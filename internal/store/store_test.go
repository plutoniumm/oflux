package store

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"oflux/internal/types"
)

// writeFile creates a file with the given content and returns its path.
func writeTemp(t *testing.T, dir, name, content string) string {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatalf("write temp %s: %v", p, err)
	}
	return p
}

// hashOf returns the hex sha256 of s.
func hashOf(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])
}

func TestOpenCreatesLayout(t *testing.T) {
	root := t.TempDir()
	s, err := Open(root)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if s.Root() != root {
		t.Errorf("Root() = %q, want %q", s.Root(), root)
	}
	for _, dir := range []string{s.BlobsDir(), s.ManifestsDir(), s.LogsDir()} {
		info, err := os.Stat(dir)
		if err != nil {
			t.Errorf("expected dir %q: %v", dir, err)
			continue
		}
		if !info.IsDir() {
			t.Errorf("%q is not a directory", dir)
		}
	}
	// Verify the subdir paths are anchored under root.
	if s.BlobsDir() != filepath.Join(root, "blobs") {
		t.Errorf("BlobsDir() = %q", s.BlobsDir())
	}
	if s.ManifestsDir() != filepath.Join(root, "manifests") {
		t.Errorf("ManifestsDir() = %q", s.ManifestsDir())
	}
	if s.LogsDir() != filepath.Join(root, "logs") {
		t.Errorf("LogsDir() = %q", s.LogsDir())
	}
}

func TestOpenEnvOverride(t *testing.T) {
	home := t.TempDir()
	t.Setenv("OFLUX_HOME", home)
	s, err := Open("")
	if err != nil {
		t.Fatalf("Open(\"\"): %v", err)
	}
	if s.Root() != home {
		t.Errorf("Root() = %q, want OFLUX_HOME %q", s.Root(), home)
	}
	if _, err := os.Stat(filepath.Join(home, "blobs")); err != nil {
		t.Errorf("blobs not created under OFLUX_HOME: %v", err)
	}
}

func TestOpenDefaultHome(t *testing.T) {
	// With no OFLUX_HOME and empty root, we fall back to ~/.oflux. Point HOME
	// at a temp dir so we don't touch the real home.
	home := t.TempDir()
	t.Setenv("OFLUX_HOME", "")
	t.Setenv("HOME", home)
	s, err := Open("")
	if err != nil {
		t.Fatalf("Open(\"\"): %v", err)
	}
	want := filepath.Join(home, ".oflux")
	if s.Root() != want {
		t.Errorf("Root() = %q, want %q", s.Root(), want)
	}
}

func TestBlobName(t *testing.T) {
	tests := []struct {
		in, want string
	}{
		{"abc123", "sha256-abc123"},
		{"sha256:abc", "sha256-abc"},
		{"sha256-abc", "sha256-abc"},
	}
	for _, tt := range tests {
		if got := BlobName(tt.in); got != tt.want {
			t.Errorf("BlobName(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

func TestBlobPath(t *testing.T) {
	root := t.TempDir()
	s, err := Open(root)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	want := filepath.Join(root, "blobs", "sha256-abc")
	for _, in := range []string{"abc", "sha256-abc", "sha256:abc"} {
		if got := s.BlobPath(in); got != want {
			t.Errorf("BlobPath(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestPutBlobAndHas(t *testing.T) {
	root := t.TempDir()
	s, err := Open(root)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	src := t.TempDir()
	content := "weights-bytes"
	h := hashOf(content)
	p := writeTemp(t, src, "file.gguf", content)

	if s.HasBlob(h) {
		t.Fatal("HasBlob true before PutBlob")
	}
	blob, err := s.PutBlob(h, p)
	if err != nil {
		t.Fatalf("PutBlob: %v", err)
	}
	if blob != BlobName(h) {
		t.Errorf("PutBlob returned %q, want %q", blob, BlobName(h))
	}
	if !s.HasBlob(h) {
		t.Error("HasBlob false after PutBlob")
	}
	// Source must have been moved away.
	if _, err := os.Stat(p); !os.IsNotExist(err) {
		t.Errorf("src still present after PutBlob: err=%v", err)
	}
	// Content in the blob store must match.
	got, err := os.ReadFile(s.BlobPath(h))
	if err != nil {
		t.Fatalf("read blob: %v", err)
	}
	if string(got) != content {
		t.Errorf("blob content = %q, want %q", got, content)
	}
}

func TestPutBlobIdempotent(t *testing.T) {
	root := t.TempDir()
	s, err := Open(root)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	src := t.TempDir()
	content := "dup-content"
	h := hashOf(content)

	p1 := writeTemp(t, src, "a.bin", content)
	if _, err := s.PutBlob(h, p1); err != nil {
		t.Fatalf("first PutBlob: %v", err)
	}
	// Second put of same hash: should dedup, remove src, no error.
	p2 := writeTemp(t, src, "b.bin", content)
	blob, err := s.PutBlob(h, p2)
	if err != nil {
		t.Fatalf("second PutBlob: %v", err)
	}
	if blob != BlobName(h) {
		t.Errorf("second PutBlob returned %q, want %q", blob, BlobName(h))
	}
	if _, err := os.Stat(p2); !os.IsNotExist(err) {
		t.Errorf("dup src not removed: err=%v", err)
	}
}

func TestManifestRoundtrip(t *testing.T) {
	root := t.TempDir()
	s, err := Open(root)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	m := types.Manifest{
		Name:         "flux-schnell",
		Architecture: "flux",
		Mode:         types.ModeGenerate,
		Components: []types.Component{
			{Role: types.RoleDiffusion, File: "flux.gguf", Source: "org/repo", Blob: "sha256-aaa", SHA256: "aaa", Size: 10},
		},
		Engine: types.EngineSpec{Flags: []string{"--diffusion-model", "{diffusion}"}},
	}
	if err := s.WriteManifest(m); err != nil {
		t.Fatalf("WriteManifest: %v", err)
	}
	got, err := s.ReadManifest("flux-schnell")
	if err != nil {
		t.Fatalf("ReadManifest: %v", err)
	}
	if !reflect.DeepEqual(got, m) {
		t.Errorf("roundtrip mismatch:\n got %+v\nwant %+v", got, m)
	}
}

func TestReadManifestNotFound(t *testing.T) {
	root := t.TempDir()
	s, err := Open(root)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	_, err = s.ReadManifest("nope")
	if !errors.Is(err, ErrManifestNotFound) {
		t.Errorf("ReadManifest(unknown) err = %v, want ErrManifestNotFound", err)
	}
}

func TestWriteManifestRejectsBadNames(t *testing.T) {
	root := t.TempDir()
	s, err := Open(root)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	for _, bad := range []string{"a/b", "..", "../x", "a/../b"} {
		m := types.Manifest{Name: bad}
		if err := s.WriteManifest(m); err == nil {
			t.Errorf("WriteManifest(name=%q) = nil, want error", bad)
		}
	}
}

func TestListManifestsSorted(t *testing.T) {
	root := t.TempDir()
	s, err := Open(root)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	names := []string{"zebra", "alpha", "mike"}
	for _, n := range names {
		if err := s.WriteManifest(types.Manifest{Name: n}); err != nil {
			t.Fatalf("WriteManifest(%q): %v", n, err)
		}
	}
	list, err := s.ListManifests()
	if err != nil {
		t.Fatalf("ListManifests: %v", err)
	}
	got := make([]string, len(list))
	for i, m := range list {
		got[i] = m.Name
	}
	want := []string{"alpha", "mike", "zebra"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("ListManifests order = %v, want %v", got, want)
	}
}

// makeBlob writes content into the store's blob dir and returns the blob name.
func makeBlob(t *testing.T, s *Store, content string) string {
	t.Helper()
	h := hashOf(content)
	src := writeTemp(t, t.TempDir(), "x", content)
	blob, err := s.PutBlob(h, src)
	if err != nil {
		t.Fatalf("PutBlob: %v", err)
	}
	return blob
}

func TestRemoveManifestGC(t *testing.T) {
	root := t.TempDir()
	s, err := Open(root)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	shared := makeBlob(t, s, "shared-weights")
	uniqA := makeBlob(t, s, "uniq-a")
	uniqB := makeBlob(t, s, "uniq-b")

	mA := types.Manifest{
		Name: "A",
		Components: []types.Component{
			{Role: types.RoleDiffusion, Blob: shared},
			{Role: types.RoleVAE, Blob: uniqA},
		},
	}
	mB := types.Manifest{
		Name: "B",
		Components: []types.Component{
			{Role: types.RoleDiffusion, Blob: shared},
			{Role: types.RoleVAE, Blob: uniqB},
		},
	}
	if err := s.WriteManifest(mA); err != nil {
		t.Fatalf("write A: %v", err)
	}
	if err := s.WriteManifest(mB); err != nil {
		t.Fatalf("write B: %v", err)
	}

	// Remove A: uniqA should be freed, shared kept (B still references it).
	freed, collected, err := s.RemoveManifest("A")
	if !collected {
		t.Fatal("GC should run when no pull holds the store")
	}
	if err != nil {
		t.Fatalf("RemoveManifest(A): %v", err)
	}
	if len(freed) != 1 || freed[0] != uniqA {
		t.Errorf("freed = %v, want [%s]", freed, uniqA)
	}
	if s.HasBlob(uniqA) {
		t.Error("uniqA still present after removing A")
	}
	if !s.HasBlob(shared) {
		t.Error("shared blob wrongly removed while B still references it")
	}
	if !s.HasBlob(uniqB) {
		t.Error("uniqB wrongly removed")
	}
	// Manifest A gone.
	if _, err := s.ReadManifest("A"); !errors.Is(err, ErrManifestNotFound) {
		t.Errorf("ReadManifest(A) after remove err = %v", err)
	}

	// Remove B: now shared + uniqB freed.
	freed, _, err = s.RemoveManifest("B")
	if err != nil {
		t.Fatalf("RemoveManifest(B): %v", err)
	}
	freedSet := map[string]bool{}
	for _, f := range freed {
		freedSet[f] = true
	}
	if !freedSet[shared] || !freedSet[uniqB] {
		t.Errorf("freed after removing B = %v, want to include %s and %s", freed, shared, uniqB)
	}
	if s.HasBlob(shared) || s.HasBlob(uniqB) {
		t.Error("blobs remain after all manifests removed")
	}
}

func TestGCStandalone(t *testing.T) {
	root := t.TempDir()
	s, err := Open(root)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	referenced := makeBlob(t, s, "keep-me")
	orphan := makeBlob(t, s, "orphan-me")

	m := types.Manifest{
		Name:       "keeper",
		Components: []types.Component{{Role: types.RoleDiffusion, Blob: referenced}},
	}
	if err := s.WriteManifest(m); err != nil {
		t.Fatalf("WriteManifest: %v", err)
	}

	freed, err := s.GC()
	if err != nil {
		t.Fatalf("GC: %v", err)
	}
	if len(freed) != 1 || freed[0] != orphan {
		t.Errorf("GC freed = %v, want [%s]", freed, orphan)
	}
	if !s.HasBlob(referenced) {
		t.Error("referenced blob wrongly GC'd")
	}
	if s.HasBlob(orphan) {
		t.Error("orphan blob not GC'd")
	}
}

func TestConfigDefaultWhenAbsent(t *testing.T) {
	root := t.TempDir()
	s, err := Open(root)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	cfg, err := s.LoadConfig()
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if !reflect.DeepEqual(cfg, types.DefaultConfig()) {
		t.Errorf("LoadConfig (no file) = %+v, want %+v", cfg, types.DefaultConfig())
	}
}

func TestConfigRoundtrip(t *testing.T) {
	root := t.TempDir()
	s, err := Open(root)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	cfg := types.DefaultConfig()
	cfg.Port = 9999
	cfg.HFToken = "secret"
	cfg.MaxLoaded = 3
	if err := s.SaveConfig(cfg); err != nil {
		t.Fatalf("SaveConfig: %v", err)
	}
	got, err := s.LoadConfig()
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if !reflect.DeepEqual(got, cfg) {
		t.Errorf("config roundtrip = %+v, want %+v", got, cfg)
	}
}

func TestConfigPartialKeepsDefaults(t *testing.T) {
	root := t.TempDir()
	s, err := Open(root)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	// Only port is specified; other fields must retain defaults.
	writeTemp(t, root, "config.json", `{"port": 12000}`)
	got, err := s.LoadConfig()
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	def := types.DefaultConfig()
	if got.Port != 12000 {
		t.Errorf("Port = %d, want 12000", got.Port)
	}
	if got.IdleTTL != def.IdleTTL {
		t.Errorf("IdleTTL = %q, want default %q", got.IdleTTL, def.IdleTTL)
	}
	if got.MaxLoaded != def.MaxLoaded {
		t.Errorf("MaxLoaded = %d, want default %d", got.MaxLoaded, def.MaxLoaded)
	}
	if got.DefaultQuant != def.DefaultQuant {
		t.Errorf("DefaultQuant = %q, want default %q", got.DefaultQuant, def.DefaultQuant)
	}
}
