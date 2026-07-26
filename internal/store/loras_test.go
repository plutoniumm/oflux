package store

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"oflux/internal/types"
)

func newTestStore(t *testing.T) *Store {
	t.Helper()
	s, err := Open(t.TempDir())
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	return s
}

func TestValidLoraName(t *testing.T) {
	ok := []string{"qwen-edit-lightning-4step", "a", "style_v1.0", "Flux-Turbo-8"}
	for _, n := range ok {
		if err := ValidLoraName(n); err != nil {
			t.Errorf("ValidLoraName(%q) = %v, want nil", n, err)
		}
	}
	// A LoRA name becomes a filename handed to the engine, so anything that
	// could escape the loras directory or name a file outside it must be
	// rejected before it reaches disk.
	bad := []string{
		"", "..", "../../etc/passwd", "a/b", `a\b`, ".hidden", "-leading",
		"has space", "quote'd", "semi;colon", strings.Repeat("x", 65),
	}
	for _, n := range bad {
		if err := ValidLoraName(n); err == nil {
			t.Errorf("ValidLoraName(%q) = nil, want an error", n)
		}
	}
}

func TestLoraPathRejectsTraversal(t *testing.T) {
	s := newTestStore(t)
	if _, err := s.LoraPath("../../evil"); err == nil {
		t.Fatal("LoraPath must reject a traversal name")
	}
	p, err := s.LoraPath("good-name")
	if err != nil {
		t.Fatal(err)
	}
	if want := filepath.Join(s.LorasDir(), "good-name.safetensors"); p != want {
		t.Fatalf("LoraPath = %q, want %q", p, want)
	}
}

func TestLoraLifecycle(t *testing.T) {
	s := newTestStore(t)

	if got, err := s.ListLoras(); err != nil || len(got) != 0 {
		t.Fatalf("fresh store: ListLoras = %v, %v", got, err)
	}
	if s.HasLora("nope") {
		t.Fatal("HasLora on an empty store should be false")
	}

	src := filepath.Join(t.TempDir(), "incoming.bin")
	if err := os.WriteFile(src, []byte("weights"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := s.PutLora("my-lora", src); err != nil {
		t.Fatalf("PutLora: %v", err)
	}
	if _, err := os.Stat(src); !os.IsNotExist(err) {
		t.Error("PutLora should consume the source file")
	}
	if !s.HasLora("my-lora") {
		t.Fatal("HasLora after PutLora should be true")
	}

	// The on-disk name is the engine's reference: sd-server resolves
	// lora[].path against --lora-model-dir.
	if _, err := os.Stat(filepath.Join(s.LorasDir(), LoraFileName("my-lora"))); err != nil {
		t.Fatalf("lora not stored under its engine filename: %v", err)
	}

	got, err := s.ListLoras()
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Name != "my-lora" || got[0].Size != int64(len("weights")) {
		t.Fatalf("ListLoras = %+v", got)
	}

	if err := s.RemoveLora("my-lora"); err != nil {
		t.Fatalf("RemoveLora: %v", err)
	}
	if s.HasLora("my-lora") {
		t.Fatal("lora still present after RemoveLora")
	}
	if err := s.RemoveLora("my-lora"); err == nil {
		t.Fatal("removing a missing lora should error")
	}
}

// `oflux rm` used to hang for the entire duration of a concurrent multi-gigabyte
// pull: the manifest was already deleted, but GC waited on the lock the pull
// holds. Removal must return promptly and report that collection was deferred.
func TestRemoveManifestDoesNotBlockOnAnInFlightPull(t *testing.T) {
	s := newTestStore(t)

	blob := writeTemp(t, t.TempDir(), "w", "weights")
	name, err := s.PutBlob(hashOf("weights"), blob)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.WriteManifest(types.Manifest{
		Name:       "doomed",
		Components: []types.Component{{Role: types.RoleDiffusion, Blob: name}},
	}); err != nil {
		t.Fatal(err)
	}

	// Simulate a pull in flight.
	done := s.BeginWrite()

	returned := make(chan struct{})
	var collected bool
	go func() {
		defer close(returned)
		_, collected, err = s.RemoveManifest("doomed")
	}()

	select {
	case <-returned:
	case <-time.After(5 * time.Second):
		done()
		t.Fatal("RemoveManifest blocked while a pull held the store")
	}
	if err != nil {
		t.Fatalf("RemoveManifest: %v", err)
	}
	if collected {
		t.Error("GC should have been skipped, not run, during a pull")
	}
	// The model itself must be gone regardless — that is what the user asked for.
	if _, err := s.ReadManifest("doomed"); !errors.Is(err, ErrManifestNotFound) {
		t.Errorf("manifest should be deleted, got %v", err)
	}
	// The blob is merely deferred, not leaked: GC reclaims it once the pull ends.
	if !s.HasBlob(name) {
		t.Error("blob should still be present while collection is deferred")
	}
	done()
	freed, collected, err := s.TryGC()
	if err != nil || !collected {
		t.Fatalf("TryGC after the pull: collected=%v err=%v", collected, err)
	}
	if len(freed) != 1 || s.HasBlob(name) {
		t.Errorf("deferred blob not reclaimed: freed=%v", freed)
	}
}

// LoRAs live outside the content-addressed blob store, so garbage collection
// (which only knows about manifest-referenced blobs) must never touch them.
func TestGCLeavesLorasAlone(t *testing.T) {
	s := newTestStore(t)
	src := filepath.Join(t.TempDir(), "incoming.bin")
	if err := os.WriteFile(src, []byte("weights"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := s.PutLora("keeper", src); err != nil {
		t.Fatal(err)
	}
	if _, err := s.GC(); err != nil {
		t.Fatal(err)
	}
	if !s.HasLora("keeper") {
		t.Fatal("GC deleted an installed lora")
	}
}
