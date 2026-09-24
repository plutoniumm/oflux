package puller

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"oflux/internal/hfclient"
	"oflux/internal/registry"
	"oflux/internal/store"
	"oflux/internal/types"
)

// fakeHub serves the minimal Hugging Face surface the puller uses: the tree
// listing and the resolve (download) endpoint, for a fixed set of repos.
func fakeHub(repos map[string]map[string][]byte) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path := r.URL.Path
		if strings.HasPrefix(path, "/api/models/") && strings.HasSuffix(path, "/tree/main") {
			repo := strings.TrimSuffix(strings.TrimPrefix(path, "/api/models/"), "/tree/main")
			files, ok := repos[repo]
			if !ok {
				http.Error(w, "no repo", http.StatusNotFound)
				return
			}
			var arr []map[string]any
			for name, content := range files {
				sum := sha256.Sum256(content)
				arr = append(arr, map[string]any{
					"type": "file", "oid": "git-" + name, "size": len(content), "path": name,
					"lfs": map[string]any{"oid": hex.EncodeToString(sum[:]), "size": len(content), "pointerSize": 100},
				})
			}
			json.NewEncoder(w).Encode(arr)
			return
		}
		if i := strings.Index(path, "/resolve/main/"); i >= 0 {
			repo := strings.TrimPrefix(path[:i], "/")
			file := path[i+len("/resolve/main/"):]
			if content, ok := repos[repo][file]; ok {
				w.Write(content)
				return
			}
		}
		http.Error(w, "not found", http.StatusNotFound)
	}))
}

func newPuller(t *testing.T, hub *httptest.Server) *Puller {
	t.Helper()
	st, err := store.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	hf := hfclient.New("")
	hf.SetBaseURL(hub.URL)
	return New(hf, st)
}

func TestResolveCurated(t *testing.T) {
	p := newPuller(t, fakeHub(nil))
	v, err := p.Resolve(context.Background(), "qwe-2511", "Q8_0", "")
	if err != nil {
		t.Fatal(err)
	}
	if !v.Compatible || v.Manifest == nil {
		t.Fatalf("expected compatible curated resolve, got %+v", v)
	}
	if !v.Manifest.Mode.CanEdit() || v.Manifest.Architecture != "qwen-image-edit" {
		t.Errorf("manifest = %+v", v.Manifest)
	}
}

func TestResolveUnknown(t *testing.T) {
	p := newPuller(t, fakeHub(nil))
	if _, err := p.Resolve(context.Background(), "not-a-real-model", "", ""); err == nil {
		t.Fatal("expected error for unknown bare name")
	}
}

func TestPullFluxRepoEndToEnd(t *testing.T) {
	repos := map[string]map[string][]byte{
		"city96/FLUX.1-dev-gguf":            {"flux1-dev-Q8_0.gguf": []byte("DIFFUSION-WEIGHTS")},
		"ffxvs/vae-flux":                    {"ae.safetensors": []byte("VAE")},
		"comfyanonymous/flux_text_encoders": {"clip_l.safetensors": []byte("CLIP-L")},
		"city96/t5-v1_1-xxl-encoder-gguf":   {"t5-v1_1-xxl-encoder-Q8_0.gguf": []byte("T5XXL")},
	}
	hub := fakeHub(repos)
	p := newPuller(t, hub)

	m, err := p.Pull(context.Background(), "city96/FLUX.1-dev-gguf", "Q8_0", Opts{}, nil)
	if err != nil {
		t.Fatalf("Pull: %v", err)
	}
	if m.Architecture != "flux" || m.Mode != types.ModeGenerate {
		t.Errorf("manifest arch/mode = %s/%s", m.Architecture, m.Mode)
	}
	// All four components must be present, downloaded (Blob set), and in the store.
	wantRoles := []types.Role{types.RoleDiffusion, types.RoleVAE, types.RoleCLIPL, types.RoleT5XXL}
	for _, role := range wantRoles {
		c, ok := m.Component(role)
		if !ok {
			t.Fatalf("missing component %s", role)
		}
		if c.Blob == "" || !p.store.HasBlob(c.Blob) {
			t.Errorf("component %s not stored: %+v", role, c)
		}
	}
	dif, _ := m.Component(types.RoleDiffusion)
	if dif.Source != "city96/FLUX.1-dev-gguf" || dif.File != "flux1-dev-Q8_0.gguf" {
		t.Errorf("diffusion = %+v", dif)
	}

	// The manifest must be persisted and re-loadable.
	got, err := p.store.ReadManifest(m.Name)
	if err != nil {
		t.Fatalf("ReadManifest: %v", err)
	}
	if len(got.Components) != 4 {
		t.Errorf("persisted manifest has %d components", len(got.Components))
	}
}

func TestPullIncompatibleRepo(t *testing.T) {
	// A repo whose architecture can't be recognized → a descriptive error.
	repos := map[string]map[string][]byte{
		"someorg/mystery-model": {"model_index.json": []byte(`{"_class_name":"TotallyUnknownPipeline"}`)},
	}
	p := newPuller(t, fakeHub(repos))
	_, err := p.Pull(context.Background(), "someorg/mystery-model", "Q8_0", Opts{}, nil)
	if err == nil {
		t.Fatal("expected incompatibility error")
	}
	if !strings.Contains(err.Error(), "not compatible") {
		t.Errorf("error = %v", err)
	}
}

func TestComponentSHAUsesTheVerdictChecksum(t *testing.T) {
	hub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("hub hit for %s though the verdict already carried the checksum", r.URL.Path)
		http.Error(w, "no", http.StatusNotFound)
	}))
	defer hub.Close()
	p := newPuller(t, hub)

	want := strings.Repeat("ab", 32)
	sha, size := p.componentSHA(context.Background(), map[string][]types.HFFile{}, types.Component{
		Role: types.RoleDiffusion, Source: "org/repo", File: "w.gguf", SHA256: want, Size: 42,
	})
	if sha != want || size != 42 {
		t.Fatalf("componentSHA = %q, %d; want %q, 42", sha, size, want)
	}
}

func TestComponentSHAFallsBackToTheRepoTree(t *testing.T) {
	content := []byte("DIFFUSION-WEIGHTS")
	sum := sha256.Sum256(content)
	hub := fakeHub(map[string]map[string][]byte{"org/repo": {"w.gguf": content}})
	defer hub.Close()
	p := newPuller(t, hub)

	for _, c := range []types.Component{
		{Source: "org/repo", File: "w.gguf"},
		// A git blob oid is a sha1: as an expected checksum it would fail every
		// transfer it guards, so it must not be mistaken for content identity.
		{Source: "org/repo", File: "w.gguf", SHA256: strings.Repeat("a", 40)},
	} {
		sha, size := p.componentSHA(context.Background(), map[string][]types.HFFile{}, c)
		if sha != hex.EncodeToString(sum[:]) || size != int64(len(content)) {
			t.Errorf("componentSHA(%+v) = %q, %d", c, sha, size)
		}
	}
}

func TestPullLoraRecordsCuratedMetadata(t *testing.T) {
	l, ok := registry.LookupLora("qwen-edit-lightning-4step")
	if !ok {
		t.Fatal("curated lora missing from the registry")
	}
	hub := fakeHub(map[string]map[string][]byte{l.Source: {l.File: []byte("ADAPTER")}})
	defer hub.Close()
	p := newPuller(t, hub)

	name, err := p.PullLora(context.Background(), l.Name, LoraOpts{}, nil)
	if err != nil {
		t.Fatalf("PullLora: %v", err)
	}
	meta, err := p.store.ReadLoraMeta(name)
	if err != nil {
		t.Fatal(err)
	}
	want := store.LoraMeta{
		Name: l.Name, Source: l.Source, File: l.File,
		For: l.Archs, Steps: l.Steps, CFG: l.CFG,
	}
	if !reflect.DeepEqual(meta, want) {
		t.Fatalf("sidecar = %+v, want the registry's pins %+v", meta, want)
	}
	rows, err := p.store.ListLoras()
	if err != nil || len(rows) != 1 || rows[0].Steps != l.Steps {
		t.Fatalf("ListLoras = %+v, %v", rows, err)
	}
}

func TestPullLoraFromRepoRecordsCallerMetadata(t *testing.T) {
	hub := fakeHub(map[string]map[string][]byte{"org/style": {"style-v2.safetensors": []byte("ADAPTER")}})
	defer hub.Close()
	p := newPuller(t, hub)

	name, err := p.PullLora(context.Background(), "org/style", LoraOpts{
		As: "my-style", For: []string{"flux"}, Steps: 8, CFG: 1.0,
	}, nil)
	if err != nil {
		t.Fatalf("PullLora: %v", err)
	}
	meta, err := p.store.ReadLoraMeta(name)
	if err != nil {
		t.Fatal(err)
	}
	want := store.LoraMeta{
		Name: "my-style", Source: "org/style", File: "style-v2.safetensors",
		For: []string{"flux"}, Steps: 8, CFG: 1.0,
	}
	if !reflect.DeepEqual(meta, want) {
		t.Fatalf("sidecar = %+v, want %+v", meta, want)
	}
}

func TestQuantPrefPutsTheRequestedQuantFirst(t *testing.T) {
	for _, want := range []string{"Q8_0", "Q4_K_M", "some-vendor-quant"} {
		got := quantPref(want)
		if len(got) == 0 || got[0] != want {
			t.Errorf("quantPref(%q) = %v", want, got)
		}
	}
	if len(quantPref("")) == 0 {
		t.Error("quantPref(\"\") must still offer a fallback chain")
	}
}
