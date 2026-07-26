package puller

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"oflux/internal/hfclient"
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
	v, err := p.Resolve(context.Background(), "qwen-image-edit", "Q8_0")
	if err != nil {
		t.Fatal(err)
	}
	if !v.Compatible || v.Manifest == nil {
		t.Fatalf("expected compatible curated resolve, got %+v", v)
	}
	if v.Manifest.Mode != types.ModeEdit || v.Manifest.Architecture != "qwen-image-edit" {
		t.Errorf("manifest = %+v", v.Manifest)
	}
}

func TestResolveUnknown(t *testing.T) {
	p := newPuller(t, fakeHub(nil))
	if _, err := p.Resolve(context.Background(), "not-a-real-model", ""); err == nil {
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

	m, err := p.Pull(context.Background(), "city96/FLUX.1-dev-gguf", "Q8_0", nil)
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
	_, err := p.Pull(context.Background(), "someorg/mystery-model", "Q8_0", nil)
	if err == nil {
		t.Fatal("expected incompatibility error")
	}
	if !strings.Contains(err.Error(), "not compatible") {
		t.Errorf("error = %v", err)
	}
}
