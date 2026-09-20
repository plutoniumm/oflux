package compat

import (
	"context"
	"errors"
	"slices"
	"testing"

	"oflux/internal/archdb"
	"oflux/internal/types"
)

// fakeFetcher is an in-memory RepoFetcher: no network. tree maps a repo id to
// its file listing; blobs maps repo -> path -> file contents.
type fakeFetcher struct {
	tree    map[string][]types.HFFile
	blobs   map[string]map[string][]byte
	treeErr error
}

func (f *fakeFetcher) Tree(_ context.Context, repo, _ string) ([]types.HFFile, error) {
	if f.treeErr != nil {
		return nil, f.treeErr
	}
	return f.tree[repo], nil
}

func (f *fakeFetcher) ReadFile(_ context.Context, repo, _, path string, _ int64) ([]byte, error) {
	m := f.blobs[repo]
	if m == nil {
		return nil, errors.New("repo not found: " + repo)
	}
	b, ok := m[path]
	if !ok {
		return nil, errors.New("file not found: " + path)
	}
	return b, nil
}

// hf builds a slice of HFFile from bare paths.
func hf(paths ...string) []types.HFFile {
	out := make([]types.HFFile, 0, len(paths))
	for _, p := range paths {
		out = append(out, types.HFFile{Path: p, IsLFS: true})
	}
	return out
}

func compByRole(t *testing.T, m *types.Manifest, role types.Role) types.Component {
	t.Helper()
	c, ok := m.Component(role)
	if !ok {
		t.Fatalf("manifest missing component for role %q", role)
	}
	return c
}

func hasBlocker(bs []types.Blocker, kind types.BlockerKind) (types.Blocker, bool) {
	for _, b := range bs {
		if b.Kind == kind {
			return b, true
		}
	}
	return types.Blocker{}, false
}

var defaultQuantPref = []string{"Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"}

func TestInspect_FluxGGUFMirror(t *testing.T) {
	repo := "city96/FLUX.1-dev-gguf"
	f := &fakeFetcher{
		tree: map[string][]types.HFFile{
			repo: hf("flux1-dev-Q8_0.gguf", "flux1-dev-Q4_K_M.gguf", "README.md"),
		},
		// no model_index.json -> keyword detection from repo id / filenames.
	}
	v, err := Inspect(context.Background(), f, repo, defaultQuantPref)
	if err != nil {
		t.Fatalf("Inspect error: %v", err)
	}
	if !v.Compatible {
		t.Fatalf("expected compatible, got blockers: %+v", v.Blockers)
	}
	if v.Manifest == nil {
		t.Fatal("expected manifest")
	}
	if v.Manifest.Architecture != "flux" {
		t.Errorf("architecture = %q, want flux", v.Manifest.Architecture)
	}
	if v.Manifest.Mode != types.ModeGenerate {
		t.Errorf("mode = %q, want generate", v.Manifest.Mode)
	}
	if v.Manifest.Name != "flux.1-dev" {
		t.Errorf("name = %q, want flux.1-dev", v.Manifest.Name)
	}
	dif := compByRole(t, v.Manifest, types.RoleDiffusion)
	if dif.File != "flux1-dev-Q8_0.gguf" || dif.Source != repo {
		t.Errorf("diffusion = %+v, want Q8_0 from repo", dif)
	}
	vae := compByRole(t, v.Manifest, types.RoleVAE)
	if vae.File != "ae.safetensors" || vae.Source != "ffxvs/vae-flux" {
		t.Errorf("vae companion = %+v", vae)
	}
	clip := compByRole(t, v.Manifest, types.RoleCLIPL)
	if clip.File != "clip_l.safetensors" || clip.Source != "comfyanonymous/flux_text_encoders" {
		t.Errorf("clip_l companion = %+v", clip)
	}
	t5 := compByRole(t, v.Manifest, types.RoleT5XXL)
	if t5.File != "t5-v1_1-xxl-encoder-Q8_0.gguf" || t5.Source != "city96/t5-v1_1-xxl-encoder-gguf" {
		t.Errorf("t5xxl companion = %+v (want {quant} filled with Q8_0)", t5)
	}
	if !slices.Contains(v.Notes, "quant: Q8_0") {
		t.Errorf("notes = %v, want quant: Q8_0", v.Notes)
	}
	if len(v.Manifest.Engine.Flags) == 0 {
		t.Error("expected engine flags")
	}
}

func TestInspect_QwenImageEditDiffusers(t *testing.T) {
	repo := "Qwen/Qwen-Image-Edit"
	f := &fakeFetcher{
		tree: map[string][]types.HFFile{
			repo: hf("model_index.json", "Qwen_Image_Edit-Q8_0.gguf"),
		},
		blobs: map[string]map[string][]byte{
			repo: {"model_index.json": []byte(`{"_class_name":"QwenImageEditPipeline"}`)},
		},
	}
	v, err := Inspect(context.Background(), f, repo, defaultQuantPref)
	if err != nil {
		t.Fatalf("Inspect error: %v", err)
	}
	if !v.Compatible {
		t.Fatalf("expected compatible, got blockers: %+v", v.Blockers)
	}
	if !v.Manifest.Mode.CanEdit() {
		t.Errorf("mode = %q, want edit", v.Manifest.Mode)
	}
	if v.Manifest.Architecture != "qwen-image-edit" {
		t.Errorf("architecture = %q", v.Manifest.Architecture)
	}
	dif := compByRole(t, v.Manifest, types.RoleDiffusion)
	if dif.File != "Qwen_Image_Edit-Q8_0.gguf" || dif.Source != repo {
		t.Errorf("diffusion = %+v", dif)
	}
	vae := compByRole(t, v.Manifest, types.RoleVAE)
	if vae.File != "split_files/vae/qwen_image_vae.safetensors" {
		t.Errorf("vae = %+v", vae)
	}
	llm := compByRole(t, v.Manifest, types.RoleLLM)
	if llm.File != "Qwen2.5-VL-7B-Instruct.Q8_0.gguf" {
		t.Errorf("llm = %+v", llm)
	}
	// mmproj is intentionally NOT included: we target Qwen-Image-Edit 2511, which
	// uses --llm only (mmproj is a 2509-only component).
	if _, ok := v.Manifest.Component(types.RoleMMProj); ok {
		t.Errorf("mmproj should not be present for qwen-image-edit (2511 target)")
	}
	if v.Manifest.Name != "qwen-image-edit" {
		t.Errorf("name = %q", v.Manifest.Name)
	}
}

func TestInspect_SDXLDiffusers(t *testing.T) {
	repo := "stabilityai/stable-diffusion-xl-base-1.0"
	f := &fakeFetcher{
		tree: map[string][]types.HFFile{
			repo: hf("model_index.json", "sd_xl_base_1.0-Q8_0.gguf"),
		},
		blobs: map[string]map[string][]byte{
			repo: {"model_index.json": []byte(`{"_class_name":"StableDiffusionXLPipeline"}`)},
		},
	}
	v, err := Inspect(context.Background(), f, repo, defaultQuantPref)
	if err != nil {
		t.Fatalf("Inspect error: %v", err)
	}
	if !v.Compatible {
		t.Fatalf("expected compatible, got blockers: %+v", v.Blockers)
	}
	if v.Manifest.Architecture != "sdxl" {
		t.Errorf("architecture = %q, want sdxl", v.Manifest.Architecture)
	}
	if v.Manifest.Mode != types.ModeGenerate {
		t.Errorf("mode = %q, want generate", v.Manifest.Mode)
	}
	dif := compByRole(t, v.Manifest, types.RoleDiffusion)
	if dif.File != "sd_xl_base_1.0-Q8_0.gguf" {
		t.Errorf("diffusion = %+v", dif)
	}
}

func TestInspect_UnknownArch(t *testing.T) {
	repo := "acme/mystery-diffuser"
	f := &fakeFetcher{
		tree: map[string][]types.HFFile{
			repo: hf("model_index.json", "model-Q8_0.gguf"),
		},
		blobs: map[string]map[string][]byte{
			repo: {"model_index.json": []byte(`{"_class_name":"SomeUnknownPipeline"}`)},
		},
	}
	v, err := Inspect(context.Background(), f, repo, defaultQuantPref)
	if err != nil {
		t.Fatalf("Inspect error: %v", err)
	}
	if v.Compatible {
		t.Fatal("expected incompatible for unknown arch")
	}
	if _, ok := hasBlocker(v.Blockers, types.BlockerArchitecture); !ok {
		t.Errorf("expected architecture blocker, got %+v", v.Blockers)
	}
	if v.Manifest != nil {
		t.Error("manifest should be nil when incompatible")
	}
}

func TestInspect_FluxFP16NoQuant(t *testing.T) {
	repo := "black-forest-labs/FLUX.1-dev"
	f := &fakeFetcher{
		tree: map[string][]types.HFFile{
			repo: hf("model_index.json", "flux1-dev.safetensors"),
		},
		blobs: map[string]map[string][]byte{
			repo: {"model_index.json": []byte(`{"_class_name":"FluxPipeline"}`)},
		},
	}
	v, err := Inspect(context.Background(), f, repo, defaultQuantPref)
	if err != nil {
		t.Fatalf("Inspect error: %v", err)
	}
	if v.Compatible {
		t.Fatal("expected incompatible: only fp16 weights")
	}
	b, ok := hasBlocker(v.Blockers, types.BlockerNoQuant)
	if !ok {
		t.Fatalf("expected no_quant blocker, got %+v", v.Blockers)
	}
	if b.Role != types.RoleDiffusion {
		t.Errorf("blocker role = %q, want diffusion", b.Role)
	}
	if b.Suggest != "city96/FLUX.1-dev-gguf" {
		t.Errorf("blocker suggest = %q, want city96/FLUX.1-dev-gguf", b.Suggest)
	}
}

func TestInspect_TreeError(t *testing.T) {
	repo := "x/y"
	f := &fakeFetcher{treeErr: errors.New("boom")}
	_, err := Inspect(context.Background(), f, repo, defaultQuantPref)
	if err == nil {
		t.Fatal("expected error propagated from Tree")
	}
}

func TestSelectQuant(t *testing.T) {
	tests := []struct {
		name     string
		cands    []string
		pref     []string
		wantFile string
		wantQ    archdb.Quant
		wantOK   bool
	}{
		{
			name:     "prefers Q8_0 over present Q4",
			cands:    []string{"m-Q4_K_M.gguf", "m-Q8_0.gguf"},
			pref:     []string{"Q8_0", "Q6_K"},
			wantFile: "m-Q8_0.gguf",
			wantQ:    "Q8_0",
			wantOK:   true,
		},
		{
			name:   "no match when only Q4 present",
			cands:  []string{"m-Q4_K_M.gguf"},
			pref:   []string{"Q8_0", "Q6_K"},
			wantOK: false,
		},
		{
			name:     "respects pref order (Q6 before Q8)",
			cands:    []string{"m-Q8_0.gguf", "m-Q6_K.gguf"},
			pref:     []string{"Q6_K", "Q8_0"},
			wantFile: "m-Q6_K.gguf",
			wantQ:    "Q6_K",
			wantOK:   true,
		},
		{
			name:     "case insensitive, canonical label",
			cands:    []string{"M-Q8_0.GGUF"},
			pref:     []string{"q8_0"},
			wantFile: "M-Q8_0.GGUF",
			wantQ:    "Q8_0",
			wantOK:   true,
		},
		{
			// Q4_K_M must never be reported as the Q4_K inside it: the label is
			// reused to build companion filenames.
			name:   "sized K-quant is not its bare label",
			cands:  []string{"m-Q4_K_M.gguf"},
			pref:   []string{"Q4_K"},
			wantOK: false,
		},
		{
			// A label archdb does not know still matches by name, so a user can
			// ask for an exotic quant by hand.
			name:     "unknown label falls back to substring",
			cands:    []string{"m-IQ4_XS.gguf"},
			pref:     []string{"IQ4_XS"},
			wantFile: "m-IQ4_XS.gguf",
			wantQ:    "IQ4_XS",
			wantOK:   true,
		},
		{
			name:   "empty pref",
			cands:  []string{"m-Q8_0.gguf"},
			pref:   nil,
			wantOK: false,
		},
		{
			name:   "empty candidates",
			cands:  nil,
			pref:   []string{"Q8_0"},
			wantOK: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			file, quant, ok := selectQuant(tt.cands, quantPrefs(tt.pref))
			if ok != tt.wantOK {
				t.Fatalf("ok = %v, want %v", ok, tt.wantOK)
			}
			if !ok {
				return
			}
			if file != tt.wantFile {
				t.Errorf("file = %q, want %q", file, tt.wantFile)
			}
			if quant != tt.wantQ {
				t.Errorf("quant = %q, want %q", quant, tt.wantQ)
			}
		})
	}
}

func TestDetectArch(t *testing.T) {
	ctx := context.Background()

	t.Run("via model_index class name", func(t *testing.T) {
		repo := "org/whatever"
		f := &fakeFetcher{
			blobs: map[string]map[string][]byte{
				repo: {"model_index.json": []byte(`{"_class_name":"FluxKontextPipeline"}`)},
			},
		}
		files := hf("model_index.json", "x.gguf")
		a, ok, err := detectArch(ctx, f, repo, files)
		if err != nil || !ok {
			t.Fatalf("ok=%v err=%v", ok, err)
		}
		if a.Name != "flux-kontext" {
			t.Errorf("arch = %q, want flux-kontext", a.Name)
		}
	})

	t.Run("via config.json architectures array", func(t *testing.T) {
		repo := "org/bare"
		f := &fakeFetcher{
			blobs: map[string]map[string][]byte{
				repo: {"config.json": []byte(`{"architectures":["FluxTransformer2DModel"]}`)},
			},
		}
		files := hf("config.json", "model.safetensors")
		a, ok, err := detectArch(ctx, f, repo, files)
		if err != nil || !ok {
			t.Fatalf("ok=%v err=%v", ok, err)
		}
		if a.Name != "flux" {
			t.Errorf("arch = %q, want flux", a.Name)
		}
	})

	t.Run("keyword fallback from repo id", func(t *testing.T) {
		repo := "someone/Qwen-Image-Edit-GGUF"
		f := &fakeFetcher{}
		files := hf("qwen_image_edit-Q8_0.gguf")
		a, ok, err := detectArch(ctx, f, repo, files)
		if err != nil || !ok {
			t.Fatalf("ok=%v err=%v", ok, err)
		}
		if a.Name != "qwen-image-edit" {
			t.Errorf("arch = %q, want qwen-image-edit", a.Name)
		}
	})

	t.Run("unknown", func(t *testing.T) {
		repo := "org/nothing"
		f := &fakeFetcher{}
		files := hf("weights.bin")
		_, ok, err := detectArch(ctx, f, repo, files)
		if err != nil {
			t.Fatalf("err = %v", err)
		}
		if ok {
			t.Error("expected ok=false for unknown arch")
		}
	})
}

func TestDeriveName(t *testing.T) {
	tests := []struct{ in, want string }{
		{"city96/FLUX.1-dev-gguf", "flux.1-dev"},
		{"QuantStack/Qwen-Image-Edit-GGUF", "qwen-image-edit"},
		{"stabilityai/stable-diffusion-xl-base-1.0", "stable-diffusion-xl-base-1.0"},
		{"noslug", "noslug"},
	}
	for _, tt := range tests {
		if got := deriveName(tt.in); got != tt.want {
			t.Errorf("deriveName(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

func TestSuggestedGGUFRepo(t *testing.T) {
	tests := []struct{ arch, want string }{
		{"flux", "city96/FLUX.1-dev-gguf"},
		{"flux-kontext", "QuantStack/FLUX.1-Kontext-dev-GGUF"},
		{"qwen-image", "QuantStack/Qwen-Image-GGUF"},
		{"qwen-image-edit", "QuantStack/Qwen-Image-Edit-GGUF"},
		{"z-image", "leejet/Z-Image-Turbo-GGUF"},
		{"sdxl", ""},
	}
	for _, tt := range tests {
		if got := suggestedGGUFRepo(tt.arch); got != tt.want {
			t.Errorf("suggestedGGUFRepo(%q) = %q, want %q", tt.arch, got, tt.want)
		}
	}
}

// The bug a real user hit: an underscored klein-9B matched no keyword, fell
// through to the flux2 arch and got flux2-dev's Mistral-3 encoder.
func TestInspect_Flux2Klein9BUnderscores(t *testing.T) {
	repo := "someone/flux2-klein-gguf"
	f := &fakeFetcher{tree: map[string][]types.HFFile{
		repo: hf("flux2_klein_9b_Q8_0.gguf", "README.md"),
	}}
	v, err := Inspect(context.Background(), f, repo, defaultQuantPref)
	if err != nil {
		t.Fatalf("Inspect error: %v", err)
	}
	if !v.Compatible {
		t.Fatalf("expected compatible, got blockers: %+v", v.Blockers)
	}
	if v.Manifest.Architecture != "flux2-klein" {
		t.Fatalf("architecture = %q, want flux2-klein", v.Manifest.Architecture)
	}
	llm := compByRole(t, v.Manifest, types.RoleLLM)
	if llm.Source != "unsloth/Qwen3-8B-GGUF" || llm.File != "Qwen3-8B-Q8_0.gguf" {
		t.Errorf("llm = %+v, want the Qwen3-8B encoder", llm)
	}
	repo4 := "someone/flux2-klein-gguf"
	f4 := &fakeFetcher{tree: map[string][]types.HFFile{
		repo4: hf("flux2_klein_4b_Q8_0.gguf"),
	}}
	v4, _ := Inspect(context.Background(), f4, repo4, defaultQuantPref)
	llm4 := compByRole(t, v4.Manifest, types.RoleLLM)
	if llm4.Source != "unsloth/Qwen3-4B-GGUF" {
		t.Errorf("klein-4B llm = %+v", llm4)
	}
}

// compat already walks every tree the puller would; recording sha256 and size
// here saves fetching them all again.
func TestInspect_FillsComponentHashAndSize(t *testing.T) {
	repo := "city96/FLUX.1-dev-gguf"
	f := &fakeFetcher{tree: map[string][]types.HFFile{
		repo: {
			{Path: "flux1-dev-Q8_0.gguf", IsLFS: true, LFSOID: "aaaa", Size: 123},
			{Path: "config.json", OID: "gitsha1", Size: 7},
		},
		"ffxvs/vae-flux": {
			{Path: "ae.safetensors", IsLFS: true, LFSOID: "bbbb", Size: 456},
		},
	}}
	v, err := Inspect(context.Background(), f, repo, defaultQuantPref)
	if err != nil || !v.Compatible {
		t.Fatalf("Inspect: %v %+v", err, v.Blockers)
	}
	dif := compByRole(t, v.Manifest, types.RoleDiffusion)
	if dif.SHA256 != "aaaa" || dif.Size != 123 {
		t.Errorf("diffusion sha/size = %q/%d", dif.SHA256, dif.Size)
	}
	vae := compByRole(t, v.Manifest, types.RoleVAE)
	if vae.SHA256 != "bbbb" || vae.Size != 456 {
		t.Errorf("companion sha/size = %q/%d", vae.SHA256, vae.Size)
	}
	t5 := compByRole(t, v.Manifest, types.RoleT5XXL)
	if t5.SHA256 != "" || t5.Size != 0 {
		t.Errorf("unknown file should carry no hash: %+v", t5)
	}
}

// A git oid is a sha1, not a sha256, and must never be recorded as one.
func TestInspect_NonLFSFileHasNoSHA256(t *testing.T) {
	repo := "org/sdxl-gguf"
	f := &fakeFetcher{tree: map[string][]types.HFFile{
		repo: {{Path: "sdxl-Q8_0.gguf", OID: "gitsha1", Size: 9}},
	}}
	v, _ := Inspect(context.Background(), f, repo, defaultQuantPref)
	dif := compByRole(t, v.Manifest, types.RoleDiffusion)
	if dif.SHA256 != "" {
		t.Errorf("SHA256 = %q, want empty for a non-LFS file", dif.SHA256)
	}
	if dif.Size != 9 {
		t.Errorf("Size = %d, want 9", dif.Size)
	}
}

func TestInspect_BaseAndRevision(t *testing.T) {
	repo := "QuantStack/Qwen-Image-Edit-GGUF"
	f := &fakeFetcher{tree: map[string][]types.HFFile{
		repo: hf("qwen-image-edit-2509-Q8_0.gguf"),
	}}
	v, _ := Inspect(context.Background(), f, repo, defaultQuantPref)
	if v.Manifest.Base != "Qwen-Image-Edit" {
		t.Errorf("base = %q", v.Manifest.Base)
	}
	if v.Manifest.Revision != "2509" {
		t.Errorf("revision = %q, want 2509", v.Manifest.Revision)
	}
}

// The two views of one fact must not drift apart.
func TestDiffusionExcludeCoversEveryRoleKeyword(t *testing.T) {
	for role, kws := range roleKeywords {
		for _, kw := range kws {
			if !slices.Contains(diffusionExcludeTokens, kw) {
				t.Errorf("role %s keyword %q would still be a diffusion candidate", role, kw)
			}
		}
	}
	files := hf("flux1-dev-Q8_0.gguf", "t5xxl_fp16.safetensors", "clip_l.safetensors", "qwen_image_vae.safetensors", "mmproj-f16.gguf")
	got := diffusionCandidates(files)
	if len(got) != 1 || got[0] != "flux1-dev-Q8_0.gguf" {
		t.Errorf("diffusionCandidates = %v", got)
	}
}
