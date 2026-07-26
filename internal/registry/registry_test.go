package registry

import (
	"slices"
	"strings"
	"testing"

	"oflux/internal/types"
)

// component returns the first component with the given role from a manifest.
func component(m types.Manifest, role types.Role) (types.Component, bool) {
	for _, c := range m.Components {
		if c.Role == role {
			return c, true
		}
	}
	return types.Component{}, false
}

func TestNamesSorted(t *testing.T) {
	got := Names()
	want := []string{
		"flux.1-dev",
		"flux.1-kontext",
		"flux.1-krea",
		"flux.1-schnell",
		"flux.2-klein",
		"qwen-image",
		"qwen-image-edit",
		"z-image-turbo",
	}
	if !slices.Equal(got, want) {
		t.Fatalf("Names() = %v, want %v", got, want)
	}
	if !slices.IsSorted(got) {
		t.Fatalf("Names() not sorted: %v", got)
	}
}

func TestLookup(t *testing.T) {
	m, ok := Lookup("qwen-image-edit")
	if !ok {
		t.Fatal("Lookup(qwen-image-edit) not found")
	}
	if m.Arch != "qwen-image-edit" {
		t.Fatalf("Arch = %q, want qwen-image-edit", m.Arch)
	}
	if _, ok := Lookup("nope"); ok {
		t.Fatal("Lookup(nope) should not be found")
	}
}

func TestResolveQwenImageEdit(t *testing.T) {
	m, ok := Resolve("qwen-image-edit", "Q8_0")
	if !ok {
		t.Fatal("Resolve(qwen-image-edit, Q8_0) not ok")
	}
	// Qwen-Image-Edit is a hybrid: the same weights edit with a reference image
	// and generate from text alone. Marking it edit-only hid half of it.
	if m.Mode != types.ModeBoth {
		t.Fatalf("Mode = %q, want %q", m.Mode, types.ModeBoth)
	}
	if !m.Mode.CanEdit() || !m.Mode.CanGenerate() {
		t.Fatalf("hybrid mode %q must allow both paths", m.Mode)
	}
	if m.Architecture != "qwen-image-edit" {
		t.Fatalf("Architecture = %q, want qwen-image-edit", m.Architecture)
	}
	if m.Name != "qwen-image-edit" {
		t.Fatalf("Name = %q, want qwen-image-edit", m.Name)
	}

	diff, ok := component(m, types.RoleDiffusion)
	if !ok {
		t.Fatal("no diffusion component")
	}
	if diff.Source != "unsloth/Qwen-Image-Edit-2511-GGUF" {
		t.Fatalf("diffusion source = %q", diff.Source)
	}
	if diff.File != "qwen-image-edit-2511-Q8_0.gguf" {
		t.Fatalf("diffusion file = %q", diff.File)
	}
	if strings.Contains(diff.File, "{quant}") {
		t.Fatalf("diffusion file still has {quant}: %q", diff.File)
	}

	if _, ok := component(m, types.RoleVAE); !ok {
		t.Fatal("no vae component")
	}
	if _, ok := component(m, types.RoleLLM); !ok {
		t.Fatal("no llm component")
	}

	// 2511 variant: zero-cond flag must be merged into engine model args.
	if v, ok := m.Engine.ModelArgs["qwen_image_zero_cond_t"]; !ok || v != true {
		t.Fatalf("qwen_image_zero_cond_t = %v (ok=%v), want true", v, ok)
	}
}

func TestResolveQwenVAEOverride(t *testing.T) {
	// The archdb default VAE filename lacks the split_files/vae/ prefix; the
	// curated model overrides it with the real path.
	m, _ := Resolve("qwen-image-edit", "Q8_0")
	vae, ok := component(m, types.RoleVAE)
	if !ok {
		t.Fatal("no vae component")
	}
	if vae.File != "split_files/vae/qwen_image_vae.safetensors" {
		t.Fatalf("vae file = %q, want split_files/vae/qwen_image_vae.safetensors", vae.File)
	}
}

func TestResolveFluxKontext(t *testing.T) {
	m, ok := Resolve("flux.1-kontext", "Q6_K")
	if !ok {
		t.Fatal("Resolve(flux.1-kontext, Q6_K) not ok")
	}
	if m.Mode != types.ModeEdit {
		t.Fatalf("Mode = %q, want %q", m.Mode, types.ModeEdit)
	}
	diff, ok := component(m, types.RoleDiffusion)
	if !ok {
		t.Fatal("no diffusion component")
	}
	if diff.Source != "QuantStack/FLUX.1-Kontext-dev-GGUF" {
		t.Fatalf("diffusion source = %q", diff.Source)
	}
	if diff.File != "flux1-kontext-dev-Q6_K.gguf" {
		t.Fatalf("diffusion file = %q", diff.File)
	}
	for _, role := range []types.Role{types.RoleVAE, types.RoleCLIPL, types.RoleT5XXL} {
		if _, ok := component(m, role); !ok {
			t.Fatalf("missing component role %q", role)
		}
	}
	// t5xxl companion is quantized -> quant token substituted.
	t5, _ := component(m, types.RoleT5XXL)
	if !strings.Contains(t5.File, "Q6_K") {
		t.Fatalf("t5xxl file = %q, expected Q6_K substituted", t5.File)
	}
}

func TestResolveZImageTurbo(t *testing.T) {
	m, ok := Resolve("z-image-turbo", "Q8_0")
	if !ok {
		t.Fatal("Resolve(z-image-turbo, Q8_0) not ok")
	}
	if m.Mode != types.ModeGenerate {
		t.Fatalf("Mode = %q, want generate", m.Mode)
	}
	diff, _ := component(m, types.RoleDiffusion)
	if diff.Source != "leejet/Z-Image-Turbo-GGUF" {
		t.Fatalf("diffusion source = %q", diff.Source)
	}
	if diff.File != "z_image_turbo-Q8_0.gguf" {
		t.Fatalf("diffusion file = %q", diff.File)
	}
	if _, ok := component(m, types.RoleLLM); !ok {
		t.Fatal("no llm component")
	}
}

func TestResolveQuantFallback(t *testing.T) {
	// A quant not offered for the model falls back to the first curated quant.
	m, ok := Resolve("flux.1-kontext", "Q2_K")
	if !ok {
		t.Fatal("Resolve fallback not ok")
	}
	want, _ := Lookup("flux.1-kontext")
	first := want.Quants[0]
	diff, _ := component(m, types.RoleDiffusion)
	if !strings.Contains(diff.File, first) {
		t.Fatalf("fallback diffusion file = %q, want quant %q", diff.File, first)
	}
}

func TestResolveUnknown(t *testing.T) {
	if _, ok := Resolve("does-not-exist", "Q8_0"); ok {
		t.Fatal("Resolve(does-not-exist) should not be ok")
	}
}

func TestResolveNoQuantLeftover(t *testing.T) {
	// No component filename should retain an unsubstituted {quant} token for any
	// curated model at any of its own quants.
	for _, name := range Names() {
		mdl, _ := Lookup(name)
		for _, q := range mdl.Quants {
			man, ok := Resolve(name, q)
			if !ok {
				t.Fatalf("Resolve(%s,%s) not ok", name, q)
			}
			for _, c := range man.Components {
				if strings.Contains(c.File, "{quant}") {
					t.Fatalf("%s@%s role %s file still has {quant}: %q", name, q, c.Role, c.File)
				}
				if c.Source == "" || c.File == "" {
					t.Fatalf("%s@%s role %s empty source/file", name, q, c.Role)
				}
				if c.Blob != "" {
					t.Fatalf("%s@%s role %s Blob should be empty in template", name, q, c.Role)
				}
			}
		}
	}
}

// FLUX.2 klein uses Qwen3-4B as its text encoder, NOT the Mistral-3 encoder
// that flux2-dev uses — getting these crossed would load the wrong weights.
func TestResolveFlux2Klein(t *testing.T) {
	m, ok := Resolve("flux.2-klein", "Q8_0")
	if !ok {
		t.Fatal("flux.2-klein should resolve")
	}
	// FLUX.2 is a hybrid — verified against the engine: klein generates from
	// text and edits from a reference image with the same weights.
	if m.Architecture != "flux2-klein" || m.Mode != types.ModeBoth {
		t.Errorf("arch/mode = %s/%s", m.Architecture, m.Mode)
	}
	dif, _ := m.Component(types.RoleDiffusion)
	if dif.Source != "leejet/FLUX.2-klein-4B-GGUF" || dif.File != "flux-2-klein-4b-Q8_0.gguf" {
		t.Errorf("diffusion = %+v", dif)
	}
	llm, ok := m.Component(types.RoleLLM)
	if !ok || llm.Source != "unsloth/Qwen3-4B-GGUF" || llm.File != "Qwen3-4B-Q8_0.gguf" {
		t.Errorf("llm should be the Qwen3-4B encoder, got %+v", llm)
	}
	if strings.Contains(strings.ToLower(llm.File), "mistral") {
		t.Error("klein must not use the flux2-dev Mistral encoder")
	}
	vae, _ := m.Component(types.RoleVAE)
	if vae.Source != "Comfy-Org/flux2-klein" {
		t.Errorf("vae should come from the ungated mirror, got %+v", vae)
	}
	// klein is few-step; the engine should launch with those defaults baked in.
	joined := strings.Join(m.Engine.Flags, " ")
	for _, want := range []string{"--steps 4", "--cfg-scale 1", "--diffusion-fa"} {
		if !strings.Contains(joined, want) {
			t.Errorf("engine flags %q missing %q", joined, want)
		}
	}
}
