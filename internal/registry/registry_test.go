package registry

import (
	"slices"
	"strings"
	"testing"

	"oflux/internal/archdb"
	"oflux/internal/types"
)

func TestNamesSorted(t *testing.T) {
	got := Names()
	want := []string{
		"flux.1-dev",
		"flux.1-kontext",
		"flux.1-krea",
		"flux.1-schnell",
		"flux.2-klein",
		"flux.2-klein-9b",
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

	diff, ok := m.Component(types.RoleDiffusion)
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

	if _, ok := m.Component(types.RoleVAE); !ok {
		t.Fatal("no vae component")
	}
	if _, ok := m.Component(types.RoleLLM); !ok {
		t.Fatal("no llm component")
	}

	// 2511 variant: zero-cond flag must be merged into engine model args.
	if v, ok := m.Engine.ModelArgs["qwen_image_zero_cond_t"]; !ok || v != true {
		t.Fatalf("qwen_image_zero_cond_t = %v (ok=%v), want true", v, ok)
	}
}

func TestResolveQwenVAE(t *testing.T) {
	m, _ := Resolve("qwen-image-edit", "Q8_0")
	vae, ok := m.Component(types.RoleVAE)
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
	diff, ok := m.Component(types.RoleDiffusion)
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
		if _, ok := m.Component(role); !ok {
			t.Fatalf("missing component role %q", role)
		}
	}
	// t5xxl companion is quantized -> quant token substituted.
	t5, _ := m.Component(types.RoleT5XXL)
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
	diff, _ := m.Component(types.RoleDiffusion)
	if diff.Source != "leejet/Z-Image-Turbo-GGUF" {
		t.Fatalf("diffusion source = %q", diff.Source)
	}
	if diff.File != "z_image_turbo-Q8_0.gguf" {
		t.Fatalf("diffusion file = %q", diff.File)
	}
	if _, ok := m.Component(types.RoleLLM); !ok {
		t.Fatal("no llm component")
	}
}

func TestResolveQuantFallback(t *testing.T) {
	// A label the model does not publish walks archdb's chain; nothing in the
	// table is unquantized, so an F16 request lands on the best quant there is.
	m, ok := Resolve("flux.1-kontext", "F16")
	if !ok {
		t.Fatal("Resolve fallback not ok")
	}
	diff, _ := m.Component(types.RoleDiffusion)
	if !strings.Contains(diff.File, "Q8_0") {
		t.Fatalf("fallback diffusion file = %q, want Q8_0", diff.File)
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
			man, ok := Resolve(name, string(q))
			if !ok {
				t.Fatalf("Resolve(%s,%s) not ok", name, q)
			}
			arch, _ := archdb.ByName(mdl.Arch)
			for _, role := range arch.Required {
				if _, ok := man.Component(role); !ok {
					t.Fatalf("%s@%s is missing required role %s", name, q, role)
				}
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
	if vae.Source != "Comfy-Org/vae-text-encorder-for-flux-klein-4b" {
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

// Handing a 9B klein the 4B's encoder does not fail cleanly: the engine dies
// with "'…q_norm.weight' not in model metadata" and a shape mismatch.
func TestResolveFlux2Klein9B(t *testing.T) {
	m, ok := Resolve("flux.2-klein-9b", "Q8_0")
	if !ok {
		t.Fatal("flux.2-klein-9b should resolve")
	}
	if m.Architecture != "flux2-klein" || m.Mode != types.ModeBoth {
		t.Errorf("arch/mode = %s/%s", m.Architecture, m.Mode)
	}
	dif, _ := m.Component(types.RoleDiffusion)
	if dif.Source != "leejet/FLUX.2-klein-9B-GGUF" || dif.File != "flux-2-klein-9b-Q8_0.gguf" {
		t.Errorf("diffusion = %+v", dif)
	}
	llm, ok := m.Component(types.RoleLLM)
	if !ok || llm.Source != "unsloth/Qwen3-8B-GGUF" || llm.File != "Qwen3-8B-Q8_0.gguf" {
		t.Errorf("llm should be the Qwen3-8B encoder, got %+v", llm)
	}
	four, _ := Resolve("flux.2-klein", "Q8_0")
	if llm4, _ := four.Component(types.RoleLLM); llm4.Source != "unsloth/Qwen3-4B-GGUF" {
		t.Errorf("klein-4B llm = %+v", llm4)
	}
}

// unsloth's Qwen3-8B has no Q4_0, so a Q4_0 klein-9B must not name one.
func TestResolveCompanionQuantSafetyNet(t *testing.T) {
	m, ok := Resolve("flux.2-klein-9b", "Q4_0")
	if !ok {
		t.Fatal("flux.2-klein-9b@Q4_0 should resolve")
	}
	dif, _ := m.Component(types.RoleDiffusion)
	if dif.File != "flux-2-klein-9b-Q4_0.gguf" {
		t.Errorf("diffusion = %q, want the requested Q4_0", dif.File)
	}
	llm, _ := m.Component(types.RoleLLM)
	if llm.File != "Qwen3-8B-Q4_K_M.gguf" {
		t.Errorf("llm = %q, want the nearest Q4 the encoder repo publishes", llm.File)
	}
}

// Every curated quant must name a file its source actually publishes.
func TestResolveCuratedQuantsArePublished(t *testing.T) {
	published := map[string][]archdb.Quant{
		"unsloth/Qwen-Image-Edit-2511-GGUF":        {"Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_1", "Q5_0", "Q4_K_M", "Q4_K_S", "Q4_1", "Q4_0", "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q2_K"},
		"QuantStack/FLUX.1-Kontext-dev-GGUF":       {"Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_1", "Q5_0", "Q4_K_M", "Q4_K_S", "Q4_1", "Q4_0", "Q3_K_M", "Q3_K_S", "Q2_K"},
		"QuantStack/FLUX.1-Krea-dev-GGUF":          {"Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_1", "Q5_0", "Q4_K_M", "Q4_K_S", "Q4_1", "Q4_0", "Q3_K_M", "Q3_K_S", "Q2_K"},
		"QuantStack/Qwen-Image-GGUF":               {"Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_1", "Q5_0", "Q4_K_M", "Q4_K_S", "Q4_1", "Q4_0", "Q3_K_M", "Q3_K_S", "Q2_K"},
		"city96/FLUX.1-dev-gguf":                   {"Q8_0", "Q6_K", "Q5_K_S", "Q5_1", "Q5_0", "Q4_K_S", "Q4_1", "Q4_0", "Q3_K_S", "Q2_K"},
		"city96/FLUX.1-schnell-gguf":               {"Q8_0", "Q6_K", "Q5_K_S", "Q5_1", "Q5_0", "Q4_K_S", "Q4_1", "Q4_0", "Q3_K_S", "Q2_K"},
		"leejet/FLUX.2-klein-4B-GGUF":              {"Q8_0", "Q4_0"},
		"leejet/FLUX.2-klein-9B-GGUF":              {"Q8_0", "Q4_0"},
		"leejet/Z-Image-Turbo-GGUF":                {"Q8_0", "Q6_K", "Q5_0", "Q4_K", "Q4_0", "Q3_K", "Q2_K"},
		"city96/t5-v1_1-xxl-encoder-gguf":          {"Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "Q3_K_L", "Q3_K_M", "Q3_K_S"},
		"mradermacher/Qwen2.5-VL-7B-Instruct-GGUF": {"Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q2_K"},
		"unsloth/Qwen3-4B-GGUF":                    {"Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "Q4_1", "Q4_0", "Q3_K_M", "Q3_K_S", "Q2_K"},
		"unsloth/Qwen3-8B-GGUF":                    {"Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "Q4_1", "Q3_K_M", "Q3_K_S", "Q2_K"},
		"unsloth/Qwen3-4B-Instruct-2507-GGUF":      {"Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "Q4_1", "Q4_0", "Q3_K_M", "Q3_K_S", "Q2_K"},
	}
	for _, name := range Names() {
		mdl, _ := Lookup(name)
		for _, q := range mdl.Quants {
			man, _ := Resolve(name, string(q))
			for _, c := range man.Components {
				have, known := published[c.Source]
				if !known {
					continue // fixed single-file source (a VAE or CLIP)
				}
				got, ok := archdb.ParseQuant(c.File)
				if !ok || !slices.Contains(have, got) {
					t.Errorf("%s@%s: %s resolves %s/%s, which that repo does not publish", name, q, c.Role, c.Source, c.File)
				}
			}
		}
	}
}

// The friendly name hides which checkpoint a model actually is.
func TestResolveBaseAndRevision(t *testing.T) {
	cases := map[string][2]string{
		"qwen-image-edit": {"Qwen-Image-Edit-2511", "2511"},
		"flux.1-dev":      {"FLUX.1-dev", ""},
		"flux.2-klein-9b": {"FLUX.2-klein-9B", ""},
		"z-image-turbo":   {"Z-Image-Turbo", ""},
	}
	for name, want := range cases {
		m, ok := Resolve(name, "Q8_0")
		if !ok {
			t.Fatalf("Resolve(%s) not ok", name)
		}
		if m.Base != want[0] || m.Revision != want[1] {
			t.Errorf("%s: base/revision = %q/%q, want %q/%q", name, m.Base, m.Revision, want[0], want[1])
		}
	}
}
