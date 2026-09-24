package archdb

import (
	"slices"
	"strings"
	"testing"

	"oflux/internal/types"
)

func TestLookupByClassName(t *testing.T) {
	cases := map[string]string{
		"FluxPipeline":              "flux",
		"FluxKontextPipeline":       "flux-kontext",
		"QwenImageEditPipeline":     "qwen-image-edit",
		"QwenImagePipeline":         "qwen-image",
		"ZImagePipeline":            "z-image",
		"StableDiffusionXLPipeline": "sdxl",
	}
	for cn, want := range cases {
		a, ok := Lookup(cn)
		if !ok || a.Name != want {
			t.Errorf("Lookup(%q) = %q,%v; want %q", cn, a.Name, ok, want)
		}
	}
	if _, ok := Lookup("NopePipeline"); ok {
		t.Error("Lookup of unknown class should fail")
	}
}

func TestMatchKeywordSpecificBeforeGeneric(t *testing.T) {
	cases := map[string]string{
		"city96/FLUX.1-dev-gguf":             "flux",
		"QuantStack/FLUX.1-Kontext-dev-GGUF": "flux-kontext",
		"unsloth/Qwen-Image-Edit-2511-GGUF":  "qwen-image-edit",
		"QuantStack/Qwen-Image-GGUF":         "qwen-image",
		"leejet/Z-Image-Turbo-GGUF":          "z-image",
	}
	for repo, want := range cases {
		a, ok := MatchKeyword(repo)
		if !ok || a.Name != want {
			t.Errorf("MatchKeyword(%q) = %q,%v; want %q", repo, a.Name, ok, want)
		}
	}
}

func TestBaseFlagsHasRolePlaceholders(t *testing.T) {
	a, _ := ByName("qwen-image-edit")
	flags := a.BaseFlags()
	// diffusion + vae + llm role flags with placeholders, then the switch flag.
	wantPairs := [][2]string{
		{"--diffusion-model", "{diffusion}"},
		{"--vae", "{vae}"},
		{"--llm", "{llm}"},
	}
	for _, wp := range wantPairs {
		i := slices.Index(flags, wp[0])
		if i < 0 || i+1 >= len(flags) || flags[i+1] != wp[1] {
			t.Errorf("flags %v missing pair %v", flags, wp)
		}
	}
	if !slices.Contains(flags, "--diffusion-fa") {
		t.Errorf("flags %v missing --diffusion-fa", flags)
	}
	// mmproj must NOT be present (2511 target).
	if slices.Contains(flags, "--llm_vision") {
		t.Errorf("qwen-image-edit should not emit --llm_vision: %v", flags)
	}
}

func TestEngineSpecBakesSamplingFlags(t *testing.T) {
	a, _ := ByName("qwen-image-edit")
	spec := a.EngineSpec()
	joined := strings.Join(spec.Flags, " ")
	for _, want := range []string{"--cfg-scale 2.5", "--flow-shift 3", "--steps 20", "--sampling-method euler"} {
		if !strings.Contains(joined, want) {
			t.Errorf("EngineSpec flags %q missing %q", joined, want)
		}
	}
}

func TestNumStr(t *testing.T) {
	cases := map[any]string{
		2.5:        "2.5",
		float64(3): "3",
		20:         "20",
		int64(8):   "8",
		1.0:        "1",
	}
	for in, want := range cases {
		if got := numStr(in); got != want {
			t.Errorf("numStr(%v) = %q; want %q", in, got, want)
		}
	}
}

func TestFlagNameMapping(t *testing.T) {
	if FlagName(types.RoleMMProj) != "--llm_vision" {
		t.Errorf("mmproj flag = %q", FlagName(types.RoleMMProj))
	}
	if FlagName(types.RoleT5XXL) != "--t5xxl" {
		t.Errorf("t5xxl flag = %q", FlagName(types.RoleT5XXL))
	}
}

// One klein ships as flux2_klein_9b_Q8_0.gguf, FLUX.2-klein and "FLUX 2 Klein";
// matching literally sent the underscore spelling to the flux2 arch.
func TestMatchKeywordSeparatorSpellings(t *testing.T) {
	for _, a := range registry {
		for _, kw := range a.Keywords {
			for _, sep := range []string{".", "-", "_", " ", ""} {
				spelled := respell(kw, sep)
				got, ok := MatchKeyword("someone/" + spelled + "-Q8_0.gguf")
				if !ok || got.Name != a.Name {
					t.Errorf("MatchKeyword(%q) = %q,%v; want %q", spelled, got.Name, ok, a.Name)
				}
			}
		}
	}
}

func TestMatchKeywordKleinUnderscores(t *testing.T) {
	for _, in := range []string{"flux2_klein_9b_Q8_0.gguf", "FLUX.2-klein-4B", "flux 2 klein", "flux-2-klein-9b-Q4_0.gguf"} {
		a, ok := MatchKeyword(in)
		if !ok || a.Name != "flux2-klein" {
			t.Errorf("MatchKeyword(%q) = %q,%v; want flux2-klein", in, a.Name, ok)
		}
	}
}

func respell(kw, sep string) string {
	out := kw
	for _, s := range []string{".", "-", "_", " "} {
		out = strings.ReplaceAll(out, s, sep)
	}
	return out
}

// klein-4B and klein-9B are one architecture with two text encoders.
func TestArchForKleinVariant(t *testing.T) {
	a, _ := ByName("flux2-klein")
	if got := a.Companions[types.RoleLLM].Source; got != "unsloth/Qwen3-4B-GGUF" {
		t.Errorf("klein default llm = %q", got)
	}
	for _, hint := range []string{"unsloth/FLUX.2-klein-9B-GGUF", "flux2_klein_9b_Q8_0.gguf", "flux-2-klein-9b-Q4_0.gguf"} {
		nine := a.For(hint)
		if got := nine.Companions[types.RoleLLM].Source; got != "unsloth/Qwen3-8B-GGUF" {
			t.Errorf("For(%q) llm = %q, want the Qwen3-8B encoder", hint, got)
		}
		if nine.Name != a.Name || nine.Mode != a.Mode {
			t.Errorf("a variant must stay the same architecture: %q/%q", nine.Name, nine.Mode)
		}
	}
	if got := a.Companions[types.RoleLLM].Source; got != "unsloth/Qwen3-4B-GGUF" {
		t.Errorf("For mutated the shared arch: llm = %q", got)
	}
	if got := a.For("leejet/FLUX.2-klein-4B-GGUF").Companions[types.RoleLLM].Source; got != "unsloth/Qwen3-4B-GGUF" {
		t.Errorf("klein-4B llm = %q", got)
	}
	flux, _ := ByName("flux")
	if got := flux.For("anything-9b"); got.Companions[types.RoleT5XXL].Source != flux.Companions[types.RoleT5XXL].Source {
		t.Errorf("an arch with no variants must be unchanged: %+v", got.Companions)
	}
}

// Manifest.Mode is frozen at pull time; archdb is not.
func TestModeOf(t *testing.T) {
	if got := ModeOf("flux2-klein", types.ModeGenerate); got != types.ModeBoth {
		t.Errorf("ModeOf(flux2-klein) = %q, want both (not the stale manifest value)", got)
	}
	if got := ModeOf("qwen-image-edit", types.ModeEdit); got != types.ModeBoth {
		t.Errorf("ModeOf(qwen-image-edit) = %q, want both", got)
	}
	if got := ModeOf("something-we-removed", types.ModeEdit); got != types.ModeEdit {
		t.Errorf("an unknown arch must fall back to the manifest: %q", got)
	}
}

// A companion's declared labels must be ones ParseQuant recognizes.
func TestCompanionQuantsAreKnownLabels(t *testing.T) {
	for _, a := range registry {
		for role, c := range a.Companions {
			for _, q := range c.Quants {
				if _, ok := ParseQuant(string(q)); !ok {
					t.Errorf("%s/%s declares unknown quant %q", a.Name, role, q)
				}
			}
			if len(c.Quants) > 0 && !c.Quantized {
				t.Errorf("%s/%s lists quants but is not quantized", a.Name, role)
			}
		}
	}
}

// Qwen-Image 2.1 contains "qwen-image" once separators are stripped, so without
// priority it resolves to the plain qwen-image arch — which would load the 2.5-VL
// encoder and the older VAE, the same class of mistake klein hit with flux2.
func TestMatchKeywordQwenImage21BeforeQwenImage(t *testing.T) {
	for _, in := range []string{
		"leejet/Qwen-Image-2.1-GGUF",
		"qwen_image_2.1-Q4_K.gguf",
		"abenzerps/Qwen-Image-2.1-Uncensored-GGUF",
		"qwen-image-2.1-UC-Q8_0.gguf",
		"Qwen/Qwen-Image-2.1",
	} {
		a, ok := MatchKeyword(in)
		if !ok || a.Name != "qwen-image-2.1" {
			t.Errorf("MatchKeyword(%q) = %q,%v; want qwen-image-2.1", in, a.Name, ok)
		}
	}
	// ...and the older checkpoints must not be dragged forward into 2.1.
	for in, want := range map[string]string{
		"unsloth/Qwen-Image-Edit-2511-GGUF": "qwen-image-edit",
		"QuantStack/Qwen-Image-GGUF":        "qwen-image",
	} {
		if a, ok := MatchKeyword(in); !ok || a.Name != want {
			t.Errorf("MatchKeyword(%q) = %q,%v; want %q", in, a.Name, ok, want)
		}
	}
}

// 2.1's VAE and encoder are not the ones the older Qwen arches use, and its
// vision tower is what makes editing work at all.
func TestQwenImage21Components(t *testing.T) {
	a, ok := ByName("qwen-image-2.1")
	if !ok {
		t.Fatal("qwen-image-2.1 not in archdb")
	}
	if !a.Mode.CanEdit() || !a.Mode.CanGenerate() {
		t.Errorf("mode = %q, want both", a.Mode)
	}
	if !a.Alpha {
		t.Error("qwen-image-2.1 emits RGBA; Alpha should be set")
	}
	if vae := a.Companions[types.RoleVAE]; vae.FilePattern == qwenImageVAE.FilePattern {
		t.Error("2.1 must not reuse the older qwen_image_vae: the docs say they are not interchangeable")
	}
	if llm := a.Companions[types.RoleLLM]; llm.Source != "Qwen/Qwen3-VL-8B-Instruct-GGUF" {
		t.Errorf("llm = %q, want the Qwen3-VL-8B encoder", llm.Source)
	}
	if !slices.Contains(a.Roles(), types.RoleMMProj) {
		t.Error("mmproj must be resolved: without --llm_vision the engine cannot read a reference image")
	}
}
