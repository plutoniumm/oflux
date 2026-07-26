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
