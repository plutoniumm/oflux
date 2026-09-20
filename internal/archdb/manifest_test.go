package archdb

import (
	"testing"

	"oflux/internal/types"
)

func TestBuildBlocksMissingRequiredRole(t *testing.T) {
	a, _ := ByName("qwen-image-edit")
	m, blockers := Build(a, BuildOpts{
		Name: "x",
		Repo: "org/Qwen-Image-Edit-2511-GGUF",
		Resolve: func(role types.Role) (types.Component, bool) {
			if role == types.RoleLLM {
				return types.Component{}, false
			}
			return types.Component{Role: role, File: "f", Source: "org/x"}, true
		},
	})
	if len(blockers) != 1 || blockers[0].Role != types.RoleLLM {
		t.Fatalf("blockers = %+v, want one for the llm role", blockers)
	}
	if len(m.Components) != 2 {
		t.Errorf("components = %+v, want the two that resolved", m.Components)
	}
}

func TestBuildSkipsUnresolvableOptionalRole(t *testing.T) {
	a, _ := ByName("sdxl") // vae is optional here
	_, blockers := Build(a, BuildOpts{
		Name: "x",
		Repo: "org/sdxl",
		Resolve: func(role types.Role) (types.Component, bool) {
			return types.Component{Role: role}, role == types.RoleDiffusion
		},
	})
	if len(blockers) != 0 {
		t.Errorf("an optional role must not block: %+v", blockers)
	}
}

func TestBaseAndRevision(t *testing.T) {
	cases := []struct{ repo, base, rev string }{
		{"unsloth/Qwen-Image-Edit-2511-GGUF", "Qwen-Image-Edit-2511", "2511"},
		{"QuantStack/Qwen-Image-Edit-GGUF", "Qwen-Image-Edit", ""},
		{"city96/FLUX.1-dev-gguf", "FLUX.1-dev", ""},
		{"leejet/FLUX.2-klein-9B-GGUF", "FLUX.2-klein-9B", ""},
		{"noslug", "noslug", ""},
	}
	for _, c := range cases {
		if got := BaseOf(c.repo); got != c.base {
			t.Errorf("BaseOf(%q) = %q, want %q", c.repo, got, c.base)
		}
		if got := RevisionOf(c.repo); got != c.rev {
			t.Errorf("RevisionOf(%q) = %q, want %q", c.repo, got, c.rev)
		}
	}
	// A release tag in the filename counts when the repo id carries none.
	if got := RevisionOf("QuantStack/Qwen-Image-Edit-GGUF", "qwen-image-edit-2509-Q8_0.gguf"); got != "2509" {
		t.Errorf("revision from filename = %q, want 2509", got)
	}
	// Four-digit numbers that are not YYMM release tags must be ignored.
	for _, s := range []string{"model-2048.gguf", "sd_xl_base_1.0", "flux1-dev-Q8_0.gguf", "upscaler-8192px"} {
		if got := RevisionOf(s); got != "" {
			t.Errorf("RevisionOf(%q) = %q, want none", s, got)
		}
	}
}
