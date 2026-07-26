package compat

import (
	"context"
	"strings"
	"testing"

	"oflux/internal/types"
)

func TestFewStepDefaults(t *testing.T) {
	cases := []struct {
		file      string
		wantSteps int // 0 = no override expected
	}{
		// Explicit step counts win wherever they appear.
		{"Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", 4},
		{"Hyper-FLUX.1-dev-8steps-lora.safetensors", 8},
		{"Qwen-Rapid-AIO-v1-8step.safetensors", 8},
		{"nitro-1step.gguf", 1},
		// Known step-distilled families with no count in the name.
		{"v19/Qwen-Rapid-AIO-NSFW-v19_Q8_0.gguf", 4},
		{"lightning-mystery-build.safetensors", 4},
		// Ordinary weights must be left alone.
		{"qwen-image-edit-2511-Q8_0.gguf", 0},
		{"flux1-dev-Q6_K.gguf", 0},
		{"z_image_turbo-Q8_0.gguf", 0},
		// A step count out of plausible range is not a distillation signal;
		// "2511" is a model date, not 2511 steps.
		{"qwen-image-edit-2511steps.gguf", 0},
	}
	for _, tc := range cases {
		got, note := fewStepDefaults(tc.file)
		if tc.wantSteps == 0 {
			if got != nil {
				t.Errorf("%s: expected no override, got %v", tc.file, got)
			}
			continue
		}
		if got == nil {
			t.Errorf("%s: expected %d-step override, got none", tc.file, tc.wantSteps)
			continue
		}
		if got["steps"] != tc.wantSteps {
			t.Errorf("%s: steps = %v, want %d", tc.file, got["steps"], tc.wantSteps)
		}
		if got["cfg_scale"] != 1.0 {
			t.Errorf("%s: cfg_scale = %v, want 1.0", tc.file, got["cfg_scale"])
		}
		if note == "" {
			t.Errorf("%s: an override must be explained in the notes", tc.file)
		}
	}
}

// rapidAIOFetcher mimics a GGUF repo that publishes many builds of the same
// model — the shape that made quant-only selection pick an arbitrary version.
type rapidAIOFetcher struct{ paths []string }

func (f rapidAIOFetcher) Tree(ctx context.Context, repo, rev string) ([]types.HFFile, error) {
	out := make([]types.HFFile, 0, len(f.paths))
	for _, p := range f.paths {
		out = append(out, types.HFFile{Path: p, IsLFS: true, LFSOID: "deadbeef", Size: 1})
	}
	return out, nil
}

func (f rapidAIOFetcher) ReadFile(ctx context.Context, repo, rev, path string, max int64) ([]byte, error) {
	return nil, context.Canceled // no diffusers config; forces keyword detection
}

var rapidRepo = rapidAIOFetcher{paths: []string{
	"v11.1/Qwen-Rapid-AIO-NSFW-v11.1_Q8_0.gguf",
	"v19/Qwen-Rapid-AIO-NSFW-v19_Q8_0.gguf",
	"v23/Qwen-Rapid-NSFW-v23_Q8_0.gguf",
}}

// Without a pinned file, quant preference alone resolves to whichever build
// sorts first — here the oldest one. That is the behaviour --file exists to fix.
func TestInspectMultiBuildRepoPicksArbitraryVersion(t *testing.T) {
	v, err := Inspect(context.Background(), rapidRepo, "Novice25/Qwen-Image-Edit-Rapid-AIO-GGUF", []string{"Q8_0"})
	if err != nil {
		t.Fatal(err)
	}
	if !v.Compatible {
		t.Fatalf("expected compatible, got %+v", v.Blockers)
	}
	diff, _ := v.Manifest.Component(types.RoleDiffusion)
	if diff.File != "v11.1/Qwen-Rapid-AIO-NSFW-v11.1_Q8_0.gguf" {
		t.Fatalf("diffusion file = %q", diff.File)
	}
}

func TestInspectFilePinsExactBuild(t *testing.T) {
	v, err := InspectFile(context.Background(), rapidRepo, "Novice25/Qwen-Image-Edit-Rapid-AIO-GGUF",
		[]string{"Q8_0"}, "v23/Qwen-Rapid-NSFW-v23_Q8_0.gguf")
	if err != nil {
		t.Fatal(err)
	}
	if !v.Compatible {
		t.Fatalf("expected compatible, got %+v", v.Blockers)
	}
	if v.Manifest.Architecture != "qwen-image-edit" || !v.Manifest.Mode.CanEdit() {
		t.Errorf("arch/mode = %s/%s", v.Manifest.Architecture, v.Manifest.Mode)
	}
	diff, _ := v.Manifest.Component(types.RoleDiffusion)
	if diff.File != "v23/Qwen-Rapid-NSFW-v23_Q8_0.gguf" {
		t.Fatalf("diffusion file = %q, want the pinned v23 build", diff.File)
	}

	// The Rapid line is a 4-step merge: launching it with qwen-image-edit's
	// 20-step / cfg-2.5 defaults is what produces burnt output.
	joined := strings.Join(v.Manifest.Engine.Flags, " ")
	for _, want := range []string{"--steps 4", "--cfg-scale 1"} {
		if !strings.Contains(joined, want) {
			t.Errorf("engine flags %q missing %q", joined, want)
		}
	}
}

func TestInspectFileUnknownFile(t *testing.T) {
	v, err := InspectFile(context.Background(), rapidRepo, "Novice25/Qwen-Image-Edit-Rapid-AIO-GGUF",
		[]string{"Q8_0"}, "v99/nope.gguf")
	if err != nil {
		t.Fatal(err)
	}
	if v.Compatible {
		t.Fatal("a missing file must not resolve as compatible")
	}
}
