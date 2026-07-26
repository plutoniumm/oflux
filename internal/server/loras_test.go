package server

import (
	"testing"

	"oflux/internal/store"
	"oflux/internal/types"
)

func qwenEditManifest() types.Manifest {
	return types.Manifest{Name: "qwen-image-edit", Architecture: "qwen-image-edit", Mode: types.ModeEdit}
}

func TestBuildImgGenLoraPathAndScale(t *testing.T) {
	req := ImageRequest{
		Prompt: "make it winter",
		Loras: []LoraRef{
			{Name: "qwen-edit-lightning-4step"},
			{Name: "custom-style", Scale: ptrF(0.6)},
		},
	}
	ig := buildImgGen(qwenEditManifest(), req, types.ModeGenerate)

	if len(ig.Loras) != 2 {
		t.Fatalf("got %d loras, want 2", len(ig.Loras))
	}
	// The engine resolves lora[].path against --lora-model-dir and needs the
	// extension; a bare name silently fails to load.
	if ig.Loras[0].Path != store.LoraFileName("qwen-edit-lightning-4step") {
		t.Errorf("path = %q, want the .safetensors filename", ig.Loras[0].Path)
	}
	if ig.Loras[0].Multiplier != 1.0 {
		t.Errorf("default multiplier = %v, want 1.0", ig.Loras[0].Multiplier)
	}
	if ig.Loras[1].Multiplier != 0.6 {
		t.Errorf("multiplier = %v, want 0.6", ig.Loras[1].Multiplier)
	}
	// The tag must not be smuggled into the prompt: sd-server does not parse
	// <lora:...> syntax and would tokenize it as literal text.
	if ig.Prompt != "make it winter" {
		t.Errorf("prompt was rewritten: %q", ig.Prompt)
	}
}

// A step-distillation LoRA changes the sampling regime. The engine was launched
// with the base model's 20 steps at cfg 2.5, so without this the adapter
// produces burnt output and nothing explains why.
func TestBuildImgGenCuratedLoraSuppliesSamplingDefaults(t *testing.T) {
	req := ImageRequest{
		Prompt: "a fox",
		Loras:  []LoraRef{{Name: "qwen-edit-lightning-4step"}},
	}
	ig := buildImgGen(qwenEditManifest(), req, types.ModeGenerate)

	if ig.SampleParams == nil || ig.SampleParams.SampleSteps == nil {
		t.Fatal("expected the adapter's step count to be applied")
	}
	if *ig.SampleParams.SampleSteps != 4 {
		t.Errorf("steps = %d, want 4", *ig.SampleParams.SampleSteps)
	}
	if ig.SampleParams.Guidance == nil || ig.SampleParams.Guidance.TxtCFG == nil {
		t.Fatal("expected the adapter's cfg to be applied")
	}
	if *ig.SampleParams.Guidance.TxtCFG != 1.0 {
		t.Errorf("txt_cfg = %v, want 1.0", *ig.SampleParams.Guidance.TxtCFG)
	}
}

func TestBuildImgGenExplicitOverridesBeatLoraDefaults(t *testing.T) {
	req := ImageRequest{
		Prompt: "a fox",
		Steps:  ptrI(12),
		CFG:    ptrF(3.5),
		Loras:  []LoraRef{{Name: "qwen-edit-lightning-4step"}},
	}
	ig := buildImgGen(qwenEditManifest(), req, types.ModeGenerate)

	if *ig.SampleParams.SampleSteps != 12 {
		t.Errorf("steps = %d, want the caller's 12", *ig.SampleParams.SampleSteps)
	}
	if *ig.SampleParams.Guidance.TxtCFG != 3.5 {
		t.Errorf("txt_cfg = %v, want the caller's 3.5", *ig.SampleParams.Guidance.TxtCFG)
	}
}

// An uncurated LoRA is applied verbatim with no sampling opinion attached.
func TestBuildImgGenUnknownLoraLeavesSamplingAlone(t *testing.T) {
	req := ImageRequest{Prompt: "a fox", Loras: []LoraRef{{Name: "some-style-lora"}}}
	ig := buildImgGen(qwenEditManifest(), req, types.ModeGenerate)

	if len(ig.Loras) != 1 || ig.Loras[0].Path != "some-style-lora.safetensors" {
		t.Fatalf("loras = %+v", ig.Loras)
	}
	if ig.SampleParams != nil {
		t.Errorf("no sampling override expected, got %+v", ig.SampleParams)
	}
}

// Flux takes its cfg through distilled_guidance, not txt_cfg.
func TestBuildImgGenLoraCFGRoutesByArchitecture(t *testing.T) {
	m := types.Manifest{Name: "flux.1-dev", Architecture: "flux", Mode: types.ModeGenerate}
	req := ImageRequest{Prompt: "a fox", Loras: []LoraRef{{Name: "flux-turbo-8step"}}}
	ig := buildImgGen(m, req, types.ModeGenerate)

	if *ig.SampleParams.SampleSteps != 8 {
		t.Errorf("steps = %d, want 8", *ig.SampleParams.SampleSteps)
	}
	if ig.SampleParams.Guidance.DistilledGuidance == nil {
		t.Fatal("flux cfg must go to distilled_guidance")
	}
	if ig.SampleParams.Guidance.TxtCFG != nil {
		t.Error("flux must not set txt_cfg")
	}
}

func TestBuildImgGenNoLorasIsUnchanged(t *testing.T) {
	ig := buildImgGen(qwenEditManifest(), ImageRequest{Prompt: "a fox"}, types.ModeGenerate)
	if len(ig.Loras) != 0 {
		t.Errorf("expected no lora entries, got %+v", ig.Loras)
	}
	if ig.SampleParams != nil {
		t.Errorf("expected no sampling overrides, got %+v", ig.SampleParams)
	}
}
