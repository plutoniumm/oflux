package registry

import "slices"

// Lora is a curated LoRA adapter: a friendly name pinned to an exact Hugging
// Face repo and file, plus the architectures it is trained against.
//
// Unlike models, a LoRA is not part of a manifest and needs no engine restart —
// sd-server applies it per request from --lora-model-dir. That makes LoRAs the
// cheap way to change a model's behaviour: a few-step distillation adapter
// turns a 20-step base model into a 4-step one for well under a gigabyte,
// instead of downloading a separately-merged multi-gigabyte checkpoint.
type Lora struct {
	Name        string   // friendly name; also the on-disk filename stem
	Source      string   // Hugging Face repo id
	File        string   // exact path within Source
	Archs       []string // archdb arch names this adapter is trained for
	Steps       int      // sampling steps this adapter is distilled for (0 = not a step distiller)
	CFG         float64  // cfg scale to use with it (0 = leave the model default)
	Description string
}

// curatedLoras is the hand-verified LoRA table. Every Source/File pair was
// checked against the live Hugging Face repo tree.
var curatedLoras = []Lora{
	{
		Name:        "qwen-edit-lightning-4step",
		Source:      "lightx2v/Qwen-Image-Edit-2511-Lightning",
		File:        "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
		Archs:       []string{"qwen-image-edit"},
		Steps:       4,
		CFG:         1.0,
		Description: "4-step distillation for Qwen-Image-Edit 2511 (~5x faster edits).",
	},
	{
		Name:        "qwen-edit-lightning-8step",
		Source:      "lightx2v/Qwen-Image-Edit-2511-Lightning",
		File:        "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors",
		Archs:       []string{"qwen-image-edit"},
		Steps:       8,
		CFG:         1.0,
		Description: "8-step distillation for Qwen-Image-Edit 2511 (better fidelity than 4-step).",
	},
	{
		Name:        "qwen-image-lightning-4step",
		Source:      "lightx2v/Qwen-Image-Lightning",
		File:        "Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors",
		Archs:       []string{"qwen-image"},
		Steps:       4,
		CFG:         1.0,
		Description: "4-step distillation for Qwen-Image text-to-image.",
	},
	{
		Name:   "flux-turbo-8step",
		Source: "alimama-creative/FLUX.1-Turbo-Alpha",
		File:   "diffusion_pytorch_model.safetensors",
		// Trained on FLUX.1-dev; Krea is a dev finetune and shares its weights
		// layout, so the same adapter applies.
		Archs:       []string{"flux"},
		Steps:       8,
		CFG:         1.0,
		Description: "8-step turbo distillation for FLUX.1 dev/Krea.",
	},
	{
		Name:        "flux-hyper-8step",
		Source:      "ByteDance/Hyper-SD",
		File:        "Hyper-FLUX.1-dev-8steps-lora.safetensors",
		Archs:       []string{"flux"},
		Steps:       8,
		CFG:         1.0,
		Description: "ByteDance Hyper-SD 8-step distillation for FLUX.1 dev/Krea.",
	},
}

// LoraNames returns the curated LoRA names, sorted.
func LoraNames() []string {
	out := make([]string, 0, len(curatedLoras))
	for _, l := range curatedLoras {
		out = append(out, l.Name)
	}
	slices.Sort(out)
	return out
}

// LookupLora returns the curated LoRA with the given name.
func LookupLora(name string) (Lora, bool) {
	for _, l := range curatedLoras {
		if l.Name == name {
			return l, true
		}
	}
	return Lora{}, false
}

// LorasForArch returns the curated LoRAs trained for an architecture, sorted by
// name. Used to tell a user which adapters are worth installing for a model
// they already have.
func LorasForArch(arch string) []Lora {
	var out []Lora
	for _, l := range curatedLoras {
		if slices.Contains(l.Archs, arch) {
			out = append(out, l)
		}
	}
	slices.SortFunc(out, func(a, b Lora) int {
		switch {
		case a.Name < b.Name:
			return -1
		case a.Name > b.Name:
			return 1
		}
		return 0
	})
	return out
}

// AllLoras returns a copy of the curated LoRA table.
func AllLoras() []Lora {
	out := make([]Lora, len(curatedLoras))
	copy(out, curatedLoras)
	return out
}
