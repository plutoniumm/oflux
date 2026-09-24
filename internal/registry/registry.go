// Package registry is the curated model registry: hand-verified templates for
// the blessed diffusion models so that `oflux pull <friendly-name>` just works.
//
// Each Model names its archdb architecture and pins the exact Hugging Face repo
// id and filename pattern for its diffusion weights; shared components come
// from the architecture's Companions unless overridden here. Every repo id,
// filename and quant list was verified against the live Hugging Face tree.
//
// This package performs no I/O: it turns a curated name plus a quant label into
// a manifest template, with Component.Blob left for the puller to fill.
package registry

import (
	"maps"
	"slices"
	"strings"

	"oflux/internal/archdb"
	"oflux/internal/types"
)

// Model is a curated entry: which archdb arch it uses and where its
// model-specific (diffusion) weights live. Shared components come from the
// arch's Companions unless overridden here.
type Model struct {
	Name           string                          // friendly name, e.g. "qwen-image-edit"
	Arch           string                          // archdb arch name
	DiffSource     string                          // HF repo id hosting the diffusion GGUF
	DiffPattern    string                          // filename with "{quant}" token
	Quants         []archdb.Quant                  // quant labels DiffSource publishes, best first
	ExtraModelArgs map[string]any                  // merged over arch.ModelArgs
	Overrides      map[types.Role]archdb.Companion // optional per-role source overrides for shared components
	Description    string
}

// curated is the hand-verified model table. Order is irrelevant; Names() sorts.
// The quant lists describe the DIFFUSION repo alone: companions resolve through
// archdb.PickQuant against their own labels, so nothing is hand-intersected.
var curated = []Model{
	{
		Name:        "qwe-2511",
		Arch:        "qwen-image-edit",
		DiffSource:  "unsloth/Qwen-Image-Edit-2511-GGUF",
		DiffPattern: "qwen-image-edit-2511-{quant}.gguf",
		Quants:      []archdb.Quant{"Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_1", "Q5_0", "Q4_K_M", "Q4_K_S", "Q4_1", "Q4_0", "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q2_K"},
		// 2511 uses zero conditioning-timestep sampling.
		ExtraModelArgs: map[string]any{"qwen_image_zero_cond_t": true},
		Description:    "Qwen-Image-Edit 2511 - instruction image editing (Qwen2.5-VL encoder).",
	},
	{
		Name:        "flx-1-kontext",
		Arch:        "flux-kontext",
		DiffSource:  "QuantStack/FLUX.1-Kontext-dev-GGUF",
		DiffPattern: "flux1-kontext-dev-{quant}.gguf",
		Quants:      []archdb.Quant{"Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_1", "Q5_0", "Q4_K_M", "Q4_K_S", "Q4_1", "Q4_0", "Q3_K_M", "Q3_K_S", "Q2_K"},
		Description: "FLUX.1 Kontext [dev] - in-context image editing.",
	},
	{
		Name:        "flx-2-klein-4b",
		Arch:        "flux2-klein",
		DiffSource:  "leejet/FLUX.2-klein-4B-GGUF", // sd.cpp author's own conversion
		DiffPattern: "flux-2-klein-4b-{quant}.gguf",
		Quants:      []archdb.Quant{"Q8_0", "Q4_0"},
		Description: "FLUX.2 klein 4B - few-step text-to-image and editing (Qwen3-4B encoder).",
	},
	{
		// The 9B klein takes the Qwen3-8B encoder; archdb's klein variant picks
		// it up from the "klein-9b" in the repo id and filename.
		Name:        "flx-2-klein-9b",
		Arch:        "flux2-klein",
		DiffSource:  "leejet/FLUX.2-klein-9B-GGUF",
		DiffPattern: "flux-2-klein-9b-{quant}.gguf",
		Quants:      []archdb.Quant{"Q8_0", "Q4_0"},
		Description: "FLUX.2 klein 9B - few-step text-to-image and editing (Qwen3-8B encoder).",
	},
	{
		Name:        "zim-1-turbo",
		Arch:        "z-image",
		DiffSource:  "leejet/Z-Image-Turbo-GGUF",
		DiffPattern: "z_image_turbo-{quant}.gguf",
		Quants:      []archdb.Quant{"Q8_0", "Q6_K", "Q5_0", "Q4_K", "Q4_0", "Q3_K", "Q2_K"},
		Description: "Z-Image Turbo - fast few-step text-to-image (Qwen3-4B encoder).",
	},
	{
		Name:        "flx-1-krea",
		Arch:        "flux",
		DiffSource:  "QuantStack/FLUX.1-Krea-dev-GGUF",
		DiffPattern: "flux1-krea-dev-{quant}.gguf",
		Quants:      []archdb.Quant{"Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_1", "Q5_0", "Q4_K_M", "Q4_K_S", "Q4_1", "Q4_0", "Q3_K_M", "Q3_K_S", "Q2_K"},
		Description: "FLUX.1 Krea [dev] - photographic text-to-image.",
	},
	{
		Name:        "flx-1-dev",
		Arch:        "flux",
		DiffSource:  "city96/FLUX.1-dev-gguf",
		DiffPattern: "flux1-dev-{quant}.gguf",
		// city96 flux repos ship the *_S K-quants, no *_M.
		Quants:      []archdb.Quant{"Q8_0", "Q6_K", "Q5_K_S", "Q5_1", "Q5_0", "Q4_K_S", "Q4_1", "Q4_0", "Q3_K_S", "Q2_K"},
		Description: "FLUX.1 [dev] - guidance-distilled text-to-image.",
	},
	{
		Name:        "flx-1-schnell",
		Arch:        "flux",
		DiffSource:  "city96/FLUX.1-schnell-gguf",
		DiffPattern: "flux1-schnell-{quant}.gguf",
		Quants:      []archdb.Quant{"Q8_0", "Q6_K", "Q5_K_S", "Q5_1", "Q5_0", "Q4_K_S", "Q4_1", "Q4_0", "Q3_K_S", "Q2_K"},
		Description: "FLUX.1 [schnell] - fast few-step text-to-image (Apache-2.0).",
	},
	{
		Name:        "qwe-2.1",
		Arch:        "qwen-image-2.1",
		DiffSource:  "leejet/Qwen-Image-2.1-GGUF", // sd.cpp author's own conversion
		DiffPattern: "qwen_image_2.1-{quant}.gguf",
		// leejet spells the K-quants bare: Q4_K and Q3_K, no _M/_S variants.
		Quants:      []archdb.Quant{"Q8_0", "Q6_K", "Q5_0", "Q4_K", "Q4_0", "Q3_K", "Q2_K"},
		Description: "Qwen-Image 2.1 - unified text-to-image and editing, up to 10 reference images, native RGBA (Qwen3-VL-8B encoder). Non-commercial license.",
	},
	{
		// Same architecture, refusal behaviour ablated by a third party. Pinned
		// separately rather than as a quant of the base so `oflux ls` never
		// hides which weights are installed.
		Name:        "qwe-2.1-uc",
		Arch:        "qwen-image-2.1",
		DiffSource:  "abenzerps/Qwen-Image-2.1-Uncensored-GGUF",
		DiffPattern: "qwen-image-2.1-UC-{quant}.gguf",
		Quants:      []archdb.Quant{"BF16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0"},
		// Ablating the diffusion weights alone leaves half the job undone: the
		// prompt is read by Qwen3-VL, so refusal behaviour lives in the encoder
		// too. This pairs the uncensored transformer with the matching
		// abliterated encoder and its vision tower.
		Overrides: map[types.Role]archdb.Companion{
			types.RoleLLM: {
				Source: "pottokao/Qwen-Image-2.1-Text-Encoder-Heretic-GGUF", FilePattern: "qwen3vl_8b_heretic-{quant}.gguf", Quantized: true,
				Quants: []archdb.Quant{"Q8_0", "Q6_K", "Q4_K_M"},
			},
			types.RoleMMProj: {Source: "pottokao/Qwen-Image-2.1-Text-Encoder-Heretic-GGUF", FilePattern: "mmproj-qwen3vl_8b_heretic-f16.gguf"},
		},
		Description: "Qwen-Image 2.1, abliterated - as above with the refusal direction removed, paired with the matching abliterated text encoder. Third-party weights.",
	},
	{
		Name:        "qwi-1",
		Arch:        "qwen-image",
		DiffSource:  "QuantStack/Qwen-Image-GGUF",
		DiffPattern: "Qwen_Image-{quant}.gguf",
		Quants:      []archdb.Quant{"Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_1", "Q5_0", "Q4_K_M", "Q4_K_S", "Q4_1", "Q4_0", "Q3_K_M", "Q3_K_S", "Q2_K"},
		Description: "Qwen-Image - text-to-image (Qwen2.5-VL encoder).",
	},
}

// source is the curated model's own diffusion repo, an override, or the
// architecture's companion. ok=false when nothing publishes the role.
func (m Model) source(role types.Role, arch archdb.Arch) (archdb.Companion, bool) {
	if role == types.RoleDiffusion {
		// The diffusion weights are the model; they are never a companion.
		return archdb.Companion{Source: m.DiffSource, FilePattern: m.DiffPattern, Quantized: true, Quants: m.Quants}, true
	}
	if c, ok := m.Overrides[role]; ok {
		return c, true
	}
	c, ok := arch.Companions[role]
	return c, ok
}

func modelName(m Model) string { return m.Name }
func loraName(l Lora) string   { return l.Name }

func sortedNames[T any](table []T, name func(T) string) []string {
	out := make([]string, len(table))
	for i, e := range table {
		out[i] = name(e)
	}
	slices.Sort(out)
	return out
}

// byName scans: the tables are tiny and read far more often than they change.
func byName[T any](table []T, name func(T) string, want string) (T, bool) {
	for _, e := range table {
		if name(e) == want {
			return e, true
		}
	}
	var zero T
	return zero, false
}

func Names() []string { return sortedNames(curated, modelName) }

func Lookup(name string) (Model, bool) { return byName(curated, modelName, name) }

// Resolve builds a manifest template for name at the given quant, falling back
// through archdb.QuantChain when the model does not publish it. ok=false for an
// unknown name.
func Resolve(name, quant string) (types.Manifest, bool) {
	m, ok := Lookup(name)
	if !ok {
		return types.Manifest{}, false
	}
	arch, ok := archdb.ByName(m.Arch)
	if !ok {
		return types.Manifest{}, false // curated data references an unknown arch
	}
	arch = arch.For(m.DiffSource, m.DiffPattern)

	pref, _ := archdb.ParseQuant(quant)
	chosen := pickQuant(pref, m.Quants)

	engine := arch.EngineSpec()
	if len(m.ExtraModelArgs) > 0 {
		if engine.ModelArgs == nil {
			engine.ModelArgs = make(map[string]any, len(m.ExtraModelArgs))
		}
		maps.Copy(engine.ModelArgs, m.ExtraModelArgs)
	}

	man, _ := archdb.Build(arch, archdb.BuildOpts{
		Name:   m.Name,
		Repo:   m.DiffSource,
		Hints:  []string{m.DiffPattern},
		Engine: engine,
		Resolve: func(role types.Role) (types.Component, bool) {
			comp, ok := m.source(role, arch)
			if !ok {
				return types.Component{}, false
			}
			file := comp.FilePattern
			if comp.Quantized {
				// Each source gets the best label IT publishes, not the
				// diffusion weights' label verbatim: unsloth's Qwen3-8B has no
				// Q4_0 at all, and a blind substitution 404s after gigabytes.
				file = strings.ReplaceAll(file, "{quant}", string(pickQuant(chosen, comp.Quants)))
			}
			return types.Component{Role: role, File: file, Source: comp.Source}, true
		},
	})
	return man, true
}

func pickQuant(pref archdb.Quant, have []archdb.Quant) archdb.Quant {
	if q, ok := archdb.PickQuant(pref, have); ok {
		return q
	}
	return pref
}
