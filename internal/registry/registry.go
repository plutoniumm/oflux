// Package registry is the curated model registry: hand-verified templates for
// the blessed diffusion models so that `oflux pull <friendly-name>` just works.
//
// Each curated Model names the archdb architecture it uses and pins the exact
// Hugging Face repo id and filename pattern for its model-specific diffusion
// weights. Shared components (VAE, text encoders, vision projector) are taken
// from the architecture's Companions unless a Model overrides them here — used
// when a companion filename in archdb does not literally exist in its source
// (e.g. the Qwen-Image VAE lives under split_files/vae/ in its repo).
//
// All repo ids, diffusion filenames and quant lists in this file were verified
// against the live Hugging Face repo trees; see the package's test coverage and
// the accompanying notes. This package performs no I/O: it turns a curated name
// plus a quant label into a manifest template (Component.Blob left empty; the
// puller fills content addresses in once files are downloaded).
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
	Mode           types.Mode                      // convenience copy of arch.Mode
	DiffSource     string                          // HF repo id hosting the diffusion GGUF
	DiffPattern    string                          // filename with "{quant}" token
	Quants         []string                        // quant labels available in DiffSource (curated, aligned with companions)
	ExtraModelArgs map[string]any                  // merged over arch.ModelArgs
	Overrides      map[types.Role]archdb.Companion // optional per-role source overrides for shared components
	Description    string
}

// curated is the hand-verified model table. Order is irrelevant; Names() sorts.
var curated = buildCurated()

func buildCurated() []Model {
	// qwenVAE is the correct on-disk path for the Qwen-Image VAE. The archdb
	// default is "qwen_image_vae.safetensors", but in Comfy-Org/Qwen-Image_ComfyUI
	// the file actually lives at split_files/vae/qwen_image_vae.safetensors.
	qwenVAE := archdb.Companion{
		Source:      "Comfy-Org/Qwen-Image_ComfyUI",
		FilePattern: "split_files/vae/qwen_image_vae.safetensors",
		Quantized:   false,
	}
	// qwenMMProj pins the Qwen2.5-VL vision projector to its fixed f16 file. Only
	// mmproj-Q8_0 and mmproj-f16 exist in the source, so the archdb per-quant
	// pattern ("...mmproj-{quant}.gguf") only resolves for Q8_0; f16 always works.
	qwenMMProj := archdb.Companion{
		Source:      "mradermacher/Qwen2.5-VL-7B-Instruct-GGUF",
		FilePattern: "Qwen2.5-VL-7B-Instruct.mmproj-f16.gguf",
		Quantized:   false,
	}
	return []Model{
		// ---- EDIT models ------------------------------------------------------
		{
			Name:        "qwen-image-edit",
			Arch:        "qwen-image-edit",
			Mode:        types.ModeEdit,
			DiffSource:  "unsloth/Qwen-Image-Edit-2511-GGUF",
			DiffPattern: "qwen-image-edit-2511-{quant}.gguf",
			Quants:      []string{"Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"},
			// 2511 uses zero conditioning-timestep sampling.
			ExtraModelArgs: map[string]any{"qwen_image_zero_cond_t": true},
			Overrides: map[types.Role]archdb.Companion{
				types.RoleVAE:    qwenVAE,
				types.RoleMMProj: qwenMMProj,
			},
			Description: "Qwen-Image-Edit 2511 - instruction image editing (Qwen2.5-VL encoder).",
		},
		{
			Name:        "flux.1-kontext",
			Arch:        "flux-kontext",
			Mode:        types.ModeEdit,
			DiffSource:  "QuantStack/FLUX.1-Kontext-dev-GGUF",
			DiffPattern: "flux1-kontext-dev-{quant}.gguf",
			Quants:      []string{"Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M"},
			Description: "FLUX.1 Kontext [dev] - in-context image editing.",
		},

		// ---- GENERATION models -----------------------------------------------
		{
			Name:        "z-image-turbo",
			Arch:        "z-image",
			Mode:        types.ModeGenerate,
			DiffSource:  "leejet/Z-Image-Turbo-GGUF",
			DiffPattern: "z_image_turbo-{quant}.gguf",
			// Aligned with the Qwen3-4B encoder companion's quant labels.
			Quants:      []string{"Q8_0", "Q6_K", "Q4_0", "Q2_K"},
			Description: "Z-Image Turbo - fast few-step text-to-image (Qwen3-4B encoder).",
		},
		{
			Name:        "flux.1-krea",
			Arch:        "flux",
			Mode:        types.ModeGenerate,
			DiffSource:  "QuantStack/FLUX.1-Krea-dev-GGUF",
			DiffPattern: "flux1-krea-dev-{quant}.gguf",
			Quants:      []string{"Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M"},
			Description: "FLUX.1 Krea [dev] - photographic text-to-image.",
		},
		{
			Name:        "flux.1-dev",
			Arch:        "flux",
			Mode:        types.ModeGenerate,
			DiffSource:  "city96/FLUX.1-dev-gguf",
			DiffPattern: "flux1-dev-{quant}.gguf",
			// city96 flux repos ship the *_S K-quants (no *_M); aligned with t5 encoder.
			Quants:      []string{"Q8_0", "Q6_K", "Q5_K_S", "Q4_K_S", "Q3_K_S"},
			Description: "FLUX.1 [dev] - guidance-distilled text-to-image.",
		},
		{
			Name:        "flux.1-schnell",
			Arch:        "flux",
			Mode:        types.ModeGenerate,
			DiffSource:  "city96/FLUX.1-schnell-gguf",
			DiffPattern: "flux1-schnell-{quant}.gguf",
			Quants:      []string{"Q8_0", "Q6_K", "Q5_K_S", "Q4_K_S", "Q3_K_S"},
			Description: "FLUX.1 [schnell] - fast few-step text-to-image (Apache-2.0).",
		},
		{
			Name:        "qwen-image",
			Arch:        "qwen-image",
			Mode:        types.ModeGenerate,
			DiffSource:  "QuantStack/Qwen-Image-GGUF",
			DiffPattern: "Qwen_Image-{quant}.gguf",
			Quants:      []string{"Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"},
			Overrides: map[types.Role]archdb.Companion{
				types.RoleVAE: qwenVAE,
			},
			Description: "Qwen-Image - text-to-image (Qwen2.5-VL encoder).",
		},
	}
}

// Names returns the curated model names, sorted.
func Names() []string {
	out := make([]string, 0, len(curated))
	for _, m := range curated {
		out = append(out, m.Name)
	}
	slices.Sort(out)
	return out
}

// Lookup returns the curated Model with the given friendly name.
func Lookup(name string) (Model, bool) {
	for _, m := range curated {
		if m.Name == name {
			return m, true
		}
	}
	return Model{}, false
}

// Resolve builds a manifest template for name at the given quant. Component
// Blob fields are left empty (filled by the puller once files are downloaded).
// If the requested quant is not among the model's Quants, it falls back to the
// first available quant. Returns ok=false for unknown names.
func Resolve(name, quant string) (types.Manifest, bool) {
	m, ok := Lookup(name)
	if !ok {
		return types.Manifest{}, false
	}
	arch, ok := archdb.ByName(m.Arch)
	if !ok {
		return types.Manifest{}, false // curated data references an unknown arch
	}

	// Choose the quant: honor the request when available, else fall back to the
	// model's first curated quant.
	chosen := quant
	if !slices.Contains(m.Quants, quant) && len(m.Quants) > 0 {
		chosen = m.Quants[0]
	}

	comps := make([]types.Component, 0, len(arch.Required)+len(arch.Optional))

	// Diffusion weights come from the curated model itself, never a companion.
	comps = append(comps, types.Component{
		Role:   types.RoleDiffusion,
		File:   strings.ReplaceAll(m.DiffPattern, "{quant}", chosen),
		Source: m.DiffSource,
	})

	// Shared components: every required role plus any optional role that has a
	// companion. Diffusion is handled above (and has no companion), so it is
	// naturally skipped by the no-companion check.
	roles := make([]types.Role, 0, len(arch.Required)+len(arch.Optional))
	roles = append(roles, arch.Required...)
	roles = append(roles, arch.Optional...)
	for _, role := range roles {
		if role == types.RoleDiffusion {
			continue
		}
		comp, ok := m.Overrides[role]
		if !ok {
			comp, ok = arch.Companions[role]
		}
		if !ok {
			continue // no known source for this role; skip
		}
		file := comp.FilePattern
		if comp.Quantized {
			file = strings.ReplaceAll(file, "{quant}", chosen)
		}
		comps = append(comps, types.Component{
			Role:   role,
			File:   file,
			Source: comp.Source,
		})
	}

	engine := arch.EngineSpec()
	if len(m.ExtraModelArgs) > 0 {
		if engine.ModelArgs == nil {
			engine.ModelArgs = make(map[string]any, len(m.ExtraModelArgs))
		}
		maps.Copy(engine.ModelArgs, m.ExtraModelArgs)
	}

	return types.Manifest{
		Name:         m.Name,
		Architecture: arch.Name,
		Mode:         arch.Mode,
		Components:   comps,
		Engine:       engine,
	}, true
}
