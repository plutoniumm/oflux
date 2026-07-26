// Package archdb is the single source of truth for what stable-diffusion.cpp
// (the bundled sd-server engine) can run. It maps diffusion model
// architectures to their required component roles, the sd-server flags used to
// launch them, and the default Hugging Face sources for shared components
// (VAEs and text encoders) that are not part of a bare diffusion-weights repo.
//
// Both the compatibility checker (internal/compat) and the curated model
// registry (internal/registry) read from this package so the two never
// diverge. It has no dependencies beyond internal/types and the standard
// library, and is pure data + lookups (no I/O), so it is trivially testable.
//
// Facts here are grounded in the stable-diffusion.cpp docs:
//   - docs/flux.md, docs/flux2.md, docs/chroma.md
//   - docs/qwen_image_edit.md, docs/z_image.md, docs/sd3.md
//   - examples/server/api.md
//
// Companion repo/filename defaults are best-known starting points; the curated
// registry pins exact, verified filenames per concrete model.
package archdb

import (
	"maps"
	"math"
	"slices"
	"strconv"
	"strings"

	"oflux/internal/types"
)

// FlagName maps a component role to the sd-server command-line flag that loads
// it. This is the canonical role->flag mapping used when building launch args.
func FlagName(role types.Role) string {
	switch role {
	case types.RoleDiffusion:
		return "--diffusion-model"
	case types.RoleVAE:
		return "--vae"
	case types.RoleCLIPL:
		return "--clip_l"
	case types.RoleT5XXL:
		return "--t5xxl"
	case types.RoleLLM:
		return "--llm"
	case types.RoleMMProj:
		return "--llm_vision"
	default:
		return ""
	}
}

// Companion describes where a shared component (a VAE or text encoder that a
// bare diffusion repo does not include) can be fetched from by default.
// FilePattern may contain a "{quant}" token that the caller fills from the
// quant preference (e.g. "qwen_2.5_vl_7b-{quant}.gguf"); a pattern with no
// token is a fixed filename (e.g. "ae.safetensors").
type Companion struct {
	Source      string // Hugging Face repo id
	FilePattern string // filename, optionally containing "{quant}"
	Quantized   bool   // whether a {quant} variant is available
}

// Arch is one supported architecture: how to recognize it, what it needs, and
// how to launch it.
type Arch struct {
	Name string // canonical short name: "flux", "flux-kontext", "qwen-image", ...

	// ClassNames are the diffusers pipeline/model _class_name values that
	// identify this architecture (from model_index.json / config.json).
	ClassNames []string

	// Keywords are lowercase substrings used to recognize the architecture from
	// a repo id or GGUF filename when no diffusers config is present (the common
	// case for GGUF mirror repos like city96/FLUX.1-dev-gguf).
	Keywords []string

	Mode     types.Mode   // edit or generate
	Required []types.Role // roles that MUST be present to launch
	Optional []types.Role // roles that improve quality when present (e.g. mmproj)

	// Companions gives the default source for each shared role (typically VAE
	// and text encoders). The diffusion role is never a companion — it comes
	// from the repo being pulled.
	Companions map[types.Role]Companion

	// ModelArgs are extra "--model-args k=v" defaults for this architecture.
	ModelArgs map[string]any

	// Sampling defaults surfaced to the user / used when the request omits them.
	Defaults map[string]any

	// DiffFlag overrides the command-line flag used to load the diffusion role.
	// Empty means the default FlagName(RoleDiffusion) == "--diffusion-model",
	// which is correct for standalone DiT weights (Flux, Qwen, Z-Image, Chroma).
	// Full-checkpoint architectures (SD1.x/SD2.x/SDXL) instead load a single
	// file that bakes in the UNet+VAE+text-encoders and are loaded with
	// "--model" ("-m"), so they set DiffFlag: "--model".
	DiffFlag string

	// switchFlags are non-value flags (no {placeholder}) for the architecture,
	// e.g. "--diffusion-fa". Appended after the role flags in BaseFlags.
	switchFlags []string
}

// BaseFlags returns the sd-server flags for this architecture with {role}
// placeholders in value positions, e.g.:
//
//	["--diffusion-model","{diffusion}","--vae","{vae}","--llm","{llm}"]
//
// plus any architecture-level switch flags (e.g. "--diffusion-fa"). The
// supervisor substitutes each "{role}" with the on-disk path of the matching
// component. Roles are emitted in a stable order: diffusion, vae, clip_l,
// t5xxl, llm, mmproj, then switches.
func (a Arch) BaseFlags() []string {
	order := []types.Role{
		types.RoleDiffusion, types.RoleVAE, types.RoleCLIPL,
		types.RoleT5XXL, types.RoleLLM, types.RoleMMProj,
	}
	present := map[types.Role]bool{}
	for _, r := range a.Required {
		present[r] = true
	}
	for _, r := range a.Optional {
		present[r] = true
	}
	var flags []string
	for _, r := range order {
		if present[r] {
			flag := FlagName(r)
			// Full-checkpoint arches (SDXL/SD) load the diffusion role via
			// --model instead of --diffusion-model; DiffFlag carries that override.
			if r == types.RoleDiffusion && a.DiffFlag != "" {
				flag = a.DiffFlag
			}
			flags = append(flags, flag, "{"+string(r)+"}")
		}
	}
	flags = append(flags, a.switchFlags...)
	return flags
}

// EngineSpec builds a types.EngineSpec for this architecture: the component
// role flags, then the sampling defaults baked in as launch flags (so the engine
// starts with the correct cfg/flow-shift/steps for the model), plus the model
// args and a copy of the defaults for reference. Component paths are resolved
// later by the supervisor; per-request overrides are sent in the img_gen body.
func (a Arch) EngineSpec() types.EngineSpec {
	flags := a.BaseFlags()
	flags = append(flags, samplingFlags(a.Defaults)...)
	return types.EngineSpec{
		Flags:     flags,
		ModelArgs: cloneAny(a.ModelArgs),
		Defaults:  cloneAny(a.Defaults),
	}
}

// samplingFlags maps known sampling defaults onto the sd-server launch flags.
// flow_shift in particular has no per-request field, so it must be set here.
func samplingFlags(d map[string]any) []string {
	if d == nil {
		return nil
	}
	var out []string
	if v, ok := d["cfg_scale"]; ok {
		out = append(out, "--cfg-scale", numStr(v))
	}
	if v, ok := d["flow_shift"]; ok {
		out = append(out, "--flow-shift", numStr(v))
	}
	if v, ok := d["steps"]; ok {
		out = append(out, "--steps", numStr(v))
	}
	if v, ok := d["sample_method"]; ok {
		if s, ok := v.(string); ok && s != "" {
			out = append(out, "--sampling-method", s)
		}
	}
	return out
}

// numStr formats a numeric default without a trailing ".0" for whole numbers.
func numStr(v any) string {
	switch n := v.(type) {
	case int:
		return strconv.Itoa(n)
	case int64:
		return strconv.FormatInt(n, 10)
	case float64:
		if n == math.Trunc(n) {
			return strconv.FormatInt(int64(n), 10)
		}
		return strconv.FormatFloat(n, 'g', -1, 64)
	default:
		return ""
	}
}

func cloneAny(m map[string]any) map[string]any {
	if m == nil {
		return nil
	}
	out := make(map[string]any, len(m))
	maps.Copy(out, m)
	return out
}

// registry is the supported-architecture table.
var registry = buildRegistry()

func buildRegistry() []Arch {
	return []Arch{
		{
			Name:       "flux",
			ClassNames: []string{"FluxPipeline", "FluxTransformer2DModel"},
			Keywords:   []string{"flux.1-dev", "flux.1-schnell", "flux1-dev", "flux1-schnell", "flux-krea", "flux.1-krea"},
			Mode:       types.ModeGenerate,
			Required:   []types.Role{types.RoleDiffusion, types.RoleVAE, types.RoleCLIPL, types.RoleT5XXL},
			Companions: map[types.Role]Companion{
				types.RoleVAE:   {Source: "ffxvs/vae-flux", FilePattern: "ae.safetensors"},
				types.RoleCLIPL: {Source: "comfyanonymous/flux_text_encoders", FilePattern: "clip_l.safetensors"},
				types.RoleT5XXL: {Source: "city96/t5-v1_1-xxl-encoder-gguf", FilePattern: "t5-v1_1-xxl-encoder-{quant}.gguf", Quantized: true},
			},
			switchFlags: []string{"--diffusion-fa"},
			Defaults:    map[string]any{"cfg_scale": 1.0, "steps": 20, "sample_method": "euler"},
		},
		{
			// FLUX.2 uses a different text-encoder stack than FLUX.1: a single
			// Mistral-3 (dev) or Qwen3-4B (klein) LLM encoder loaded via --llm,
			// with NO CLIP-L / T5-XXL. Diffusion is still standalone DiT weights
			// (--diffusion-model). Ground truth: docs/flux2.md, e.g.
			//   --diffusion-model flux2-dev-*.gguf --vae flux2_ae.safetensors \
			//   --llm Mistral-Small-3.2-24B-Instruct-2506-*.gguf --diffusion-fa
			Name:       "flux2",
			ClassNames: []string{"Flux2Pipeline", "Flux2Transformer2DModel"},
			Keywords:   []string{"flux.2", "flux2", "flux-2"},
			Mode:       types.ModeGenerate,
			Required:   []types.Role{types.RoleDiffusion, types.RoleVAE, types.RoleLLM},
			Companions: map[types.Role]Companion{
				// Comfy-Org ships the flux2 VAE and a purpose-built fp8 Mistral-3
				// text encoder as fixed single files (no per-quant variants).
				types.RoleVAE: {Source: "Comfy-Org/flux2-dev", FilePattern: "split_files/vae/flux2-vae.safetensors"},
				types.RoleLLM: {Source: "Comfy-Org/flux2-dev", FilePattern: "split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors"},
			},
			switchFlags: []string{"--diffusion-fa"},
			Defaults:    map[string]any{"cfg_scale": 1.0, "steps": 28, "sample_method": "euler"},
		},
		{
			Name:       "flux-kontext",
			ClassNames: []string{"FluxKontextPipeline"},
			Keywords:   []string{"kontext"},
			Mode:       types.ModeEdit,
			Required:   []types.Role{types.RoleDiffusion, types.RoleVAE, types.RoleCLIPL, types.RoleT5XXL},
			Companions: map[types.Role]Companion{
				types.RoleVAE:   {Source: "ffxvs/vae-flux", FilePattern: "ae.safetensors"},
				types.RoleCLIPL: {Source: "comfyanonymous/flux_text_encoders", FilePattern: "clip_l.safetensors"},
				types.RoleT5XXL: {Source: "city96/t5-v1_1-xxl-encoder-gguf", FilePattern: "t5-v1_1-xxl-encoder-{quant}.gguf", Quantized: true},
			},
			switchFlags: []string{"--diffusion-fa"},
			Defaults:    map[string]any{"cfg_scale": 1.0, "steps": 28, "sample_method": "euler"},
		},
		{
			Name:       "qwen-image",
			ClassNames: []string{"QwenImagePipeline"},
			Keywords:   []string{"qwen-image", "qwen_image"},
			Mode:       types.ModeGenerate,
			Required:   []types.Role{types.RoleDiffusion, types.RoleVAE, types.RoleLLM},
			Companions: map[types.Role]Companion{
				types.RoleVAE: {Source: "Comfy-Org/Qwen-Image_ComfyUI", FilePattern: "split_files/vae/qwen_image_vae.safetensors"},
				types.RoleLLM: {Source: "mradermacher/Qwen2.5-VL-7B-Instruct-GGUF", FilePattern: "Qwen2.5-VL-7B-Instruct.{quant}.gguf", Quantized: true},
			},
			switchFlags: []string{"--diffusion-fa"},
			Defaults:    map[string]any{"cfg_scale": 2.5, "flow_shift": 3, "steps": 20, "sample_method": "euler"},
		},
		{
			Name:       "qwen-image-edit",
			ClassNames: []string{"QwenImageEditPipeline", "QwenImageEditPlusPipeline"},
			Keywords:   []string{"qwen-image-edit", "qwen_image_edit"},
			Mode:       types.ModeEdit,
			Required:   []types.Role{types.RoleDiffusion, types.RoleVAE, types.RoleLLM},
			// NOTE: mmproj (--llm_vision) is only used by the 2509 variant. We ship
			// 2511 (which uses --llm only), so mmproj is intentionally NOT listed as
			// a required/optional role — that keeps it out of both the launch flags
			// and the resolved component set. The companion below is kept (pinned to
			// the fixed f16 projector, which is the only non-Q8_0 file that exists)
			// so a future 2509 curated entry can opt back in.
			Companions: map[types.Role]Companion{
				types.RoleVAE:    {Source: "Comfy-Org/Qwen-Image_ComfyUI", FilePattern: "split_files/vae/qwen_image_vae.safetensors"},
				types.RoleLLM:    {Source: "mradermacher/Qwen2.5-VL-7B-Instruct-GGUF", FilePattern: "Qwen2.5-VL-7B-Instruct.{quant}.gguf", Quantized: true},
				types.RoleMMProj: {Source: "mradermacher/Qwen2.5-VL-7B-Instruct-GGUF", FilePattern: "Qwen2.5-VL-7B-Instruct.mmproj-f16.gguf"},
			},
			switchFlags: []string{"--diffusion-fa"},
			Defaults:    map[string]any{"cfg_scale": 2.5, "flow_shift": 3, "steps": 20, "sample_method": "euler"},
		},
		{
			Name:       "z-image",
			ClassNames: []string{"ZImagePipeline"},
			Keywords:   []string{"z-image", "z_image"},
			Mode:       types.ModeGenerate,
			Required:   []types.Role{types.RoleDiffusion, types.RoleVAE, types.RoleLLM},
			Companions: map[types.Role]Companion{
				// Z-Image reuses the FLUX.1-schnell VAE (ae.sft) and a Qwen3-4B encoder.
				types.RoleVAE: {Source: "ffxvs/vae-flux", FilePattern: "ae.safetensors"},
				types.RoleLLM: {Source: "unsloth/Qwen3-4B-Instruct-2507-GGUF", FilePattern: "Qwen3-4B-Instruct-2507-{quant}.gguf", Quantized: true},
			},
			Defaults: map[string]any{"cfg_scale": 1.0, "steps": 8, "sample_method": "euler"},
		},
		{
			// Chroma is a FLUX.1-schnell-derived DiT that drops CLIP-L and keeps
			// only the T5-XXL encoder. Diffusion loads via --diffusion-model.
			// Ground truth: docs/chroma.md, e.g.
			//   --diffusion-model chroma-*.gguf --vae ae.sft --t5xxl t5xxl_fp16 \
			//   --model-args chroma_use_dit_mask=false
			Name:       "chroma",
			ClassNames: []string{"ChromaPipeline", "ChromaTransformer2DModel"},
			Keywords:   []string{"chroma"},
			Mode:       types.ModeGenerate,
			Required:   []types.Role{types.RoleDiffusion, types.RoleVAE, types.RoleT5XXL},
			Companions: map[types.Role]Companion{
				// Reuses the FLUX.1-schnell VAE (ae.safetensors) and the same
				// city96 T5-XXL encoder GGUFs as Flux.
				types.RoleVAE:   {Source: "ffxvs/vae-flux", FilePattern: "ae.safetensors"},
				types.RoleT5XXL: {Source: "city96/t5-v1_1-xxl-encoder-gguf", FilePattern: "t5-v1_1-xxl-encoder-{quant}.gguf", Quantized: true},
			},
			ModelArgs: map[string]any{"chroma_use_dit_mask": false},
			Defaults:  map[string]any{"cfg_scale": 4.0, "steps": 26, "sample_method": "euler"},
		},
		{
			Name:       "sdxl",
			ClassNames: []string{"StableDiffusionXLPipeline"},
			Keywords:   []string{"sdxl", "stable-diffusion-xl"},
			Mode:       types.ModeGenerate,
			Required:   []types.Role{types.RoleDiffusion},
			Optional:   []types.Role{types.RoleVAE},
			// SDXL is a full checkpoint: the single -m/--model file bakes in the
			// UNet, VAE and text encoders, so it must NOT use --diffusion-model.
			// A separate fp16-fix --vae is optional (only used when supplied
			// in-repo); there is no default VAE companion.
			DiffFlag: "--model",
			Defaults: map[string]any{"cfg_scale": 7.0, "steps": 30},
		},
	}
}

// Lookup returns the architecture whose ClassNames contains className.
func Lookup(className string) (Arch, bool) {
	for _, a := range registry {
		if slices.Contains(a.ClassNames, className) {
			return a, true
		}
	}
	return Arch{}, false
}

// ByName returns the architecture with the given canonical Name.
func ByName(name string) (Arch, bool) {
	for _, a := range registry {
		if a.Name == name {
			return a, true
		}
	}
	return Arch{}, false
}

// MatchKeyword returns the first architecture whose Keywords appear in the
// given haystack (a repo id or filename; matched case-insensitively). Used as a
// fallback when no diffusers config is available. More specific archs (e.g.
// "flux-kontext", "qwen-image-edit") are checked before their generic bases.
func MatchKeyword(haystack string) (Arch, bool) {
	h := strings.ToLower(haystack)
	// Order matters: specific before generic. flux2 precedes flux so "FLUX.2"
	// repos never fall through to the FLUX.1 arch.
	priority := []string{"flux-kontext", "flux2", "qwen-image-edit", "qwen-image", "z-image", "chroma", "flux", "sdxl"}
	for _, name := range priority {
		a, ok := ByName(name)
		if !ok {
			continue
		}
		for _, kw := range a.Keywords {
			if strings.Contains(h, kw) {
				return a, true
			}
		}
	}
	return Arch{}, false
}

// All returns a copy of the supported-architecture table.
func All() []Arch {
	out := make([]Arch, len(registry))
	copy(out, registry)
	return out
}
