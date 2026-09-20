// Package archdb is the single source of truth for what stable-diffusion.cpp
// (the bundled sd-server engine) can run: architecture -> required component
// roles, the flags that launch them, and the Hugging Face sources for the
// shared components (VAEs, text encoders) a bare diffusion repo lacks.
//
// Both internal/compat and internal/registry read from here so the two never
// diverge. Pure data and lookups, no I/O.
//
// Facts here are grounded in the stable-diffusion.cpp docs:
//   - docs/flux.md, docs/flux2.md, docs/chroma.md
//   - docs/qwen_image_edit.md, docs/z_image.md, docs/sd3.md
//   - examples/server/api.md
package archdb

import (
	"maps"
	"math"
	"slices"
	"strconv"
	"strings"

	"oflux/internal/types"
)

// roleFlags is the canonical role->flag mapping used when building launch args.
var roleFlags = map[types.Role]string{
	types.RoleDiffusion:  "--diffusion-model",
	types.RoleVAE:        "--vae",
	types.RoleCLIPL:      "--clip_l",
	types.RoleT5XXL:      "--t5xxl",
	types.RoleLLM:        "--llm",
	types.RoleMMProj:     "--llm_vision",
	types.RoleControlNet: "--control-net",
}

// FlagName returns the flag that loads a role, "" if the engine has none.
func FlagName(role types.Role) string { return roleFlags[role] }

// Companion is where a shared component a bare diffusion repo lacks comes from.
// FilePattern may contain a "{quant}" token the caller fills; without one it is
// a fixed filename.
type Companion struct {
	Source      string // Hugging Face repo id
	FilePattern string // filename, optionally containing "{quant}"
	Quantized   bool   // whether a {quant} variant is available

	// Quants are the labels Source actually publishes, read off the live tree.
	// Publishers disagree on granularity (unsloth's Qwen3-8B has no Q4_0 at
	// all), so substituting the diffusion weights' label blindly resolves a
	// filename that 404s after the multi-gigabyte download. Empty means unknown.
	Quants []Quant
}

// Arch is one architecture: how to recognize it, what it needs, how to launch it.
type Arch struct {
	Name string // canonical short name: "flux", "flux-kontext", "qwen-image", ...

	// ClassNames are the diffusers _class_name values identifying this arch.
	ClassNames []string

	// Keywords recognize the architecture from a repo id or GGUF filename when
	// no diffusers config is present (the common case for mirror repos).
	Keywords []string

	Mode     types.Mode   // edit, generate, or both (hybrid)
	Required []types.Role // roles that MUST be present to launch
	Optional []types.Role // roles that improve quality when present (e.g. mmproj)

	// Companions source the shared roles; diffusion comes from the pulled repo.
	Companions map[types.Role]Companion

	// ModelArgs are extra "--model-args k=v" defaults for this architecture.
	ModelArgs map[string]any

	// Sampling defaults surfaced to the user / used when the request omits them.
	Defaults map[string]any

	// DiffFlag overrides the flag that loads the diffusion role. Empty means
	// "--diffusion-model", right for standalone DiT weights. Full-checkpoint
	// arches (SD/SDXL) bake UNet+VAE+encoders into one file loaded with
	// "--model", so they set DiffFlag.
	DiffFlag string

	// switchFlags are non-value flags, e.g. "--diffusion-fa".
	switchFlags []string

	// variants refine the architecture for a concrete checkpoint; see For.
	variants []variant
}

// variant is a sibling checkpoint that launches through the same architecture
// but needs different companions: klein-4B is encoded by Qwen3-4B, klein-9B by
// Qwen3-8B. Handing a 9B the 4B encoder does not fail cleanly — the engine dies
// with "'…q_norm.weight' not in model metadata" and a tensor-shape mismatch,
// which reads like a corrupt download rather than the wrong encoder.
type variant struct {
	keywords   []string // matched like Arch.Keywords: separators ignored
	companions map[types.Role]Companion
}

// For returns the architecture specialized for a concrete checkpoint, given
// hints such as the repo id and the chosen filename. Variants share a name,
// mode and launch recipe, so they are not separate Arch entries.
func (a Arch) For(hints ...string) Arch {
	for _, v := range a.variants {
		if !matchesAny(v.keywords, hints) {
			continue
		}
		merged := make(map[types.Role]Companion, len(a.Companions)+len(v.companions))
		maps.Copy(merged, a.Companions)
		maps.Copy(merged, v.companions)
		a.Companions = merged
		return a
	}
	return a
}

// BaseFlags returns the sd-server flags with {role} placeholders in value
// positions, e.g. ["--diffusion-model","{diffusion}","--vae","{vae}"], then the
// architecture's switch flags. The supervisor substitutes each placeholder with
// the matching component's on-disk path.
func (a Arch) BaseFlags() []string {
	var flags []string
	for _, r := range flagOrder {
		if !a.has(r) {
			continue
		}
		flag := FlagName(r)
		// Full-checkpoint arches (SDXL/SD) load the diffusion role via
		// --model instead of --diffusion-model; DiffFlag carries that override.
		if r == types.RoleDiffusion && a.DiffFlag != "" {
			flag = a.DiffFlag
		}
		flags = append(flags, flag, "{"+string(r)+"}")
	}
	return append(flags, a.switchFlags...)
}

// flagOrder is the stable argv order for component role flags.
var flagOrder = []types.Role{
	types.RoleDiffusion, types.RoleVAE, types.RoleCLIPL,
	types.RoleT5XXL, types.RoleLLM, types.RoleMMProj,
}

func (a Arch) Roles() []types.Role { return slices.Concat(a.Required, a.Optional) }

func (a Arch) has(role types.Role) bool {
	return a.requires(role) || slices.Contains(a.Optional, role)
}

func (a Arch) requires(role types.Role) bool { return slices.Contains(a.Required, role) }

// EngineSpec is the role flags plus the sampling defaults baked in as launch
// flags, so the engine starts with the right cfg/flow-shift/steps.
func (a Arch) EngineSpec() types.EngineSpec { return a.EngineSpecWith(nil) }

// EngineSpecWith layers per-model defaults over the architecture's. A
// step-distilled merge wants 4 steps at cfg 1.0 where its base wants 20 at 2.5,
// and those are baked into the launch flags, so they resolve before launch.
func (a Arch) EngineSpecWith(overrides map[string]any) types.EngineSpec {
	defaults := maps.Clone(a.Defaults)
	if len(overrides) > 0 {
		if defaults == nil {
			defaults = make(map[string]any, len(overrides))
		}
		maps.Copy(defaults, overrides)
	}
	return types.EngineSpec{
		Flags:     append(a.BaseFlags(), samplingFlags(defaults)...),
		ModelArgs: maps.Clone(a.ModelArgs),
		Defaults:  defaults,
	}
}

// samplingFlagTable maps a sampling default onto its launch flag.
var samplingFlagTable = []struct{ key, flag string }{
	{"cfg_scale", "--cfg-scale"},
	{"flow_shift", "--flow-shift"},
	{"steps", "--steps"},
	{"sample_method", "--sampling-method"},
}

// flow_shift in particular has no per-request field, so it must be set here.
func samplingFlags(d map[string]any) []string {
	var out []string
	for _, sf := range samplingFlagTable {
		v, ok := d[sf.key]
		if !ok {
			continue
		}
		s, isStr := v.(string)
		if !isStr {
			s = numStr(v)
		}
		if s != "" {
			out = append(out, sf.flag, s)
		}
	}
	return out
}

// IsSamplingFlag reports whether arg is one of the flags EngineSpec appends
// after the role flags. Anything inserting a role flag into an existing spec
// must land in front of them to keep each flag next to its own placeholder.
func IsSamplingFlag(arg string) bool {
	for _, sf := range samplingFlagTable {
		if strings.HasPrefix(arg, sf.flag) {
			return true
		}
	}
	return false
}

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

// Companions shared by several architectures. Every Source/FilePattern/Quants
// triple was checked against the live Hugging Face tree; the {quant} labels are
// only the ones that repo really publishes.
var (
	// FLUX.1's autoencoder, reused by Chroma and Z-Image.
	fluxVAE = Companion{Source: "ffxvs/vae-flux", FilePattern: "ae.safetensors"}
	clipL   = Companion{Source: "comfyanonymous/flux_text_encoders", FilePattern: "clip_l.safetensors"}
	// city96 also publishes f16/f32 builds, but spells them lowercase while
	// every quant label is upper — substituting "F16" into the pattern 404s,
	// so they are deliberately not listed.
	t5xxlGGUF = Companion{
		Source: "city96/t5-v1_1-xxl-encoder-gguf", FilePattern: "t5-v1_1-xxl-encoder-{quant}.gguf", Quantized: true,
		Quants: []Quant{"Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "Q3_K_L", "Q3_K_M", "Q3_K_S"},
	}
	qwenImageVAE = Companion{Source: "Comfy-Org/Qwen-Image_ComfyUI", FilePattern: "split_files/vae/qwen_image_vae.safetensors"}
	qwen25VL     = Companion{
		Source: "mradermacher/Qwen2.5-VL-7B-Instruct-GGUF", FilePattern: "Qwen2.5-VL-7B-Instruct.{quant}.gguf", Quantized: true,
		Quants: []Quant{"Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q2_K"},
	}
)

// registry is the supported-architecture table.
var registry = []Arch{
	{
		Name:       "flux",
		ClassNames: []string{"FluxPipeline", "FluxTransformer2DModel"},
		Keywords:   []string{"flux.1-dev", "flux.1-schnell", "flux1-dev", "flux1-schnell", "flux-krea", "flux.1-krea"},
		Mode:       types.ModeGenerate,
		Required:   []types.Role{types.RoleDiffusion, types.RoleVAE, types.RoleCLIPL, types.RoleT5XXL},
		Companions: map[types.Role]Companion{
			types.RoleVAE:   fluxVAE,
			types.RoleCLIPL: clipL,
			types.RoleT5XXL: t5xxlGGUF,
		},
		switchFlags: []string{"--diffusion-fa"},
		Defaults:    map[string]any{"cfg_scale": 1.0, "steps": 20, "sample_method": "euler"},
	},
	{
		// FLUX.2 encodes with a single LLM via --llm — Mistral-3 for dev,
		// Qwen3 for klein — and no CLIP-L / T5-XXL. Ground truth: docs/flux2.md
		//   --diffusion-model flux2-dev-*.gguf --vae flux2_ae.safetensors \
		//   --llm Mistral-Small-3.2-24B-Instruct-2506-*.gguf --diffusion-fa
		Name:       "flux2",
		ClassNames: []string{"Flux2Pipeline", "Flux2Transformer2DModel"},
		Keywords:   []string{"flux.2", "flux2", "flux-2"},
		Mode:       types.ModeBoth,
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
		// klein is a distilled sibling of flux2-dev encoded by Qwen3, not
		// Mistral-3, and few-step. Ground truth: docs/flux2.md, e.g.
		//   --diffusion-model flux-2-klein-4b.safetensors \
		//   --vae flux2_ae.safetensors --llm qwen_3_4b.safetensors \
		//   --cfg-scale 1.0 --steps 4 --diffusion-fa
		Name:       "flux2-klein",
		ClassNames: []string{"Flux2KleinPipeline"},
		Keywords:   []string{"flux.2-klein", "flux2-klein", "flux-2-klein"},
		Mode:       types.ModeBoth,
		Required:   []types.Role{types.RoleDiffusion, types.RoleVAE, types.RoleLLM},
		Companions: map[types.Role]Companion{
			// Comfy-Org's mirror is ungated where black-forest-labs/FLUX.2-dev
			// (docs/flux2.md's link for the VAE) needs a licence accepted. It
			// was renamed from Comfy-Org/flux2-klein, which now only 307s here.
			types.RoleVAE: {Source: "Comfy-Org/vae-text-encorder-for-flux-klein-4b", FilePattern: "split_files/vae/flux2-vae.safetensors"},
			types.RoleLLM: {
				Source: "unsloth/Qwen3-4B-GGUF", FilePattern: "Qwen3-4B-{quant}.gguf", Quantized: true,
				Quants: []Quant{"Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "Q4_1", "Q4_0", "Q3_K_M", "Q3_K_S", "Q2_K"},
			},
		},
		variants: []variant{{
			// klein-9B is encoded by Qwen3-8B, not the 4B its sibling uses.
			keywords: []string{"klein-9b"},
			companions: map[types.Role]Companion{
				types.RoleVAE: {Source: "Comfy-Org/vae-text-encorder-for-flux-klein-9b", FilePattern: "split_files/vae/flux2-vae.safetensors"},
				types.RoleLLM: {
					Source: "unsloth/Qwen3-8B-GGUF", FilePattern: "Qwen3-8B-{quant}.gguf", Quantized: true,
					// No Q4_0 in this repo, unlike the 4B's.
					Quants: []Quant{"Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "Q4_1", "Q3_K_M", "Q3_K_S", "Q2_K"},
				},
			},
		}},
		switchFlags: []string{"--diffusion-fa"},
		Defaults:    map[string]any{"cfg_scale": 1.0, "steps": 4, "sample_method": "euler"},
	},
	{
		Name:       "flux-kontext",
		ClassNames: []string{"FluxKontextPipeline"},
		Keywords:   []string{"kontext"},
		Mode:       types.ModeEdit,
		Required:   []types.Role{types.RoleDiffusion, types.RoleVAE, types.RoleCLIPL, types.RoleT5XXL},
		Companions: map[types.Role]Companion{
			types.RoleVAE:   fluxVAE,
			types.RoleCLIPL: clipL,
			types.RoleT5XXL: t5xxlGGUF,
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
			types.RoleVAE: qwenImageVAE,
			types.RoleLLM: qwen25VL,
		},
		switchFlags: []string{"--diffusion-fa"},
		Defaults:    map[string]any{"cfg_scale": 2.5, "flow_shift": 3, "steps": 20, "sample_method": "euler"},
	},
	{
		Name:       "qwen-image-edit",
		ClassNames: []string{"QwenImageEditPipeline", "QwenImageEditPlusPipeline"},
		Keywords:   []string{"qwen-image-edit", "qwen_image_edit"},
		Mode:       types.ModeBoth,
		Required:   []types.Role{types.RoleDiffusion, types.RoleVAE, types.RoleLLM},
		// mmproj (--llm_vision) is a 2509-only component. We ship 2511, so it is
		// deliberately not a required/optional role — nothing resolves the
		// companion below, which is kept only so a 2509 entry can opt back in.
		//
		// 2511 samples with a zero conditioning timestep (docs/qwen_image_edit.md),
		// and that belongs here rather than on the curated model: every
		// third-party 2511 derivative pulled by repo id needs it too.
		ModelArgs: map[string]any{"qwen_image_zero_cond_t": true},
		Companions: map[types.Role]Companion{
			types.RoleVAE:    qwenImageVAE,
			types.RoleLLM:    qwen25VL,
			types.RoleMMProj: {Source: qwen25VL.Source, FilePattern: "Qwen2.5-VL-7B-Instruct.mmproj-f16.gguf"},
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
			types.RoleVAE: fluxVAE,
			types.RoleLLM: {
				Source: "unsloth/Qwen3-4B-Instruct-2507-GGUF", FilePattern: "Qwen3-4B-Instruct-2507-{quant}.gguf", Quantized: true,
				Quants: []Quant{"Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "Q4_1", "Q4_0", "Q3_K_M", "Q3_K_S", "Q2_K"},
			},
		},
		Defaults: map[string]any{"cfg_scale": 1.0, "steps": 8, "sample_method": "euler"},
	},
	{
		// Chroma is a FLUX.1-schnell-derived DiT that drops CLIP-L and keeps
		// only T5-XXL. Ground truth: docs/chroma.md, e.g.
		//   --diffusion-model chroma-*.gguf --vae ae.sft --t5xxl t5xxl_fp16 \
		//   --model-args chroma_use_dit_mask=false
		Name:       "chroma",
		ClassNames: []string{"ChromaPipeline", "ChromaTransformer2DModel"},
		Keywords:   []string{"chroma"},
		Mode:       types.ModeGenerate,
		Required:   []types.Role{types.RoleDiffusion, types.RoleVAE, types.RoleT5XXL},
		Companions: map[types.Role]Companion{
			types.RoleVAE:   fluxVAE,
			types.RoleT5XXL: t5xxlGGUF,
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
		// The fp16-fix --vae is optional and only used when supplied in-repo.
		DiffFlag: "--model",
		Defaults: map[string]any{"cfg_scale": 7.0, "steps": 30},
	},
}

func Lookup(className string) (Arch, bool) {
	return find(func(a Arch) bool { return slices.Contains(a.ClassNames, className) })
}

func ByName(name string) (Arch, bool) {
	return find(func(a Arch) bool { return a.Name == name })
}

func find(match func(Arch) bool) (Arch, bool) {
	if i := slices.IndexFunc(registry, match); i >= 0 {
		return registry[i], true
	}
	return Arch{}, false
}

// keywordPriority orders matching specific before generic, so "FLUX.2" never
// falls through to FLUX.1. An arch missing here is never keyword-matched.
var keywordPriority = []string{"flux-kontext", "flux2-klein", "flux2", "qwen-image-edit", "qwen-image", "z-image", "chroma", "flux", "sdxl"}

// MatchKeyword returns the architecture whose Keywords appear in the given
// haystack (a repo id or filename), most specific first. Used as a fallback
// when no diffusers config is available.
func MatchKeyword(haystack string) (Arch, bool) {
	for _, name := range keywordPriority {
		if a, ok := ByName(name); ok && matches(a.Keywords, haystack) {
			return a, true
		}
	}
	return Arch{}, false
}

// ModeOf returns the mode archdb currently believes arch supports, falling back
// to fallback for an architecture it does not know.
//
// Manifest.Mode is frozen at pull time and goes stale: a FLUX.2 installed
// before the arch learned it could edit as well as generate still claims
// "generate" on disk. Every caller deciding edit-vs-generate must ask here.
func ModeOf(arch string, fallback types.Mode) types.Mode {
	if a, ok := ByName(arch); ok {
		return a.Mode
	}
	return fallback
}

// Keywords are spelled "flux.2-klein" but the same model ships as
// flux2_klein_9b_Q8_0.gguf and "FLUX 2 Klein"; matching literally sent that
// klein through the flux2 arch, which loads flux2-dev's Mistral-3 encoder.
var sepStripper = strings.NewReplacer(".", "", "-", "", "_", "", " ", "")

func normKeyword(s string) string { return sepStripper.Replace(strings.ToLower(s)) }

func matches(keywords []string, haystack string) bool {
	h := normKeyword(haystack)
	return slices.ContainsFunc(keywords, func(kw string) bool {
		return strings.Contains(h, normKeyword(kw))
	})
}

func matchesAny(keywords, haystacks []string) bool {
	return slices.ContainsFunc(haystacks, func(h string) bool { return matches(keywords, h) })
}
