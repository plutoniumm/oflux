// Package types holds the shared data contracts used across oflux packages.
//
// It intentionally has no dependencies beyond the standard library and defines
// only types that cross package boundaries. Package-local types (e.g. a
// package's internal helpers) should stay in that package.
package types

// Mode is what a model can do.
type Mode string

const (
	ModeEdit     Mode = "edit"
	ModeGenerate Mode = "generate"
	// ModeBoth is a hybrid: the same weights generate from text alone and edit
	// when given reference images. FLUX.2 (dev and klein) and Qwen-Image-Edit
	// work this way — verified by running both paths against the engine — so
	// describing them as one or the other hides half of what they do.
	ModeBoth Mode = "both"
)

// CanEdit reports whether a model in this mode accepts an input image.
func (m Mode) CanEdit() bool { return m == ModeEdit || m == ModeBoth }

// CanGenerate reports whether a model in this mode can run from text alone.
func (m Mode) CanGenerate() bool { return m == ModeGenerate || m == ModeBoth }

// Role identifies what a component file is within a diffusion pipeline. The
// role names also map 1:1 onto the {placeholder} tokens used in
// EngineSpec.Flags, which the supervisor substitutes with the on-disk path of
// the matching component before launching sd-server.
type Role string

const (
	RoleDiffusion Role = "diffusion" // the DiT/UNet weights   -> --diffusion-model
	RoleVAE       Role = "vae"       // autoencoder             -> --vae
	RoleCLIPL     Role = "clip_l"    // CLIP-L text encoder     -> --clip_l   (Flux)
	RoleT5XXL     Role = "t5xxl"     // T5-XXL text encoder     -> --t5xxl    (Flux)
	RoleLLM       Role = "llm"       // LLM text encoder        -> --llm      (Qwen/Z-Image)
	RoleMMProj    Role = "mmproj"    // vision projector        -> --llm_vision (Qwen-Image-Edit 2509)

	// RoleControlNet is a ControlNet the engine loads alongside the model.
	// Unlike a LoRA, sd-server takes it only at launch (--control-net) and
	// exposes no way to load or switch one over HTTP, so it is part of the
	// installed model rather than a per-request choice. Attached explicitly at
	// pull time; no architecture has a default.
	RoleControlNet Role = "control_net" // ControlNet weights  -> --control-net
)

// Component is one weight file that makes up a model, addressed by content hash
// so it can be shared across manifests in the blob store.
type Component struct {
	Role   Role   `json:"role"`
	File   string `json:"file"`             // filename as it lives in Source
	Source string `json:"source"`           // Hugging Face repo id it came from
	Blob   string `json:"blob"`             // content address in the store, "sha256-<hex>"
	SHA256 string `json:"sha256,omitempty"` // hex sha256 of the content
	Size   int64  `json:"size,omitempty"`   // bytes
}

// EngineSpec is everything needed to launch sd-server for a model, minus the
// resolved file paths. Flags may contain {role} placeholders (e.g.
// "{diffusion}") that the supervisor replaces with the component's blob path.
type EngineSpec struct {
	Flags     []string       `json:"flags"`                // e.g. ["--diffusion-model","{diffusion}","--vae","{vae}"]
	ModelArgs map[string]any `json:"model_args,omitempty"` // appended as --model-args k=v (e.g. qwen_image_zero_cond_t=true)
	Defaults  map[string]any `json:"defaults,omitempty"`   // sampling defaults: cfg_scale, flow_shift, steps, sample_method...
}

// Manifest is the resolved, on-disk description of an installed model. It is
// written to ~/.oflux/manifests/<name>.json once a pull completes.
type Manifest struct {
	Name         string      `json:"name"`
	Architecture string      `json:"architecture"` // "flux", "qwen-image-edit", "z-image", ...
	Mode         Mode        `json:"mode"`
	Components   []Component `json:"components"`
	Engine       EngineSpec  `json:"engine"`
}

// Component returns the component with the given role, if present.
func (m Manifest) Component(role Role) (Component, bool) {
	for _, c := range m.Components {
		if c.Role == role {
			return c, true
		}
	}
	return Component{}, false
}

// HFFile is one entry from a Hugging Face repo tree listing. For LFS-backed
// files (the actual weights), LFSOID is the sha256 of the content and Size is
// the true content size; for small git files, OID is the git blob sha1.
type HFFile struct {
	Path   string `json:"path"`
	Size   int64  `json:"size"`
	OID    string `json:"oid"`     // git blob sha1 (small files)
	LFSOID string `json:"lfs_oid"` // sha256 of content (LFS files)
	IsLFS  bool   `json:"is_lfs"`
}

// ContentHash returns the best available content identity for dedup: the LFS
// sha256 when present, otherwise the git oid.
func (f HFFile) ContentHash() string {
	if f.IsLFS && f.LFSOID != "" {
		return f.LFSOID
	}
	return f.OID
}

// BlockerKind categorizes why a repo is not usable.
type BlockerKind string

const (
	BlockerArchitecture BlockerKind = "architecture"      // arch not in the sd.cpp capability table
	BlockerMissingRole  BlockerKind = "missing_component" // a required component is absent and no known source
	BlockerNoQuant      BlockerKind = "no_quant"          // present but no acceptable quantization
)

// Blocker explains one reason a repo cannot be pulled, with an optional
// actionable suggestion (e.g. a companion GGUF repo that would work).
type Blocker struct {
	Kind    BlockerKind `json:"kind"`
	Role    Role        `json:"role,omitempty"` // set for missing_component / no_quant
	Detail  string      `json:"detail"`
	Suggest string      `json:"suggest,omitempty"`
}

// Verdict is the result of inspecting a Hugging Face repo (or resolving a
// curated name) for compatibility.
type Verdict struct {
	Repo       string    `json:"repo"`
	Compatible bool      `json:"compatible"`
	Manifest   *Manifest `json:"manifest,omitempty"` // populated when Compatible; components have Source+File but empty Blob until downloaded
	Blockers   []Blocker `json:"blockers,omitempty"` // populated when not Compatible
	Notes      []string  `json:"notes,omitempty"`    // non-fatal remarks (e.g. "gated repo, HF token required")
}

// Config is the daemon configuration persisted at ~/.oflux/config.json.
type Config struct {
	Port         int    `json:"port"`          // default 11534
	IdleTTL      string `json:"idle_ttl"`      // Go duration string, default "2m"
	MaxLoaded    int    `json:"max_loaded"`    // default 1
	DefaultQuant string `json:"default_quant"` // default "Q8_0"
	HFToken      string `json:"hf_token,omitempty"`
	ModelsDir    string `json:"models_dir,omitempty"` // override for ~/.oflux
}

// DefaultConfig returns the built-in defaults.
func DefaultConfig() Config {
	return Config{
		Port:         11534,
		IdleTTL:      "2m",
		MaxLoaded:    1,
		DefaultQuant: "Q8_0",
	}
}
