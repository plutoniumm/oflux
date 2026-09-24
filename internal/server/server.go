// Package server exposes oflux's HTTP API on the daemon port (default 11534):
// a clean JSON edit/generate surface that maps onto the sd-server native
// img_gen API, plus model-management endpoints (pull/list/delete/ps).
//
// Request decoding, validation and error writing live in request.go; streaming
// and job endpoints in stream.go; handlers here only parse, delegate and write.
package server

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"

	"oflux/internal/archdb"
	"oflux/internal/engineclient"
	"oflux/internal/puller"
	"oflux/internal/registry"
	"oflux/internal/store"
	"oflux/internal/supervisor"
	"oflux/internal/types"
)

type Server struct {
	store *store.Store
	sup   *supervisor.Supervisor
	pull  *puller.Puller
	cfg   types.Config
	jobs  *jobNotes
}

func New(st *store.Store, sup *supervisor.Supervisor, pull *puller.Puller, cfg types.Config) *Server {
	return &Server{store: st, sup: sup, pull: pull, cfg: cfg, jobs: newJobNotes()}
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /{$}", s.handleUI) // "/{$}" matches the root exactly
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) { w.Write([]byte("ok")) })
	mux.HandleFunc("POST /v1/edit", s.handleImage(types.ModeEdit))
	mux.HandleFunc("POST /v1/generate", s.handleImage(types.ModeGenerate))
	mux.HandleFunc("POST /v1/segment", s.handleSegment)
	mux.HandleFunc("GET /v1/jobs/{id}", s.handleJob)
	mux.HandleFunc("DELETE /v1/jobs/{id}", s.handleJobCancel)
	mux.HandleFunc("POST /v1/images/edits", s.handleOpenAIEdit)
	mux.HandleFunc("POST /v1/images/generations", s.handleOpenAIGenerate)
	mux.HandleFunc("POST /api/pull", s.handlePull)
	mux.HandleFunc("GET /api/tags", s.handleTags)
	mux.HandleFunc("POST /api/delete", s.handleDelete)
	mux.HandleFunc("GET /api/ps", s.handlePS)
	mux.HandleFunc("GET /api/presets", s.handlePresets)
	mux.HandleFunc("POST /api/presets/create", s.handlePresetCreate)
	mux.HandleFunc("POST /api/presets/delete", s.handlePresetDelete)
	mux.HandleFunc("GET /api/loras", s.handleLoras)
	mux.HandleFunc("POST /api/loras/pull", s.handleLoraPull)
	mux.HandleFunc("POST /api/loras/delete", s.handleLoraDelete)
	mux.HandleFunc("GET /v1/models", s.handleModels)
	return guard(mux)
}

// maxBody is the ONE limit on inbound size: ParseMultipartForm's argument, which
// looked like a second smaller one, is a memory budget rather than a cap.
const (
	maxBody         = 256 << 20 // 256 MiB
	multipartMemory = 32 << 20  // in-RAM slice of a multipart upload; the rest spills to disk
)

// Without guard, any site the user visits could POST to 127.0.0.1:11534 (a CORS
// "simple request" needs no preflight) and delete models, and a DNS-rebinding
// page could read the answers. Hence loopback Host only, no cross-origin.
func guard(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !loopbackHost(r.Host) {
			fail(w, statusErr(http.StatusForbidden, errors.New("forbidden: unexpected Host header")))
			return
		}
		if origin := r.Header.Get("Origin"); origin != "" && !loopbackOrigin(origin) {
			fail(w, statusErr(http.StatusForbidden, errors.New("forbidden: cross-origin request")))
			return
		}
		r.Body = http.MaxBytesReader(w, r.Body, maxBody)
		next.ServeHTTP(w, r)
	})
}

func loopbackHost(host string) bool {
	h, _, err := net.SplitHostPort(host)
	if err != nil {
		h = host // no port
	}
	h = strings.TrimSuffix(strings.Trim(h, "[]"), ".")
	if strings.EqualFold(h, "localhost") {
		return true
	}
	ip := net.ParseIP(h)
	return ip != nil && ip.IsLoopback()
}

func loopbackOrigin(origin string) bool {
	u, err := url.Parse(origin)
	if err != nil {
		return false
	}
	return loopbackHost(u.Host)
}

// LoraRef selects an adapter; Scale is its multiplier, omitted meaning 1.0.
type LoraRef struct {
	Name  string   `json:"name"`
	Scale *float64 `json:"scale,omitempty"`
}

// Images are base64 or data URLs; Image is one string, or an array led by the
// subject.
type ImageRequest struct {
	// Model is an installed model, or a preset that resolves to one.
	Model          string     `json:"model"`
	Prompt         string     `json:"prompt"`
	Loras          []LoraRef  `json:"loras,omitempty"`
	NegativePrompt string     `json:"negative_prompt,omitempty"`
	Image          ImageInput `json:"image,omitempty"`
	RefImages      []string   `json:"ref_images,omitempty"`
	ControlImage   string     `json:"control_image,omitempty"`
	// One field, two names; WHITE marks the region to edit.
	MaskImage string `json:"mask_image,omitempty"`
	Mask      string `json:"mask,omitempty"`
	// MaskPrompt names the region in words and lets SAM3 find it, so one call
	// segments and edits. An explicit mask wins.
	MaskPrompt      string                 `json:"mask_prompt,omitempty"`
	ControlStrength *float64               `json:"control_strength,omitempty"`
	Strength        *float64               `json:"strength,omitempty"`
	Steps           *int                   `json:"steps,omitempty"`
	Sampler         string                 `json:"sampler,omitempty"`
	Scheduler       string                 `json:"scheduler,omitempty"`
	Seed            *int64                 `json:"seed,omitempty"`
	Width           *int                   `json:"width,omitempty"`
	Height          *int                   `json:"height,omitempty"`
	CFG             *float64               `json:"cfg,omitempty"`
	Guidance        *engineclient.Guidance `json:"guidance,omitempty"`
	// Transparent asks for an RGBA result. Only some architectures can, and
	// none take a flag for it — see archdb.AlphaPrompt.
	Transparent bool `json:"transparent,omitempty"`
	// KeepAlive: "10m" or seconds; zero means the configured idle TTL.
	KeepAlive Duration `json:"keep_alive,omitempty"`
	// Stream makes the response NDJSON whose last line is this ImageResponse.
	Stream bool `json:"stream,omitempty"`
}

// The seed is echoed because the engine picks its own and never reports it.
type ImageResponse struct {
	Model  string   `json:"model"`
	Images []string `json:"images"`
	Seed   int64    `json:"seed"`
}

func (s *Server) handleImage(mode types.Mode) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req ImageRequest
		if err := decodeJSON(r, &req); err != nil {
			fail(w, err)
			return
		}
		if req.Stream {
			s.streamImage(w, r, &req, mode)
			return
		}
		res, err := s.generate(r.Context(), &req, mode)
		if err != nil {
			fail(w, err)
			return
		}
		writeJSON(w, http.StatusOK, res)
	}
}

// Both API surfaces funnel through generate, so they accept the same things.
func (s *Server) generate(ctx context.Context, req *ImageRequest, mode types.Mode) (ImageResponse, error) {
	m, err := s.validate(req, mode)
	if err != nil {
		return ImageResponse{}, withModel(err, req.Model)
	}
	if err := resolveMask(ctx, req, nil); err != nil {
		return ImageResponse{}, withModel(err, m.Name)
	}
	// The engine hands back base64 and so do we: never decoded on the way.
	img, err := s.sup.Generate(ctx, m, buildImgGen(m, *req, mode), s.genOpts(req))
	if err != nil {
		return ImageResponse{}, withModel(generationErr(err), m.Name)
	}
	return ImageResponse{Model: m.Name, Images: []string{img}, Seed: *req.Seed}, nil
}

func (s *Server) genOpts(req *ImageRequest) supervisor.GenOpts {
	return supervisor.GenOpts{KeepAlive: time.Duration(req.KeepAlive)}
}

// A timeout is the caller's deadline; anything else keeps what the engine said.
func generationErr(err error) error {
	if errors.Is(err, supervisor.ErrBusy) {
		// 429, not 409: the request is well formed, the server is saturated.
		return &statusError{code: http.StatusTooManyRequests, slug: codeQueueFull,
			err: errors.New("busy: the generation queue is full")}
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return statusErr(http.StatusGatewayTimeout, fmt.Errorf("generation timed out: %w", err))
	}
	return fmt.Errorf("generation failed: %w", err)
}

// Model defaults (steps, cfg, flow-shift, sampling method) are baked into the
// engine launch flags at pull time, so only caller-supplied OVERRIDES are sent
// here — an omitted field keeps the model's default. Assumes validate() ran.
func buildImgGen(m types.Manifest, r ImageRequest, mode types.Mode) engineclient.ImgGenRequest {
	prompt := r.Prompt
	if r.Transparent {
		// The only way to ask: 2.1 decides transparency from the wording alone,
		// and upstream ships this exact framing as the reliable one.
		prompt = fmt.Sprintf(archdb.AlphaPrompt, strings.TrimSuffix(strings.TrimSpace(prompt), ".")+".")
	}
	ig := engineclient.ImgGenRequest{
		Prompt:          prompt,
		NegativePrompt:  r.NegativePrompt,
		Width:           r.Width,
		Height:          r.Height,
		Strength:        r.Strength,
		Seed:            r.Seed,
		RefImages:       r.RefImages,
		ControlImage:    r.ControlImage,
		ControlStrength: r.ControlStrength,
		MaskImage:       r.MaskImage,
		OutputFormat:    "png",
	}
	if mode == types.ModeEdit && len(r.Image) > 0 {
		// Instruction-edit models (Flux-Kontext, Qwen-Image-Edit) consume the
		// input as a REFERENCE image, not an img2img init image — init_image would
		// denoise the content away at the default strength, ignoring the input.
		// The subject leads: these models read the first reference as the subject.
		ig.RefImages = append(slices.Clone([]string(r.Image)), ig.RefImages...)
	}

	steps, cfg := r.Steps, r.CFG
	for _, l := range r.Loras {
		scale := 1.0
		if l.Scale != nil {
			scale = *l.Scale
		}
		ig.Loras = append(ig.Loras, engineclient.Lora{Path: store.LoraFileName(l.Name), Multiplier: scale})
		// A step-distillation adapter changes the sampling regime: a 4-step LoRA at
		// the base model's 20 steps and cfg 2.5 burns the output. The engine
		// launched with the base defaults, so supply the adapter's.
		known, ok := registry.LookupLora(l.Name)
		if !ok {
			continue
		}
		if steps == nil && known.Steps > 0 {
			steps = &known.Steps
		}
		if cfg == nil && known.CFG > 0 {
			cfg = &known.CFG
		}
	}
	ig.SampleParams = sampleParams(m, r, steps, cfg)
	return ig
}

// Nil when the caller asked for nothing: the model's launch defaults stand.
func sampleParams(m types.Manifest, r ImageRequest, steps *int, cfg *float64) *engineclient.SampleParams {
	// Sampler and scheduler matter for step-distilled checkpoints: their authors
	// name a pair (Rapid wants euler_a or er_sde with beta), rarely the default.
	sp := engineclient.SampleParams{
		SampleSteps:  steps,
		SampleMethod: r.Sampler,
		Scheduler:    r.Scheduler,
		Guidance:     r.Guidance,
	}
	if cfg != nil {
		// Flux is distilled (distilled_guidance); everything else uses txt_cfg.
		if sp.Guidance == nil {
			sp.Guidance = &engineclient.Guidance{}
		}
		if strings.HasPrefix(m.Architecture, "flux") {
			if sp.Guidance.DistilledGuidance == nil {
				sp.Guidance.DistilledGuidance = cfg
			}
		} else if sp.Guidance.TxtCFG == nil {
			sp.Guidance.TxtCFG = cfg
		}
	}
	if sp == (engineclient.SampleParams{}) {
		return nil
	}
	return &sp
}

// PullRequest is the body for POST /api/pull.
type PullRequest struct {
	Name  string `json:"name"`
	Quant string `json:"quant,omitempty"`
	// File pins the weights inside a repo publishing many; curated models ignore it.
	File string `json:"file,omitempty"`
	// ControlNet attaches a ControlNet to the installed model; the engine can
	// only load one at startup, so it is chosen here rather than per request.
	ControlNet     string `json:"control_net,omitempty"`
	ControlNetFile string `json:"control_net_file,omitempty"`
	// As installs under this name instead of the derived one.
	As string `json:"as,omitempty"`
}

// handlePull streams NDJSON progress lines, one JSON object per line.
func (s *Server) handlePull(w http.ResponseWriter, r *http.Request) {
	var req PullRequest
	if err := decodeJSON(r, &req); err != nil {
		fail(w, err)
		return
	}
	// Only the install name has to be a legal manifest name, checked up front:
	// finding out after a multi-gigabyte download is brutal.
	if err := requireName("name", req.Name, nil); err != nil {
		fail(w, err)
		return
	}
	if req.As != "" {
		if err := requireName("as", req.As, store.ValidModelName); err != nil {
			fail(w, err)
			return
		}
	}

	out := newNDJSON(w)
	m, err := s.pull.Pull(r.Context(), req.Name, cmp.Or(req.Quant, s.cfg.DefaultQuant), puller.Opts{
		File:           req.File,
		ControlNet:     req.ControlNet,
		ControlNetFile: req.ControlNetFile,
		As:             req.As,
	}, puller.Progress(out.status))
	if err != nil {
		out.finish("", err)
		return
	}
	out.finish(m.Name, nil)
}

// ModelRow is one GET /api/tags entry; presets list here too, being callable.
type ModelRow struct {
	Name         string     `json:"name"`
	Architecture string     `json:"architecture,omitempty"`
	Mode         types.Mode `json:"mode,omitempty"`
	Loaded       bool       `json:"loaded"`
	Base         string     `json:"base,omitempty"`     // checkpoint identity, e.g. "Qwen-Image-Edit-2511"
	Revision     string     `json:"revision,omitempty"` // HF revision or release tag
	Preset       bool       `json:"preset,omitempty"`
	Label        string     `json:"label,omitempty"`
	// Alpha reports that this model can return a transparent image, so a client
	// can offer the option instead of hard-coding which checkpoints have RGBA.
	Alpha bool `json:"alpha,omitempty"`
}

// Mode comes from archdb, not the manifest: Manifest.Mode is frozen at pull
// time, so a FLUX.2 installed before oflux learned it edits by reference
// reported "generate" forever — which is why clients grew editArches tables.
func modelRow(m types.Manifest, loaded bool) ModelRow {
	return ModelRow{
		Name:         m.Name,
		Architecture: m.Architecture,
		Mode:         archdb.ModeOf(m.Architecture, m.Mode),
		Loaded:       loaded,
		Base:         m.Base,
		Revision:     m.Revision,
		Alpha:        archdb.SupportsAlpha(m.Architecture),
	}
}

// /api/tags, /v1/models and /api/ps project this into three different wire
// shapes. The shapes are contracts; the data behind them is read in one place.
func (s *Server) installed() ([]types.Manifest, map[string]bool, error) {
	ms, err := s.store.ListManifests()
	if err != nil {
		return nil, nil, err
	}
	loaded := map[string]bool{}
	for _, n := range s.sup.Loaded() {
		loaded[n] = true
	}
	return ms, loaded, nil
}

func (s *Server) handleTags(w http.ResponseWriter, r *http.Request) {
	ms, loaded, err := s.installed()
	if err != nil {
		fail(w, err)
		return
	}
	out := make([]ModelRow, 0, len(ms))
	for _, m := range ms {
		out = append(out, modelRow(m, loaded[m.Name]))
	}
	resp := map[string]any{"models": out}
	// A malformed preset must not make the model list unreadable.
	presets, err := s.store.ListPresets()
	if err != nil {
		resp["warning"] = "presets unavailable: " + err.Error()
	}
	for _, p := range presets {
		row := ModelRow{Name: p.Name, Label: p.Label, Preset: true}
		if m, err := s.store.ReadManifest(p.Model); err == nil {
			t := modelRow(m, loaded[m.Name])
			row.Architecture, row.Mode, row.Loaded = t.Architecture, t.Mode, t.Loaded
			row.Base, row.Revision = t.Base, t.Revision
		}
		out = append(out, row)
	}
	resp["models"] = out
	writeJSON(w, http.StatusOK, resp)
}

// DeleteRequest is the body for POST /api/delete and POST /api/loras/delete.
type DeleteRequest struct {
	Name string `json:"name"`
}

func (s *Server) handleDelete(w http.ResponseWriter, r *http.Request) {
	var req DeleteRequest
	if err := decodeJSON(r, &req); err != nil {
		fail(w, err)
		return
	}
	if err := requireName("name", req.Name, store.ValidModelName); err != nil {
		fail(w, err)
		return
	}
	_ = s.sup.Unload(req.Name) // best-effort stop if running
	freed, collected, err := s.store.RemoveManifest(req.Name)
	if errors.Is(err, store.ErrManifestNotFound) {
		fail(w, coded(codeModelNotFound, notFound("model %q not installed", req.Name)))
		return
	}
	if err != nil {
		fail(w, err)
		return
	}
	resp := map[string]any{"status": "deleted", "freed_blobs": len(freed)}
	if !collected {
		// The model is gone either way; say why the disk space is not back yet.
		resp["note"] = "a pull is in progress, so unused files were left for later — they are freed by the next `oflux rm`"
	}
	writeJSON(w, http.StatusOK, resp)
}

// LoraRow is one GET /api/loras entry; uninstalled curated ones list too, so a
// client can offer them. For/Steps/CFG come from the sidecar, then the curated
// table: a hand-dropped adapter has neither and used to list blank.
type LoraRow struct {
	Name      string   `json:"name"`
	Installed bool     `json:"installed"`
	Size      int64    `json:"size,omitempty"`
	For       []string `json:"for,omitempty"`
	// Archs is the older spelling of For, still emitted for old clients.
	Archs       []string `json:"archs,omitempty"`
	Steps       int      `json:"steps,omitempty"`
	CFG         float64  `json:"cfg,omitempty"`
	Description string   `json:"description,omitempty"`
}

func (s *Server) handleLoras(w http.ResponseWriter, r *http.Request) {
	installed, err := s.store.ListLoras()
	if err != nil {
		fail(w, err)
		return
	}
	rows := make([]LoraRow, 0, len(installed))
	seen := make(map[string]bool, len(installed))
	for _, l := range installed {
		// The sidecar wins: the curated table may not know the adapter at all.
		row := LoraRow{Name: l.Name, Installed: true, Size: l.Size, For: l.For, Steps: l.Steps, CFG: l.CFG}
		if c, ok := registry.LookupLora(l.Name); ok {
			if len(row.For) == 0 {
				row.For = c.Archs
			}
			row.Steps = cmp.Or(row.Steps, c.Steps)
			row.CFG = cmp.Or(row.CFG, c.CFG)
			row.Description = c.Description
		}
		row.Archs = row.For
		rows = append(rows, row)
		seen[l.Name] = true
	}
	for _, c := range registry.AllLoras() {
		if seen[c.Name] {
			continue
		}
		rows = append(rows, LoraRow{
			Name: c.Name, For: c.Archs, Archs: c.Archs,
			Steps: c.Steps, CFG: c.CFG, Description: c.Description,
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{"loras": rows})
}

type LoraPullRequest struct {
	Name string `json:"name"`           // curated lora name or org/repo
	File string `json:"file,omitempty"` // path within the repo, when ambiguous
	As   string `json:"as,omitempty"`   // install under this name instead
	// For an arbitrary repo the sidecar is all oflux will ever know.
	For   []string `json:"for,omitempty"`
	Steps int      `json:"steps,omitempty"`
	CFG   float64  `json:"cfg,omitempty"`
}

func (s *Server) handleLoraPull(w http.ResponseWriter, r *http.Request) {
	var req LoraPullRequest
	if err := decodeJSON(r, &req); err != nil {
		fail(w, err)
		return
	}
	if err := requireName("name", req.Name, nil); err != nil {
		fail(w, err)
		return
	}
	// The install name becomes the filename the engine loads by.
	if req.As != "" {
		if err := requireName("as", req.As, store.ValidLoraName); err != nil {
			fail(w, err)
			return
		}
	}

	out := newNDJSON(w)
	name, err := s.pull.PullLora(r.Context(), req.Name, puller.LoraOpts{
		File: req.File, As: req.As, For: req.For, Steps: req.Steps, CFG: req.CFG,
	}, puller.Progress(out.status))
	out.finish(name, err)
}

func (s *Server) handleLoraDelete(w http.ResponseWriter, r *http.Request) {
	var req DeleteRequest
	if err := decodeJSON(r, &req); err != nil {
		fail(w, err)
		return
	}
	if err := requireName("name", req.Name, store.ValidLoraName); err != nil {
		fail(w, err)
		return
	}
	err := s.store.RemoveLora(req.Name)
	if errors.Is(err, store.ErrLoraNotFound) {
		fail(w, coded(codeLoraNotFound, notFound("lora %q not installed", req.Name)))
		return
	}
	if err != nil {
		fail(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"status": "deleted"})
}

func (s *Server) handlePresets(w http.ResponseWriter, r *http.Request) {
	presets, err := s.store.ListPresets()
	if err != nil {
		fail(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"presets": presets})
}

func (s *Server) handlePresetCreate(w http.ResponseWriter, r *http.Request) {
	var p store.Preset
	if err := decodeJSON(r, &p); err != nil {
		fail(w, err)
		return
	}
	if err := requireName("name", p.Name, store.ValidModelName); err != nil {
		fail(w, err)
		return
	}
	if err := requireName("model", p.Model, store.ValidModelName); err != nil {
		fail(w, err)
		return
	}
	// A bad sampler should fail now, not minutes into the first generation.
	sampler, scheduler, err := normalizeSampling(p.Sampler, p.Scheduler)
	if err != nil {
		fail(w, statusErr(http.StatusBadRequest, err))
		return
	}
	p.Sampler, p.Scheduler = sampler, scheduler
	if err := s.store.WritePreset(p); err != nil {
		fail(w, err)
		return
	}
	resp := map[string]any{"status": "created", "name": p.Name}
	if _, err := s.store.ReadManifest(p.Model); err != nil {
		resp["note"] = fmt.Sprintf("model %q is not installed yet — run: oflux pull %s", p.Model, p.Model)
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) handlePresetDelete(w http.ResponseWriter, r *http.Request) {
	var req DeleteRequest
	if err := decodeJSON(r, &req); err != nil {
		fail(w, err)
		return
	}
	if err := requireName("name", req.Name, store.ValidModelName); err != nil {
		fail(w, err)
		return
	}
	err := s.store.RemovePreset(req.Name)
	if errors.Is(err, store.ErrPresetNotFound) {
		fail(w, coded(codePresetNotFound, notFound("preset %q not found", req.Name)))
		return
	}
	if err != nil {
		fail(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"status": "deleted"})
}

func (s *Server) handlePS(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"loaded": s.sup.Loaded(), "pending": s.sup.Pending()})
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	ms, _, err := s.installed()
	if err != nil {
		fail(w, err)
		return
	}
	type model struct {
		ID     string `json:"id"`
		Object string `json:"object"`
	}
	data := make([]model, 0, len(ms))
	for _, m := range ms {
		data = append(data, model{ID: m.Name, Object: "model"})
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": data})
}
