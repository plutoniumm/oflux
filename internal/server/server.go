// Package server exposes oflux's HTTP API on the daemon port (default 11534):
// a clean JSON edit/generate surface that maps onto the sd-server native
// img_gen API, plus model-management endpoints (pull/list/delete/ps).
package server

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strings"

	"oflux/internal/engineclient"
	"oflux/internal/puller"
	"oflux/internal/registry"
	"oflux/internal/store"
	"oflux/internal/supervisor"
	"oflux/internal/types"
)

// Server wires the HTTP API to the store, supervisor, and puller.
type Server struct {
	store *store.Store
	sup   *supervisor.Supervisor
	pull  *puller.Puller
	cfg   types.Config
}

// New constructs a Server.
func New(st *store.Store, sup *supervisor.Supervisor, pull *puller.Puller, cfg types.Config) *Server {
	return &Server{store: st, sup: sup, pull: pull, cfg: cfg}
}

// Handler returns the HTTP handler with all routes registered.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /{$}", s.handleUI) // "/{$}" matches the root exactly
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) { w.Write([]byte("ok")) })
	mux.HandleFunc("POST /v1/edit", s.handleImage(types.ModeEdit))
	mux.HandleFunc("POST /v1/generate", s.handleImage(types.ModeGenerate))
	mux.HandleFunc("POST /v1/images/edits", s.handleOpenAIEdit)
	mux.HandleFunc("POST /v1/images/generations", s.handleOpenAIGenerate)
	mux.HandleFunc("POST /api/pull", s.handlePull)
	mux.HandleFunc("GET /api/tags", s.handleTags)
	mux.HandleFunc("POST /api/delete", s.handleDelete)
	mux.HandleFunc("GET /api/ps", s.handlePS)
	mux.HandleFunc("GET /api/loras", s.handleLoras)
	mux.HandleFunc("POST /api/loras/pull", s.handleLoraPull)
	mux.HandleFunc("POST /api/loras/delete", s.handleLoraDelete)
	mux.HandleFunc("GET /v1/models", s.handleModels)
	return guard(mux)
}

// maxBody caps request bodies. Images arrive base64-encoded in JSON, so this is
// generous enough for large inputs while preventing a single request from
// exhausting memory.
const maxBody = 256 << 20 // 256 MiB

// guard protects the loopback API from web pages. Without it any site the user
// visits could POST to 127.0.0.1:11534 (a CORS "simple request" needs no
// preflight) and delete models or queue pulls, and a DNS-rebinding page could
// read the responses. We therefore require the Host header to be loopback and
// reject cross-origin requests outright. It also bounds request bodies.
func guard(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !loopbackHost(r.Host) {
			writeErr(w, http.StatusForbidden, "forbidden: unexpected Host header")
			return
		}
		if origin := r.Header.Get("Origin"); origin != "" && !loopbackOrigin(origin) {
			writeErr(w, http.StatusForbidden, "forbidden: cross-origin request")
			return
		}
		r.Body = http.MaxBytesReader(w, r.Body, maxBody)
		next.ServeHTTP(w, r)
	})
}

// loopbackHost reports whether a Host header names the local machine.
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

// loopbackOrigin reports whether an Origin header refers to the local machine.
func loopbackOrigin(origin string) bool {
	u, err := url.Parse(origin)
	if err != nil {
		return false
	}
	return loopbackHost(u.Host)
}

// ---- request/response types (the clean oflux JSON API) ----

// LoraRef selects an installed LoRA adapter for one request. Scale is the
// adapter multiplier; omitted means 1.0 (full strength).
type LoraRef struct {
	Name  string   `json:"name"`
	Scale *float64 `json:"scale,omitempty"`
}

// ImageRequest is the body for /v1/edit and /v1/generate. Images are base64 or
// data URLs. For /v1/edit, Image is the image being edited.
type ImageRequest struct {
	Model           string                 `json:"model"`
	Prompt          string                 `json:"prompt"`
	Loras           []LoraRef              `json:"loras,omitempty"`
	NegativePrompt  string                 `json:"negative_prompt,omitempty"`
	Image           string                 `json:"image,omitempty"`
	RefImages       []string               `json:"ref_images,omitempty"`
	ControlImage    string                 `json:"control_image,omitempty"`
	ControlStrength *float64               `json:"control_strength,omitempty"`
	MaskImage       string                 `json:"mask_image,omitempty"`
	Strength        *float64               `json:"strength,omitempty"`
	Steps           *int                   `json:"steps,omitempty"`
	Sampler         string                 `json:"sampler,omitempty"`
	Scheduler       string                 `json:"scheduler,omitempty"`
	Seed            *int64                 `json:"seed,omitempty"`
	Width           *int                   `json:"width,omitempty"`
	Height          *int                   `json:"height,omitempty"`
	CFG             *float64               `json:"cfg,omitempty"`
	Guidance        *engineclient.Guidance `json:"guidance,omitempty"`
}

// ImageResponse returns the generated image(s) as base64 PNG.
type ImageResponse struct {
	Model  string   `json:"model"`
	Images []string `json:"images"`
}

func (s *Server) handleImage(mode types.Mode) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req ImageRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeErr(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
			return
		}
		if req.Model == "" {
			writeErr(w, http.StatusBadRequest, "model is required")
			return
		}
		if mode == types.ModeEdit && req.Image == "" && len(req.RefImages) == 0 {
			writeErr(w, http.StatusBadRequest, "edit requires an image (or ref_images)")
			return
		}
		// The route, not the manifest, decides how the input is used — a hybrid
		// model serves both endpoints with the same weights.
		name, img, code, err := s.generate(r.Context(), req.Model, req, mode)
		if err != nil {
			writeErr(w, code, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, ImageResponse{
			Model:  name,
			Images: []string{base64.StdEncoding.EncodeToString(img)},
		})
	}
}

// generate reads the model manifest and runs one image through the supervisor.
// It returns the resolved model name, the image bytes, and an HTTP status code
// to use if err is non-nil. Shared by the native and OpenAI-compatible handlers.
func (s *Server) generate(ctx context.Context, model string, req ImageRequest, mode types.Mode) (string, []byte, int, error) {
	m, err := s.store.ReadManifest(model)
	if errors.Is(err, store.ErrManifestNotFound) {
		return "", nil, http.StatusNotFound, fmt.Errorf("model %q not installed — run: oflux pull %s", model, model)
	}
	if err != nil {
		return "", nil, http.StatusInternalServerError, err
	}
	// Refuse a request the model cannot serve, rather than letting the engine
	// silently ignore the input image (or invent a subject from nothing).
	if mode == types.ModeEdit && !m.Mode.CanEdit() {
		return "", nil, http.StatusBadRequest,
			fmt.Errorf("model %q is generate-only; use /v1/generate", m.Name)
	}
	if mode == types.ModeGenerate && !m.Mode.CanGenerate() {
		return "", nil, http.StatusBadRequest,
			fmt.Errorf("model %q is edit-only; use /v1/edit with an image", m.Name)
	}
	// Reject an unknown sampler/scheduler here: the engine surfaces one as an
	// opaque job failure, and only after loading the model.
	sampler, scheduler, err := normalizeSampling(req.Sampler, req.Scheduler)
	if err != nil {
		return "", nil, http.StatusBadRequest, err
	}
	req.Sampler, req.Scheduler = sampler, scheduler

	// A ControlNet is loaded at engine startup, so a model installed without one
	// can never honour control_image. Forwarding it anyway meant the engine
	// quietly ignored it and returned an ordinary image — the request looked
	// like it worked. Say so instead.
	if req.ControlImage != "" {
		if _, ok := m.Component(types.RoleControlNet); !ok {
			return "", nil, http.StatusBadRequest, fmt.Errorf(
				"model %q has no control net, so control_image would be ignored — reinstall it with one: oflux pull <repo> --control-net <org/repo> --as %s-control",
				m.Name, m.Name)
		}
	}

	// Validate LoRAs before spawning an engine: a bad name is a client error,
	// and a missing adapter would otherwise surface as an opaque engine failure
	// several minutes into a model load.
	for _, l := range req.Loras {
		if err := store.ValidLoraName(l.Name); err != nil {
			return "", nil, http.StatusBadRequest, err
		}
		if !s.store.HasLora(l.Name) {
			return "", nil, http.StatusNotFound,
				fmt.Errorf("lora %q not installed — run: oflux lora pull %s", l.Name, l.Name)
		}
	}
	img, err := s.sup.Generate(ctx, m, buildImgGen(m, req, mode))
	if err != nil {
		return "", nil, http.StatusInternalServerError, fmt.Errorf("generation failed: %w", err)
	}
	return m.Name, img, http.StatusOK, nil
}

// buildImgGen maps the oflux request onto the native sd-server img_gen request.
// Model defaults (steps, cfg, flow-shift, sampling method) are baked into the
// engine launch flags at pull time, so only caller-supplied OVERRIDES are sent
// here — an omitted field keeps the model's default.
func buildImgGen(m types.Manifest, r ImageRequest, mode types.Mode) engineclient.ImgGenRequest {
	ig := engineclient.ImgGenRequest{
		Prompt:          r.Prompt,
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
	if mode == types.ModeEdit && r.Image != "" {
		// Instruction-edit models (Flux-Kontext, Qwen-Image-Edit) consume the
		// input as a REFERENCE image, not an img2img init image — init_image would
		// denoise the content away at the default strength, ignoring the input.
		ig.RefImages = append([]string{r.Image}, ig.RefImages...)
	}

	steps, cfg := r.Steps, r.CFG
	for _, l := range r.Loras {
		scale := 1.0
		if l.Scale != nil {
			scale = *l.Scale
		}
		ig.Loras = append(ig.Loras, engineclient.Lora{
			Path:       store.LoraFileName(l.Name),
			Multiplier: scale,
		})
		// A step-distillation adapter changes the sampling regime of the model it
		// is applied to: running a 4-step LoRA at the base model's 20 steps and
		// cfg 2.5 produces burnt, over-saturated output. The engine launched with
		// the base model's defaults, so supply the adapter's instead — unless the
		// caller asked for specific values, which always win.
		if known, ok := registry.LookupLora(l.Name); ok {
			if steps == nil && known.Steps > 0 {
				steps = &known.Steps
			}
			if cfg == nil && known.CFG > 0 {
				cfg = &known.CFG
			}
		}
	}

	sp := &engineclient.SampleParams{}
	used := false
	if steps != nil {
		sp.SampleSteps = steps
		used = true
	}
	// Sampler and scheduler matter a lot for step-distilled checkpoints — their
	// authors usually name a specific pair (the Rapid line wants euler_a or
	// er_sde with the beta schedule) and the model's launch default is rarely it.
	if r.Sampler != "" {
		sp.SampleMethod = r.Sampler
		used = true
	}
	if r.Scheduler != "" {
		sp.Scheduler = r.Scheduler
		used = true
	}

	// cfg override: Flux models are distilled (distilled_guidance); everything
	// else uses a true txt_cfg.
	g := r.Guidance
	if cfg != nil {
		if g == nil {
			g = &engineclient.Guidance{}
		}
		if strings.HasPrefix(m.Architecture, "flux") {
			if g.DistilledGuidance == nil {
				g.DistilledGuidance = cfg
			}
		} else if g.TxtCFG == nil {
			g.TxtCFG = cfg
		}
	}
	if g != nil {
		sp.Guidance = g
		used = true
	}
	if used {
		ig.SampleParams = sp
	}
	return ig
}

// ---- model management ----

// PullRequest is the body for POST /api/pull.
type PullRequest struct {
	Name  string `json:"name"`
	Quant string `json:"quant,omitempty"`
	// File pins the exact diffusion weights inside a Hugging Face repo that
	// publishes many builds. Ignored for curated models.
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
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}
	if req.Name == "" {
		writeErr(w, http.StatusBadRequest, "name is required")
		return
	}
	quant := req.Quant
	if quant == "" {
		quant = s.cfg.DefaultQuant
	}

	w.Header().Set("Content-Type", "application/x-ndjson")
	flusher, _ := w.(http.Flusher)
	enc := json.NewEncoder(w)
	emit := func(status string) {
		enc.Encode(map[string]string{"status": status})
		if flusher != nil {
			flusher.Flush()
		}
	}

	m, err := s.pull.Pull(r.Context(), req.Name, quant, puller.Opts{
		File:           req.File,
		ControlNet:     req.ControlNet,
		ControlNetFile: req.ControlNetFile,
		As:             req.As,
	}, puller.Progress(emit))
	if err != nil {
		enc.Encode(map[string]string{"error": err.Error()})
		if flusher != nil {
			flusher.Flush()
		}
		return
	}
	enc.Encode(map[string]string{"status": "success", "name": m.Name})
	if flusher != nil {
		flusher.Flush()
	}
}

func (s *Server) handleTags(w http.ResponseWriter, r *http.Request) {
	ms, err := s.store.ListManifests()
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	loaded := map[string]bool{}
	for _, n := range s.sup.Loaded() {
		loaded[n] = true
	}
	type row struct {
		Name         string     `json:"name"`
		Architecture string     `json:"architecture"`
		Mode         types.Mode `json:"mode"`
		Loaded       bool       `json:"loaded"`
	}
	out := make([]row, 0, len(ms))
	for _, m := range ms {
		out = append(out, row{m.Name, m.Architecture, m.Mode, loaded[m.Name]})
	}
	writeJSON(w, http.StatusOK, map[string]any{"models": out})
}

// DeleteRequest is the body for POST /api/delete.
type DeleteRequest struct {
	Name string `json:"name"`
}

func (s *Server) handleDelete(w http.ResponseWriter, r *http.Request) {
	var req DeleteRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}
	_ = s.sup.Unload(req.Name) // best-effort stop if running
	freed, collected, err := s.store.RemoveManifest(req.Name)
	if errors.Is(err, store.ErrManifestNotFound) {
		writeErr(w, http.StatusNotFound, fmt.Sprintf("model %q not installed", req.Name))
		return
	}
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	resp := map[string]any{"status": "deleted", "freed_blobs": len(freed)}
	if !collected {
		// The model is gone either way; say why the disk space is not back yet.
		resp["note"] = "a pull is in progress, so unused files were left for later — they are freed by the next `oflux rm`"
	}
	writeJSON(w, http.StatusOK, resp)
}

// ---- LoRA management ----

// LoraRow is one entry in GET /api/loras. Installed adapters carry a size;
// curated-but-not-installed ones are listed too so a client can offer them.
type LoraRow struct {
	Name        string   `json:"name"`
	Installed   bool     `json:"installed"`
	Size        int64    `json:"size,omitempty"`
	Archs       []string `json:"archs,omitempty"`
	Steps       int      `json:"steps,omitempty"`
	Description string   `json:"description,omitempty"`
}

func (s *Server) handleLoras(w http.ResponseWriter, r *http.Request) {
	installed, err := s.store.ListLoras()
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	rows := make([]LoraRow, 0, len(installed))
	seen := make(map[string]bool, len(installed))
	for _, l := range installed {
		row := LoraRow{Name: l.Name, Installed: true, Size: l.Size}
		if c, ok := registry.LookupLora(l.Name); ok {
			row.Archs, row.Steps, row.Description = c.Archs, c.Steps, c.Description
		}
		rows = append(rows, row)
		seen[l.Name] = true
	}
	for _, c := range registry.AllLoras() {
		if seen[c.Name] {
			continue
		}
		rows = append(rows, LoraRow{
			Name: c.Name, Archs: c.Archs, Steps: c.Steps, Description: c.Description,
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{"loras": rows})
}

// LoraPullRequest is the body for POST /api/loras/pull.
type LoraPullRequest struct {
	Name string `json:"name"`           // curated lora name or org/repo
	File string `json:"file,omitempty"` // path within the repo, when ambiguous
	As   string `json:"as,omitempty"`   // install under this name instead
}

// handleLoraPull streams NDJSON progress, matching /api/pull.
func (s *Server) handleLoraPull(w http.ResponseWriter, r *http.Request) {
	var req LoraPullRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}
	if req.Name == "" {
		writeErr(w, http.StatusBadRequest, "name is required")
		return
	}

	w.Header().Set("Content-Type", "application/x-ndjson")
	flusher, _ := w.(http.Flusher)
	enc := json.NewEncoder(w)
	emit := func(status string) {
		enc.Encode(map[string]string{"status": status})
		if flusher != nil {
			flusher.Flush()
		}
	}

	name, err := s.pull.PullLora(r.Context(), req.Name, req.File, req.As, puller.Progress(emit))
	if err != nil {
		enc.Encode(map[string]string{"error": err.Error()})
	} else {
		enc.Encode(map[string]string{"status": "success", "name": name})
	}
	if flusher != nil {
		flusher.Flush()
	}
}

func (s *Server) handleLoraDelete(w http.ResponseWriter, r *http.Request) {
	var req DeleteRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}
	err := s.store.RemoveLora(req.Name)
	if errors.Is(err, store.ErrLoraNotFound) {
		writeErr(w, http.StatusNotFound, fmt.Sprintf("lora %q not installed", req.Name))
		return
	}
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"status": "deleted"})
}

func (s *Server) handlePS(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"loaded": s.sup.Loaded()})
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	ms, _ := s.store.ListManifests()
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

// ---- helpers ----

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, code int, msg string) {
	writeJSON(w, code, map[string]any{"error": msg})
}
