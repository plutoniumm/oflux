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

// ImageRequest is the body for /v1/edit and /v1/generate. Images are base64 or
// data URLs. For /v1/edit, Image is the image being edited.
type ImageRequest struct {
	Model           string                 `json:"model"`
	Prompt          string                 `json:"prompt"`
	NegativePrompt  string                 `json:"negative_prompt,omitempty"`
	Image           string                 `json:"image,omitempty"`
	RefImages       []string               `json:"ref_images,omitempty"`
	ControlImage    string                 `json:"control_image,omitempty"`
	ControlStrength *float64               `json:"control_strength,omitempty"`
	MaskImage       string                 `json:"mask_image,omitempty"`
	Strength        *float64               `json:"strength,omitempty"`
	Steps           *int                   `json:"steps,omitempty"`
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

	sp := &engineclient.SampleParams{}
	used := false
	if r.Steps != nil {
		sp.SampleSteps = r.Steps
		used = true
	}

	// cfg override: Flux models are distilled (distilled_guidance); everything
	// else uses a true txt_cfg.
	g := r.Guidance
	if r.CFG != nil {
		if g == nil {
			g = &engineclient.Guidance{}
		}
		if strings.HasPrefix(m.Architecture, "flux") {
			if g.DistilledGuidance == nil {
				g.DistilledGuidance = r.CFG
			}
		} else if g.TxtCFG == nil {
			g.TxtCFG = r.CFG
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

	m, err := s.pull.Pull(r.Context(), req.Name, quant, puller.Progress(emit))
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
	freed, err := s.store.RemoveManifest(req.Name)
	if errors.Is(err, store.ErrManifestNotFound) {
		writeErr(w, http.StatusNotFound, fmt.Sprintf("model %q not installed", req.Name))
		return
	}
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"status": "deleted", "freed_blobs": len(freed)})
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
