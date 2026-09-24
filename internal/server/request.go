package server

// The API edge: one decoder, one error writer, one NDJSON writer and every
// request check, so the two API surfaces cannot drift apart.

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"io"
	"math/rand/v2"
	"net/http"
	"strings"
	"time"

	"oflux/internal/archdb"
	"oflux/internal/engineclient"
	"oflux/internal/store"
	"oflux/internal/types"
)

// Codes are wire contract: one may be added, none may change meaning.
const (
	codeInvalidRequest = "invalid_request"
	codeModelNotFound  = "model_not_found"
	codeLoraNotFound   = "lora_not_found"
	codePresetNotFound = "preset_not_found"
	codeNoMatch        = "no_match"
	codeJobNotFound    = "job_not_found"
	codeNotFound       = "not_found"
	codeForbidden      = "forbidden"
	codeTooLarge       = "payload_too_large"
	codeTimeout        = "timeout"
	codeEngineFailed   = "engine_failed"
	codeQueueFull      = "queue_full"
	codeCancelled      = "cancelled"
	codeUnsupported    = "unsupported"
	codeInternal       = "internal"
)

// A check returns a plain error; one writer decides the wire. No status = 500.
type statusError struct {
	code   int    // HTTP status
	slug   string // stable machine-readable code
	model  string // the model the request resolved to, when known
	detail string // subsystem text, possibly long; truncated on the way out
	err    error
}

func (e *statusError) Error() string { return e.err.Error() }
func (e *statusError) Unwrap() error { return e.err }

func statusErr(code int, err error) error {
	return &statusError{code: code, slug: slugFor(code), err: err}
}

// coded narrows a slug where it helps a client act: a model can be pulled.
func coded(slug string, err error) error {
	var se *statusError
	if errors.As(err, &se) {
		se.slug = slug
	}
	return err
}

// withModel names the model, so a supervisor failure says what was running.
func withModel(err error, model string) error {
	if err == nil || model == "" {
		return err
	}
	var se *statusError
	if errors.As(err, &se) {
		if se.model == "" {
			se.model = model
		}
		return err
	}
	return &statusError{code: http.StatusInternalServerError, slug: codeInternal, model: model, err: err}
}

func badRequest(format string, a ...any) error {
	return &statusError{code: http.StatusBadRequest, slug: codeInvalidRequest, err: fmt.Errorf(format, a...)}
}

func notFound(format string, a ...any) error {
	return &statusError{code: http.StatusNotFound, slug: codeNotFound, err: fmt.Errorf(format, a...)}
}

func slugFor(code int) string {
	switch {
	case code == http.StatusRequestEntityTooLarge:
		return codeTooLarge
	case code == http.StatusGatewayTimeout || code == http.StatusRequestTimeout:
		return codeTimeout
	case code == http.StatusNotImplemented:
		return codeUnsupported
	case code == http.StatusNotFound:
		return codeNotFound
	case code == http.StatusForbidden:
		return codeForbidden
	case code >= 500:
		return codeInternal
	case code >= 400:
		return codeInvalidRequest
	}
	return codeInternal
}

// "error" stays the short message existing clients print.
type errorBody struct {
	Error  string `json:"error"`
	Code   string `json:"code"`
	Model  string `json:"model,omitempty"`
	Detail string `json:"detail,omitempty"`
}

// A failed sd.cpp job reports its whole log, hundreds of lines, and the part
// that explains the failure is at the end — so the tail is what survives.
const maxDetail = 2000

// 400 caller-fixable, 404 missing, 413 oversized, 429 saturated, 502 engine.
func apiError(err error) (int, errorBody) {
	code := http.StatusInternalServerError
	body := errorBody{Error: err.Error(), Code: codeInternal}

	var tooBig *http.MaxBytesError
	var se *statusError
	switch {
	case errors.As(err, &tooBig):
		code, body.Code = http.StatusRequestEntityTooLarge, codeTooLarge
	case errors.As(err, &se):
		code, body.Code, body.Model, body.Detail = se.code, se.slug, se.model, se.detail
	}

	// Otherwise every failed edit dumps the engine log into a client that prints
	// nothing but "error".
	var je *engineclient.JobError
	if errors.As(err, &je) {
		if line := shortReason(je.Reason); line != "" {
			body.Error = fmt.Sprintf("engine job %s: %s", je.Status, line)
		} else {
			body.Error = fmt.Sprintf("engine job %s", je.Status)
		}
		if body.Detail == "" {
			body.Detail = truncateDetail(je.Reason)
		}
		if se == nil {
			code = http.StatusBadGateway
			body.Code = codeEngineFailed
		}
		if je.Status == "cancelled" && body.Code == codeEngineFailed {
			body.Code = codeCancelled
		}
	}
	return code, body
}

func truncateDetail(s string) string {
	s = strings.TrimSpace(s)
	if len(s) <= maxDetail {
		return s
	}
	cut := s[len(s)-maxDetail:]
	if i := strings.IndexByte(cut, '\n'); i >= 0 {
		cut = cut[i+1:]
	}
	cut = strings.ToValidUTF8(cut, "")
	return fmt.Sprintf("… (%d earlier bytes elided)\n%s", len(s)-len(cut), cut)
}

// The last non-empty line is where the actual error lands, after the load chatter.
func shortReason(s string) string {
	lines := strings.Split(strings.TrimSpace(s), "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		t := strings.TrimSpace(lines[i])
		if t == "" {
			continue
		}
		if len(t) > 200 {
			t = strings.ToValidUTF8(t[:200], "") + "…"
		}
		return t
	}
	return ""
}

func fail(w http.ResponseWriter, err error) {
	code, body := apiError(err)
	retryAfter(w, code)
	writeJSON(w, code, body)
}

// At least the tail of a sampling run; the supervisor exposes no queue timings.
func retryAfter(w http.ResponseWriter, code int) {
	if code == http.StatusTooManyRequests {
		w.Header().Set("Retry-After", "5")
	}
}

// OpenAI clients parse {"error":{"message","type"}}; handed the native oflux
// object they surface a plain 400 as "unknown error".
func failOpenAI(w http.ResponseWriter, err error) {
	code, body := apiError(err)
	retryAfter(w, code)
	kind := "api_error"
	if code < 500 {
		kind = "invalid_request_error"
	}
	writeJSON(w, code, map[string]any{"error": map[string]any{
		"message": body.Error,
		"type":    kind,
		"code":    body.Code,
	}})
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

// One JSON object per line, flushed as written. /api/pull and /api/loras/pull
// share a shape: {"status":...}, then {"status":"success","name":...} or an error.
type ndjson struct {
	enc   *json.Encoder
	flush func()
	wrote bool
}

func newNDJSON(w http.ResponseWriter) *ndjson {
	w.Header().Set("Content-Type", "application/x-ndjson")
	n := &ndjson{enc: json.NewEncoder(w), flush: func() {}}
	if f, ok := w.(http.Flusher); ok {
		n.flush = f.Flush
	}
	return n
}

func (n *ndjson) line(v any) {
	n.wrote = true
	_ = n.enc.Encode(v)
	n.flush()
}

func (n *ndjson) status(s string) { n.line(map[string]string{"status": s}) }

// finish writes the one terminal line of the stream.
func (n *ndjson) finish(name string, err error) {
	if err != nil {
		n.line(map[string]string{"error": err.Error()})
		return
	}
	n.line(map[string]string{"status": "success", "name": name})
}

// An empty or oversized body used to arrive as a confusing "invalid JSON" parse
// error; they now report themselves, and fail maps the oversized one to 413.
func decodeJSON(r *http.Request, v any) error {
	switch err := json.NewDecoder(r.Body).Decode(v); {
	case err == nil:
		return nil
	case errors.Is(err, io.EOF):
		return badRequest("request body is empty: expected a JSON object")
	default:
		return badRequest("invalid JSON: %w", err)
	}
}

// ImageInput accepts one base64 string or an array: the instruction-edit models
// take several, the first the subject and the rest references.
type ImageInput []string

func (im *ImageInput) UnmarshalJSON(b []byte) error {
	b = bytes.TrimSpace(b)
	switch {
	case len(b) == 0 || string(b) == "null":
		*im = nil
	case b[0] == '[':
		var a []string
		if err := json.Unmarshal(b, &a); err != nil {
			return errors.New("image array must contain base64 strings")
		}
		*im = a
	case b[0] == '"':
		var s string
		if err := json.Unmarshal(b, &s); err != nil {
			return err
		}
		if s == "" {
			*im = nil
			return nil
		}
		*im = ImageInput{s}
	default:
		return errors.New("image must be a base64 string or an array of them")
	}
	return nil
}

func (im ImageInput) First() string {
	if len(im) == 0 {
		return ""
	}
	return im[0]
}

// Duration accepts "10m" or a number of seconds, the two forms clients send.
type Duration time.Duration

func (d *Duration) UnmarshalJSON(b []byte) error {
	b = bytes.TrimSpace(b)
	if len(b) == 0 || string(b) == "null" {
		*d = 0
		return nil
	}
	if b[0] == '"' {
		var s string
		if err := json.Unmarshal(b, &s); err != nil {
			return err
		}
		if strings.TrimSpace(s) == "" {
			*d = 0
			return nil
		}
		v, err := time.ParseDuration(s)
		if err != nil {
			return fmt.Errorf("keep_alive %q: want a duration like \"10m\" or a number of seconds", s)
		}
		*d = Duration(v)
		return nil
	}
	var secs float64
	if err := json.Unmarshal(b, &secs); err != nil {
		return errors.New("keep_alive must be a duration string or a number of seconds")
	}
	*d = Duration(time.Duration(secs * float64(time.Second)))
	return nil
}

// A name becomes a path segment or an engine argument. Without this check an
// empty or traversal-shaped one reached the store and came back as a 500.
func requireName(field, name string, valid func(string) error) error {
	if name == "" {
		return badRequest("%s is required", field)
	}
	if valid != nil {
		if err := valid(name); err != nil {
			return statusErr(http.StatusBadRequest, err)
		}
	}
	return nil
}

// The engine reports a value it cannot honour as an opaque job failure *after*
// loading the model, so a nonsensical request has to die here instead.
const (
	minDimension = 64
	maxDimension = 8192
	maxSteps     = 500
	maxCFG       = 100
	maxLoraScale = 10
	// So a runaway client cannot hand the engine 100MB of references to tokenize.
	maxImages = 16
)

// Every image field accepts a data URL because browsers produce one
// (FileReader.readAsDataURL — the UI sends exactly that) while sd-server takes
// bare base64. The payload is checked here too: an unusable image otherwise
// surfaces as a failed job minutes into a model load.
func normalizeImage(field, v string) (string, error) {
	s := strings.TrimSpace(v)
	if rest, ok := strings.CutPrefix(s, "data:"); ok {
		comma := strings.IndexByte(rest, ',')
		if comma < 0 {
			return "", badRequest("%s: malformed data URL", field)
		}
		if !strings.Contains(strings.ToLower(rest[:comma]), ";base64") {
			return "", badRequest("%s: data URL must be base64-encoded", field)
		}
		s = rest[comma+1:] // a substring shares the backing array: no copy
	}
	if s == "" {
		return "", badRequest("%s is empty", field)
	}
	if strings.ContainsAny(s, " \t\r\n") {
		// `openssl base64` wraps at 76 columns and the engine's decoder does not
		// skip line breaks. Rare enough that the copy never happens normally.
		s = strings.Join(strings.Fields(s), "")
	}
	if !isBase64(s) {
		return "", badRequest("%s is not valid base64", field)
	}
	return s, nil
}

// Tolerates line breaks and missing padding as the engine's decoder does, and
// does not decode: megabytes materialized to be thrown away would double the
// memory cost of every request.
func isBase64(s string) bool {
	n, pad := 0, 0
	for i := 0; i < len(s); i++ {
		switch c := s[i]; {
		case c == '\n', c == '\r', c == ' ', c == '\t':
		case c == '=':
			pad++
		case pad > 0:
			return false // data after the padding
		case c >= 'A' && c <= 'Z', c >= 'a' && c <= 'z', c >= '0' && c <= '9', c == '+', c == '/':
			n++
		default:
			return false
		}
	}
	if pad > 2 {
		return false
	}
	if pad > 0 {
		return (n+pad)%4 == 0
	}
	return n%4 != 1
}

// One fixed order: request shape (400, needs no model), then the model (404),
// then everything depending on it. Past this is the point of no return — a bad
// field there costs a multi-minute model load before the engine fails the job
// unhelpfully. It normalizes in place, so nothing downstream sees a data URL.
func (s *Server) validate(req *ImageRequest, mode types.Mode) (types.Manifest, error) {
	if err := requireName("model", req.Model, store.ValidModelName); err != nil {
		return types.Manifest{}, err
	}
	if err := s.applyPreset(req); err != nil {
		return types.Manifest{}, err
	}
	// mask is the short spelling of mask_image: WHITE marks the region to edit.
	if req.MaskImage == "" {
		req.MaskImage = req.Mask
	}
	if req.MaskPrompt != "" {
		if mode != types.ModeEdit {
			return types.Manifest{}, badRequest("mask_prompt is only used by /v1/edit")
		}
		if len(req.Image) == 0 {
			return types.Manifest{}, badRequest("mask_prompt needs an image to segment")
		}
		if err := checkPrompt("mask_prompt", req.MaskPrompt); err != nil {
			return types.Manifest{}, err
		}
	}
	// The engine picks its own seed and never reports it: resolve it here, so the
	// response can echo what ran. A negative seed is the same "surprise me".
	if req.Seed == nil || *req.Seed < 0 {
		seed := randomSeed()
		req.Seed = &seed
	}

	// The route, not the manifest, decides how the input image is used — a
	// hybrid model serves both endpoints with the same weights.
	if mode == types.ModeEdit && len(req.Image) == 0 && len(req.RefImages) == 0 {
		return types.Manifest{}, badRequest("edit requires an image (or ref_images)")
	}
	if mode == types.ModeGenerate && len(req.Image) > 0 {
		return types.Manifest{}, badRequest("image is only used by /v1/edit; POST there, or pass ref_images to generate from a reference")
	}
	if n := len(req.Image) + len(req.RefImages); n > maxImages {
		return types.Manifest{}, badRequest("too many images: %d, the limit is %d", n, maxImages)
	}
	if err := validateRanges(req); err != nil {
		return types.Manifest{}, err
	}

	m, err := s.store.ReadManifest(req.Model)
	if errors.Is(err, store.ErrManifestNotFound) {
		return types.Manifest{}, coded(codeModelNotFound,
			notFound("model %q not installed — run: oflux pull %s", req.Model, req.Model))
	}
	if err != nil {
		return types.Manifest{}, err
	}

	// From archdb, not the manifest: Manifest.Mode is frozen at pull time, so a
	// FLUX.2 pulled before oflux knew it edits would have its edits refused.
	can := archdb.ModeOf(m.Architecture, m.Mode)

	// Else the engine silently ignores the input image, or invents a subject.
	if mode == types.ModeEdit && !can.CanEdit() {
		return m, badRequest("model %q is generate-only; use /v1/generate", m.Name)
	}
	if mode == types.ModeGenerate && !can.CanGenerate() {
		return m, badRequest("model %q is edit-only; use /v1/edit with an image", m.Name)
	}
	if len(req.RefImages) > 0 && !can.CanEdit() {
		return m, badRequest("model %q takes no reference images, so ref_images would be ignored", m.Name)
	}

	// Reject an unknown sampler/scheduler here: the engine surfaces one as an
	// opaque job failure, and only after loading the model.
	sampler, scheduler, err := normalizeSampling(req.Sampler, req.Scheduler)
	if err != nil {
		return m, statusErr(http.StatusBadRequest, err)
	}
	req.Sampler, req.Scheduler = sampler, scheduler

	// A ControlNet is loaded at engine startup, so a model installed without one
	// can never honour control_image. Forwarding it anyway meant the engine
	// quietly ignored it and returned an ordinary image — the request looked
	// like it worked. Say so instead.
	if req.ControlImage != "" {
		if _, ok := m.Component(types.RoleControlNet); !ok {
			return m, badRequest(
				"model %q has no control net, so control_image would be ignored — reinstall it with one: oflux pull <repo> --control-net <org/repo> --as %s-control",
				m.Name, m.Name)
		}
	}

	// Alpha is a property of the weights, not a switch, so a model that lacks
	// it would silently return an opaque image and look like it worked.
	if req.Transparent && !archdb.SupportsAlpha(m.Architecture) {
		return m, badRequest("model %q cannot emit transparency — install one that can: oflux pull qwe-2.1", m.Name)
	}

	// Before spawning an engine: a missing adapter would otherwise surface as an
	// opaque failure several minutes into a model load.
	for _, l := range req.Loras {
		if err := store.ValidLoraName(l.Name); err != nil {
			return m, statusErr(http.StatusBadRequest, err)
		}
		if !s.store.HasLora(l.Name) {
			return m, coded(codeLoraNotFound,
				notFound("lora %q not installed — run: oflux lora pull %s", l.Name, l.Name))
		}
		if l.Scale != nil && (*l.Scale < -maxLoraScale || *l.Scale > maxLoraScale) {
			return m, badRequest("lora %q scale must be between %d and %d, got %g", l.Name, -maxLoraScale, maxLoraScale, *l.Scale)
		}
	}

	// Images last: they are the megabytes, and scanning them is only worth it
	// once the rest of the request is known to be good.
	maskField := "mask_image"
	if req.Mask != "" && req.Mask == req.MaskImage {
		maskField = "mask"
	}
	for _, f := range []struct {
		name string
		p    *string
	}{
		{"control_image", &req.ControlImage},
		{maskField, &req.MaskImage},
	} {
		if *f.p == "" {
			continue
		}
		v, err := normalizeImage(f.name, *f.p)
		if err != nil {
			return m, err
		}
		*f.p = v
	}
	for i := range req.Image {
		v, err := normalizeImage(imageField(i, len(req.Image)), req.Image[i])
		if err != nil {
			return m, err
		}
		req.Image[i] = v
	}
	for i := range req.RefImages {
		v, err := normalizeImage(fmt.Sprintf("ref_images[%d]", i), req.RefImages[i])
		if err != nil {
			return m, err
		}
		req.RefImages[i] = v
	}

	// sd-server's canvas defaults to 512x512 and it never infers one from the
	// reference image, so an edit that named no size came back downscaled —
	// silently, however large the input was. Every edit through the web UI was
	// a 512x512 thumbnail of itself. Match the subject instead.
	if mode == types.ModeEdit && req.Width == nil && req.Height == nil {
		src := req.Image.First()
		if src == "" && len(req.RefImages) > 0 {
			src = req.RefImages[0]
		}
		if w, h, ok := canvasFor(src); ok {
			req.Width, req.Height = &w, &h
		}
	}
	return m, nil
}

// editCanvasMax caps the canvas an input image can ask for on its own. Cost
// grows with the pixel count, so an unclamped phone photograph would quietly
// commission an hour of sampling; 2048 is the largest any curated model claims
// to render natively. An explicit width/height is still honoured up to
// maxDimension — this ceiling only applies to the size we infer.
// ponytail: fixed ceiling, make it per-arch if a model lands that wants more.
const editCanvasMax = 2048

// canvasFor reads a base64 image's header and returns the nearest canvas the
// engine will accept: a multiple of 64, which every supported architecture's
// patch size divides. ok=false when there is nothing decodable to measure, and
// the caller then leaves the engine on its own default.
func canvasFor(b64 string) (w, h int, ok bool) {
	if b64 == "" {
		return 0, 0, false
	}
	raw, err := decodeB64(b64)
	if err != nil {
		return 0, 0, false
	}
	// Header only: the pixels are never allocated just to measure them.
	cfg, _, err := image.DecodeConfig(bytes.NewReader(raw))
	if err != nil || cfg.Width <= 0 || cfg.Height <= 0 {
		return 0, 0, false
	}
	w, h = cfg.Width, cfg.Height
	if longest := max(w, h); longest > editCanvasMax {
		w = w * editCanvasMax / longest // aspect first, rounding second
		h = h * editCanvasMax / longest
	}
	return roundCanvas(w), roundCanvas(h), true
}

func roundCanvas(v int) int {
	return min(max((v+32)/64*64, minDimension), editCanvasMax)
}

func imageField(i, n int) string {
	if n == 1 {
		return "image"
	}
	return fmt.Sprintf("image[%d]", i)
}

// Bounded to 32 bits, the width other diffusion tooling treats a seed as.
func randomSeed() int64 { return rand.Int64N(1 << 32) }

// A preset is a starting point, not an override: an explicit request field wins.
func (s *Server) applyPreset(req *ImageRequest) error {
	// An installed model wins the name: a preset must not shadow a real model.
	if _, err := s.store.ReadManifest(req.Model); err == nil {
		return nil
	}
	// A name that is neither is not an error here: validate reports it better.
	p, err := s.store.ReadPreset(req.Model)
	if err != nil || p.Model == "" {
		return nil
	}
	if err := store.ValidModelName(p.Model); err != nil {
		return badRequest("preset %q names an unusable model %q", req.Model, p.Model)
	}
	req.Model = p.Model
	if len(req.Loras) == 0 {
		for _, n := range p.Loras {
			req.Loras = append(req.Loras, LoraRef{Name: n})
		}
	}
	if req.Steps == nil {
		req.Steps = p.Steps
	}
	if req.CFG == nil {
		req.CFG = p.CFG
	}
	if req.Sampler == "" {
		req.Sampler = p.Sampler
	}
	if req.Scheduler == "" {
		req.Scheduler = p.Scheduler
	}
	if req.NegativePrompt == "" {
		req.NegativePrompt = p.NegativePrompt
	}
	return nil
}

// Forwarded unchecked, a typo (steps: 5000, width: 0) either burnt an hour of
// GPU time or failed the job with no explanation.
func validateRanges(r *ImageRequest) error {
	for _, d := range []struct {
		name string
		v    *int
	}{{"width", r.Width}, {"height", r.Height}} {
		if d.v != nil && (*d.v < minDimension || *d.v > maxDimension) {
			return badRequest("%s must be between %d and %d, got %d", d.name, minDimension, maxDimension, *d.v)
		}
	}
	if r.Steps != nil && (*r.Steps < 1 || *r.Steps > maxSteps) {
		return badRequest("steps must be between 1 and %d, got %d", maxSteps, *r.Steps)
	}
	if err := inRange("strength", r.Strength, 0, 1); err != nil {
		return err
	}
	if err := inRange("control_strength", r.ControlStrength, 0, 1); err != nil {
		return err
	}
	if err := inRange("cfg", r.CFG, 0, maxCFG); err != nil {
		return err
	}
	if g := r.Guidance; g != nil {
		for _, f := range []struct {
			name string
			v    *float64
		}{
			{"guidance.txt_cfg", g.TxtCFG},
			{"guidance.img_cfg", g.ImgCFG},
			{"guidance.distilled_guidance", g.DistilledGuidance},
		} {
			if err := inRange(f.name, f.v, 0, maxCFG); err != nil {
				return err
			}
		}
	}
	return nil
}

func inRange(name string, v *float64, lo, hi float64) error {
	if v != nil && (*v < lo || *v > hi) {
		return badRequest("%s must be between %g and %g, got %g", name, lo, hi, *v)
	}
	return nil
}
