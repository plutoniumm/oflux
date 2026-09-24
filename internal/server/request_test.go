package server

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/png"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"oflux/internal/engineclient"
	"oflux/internal/store"
	"oflux/internal/supervisor"
	"oflux/internal/types"
)

// statusOf reports the HTTP status a validation error carries.
func statusOf(err error) int {
	if err == nil {
		return http.StatusOK
	}
	var se *statusError
	if errors.As(err, &se) {
		return se.code
	}
	return http.StatusInternalServerError
}

func TestIsBase64(t *testing.T) {
	good := []string{"SU1H", "aGVsbG8=", "aGVsbG9v", "aGVsbG8", "AAAA\nBBBB", "AA=="}
	for _, s := range good {
		if !isBase64(s) {
			t.Errorf("isBase64(%q) = false, want true", s)
		}
	}
	bad := []string{"!!!", "a", "AAAAA", "aGVsbG8=x", "SU1H====", "data:image/png"}
	for _, s := range bad {
		if isBase64(s) {
			t.Errorf("isBase64(%q) = true, want false", s)
		}
	}
}

// A browser reads a file with FileReader.readAsDataURL, so the built-in UI
// sends "data:image/png;base64,...". The engine takes raw base64 and cannot
// decode the prefix, so it has to come off here.
func TestNormalizeImage(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"SU1H", "SU1H"},
		{"  SU1H\n", "SU1H"},
		{"data:image/png;base64,SU1H", "SU1H"},
		{"data:image/jpeg;charset=utf-8;base64,SU1H", "SU1H"},
	} {
		got, err := normalizeImage("image", tc.in)
		if err != nil {
			t.Errorf("normalizeImage(%q) = error %v", tc.in, err)
		} else if got != tc.want {
			t.Errorf("normalizeImage(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
	for _, bad := range []string{
		"", "data:image/png,SU1H", "data:image/png;base64", "data:image/png;base64,",
		"not base64!", "DATA:image/png;BASE64,SU1H",
	} {
		if _, err := normalizeImage("image", bad); statusOf(err) != http.StatusBadRequest {
			t.Errorf("normalizeImage(%q) = %v, want a 400", bad, err)
		}
	}
}

// Everything the engine would otherwise reject as an opaque job failure — after
// a multi-minute model load — has to be refused at the edge, with the same
// status whichever API surface asked.
func TestValidate(t *testing.T) {
	srv, st := newTestServer(t)
	for _, m := range []types.Manifest{
		{Name: "e1", Architecture: "qwen-image-edit", Mode: types.ModeEdit},
		{Name: "k1", Architecture: "flux-kontext", Mode: types.ModeEdit},
		{Name: "g1", Architecture: "flux", Mode: types.ModeGenerate},
	} {
		if err := st.WriteManifest(m); err != nil {
			t.Fatal(err)
		}
	}

	// An adapter has to exist before its scale can be the thing that is wrong.
	if err := os.WriteFile(filepath.Join(st.LorasDir(), "installed"+store.LoraExt), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	img := "SU1H"
	for _, tc := range []struct {
		name string
		mode types.Mode
		req  ImageRequest
		want int
	}{
		{"no model", types.ModeGenerate, ImageRequest{}, 400},
		{"model name with a separator", types.ModeGenerate, ImageRequest{Model: "../etc/passwd"}, 400},
		{"unknown model", types.ModeEdit, ImageRequest{Model: "ghost", Image: ImageInput{img}}, 404},
		{"edit without an image", types.ModeEdit, ImageRequest{Model: "e1"}, 400},
		{"edit against a generate-only model", types.ModeEdit, ImageRequest{Model: "g1", Image: ImageInput{img}}, 400},
		{"generate against an edit-only model", types.ModeGenerate, ImageRequest{Model: "k1"}, 400},
		// e1's manifest says edit, but archdb knows qwen-image-edit is a hybrid:
		// the live table decides, or a stale manifest hides half the model.
		{"generate against a stale edit-only manifest", types.ModeGenerate, ImageRequest{Model: "e1"}, 200},
		{"image posted to generate", types.ModeGenerate, ImageRequest{Model: "g1", Image: ImageInput{img}}, 400},
		{"ref images on a generate-only model", types.ModeGenerate, ImageRequest{Model: "g1", RefImages: []string{img}}, 400},
		{"unknown sampler", types.ModeGenerate, ImageRequest{Model: "g1", Sampler: "nope"}, 400},
		{"unknown scheduler", types.ModeGenerate, ImageRequest{Model: "g1", Scheduler: "nope"}, 400},
		{"zero steps", types.ModeGenerate, ImageRequest{Model: "g1", Steps: ptrI(0)}, 400},
		{"absurd steps", types.ModeGenerate, ImageRequest{Model: "g1", Steps: ptrI(9999)}, 400},
		{"tiny width", types.ModeGenerate, ImageRequest{Model: "g1", Width: ptrI(8)}, 400},
		{"huge height", types.ModeGenerate, ImageRequest{Model: "g1", Height: ptrI(99999)}, 400},
		{"strength out of range", types.ModeGenerate, ImageRequest{Model: "g1", Strength: ptrF(4)}, 400},
		{"cfg out of range", types.ModeGenerate, ImageRequest{Model: "g1", CFG: ptrF(1e6)}, 400},
		{"unknown lora", types.ModeGenerate, ImageRequest{Model: "g1", Loras: []LoraRef{{Name: "nope"}}}, 404},
		{"lora scale out of range", types.ModeGenerate, ImageRequest{Model: "g1", Loras: []LoraRef{{Name: "installed", Scale: ptrF(99)}}}, 400},
		{"unparsable image", types.ModeEdit, ImageRequest{Model: "e1", Image: ImageInput{"!!!!"}}, 400},
		{"unparsable ref image", types.ModeEdit, ImageRequest{Model: "e1", RefImages: []string{"!!!!"}}, 400},
		{"ok", types.ModeEdit, ImageRequest{Model: "e1", Image: ImageInput{img}}, 200},
		{"ok with a data URL", types.ModeEdit, ImageRequest{Model: "e1", Image: ImageInput{"data:image/png;base64," + img}}, 200},
	} {
		req := tc.req
		_, err := srv.validate(&req, tc.mode)
		if got := statusOf(err); got != tc.want {
			t.Errorf("%s: status = %d (%v), want %d", tc.name, got, err, tc.want)
		}
	}
}

// validate canonicalizes in place, so buildImgGen never sees a data URL or a
// ComfyUI sampler alias.
func TestValidateNormalizesInPlace(t *testing.T) {
	srv, st := newTestServer(t)
	if err := st.WriteManifest(types.Manifest{Name: "e1", Architecture: "qwen-image-edit", Mode: types.ModeEdit}); err != nil {
		t.Fatal(err)
	}
	req := ImageRequest{
		Model:     "e1",
		Image:     ImageInput{"data:image/png;base64,SU1H"},
		RefImages: []string{"data:image/jpeg;base64,Q05H"},
		Sampler:   "euler_ancestral",
	}
	if _, err := srv.validate(&req, types.ModeEdit); err != nil {
		t.Fatalf("validate: %v", err)
	}
	if req.Image.First() != "SU1H" {
		t.Errorf("image = %q, want the bare payload", req.Image)
	}
	if req.RefImages[0] != "Q05H" {
		t.Errorf("ref_images[0] = %q, want the bare payload", req.RefImages[0])
	}
	if req.Sampler != "euler_a" {
		t.Errorf("sampler = %q, want the canonical euler_a", req.Sampler)
	}
}

// A name the store cannot turn into a path is a client mistake; it used to come
// back as a 500.
func TestBadNamesAreClientErrors(t *testing.T) {
	srv, _ := newTestServer(t)
	for _, tc := range []struct{ path, body string }{
		{"/api/delete", `{"name":""}`},
		{"/api/delete", `{"name":"../../etc/passwd"}`},
		{"/api/loras/delete", `{"name":""}`},
		{"/api/loras/delete", `{"name":"../../etc/passwd"}`},
		{"/api/pull", `{"name":"org/repo","as":"../evil"}`},
		{"/api/loras/pull", `{"name":"org/repo","as":"../evil"}`},
	} {
		rec := httptest.NewRecorder()
		srv.Handler().ServeHTTP(rec, localReq(http.MethodPost, tc.path, strings.NewReader(tc.body)))
		if rec.Code != http.StatusBadRequest {
			t.Errorf("POST %s %s: status = %d, want 400 (body: %s)", tc.path, tc.body, rec.Code, rec.Body.String())
		}
	}
}

func TestEmptyBodyIsBadRequest(t *testing.T) {
	srv, _ := newTestServer(t)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, localReq(http.MethodPost, "/api/delete", strings.NewReader("")))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "empty") {
		t.Errorf("error should say the body is empty: %s", rec.Body.String())
	}
}

// A bare string and an array must both land in the same field, because the
// array form is the only way to reach multi-image editing and the string form
// is what every existing client sends.
func TestImageInputAcceptsStringAndArray(t *testing.T) {
	for _, tc := range []struct {
		body string
		want []string
	}{
		{`{"image":"A"}`, []string{"A"}},
		{`{"image":["A","B"]}`, []string{"A", "B"}},
		{`{"image":null}`, nil},
		{`{"image":""}`, nil},
		{`{}`, nil},
	} {
		var req ImageRequest
		if err := json.Unmarshal([]byte(tc.body), &req); err != nil {
			t.Errorf("%s: %v", tc.body, err)
			continue
		}
		if !slices.Equal([]string(req.Image), tc.want) {
			t.Errorf("%s: image = %v, want %v", tc.body, req.Image, tc.want)
		}
	}
	for _, bad := range []string{`{"image":7}`, `{"image":{"a":1}}`, `{"image":[1,2]}`} {
		var req ImageRequest
		if err := json.Unmarshal([]byte(bad), &req); err == nil {
			t.Errorf("%s: want an error, got image=%v", bad, req.Image)
		}
	}
}

func TestKeepAliveAcceptsDurationOrSeconds(t *testing.T) {
	for _, tc := range []struct {
		body string
		want time.Duration
	}{
		{`{"keep_alive":"10m"}`, 10 * time.Minute},
		{`{"keep_alive":90}`, 90 * time.Second},
		{`{"keep_alive":1.5}`, 1500 * time.Millisecond},
		{`{"keep_alive":-1}`, -time.Second},
		{`{"keep_alive":""}`, 0},
		{`{}`, 0},
	} {
		var req ImageRequest
		if err := json.Unmarshal([]byte(tc.body), &req); err != nil {
			t.Errorf("%s: %v", tc.body, err)
			continue
		}
		if got := time.Duration(req.KeepAlive); got != tc.want {
			t.Errorf("%s: keep_alive = %v, want %v", tc.body, got, tc.want)
		}
	}
	if err := json.Unmarshal([]byte(`{"keep_alive":"soon"}`), &ImageRequest{}); err == nil {
		t.Error(`keep_alive "soon" should be rejected`)
	}
}

// mask is the spelling every other image API uses; it must reach the engine
// field mask_image.
func TestMaskAliasesMaskImage(t *testing.T) {
	srv, st := newTestServer(t)
	if err := st.WriteManifest(types.Manifest{Name: "e1", Architecture: "qwen-image-edit", Mode: types.ModeEdit}); err != nil {
		t.Fatal(err)
	}
	req := ImageRequest{Model: "e1", Image: ImageInput{"SU1H"}, Mask: "data:image/png;base64,Q05H"}
	m, err := srv.validate(&req, types.ModeEdit)
	if err != nil {
		t.Fatalf("validate: %v", err)
	}
	if req.MaskImage != "Q05H" {
		t.Fatalf("mask_image = %q, want the normalized mask payload", req.MaskImage)
	}
	if ig := buildImgGen(m, req, types.ModeEdit); ig.MaskImage != "Q05H" {
		t.Errorf("engine mask_image = %q", ig.MaskImage)
	}
	// An explicit mask_image wins over the alias.
	req2 := ImageRequest{Model: "e1", Image: ImageInput{"SU1H"}, Mask: "Q05H", MaskImage: "TUFTSw=="}
	if _, err := srv.validate(&req2, types.ModeEdit); err != nil {
		t.Fatal(err)
	}
	if req2.MaskImage != "TUFTSw==" {
		t.Errorf("mask_image = %q, want the explicit field", req2.MaskImage)
	}
}

// The engine picks a seed when none is sent and never reports it, so the edge
// resolves one; an explicit seed must survive untouched.
func TestSeedIsResolvedAtTheEdge(t *testing.T) {
	srv, st := newTestServer(t)
	if err := st.WriteManifest(types.Manifest{Name: "g1", Architecture: "flux", Mode: types.ModeGenerate}); err != nil {
		t.Fatal(err)
	}
	req := ImageRequest{Model: "g1", Prompt: "p"}
	if _, err := srv.validate(&req, types.ModeGenerate); err != nil {
		t.Fatal(err)
	}
	if req.Seed == nil || *req.Seed < 0 {
		t.Fatalf("seed = %v, want a resolved non-negative value", req.Seed)
	}
	// A negative seed is the engine's "surprise me" and is resolved the same way.
	neg := int64(-1)
	req2 := ImageRequest{Model: "g1", Prompt: "p", Seed: &neg}
	if _, err := srv.validate(&req2, types.ModeGenerate); err != nil {
		t.Fatal(err)
	}
	if *req2.Seed < 0 {
		t.Errorf("seed = %d, want it resolved", *req2.Seed)
	}
	want := int64(42)
	req3 := ImageRequest{Model: "g1", Prompt: "p", Seed: &want}
	if _, err := srv.validate(&req3, types.ModeGenerate); err != nil {
		t.Fatal(err)
	}
	if *req3.Seed != want {
		t.Errorf("seed = %d, want the caller's %d", *req3.Seed, want)
	}
}

// A preset is callable as a model and fills in what the request left out; an
// explicit field always wins.
func TestPresetResolvesModelAndDefaults(t *testing.T) {
	srv, st := newTestServer(t)
	if err := st.WriteManifest(types.Manifest{Name: "g1", Architecture: "flux", Mode: types.ModeGenerate}); err != nil {
		t.Fatal(err)
	}
	steps, cfg := 4, 1.0
	if err := st.WritePreset(store.Preset{
		Name: "fast", Label: "Fast", Model: "g1", Steps: &steps, CFG: &cfg,
		Sampler: "euler_a", Scheduler: "beta", NegativePrompt: "blurry",
	}); err != nil {
		t.Fatal(err)
	}

	req := ImageRequest{Model: "fast", Prompt: "p"}
	m, err := srv.validate(&req, types.ModeGenerate)
	if err != nil {
		t.Fatalf("validate: %v", err)
	}
	if m.Name != "g1" || req.Model != "g1" {
		t.Fatalf("model = %q/%q, want g1", m.Name, req.Model)
	}
	if req.Steps == nil || *req.Steps != 4 || req.Sampler != "euler_a" || req.NegativePrompt != "blurry" {
		t.Errorf("preset defaults not applied: %+v", req)
	}

	override := 20
	req2 := ImageRequest{Model: "fast", Prompt: "p", Steps: &override, NegativePrompt: "mine"}
	if _, err := srv.validate(&req2, types.ModeGenerate); err != nil {
		t.Fatal(err)
	}
	if *req2.Steps != 20 || req2.NegativePrompt != "mine" {
		t.Errorf("explicit fields must beat the preset: %+v", req2)
	}
}

// A preset must not be able to shadow an installed model of the same name.
func TestInstalledModelBeatsPresetOfTheSameName(t *testing.T) {
	srv, st := newTestServer(t)
	for _, m := range []types.Manifest{
		{Name: "g1", Architecture: "flux", Mode: types.ModeGenerate},
		{Name: "g2", Architecture: "flux", Mode: types.ModeGenerate},
	} {
		if err := st.WriteManifest(m); err != nil {
			t.Fatal(err)
		}
	}
	if err := st.WritePreset(store.Preset{Name: "g1", Model: "g2"}); err != nil {
		t.Fatal(err)
	}
	req := ImageRequest{Model: "g1", Prompt: "p"}
	m, err := srv.validate(&req, types.ModeGenerate)
	if err != nil {
		t.Fatal(err)
	}
	if m.Name != "g1" {
		t.Errorf("model = %q, want the installed g1", m.Name)
	}
}

// Every failure carries a stable code, and an engine failure's hundreds of log
// lines are truncated into detail rather than dumped into the message.
func TestApiErrorShape(t *testing.T) {
	code, body := apiError(notFound("nope"))
	if code != http.StatusNotFound || body.Code != codeNotFound {
		t.Errorf("notFound -> %d/%s", code, body.Code)
	}
	code, body = apiError(withModel(coded(codeModelNotFound, notFound("model %q not installed", "ghost")), "ghost"))
	if code != http.StatusNotFound || body.Code != codeModelNotFound || body.Model != "ghost" {
		t.Errorf("model not found -> %d/%s/%s", code, body.Code, body.Model)
	}

	reason := strings.Repeat("loading tensor blah blah blah\n", 400) + "ggml_new_object: not enough space"
	je := &engineclient.JobError{Op: "img_gen", Status: "failed", Reason: reason}
	code, body = apiError(fmt.Errorf("generation failed: %w", je))
	if code != http.StatusBadGateway || body.Code != codeEngineFailed {
		t.Fatalf("engine failure -> %d/%s", code, body.Code)
	}
	if len(body.Error) > 300 {
		t.Errorf("the human message must stay short, got %d bytes", len(body.Error))
	}
	if !strings.Contains(body.Error, "not enough space") {
		t.Errorf("the message should name the failure: %q", body.Error)
	}
	if len(body.Detail) > maxDetail+80 {
		t.Errorf("detail = %d bytes, want it capped near %d", len(body.Detail), maxDetail)
	}
	if !strings.HasSuffix(body.Detail, "not enough space") {
		t.Errorf("detail must keep the TAIL of the log, got %q", body.Detail[max(0, len(body.Detail)-60):])
	}

	code, body = apiError(generationErr(supervisor.ErrBusy))
	if code != http.StatusTooManyRequests || body.Code != codeQueueFull {
		t.Errorf("a full queue -> %d/%s, want 429/queue_full", code, body.Code)
	}
}

func TestErrorResponseCarriesCodeAndRetryAfter(t *testing.T) {
	srv, _ := newTestServer(t)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, localReq(http.MethodPost, "/v1/generate", strings.NewReader(`{"model":"ghost","prompt":"x"}`)))
	var body errorBody
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if rec.Code != http.StatusNotFound || body.Code != codeModelNotFound || body.Model != "ghost" {
		t.Errorf("body = %+v (status %d)", body, rec.Code)
	}
	if body.Error == "" {
		t.Error("the short human message must survive for existing clients")
	}

	rec2 := httptest.NewRecorder()
	fail(rec2, generationErr(supervisor.ErrBusy))
	if rec2.Code != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want 429", rec2.Code)
	}
	if rec2.Header().Get("Retry-After") == "" {
		t.Error("a 429 should tell the client when to come back")
	}
}

// The OpenAI-compatible endpoints must fail in OpenAI's shape: real clients
// parse {"error":{"message","type"}} and show nothing useful otherwise.
func TestOpenAIErrorShape(t *testing.T) {
	srv, _ := newTestServer(t)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, localReq(http.MethodPost, "/v1/images/generations",
		strings.NewReader(`{"model":"ghost","prompt":"x"}`)))
	var body struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body.Error.Message == "" || body.Error.Type != "invalid_request_error" || body.Error.Code != codeModelNotFound {
		t.Errorf("error = %+v", body.Error)
	}
}

func TestTooManyImagesIsRejected(t *testing.T) {
	srv, st := newTestServer(t)
	if err := st.WriteManifest(types.Manifest{Name: "e1", Architecture: "qwen-image-edit", Mode: types.ModeEdit}); err != nil {
		t.Fatal(err)
	}
	imgs := make(ImageInput, maxImages+1)
	for i := range imgs {
		imgs[i] = "SU1H"
	}
	req := ImageRequest{Model: "e1", Image: imgs}
	if _, err := srv.validate(&req, types.ModeEdit); statusOf(err) != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", statusOf(err))
	}
}

// The preset endpoints are what `oflux preset add/ls/rm` drives.
func TestPresetEndpoints(t *testing.T) {
	srv, st := newTestServer(t)
	if err := st.WriteManifest(types.Manifest{Name: "g1", Architecture: "flux", Mode: types.ModeGenerate}); err != nil {
		t.Fatal(err)
	}

	rec := postJSON(t, srv, "/api/presets/create", map[string]any{
		"name": "fast", "label": "Fast", "model": "g1", "steps": 4, "sampler": "euler_ancestral",
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("create status = %d: %s", rec.Code, rec.Body.String())
	}
	saved, err := st.ReadPreset("fast")
	if err != nil {
		t.Fatal(err)
	}
	// A sampler alias is canonicalized at creation, not at first use.
	if saved.Sampler != "euler_a" {
		t.Errorf("sampler = %q, want euler_a", saved.Sampler)
	}

	lrec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(lrec, localReq(http.MethodGet, "/api/presets", nil))
	var list struct {
		Presets []store.Preset `json:"presets"`
	}
	if err := json.Unmarshal(lrec.Body.Bytes(), &list); err != nil {
		t.Fatal(err)
	}
	if len(list.Presets) != 1 || list.Presets[0].Name != "fast" {
		t.Errorf("presets = %+v", list.Presets)
	}

	if rec := postJSON(t, srv, "/api/presets/delete", map[string]any{"name": "fast"}); rec.Code != http.StatusOK {
		t.Fatalf("delete status = %d: %s", rec.Code, rec.Body.String())
	}
	rec = postJSON(t, srv, "/api/presets/delete", map[string]any{"name": "fast"})
	if rec.Code != http.StatusNotFound {
		t.Errorf("second delete = %d, want 404", rec.Code)
	}
	var body errorBody
	json.Unmarshal(rec.Body.Bytes(), &body)
	if body.Code != codePresetNotFound {
		t.Errorf("code = %q, want %q", body.Code, codePresetNotFound)
	}

	for _, bad := range []map[string]any{
		{"name": "", "model": "g1"},
		{"name": "../evil", "model": "g1"},
		{"name": "ok", "model": ""},
		{"name": "ok", "model": "g1", "sampler": "nope"},
	} {
		if rec := postJSON(t, srv, "/api/presets/create", bad); rec.Code != http.StatusBadRequest {
			t.Errorf("create %v = %d, want 400", bad, rec.Code)
		}
	}
}

// Transparency has no engine flag: 2.1 reads it out of the prompt, so the edge
// supplies the wording upstream prescribes. A model without RGBA weights would
// otherwise return an ordinary opaque image and look like it had worked.
func TestTransparentWrapsPromptAndRejectsModelsWithoutAlpha(t *testing.T) {
	srv, st := newTestServer(t)
	for _, m := range []types.Manifest{
		{Name: "q21", Architecture: "qwen-image-2.1", Mode: types.ModeBoth},
		{Name: "f1", Architecture: "flux", Mode: types.ModeGenerate},
	} {
		if err := st.WriteManifest(m); err != nil {
			t.Fatal(err)
		}
	}

	req := ImageRequest{Model: "q21", Prompt: "a dragon sticker", Transparent: true}
	m, err := srv.validate(&req, types.ModeGenerate)
	if err != nil {
		t.Fatalf("validate: %v", err)
	}
	got := buildImgGen(m, req, types.ModeGenerate).Prompt
	want := "This is an RGBA image with transparency. a dragon sticker. The image has alpha channel and the background is transparent."
	if got != want {
		t.Errorf("prompt = %q, want %q", got, want)
	}
	// An author who already punctuated must not get two full stops.
	req.Prompt = "a dragon sticker."
	if got := buildImgGen(m, req, types.ModeGenerate).Prompt; got != want {
		t.Errorf("prompt = %q, want %q", got, want)
	}
	// Off by default, the prompt is untouched.
	plain := ImageRequest{Model: "q21", Prompt: "a dragon sticker"}
	if got := buildImgGen(m, plain, types.ModeGenerate).Prompt; got != "a dragon sticker" {
		t.Errorf("prompt = %q, want it verbatim", got)
	}

	bad := ImageRequest{Model: "f1", Prompt: "p", Transparent: true}
	if _, err := srv.validate(&bad, types.ModeGenerate); err == nil {
		t.Fatal("transparent on a model without alpha should be a 400")
	}
}

// The engine's own canvas default is 512x512 and it never looks at the
// reference image, so an edit that named no size used to come back as a
// downscaled thumbnail of whatever went in — through the web UI, always.
func TestEditCanvasFollowsTheInputImage(t *testing.T) {
	srv, st := newTestServer(t)
	if err := st.WriteManifest(types.Manifest{Name: "e1", Architecture: "qwen-image-edit", Mode: types.ModeEdit}); err != nil {
		t.Fatal(err)
	}
	pngOf := func(w, h int) string {
		var buf bytes.Buffer
		if err := png.Encode(&buf, image.NewRGBA(image.Rect(0, 0, w, h))); err != nil {
			t.Fatal(err)
		}
		return base64.StdEncoding.EncodeToString(buf.Bytes())
	}

	for _, tc := range []struct{ inW, inH, wantW, wantH int }{
		{1024, 1024, 1024, 1024},
		{768, 512, 768, 512},
		{1000, 600, 1024, 576},   // snapped to the nearest multiple of 64
		{6000, 3000, 2048, 1024}, // clamped to editCanvasMax, aspect kept
	} {
		req := ImageRequest{Model: "e1", Prompt: "p", Image: ImageInput{pngOf(tc.inW, tc.inH)}}
		if _, err := srv.validate(&req, types.ModeEdit); err != nil {
			t.Fatalf("%dx%d: validate: %v", tc.inW, tc.inH, err)
		}
		if req.Width == nil || req.Height == nil {
			t.Fatalf("%dx%d: canvas not inferred", tc.inW, tc.inH)
		}
		if *req.Width != tc.wantW || *req.Height != tc.wantH {
			t.Errorf("%dx%d -> %dx%d, want %dx%d", tc.inW, tc.inH, *req.Width, *req.Height, tc.wantW, tc.wantH)
		}
	}

	// An explicit size is the caller's business and must survive untouched.
	w, h := 640, 480
	req := ImageRequest{Model: "e1", Prompt: "p", Image: ImageInput{pngOf(2048, 2048)}, Width: &w, Height: &h}
	if _, err := srv.validate(&req, types.ModeEdit); err != nil {
		t.Fatal(err)
	}
	if *req.Width != 640 || *req.Height != 480 {
		t.Errorf("explicit size overwritten: %dx%d", *req.Width, *req.Height)
	}

	// Generate has no subject to measure: the model's own default canvas stands.
	if err := st.WriteManifest(types.Manifest{Name: "g1", Architecture: "flux", Mode: types.ModeGenerate}); err != nil {
		t.Fatal(err)
	}
	gen := ImageRequest{Model: "g1", Prompt: "p"}
	if _, err := srv.validate(&gen, types.ModeGenerate); err != nil {
		t.Fatal(err)
	}
	if gen.Width != nil || gen.Height != nil {
		t.Errorf("generate should not invent a canvas: %v x %v", gen.Width, gen.Height)
	}
}
