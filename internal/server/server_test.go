package server

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"oflux/internal/hfclient"
	"oflux/internal/puller"
	"oflux/internal/store"
	"oflux/internal/supervisor"
	"oflux/internal/types"
)

func ptrF(f float64) *float64 { return &f }
func ptrI(i int) *int         { return &i }

func TestBuildImgGenEditVsGenerate(t *testing.T) {
	m := types.Manifest{Architecture: "qwen-image-edit"}
	edit := buildImgGen(m, ImageRequest{Prompt: "p", Image: ImageInput{"IMG"}}, types.ModeEdit)
	if len(edit.RefImages) == 0 || edit.RefImages[0] != "IMG" {
		t.Errorf("edit should pass the input as a ref image, got RefImages=%v", edit.RefImages)
	}
	if edit.InitImage != "" {
		t.Errorf("edit must not set InitImage (that's img2img), got %q", edit.InitImage)
	}
	gen := buildImgGen(m, ImageRequest{Prompt: "p", Image: ImageInput{"IMG"}}, types.ModeGenerate)
	if len(gen.RefImages) != 0 || gen.InitImage != "" {
		t.Errorf("generate must not carry the input image")
	}
}

func TestBuildImgGenCFGRouting(t *testing.T) {
	// Flux -> distilled_guidance; others -> txt_cfg.
	flux := buildImgGen(types.Manifest{Architecture: "flux"}, ImageRequest{CFG: ptrF(3.5)}, types.ModeGenerate)
	if flux.SampleParams == nil || flux.SampleParams.Guidance == nil || flux.SampleParams.Guidance.DistilledGuidance == nil {
		t.Fatalf("flux cfg should map to distilled_guidance: %+v", flux.SampleParams)
	}
	if flux.SampleParams.Guidance.TxtCFG != nil {
		t.Errorf("flux should not set txt_cfg")
	}
	qwen := buildImgGen(types.Manifest{Architecture: "qwen-image-edit"}, ImageRequest{CFG: ptrF(2.5)}, types.ModeEdit)
	if qwen.SampleParams == nil || qwen.SampleParams.Guidance == nil || qwen.SampleParams.Guidance.TxtCFG == nil {
		t.Fatalf("qwen cfg should map to txt_cfg: %+v", qwen.SampleParams)
	}
}

func TestBuildImgGenNoOverrides(t *testing.T) {
	// No steps/cfg/guidance -> no SampleParams (model launch defaults apply).
	ig := buildImgGen(types.Manifest{Architecture: "flux"}, ImageRequest{Prompt: "p"}, types.ModeGenerate)
	if ig.SampleParams != nil {
		t.Errorf("expected nil SampleParams, got %+v", ig.SampleParams)
	}
	ig2 := buildImgGen(types.Manifest{Architecture: "flux"}, ImageRequest{Steps: ptrI(12)}, types.ModeGenerate)
	if ig2.SampleParams == nil || ig2.SampleParams.SampleSteps == nil || *ig2.SampleParams.SampleSteps != 12 {
		t.Errorf("steps override not applied: %+v", ig2.SampleParams)
	}
}

// localReq builds a request that passes the loopback guard (httptest defaults
// the Host to example.com, which the guard correctly rejects).
func localReq(method, target string, body io.Reader) *http.Request {
	r := httptest.NewRequest(method, target, body)
	r.Host = "127.0.0.1:11534"
	return r
}

func TestGuardRejectsForeignHostAndOrigin(t *testing.T) {
	srv, _ := newTestServer(t)
	// A web page POSTing to the daemon must not be able to delete models.
	req := localReq(http.MethodPost, "/api/delete", strings.NewReader(`{"name":"m1"}`))
	req.Host = "evil.example.com"
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusForbidden {
		t.Errorf("foreign Host: status = %d, want 403", rec.Code)
	}

	req2 := localReq(http.MethodPost, "/api/delete", strings.NewReader(`{"name":"m1"}`))
	req2.Header.Set("Origin", "https://evil.example.com")
	rec2 := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec2, req2)
	if rec2.Code != http.StatusForbidden {
		t.Errorf("cross Origin: status = %d, want 403", rec2.Code)
	}
}

func TestLoopbackHost(t *testing.T) {
	for _, h := range []string{"127.0.0.1:11534", "localhost:11534", "[::1]:11534", "127.0.0.1", "localhost"} {
		if !loopbackHost(h) {
			t.Errorf("loopbackHost(%q) = false, want true", h)
		}
	}
	for _, h := range []string{"evil.example.com", "example.com:11534", "10.0.0.5:11534", ""} {
		if loopbackHost(h) {
			t.Errorf("loopbackHost(%q) = true, want false", h)
		}
	}
}

func newTestServer(t *testing.T) (*Server, *store.Store) {
	t.Helper()
	st, err := store.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	// LogDir must be set: the supervisor creates <model>.log before it execs,
	// so an empty LogDir drops engine logs into the package directory whenever
	// a test reaches a spawn.
	sup := supervisor.New(supervisor.Options{LogDir: t.TempDir()})
	pl := puller.New(hfclient.New(""), st)
	return New(st, sup, pl, types.DefaultConfig()), st
}

func TestHandleImageModelNotInstalled(t *testing.T) {
	srv, _ := newTestServer(t)
	body := `{"model":"ghost","prompt":"hi","image":"x"}`
	req := localReq(http.MethodPost, "/v1/edit", strings.NewReader(body))
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404; body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "oflux pull ghost") {
		t.Errorf("expected pull hint, got %s", rec.Body.String())
	}
}

func TestHandleTagsAndDelete(t *testing.T) {
	srv, st := newTestServer(t)
	st.WriteManifest(types.Manifest{Name: "m1", Architecture: "flux", Mode: types.ModeGenerate})

	// list
	req := localReq(http.MethodGet, "/api/tags", nil)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("tags status = %d", rec.Code)
	}
	var tags struct {
		Models []struct {
			Name   string `json:"name"`
			Loaded bool   `json:"loaded"`
		} `json:"models"`
	}
	json.Unmarshal(rec.Body.Bytes(), &tags)
	if len(tags.Models) != 1 || tags.Models[0].Name != "m1" || tags.Models[0].Loaded {
		t.Fatalf("tags = %+v", tags.Models)
	}

	// delete
	dreq := localReq(http.MethodPost, "/api/delete", strings.NewReader(`{"name":"m1"}`))
	drec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(drec, dreq)
	if drec.Code != http.StatusOK {
		t.Fatalf("delete status = %d body=%s", drec.Code, drec.Body.String())
	}
	if _, err := st.ReadManifest("m1"); err == nil {
		t.Error("manifest should be gone after delete")
	}
}

func TestHandlePS(t *testing.T) {
	srv, _ := newTestServer(t)
	req := localReq(http.MethodGet, "/api/ps", nil)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("ps status = %d", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "loaded") {
		t.Errorf("ps body = %s", rec.Body.String())
	}
}

// The first image is the subject and the rest are references, ahead of any
// ref_images — the ordering is what makes "put this in that" work.
func TestBuildImgGenMultiImageOrder(t *testing.T) {
	m := types.Manifest{Architecture: "qwen-image-edit"}
	ig := buildImgGen(m, ImageRequest{
		Image:     ImageInput{"SUBJ", "STYLE"},
		RefImages: []string{"EXTRA"},
	}, types.ModeEdit)
	if got := strings.Join(ig.RefImages, ","); got != "SUBJ,STYLE,EXTRA" {
		t.Errorf("ref_images = %q, want SUBJ,STYLE,EXTRA", got)
	}
	if ig.InitImage != "" {
		t.Error("edit must never set InitImage")
	}
}

// Manifest.Mode is frozen at pull time; /api/tags must report what archdb
// believes now, or a FLUX.2 pulled as generate-only never shows as editable.
func TestTagsReportsModeFromArchdb(t *testing.T) {
	srv, st := newTestServer(t)
	if err := st.WriteManifest(types.Manifest{
		Name: "f2", Architecture: "flux2", Mode: types.ModeGenerate,
		Base: "FLUX.2-dev", Revision: "2",
	}); err != nil {
		t.Fatal(err)
	}
	rows := tagRows(t, srv)
	if len(rows) != 1 {
		t.Fatalf("rows = %+v", rows)
	}
	if rows[0].Mode != types.ModeBoth {
		t.Errorf("mode = %q, want %q from archdb", rows[0].Mode, types.ModeBoth)
	}
	if rows[0].Base != "FLUX.2-dev" || rows[0].Revision != "2" {
		t.Errorf("base/revision = %q/%q", rows[0].Base, rows[0].Revision)
	}
}

// A preset is a callable name, so it belongs in the same listing as models —
// flagged, labelled, and carrying the target's architecture and mode.
func TestTagsListsPresets(t *testing.T) {
	srv, st := newTestServer(t)
	if err := st.WriteManifest(types.Manifest{Name: "e1", Architecture: "qwen-image-edit", Mode: types.ModeEdit}); err != nil {
		t.Fatal(err)
	}
	if err := st.WritePreset(store.Preset{Name: "portrait", Label: "Portrait fix", Model: "e1"}); err != nil {
		t.Fatal(err)
	}
	rows := tagRows(t, srv)
	var preset *ModelRow
	for i := range rows {
		if rows[i].Name == "portrait" {
			preset = &rows[i]
		}
	}
	if preset == nil {
		t.Fatal("preset missing from /api/tags")
	}
	if !preset.Preset || preset.Label != "Portrait fix" {
		t.Errorf("preset row = %+v", *preset)
	}
	if preset.Architecture != "qwen-image-edit" || !preset.Mode.CanEdit() {
		t.Errorf("a preset should carry its model's architecture and mode: %+v", *preset)
	}
}

func tagRows(t *testing.T, srv *Server) []ModelRow {
	t.Helper()
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, localReq(http.MethodGet, "/api/tags", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("tags status = %d: %s", rec.Code, rec.Body.String())
	}
	var out struct {
		Models []ModelRow `json:"models"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	return out.Models
}

// A hand-dropped adapter has no curated entry, so its sidecar is the only place
// its sampling regime is recorded — it used to list with a blank STEPS and FOR.
func TestLorasRowsCarrySidecarMetadata(t *testing.T) {
	srv, st := newTestServer(t)
	if err := os.WriteFile(filepath.Join(st.LorasDir(), "homemade"+store.LoraExt), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := st.WriteLoraMeta(store.LoraMeta{
		Name: "homemade", For: []string{"qwen-image-edit"}, Steps: 6, CFG: 1.5,
	}); err != nil {
		t.Fatal(err)
	}
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, localReq(http.MethodGet, "/api/loras", nil))
	var out struct {
		Loras []LoraRow `json:"loras"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	var row *LoraRow
	for i := range out.Loras {
		if out.Loras[i].Name == "homemade" {
			row = &out.Loras[i]
		}
	}
	if row == nil {
		t.Fatalf("homemade missing: %+v", out.Loras)
	}
	if row.Steps != 6 || row.CFG != 1.5 || len(row.For) != 1 || row.For[0] != "qwen-image-edit" {
		t.Errorf("row = %+v", *row)
	}
	// archs is the older spelling and still has to be there for old clients.
	if len(row.Archs) != len(row.For) {
		t.Errorf("archs = %v, want the same as for", row.Archs)
	}
}
