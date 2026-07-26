package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"oflux/internal/types"
)

// postJSON drives the real handler stack, including the loopback guard.
func postJSON(t *testing.T, srv *Server, path string, body any) *httptest.ResponseRecorder {
	t.Helper()
	b, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, path, bytes.NewReader(b))
	req.Host = "127.0.0.1:11534"
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)
	return w
}

// control_image used to be forwarded to an engine with no ControlNet loaded,
// which quietly ignored it and returned an ordinary image — so the request
// looked like it had worked. It must be refused instead.
func TestControlImageWithoutControlNetIsRejected(t *testing.T) {
	srv, st := newTestServer(t)
	if err := st.WriteManifest(types.Manifest{
		Name:         "plain",
		Architecture: "qwen-image-edit",
		Mode:         types.ModeEdit,
		Components:   []types.Component{{Role: types.RoleDiffusion, Blob: "sha256-x"}},
	}); err != nil {
		t.Fatal(err)
	}

	w := postJSON(t, srv, "/v1/edit", map[string]any{
		"model":         "plain",
		"prompt":        "p",
		"image":         "SU1H",
		"control_image": "Q05H",
	})
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400 (body: %s)", w.Code, w.Body.String())
	}
	// The message has to say how to fix it, not just that it failed.
	body := w.Body.String()
	for _, want := range []string{"control net", "--control-net"} {
		if !strings.Contains(body, want) {
			t.Errorf("error %q missing %q", body, want)
		}
	}
}

// A request that sends no control_image is unaffected.
func TestNoControlImageIsNotRejected(t *testing.T) {
	srv, st := newTestServer(t)
	if err := st.WriteManifest(types.Manifest{
		Name:         "plain",
		Architecture: "qwen-image-edit",
		Mode:         types.ModeEdit,
		Components:   []types.Component{{Role: types.RoleDiffusion, Blob: "sha256-x"}},
	}); err != nil {
		t.Fatal(err)
	}
	w := postJSON(t, srv, "/v1/edit", map[string]any{
		"model": "plain", "prompt": "p", "image": "SU1H",
	})
	if w.Code == http.StatusBadRequest {
		t.Fatalf("a plain edit must not be rejected: %s", w.Body.String())
	}
}

// With a ControlNet attached, control_image passes validation and reaches the
// engine request.
func TestControlImageWithControlNetIsForwarded(t *testing.T) {
	m := types.Manifest{
		Name:         "sd15-canny",
		Architecture: "sdxl",
		Mode:         types.ModeGenerate,
		Components: []types.Component{
			{Role: types.RoleDiffusion, Blob: "sha256-x"},
			{Role: types.RoleControlNet, Blob: "sha256-cn"},
		},
	}
	if _, ok := m.Component(types.RoleControlNet); !ok {
		t.Fatal("fixture is missing its control net")
	}
	strength := 0.7
	ig := buildImgGen(m, ImageRequest{
		Prompt:          "p",
		ControlImage:    "Q05H",
		ControlStrength: &strength,
	}, types.ModeGenerate)

	if ig.ControlImage != "Q05H" {
		t.Errorf("control_image = %q", ig.ControlImage)
	}
	if ig.ControlStrength == nil || *ig.ControlStrength != 0.7 {
		t.Errorf("control_strength = %v", ig.ControlStrength)
	}
}

// A LoRA that isn't installed must fail fast with a fix, not after a multi-minute
// model load.
func TestUnknownLoraIsRejected(t *testing.T) {
	srv, st := newTestServer(t)
	if err := st.WriteManifest(types.Manifest{
		Name: "plain", Architecture: "qwen-image-edit", Mode: types.ModeEdit,
		Components: []types.Component{{Role: types.RoleDiffusion, Blob: "sha256-x"}},
	}); err != nil {
		t.Fatal(err)
	}
	w := postJSON(t, srv, "/v1/edit", map[string]any{
		"model": "plain", "prompt": "p", "image": "SU1H",
		"loras": []map[string]any{{"name": "not-installed"}},
	})
	if w.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404 (body: %s)", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "oflux lora pull") {
		t.Errorf("error should say how to install it: %s", w.Body.String())
	}
}

// A traversal-shaped LoRA name must be refused before it becomes a path.
func TestMaliciousLoraNameIsRejected(t *testing.T) {
	srv, st := newTestServer(t)
	if err := st.WriteManifest(types.Manifest{
		Name: "plain", Architecture: "qwen-image-edit", Mode: types.ModeEdit,
		Components: []types.Component{{Role: types.RoleDiffusion, Blob: "sha256-x"}},
	}); err != nil {
		t.Fatal(err)
	}
	w := postJSON(t, srv, "/v1/edit", map[string]any{
		"model": "plain", "prompt": "p", "image": "SU1H",
		"loras": []map[string]any{{"name": "../../../../etc/passwd"}},
	})
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400 (body: %s)", w.Code, w.Body.String())
	}
}
