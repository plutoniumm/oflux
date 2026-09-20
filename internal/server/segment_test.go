package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"oflux/internal/sam3"
	"oflux/internal/types"
)

func testImage(t *testing.T, format string, w, h int) string {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	img.Set(0, 0, color.White)
	var buf bytes.Buffer
	var err error
	if format == "jpeg" {
		err = jpeg.Encode(&buf, img, nil)
	} else {
		err = png.Encode(&buf, img)
	}
	if err != nil {
		t.Fatal(err)
	}
	return base64.StdEncoding.EncodeToString(buf.Bytes())
}

// libsam3 validates none of its inputs, so anything that is not a real image
// has to be refused before it crosses that boundary.
func TestDecodeToPNGGuardsTheCgoBoundary(t *testing.T) {
	good := testImage(t, "png", 32, 32)
	raw, cfg, err := decodeToPNG("image", good)
	if err != nil {
		t.Fatalf("a valid PNG was rejected: %v", err)
	}
	if cfg.Width != 32 || cfg.Height != 32 {
		t.Errorf("config = %dx%d, want 32x32", cfg.Width, cfg.Height)
	}
	if _, _, err := image.Decode(bytes.NewReader(raw)); err != nil {
		t.Fatalf("passed-through bytes are not decodable: %v", err)
	}
	if _, _, err := decodeToPNG("image", "data:image/png;base64,"+good); err != nil {
		t.Errorf("a data URL should be accepted: %v", err)
	}

	// A JPEG is re-encoded, because Segment takes PNG bytes.
	conv, _, err := decodeToPNG("image", testImage(t, "jpeg", 32, 32))
	if err != nil {
		t.Fatalf("jpeg: %v", err)
	}
	if _, err := png.Decode(bytes.NewReader(conv)); err != nil {
		t.Errorf("a jpeg should arrive as PNG: %v", err)
	}

	for _, bad := range []string{
		"",
		"!!!!",
		base64.StdEncoding.EncodeToString([]byte("not an image at all")),
		base64.StdEncoding.EncodeToString([]byte("\x89PNG\r\n\x1a\n truncated")),
		testImage(t, "png", 4, 4), // smaller than libsam3 accepts
	} {
		if _, _, err := decodeToPNG("image", bad); statusOf(err) != http.StatusBadRequest {
			t.Errorf("decodeToPNG(%.20q) = %v, want a 400", bad, err)
		}
	}
}

func TestSegmentPromptsAreBoundsChecked(t *testing.T) {
	cfg := image.Config{Width: 64, Height: 48}
	ok := SegmentRequest{
		Points: []SegmentPoint{{X: 0, Y: 0}, {X: 63, Y: 47, Negative: true}},
		Boxes:  []SegmentBox{{X0: 0, Y0: 0, X1: 64, Y1: 48}},
	}
	points, boxes, err := segmentPrompts(ok, cfg)
	if err != nil {
		t.Fatalf("valid prompts rejected: %v", err)
	}
	if len(points) != 2 || !points[1].Negative || len(boxes) != 1 {
		t.Errorf("points = %+v boxes = %+v", points, boxes)
	}

	for name, bad := range map[string]SegmentRequest{
		"point off the right edge": {Points: []SegmentPoint{{X: 64, Y: 0}}},
		"negative point":           {Points: []SegmentPoint{{X: -1, Y: 0}}},
		"point below the image":    {Points: []SegmentPoint{{X: 0, Y: 48}}},
		"inverted box":             {Boxes: []SegmentBox{{X0: 10, Y0: 0, X1: 5, Y1: 10}}},
		"empty box":                {Boxes: []SegmentBox{{X0: 5, Y0: 5, X1: 5, Y1: 10}}},
		"box past the edge":        {Boxes: []SegmentBox{{X0: 0, Y0: 0, X1: 65, Y1: 10}}},
		"too many points":          {Points: make([]SegmentPoint, sam3.MaxPoints+1)},
	} {
		if _, _, err := segmentPrompts(bad, cfg); statusOf(err) != http.StatusBadRequest {
			t.Errorf("%s: err = %v, want a 400", name, err)
		}
	}
}

// Every check that can be made without SAM3 has to happen before SAM3 is
// touched, so a malformed request is a 400 whatever the build can do.
func TestSegmentValidatesBeforeReachingSAM3(t *testing.T) {
	srv, _ := newTestServer(t)
	img := testImage(t, "png", 32, 32)
	for name, body := range map[string]string{
		"no image":          `{"prompt":"a cat"}`,
		"no prompt form":    `{"image":"` + img + `"}`,
		"unusable image":    `{"image":"!!!!","prompt":"a cat"}`,
		"prompt and points": `{"image":"` + img + `","prompt":"a cat","points":[{"x":1,"y":1}]}`,
		"point off image":   `{"image":"` + img + `","points":[{"x":999,"y":1}]}`,
		"endless prompt":    `{"image":"` + img + `","prompt":"` + strings.Repeat("a", sam3.MaxPromptBytes+1) + `"}`,
	} {
		rec := httptest.NewRecorder()
		srv.Handler().ServeHTTP(rec, localReq(http.MethodPost, "/v1/segment", strings.NewReader(body)))
		if rec.Code != http.StatusBadRequest {
			t.Errorf("%s: status = %d, want 400: %s", name, rec.Code, rec.Body.String())
		}
	}
}

// Without the sam3 build tag the daemon cannot segment at all; that is a 501,
// and nothing a retry or a download would fix.
func TestSegmentWithoutSAM3Is501(t *testing.T) {
	if sam3.Compiled() {
		t.Skip("this build links libsam3")
	}
	srv, _ := newTestServer(t)
	rec := httptest.NewRecorder()
	body := `{"image":"` + testImage(t, "png", 32, 32) + `","prompt":"the cat","separate":true}`
	srv.Handler().ServeHTTP(rec, localReq(http.MethodPost, "/v1/segment", strings.NewReader(body)))
	if rec.Code != http.StatusNotImplemented {
		t.Fatalf("status = %d, want 501: %s", rec.Code, rec.Body.String())
	}
	var out errorBody
	json.Unmarshal(rec.Body.Bytes(), &out)
	if out.Code != codeUnsupported {
		t.Errorf("code = %q, want %q", out.Code, codeUnsupported)
	}
}

// mask_prompt is the whole point of the feature: segment and edit in one call.
func TestMaskPromptValidation(t *testing.T) {
	srv, st := newTestServer(t)
	editModel(t, st)
	img := testImage(t, "png", 32, 32)

	// It describes a region of the image being edited, so it needs one, and
	// there is nothing to segment on the generate route.
	for name, tc := range map[string]struct {
		mode types.Mode
		req  ImageRequest
	}{
		"no image":       {types.ModeEdit, ImageRequest{Model: "e1", RefImages: []string{img}, MaskPrompt: "the cat"}},
		"generate route": {types.ModeGenerate, ImageRequest{Model: "e1", MaskPrompt: "the cat"}},
		"endless prompt": {types.ModeEdit, ImageRequest{Model: "e1", Image: ImageInput{img}, MaskPrompt: strings.Repeat("a", sam3.MaxPromptBytes+1)}},
	} {
		req := tc.req
		if _, err := srv.validate(&req, tc.mode); statusOf(err) != http.StatusBadRequest {
			t.Errorf("%s: err = %v, want a 400", name, err)
		}
	}
}

// A mask the caller painted themselves wins, and SAM3 is never consulted —
// which is also why this passes in a build that has no SAM3.
func TestExplicitMaskBeatsMaskPrompt(t *testing.T) {
	srv, st := newTestServer(t)
	editModel(t, st)
	req := ImageRequest{
		Model: "e1", Image: ImageInput{testImage(t, "png", 32, 32)},
		MaskImage: "TUFTSw==", MaskPrompt: "the cat",
	}
	if _, err := srv.validate(&req, types.ModeEdit); err != nil {
		t.Fatal(err)
	}
	if err := resolveMask(context.Background(), &req, nil); err != nil {
		t.Fatalf("resolveMask should be a no-op here: %v", err)
	}
	if req.MaskImage != "TUFTSw==" {
		t.Errorf("mask = %q, want the caller's own", req.MaskImage)
	}
}

// With no mask of its own, an edit that asks for one by prompt has to reach
// SAM3 — so in this build it fails as unavailable rather than quietly editing
// the whole image.
func TestMaskPromptReachesSAM3(t *testing.T) {
	if sam3.Compiled() {
		t.Skip("this build links libsam3")
	}
	srv, st := newTestServer(t)
	editModel(t, st)
	rec := postJSON(t, srv, "/v1/edit", map[string]any{
		"model": "e1", "prompt": "remove it",
		"image": testImage(t, "png", 32, 32), "mask_prompt": "the cat",
	})
	if rec.Code != http.StatusNotImplemented {
		t.Fatalf("status = %d, want 501: %s", rec.Code, rec.Body.String())
	}
}
