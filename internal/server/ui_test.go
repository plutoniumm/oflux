package server

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestServesUIAtRoot(t *testing.T) {
	srv, _ := newTestServer(t)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, localReq(http.MethodGet, "/", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/html") {
		t.Errorf("content-type = %q", ct)
	}
	body := rec.Body.String()
	for _, want := range []string{
		"<title>oflux</title>", "/v1/edit", "/api/tags", "/v1/segment",
		// A drawn selection has to reach the engine as the mask: dropped, the
		// editor would still look like it worked and repaint the whole image.
		"body.mask_image = maskPNG()",
		// Several selected objects are a subject plus references; flattened into
		// one picture, the model could no longer tell which is which.
		"body.image = sub.refs.length ? [sub.img, ...sub.refs] : sub.img",
		// Nothing else on the page says Shift-click adds a reference.
		"Click: add as a reference",
		// The board autosaves: without it a reload silently loses the work.
		"indexedDB.open(",
	} {
		if !strings.Contains(body, want) {
			t.Errorf("page is missing %q", want)
		}
	}
	if rec.Header().Get("Content-Security-Policy") == "" {
		t.Error("expected a CSP header on the UI")
	}
}

// The UI must not become a second, unguarded surface.
func TestUIStillBehindGuard(t *testing.T) {
	srv, _ := newTestServer(t)
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.Host = "evil.example.com"
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusForbidden {
		t.Errorf("status = %d, want 403", rec.Code)
	}
}
