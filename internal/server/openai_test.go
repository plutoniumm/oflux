package server

import (
	"bytes"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestParseSize(t *testing.T) {
	w, h := parseSize("1024x768")
	if w == nil || h == nil || *w != 1024 || *h != 768 {
		t.Errorf("1024x768 -> %v,%v", w, h)
	}
	if w, h := parseSize("512X512"); w == nil || *w != 512 || h == nil || *h != 512 {
		t.Errorf("case-insensitive x failed")
	}
	for _, bad := range []string{"", "abc", "10", "0x10", "10x-1"} {
		if w, h := parseSize(bad); w != nil || h != nil {
			t.Errorf("parseSize(%q) should be nil,nil", bad)
		}
	}
}

func TestOpenAIGenerateModelNotInstalled(t *testing.T) {
	srv, _ := newTestServer(t)
	req := localReq(http.MethodPost, "/v1/images/generations", strings.NewReader(`{"model":"ghost","prompt":"x"}`))
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404; body=%s", rec.Code, rec.Body.String())
	}
}

func TestOpenAIEditMissingImage(t *testing.T) {
	srv, _ := newTestServer(t)
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	mw.WriteField("model", "m1")
	mw.WriteField("prompt", "make it red")
	mw.Close()
	req := localReq(http.MethodPost, "/v1/images/edits", &buf)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400 (missing image); body=%s", rec.Code, rec.Body.String())
	}
}
