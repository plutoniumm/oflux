package server

import (
	"bufio"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"oflux/internal/store"
	"oflux/internal/types"
)

func editModel(t *testing.T, st *store.Store) {
	t.Helper()
	if err := st.WriteManifest(types.Manifest{
		Name: "e1", Architecture: "qwen-image-edit", Mode: types.ModeEdit,
		Components: []types.Component{{Role: types.RoleDiffusion, Blob: "sha256-x"}},
	}); err != nil {
		t.Fatal(err)
	}
}

// A request that fails validation has written nothing yet, so it must come back
// as an ordinary HTTP error rather than a 200 stream with an error line.
func TestStreamValidationIsAPlainHTTPError(t *testing.T) {
	srv, _ := newTestServer(t)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, localReq(http.MethodPost, "/v1/edit",
		strings.NewReader(`{"model":"ghost","prompt":"p","image":"SU1H","stream":true}`)))
	if rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404: %s", rec.Code, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
		t.Errorf("content-type = %q, want json", ct)
	}
}

// Once the job is submitted the response is committed, so a failure arrives as
// the stream's terminal line — carrying the same fields a blocking call fails
// with. There is no engine binary in the test store, so the job fails fast.
func TestStreamEmitsProgressThenATerminalLine(t *testing.T) {
	srv, st := newTestServer(t)
	editModel(t, st)

	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, localReq(http.MethodPost, "/v1/edit",
		strings.NewReader(`{"model":"e1","prompt":"p","image":"SU1H","stream":true}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", rec.Code, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "application/x-ndjson") {
		t.Errorf("content-type = %q, want ndjson", ct)
	}

	var lines []map[string]any
	sc := bufio.NewScanner(strings.NewReader(rec.Body.String()))
	for sc.Scan() {
		var m map[string]any
		if err := json.Unmarshal(sc.Bytes(), &m); err != nil {
			t.Fatalf("line %q: %v", sc.Text(), err)
		}
		lines = append(lines, m)
	}
	if len(lines) < 2 {
		t.Fatalf("want at least a progress and a terminal line, got %v", lines)
	}
	first, last := lines[0], lines[len(lines)-1]
	if first["status"] != "queued" {
		t.Errorf("first line = %v, want a queued status", first)
	}
	if first["job"] == "" || first["job"] == nil {
		t.Errorf("the first line must name the job so a dropped client can reattach: %v", first)
	}
	if _, ok := last["error"]; !ok {
		t.Errorf("terminal line = %v, want the error object a blocking call returns", last)
	}
	if _, ok := last["code"]; !ok {
		t.Errorf("a streamed failure needs the stable code too: %v", last)
	}

	// The job outlives the stream, so the same id answers on /v1/jobs.
	id, _ := first["job"].(string)
	rec2 := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec2, localReq(http.MethodGet, "/v1/jobs/"+id, nil))
	if rec2.Code == http.StatusNotFound {
		t.Errorf("job %q should still be known after the stream ends", id)
	}
}

func TestJobEndpointsRejectUnknownIDs(t *testing.T) {
	srv, _ := newTestServer(t)
	for _, method := range []string{http.MethodGet, http.MethodDelete} {
		rec := httptest.NewRecorder()
		srv.Handler().ServeHTTP(rec, localReq(method, "/v1/jobs/nope", nil))
		if rec.Code != http.StatusNotFound {
			t.Errorf("%s: status = %d, want 404", method, rec.Code)
		}
		var body errorBody
		json.Unmarshal(rec.Body.Bytes(), &body)
		if body.Code != codeJobNotFound {
			t.Errorf("%s: code = %q, want %q", method, body.Code, codeJobNotFound)
		}
	}
}

func TestJobNotesEvictOldestWhenFull(t *testing.T) {
	j := newJobNotes()
	for i := 0; i < maxJobNotes+10; i++ {
		j.remember(string(rune('a'+i%26))+strings.Repeat("x", i), "m", int64(i))
	}
	if len(j.m) > maxJobNotes {
		t.Errorf("kept %d notes, want at most %d", len(j.m), maxJobNotes)
	}
}
