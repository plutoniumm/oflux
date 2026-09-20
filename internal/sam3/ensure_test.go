package sam3

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// fakeHub serves just enough of the Hub for hfclient: the recursive tree
// listing and the resolve endpoint.
type fakeHub struct {
	body      []byte
	sha       string // advertised LFS oid; a wrong one must fail the download
	downloads atomic.Int32
	trees     atomic.Int32
	delay     time.Duration
}

func newFakeHub(t *testing.T, body string) (*fakeHub, *httptest.Server) {
	t.Helper()
	sum := sha256.Sum256([]byte(body))
	hub := &fakeHub{body: []byte(body), sha: hex.EncodeToString(sum[:])}
	srv := httptest.NewServer(hub)
	t.Cleanup(srv.Close)
	return hub, srv
}

func (f *fakeHub) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch {
	case strings.Contains(r.URL.Path, "/api/models/"):
		f.trees.Add(1)
		_ = json.NewEncoder(w).Encode([]map[string]any{
			{"type": "file", "path": "README.md", "oid": "abc", "size": 12},
			{
				"type": "file", "path": ModelFile, "oid": "def", "size": len(f.body),
				"lfs": map[string]any{"oid": f.sha, "size": len(f.body), "pointerSize": 134},
			},
		})
	case strings.HasSuffix(r.URL.Path, ModelFile):
		f.downloads.Add(1)
		if f.delay > 0 {
			time.Sleep(f.delay)
		}
		_, _ = w.Write(f.body)
	default:
		http.NotFound(w, r)
	}
}

func useFakeHub(t *testing.T, hub *httptest.Server) string {
	t.Helper()
	dir := isolateModelEnv(t)
	t.Setenv("OFLUX_SAM3_BASE_URL", hub.URL)
	t.Setenv("HF_TOKEN", "")
	return dir
}

func TestEnsureModelDownloads(t *testing.T) {
	hub, srv := newFakeHub(t, "pretend this is 707MB of weights")
	dir := useFakeHub(t, srv)

	var log []string
	got, err := ensureModel(t.Context(), func(m string) { log = append(log, m) })
	if err != nil {
		t.Fatalf("ensureModel: %v", err)
	}
	want := filepath.Join(dir, ModelFile)
	if got != want {
		t.Fatalf("EnsureModel = %s, want %s", got, want)
	}
	body, err := os.ReadFile(got)
	if err != nil || string(body) != string(hub.body) {
		t.Fatalf("downloaded content = %q, %v", body, err)
	}
	if !Available() && Status() != "" {
		// Available() is false in the untagged build by construction; the
		// point here is that resolveModel now finds the file.
		if ModelPath() != want {
			t.Fatalf("ModelPath = %q, want %s", ModelPath(), want)
		}
	}
	joined := strings.Join(log, "\n")
	if !strings.Contains(joined, "↓ "+ModelFile) || !strings.Contains(joined, "installed "+ModelFile) {
		t.Fatalf("progress log = %q, want a download and an installed line", joined)
	}
	if !strings.Contains(joined, ModelRepo) {
		t.Fatalf("progress log = %q, want it to name the source repo", joined)
	}
}

func TestEnsureModelIsCachedOnSecondCall(t *testing.T) {
	hub, srv := newFakeHub(t, "weights")
	useFakeHub(t, srv)

	if _, err := ensureModel(t.Context(), nil); err != nil {
		t.Fatalf("first ensureModel: %v", err)
	}
	var log []string
	if _, err := ensureModel(t.Context(), func(m string) { log = append(log, m) }); err != nil {
		t.Fatalf("second ensureModel: %v", err)
	}
	if n := hub.downloads.Load(); n != 1 {
		t.Fatalf("%d downloads, want exactly 1", n)
	}
	if len(log) != 1 || !strings.Contains(log[0], "(cached)") {
		t.Fatalf("progress log = %q, want one cached line", log)
	}
}

func TestEnsureModelConcurrentCallersDownloadOnce(t *testing.T) {
	hub, srv := newFakeHub(t, "weights for everyone")
	hub.delay = 150 * time.Millisecond
	useFakeHub(t, srv)

	const n = 4
	var wg sync.WaitGroup
	paths := make([]string, n)
	errs := make([]error, n)
	for i := range n {
		wg.Add(1)
		go func() {
			defer wg.Done()
			paths[i], errs[i] = ensureModel(context.Background(), nil)
		}()
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Fatalf("caller %d: %v", i, err)
		}
		if paths[i] != paths[0] {
			t.Fatalf("caller %d got %s, caller 0 got %s", i, paths[i], paths[0])
		}
	}
	if got := hub.downloads.Load(); got != 1 {
		t.Fatalf("%d downloads for %d concurrent callers, want 1", got, n)
	}
}

func TestEnsureModelCorruptDownloadLeavesNothingLoadable(t *testing.T) {
	hub, srv := newFakeHub(t, "weights")
	hub.sha = strings.Repeat("0", 64) // advertise a hash the body will not match
	dir := useFakeHub(t, srv)

	if _, err := ensureModel(t.Context(), nil); err == nil {
		t.Fatal("ensureModel accepted a checkpoint whose sha256 did not match")
	}
	if _, err := os.Stat(filepath.Join(dir, ModelFile)); !os.IsNotExist(err) {
		t.Fatalf("a failed download left %s behind (%v)", ModelFile, err)
	}
	if _, err := resolveModel(); err == nil {
		t.Fatal("resolveModel found a model after a failed download")
	}
}

func TestEnsureModelIgnoresStrayPartFile(t *testing.T) {
	_, srv := newFakeHub(t, "weights")
	dir := useFakeHub(t, srv)
	// What a SIGKILLed download leaves behind.
	if err := os.WriteFile(filepath.Join(dir, ModelFile+".part"), []byte("half"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := resolveModel(); err == nil {
		t.Fatal("resolveModel accepted a .part file as a checkpoint")
	}
	if _, err := ensureModel(t.Context(), nil); err != nil {
		t.Fatalf("ensureModel could not recover from a stray .part: %v", err)
	}
}

func TestEnsureModelRefusesToDownloadOverAnExplicitPath(t *testing.T) {
	_, srv := newFakeHub(t, "weights")
	dir := useFakeHub(t, srv)
	t.Setenv("OFLUX_SAM3_MODEL", filepath.Join(dir, "nope.ggml"))

	_, err := ensureModel(t.Context(), nil)
	if err == nil || !strings.Contains(err.Error(), "refusing to download") {
		t.Fatalf("err = %v, want a refusal naming OFLUX_SAM3_MODEL", err)
	}
}

func TestEnsureModelReportsAMissingFile(t *testing.T) {
	useFakeHub(t, httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode([]map[string]any{
			{"type": "file", "path": "README.md", "oid": "abc", "size": 12},
		})
	})))
	_, err := ensureModel(t.Context(), nil)
	if err == nil || !strings.Contains(err.Error(), ModelFile) {
		t.Fatalf("err = %v, want it to name the missing %s", err, ModelFile)
	}
}

func TestHumanBytes(t *testing.T) {
	for _, tt := range []struct {
		in   int64
		want string
	}{
		{512, "512 B"},
		{2 << 20, "2 MB"},
		{706606590, "674 MB"},
		{2 << 30, "2.0 GB"},
	} {
		if got := humanBytes(tt.in); got != tt.want {
			t.Fatalf("humanBytes(%d) = %s, want %s", tt.in, got, tt.want)
		}
	}
}
