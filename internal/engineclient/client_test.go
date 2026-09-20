package engineclient

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func tinyPNG(t *testing.T) ([]byte, string) {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 1, 1))
	img.Set(0, 0, color.RGBA{R: 12, G: 34, B: 56, A: 255})
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encode png: %v", err)
	}
	return buf.Bytes(), base64.StdEncoding.EncodeToString(buf.Bytes())
}

func newTestClient(t *testing.T, srv *httptest.Server) *Client {
	t.Helper()
	c := New(srv.URL)
	c.SetHTTPClient(srv.Client())
	return c
}

func TestCapabilities(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/sdcpp/v1/capabilities" {
			t.Errorf("unexpected path %q", r.URL.Path)
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"modes":["img_gen"]}`))
	}))
	defer srv.Close()

	if err := newTestClient(t, srv).Capabilities(context.Background()); err != nil {
		t.Fatalf("Capabilities: %v", err)
	}
}

func TestCapabilitiesError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	if err := newTestClient(t, srv).Capabilities(context.Background()); err == nil {
		t.Fatal("expected error on 500, got nil")
	}
}

func TestSubmit(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/sdcpp/v1/img_gen" {
			t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
		}
		var req ImgGenRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
		}
		if req.Prompt != "a cat" {
			t.Errorf("prompt = %q", req.Prompt)
		}
		w.WriteHeader(http.StatusAccepted)
		_, _ = w.Write([]byte(`{"id":"job_1","kind":"img_gen","status":"queued","created":1,"poll_url":"/sdcpp/v1/jobs/job_1"}`))
	}))
	defer srv.Close()

	job, err := newTestClient(t, srv).Submit(context.Background(), ImgGenRequest{Prompt: "a cat"})
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}
	if job.ID != "job_1" || job.Status != "queued" {
		t.Fatalf("job = %+v", job)
	}
}

func TestSubmitQueueFull(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
	}))
	defer srv.Close()

	_, err := newTestClient(t, srv).Submit(context.Background(), ImgGenRequest{Prompt: "x"})
	if !errors.Is(err, ErrQueueFull) {
		t.Fatalf("expected ErrQueueFull, got %v", err)
	}
}

func TestPollCompleted(t *testing.T) {
	_, b64 := tinyPNG(t)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/sdcpp/v1/jobs/job_1" {
			t.Errorf("unexpected path %q", r.URL.Path)
		}
		fmt.Fprintf(w, `{"id":"job_1","kind":"img_gen","status":"completed","result":{"output_format":"png","images":[{"index":0,"b64_json":%q}]},"error":null}`, b64)
	}))
	defer srv.Close()

	job, err := newTestClient(t, srv).Poll(context.Background(), "job_1")
	if err != nil {
		t.Fatalf("Poll: %v", err)
	}
	if job.Status != "completed" {
		t.Fatalf("status = %q", job.Status)
	}
}

func TestWaitTransitions(t *testing.T) {
	_, b64 := tinyPNG(t)
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt32(&calls, 1)
		if n < 2 {
			_, _ = w.Write([]byte(`{"id":"job_1","status":"generating","error":null}`))
			return
		}
		fmt.Fprintf(w, `{"id":"job_1","status":"completed","result":{"images":[{"b64_json":%q}]},"error":null}`, b64)
	}))
	defer srv.Close()

	job, err := newTestClient(t, srv).Wait(context.Background(), "job_1", 10*time.Millisecond)
	if err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if job.Status != "completed" {
		t.Fatalf("status = %q", job.Status)
	}
	if atomic.LoadInt32(&calls) < 2 {
		t.Fatalf("expected at least 2 polls, got %d", calls)
	}
}

func TestWaitContextCancel(t *testing.T) {
	var cancelHit int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/cancel") {
			atomic.StoreInt32(&cancelHit, 1)
			w.WriteHeader(http.StatusOK)
			return
		}
		// Never terminal.
		_, _ = w.Write([]byte(`{"id":"job_1","status":"generating","error":null}`))
	}))
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(60 * time.Millisecond)
		cancel()
	}()

	_, err := newTestClient(t, srv).Wait(ctx, "job_1", 20*time.Millisecond)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
	if atomic.LoadInt32(&cancelHit) != 1 {
		t.Fatal("expected best-effort cancel to hit the engine")
	}
}

func TestCancel(t *testing.T) {
	var mu sync.Mutex
	var gotPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		gotPath = r.URL.Path
		mu.Unlock()
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	if err := newTestClient(t, srv).Cancel(context.Background(), "job_9"); err != nil {
		t.Fatalf("Cancel: %v", err)
	}
	mu.Lock()
	defer mu.Unlock()
	if gotPath != "/sdcpp/v1/jobs/job_9/cancel" {
		t.Fatalf("cancel path = %q", gotPath)
	}
}

func TestImagesB64NativeShape(t *testing.T) {
	_, b64 := tinyPNG(t)
	job := Job{Result: json.RawMessage(fmt.Sprintf(`{"output_format":"png","images":[{"index":0,"b64_json":%q}]}`, b64))}
	imgs, err := job.ImagesB64()
	if err != nil {
		t.Fatalf("ImagesB64: %v", err)
	}
	if len(imgs) != 1 || imgs[0] != b64 {
		t.Fatalf("base64 mismatch (len=%d)", len(imgs))
	}
}

func TestImagesB64ImagesDataShape(t *testing.T) {
	_, b64 := tinyPNG(t)
	job := Job{Result: json.RawMessage(fmt.Sprintf(`{"images":[{"data":%q}]}`, b64))}
	imgs, err := job.ImagesB64()
	if err != nil {
		t.Fatalf("ImagesB64: %v", err)
	}
	if len(imgs) != 1 || imgs[0] != b64 {
		t.Fatal("base64 mismatch for images[].data shape")
	}
}

func TestImagesB64OpenAIShape(t *testing.T) {
	_, b64 := tinyPNG(t)
	job := Job{Result: json.RawMessage(fmt.Sprintf(`{"data":[{"b64_json":%q}]}`, b64))}
	imgs, err := job.ImagesB64()
	if err != nil {
		t.Fatalf("ImagesB64: %v", err)
	}
	if len(imgs) != 1 || imgs[0] != b64 {
		t.Fatal("base64 mismatch for data[].b64_json shape")
	}
}

// Unpadded base64 from the engine must not reach a client that cannot read it.
func TestImagesB64RestoresPadding(t *testing.T) {
	raw, b64 := tinyPNG(t)
	unpadded := strings.TrimRight(b64, "=")
	if unpadded == b64 {
		t.Skip("fixture happens to need no padding")
	}
	job := Job{Result: json.RawMessage(fmt.Sprintf(`{"images":[{"b64_json":%q}]}`, unpadded))}
	imgs, err := job.ImagesB64()
	if err != nil {
		t.Fatalf("ImagesB64: %v", err)
	}
	if imgs[0] != b64 {
		t.Fatalf("padding not restored: %q", imgs[0])
	}
	got, err := base64.StdEncoding.DecodeString(imgs[0])
	if err != nil {
		t.Fatalf("result is not standard base64: %v", err)
	}
	if !bytes.Equal(got, raw) {
		t.Fatal("padded result decodes to the wrong bytes")
	}
}

func TestImagesB64NoResult(t *testing.T) {
	if _, err := (Job{}).ImagesB64(); err == nil {
		t.Fatal("expected error for empty result")
	}
	if _, err := (Job{Result: json.RawMessage("null")}).ImagesB64(); err == nil {
		t.Fatal("expected error for null result")
	}
	if _, err := (Job{Result: json.RawMessage(`{"images":[]}`)}).ImagesB64(); err == nil {
		t.Fatal("expected error for an empty image list")
	}
}

func TestJobError(t *testing.T) {
	err := error(&JobError{Op: "img_gen", Status: "failed", Reason: "ggml_metal_encode: error"})
	if got := err.Error(); !strings.Contains(got, "img_gen") || !strings.Contains(got, "ggml_metal_encode") {
		t.Fatalf("Error() = %q", got)
	}
	// No reason: the message must still say what happened, not trail a colon.
	if got := (&JobError{Status: "cancelled"}).Error(); got != "engine: job cancelled" {
		t.Fatalf("bare Error() = %q", got)
	}
	var je *JobError
	if !errors.As(fmt.Errorf("wrapped: %w", err), &je) || je.Reason == "" {
		t.Fatal("JobError must survive wrapping for errors.As")
	}
}

// One flaky poll must not lose a generation.
func TestWaitToleratesTransientPollFailure(t *testing.T) {
	_, b64 := tinyPNG(t)
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch atomic.AddInt32(&calls, 1) {
		case 1:
			w.WriteHeader(http.StatusInternalServerError)
		case 2:
			_, _ = w.Write([]byte(`{"id":"job_1","status":"generating","error":null}`))
		default:
			fmt.Fprintf(w, `{"id":"job_1","status":"completed","result":{"images":[{"b64_json":%q}]},"error":null}`, b64)
		}
	}))
	defer srv.Close()

	job, err := newTestClient(t, srv).Wait(context.Background(), "job_1", time.Millisecond)
	if err != nil {
		t.Fatalf("Wait should have ridden out one 500: %v", err)
	}
	if job.Status != "completed" {
		t.Fatalf("status = %q", job.Status)
	}
}

func TestWaitGivesUpAfterRepeatedPollFailures(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	if _, err := newTestClient(t, srv).Wait(context.Background(), "job_1", time.Millisecond); err == nil {
		t.Fatal("expected Wait to give up on a permanently broken engine")
	}
	if got := atomic.LoadInt32(&calls); got != maxPollFailures {
		t.Fatalf("polled %d times, want exactly %d", got, maxPollFailures)
	}
}

// Without these a wedged engine hangs a call until the OS gives up on the socket.
func TestClientIsBounded(t *testing.T) {
	c := New("http://127.0.0.1:1")
	if c.http.Timeout <= 0 {
		t.Fatal("engine client has no request timeout")
	}
	tr, ok := c.http.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("transport = %T, want *http.Transport", c.http.Transport)
	}
	if tr.ResponseHeaderTimeout <= 0 {
		t.Fatal("transport has no response-header timeout")
	}
	if tr.MaxIdleConnsPerHost < 1 {
		t.Fatal("poll loop needs connection reuse")
	}
}
