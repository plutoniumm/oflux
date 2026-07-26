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

// tinyPNG returns a valid 1x1 PNG and its standard base64 encoding.
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
	if job.PollURL != "/sdcpp/v1/jobs/job_1" {
		t.Fatalf("poll_url = %q", job.PollURL)
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

func TestImagesPNGNativeShape(t *testing.T) {
	raw, b64 := tinyPNG(t)
	job := Job{Result: json.RawMessage(fmt.Sprintf(`{"output_format":"png","images":[{"index":0,"b64_json":%q}]}`, b64))}
	imgs, err := job.ImagesPNG()
	if err != nil {
		t.Fatalf("ImagesPNG: %v", err)
	}
	if len(imgs) != 1 || !bytes.Equal(imgs[0], raw) {
		t.Fatalf("decoded image mismatch (len=%d)", len(imgs))
	}
}

func TestImagesPNGImagesDataShape(t *testing.T) {
	raw, b64 := tinyPNG(t)
	job := Job{Result: json.RawMessage(fmt.Sprintf(`{"images":[{"data":%q}]}`, b64))}
	imgs, err := job.ImagesPNG()
	if err != nil {
		t.Fatalf("ImagesPNG: %v", err)
	}
	if len(imgs) != 1 || !bytes.Equal(imgs[0], raw) {
		t.Fatal("decoded image mismatch for images[].data shape")
	}
}

func TestImagesPNGOpenAIShape(t *testing.T) {
	raw, b64 := tinyPNG(t)
	job := Job{Result: json.RawMessage(fmt.Sprintf(`{"data":[{"b64_json":%q}]}`, b64))}
	imgs, err := job.ImagesPNG()
	if err != nil {
		t.Fatalf("ImagesPNG: %v", err)
	}
	if len(imgs) != 1 || !bytes.Equal(imgs[0], raw) {
		t.Fatal("decoded image mismatch for data[].b64_json shape")
	}
}

func TestImagesPNGNoResult(t *testing.T) {
	if _, err := (Job{}).ImagesPNG(); err == nil {
		t.Fatal("expected error for empty result")
	}
	if _, err := (Job{Result: json.RawMessage("null")}).ImagesPNG(); err == nil {
		t.Fatal("expected error for null result")
	}
}
