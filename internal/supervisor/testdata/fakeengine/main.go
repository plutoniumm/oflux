// Command fakeengine is a stand-in for the real sd-server binary, used only by
// the supervisor package tests. It implements just enough of the sd-server
// native async API (see api.md) for the supervisor to spawn it, health-probe
// it, submit a job, poll to completion, and cancel.
//
// It accepts the same bind flags the supervisor passes to the real engine:
// --listen-ip <ip> and --listen-port <port>. All other flags are ignored, so
// the supervisor can pass through arbitrary manifest flags unchanged.
//
// Job lifecycle: the first GET of a job returns "generating"; the second and
// later GETs return "completed" with a real 1x1 PNG base64-encoded at
// result.images[0].b64_json, matching the shape confirmed from api.md.
package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

type fakeServer struct {
	mu        sync.Mutex
	polls     map[string]int
	cancelled map[string]bool
	counter   int
	pngB64    string
}

func main() {
	host := "127.0.0.1"
	port := ""

	// Manual flag scan: recognise only the bind flags, ignore everything else
	// (the supervisor passes real manifest flags through to the engine).
	args := os.Args[1:]
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--listen-ip":
			if i+1 < len(args) {
				host = args[i+1]
				i++
			}
		case "--listen-port":
			if i+1 < len(args) {
				port = args[i+1]
				i++
			}
		}
	}
	if port == "" {
		fmt.Fprintln(os.Stderr, "fakeengine: --listen-port is required")
		os.Exit(2)
	}

	srv := &fakeServer{
		polls:     map[string]int{},
		cancelled: map[string]bool{},
		pngB64:    onePixelPNGBase64(),
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/sdcpp/v1/capabilities", srv.capabilities)
	mux.HandleFunc("/sdcpp/v1/img_gen", srv.imgGen)
	mux.HandleFunc("/sdcpp/v1/jobs/", srv.jobs)

	addr := host + ":" + port
	if err := http.ListenAndServe(addr, mux); err != nil {
		fmt.Fprintln(os.Stderr, "fakeengine:", err)
		os.Exit(1)
	}
}

func onePixelPNGBase64() string {
	img := image.NewRGBA(image.Rect(0, 0, 1, 1))
	img.Set(0, 0, color.RGBA{R: 12, G: 34, B: 56, A: 255})
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		panic(err)
	}
	return base64.StdEncoding.EncodeToString(buf.Bytes())
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func (s *fakeServer) capabilities(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"modes":          []string{"img_gen"},
		"output_formats": []string{"png"},
	})
}

func (s *fakeServer) imgGen(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	s.mu.Lock()
	s.counter++
	id := fmt.Sprintf("job_%d", s.counter)
	s.mu.Unlock()

	writeJSON(w, http.StatusAccepted, map[string]any{
		"id":       id,
		"kind":     "img_gen",
		"status":   "queued",
		"created":  time.Now().Unix(),
		"poll_url": "/sdcpp/v1/jobs/" + id,
	})
}

func (s *fakeServer) jobs(w http.ResponseWriter, r *http.Request) {
	rest := strings.TrimPrefix(r.URL.Path, "/sdcpp/v1/jobs/")
	if strings.HasSuffix(rest, "/cancel") {
		s.cancel(w, strings.TrimSuffix(rest, "/cancel"))
		return
	}
	s.getJob(w, rest)
}

func (s *fakeServer) cancel(w http.ResponseWriter, id string) {
	s.mu.Lock()
	s.cancelled[id] = true
	s.mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]any{"id": id, "status": "cancelled"})
}

func (s *fakeServer) getJob(w http.ResponseWriter, id string) {
	s.mu.Lock()
	s.polls[id]++
	n := s.polls[id]
	cancelled := s.cancelled[id]
	s.mu.Unlock()

	status := "generating"
	var result json.RawMessage
	switch {
	case cancelled:
		status = "cancelled"
	case n >= 2:
		status = "completed"
		result = json.RawMessage(fmt.Sprintf(`{"output_format":"png","images":[{"index":0,"b64_json":%q}]}`, s.pngB64))
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"id":      id,
		"kind":    "img_gen",
		"status":  status,
		"created": time.Now().Unix(),
		"result":  result,
		"error":   nil,
	})
}
