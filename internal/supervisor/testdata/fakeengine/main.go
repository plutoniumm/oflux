// Command fakeengine is a stand-in for sd-server, used only by the supervisor
// tests. It implements just enough of the native async API (see api.md) to
// spawn, probe, submit, poll and cancel, and honours --listen-ip/--listen-port
// while ignoring every other flag the supervisor passes through.
//
// The first GET of a job returns "generating", later ones "completed" with a 1x1
// PNG at result.images[0].b64_json. Prompt keywords steer that: "hold" never
// finishes, "fail" fails with a multi-line reason, and "progress" writes real
// sd.cpp-shaped sampling bars to stdout, which the supervisor captures into the
// model log it scrapes progress from.
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
	prompts   map[string]string
	counter   int
	pngB64    string
}

const sampleSteps = 4

func main() {
	host := "127.0.0.1"
	port := ""

	// Recognise only the bind flags; manifest flags pass through untouched.
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
		prompts:   map[string]string{},
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
	var body struct {
		Prompt string `json:"prompt"`
	}
	_ = json.NewDecoder(r.Body).Decode(&body)

	s.mu.Lock()
	s.counter++
	id := fmt.Sprintf("job_%d", s.counter)
	s.prompts[id] = body.Prompt
	s.mu.Unlock()

	// Lets tests tell "holds the slot" from "the engine has the job".
	fmt.Printf("SUBMITTED %s\n", id)
	if strings.Contains(body.Prompt, "progress") {
		go emitSamplingBars()
	}

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
	// On stdout so tests can prove a cancel reached the engine via the log.
	fmt.Printf("CANCELLED %s\n", id)
	writeJSON(w, http.StatusOK, map[string]any{"id": id, "status": "cancelled"})
}

func (s *fakeServer) getJob(w http.ResponseWriter, id string) {
	s.mu.Lock()
	s.polls[id]++
	n := s.polls[id]
	cancelled := s.cancelled[id]
	prompt := s.prompts[id]
	s.mu.Unlock()

	status := "generating"
	var result json.RawMessage
	var jobErr any
	done := n >= 2 && !strings.Contains(prompt, "hold")
	switch {
	case cancelled:
		status = "cancelled"
	case done && strings.Contains(prompt, "fail"):
		status = "failed"
		jobErr = "ggml_metal_graph_compute: command buffer 0 failed\nstack trace line 1\nstack trace line 2"
	case done:
		status = "completed"
		result = json.RawMessage(fmt.Sprintf(`{"output_format":"png","images":[{"index":0,"b64_json":%q}]}`, s.pngB64))
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"id":             id,
		"kind":           "img_gen",
		"status":         status,
		"queue_position": 0,
		"created":        time.Now().Unix(),
		"result":         result,
		"error":          jobErr,
	})
}

// The same shape sd.cpp emits, including the tensor-loading bar the supervisor
// has to tell apart from sampling.
func emitSamplingBars() {
	fmt.Printf("\r  |##########| 219/219 - 1.14GB/s\x1b[K\n")
	for i := 1; i <= sampleSteps; i++ {
		fmt.Printf("\r  |=====>    | %d/%d - 6.79s/it\x1b[K", i, sampleSteps)
		time.Sleep(40 * time.Millisecond)
	}
}
