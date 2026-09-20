package supervisor

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"oflux/internal/engineclient"
	"oflux/internal/types"
)

var fakeEnginePath string

func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "fakeengine-bin")
	if err != nil {
		fmt.Fprintln(os.Stderr, "mktemp:", err)
		os.Exit(1)
	}
	bin := filepath.Join(dir, "fakeengine")
	build := exec.Command("go", "build", "-o", bin, "./testdata/fakeengine")
	if out, err := build.CombinedOutput(); err != nil {
		fmt.Fprintf(os.Stderr, "build fakeengine: %v\n%s", err, out)
		os.RemoveAll(dir)
		os.Exit(1)
	}
	fakeEnginePath = bin

	code := m.Run()
	os.RemoveAll(dir)
	os.Exit(code)
}

// The fake engine ignores the flags, but buildArgs needs the component.
func testManifest(name string) types.Manifest {
	return types.Manifest{
		Name:         name,
		Architecture: "flux",
		Mode:         types.ModeGenerate,
		Components:   []types.Component{{Role: types.RoleDiffusion, Blob: "sha256-" + name}},
		Engine:       types.EngineSpec{Flags: []string{"--diffusion-model", "{diffusion}"}},
	}
}

func newTestSupervisor(t *testing.T, opts Options) *Supervisor {
	t.Helper()
	if opts.EnginePath == "" {
		opts.EnginePath = fakeEnginePath
	}
	if opts.LogDir == "" {
		opts.LogDir = t.TempDir()
	}
	if opts.BlobPath == nil {
		opts.BlobPath = func(blob string) string { return "/fake/blobs/" + blob }
	}
	s := New(opts)
	t.Cleanup(s.Shutdown)
	return s
}

func TestBuildArgs(t *testing.T) {
	m := types.Manifest{
		Name: "flux-dev",
		Components: []types.Component{
			{Role: types.RoleDiffusion, Blob: "sha256-diff"},
			{Role: types.RoleVAE, Blob: "sha256-vae"},
		},
		Engine: types.EngineSpec{
			Flags:     []string{"--diffusion-model", "{diffusion}", "--vae", "{vae}", "--diffusion-fa"},
			ModelArgs: map[string]any{"qwen_image_zero_cond_t": true},
		},
	}
	blobPath := func(blob string) string { return "/blobs/" + blob }

	got, err := buildArgs(m, blobPath, "127.0.0.1", "8080", "/store/loras")
	if err != nil {
		t.Fatalf("buildArgs: %v", err)
	}
	want := []string{
		"--diffusion-model", "/blobs/sha256-diff",
		"--vae", "/blobs/sha256-vae",
		"--diffusion-fa",
		"--model-args", "qwen_image_zero_cond_t=true",
		"--lora-model-dir", "/store/loras",
		"--listen-ip", "127.0.0.1",
		"--listen-port", "8080",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("argv mismatch\n got: %q\nwant: %q", got, want)
	}

	// With no LoRA directory configured the flag must be omitted entirely —
	// sd-server treats an empty --lora-model-dir as the current directory.
	got, err = buildArgs(m, blobPath, "127.0.0.1", "8080", "")
	if err != nil {
		t.Fatalf("buildArgs (no lora dir): %v", err)
	}
	if slices.Contains(got, "--lora-model-dir") {
		t.Fatalf("empty LoraDir should omit the flag, got: %q", got)
	}
}

// A ControlNet is a launch-time flag, so its blob path must land in argv.
func TestBuildArgsControlNet(t *testing.T) {
	m := types.Manifest{
		Name: "sd15-canny",
		Components: []types.Component{
			{Role: types.RoleDiffusion, Blob: "sha256-diff"},
			{Role: types.RoleControlNet, Blob: "sha256-cn"},
		},
		Engine: types.EngineSpec{
			Flags: []string{"--model", "{diffusion}", "--control-net", "{control_net}"},
		},
	}
	got, err := buildArgs(m, func(b string) string { return "/blobs/" + b }, "127.0.0.1", "8080", "")
	if err != nil {
		t.Fatalf("buildArgs: %v", err)
	}
	i := slices.Index(got, "--control-net")
	if i < 0 || got[i+1] != "/blobs/sha256-cn" {
		t.Fatalf("control net path not substituted: %q", got)
	}
}

func TestBuildArgsMissingComponent(t *testing.T) {
	m := types.Manifest{
		Name:   "broken",
		Engine: types.EngineSpec{Flags: []string{"--vae", "{vae}"}},
	}
	if _, err := buildArgs(m, nil, "127.0.0.1", "1", ""); err == nil {
		t.Fatal("expected error for missing vae component")
	}
}

func TestGenerateReturnsPNG(t *testing.T) {
	s := newTestSupervisor(t, Options{})

	b64, err := s.Generate(context.Background(), testManifest("modelA"), engineclient.ImgGenRequest{Prompt: "a cat"}, GenOpts{})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	img, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		t.Fatalf("Generate must return standard base64: %v", err)
	}
	if !bytes.HasPrefix(img, []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'}) {
		t.Fatalf("result is not a PNG (len=%d, prefix=%x)", len(img), img[:min(8, len(img))])
	}
	if got := s.Loaded(); len(got) != 1 || got[0] != "modelA" {
		t.Fatalf("Loaded = %v, want [modelA]", got)
	}
}

func TestLazyLoad(t *testing.T) {
	s := newTestSupervisor(t, Options{})

	if got := s.Loaded(); len(got) != 0 {
		t.Fatalf("expected nothing loaded before Generate, got %v", got)
	}
	if _, err := s.Generate(context.Background(), testManifest("lazy"), engineclient.ImgGenRequest{Prompt: "x"}, GenOpts{}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if got := s.Loaded(); len(got) != 1 || got[0] != "lazy" {
		t.Fatalf("expected [lazy] loaded after Generate, got %v", got)
	}
}

func TestMaxLoadedEviction(t *testing.T) {
	s := newTestSupervisor(t, Options{MaxLoaded: 1})

	if _, err := s.Generate(context.Background(), testManifest("A"), engineclient.ImgGenRequest{Prompt: "x"}, GenOpts{}); err != nil {
		t.Fatalf("Generate A: %v", err)
	}
	if got := s.Loaded(); len(got) != 1 || got[0] != "A" {
		t.Fatalf("after A: Loaded = %v", got)
	}
	if _, err := s.Generate(context.Background(), testManifest("B"), engineclient.ImgGenRequest{Prompt: "y"}, GenOpts{}); err != nil {
		t.Fatalf("Generate B: %v", err)
	}
	if got := s.Loaded(); len(got) != 1 || got[0] != "B" {
		t.Fatalf("after B (MaxLoaded=1): Loaded = %v, want [B] (A evicted)", got)
	}
}

func TestIdleReap(t *testing.T) {
	s := newTestSupervisor(t, Options{IdleTTL: 250 * time.Millisecond})

	if _, err := s.Generate(context.Background(), testManifest("ephemeral"), engineclient.ImgGenRequest{Prompt: "x"}, GenOpts{}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if got := s.Loaded(); len(got) != 1 {
		t.Fatalf("expected 1 loaded right after Generate, got %v", got)
	}

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if len(s.Loaded()) == 0 {
			return // reaped
		}
		time.Sleep(25 * time.Millisecond)
	}
	t.Fatalf("model was not reaped after idle TTL; still loaded: %v", s.Loaded())
}

func TestUnload(t *testing.T) {
	s := newTestSupervisor(t, Options{})

	if _, err := s.Generate(context.Background(), testManifest("u"), engineclient.ImgGenRequest{Prompt: "x"}, GenOpts{}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if err := s.Unload("u"); err != nil {
		t.Fatalf("Unload: %v", err)
	}
	if got := s.Loaded(); len(got) != 0 {
		t.Fatalf("expected nothing loaded after Unload, got %v", got)
	}
	if err := s.Unload("nope"); err == nil {
		t.Fatal("expected error unloading unknown model")
	}
}

func TestStartTimeoutBadBinary(t *testing.T) {
	s := newTestSupervisor(t, Options{EnginePath: "/bin/true", StartTimeout: 300 * time.Millisecond})

	_, err := s.Generate(context.Background(), testManifest("dead"), engineclient.ImgGenRequest{Prompt: "x"}, GenOpts{})
	if err == nil {
		t.Fatal("expected error when engine never becomes healthy")
	}
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

func statusOf(t *testing.T, s *Supervisor, id string) string {
	t.Helper()
	st, ok := s.JobState(id)
	if !ok {
		t.Fatalf("job %s vanished from the registry", id)
	}
	return st.Status
}

func engineLog(dir, model string) string {
	data, _ := os.ReadFile(filepath.Join(dir, model+".log"))
	return string(data)
}

// Later than "running", which starts as soon as the queue slot is held.
func waitSubmitted(t *testing.T, dir, model string) {
	t.Helper()
	waitFor(t, "the engine to receive a job", func() bool {
		return strings.Contains(engineLog(dir, model), "SUBMITTED ")
	})
}

// A failure must arrive as a JobError carrying the engine's own text.
func TestGenerateSurfacesEngineFailure(t *testing.T) {
	s := newTestSupervisor(t, Options{})

	_, err := s.Generate(context.Background(), testManifest("boom"), engineclient.ImgGenRequest{Prompt: "fail"}, GenOpts{})
	var je *engineclient.JobError
	if !errors.As(err, &je) {
		t.Fatalf("err = %v (%T), want *engineclient.JobError", err, err)
	}
	if je.Status != "failed" || !strings.Contains(je.Reason, "ggml_metal_graph_compute") {
		t.Fatalf("JobError = %+v", je)
	}
}

func TestKeepAliveOverridesIdleTTL(t *testing.T) {
	s := newTestSupervisor(t, Options{IdleTTL: 200 * time.Millisecond})

	req := engineclient.ImgGenRequest{Prompt: "x"}
	if _, err := s.Generate(context.Background(), testManifest("warm"), req, GenOpts{KeepAlive: time.Hour}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	time.Sleep(600 * time.Millisecond)
	if got := s.Loaded(); len(got) != 1 {
		t.Fatalf("keep_alive=1h did not survive a 200ms IdleTTL; loaded = %v", got)
	}

	// A later request with no keep_alive goes back to the configured TTL.
	if _, err := s.Generate(context.Background(), testManifest("warm"), req, GenOpts{}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	waitFor(t, "the model to be reaped once keep_alive is back to default", func() bool {
		return len(s.Loaded()) == 0
	})
}

func TestKeepAliveNegativeKeepsResident(t *testing.T) {
	s := newTestSupervisor(t, Options{IdleTTL: 100 * time.Millisecond})

	req := engineclient.ImgGenRequest{Prompt: "x"}
	if _, err := s.Generate(context.Background(), testManifest("pinned"), req, GenOpts{KeepAlive: -1}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	time.Sleep(500 * time.Millisecond)
	if got := s.Loaded(); len(got) != 1 {
		t.Fatalf("keep_alive<0 should pin the model, loaded = %v", got)
	}
	if err := s.Unload("pinned"); err != nil {
		t.Fatalf("Unload: %v", err)
	}
}

func TestSubmitJobCompletes(t *testing.T) {
	s := newTestSupervisor(t, Options{})

	id, err := s.SubmitJob(context.Background(), testManifest("bg"), engineclient.ImgGenRequest{Prompt: "x"}, GenOpts{})
	if err != nil {
		t.Fatalf("SubmitJob: %v", err)
	}
	waitFor(t, "the job to finish", func() bool { return statusOf(t, s, id) == statusDone })

	st, _ := s.JobState(id)
	if len(st.Images) != 1 {
		t.Fatalf("done job carries %d images", len(st.Images))
	}
	if _, err := base64.StdEncoding.DecodeString(st.Images[0]); err != nil {
		t.Fatalf("job image is not base64: %v", err)
	}
	if _, ok := s.JobState("job-nope"); ok {
		t.Fatal("an unknown job id must not resolve")
	}
	if s.CancelJob(id) {
		t.Fatal("cancelling a finished job must report false")
	}
}

// Cancelling must reach the engine, or a long edit keeps burning GPU.
func TestCancelJobReachesEngine(t *testing.T) {
	logDir := t.TempDir()
	s := newTestSupervisor(t, Options{LogDir: logDir})

	id, err := s.SubmitJob(context.Background(), testManifest("slow"), engineclient.ImgGenRequest{Prompt: "hold"}, GenOpts{})
	if err != nil {
		t.Fatalf("SubmitJob: %v", err)
	}
	waitSubmitted(t, logDir, "slow")

	if !s.CancelJob(id) {
		t.Fatal("CancelJob reported nothing to cancel")
	}
	if got := statusOf(t, s, id); got != statusCancelled {
		t.Fatalf("status after cancel = %q", got)
	}
	waitFor(t, "the engine to be told to cancel", func() bool {
		return strings.Contains(engineLog(logDir, "slow"), "CANCELLED ")
	})
}

// A job still in oflux's queue is dropped from it and must never be submitted.
func TestCancelQueuedJobNeverReachesEngine(t *testing.T) {
	logDir := t.TempDir()
	s := newTestSupervisor(t, Options{LogDir: logDir})
	m := testManifest("q")

	if _, err := s.SubmitJob(context.Background(), m, engineclient.ImgGenRequest{Prompt: "hold"}, GenOpts{}); err != nil {
		t.Fatalf("SubmitJob: %v", err)
	}
	waitSubmitted(t, logDir, "q")

	queued, err := s.SubmitJob(context.Background(), m, engineclient.ImgGenRequest{Prompt: "second"}, GenOpts{})
	if err != nil {
		t.Fatalf("SubmitJob: %v", err)
	}
	if got := statusOf(t, s, queued); got != statusQueued {
		t.Fatalf("second job status = %q, want %q", got, statusQueued)
	}
	if !s.CancelJob(queued) {
		t.Fatal("CancelJob reported nothing to cancel")
	}
	if got := statusOf(t, s, queued); got != statusCancelled {
		t.Fatalf("status after cancel = %q", got)
	}
	if n := strings.Count(engineLog(logDir, "q"), "CANCELLED "); n != 0 {
		t.Fatalf("a queued job must never touch the engine, saw %d cancels", n)
	}
}

func TestQueueSerializesPerModel(t *testing.T) {
	s := newTestSupervisor(t, Options{})
	m := testManifest("serial")

	first, err := s.SubmitJob(context.Background(), m, engineclient.ImgGenRequest{Prompt: "hold"}, GenOpts{})
	if err != nil {
		t.Fatalf("SubmitJob: %v", err)
	}
	waitFor(t, "the first job to start", func() bool { return statusOf(t, s, first) == statusRunning })

	second, err := s.SubmitJob(context.Background(), m, engineclient.ImgGenRequest{Prompt: "x"}, GenOpts{})
	if err != nil {
		t.Fatalf("SubmitJob: %v", err)
	}
	time.Sleep(300 * time.Millisecond)
	if got := statusOf(t, s, second); got != statusQueued {
		t.Fatalf("second job = %q while the first still holds the model", got)
	}

	s.CancelJob(first)
	waitFor(t, "the queued job to run once the slot frees", func() bool {
		return statusOf(t, s, second) == statusDone
	})
}

func TestQueueDepthRejectsWithErrBusy(t *testing.T) {
	s := newTestSupervisor(t, Options{QueueDepth: 1})
	m := testManifest("full")

	held, err := s.SubmitJob(context.Background(), m, engineclient.ImgGenRequest{Prompt: "hold"}, GenOpts{})
	if err != nil {
		t.Fatalf("SubmitJob: %v", err)
	}
	waitFor(t, "the first job to start", func() bool { return statusOf(t, s, held) == statusRunning })

	if _, err := s.SubmitJob(context.Background(), m, engineclient.ImgGenRequest{Prompt: "x"}, GenOpts{}); err != nil {
		t.Fatalf("the one queue slot should have been free: %v", err)
	}
	_, err = s.SubmitJob(context.Background(), m, engineclient.ImgGenRequest{Prompt: "x"}, GenOpts{})
	if !errors.Is(err, ErrBusy) {
		t.Fatalf("err = %v, want ErrBusy", err)
	}
	if _, err := s.Generate(context.Background(), m, engineclient.ImgGenRequest{Prompt: "x"}, GenOpts{}); !errors.Is(err, ErrBusy) {
		t.Fatalf("Generate err = %v, want ErrBusy", err)
	}
}

// A client that goes away must give its queue place back.
func TestQueuedRequestIsCancellable(t *testing.T) {
	s := newTestSupervisor(t, Options{QueueDepth: 1})
	m := testManifest("waiter")

	held, err := s.SubmitJob(context.Background(), m, engineclient.ImgGenRequest{Prompt: "hold"}, GenOpts{})
	if err != nil {
		t.Fatalf("SubmitJob: %v", err)
	}
	waitFor(t, "the first job to start", func() bool { return statusOf(t, s, held) == statusRunning })

	ctx, cancel := context.WithCancel(context.Background())
	var wg sync.WaitGroup
	wg.Add(1)
	var genErr error
	go func() {
		defer wg.Done()
		_, genErr = s.Generate(ctx, m, engineclient.ImgGenRequest{Prompt: "x"}, GenOpts{})
	}()
	time.Sleep(200 * time.Millisecond)
	cancel()
	wg.Wait()
	if !errors.Is(genErr, context.Canceled) {
		t.Fatalf("a queued request that is abandoned should return context.Canceled, got %v", genErr)
	}
	if _, err := s.SubmitJob(context.Background(), m, engineclient.ImgGenRequest{Prompt: "x"}, GenOpts{}); err != nil {
		t.Fatalf("the abandoned queue place was never released: %v", err)
	}
}

// Loading another model must not evict one that still has work against it.
func TestQueuedWorkBlocksEviction(t *testing.T) {
	logDir := t.TempDir()
	s := newTestSupervisor(t, Options{MaxLoaded: 1, LogDir: logDir})

	if _, err := s.SubmitJob(context.Background(), testManifest("busy"), engineclient.ImgGenRequest{Prompt: "hold"}, GenOpts{}); err != nil {
		t.Fatalf("SubmitJob: %v", err)
	}
	waitSubmitted(t, logDir, "busy")

	if _, err := s.Generate(context.Background(), testManifest("other"), engineclient.ImgGenRequest{Prompt: "x"}, GenOpts{}); err != nil {
		t.Fatalf("Generate on a second model: %v", err)
	}
	if !slices.Contains(s.Loaded(), "busy") {
		t.Fatalf("a busy model was evicted for another model; loaded = %v", s.Loaded())
	}
}

// The fake emits a tensor-loading bar of the same N/M shape first, so landing
// on 4/4 also proves the two kinds of bar are told apart.
func TestProgressFromEngineLog(t *testing.T) {
	s := newTestSupervisor(t, Options{})

	var mu sync.Mutex
	var steps [][2]int
	opts := GenOpts{OnStep: func(step, total int) {
		mu.Lock()
		steps = append(steps, [2]int{step, total})
		mu.Unlock()
	}}
	id, err := s.SubmitJob(context.Background(), testManifest("prog"), engineclient.ImgGenRequest{Prompt: "hold progress"}, opts)
	if err != nil {
		t.Fatalf("SubmitJob: %v", err)
	}
	waitFor(t, "sampling progress to be reported", func() bool {
		st, _ := s.JobState(id)
		return st.Total > 0
	})

	st, _ := s.JobState(id)
	if st.Step != 4 || st.Total != 4 {
		t.Fatalf("JobState progress = %d/%d, want 4/4 (219/219 would mean a tensor-loading bar was mistaken for sampling)", st.Step, st.Total)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(steps) == 0 || steps[len(steps)-1] != [2]int{4, 4} {
		t.Fatalf("OnStep saw %v", steps)
	}
}

func TestLastSamplingBar(t *testing.T) {
	cases := []struct {
		name        string
		in          string
		step, total int
		ok          bool
	}{
		{"sampling", "\r  |=====>   | 3/28 - 6.69s/it\x1b[K", 3, 28, true},
		{"newest wins", "| 1/28 - 6.7s/it| 2/28 - 6.7s/it", 2, 28, true},
		{"iterations per second", "| 9/10 - 1.50it/s", 9, 10, true},
		{"tensor load is not sampling", "  |####| 129/219 - 1.02GB/s", 0, 0, false},
		{"plain log line", "[INFO ] sampling completed, taking 195.19s", 0, 0, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			step, total, ok := lastSamplingBar([]byte(tc.in))
			if ok != tc.ok || step != tc.step || total != tc.total {
				t.Fatalf("got %d/%d ok=%v, want %d/%d ok=%v", step, total, ok, tc.step, tc.total, tc.ok)
			}
		})
	}
}

func TestGateFIFO(t *testing.T) {
	g := newGate(1, 2)
	if _, err := g.enqueue(); err != nil {
		t.Fatalf("first enqueue: %v", err)
	}
	first, err := g.enqueue()
	if err != nil {
		t.Fatalf("second enqueue: %v", err)
	}
	second, err := g.enqueue()
	if err != nil {
		t.Fatalf("third enqueue: %v", err)
	}
	if _, err := g.enqueue(); !errors.Is(err, ErrBusy) {
		t.Fatalf("err = %v, want ErrBusy at depth", err)
	}

	g.release()
	if err := g.wait(context.Background(), first); err != nil {
		t.Fatalf("the earliest waiter should have been served first: %v", err)
	}
	select {
	case <-second:
		t.Fatal("the later waiter was served out of order")
	default:
	}
	g.release()
	if err := g.wait(context.Background(), second); err != nil {
		t.Fatalf("second waiter: %v", err)
	}
}
