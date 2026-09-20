// Package supervisor spawns and reaps sd-server subprocesses, one per model, and
// drives generation requests through them. sd-server is an opaque bundled
// binary: the supervisor only launches it with CLI flags and talks to its HTTP
// API. Models load lazily, are capped at MaxLoaded (LRU eviction), and unload
// after IdleTTL.
package supervisor

import (
	"cmp"
	"context"
	"fmt"
	"maps"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"oflux/internal/engineclient"
	"oflux/internal/types"
)

type Options struct {
	EnginePath   string                   // path to the sd-server binary
	IdleTTL      time.Duration            // unload after this much inactivity; default 15m
	MaxLoaded    int                      // max concurrently loaded models; default 1
	LogDir       string                   // per-model logs written here as <name>.log
	BlobPath     func(blob string) string // resolves a Component.Blob to an absolute file path
	Host         string                   // engine bind host; default "127.0.0.1"
	StartTimeout time.Duration            // health-probe timeout; default 120s
	// LoraDir is the engine's --lora-model-dir. Requests name adapters in it by
	// filename, so every model shares it and a new LoRA needs no reload.
	LoraDir       string
	MaxConcurrent int // generations at once per loaded model; default 1, see gate
	QueueDepth    int // requests that may wait per model; default 8, <=0 unbounded
}

// cancel and timer are set before a runner is published, so every runner in
// Supervisor.runners has them.
type runner struct {
	name     string
	cancel   context.CancelFunc
	client   *engineclient.Client
	lastUsed time.Time
	timer    *time.Timer
	logPath  string // stdout+stderr; progress.go scrapes sampling progress from it
	// ttl starts at Options.IdleTTL and GenOpts.KeepAlive can replace it, so
	// "keep this warm" is a property of the model, not of one request.
	ttl time.Duration

	dead bool          // process exited, killed or crashed; guarded by Supervisor.mu
	done chan struct{} // closed once the process has exited and been reaped
}

type Supervisor struct {
	mu      sync.Mutex
	runners map[string]*runner
	// loading holds an in-progress load per model, closed when it finishes, so
	// concurrent callers can wait without holding mu (a load takes minutes).
	loading map[string]chan struct{}
	// pending counts requests queued or running per model, loaded or not.
	// Eviction and reaping consult it, so one model's request can never pull
	// the weights out from under another model's waiting queue.
	gates   map[string]*gate
	pending map[string]int
	opts    Options

	// jobsMu guards the registry and every field of the jobs in it. It is never
	// held together with mu: the registry knows nothing about model lifecycle,
	// so nesting would only add an order to get wrong.
	jobsMu sync.Mutex
	jobs   map[string]*job
}

type GenOpts struct {
	// KeepAlive overrides IdleTTL for this model's idle timer: 0 keeps the
	// configured value, negative pins the model until something unloads it.
	// Chained edits otherwise reload ~22GB of weights between turns.
	KeepAlive time.Duration
	// OnStep may fire from another goroutine, never after the call returns.
	OnStep func(step, total int)
}

func New(opts Options) *Supervisor {
	if opts.IdleTTL <= 0 {
		opts.IdleTTL = 15 * time.Minute
	}
	if opts.MaxLoaded <= 0 {
		opts.MaxLoaded = 1
	}
	if opts.Host == "" {
		opts.Host = "127.0.0.1"
	}
	if opts.StartTimeout <= 0 {
		opts.StartTimeout = 120 * time.Second
	}
	if opts.BlobPath == nil {
		opts.BlobPath = func(blob string) string { return blob }
	}
	if opts.MaxConcurrent <= 0 {
		opts.MaxConcurrent = 1
	}
	if opts.QueueDepth == 0 {
		opts.QueueDepth = 8
	}
	if opts.LogDir == "" {
		// An invariant, not a per-path conditional. See spawn for why.
		opts.LogDir = os.TempDir()
	}
	return &Supervisor{
		runners: make(map[string]*runner),
		loading: make(map[string]chan struct{}),
		gates:   make(map[string]*gate),
		pending: make(map[string]int),
		jobs:    make(map[string]*job),
		opts:    opts,
	}
}

var placeholderRoles = map[string]types.Role{
	"{diffusion}":   types.RoleDiffusion,
	"{vae}":         types.RoleVAE,
	"{clip_l}":      types.RoleCLIPL,
	"{t5xxl}":       types.RoleT5XXL,
	"{llm}":         types.RoleLLM,
	"{mmproj}":      types.RoleMMProj,
	"{control_net}": types.RoleControlNet,
}

// buildArgs resolves {role} placeholders to blob paths, appends --model-args
// (sorted for determinism) and the LoRA dir, then the bind flags. Those are
// --listen-ip / --listen-port, verified from stable-diffusion.cpp's server
// (examples/server/README.md + main.cpp: listen_ip / listen_port).
func buildArgs(m types.Manifest, blobPath func(string) string, host, port, loraDir string) ([]string, error) {
	args := make([]string, 0, len(m.Engine.Flags)+2*len(m.Engine.ModelArgs)+4)
	for _, f := range m.Engine.Flags {
		if role, ok := placeholderRoles[f]; ok {
			comp, ok := m.Component(role)
			if !ok {
				return nil, fmt.Errorf("supervisor: manifest %q references %s but has no %s component", m.Name, f, role)
			}
			args = append(args, blobPath(comp.Blob))
			continue
		}
		args = append(args, f)
	}
	if len(m.Engine.ModelArgs) > 0 {
		// sd-server takes ONE --model-args with a comma-separated key=value
		// list, not a repeated flag, per `sd-server --help`.
		pairs := make([]string, 0, len(m.Engine.ModelArgs))
		for _, k := range slices.Sorted(maps.Keys(m.Engine.ModelArgs)) {
			pairs = append(pairs, fmt.Sprintf("%s=%v", k, m.Engine.ModelArgs[k]))
		}
		args = append(args, "--model-args", strings.Join(pairs, ","))
	}
	if loraDir != "" {
		args = append(args, "--lora-model-dir", loraDir)
	}
	args = append(args, "--listen-ip", host, "--listen-port", port)
	return args, nil
}

// Generate returns the first image as RAW base64 PNG: the engine sends base64
// and oflux sends base64 out, so a decode here is only re-encoded upstream.
func (s *Supervisor) Generate(ctx context.Context, m types.Manifest, req engineclient.ImgGenRequest, opts GenOpts) (string, error) {
	t, err := s.enqueue(m.Name)
	if err != nil {
		return "", err
	}
	imgs, err := s.run(ctx, m, req, opts, nil, t)
	if err != nil {
		return "", err
	}
	return imgs[0], nil
}

// run owns the ticket it is given; j, when non-nil, is its registry entry.
func (s *Supervisor) run(ctx context.Context, m types.Manifest, req engineclient.ImgGenRequest, opts GenOpts, j *job, t *ticket) ([]string, error) {
	defer s.release(t)
	if err := t.wait(ctx); err != nil {
		return nil, err
	}
	if j != nil {
		s.jobsMu.Lock()
		if j.status == statusQueued {
			j.status = statusRunning
		}
		s.jobsMu.Unlock()
	}

	// ensureLoaded may spawn an engine and wait minutes for it to become
	// healthy; it deliberately does not hold s.mu for that.
	r, err := s.ensureLoaded(ctx, m)
	if err != nil {
		return nil, err
	}

	s.mu.Lock()
	r.ttl = cmp.Or(opts.KeepAlive, s.opts.IdleTTL)
	s.touchLocked(r, true)
	client := r.client
	logPath := r.logPath
	name := m.Name
	s.mu.Unlock()

	// Only a *successful* request refreshes the idle clock. Refreshing on the
	// failure path would let a wedged engine keep resetting its own TTL and
	// never be reaped.
	ok := false
	defer func() {
		s.mu.Lock()
		if rr, found := s.runners[name]; found && rr == r {
			s.touchLocked(rr, ok)
		}
		s.mu.Unlock()
	}()

	if opts.OnStep != nil {
		tailCtx, stopTail := context.WithCancel(context.Background())
		tailDone := make(chan struct{})
		go func() {
			defer close(tailDone)
			tailProgress(tailCtx, logPath, opts.OnStep)
		}()
		// Joining the tailer is what makes the GenOpts.OnStep promise true.
		defer func() { stopTail(); <-tailDone }()
	}

	eng, err := client.Submit(ctx, req)
	if err != nil {
		return nil, err
	}
	if j != nil {
		s.jobsMu.Lock()
		j.engineID, j.client = eng.ID, client
		s.jobsMu.Unlock()
	}
	eng, err = client.Wait(ctx, eng.ID, pollInterval)
	if err != nil {
		return nil, err
	}
	if eng.Status != "completed" {
		return nil, &engineclient.JobError{Op: "img_gen", Status: eng.Status, Reason: eng.Error}
	}
	imgs, err := eng.ImagesB64()
	if err != nil {
		return nil, err
	}
	ok = true
	return imgs, nil
}

func (s *Supervisor) Loaded() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Sorted(maps.Keys(s.runners))
}

func (s *Supervisor) Unload(name string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	r, ok := s.runners[name]
	if !ok {
		return fmt.Errorf("supervisor: model %q not loaded", name)
	}
	s.dropLocked(r)
	return nil
}

// Shutdown waits (briefly) for the engines to actually exit: sd-server is not
// killed by its parent dying, so returning early orphans multi-GB processes.
func (s *Supervisor) Shutdown() {
	s.cancelAllJobs()

	s.mu.Lock()
	stopped := make([]*runner, 0, len(s.runners))
	for _, r := range s.runners {
		stopped = append(stopped, r)
		s.dropLocked(r)
	}
	s.mu.Unlock()

	deadline := time.After(10 * time.Second)
	for _, r := range stopped {
		select {
		case <-r.done:
		case <-deadline:
			return // give up rather than hang shutdown; startup reaps stragglers
		}
	}
}

// s.mu is NOT held while the engine spawns and health-probes: that takes minutes
// for a large checkpoint and would block every other call. Concurrent callers
// for one model wait on a shared latch rather than each spawning an engine.
func (s *Supervisor) ensureLoaded(ctx context.Context, m types.Manifest) (*runner, error) {
	for {
		s.mu.Lock()
		if r, found := s.runners[m.Name]; found {
			if !r.dead {
				s.mu.Unlock()
				return r, nil
			}
			// Crash or OOM kill: drop the corpse so the model self-heals.
			s.dropLocked(r)
		}
		if ch, loading := s.loading[m.Name]; loading {
			s.mu.Unlock()
			select {
			case <-ch: // another caller's load finished; re-check the map
				continue
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
		ch := make(chan struct{})
		s.loading[m.Name] = ch
		s.evictForLoadLocked()
		s.mu.Unlock()

		r, err := s.spawn(m)

		s.mu.Lock()
		delete(s.loading, m.Name)
		close(ch)
		if err != nil {
			s.mu.Unlock()
			return nil, err
		}
		r.timer = time.AfterFunc(r.ttl, func() { s.reap(m.Name) })
		s.runners[m.Name] = r
		s.mu.Unlock()
		return r, nil
	}
}

// Never evicts a model with queued or in-flight work, which would throw away
// minutes of GPU work; if all are busy it overshoots MaxLoaded instead.
func (s *Supervisor) evictForLoadLocked() {
	for len(s.runners) >= s.opts.MaxLoaded {
		victim := s.lruIdleLocked()
		if victim == nil {
			return // all loaded models are busy; allow a temporary overshoot
		}
		s.dropLocked(victim)
	}
}

func (s *Supervisor) spawn(m types.Manifest) (*runner, error) {
	port, err := freePort(s.opts.Host)
	if err != nil {
		return nil, fmt.Errorf("supervisor: pick port: %w", err)
	}
	args, err := buildArgs(m, s.opts.BlobPath, s.opts.Host, port, s.opts.LoraDir)
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(s.opts.LogDir, 0o755); err != nil {
		return nil, fmt.Errorf("supervisor: create log dir: %w", err)
	}
	logPath := filepath.Join(s.opts.LogDir, m.Name+".log")
	logFile, err := os.Create(logPath)
	if err != nil {
		return nil, fmt.Errorf("supervisor: create log file: %w", err)
	}

	procCtx, cancel := context.WithCancel(context.Background())
	cmd := exec.CommandContext(procCtx, s.opts.EnginePath, args...)
	// sd-server writes scratch files to its working directory. Under launchd /
	// LaunchServices the inherited cwd is "/", which the user can't write, and
	// the engine then fails every request with a 500. Run it in a writable dir.
	cmd.Dir = s.opts.LogDir
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	if err := cmd.Start(); err != nil {
		cancel()
		_ = logFile.Close()
		return nil, fmt.Errorf("supervisor: start engine: %w", err)
	}

	r := &runner{
		name:     m.Name,
		cancel:   cancel,
		client:   engineclient.New("http://" + net.JoinHostPort(s.opts.Host, port)),
		lastUsed: time.Now(),
		logPath:  logPath,
		ttl:      s.opts.IdleTTL,
		done:     make(chan struct{}),
	}

	// One goroutine owns Wait(): it reaps the child and records its death.
	go func() {
		_ = cmd.Wait()
		s.mu.Lock()
		r.dead = true
		s.mu.Unlock()
		_ = logFile.Close()
		close(r.done)
	}()

	// Bounded by StartTimeout, NOT the client's context: a client that gives up
	// must not kill an engine that is still loading.
	probeCtx, probeCancel := context.WithTimeout(context.Background(), s.opts.StartTimeout)
	defer probeCancel()
	if err := s.probe(probeCtx, r); err != nil {
		cancel()
		<-r.done
		return nil, fmt.Errorf("supervisor: engine %q did not become healthy: %w\n--- %s tail ---\n%s",
			m.Name, err, logPath, tailFile(logPath, 2048))
	}
	return r, nil
}

// Watching for process death turns an engine that exits at once (bad flags,
// missing weights) into a fast failure instead of a StartTimeout-long wait.
func (s *Supervisor) probe(ctx context.Context, r *runner) error {
	var lastErr error
	for {
		pctx, cancel := context.WithTimeout(ctx, 2*time.Second)
		lastErr = r.client.Capabilities(pctx)
		cancel()
		if lastErr == nil {
			return nil
		}
		select {
		case <-r.done:
			return fmt.Errorf("engine exited during startup: %v", lastErr)
		case <-ctx.Done():
			return fmt.Errorf("%w (last probe: %v)", ctx.Err(), lastErr)
		case <-time.After(250 * time.Millisecond):
		}
	}
}

func (s *Supervisor) reap(name string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	r, ok := s.runners[name]
	if !ok {
		return
	}
	if r.ttl < 0 || s.pending[name] > 0 || time.Since(r.lastUsed) < r.ttl {
		s.touchLocked(r, false)
		return
	}
	s.dropLocked(r)
}

func (s *Supervisor) lruIdleLocked() *runner {
	var oldest *runner
	for _, r := range s.runners {
		if s.pending[r.name] > 0 {
			continue
		}
		if oldest == nil || r.lastUsed.Before(oldest.lastUsed) {
			oldest = r
		}
	}
	return oldest
}

// dropLocked does not wait: the per-runner reaper goroutine calls Wait and
// closes the log, so s.mu is never held across a process teardown.
func (s *Supervisor) dropLocked(r *runner) {
	r.timer.Stop()
	r.cancel()
	delete(s.runners, r.name)
}

func (s *Supervisor) touchLocked(r *runner, used bool) {
	if used {
		r.lastUsed = time.Now()
	}
	if r.ttl < 0 {
		r.timer.Stop() // keep_alive < 0: resident until something unloads it
		return
	}
	r.timer.Reset(r.ttl)
}

func freePort(host string) (string, error) {
	ln, err := net.Listen("tcp", net.JoinHostPort(host, "0"))
	if err != nil {
		return "", err
	}
	defer ln.Close()
	_, port, err := net.SplitHostPort(ln.Addr().String())
	if err != nil {
		return "", err
	}
	return port, nil
}

func tailFile(path string, n int) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	if len(data) > n {
		data = data[len(data)-n:]
	}
	return string(data)
}
