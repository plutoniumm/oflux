// Package supervisor spawns and reaps sd-server subprocesses, one per model, and
// drives generation requests through them via the engineclient HTTP client.
//
// sd-server is treated as an opaque bundled binary: the supervisor only ever
// launches it with CLI flags and talks to it over its native HTTP API. Models
// are loaded lazily on first use, capped at MaxLoaded (least-recently-used
// eviction), and unloaded after IdleTTL of inactivity.
package supervisor

import (
	"context"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"oflux/internal/engineclient"
	"oflux/internal/types"
)

// Options configures a Supervisor.
type Options struct {
	EnginePath   string                   // path to the sd-server binary
	IdleTTL      time.Duration            // unload after this much inactivity; default 15m
	MaxLoaded    int                      // max concurrently loaded models; default 1
	LogDir       string                   // per-model logs written here as <name>.log
	BlobPath     func(blob string) string // resolves a Component.Blob to an absolute file path
	Host         string                   // engine bind host; default "127.0.0.1"
	StartTimeout time.Duration            // health-probe timeout; default 120s
	// LoraDir is handed to the engine as --lora-model-dir. Requests reference
	// adapters in it by filename, so every model gets the same directory and
	// LoRAs need no reload to become available.
	LoraDir string
}

// runner is one live sd-server subprocess.
type runner struct {
	name     string
	cmd      *exec.Cmd
	cancel   context.CancelFunc
	client   *engineclient.Client
	logFile  *os.File
	logPath  string
	lastUsed time.Time
	inFlight int
	timer    *time.Timer

	// dead is set by the reaper goroutine once the process has exited (whether
	// we killed it or it crashed on its own). Guarded by Supervisor.mu.
	dead bool
	// done is closed after the process has exited and been reaped.
	done chan struct{}
}

// Supervisor manages a set of sd-server subprocesses.
type Supervisor struct {
	mu      sync.Mutex
	runners map[string]*runner
	// loading holds an in-progress load per model name; the channel is closed
	// when that load finishes. It lets concurrent callers wait for a load
	// without holding mu (a load can take minutes).
	loading map[string]chan struct{}
	opts    Options
}

// New returns a Supervisor with defaults applied for any zero-valued option.
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
	return &Supervisor{
		runners: make(map[string]*runner),
		loading: make(map[string]chan struct{}),
		opts:    opts,
	}
}

// placeholderRoles maps a "{role}" flag token to the role whose component path
// should be substituted in its place.
var placeholderRoles = map[string]types.Role{
	"{diffusion}":   types.RoleDiffusion,
	"{vae}":         types.RoleVAE,
	"{clip_l}":      types.RoleCLIPL,
	"{t5xxl}":       types.RoleT5XXL,
	"{llm}":         types.RoleLLM,
	"{mmproj}":      types.RoleMMProj,
	"{control_net}": types.RoleControlNet,
}

// buildArgs turns a manifest into the argv for sd-server: the engine flags with
// {role} placeholders resolved to on-disk blob paths, followed by --model-args
// entries (sorted for determinism), the LoRA directory, then the bind flags.
// The bind flags are --listen-ip / --listen-port, as verified from
// stable-diffusion.cpp's server (examples/server/README.md + main.cpp:
// listen_ip / listen_port).
func buildArgs(m types.Manifest, blobPath func(string) string, host, port, loraDir string) ([]string, error) {
	if blobPath == nil {
		blobPath = func(blob string) string { return blob }
	}
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
		keys := make([]string, 0, len(m.Engine.ModelArgs))
		for k := range m.Engine.ModelArgs {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		// sd-server takes a single --model-args with a comma-separated key=value
		// list (not a repeated flag), per `sd-server --help`.
		pairs := make([]string, len(keys))
		for i, k := range keys {
			pairs[i] = fmt.Sprintf("%s=%v", k, m.Engine.ModelArgs[k])
		}
		args = append(args, "--model-args", strings.Join(pairs, ","))
	}
	if loraDir != "" {
		args = append(args, "--lora-model-dir", loraDir)
	}
	args = append(args, "--listen-ip", host, "--listen-port", port)
	return args, nil
}

// Generate ensures m is loaded, submits req, waits for completion and returns
// the first result image's bytes. It resets the model's idle timer. If ctx is
// cancelled while waiting, the underlying job is cancelled best-effort.
func (s *Supervisor) Generate(ctx context.Context, m types.Manifest, req engineclient.ImgGenRequest) ([]byte, error) {
	// ensureLoaded may spawn an engine and wait minutes for it to become
	// healthy; it deliberately does not hold s.mu for that.
	r, err := s.ensureLoaded(ctx, m)
	if err != nil {
		return nil, err
	}

	s.mu.Lock()
	r.inFlight++
	r.lastUsed = time.Now()
	if r.timer != nil {
		r.timer.Reset(s.opts.IdleTTL)
	}
	client := r.client
	name := m.Name
	s.mu.Unlock()

	// Only a *successful* request refreshes the idle clock. Refreshing on the
	// failure path would let a wedged engine keep resetting its own TTL and
	// never be reaped.
	ok := false
	defer func() {
		s.mu.Lock()
		if rr, found := s.runners[name]; found && rr == r {
			if rr.inFlight > 0 {
				rr.inFlight--
			}
			if ok {
				rr.lastUsed = time.Now()
			}
			if rr.timer != nil {
				rr.timer.Reset(s.opts.IdleTTL)
			}
		}
		s.mu.Unlock()
	}()

	job, err := client.Submit(ctx, req)
	if err != nil {
		return nil, err
	}
	job, err = client.Wait(ctx, job.ID, 250*time.Millisecond)
	if err != nil {
		return nil, err
	}
	switch job.Status {
	case "completed":
		imgs, err := job.ImagesPNG()
		if err != nil {
			return nil, err
		}
		if len(imgs) == 0 {
			return nil, fmt.Errorf("supervisor: engine returned no images")
		}
		ok = true
		return imgs[0], nil
	case "failed":
		return nil, fmt.Errorf("supervisor: engine job failed: %s", job.Error)
	default:
		return nil, fmt.Errorf("supervisor: engine job ended with status %q", job.Status)
	}
}

// Loaded returns the names of currently running models, sorted.
func (s *Supervisor) Loaded() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	names := make([]string, 0, len(s.runners))
	for n := range s.runners {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// Unload stops the runner for name immediately.
func (s *Supervisor) Unload(name string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	r, ok := s.runners[name]
	if !ok {
		return fmt.Errorf("supervisor: model %q not loaded", name)
	}
	s.stopRunnerLocked(r)
	delete(s.runners, name)
	return nil
}

// Shutdown stops all runners.
// Shutdown stops all runners and waits (briefly) for the engine processes to
// actually exit. Waiting matters: sd-server is not killed by its parent dying,
// so returning early would leave multi-GB engines orphaned.
func (s *Supervisor) Shutdown() {
	s.mu.Lock()
	stopped := make([]*runner, 0, len(s.runners))
	for name, r := range s.runners {
		s.stopRunnerLocked(r)
		stopped = append(stopped, r)
		delete(s.runners, name)
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

// ensureLoaded returns a live runner for m, spawning sd-server if needed.
//
// s.mu is NOT held while the engine spawns and health-probes — that can take
// minutes for a large checkpoint, and holding the lock would block Loaded(),
// Unload(), Shutdown() and every other request for the duration. Concurrent
// callers for the same model wait on a shared latch instead of each spawning
// their own engine.
func (s *Supervisor) ensureLoaded(ctx context.Context, m types.Manifest) (*runner, error) {
	for {
		s.mu.Lock()
		if r, found := s.runners[m.Name]; found {
			if !r.dead {
				s.mu.Unlock()
				return r, nil
			}
			// The engine exited on its own (crash / OOM kill). Drop the corpse
			// and fall through to spawn a fresh one, so the model self-heals
			// instead of failing every subsequent request.
			s.stopRunnerLocked(r)
			delete(s.runners, m.Name)
		}
		if ch, loading := s.loading[m.Name]; loading {
			s.mu.Unlock()
			select {
			case <-ch: // another caller's load finished; re-check the map
				continue
			case <-ctx.Done():
				// The caller gave up; the load continues in the background so
				// the next request benefits from it.
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
		r.timer = time.AfterFunc(s.opts.IdleTTL, func() { s.reap(m.Name) })
		s.runners[m.Name] = r
		s.mu.Unlock()
		return r, nil
	}
}

// evictForLoadLocked frees capacity for one more model, never evicting a runner
// with a request in flight — killing a busy engine would fail a user's
// in-progress generation and throw away minutes of GPU work. If every runner is
// busy we temporarily exceed MaxLoaded rather than do that. Callers hold s.mu.
func (s *Supervisor) evictForLoadLocked() {
	for len(s.runners) >= s.opts.MaxLoaded {
		victim := s.lruIdleLocked()
		if victim == nil {
			return // all loaded models are busy; allow a temporary overshoot
		}
		s.stopRunnerLocked(victim)
		delete(s.runners, victim.name)
	}
}

// spawn starts an engine for m and waits for it to become healthy. It does not
// touch supervisor state, so it is safe to call without s.mu held.
func (s *Supervisor) spawn(m types.Manifest) (*runner, error) {
	port, err := freePort(s.opts.Host)
	if err != nil {
		return nil, fmt.Errorf("supervisor: pick port: %w", err)
	}
	args, err := buildArgs(m, s.opts.BlobPath, s.opts.Host, port, s.opts.LoraDir)
	if err != nil {
		return nil, err
	}
	if s.opts.LogDir != "" {
		if err := os.MkdirAll(s.opts.LogDir, 0o755); err != nil {
			return nil, fmt.Errorf("supervisor: create log dir: %w", err)
		}
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
	if s.opts.LogDir != "" {
		cmd.Dir = s.opts.LogDir
	} else {
		cmd.Dir = os.TempDir()
	}
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	if err := cmd.Start(); err != nil {
		cancel()
		_ = logFile.Close()
		return nil, fmt.Errorf("supervisor: start engine: %w", err)
	}

	r := &runner{
		name:     m.Name,
		cmd:      cmd,
		cancel:   cancel,
		client:   engineclient.New("http://" + net.JoinHostPort(s.opts.Host, port)),
		logFile:  logFile,
		logPath:  logPath,
		lastUsed: time.Now(),
		done:     make(chan struct{}),
	}

	// One goroutine owns Wait(): it reaps the child (no zombies) and records
	// that the process is gone, whether we killed it or it died on its own.
	go func() {
		_ = cmd.Wait()
		s.mu.Lock()
		r.dead = true
		s.mu.Unlock()
		_ = logFile.Close()
		close(r.done)
	}()

	// The probe is bounded by StartTimeout, NOT by the requesting client's
	// context: a client that times out or disconnects must not kill an engine
	// that is still loading (the next request would restart it from scratch).
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

// probe polls the engine's Capabilities endpoint until it answers, ctx expires,
// or the process dies. Watching for process death turns an engine that exits
// immediately (bad flags, missing weights) into a fast, clear failure instead of
// a StartTimeout-long wait.
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

// reap stops a runner once it has been idle for IdleTTL. Skips (and reschedules)
// while a generation is in flight or the runner was used more recently than
// IdleTTL ago.
func (s *Supervisor) reap(name string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	r, ok := s.runners[name]
	if !ok {
		return
	}
	if r.inFlight > 0 || time.Since(r.lastUsed) < s.opts.IdleTTL {
		r.timer.Reset(s.opts.IdleTTL)
		return
	}
	s.stopRunnerLocked(r)
	delete(s.runners, name)
}

// lruIdleLocked returns the least-recently-used runner that has no request in
// flight, or nil if every runner is busy. Callers must hold s.mu.
func (s *Supervisor) lruIdleLocked() *runner {
	var oldest *runner
	for _, r := range s.runners {
		if r.inFlight > 0 {
			continue
		}
		if oldest == nil || r.lastUsed.Before(oldest.lastUsed) {
			oldest = r
		}
	}
	return oldest
}

// stopRunnerLocked signals a runner's process to die. It does not wait: the
// per-runner reaper goroutine calls Wait and closes the log, so the supervisor
// mutex is never held across a process teardown. Callers must hold s.mu.
func (s *Supervisor) stopRunnerLocked(r *runner) {
	if r.timer != nil {
		r.timer.Stop()
	}
	if r.cancel != nil {
		r.cancel()
	}
}

// freePort asks the OS for a free TCP port on host and returns it as a string.
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

// tailFile returns up to max trailing bytes of the file at path, or "" on error.
func tailFile(path string, max int) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	if len(data) > max {
		data = data[len(data)-max:]
	}
	return string(data)
}
