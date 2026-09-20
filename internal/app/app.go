// Package app wires the oflux subsystems (store, supervisor, puller, HTTP
// server) into a single runnable daemon. Both `oflux serve` and the menu-bar
// app build an App via Setup and host it, so the daemon behaves identically
// however it is launched.
package app

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"time"

	"oflux/internal/hfclient"
	"oflux/internal/puller"
	"oflux/internal/server"
	"oflux/internal/store"
	"oflux/internal/supervisor"
	"oflux/internal/types"
	"oflux/internal/updater"
)

// App holds the constructed subsystems.
type App struct {
	Store  *store.Store
	Sup    *supervisor.Supervisor
	Server *server.Server
	Cfg    types.Config

	enginePath string
	// exePath is os.Executable's answer with symlinks left unresolved, which is
	// the form resolveEngineFrom needs.
	exePath string
}

// Setup opens the store, loads config, locates the engine, and constructs the
// supervisor/puller/server. A missing engine binary is a warning, not an error:
// pulling still works and generation fails clearly until the engine is present.
func Setup() (*App, error) {
	st, err := store.Open(os.Getenv("OFLUX_HOME"))
	if err != nil {
		return nil, err
	}
	cfg, err := st.LoadConfig()
	if err != nil {
		return nil, err
	}

	// os.Executable's error needs no handling: resolveEngineFrom treats "" as
	// "no idea where I am" and falls back to $PATH.
	exe, _ := os.Executable()
	enginePath, engErr := resolveEngineFrom(exe)
	if engErr != nil {
		fmt.Fprintln(os.Stderr, "warning:", engErr, "(pull works; generation will fail until the engine is available)")
	}

	// An unparseable or non-positive TTL lands on supervisor.New's own default,
	// so there is nothing to fix up here.
	idle, _ := time.ParseDuration(cfg.IdleTTL)
	sup := supervisor.New(supervisor.Options{
		EnginePath:    enginePath,
		IdleTTL:       idle,
		MaxLoaded:     cfg.MaxLoaded,
		MaxConcurrent: cfg.MaxConcurrent,
		QueueDepth:    cfg.QueueDepth,
		LogDir:        st.LogsDir(),
		BlobPath:      st.BlobPath,
		LoraDir:       st.LorasDir(),
		// Large diffusion checkpoints (12-20B) take minutes to load into Metal;
		// give the startup health-probe generous headroom before giving up.
		StartTimeout: 10 * time.Minute,
	})

	hf := hfclient.New(cfg.HFToken)
	pl := puller.New(hf, st)
	srv := server.New(st, sup, pl, cfg)
	return &App{Store: st, Sup: sup, Server: srv, Cfg: cfg, enginePath: enginePath, exePath: exe}, nil
}

// Addr is the daemon listen address.
func (a *App) Addr() string { return net.JoinHostPort("127.0.0.1", strconv.Itoa(a.Cfg.Port)) }

// Serve runs the HTTP server until ctx is cancelled, then shuts the supervisor
// down. It returns nil on a clean shutdown.
func (a *App) Serve(ctx context.Context) error {
	// Claim the port FIRST. Everything below mutates machine-wide state (killing
	// stray engines, repairing the app bundle), and must not run if another
	// daemon already owns the port — a second `oflux serve` would otherwise
	// SIGKILL the live daemon's engines before discovering it can't listen.
	lns, err := a.listenLoopback()
	if err != nil {
		return fmt.Errorf("oflux is already running on port %d (%w)", a.Cfg.Port, err)
	}
	// Announce only once the port is actually ours. Printing before the bind
	// claimed to be serving and then failed on the next line.
	fmt.Printf("oflux serving on http://%s\n", a.Addr())

	// Reap any sd-server orphaned by a previously-killed daemon: it does not die
	// with its parent, and stale engines hold ports and many GB of memory.
	a.reapOrphanEngines()

	// Repair a bundle left half-swapped by an update that was interrupted
	// (logout, reboot, force-quit) so the app can never be stranded.
	if msg := updater.Recover(updater.AppPathFromExe(realPath(a.exePath))); msg != "" {
		fmt.Fprintln(os.Stderr, "oflux:", msg)
	}

	httpSrv := &http.Server{Handler: a.Server.Handler()}
	go func() {
		<-ctx.Done()
		shutCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = httpSrv.Shutdown(shutCtx)
	}()

	errCh := make(chan error, len(lns))
	for _, ln := range lns {
		go func() { errCh <- httpSrv.Serve(ln) }()
	}
	serveErr := <-errCh // first listener to stop ends the daemon

	// Stop engines only after the HTTP server has finished draining, so we never
	// kill an engine out from under a request the server is still completing.
	a.Sup.Shutdown()
	if errors.Is(serveErr, http.ErrServerClosed) {
		return nil
	}
	return serveErr
}

// listenLoopback binds the daemon port on both loopback families.
//
// This matters for the web UI: on macOS "localhost" resolves to ::1 before
// 127.0.0.1, so a browser pointed at http://localhost:11534 tries IPv6 first
// and stalls if only IPv4 is bound. Binding both means every spelling of
// loopback works. We deliberately never bind 0.0.0.0 — the daemon has no auth
// and must stay off the network.
func (a *App) listenLoopback() ([]net.Listener, error) {
	port := strconv.Itoa(a.Cfg.Port)
	var lns []net.Listener
	for _, addr := range []string{a.Addr(), net.JoinHostPort("::1", port)} {
		ln, err := net.Listen("tcp", addr)
		if err != nil {
			if len(lns) == 0 {
				return nil, err // couldn't get IPv4: the port is taken
			}
			// No IPv6 loopback on this machine; IPv4 alone is fine.
			continue
		}
		lns = append(lns, ln)
	}
	return lns, nil
}

// reapOrphanEngines kills any sd-server left running from our engine path by a
// previous daemon that didn't shut down cleanly.
func (a *App) reapOrphanEngines() {
	if a.enginePath == "" {
		return
	}
	_ = exec.Command("pkill", "-9", "-f", a.enginePath).Run()
}

// resolveEngineFrom locates the bundled sd-server binary given the path
// os.Executable() reported ("" searches $PATH only): $OFLUX_ENGINE, then next
// to the executable (including the .app Resources dir), then $PATH. exe is
// injected so the bundle/symlink layouts can be tested without reinstalling.
//
// It is also resolved through symlinks: `oflux` on the user's PATH is a symlink
// into oflux.app/Contents/MacOS/, and os.Executable() returns that symlink — so
// searching relative to it looks beside the symlink (e.g. /opt/homebrew/bin) and
// never finds Contents/Resources/sd-server. The symptom is `oflux serve` warning
// that the engine is missing while the menu-bar app, launched from the bundle,
// works fine.
func resolveEngineFrom(exe string) (string, error) {
	if p := os.Getenv("OFLUX_ENGINE"); p != "" {
		if !isExec(p) {
			return "", fmt.Errorf("OFLUX_ENGINE=%s is not an executable", p)
		}
		return p, nil
	}
	if exe != "" {
		// Search beside the real binary too, but keep the symlink's own
		// directory first so a side-by-side dev build still wins.
		dirs := []string{filepath.Dir(exe)}
		if d := filepath.Dir(realPath(exe)); d != dirs[0] {
			dirs = append(dirs, d)
		}
		for _, dir := range dirs {
			for _, cand := range []string{
				filepath.Join(dir, "sd-server"),
				filepath.Join(dir, "..", "Resources", "sd-server"),
			} {
				if isExec(cand) {
					return cand, nil
				}
			}
		}
	}
	if p, err := exec.LookPath("sd-server"); err == nil {
		return p, nil
	}
	return "", errors.New("sd-server engine binary not found (set OFLUX_ENGINE)")
}

// realPath resolves symlinks in p, falling back to p when it cannot.
func realPath(p string) string {
	if resolved, err := filepath.EvalSymlinks(p); err == nil {
		return resolved
	}
	return p
}

func isExec(p string) bool {
	fi, err := os.Stat(p)
	return err == nil && !fi.IsDir() && fi.Mode()&0o111 != 0
}
