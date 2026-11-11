package app

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// fakeBundle builds an .app-shaped tree with the engine in Contents/Resources,
// plus a symlink to the executable standing in for the PATH entry.
func fakeBundle(t *testing.T) (exe, symlink, engine string) {
	t.Helper()
	root := t.TempDir()
	macos := filepath.Join(root, "oflux.app", "Contents", "MacOS")
	res := filepath.Join(root, "oflux.app", "Contents", "Resources")
	for _, d := range []string{macos, res, filepath.Join(root, "bin")} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	exe = filepath.Join(macos, "oflux")
	engine = filepath.Join(res, "sd-server")
	for _, p := range []string{exe, engine} {
		if err := os.WriteFile(p, []byte("#!/bin/sh\n"), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	symlink = filepath.Join(root, "bin", "oflux")
	if err := os.Symlink(exe, symlink); err != nil {
		t.Fatal(err)
	}
	return exe, symlink, engine
}

// Regression: `oflux` on PATH is a symlink into the bundle. Resolving the engine
// relative to the symlink's own directory finds nothing, so `oflux serve`
// warned the engine was missing even though the bundle shipped it.
func TestResolveEngineFollowsSymlinkIntoBundle(t *testing.T) {
	_, symlink, engine := fakeBundle(t)
	t.Setenv("OFLUX_ENGINE", "")

	got, err := resolveEngineFrom(symlink)
	if err != nil {
		t.Fatalf("resolve via symlink: %v", err)
	}
	if !sameFile(t, got, engine) {
		t.Fatalf("got %q, want the bundled engine %q", got, engine)
	}
}

func TestResolveEngineFromRealExecutable(t *testing.T) {
	exe, _, engine := fakeBundle(t)
	t.Setenv("OFLUX_ENGINE", "")

	got, err := resolveEngineFrom(exe)
	if err != nil {
		t.Fatalf("resolve via real exe: %v", err)
	}
	if !sameFile(t, got, engine) {
		t.Fatalf("got %q, want %q", got, engine)
	}
}

// A dev build with sd-server sitting next to it must win over anything the
// symlink target would resolve to.
func TestResolveEnginePrefersSideBySide(t *testing.T) {
	_, symlink, _ := fakeBundle(t)
	t.Setenv("OFLUX_ENGINE", "")

	local := filepath.Join(filepath.Dir(symlink), "sd-server")
	if err := os.WriteFile(local, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	got, err := resolveEngineFrom(symlink)
	if err != nil {
		t.Fatal(err)
	}
	if !sameFile(t, got, local) {
		t.Fatalf("got %q, want the side-by-side engine %q", got, local)
	}
}

func TestResolveEngineEnvOverride(t *testing.T) {
	_, symlink, _ := fakeBundle(t)

	other := filepath.Join(t.TempDir(), "sd-server")
	if err := os.WriteFile(other, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("OFLUX_ENGINE", other)
	got, err := resolveEngineFrom(symlink)
	if err != nil || got != other {
		t.Fatalf("OFLUX_ENGINE should win: got %q, %v", got, err)
	}

	// A bad override is an error, not a silent fallback: it would otherwise
	// look like the override took effect when it did not.
	t.Setenv("OFLUX_ENGINE", filepath.Join(t.TempDir(), "nope"))
	if _, err := resolveEngineFrom(symlink); err == nil {
		t.Fatal("a non-executable OFLUX_ENGINE must error")
	}
}

func sameFile(t *testing.T, a, b string) bool {
	t.Helper()
	fa, err := os.Stat(a)
	if err != nil {
		return false
	}
	fb, err := os.Stat(b)
	if err != nil {
		return false
	}
	return os.SameFile(fa, fb)
}

func TestResolveEngineFallsBackToPath(t *testing.T) {
	t.Setenv("OFLUX_ENGINE", "")
	bin := t.TempDir()
	t.Setenv("PATH", bin)
	// exe "" is os.Executable having failed: $PATH is all that is left.
	if _, err := resolveEngineFrom(""); err == nil {
		t.Fatal("no engine anywhere must be an error")
	}
	engine := filepath.Join(bin, "sd-server")
	if err := os.WriteFile(engine, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	if got, err := resolveEngineFrom(""); err != nil || got != engine {
		t.Fatalf("got %q, %v; want %q from PATH", got, err, engine)
	}
}

func TestIsExecAndRealPath(t *testing.T) {
	dir := t.TempDir()
	exe, plain := filepath.Join(dir, "exe"), filepath.Join(dir, "plain")
	os.WriteFile(exe, nil, 0o755)
	os.WriteFile(plain, nil, 0o644)
	for p, want := range map[string]bool{exe: true, plain: false, dir: false, filepath.Join(dir, "nope"): false} {
		if isExec(p) != want {
			t.Errorf("isExec(%s) = %v, want %v", filepath.Base(p), !want, want)
		}
	}
	link := filepath.Join(dir, "link")
	if err := os.Symlink(exe, link); err != nil {
		t.Fatal(err)
	}
	if !sameFile(t, realPath(link), exe) {
		t.Errorf("realPath(link) = %q", realPath(link))
	}
	// Unresolvable stays as given rather than becoming "".
	if got := realPath(filepath.Join(dir, "nope")); got != filepath.Join(dir, "nope") {
		t.Errorf("realPath(missing) = %q", got)
	}
}

// newApp is a daemon confined to temp dirs and a free port. No engine: with one
// set, Serve would pkill by its path, and a test has no business doing that.
func newApp(t *testing.T, port int) *App {
	t.Helper()
	t.Setenv("OFLUX_HOME", t.TempDir())
	t.Setenv("OFLUX_BIN_DIR", t.TempDir())
	t.Setenv("OFLUX_ENGINE", "")
	t.Setenv("PATH", t.TempDir())
	a, err := Setup()
	if err != nil {
		t.Fatal(err)
	}
	if a.enginePath != "" {
		t.Fatalf("test daemon found an engine: %s", a.enginePath)
	}
	a.Cfg.Port = port
	return a
}

func freePort(t *testing.T) int {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	return ln.Addr().(*net.TCPAddr).Port
}

// There is no auth, so every listener must be loopback — never a wildcard.
func TestListenLoopback(t *testing.T) {
	a := newApp(t, freePort(t))
	if want := fmt.Sprintf("127.0.0.1:%d", a.Cfg.Port); a.Addr() != want {
		t.Errorf("Addr = %q, want %q", a.Addr(), want)
	}
	lns, err := a.listenLoopback()
	if err != nil {
		t.Fatal(err)
	}
	if len(lns) == 0 {
		t.Fatal("no listeners")
	}
	for _, ln := range lns {
		defer ln.Close()
		addr := ln.Addr().(*net.TCPAddr)
		if !addr.IP.IsLoopback() || addr.Port != a.Cfg.Port {
			t.Errorf("listening on %s", addr)
		}
	}
	if !lns[0].Addr().(*net.TCPAddr).IP.Equal(net.IPv4(127, 0, 0, 1)) {
		t.Errorf("first listener is %s, want IPv4 loopback", lns[0].Addr())
	}
	// The port is ours now, so a second claim has to fail outright.
	if again, err := a.listenLoopback(); err == nil {
		for _, ln := range again {
			ln.Close()
		}
		t.Error("claimed a port that was already bound")
	}
}

// One daemon end to end: both loopback spellings answer, a second daemon on the
// same port refuses to start, and cancelling the context is a clean exit.
func TestServe(t *testing.T) {
	a := newApp(t, freePort(t))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- a.Serve(ctx) }()

	get := func(host string) (*http.Response, error) {
		return http.Get(fmt.Sprintf("http://%s/healthz", net.JoinHostPort(host, strconv.Itoa(a.Cfg.Port))))
	}
	var resp *http.Response
	var err error
	for range 100 { // the bind is asynchronous to this goroutine
		if resp, err = get("127.0.0.1"); err == nil {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if err != nil {
		t.Fatalf("daemon never answered: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("healthz = %s", resp.Status)
	}
	// macOS resolves localhost to ::1 first; skip only where there is no IPv6 loopback.
	if ln, lerr := net.Listen("tcp", "[::1]:0"); lerr == nil {
		ln.Close()
		resp, err := get("::1")
		if err != nil {
			t.Errorf("::1 not served: %v", err)
		} else {
			resp.Body.Close()
		}
	}

	second := newApp(t, a.Cfg.Port)
	if err := second.Serve(context.Background()); err == nil || !strings.Contains(err.Error(), "already running") {
		t.Errorf("second daemon on the same port: err = %v", err)
	}

	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Errorf("clean shutdown returned %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("Serve did not return after cancel")
	}
}
