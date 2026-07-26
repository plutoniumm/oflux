package app

import (
	"os"
	"path/filepath"
	"testing"
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
