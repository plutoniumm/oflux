package selfinstall

import (
	"os"
	"path/filepath"
	"testing"
)

// A real binary at the target path (a hand-built or Homebrew-installed oflux)
// must never be deleted to make room for our symlink.
func TestLinkCLINeverDeletesARealBinary(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "oflux")
	if err := os.WriteFile(target, []byte("#!/bin/sh\necho real\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	exe := filepath.Join(dir, "bundled-oflux")
	if err := os.WriteFile(exe, []byte("x"), 0o755); err != nil {
		t.Fatal(err)
	}

	// Point the candidate search at our temp dir by making it the home-based one.
	t.Setenv("HOME", dir)
	// BinDirs() includes $HOME/.local/bin; create a real file there too.
	localBin := filepath.Join(dir, ".local", "bin")
	if err := os.MkdirAll(localBin, 0o755); err != nil {
		t.Fatal(err)
	}
	realIn := filepath.Join(localBin, "oflux")
	if err := os.WriteFile(realIn, []byte("real"), 0o755); err != nil {
		t.Fatal(err)
	}

	_, _, _ = LinkCLI(exe)

	data, err := os.ReadFile(realIn)
	if err != nil {
		t.Fatalf("the real binary was deleted: %v", err)
	}
	if string(data) != "real" {
		t.Errorf("the real binary was replaced, content = %q", data)
	}
}

// An existing symlink of ours is replaced (idempotent re-install).
func TestLinkCLIReplacesOwnSymlink(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("HOME", dir)
	localBin := filepath.Join(dir, ".local", "bin")
	if err := os.MkdirAll(localBin, 0o755); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(localBin, "oflux")
	if err := os.Symlink("/old/oflux.app/Contents/MacOS/oflux", link); err != nil {
		t.Fatal(err)
	}
	exe := filepath.Join(dir, "new-oflux")
	if err := os.WriteFile(exe, []byte("x"), 0o755); err != nil {
		t.Fatal(err)
	}

	target, _, err := LinkCLI(exe)
	if err != nil {
		t.Fatalf("LinkCLI: %v", err)
	}
	got, err := os.Readlink(target)
	if err != nil || got != exe {
		t.Errorf("symlink = %q (err %v), want %q", got, err, exe)
	}
}
