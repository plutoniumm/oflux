// Package selfinstall wires oflux into the user's environment: a PATH symlink
// so the `oflux` CLI works from a terminal (Ollama-style). Used by the
// `oflux install` command and by the menu-bar app's first-run self-setup.
package selfinstall

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// BinDirs are the PATH directories tried when linking the CLI, most
// conventional first. ~/.local/bin is created if it doesn't exist.
func BinDirs() []string {
	home, _ := os.UserHomeDir()
	return []string{"/usr/local/bin", "/opt/homebrew/bin", filepath.Join(home, ".local", "bin")}
}

// LinkCLI symlinks the oflux binary at exe into a PATH directory. It returns the
// link path and whether that directory is actually on $PATH. Idempotent: if the
// correct symlink already exists it is left in place.
func LinkCLI(exe string) (target string, onPath bool, err error) {
	pathset := map[string]bool{}
	for _, p := range filepath.SplitList(os.Getenv("PATH")) {
		if p != "" {
			pathset[p] = true
		}
	}
	home, _ := os.UserHomeDir()
	var last error
	for _, dir := range BinDirs() {
		if home != "" && strings.HasPrefix(dir, home) {
			_ = os.MkdirAll(dir, 0o755)
		}
		if fi, e := os.Stat(dir); e != nil || !fi.IsDir() {
			continue
		}
		target = filepath.Join(dir, "oflux")
		switch fi, e := os.Lstat(target); {
		case e != nil:
			// Nothing there — free to create the link.
		case fi.Mode()&os.ModeSymlink == 0:
			// A REAL file (e.g. a hand-built or Homebrew-installed oflux).
			// Never delete a user's binary; try the next directory instead.
			last = fmt.Errorf("%s exists and is not a symlink; leaving it alone", target)
			continue
		default:
			existing, _ := os.Readlink(target)
			if existing == exe {
				return target, pathset[dir], nil // already linked correctly
			}
			_ = os.Remove(target) // our own (or another) symlink: safe to replace
		}
		if e := os.Symlink(exe, target); e != nil {
			last = e
			continue
		}
		return target, pathset[dir], nil
	}
	if last == nil {
		last = errors.New("no writable bin directory found on PATH")
	}
	return "", false, last
}

// UnlinkCLI removes oflux symlinks we created (only those pointing into an
// oflux.app bundle), best effort.
func UnlinkCLI() {
	for _, dir := range BinDirs() {
		p := filepath.Join(dir, "oflux")
		if dest, err := os.Readlink(p); err == nil && strings.Contains(dest, "oflux.app/") {
			_ = os.Remove(p)
		}
	}
}
