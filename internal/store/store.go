// Package store is the on-disk model store for oflux. It mirrors Ollama's
// layout: content-addressed blobs plus per-model manifests, rooted at
// ~/.oflux (or $OFLUX_HOME).
//
//	<root>/blobs/sha256-<hex>   content-addressed weight files
//	<root>/manifests/<name>.json installed-model descriptions
//	<root>/logs/                 daemon/engine logs
//	<root>/config.json           daemon configuration
//
// Blobs use a dash ("sha256-<hex>") rather than a colon so the filenames are
// safe on every filesystem.
package store

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"syscall"

	"oflux/internal/types"
)

// ErrManifestNotFound is returned by ReadManifest when no manifest with the
// requested name is installed.
var ErrManifestNotFound = errors.New("store: manifest not found")

// Store is a handle to an on-disk oflux model store rooted at a directory.
type Store struct {
	root string
	// gcMu orders garbage collection against in-progress pulls. A pull holds it
	// for reading from its first download until its manifest is committed; GC
	// takes it for writing. Without this, GC (triggered by an unrelated
	// `oflux rm`) can delete freshly-downloaded blobs that no manifest
	// references yet, producing an installed model whose files are gone.
	gcMu sync.RWMutex
}

// BeginWrite marks the start of a multi-step write (a pull) that will end with
// WriteManifest. Garbage collection blocks until the returned function is
// called, so blobs staged for a not-yet-written manifest are never swept.
func (s *Store) BeginWrite() func() {
	s.gcMu.RLock()
	return s.gcMu.RUnlock
}

// Open roots the store at root, creating blobs/, manifests/ and logs/ if they
// are missing. When root is empty it uses $OFLUX_HOME if set, otherwise
// ~/.oflux.
func Open(root string) (*Store, error) {
	if root == "" {
		if env := os.Getenv("OFLUX_HOME"); env != "" {
			root = env
		} else {
			home, err := os.UserHomeDir()
			if err != nil {
				return nil, fmt.Errorf("store: resolve home dir: %w", err)
			}
			root = filepath.Join(home, ".oflux")
		}
	}
	s := &Store{root: root}
	for _, dir := range []string{s.root, s.BlobsDir(), s.ManifestsDir(), s.LogsDir()} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, fmt.Errorf("store: create %s: %w", dir, err)
		}
	}
	return s, nil
}

// Root returns the store's root directory.
func (s *Store) Root() string { return s.root }

// BlobsDir returns the directory holding content-addressed blobs.
func (s *Store) BlobsDir() string { return filepath.Join(s.root, "blobs") }

// ManifestsDir returns the directory holding installed-model manifests.
func (s *Store) ManifestsDir() string { return filepath.Join(s.root, "manifests") }

// LogsDir returns the directory for daemon and engine logs.
func (s *Store) LogsDir() string { return filepath.Join(s.root, "logs") }

// BlobName converts a hex sha256 (optionally already prefixed with "sha256:"
// or "sha256-") into the on-disk blob filename "sha256-<hex>".
func BlobName(sha256hex string) string {
	h := sha256hex
	switch {
	case strings.HasPrefix(h, "sha256:"):
		h = strings.TrimPrefix(h, "sha256:")
	case strings.HasPrefix(h, "sha256-"):
		h = strings.TrimPrefix(h, "sha256-")
	}
	return "sha256-" + h
}

// BlobPath returns the absolute path for a blob address. It accepts either a
// full "sha256-<hex>" name or a bare hex string.
func (s *Store) BlobPath(blob string) string {
	return filepath.Join(s.BlobsDir(), BlobName(blob))
}

// HasBlob reports whether the blob is present in the store.
func (s *Store) HasBlob(blob string) bool {
	info, err := os.Stat(s.BlobPath(blob))
	return err == nil && !info.IsDir()
}

// PutBlob moves the file at srcPath into the blob store under the address
// derived from sha256hex. It is idempotent: if the blob already exists it
// removes srcPath and returns the existing name.
func (s *Store) PutBlob(sha256hex, srcPath string) (string, error) {
	name := BlobName(sha256hex)
	dst := filepath.Join(s.BlobsDir(), name)

	if s.HasBlob(name) {
		// Already have identical content; drop the incoming copy.
		if err := os.Remove(srcPath); err != nil && !os.IsNotExist(err) {
			return "", fmt.Errorf("store: remove duplicate src %s: %w", srcPath, err)
		}
		return name, nil
	}

	if err := os.MkdirAll(s.BlobsDir(), 0o755); err != nil {
		return "", fmt.Errorf("store: ensure blobs dir: %w", err)
	}

	if err := os.Rename(srcPath, dst); err != nil {
		// Cross-device rename (EXDEV) can't be done atomically; fall back to a
		// copy + remove.
		if isCrossDevice(err) {
			// Copy to a temp file and rename into place, never straight to the
			// content address: a copy interrupted by a full disk or a kill
			// would otherwise leave truncated bytes at a valid-looking blob
			// name, which HasBlob would then trust forever.
			tmp := dst + ".incoming"
			if cerr := copyFile(srcPath, tmp); cerr != nil {
				_ = os.Remove(tmp)
				return "", fmt.Errorf("store: copy blob: %w", cerr)
			}
			if rerr := os.Rename(tmp, dst); rerr != nil {
				_ = os.Remove(tmp)
				return "", fmt.Errorf("store: move copied blob into place: %w", rerr)
			}
			if rerr := os.Remove(srcPath); rerr != nil && !os.IsNotExist(rerr) {
				return "", fmt.Errorf("store: remove src after copy: %w", rerr)
			}
			return name, nil
		}
		return "", fmt.Errorf("store: move blob into place: %w", err)
	}
	return name, nil
}

// isCrossDevice reports whether err is an EXDEV cross-device link error, which
// os.Rename returns when src and dst live on different filesystems.
func isCrossDevice(err error) bool {
	return errors.Is(err, syscall.EXDEV)
}

// copyFile copies src to dst, creating dst with 0644 permissions.
func copyFile(src, dst string) (err error) {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	out, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	defer func() {
		cerr := out.Close()
		if err == nil {
			err = cerr
		}
	}()

	if _, err = io.Copy(out, in); err != nil {
		return err
	}
	return out.Sync()
}

// manifestPath returns the on-disk path for a manifest name, rejecting names
// that contain path separators or "..".
func (s *Store) manifestPath(name string) (string, error) {
	if name == "" {
		return "", fmt.Errorf("store: empty manifest name")
	}
	if strings.ContainsAny(name, "/\\") || strings.Contains(name, "..") {
		return "", fmt.Errorf("store: invalid manifest name %q", name)
	}
	return filepath.Join(s.ManifestsDir(), name+".json"), nil
}

// WriteManifest writes m to manifests/<name>.json as pretty JSON.
func (s *Store) WriteManifest(m types.Manifest) error {
	path, err := s.manifestPath(m.Name)
	if err != nil {
		return err
	}
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return fmt.Errorf("store: marshal manifest %q: %w", m.Name, err)
	}
	data = append(data, '\n')
	if err := os.MkdirAll(s.ManifestsDir(), 0o755); err != nil {
		return fmt.Errorf("store: ensure manifests dir: %w", err)
	}
	// Write atomically: a manifest truncated by a crash would be unparsable,
	// and one bad file makes ListManifests (and therefore GC) fail for the
	// whole store.
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return fmt.Errorf("store: write manifest %q: %w", m.Name, err)
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("store: commit manifest %q: %w", m.Name, err)
	}
	return nil
}

// ReadManifest reads manifests/<name>.json. It returns ErrManifestNotFound if
// no such manifest exists.
func (s *Store) ReadManifest(name string) (types.Manifest, error) {
	path, err := s.manifestPath(name)
	if err != nil {
		return types.Manifest{}, err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return types.Manifest{}, fmt.Errorf("%q: %w", name, ErrManifestNotFound)
		}
		return types.Manifest{}, fmt.Errorf("store: read manifest %q: %w", name, err)
	}
	var m types.Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return types.Manifest{}, fmt.Errorf("store: parse manifest %q: %w", name, err)
	}
	return m, nil
}

// ListManifests returns all installed manifests, sorted by Name.
func (s *Store) ListManifests() ([]types.Manifest, error) {
	entries, err := os.ReadDir(s.ManifestsDir())
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("store: list manifests: %w", err)
	}
	var out []types.Manifest
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		name := strings.TrimSuffix(e.Name(), ".json")
		m, err := s.ReadManifest(name)
		if err != nil {
			return nil, err
		}
		out = append(out, m)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out, nil
}

// RemoveManifest deletes the named manifest and then garbage-collects any
// blobs that no remaining manifest references. It returns the freed blob names.
func (s *Store) RemoveManifest(name string) ([]string, error) {
	path, err := s.manifestPath(name)
	if err != nil {
		return nil, err
	}
	// Confirm the manifest set is readable BEFORE deleting anything: if some
	// other manifest is unparsable, GC would fail and we'd have deleted this
	// one with its blobs left orphaned and un-collectable.
	if _, err := s.referencedBlobs(); err != nil {
		return nil, err
	}
	if err := os.Remove(path); err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("%q: %w", name, ErrManifestNotFound)
		}
		return nil, fmt.Errorf("store: remove manifest %q: %w", name, err)
	}
	return s.GC()
}

// referencedBlobs returns the set of blob names referenced by any installed
// manifest's components.
func (s *Store) referencedBlobs() (map[string]bool, error) {
	manifests, err := s.ListManifests()
	if err != nil {
		return nil, err
	}
	set := make(map[string]bool)
	for _, m := range manifests {
		for _, c := range m.Components {
			if c.Blob == "" {
				continue
			}
			set[BlobName(c.Blob)] = true
		}
	}
	return set, nil
}

// GC removes any blob not referenced by an installed manifest and returns the
// freed blob names, sorted.
func (s *Store) GC() ([]string, error) {
	// Wait for any in-progress pull to commit its manifest first.
	s.gcMu.Lock()
	defer s.gcMu.Unlock()

	referenced, err := s.referencedBlobs()
	if err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(s.BlobsDir())
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("store: list blobs: %w", err)
	}
	var freed []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if referenced[name] {
			continue
		}
		if err := os.Remove(filepath.Join(s.BlobsDir(), name)); err != nil {
			return nil, fmt.Errorf("store: gc remove %s: %w", name, err)
		}
		freed = append(freed, name)
	}
	sort.Strings(freed)
	return freed, nil
}

// configPath returns the path to the store's config file.
func (s *Store) configPath() string { return filepath.Join(s.root, "config.json") }

// LoadConfig returns the store's configuration. When config.json is absent it
// returns types.DefaultConfig(); otherwise the file is merged over the
// defaults so unspecified fields keep their default values.
func (s *Store) LoadConfig() (types.Config, error) {
	cfg := types.DefaultConfig()
	data, err := os.ReadFile(s.configPath())
	if err != nil {
		if os.IsNotExist(err) {
			return cfg, nil
		}
		return types.Config{}, fmt.Errorf("store: read config: %w", err)
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return types.Config{}, fmt.Errorf("store: parse config: %w", err)
	}
	return cfg, nil
}

// SaveConfig writes cfg to config.json at the store root as pretty JSON.
func (s *Store) SaveConfig(cfg types.Config) error {
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return fmt.Errorf("store: marshal config: %w", err)
	}
	data = append(data, '\n')
	if err := os.MkdirAll(s.root, 0o755); err != nil {
		return fmt.Errorf("store: ensure root: %w", err)
	}
	if err := os.WriteFile(s.configPath(), data, 0o644); err != nil {
		return fmt.Errorf("store: write config: %w", err)
	}
	return nil
}
