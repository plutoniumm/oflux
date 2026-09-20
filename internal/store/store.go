// Package store is the on-disk model store for oflux. It mirrors Ollama's
// layout: content-addressed blobs plus per-model manifests, rooted at
// ~/.oflux (or $OFLUX_HOME).
//
//	<root>/blobs/sha256-<hex>   content-addressed weight files
//	<root>/manifests/<name>.json installed-model descriptions
//	<root>/loras/<name>.safetensors  LoRA adapters, addressed by name
//	<root>/loras/<name>.json     what an adapter is for (archs, steps, cfg)
//	<root>/presets/<name>.json   saved request defaults
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
	"slices"
	"strings"
	"sync"
	"syscall"

	"oflux/internal/types"
)

// ErrManifestNotFound is returned when no such manifest is installed.
var ErrManifestNotFound = errors.New("store: manifest not found")

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

// Open creates the store's directories. An empty root means $OFLUX_HOME, else
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
	for _, dir := range []string{s.root, s.BlobsDir(), s.ManifestsDir(), s.LogsDir(), s.LorasDir(), s.PresetsDir()} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, fmt.Errorf("store: create %s: %w", dir, err)
		}
	}
	return s, nil
}

func (s *Store) Root() string { return s.root }

func (s *Store) BlobsDir() string { return filepath.Join(s.root, "blobs") }

func (s *Store) ManifestsDir() string { return filepath.Join(s.root, "manifests") }

func (s *Store) LogsDir() string { return filepath.Join(s.root, "logs") }

// LoRAs are stored under their friendly name rather than content-addressed like
// weights: the engine is handed this directory as --lora-model-dir and resolves
// each request's LoRA by filename, so the name on disk IS the API identifier.
func (s *Store) LorasDir() string { return filepath.Join(s.root, "loras") }

// BlobName normalizes a hex sha256, with or without a "sha256:"/"sha256-"
// prefix, to the on-disk blob filename.
func BlobName(sha256hex string) string {
	h, _ := strings.CutPrefix(sha256hex, "sha256:")
	h, _ = strings.CutPrefix(h, "sha256-")
	return "sha256-" + h
}

func (s *Store) BlobPath(blob string) string {
	return filepath.Join(s.BlobsDir(), BlobName(blob))
}

func (s *Store) HasBlob(blob string) bool { return isFile(s.BlobPath(blob)) }

// PutBlob is idempotent: an existing blob wins and srcPath is removed.
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
	if err := placeFile(srcPath, dst); err != nil {
		return "", fmt.Errorf("store: move blob into place: %w", err)
	}
	return name, nil
}

func isFile(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

// placeFile moves src onto dst, consuming src.
//
// A cross-device rename (EXDEV) can't be done atomically, so it falls back to a
// copy. The copy goes to a temp file next to dst and is then renamed into
// place, never straight to the final name: a copy interrupted by a full disk or
// a kill would otherwise leave truncated bytes under a name the store trusts
// forever — a content address HasBlob believes, or a LoRA filename the engine
// loads.
func placeFile(src, dst string) error {
	err := os.Rename(src, dst)
	if err == nil || !errors.Is(err, syscall.EXDEV) {
		return err
	}
	tmp := dst + ".incoming"
	if err := copyFile(src, tmp); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("copy to %s: %w", tmp, err)
	}
	if err := os.Rename(tmp, dst); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("commit copy: %w", err)
	}
	if err := os.Remove(src); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("remove src after copy: %w", err)
	}
	return nil
}

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

// writeJSONAtomic writes via a temp file plus rename: in place, a crash leaves
// a truncated file, and one unparsable manifest fails ListManifests — and so
// GC — for the whole store.
func writeJSONAtomic(path string, v any) error {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}

// readJSON reports whether the file existed; a missing one is not an error.
func readJSON(path string, v any) (bool, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, err
	}
	if err := json.Unmarshal(data, v); err != nil {
		return true, err
	}
	return true, nil
}

const jsonExt = ".json"

// ValidName reports whether name is usable as a model, LoRA or preset name.
// All three become a path segment under the store, and a LoRA name is also the
// filename sd-server resolves against --lora-model-dir, so separators and ".."
// are refused. Nothing else is: model names come from Hugging Face repo ids and
// from --as, and a stricter charset would reject names that install fine today.
func ValidName(name string) error {
	if name == "" {
		return errors.New("store: empty name")
	}
	if strings.ContainsAny(name, `/\`) || strings.Contains(name, "..") {
		return fmt.Errorf("store: invalid name %q: must not contain '/', '\\' or '..'", name)
	}
	return nil
}

// ValidModelName is ValidName, spelled the way its call sites read.
func ValidModelName(name string) error { return ValidName(name) }

// pathFor is the only place a caller-supplied name becomes a path, so the name
// policy cannot drift between manifests, LoRAs and presets.
func (s *Store) pathFor(dir, name, ext string) (string, error) {
	if err := ValidName(name); err != nil {
		return "", err
	}
	return filepath.Join(dir, name+ext), nil
}

func (s *Store) manifestPath(name string) (string, error) {
	return s.pathFor(s.ManifestsDir(), name, jsonExt)
}

func (s *Store) WriteManifest(m types.Manifest) error {
	path, err := s.manifestPath(m.Name)
	if err != nil {
		return err
	}
	if err := writeJSONAtomic(path, m); err != nil {
		return fmt.Errorf("store: write manifest %q: %w", m.Name, err)
	}
	return nil
}

// ReadManifest returns ErrManifestNotFound if no such manifest exists.
func (s *Store) ReadManifest(name string) (types.Manifest, error) {
	path, err := s.manifestPath(name)
	if err != nil {
		return types.Manifest{}, err
	}
	var m types.Manifest
	found, err := readJSON(path, &m)
	if err != nil {
		return types.Manifest{}, fmt.Errorf("store: read manifest %q: %w", name, err)
	}
	if !found {
		return types.Manifest{}, fmt.Errorf("%q: %w", name, ErrManifestNotFound)
	}
	return m, nil
}

// eachManifest is the single reader of the manifest directory: ListManifests
// sorts what it yields, GC only wants the blob names.
func (s *Store) eachManifest(fn func(path string, m types.Manifest)) error {
	entries, err := os.ReadDir(s.ManifestsDir())
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("store: list manifests: %w", err)
	}
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), jsonExt) {
			continue
		}
		path := filepath.Join(s.ManifestsDir(), e.Name())
		var m types.Manifest
		found, err := readJSON(path, &m)
		if err != nil {
			return fmt.Errorf("store: read manifest %s: %w", e.Name(), err)
		}
		if !found {
			continue // removed while we were listing
		}
		fn(path, m)
	}
	return nil
}

// ListManifests returns all installed manifests, sorted by Name.
func (s *Store) ListManifests() ([]types.Manifest, error) {
	var out []types.Manifest
	if err := s.eachManifest(func(_ string, m types.Manifest) { out = append(out, m) }); err != nil {
		return nil, err
	}
	slices.SortFunc(out, func(a, b types.Manifest) int { return strings.Compare(a.Name, b.Name) })
	return out, nil
}

// RemoveManifest deletes the manifest, then collects the blobs no remaining
// manifest references, reporting the freed names and whether it ran at all.
//
// Collection is skipped rather than waited for when a pull is in flight. GC has
// to exclude in-progress downloads, and a multi-gigabyte pull holds that lock
// for many minutes — so blocking here hung `oflux rm` long after the manifest
// itself was gone, with nothing on screen to explain why. Blobs orphaned by the
// skip are collected by the next removal or GC.
func (s *Store) RemoveManifest(name string) (freed []string, collected bool, err error) {
	path, err := s.manifestPath(name)
	if err != nil {
		return nil, false, err
	}
	if !s.gcMu.TryLock() {
		// A pull is in flight, so nothing may be swept.
		return nil, false, removeManifestFile(path, name)
	}
	defer s.gcMu.Unlock()

	// Survey the SURVIVING manifests before deleting anything: if one of them
	// is unparsable the sweep cannot run, and having already deleted this one
	// would leave its blobs orphaned with nothing left to name them. It is also
	// why the manifest directory is walked once per removal, not twice.
	referenced, err := s.referencedBlobs(path)
	if err != nil {
		return nil, false, err
	}
	if err := removeManifestFile(path, name); err != nil {
		return nil, false, err
	}
	freed, err = s.sweep(referenced)
	return freed, true, err
}

func removeManifestFile(path, name string) error {
	if err := os.Remove(path); err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("%q: %w", name, ErrManifestNotFound)
		}
		return fmt.Errorf("store: remove manifest %q: %w", name, err)
	}
	return nil
}

// TryGC removes any blob not referenced by an installed manifest and returns
// the freed blob names, sorted. It runs only if no pull currently holds the
// store, reporting collected=false when it backed off instead of waiting.
func (s *Store) TryGC() (freed []string, collected bool, err error) {
	if !s.gcMu.TryLock() {
		return nil, false, nil
	}
	defer s.gcMu.Unlock()
	referenced, err := s.referencedBlobs("")
	if err != nil {
		return nil, true, err
	}
	freed, err = s.sweep(referenced)
	return freed, true, err
}

// referencedBlobs ignores the manifest at skipPath: the one being removed.
func (s *Store) referencedBlobs(skipPath string) (map[string]bool, error) {
	referenced := make(map[string]bool)
	err := s.eachManifest(func(path string, m types.Manifest) {
		if path == skipPath {
			return
		}
		for _, c := range m.Components {
			if c.Blob != "" {
				referenced[BlobName(c.Blob)] = true
			}
		}
	})
	if err != nil {
		return nil, err
	}
	return referenced, nil
}

// sweep requires s.gcMu held.
func (s *Store) sweep(referenced map[string]bool) ([]string, error) {
	entries, err := os.ReadDir(s.BlobsDir())
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("store: list blobs: %w", err)
	}
	var freed []string
	for _, e := range entries {
		if e.IsDir() || referenced[e.Name()] {
			continue
		}
		if err := os.Remove(filepath.Join(s.BlobsDir(), e.Name())); err != nil {
			return nil, fmt.Errorf("store: gc remove %s: %w", e.Name(), err)
		}
		freed = append(freed, e.Name())
	}
	slices.Sort(freed)
	return freed, nil
}

func (s *Store) configPath() string { return filepath.Join(s.root, "config.json") }

// LoadConfig merges config.json over types.DefaultConfig(), so an unspecified
// field keeps its default.
func (s *Store) LoadConfig() (types.Config, error) {
	cfg := types.DefaultConfig()
	if _, err := readJSON(s.configPath(), &cfg); err != nil {
		return types.Config{}, fmt.Errorf("store: read config: %w", err)
	}
	return cfg, nil
}

func (s *Store) SaveConfig(cfg types.Config) error {
	if err := writeJSONAtomic(s.configPath(), cfg); err != nil {
		return fmt.Errorf("store: write config: %w", err)
	}
	return nil
}
