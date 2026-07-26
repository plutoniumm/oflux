package store

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// LoraExt is the on-disk extension for LoRA adapters. sd-server loads LoRAs
// from --lora-model-dir in safetensors format only, so every adapter is stored
// as "<name>.safetensors" and referenced by <name> in the oflux API.
const LoraExt = ".safetensors"

// ErrLoraNotFound is returned when no LoRA with the requested name is installed.
var ErrLoraNotFound = errors.New("store: lora not found")

// loraNameRe constrains LoRA names to characters that are safe both as a path
// segment and as an engine argument. A LoRA name reaches the engine as a
// filename, so anything permitting traversal ("../") or separators would let an
// API caller point the engine at an arbitrary file on disk.
var loraNameRe = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`)

// ValidLoraName reports whether name is usable as a LoRA identifier, explaining
// why if not.
func ValidLoraName(name string) error {
	if name == "" {
		return errors.New("lora name is empty")
	}
	if strings.Contains(name, "..") || !loraNameRe.MatchString(name) {
		return fmt.Errorf("invalid lora name %q: use letters, digits, '.', '_' or '-' (max 64 chars, must start alphanumeric)", name)
	}
	return nil
}

// LoraFile describes one installed LoRA adapter.
type LoraFile struct {
	Name string `json:"name"`
	Size int64  `json:"size"`
}

// LoraPath returns the on-disk path for a LoRA name, rejecting unsafe names.
func (s *Store) LoraPath(name string) (string, error) {
	if err := ValidLoraName(name); err != nil {
		return "", err
	}
	return filepath.Join(s.LorasDir(), name+LoraExt), nil
}

// LoraFileName returns the filename the engine uses to reference a LoRA. This
// is what goes into an img_gen request's lora[].path, resolved by the engine
// against --lora-model-dir.
func LoraFileName(name string) string { return name + LoraExt }

// HasLora reports whether the named LoRA is installed.
func (s *Store) HasLora(name string) bool {
	p, err := s.LoraPath(name)
	if err != nil {
		return false
	}
	info, err := os.Stat(p)
	return err == nil && !info.IsDir()
}

// ListLoras returns the installed LoRA adapters, sorted by name.
func (s *Store) ListLoras() ([]LoraFile, error) {
	entries, err := os.ReadDir(s.LorasDir())
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("store: list loras: %w", err)
	}
	var out []LoraFile
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), LoraExt) {
			continue
		}
		lf := LoraFile{Name: strings.TrimSuffix(e.Name(), LoraExt)}
		if info, err := e.Info(); err == nil {
			lf.Size = info.Size()
		}
		out = append(out, lf)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out, nil
}

// PutLora moves the file at srcPath into the LoRA directory under name,
// replacing any existing adapter with that name.
func (s *Store) PutLora(name, srcPath string) error {
	dst, err := s.LoraPath(name)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(s.LorasDir(), 0o755); err != nil {
		return fmt.Errorf("store: ensure loras dir: %w", err)
	}
	if err := os.Rename(srcPath, dst); err != nil {
		if !isCrossDevice(err) {
			return fmt.Errorf("store: install lora %q: %w", name, err)
		}
		// Cross-device: copy via a temp file so an interrupted copy never leaves
		// truncated bytes at the real name, where the engine would load them.
		tmp := dst + ".incoming"
		if cerr := copyFile(srcPath, tmp); cerr != nil {
			_ = os.Remove(tmp)
			return fmt.Errorf("store: copy lora %q: %w", name, cerr)
		}
		if rerr := os.Rename(tmp, dst); rerr != nil {
			_ = os.Remove(tmp)
			return fmt.Errorf("store: commit lora %q: %w", name, rerr)
		}
		if rerr := os.Remove(srcPath); rerr != nil && !os.IsNotExist(rerr) {
			return fmt.Errorf("store: remove lora src: %w", rerr)
		}
	}
	return nil
}

// RemoveLora deletes an installed LoRA adapter.
func (s *Store) RemoveLora(name string) error {
	p, err := s.LoraPath(name)
	if err != nil {
		return err
	}
	if err := os.Remove(p); err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("%q: %w", name, ErrLoraNotFound)
		}
		return fmt.Errorf("store: remove lora %q: %w", name, err)
	}
	return nil
}
