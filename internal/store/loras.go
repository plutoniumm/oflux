package store

import (
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"
)

// LoraExt is the on-disk extension for LoRA adapters. sd-server loads LoRAs
// from --lora-model-dir in safetensors format only, so every adapter is stored
// as "<name>.safetensors" and referenced by <name> in the oflux API.
const LoraExt = ".safetensors"

// ErrLoraNotFound is returned when no LoRA with the requested name is installed.
var ErrLoraNotFound = errors.New("store: lora not found")

// ValidLoraName is ValidName: a LoRA name reaches the engine as a filename
// resolved against --lora-model-dir, and ValidName is what keeps it from
// naming a file outside that directory.
func ValidLoraName(name string) error { return ValidName(name) }

type LoraFile struct {
	Name  string   `json:"name"`
	Size  int64    `json:"size"`
	For   []string `json:"for,omitempty"`
	Steps int      `json:"steps,omitempty"`
	CFG   float64  `json:"cfg,omitempty"`
}

// LoraMeta is the sidecar at loras/<name>.json, beside <name>.safetensors.
//
// A step-distillation adapter is only correct at the steps and cfg it was
// distilled for — applying one at the base model's defaults burns the output —
// and nothing inside the .safetensors says which those are. The registry knows
// it for the adapters it pins; one from an arbitrary repo has nowhere else to
// record it.
type LoraMeta struct {
	Name   string   `json:"name"`
	Source string   `json:"source,omitempty"` // Hugging Face repo it came from
	File   string   `json:"file,omitempty"`   // path within Source
	For    []string `json:"for,omitempty"`    // archdb arch names it is trained for
	Steps  int      `json:"steps,omitempty"`  // steps it is distilled for (0 = not a step distiller)
	CFG    float64  `json:"cfg,omitempty"`    // cfg to use with it (0 = keep the model default)
}

func (s *Store) LoraPath(name string) (string, error) {
	return s.pathFor(s.LorasDir(), name, LoraExt)
}

func (s *Store) loraMetaPath(name string) (string, error) {
	return s.pathFor(s.LorasDir(), name, jsonExt)
}

// ReadLoraMeta returns the zero value for an adapter with no sidecar: one
// dropped into the directory by hand loads perfectly well without metadata.
func (s *Store) ReadLoraMeta(name string) (LoraMeta, error) {
	path, err := s.loraMetaPath(name)
	if err != nil {
		return LoraMeta{}, err
	}
	var m LoraMeta
	found, err := readJSON(path, &m)
	if err != nil {
		return LoraMeta{}, fmt.Errorf("store: read lora metadata %q: %w", name, err)
	}
	if !found {
		return LoraMeta{}, nil
	}
	m.Name = name // the filename is the identity; the record only copies it
	return m, nil
}

func (s *Store) WriteLoraMeta(m LoraMeta) error {
	path, err := s.loraMetaPath(m.Name)
	if err != nil {
		return err
	}
	if err := writeJSONAtomic(path, m); err != nil {
		return fmt.Errorf("store: write lora metadata %q: %w", m.Name, err)
	}
	return nil
}

// LoraFileName returns the filename the engine uses to reference a LoRA. This
// is what goes into an img_gen request's lora[].path, resolved by the engine
// against --lora-model-dir.
func LoraFileName(name string) string { return name + LoraExt }

func (s *Store) HasLora(name string) bool {
	p, err := s.LoraPath(name)
	return err == nil && isFile(p)
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
		// The directory also holds the .json sidecars, and is handed to the
		// engine wholesale as --lora-model-dir: only adapters are LoRAs.
		if e.IsDir() || !strings.HasSuffix(e.Name(), LoraExt) {
			continue
		}
		lf := LoraFile{Name: strings.TrimSuffix(e.Name(), LoraExt)}
		if info, err := e.Info(); err == nil {
			lf.Size = info.Size()
		}
		// A corrupt sidecar must not hide an adapter the engine can still load.
		if meta, err := s.ReadLoraMeta(lf.Name); err == nil {
			lf.For, lf.Steps, lf.CFG = meta.For, meta.Steps, meta.CFG
		}
		out = append(out, lf)
	}
	slices.SortFunc(out, func(a, b LoraFile) int { return strings.Compare(a.Name, b.Name) })
	return out, nil
}

// PutLora replaces any existing adapter with that name.
func (s *Store) PutLora(name, srcPath string) error {
	dst, err := s.LoraPath(name)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(s.LorasDir(), 0o755); err != nil {
		return fmt.Errorf("store: ensure loras dir: %w", err)
	}
	if err := placeFile(srcPath, dst); err != nil {
		return fmt.Errorf("store: install lora %q: %w", name, err)
	}
	return nil
}

// RemoveLora deletes an installed LoRA adapter and its metadata sidecar.
func (s *Store) RemoveLora(name string) error {
	p, err := s.LoraPath(name)
	if err != nil {
		return err
	}
	rmErr := os.Remove(p)
	// Left behind, the sidecar would attach this adapter's steps and cfg to
	// whatever is next installed under the same name.
	if mp, err := s.loraMetaPath(name); err == nil {
		if err := os.Remove(mp); err != nil && !os.IsNotExist(err) {
			return fmt.Errorf("store: remove lora metadata %q: %w", name, err)
		}
	}
	if rmErr != nil {
		if os.IsNotExist(rmErr) {
			return fmt.Errorf("%q: %w", name, ErrLoraNotFound)
		}
		return fmt.Errorf("store: remove lora %q: %w", name, rmErr)
	}
	return nil
}
