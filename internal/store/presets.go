package store

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

// ErrPresetNotFound is returned when no preset with the requested name exists.
var ErrPresetNotFound = errors.New("store: preset not found")

// ErrPresetExists is returned when a rename would overwrite another preset.
var ErrPresetExists = errors.New("store: preset already exists")

// Preset is a saved bundle of request defaults at presets/<name>.json, callable
// in place of a model name: a model plus the LoRA and the steps/cfg it needs,
// named once instead of retyped — correctly — on every request.
//
// Steps and CFG are pointers because unset must stay distinguishable from 0: an
// unset field falls through to the model's launch-time default, as an omitted
// request field does.
type Preset struct {
	Name           string   `json:"name"`
	Label          string   `json:"label,omitempty"`
	Model          string   `json:"model"`
	Loras          []string `json:"loras,omitempty"`
	Steps          *int     `json:"steps,omitempty"`
	CFG            *float64 `json:"cfg,omitempty"`
	Sampler        string   `json:"sampler,omitempty"`
	Scheduler      string   `json:"scheduler,omitempty"`
	NegativePrompt string   `json:"negative_prompt,omitempty"`
}

func (s *Store) PresetsDir() string { return filepath.Join(s.root, "presets") }

func (s *Store) presetPath(name string) (string, error) {
	return s.pathFor(s.PresetsDir(), name, jsonExt)
}

// ReadPreset returns ErrPresetNotFound if no such preset exists.
func (s *Store) ReadPreset(name string) (Preset, error) {
	path, err := s.presetPath(name)
	if err != nil {
		return Preset{}, err
	}
	var p Preset
	found, err := readJSON(path, &p)
	if err != nil {
		return Preset{}, fmt.Errorf("store: read preset %q: %w", name, err)
	}
	if !found {
		return Preset{}, fmt.Errorf("%q: %w", name, ErrPresetNotFound)
	}
	p.Name = name // the filename is what a request names, so it wins
	return p, nil
}

// WritePreset refuses a preset with no model: it would be a callable name that
// resolves to nothing.
func (s *Store) WritePreset(p Preset) error {
	path, err := s.presetPath(p.Name)
	if err != nil {
		return err
	}
	if p.Model == "" {
		return fmt.Errorf("store: preset %q has no model", p.Name)
	}
	if err := writeJSONAtomic(path, p); err != nil {
		return fmt.Errorf("store: write preset %q: %w", p.Name, err)
	}
	return nil
}

// ListPresets sorts by Name.
func (s *Store) ListPresets() ([]Preset, error) {
	entries, err := os.ReadDir(s.PresetsDir())
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("store: list presets: %w", err)
	}
	var out []Preset
	for _, e := range entries {
		// An interrupted write leaves "<name>.json.tmp" behind.
		if e.IsDir() || !strings.HasSuffix(e.Name(), jsonExt) {
			continue
		}
		name := strings.TrimSuffix(e.Name(), jsonExt)
		p, err := s.ReadPreset(name)
		if err != nil {
			if errors.Is(err, ErrPresetNotFound) {
				continue // removed while we were listing
			}
			return nil, err
		}
		out = append(out, p)
	}
	slices.SortFunc(out, func(a, b Preset) int { return strings.Compare(a.Name, b.Name) })
	return out, nil
}

// RenamePreset writes the preset under the new name and removes the old file.
// The name inside the preset is rewritten too: it is what a response echoes,
// and a stale one would name a preset that no longer exists.
func (s *Store) RenamePreset(from, to string) error {
	p, err := s.ReadPreset(from)
	if err != nil {
		return err
	}
	dst, err := s.presetPath(to)
	if err != nil {
		return err
	}
	if isFile(dst) {
		return fmt.Errorf("%q: %w", to, ErrPresetExists)
	}
	p.Name = to
	if err := s.WritePreset(p); err != nil {
		return err
	}
	return s.RemovePreset(from)
}

func (s *Store) RemovePreset(name string) error {
	path, err := s.presetPath(name)
	if err != nil {
		return err
	}
	if err := os.Remove(path); err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("%q: %w", name, ErrPresetNotFound)
		}
		return fmt.Errorf("store: remove preset %q: %w", name, err)
	}
	return nil
}
