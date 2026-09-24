package store

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestPresetRoundtrip(t *testing.T) {
	s := newTestStore(t)

	if got, err := s.ListPresets(); err != nil || len(got) != 0 {
		t.Fatalf("fresh store: ListPresets = %v, %v", got, err)
	}
	if _, err := s.ReadPreset("nope"); !errors.Is(err, ErrPresetNotFound) {
		t.Fatalf("ReadPreset on an empty store = %v", err)
	}

	steps, cfg := 4, 1.0
	want := Preset{
		Name: "fast-edit", Label: "Fast edit", Model: "qwen-image-edit",
		Loras: []string{"qwen-edit-lightning-4step"}, Steps: &steps, CFG: &cfg,
		Sampler: "euler", Scheduler: "simple", NegativePrompt: "blurry",
	}
	if err := s.WritePreset(want); err != nil {
		t.Fatalf("WritePreset: %v", err)
	}
	got, err := s.ReadPreset("fast-edit")
	if err != nil {
		t.Fatalf("ReadPreset: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ReadPreset = %+v, want %+v", got, want)
	}

	// Unset must survive as unset: 0 steps is a value the engine would take.
	if err := s.WritePreset(Preset{Name: "bare", Model: "flux.1-dev"}); err != nil {
		t.Fatal(err)
	}
	bare, err := s.ReadPreset("bare")
	if err != nil {
		t.Fatal(err)
	}
	if bare.Steps != nil || bare.CFG != nil {
		t.Errorf("unset steps/cfg came back set: %+v", bare)
	}

	list, err := s.ListPresets()
	if err != nil {
		t.Fatal(err)
	}
	if len(list) != 2 || list[0].Name != "bare" || list[1].Name != "fast-edit" {
		t.Fatalf("ListPresets = %+v, want sorted [bare fast-edit]", list)
	}

	if err := s.RemovePreset("fast-edit"); err != nil {
		t.Fatalf("RemovePreset: %v", err)
	}
	if _, err := s.ReadPreset("fast-edit"); !errors.Is(err, ErrPresetNotFound) {
		t.Errorf("preset survived removal: %v", err)
	}
	if err := s.RemovePreset("fast-edit"); !errors.Is(err, ErrPresetNotFound) {
		t.Errorf("removing a missing preset = %v", err)
	}
}

func TestPresetNamesAreValidatedLikeModels(t *testing.T) {
	s := newTestStore(t)
	for _, name := range []string{"", "..", "../../evil", "a/b"} {
		if err := s.WritePreset(Preset{Name: name, Model: "m"}); err == nil {
			t.Errorf("WritePreset(%q) = nil, want an error", name)
		}
		if _, err := s.ReadPreset(name); err == nil {
			t.Errorf("ReadPreset(%q) = nil, want an error", name)
		}
	}
	// A preset is callable in place of a model, so it needs one to resolve to.
	if err := s.WritePreset(Preset{Name: "empty"}); err == nil {
		t.Error("WritePreset without a model should fail")
	}
	if err := os.WriteFile(filepath.Join(s.PresetsDir(), "half.json.tmp"), []byte("{ trunc"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got, err := s.ListPresets(); err != nil || len(got) != 0 {
		t.Fatalf("ListPresets with a stray temp file = %+v, %v", got, err)
	}
}

func TestRenamePreset(t *testing.T) {
	s := newTestStore(t)
	if err := s.WritePreset(Preset{Name: "old", Model: "flux.1-dev", Label: "Old"}); err != nil {
		t.Fatal(err)
	}
	if err := s.WritePreset(Preset{Name: "taken", Model: "flux.1-dev"}); err != nil {
		t.Fatal(err)
	}

	if err := s.RenamePreset("old", "new"); err != nil {
		t.Fatalf("RenamePreset: %v", err)
	}
	if _, err := s.ReadPreset("old"); !errors.Is(err, ErrPresetNotFound) {
		t.Errorf("old preset still readable: %v", err)
	}
	got, err := s.ReadPreset("new")
	if err != nil {
		t.Fatalf("ReadPreset: %v", err)
	}
	if got.Label != "Old" || got.Model != "flux.1-dev" {
		t.Errorf("renamed preset lost fields: %+v", got)
	}
	// ReadPreset overwrites Name from the filename, so read the file itself:
	// a stale name inside would name a preset that no longer exists.
	var onDisk Preset
	if _, err := readJSON(filepath.Join(s.PresetsDir(), "new.json"), &onDisk); err != nil {
		t.Fatal(err)
	}
	if onDisk.Name != "new" {
		t.Errorf("on-disk name = %q, want \"new\"", onDisk.Name)
	}

	if err := s.RenamePreset("ghost", "x"); !errors.Is(err, ErrPresetNotFound) {
		t.Errorf("rename of a missing preset = %v", err)
	}
	if err := s.RenamePreset("new", "taken"); !errors.Is(err, ErrPresetExists) {
		t.Errorf("rename onto an existing preset = %v", err)
	}
	if _, err := s.ReadPreset("new"); err != nil {
		t.Errorf("a refused rename must leave the source alone: %v", err)
	}
}
