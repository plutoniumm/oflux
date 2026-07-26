package registry

import (
	"slices"
	"strings"
	"testing"

	"oflux/internal/archdb"
	"oflux/internal/store"
)

func TestCuratedLorasWellFormed(t *testing.T) {
	seen := map[string]bool{}
	for _, l := range AllLoras() {
		// The name becomes a filename handed to the engine.
		if err := store.ValidLoraName(l.Name); err != nil {
			t.Errorf("%s: %v", l.Name, err)
		}
		if seen[l.Name] {
			t.Errorf("duplicate lora name %q", l.Name)
		}
		seen[l.Name] = true

		if l.Source == "" || !strings.Contains(l.Source, "/") {
			t.Errorf("%s: source %q is not an org/repo", l.Name, l.Source)
		}
		// sd-server loads adapters from --lora-model-dir in safetensors only.
		if !strings.HasSuffix(l.File, store.LoraExt) {
			t.Errorf("%s: file %q is not a %s", l.Name, l.File, store.LoraExt)
		}
		if len(l.Archs) == 0 {
			t.Errorf("%s: no target architecture", l.Name)
		}
		// An adapter aimed at an architecture the engine can't run is dead weight.
		for _, a := range l.Archs {
			if _, ok := archdb.ByName(a); !ok {
				t.Errorf("%s: unknown architecture %q", l.Name, a)
			}
		}
		// Step distillers must carry the sampling regime they need, or applying
		// one silently runs at the base model's steps and burns the output.
		if l.Steps > 0 && l.CFG <= 0 {
			t.Errorf("%s: %d-step adapter has no cfg", l.Name, l.Steps)
		}
	}
}

func TestLoraNamesSorted(t *testing.T) {
	got := LoraNames()
	if !slices.IsSorted(got) {
		t.Fatalf("LoraNames() not sorted: %v", got)
	}
	if len(got) != len(AllLoras()) {
		t.Fatalf("LoraNames() = %d entries, table has %d", len(got), len(AllLoras()))
	}
}

func TestLookupLora(t *testing.T) {
	l, ok := LookupLora("qwen-edit-lightning-4step")
	if !ok {
		t.Fatal("qwen-edit-lightning-4step should be curated")
	}
	if l.Steps != 4 || l.CFG != 1.0 {
		t.Errorf("steps/cfg = %d/%v, want 4/1", l.Steps, l.CFG)
	}
	if l.Source != "lightx2v/Qwen-Image-Edit-2511-Lightning" {
		t.Errorf("source = %q", l.Source)
	}
	if _, ok := LookupLora("nope"); ok {
		t.Error("LookupLora(nope) should miss")
	}
}

func TestLorasForArch(t *testing.T) {
	// Every curated model's architecture should be reachable: this is what
	// tells a user which adapters are worth installing for what they have.
	got := LorasForArch("qwen-image-edit")
	if len(got) == 0 {
		t.Fatal("qwen-image-edit should have curated adapters")
	}
	for _, l := range got {
		if !slices.Contains(l.Archs, "qwen-image-edit") {
			t.Errorf("%s does not target qwen-image-edit", l.Name)
		}
	}
	if len(LorasForArch("z-image")) != 0 {
		t.Error("z-image has no curated adapters; expected an empty result")
	}
}
