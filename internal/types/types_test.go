package types

import "testing"

// A hybrid must satisfy both capability checks; the single-purpose modes must
// satisfy exactly one. Getting this wrong either hides half a model (FLUX.2 was
// listed generate-only despite editing fine) or routes a request the engine
// cannot serve.
func TestModeCapabilities(t *testing.T) {
	for _, tc := range []struct {
		mode                 Mode
		canEdit, canGenerate bool
	}{
		{ModeEdit, true, false},
		{ModeGenerate, false, true},
		{ModeBoth, true, true},
		{Mode("nonsense"), false, false},
		{Mode(""), false, false},
	} {
		if got := tc.mode.CanEdit(); got != tc.canEdit {
			t.Errorf("%q.CanEdit() = %v, want %v", tc.mode, got, tc.canEdit)
		}
		if got := tc.mode.CanGenerate(); got != tc.canGenerate {
			t.Errorf("%q.CanGenerate() = %v, want %v", tc.mode, got, tc.canGenerate)
		}
	}
}
