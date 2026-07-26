package updater

import "testing"

func TestIsNewer(t *testing.T) {
	cases := []struct {
		latest, current string
		want            bool
	}{
		{"0.2.0", "0.1.0", true},
		{"v0.2.0", "0.1.0", true}, // leading v tolerated
		{"0.1.1", "0.1.0", true},
		{"1.0.0", "0.9.9", true},
		{"0.1.0", "0.1.0", false},
		{"0.1.0", "0.2.0", false},
		{"0.1.0", "dev", false}, // dev builds are not auto-updatable
		{"garbage", "0.1.0", false},
	}
	for _, c := range cases {
		if got := IsNewer(c.latest, c.current); got != c.want {
			t.Errorf("IsNewer(%q,%q) = %v, want %v", c.latest, c.current, got, c.want)
		}
	}
}

func TestAppPathFromExe(t *testing.T) {
	got := AppPathFromExe("/Applications/oflux.app/Contents/MacOS/oflux")
	if got != "/Applications/oflux.app" {
		t.Errorf("AppPathFromExe = %q", got)
	}
	if p := AppPathFromExe("/usr/local/bin/oflux"); p != "" {
		t.Errorf("non-bundle exe should give empty, got %q", p)
	}
}
