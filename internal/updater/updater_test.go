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
		{"1.10.0", "1.9.0", true}, // numeric, not lexical
		{"1.4", "1.3.9", true},    // a missing part is zero
		{"1.4", "1.4.0", false},
		{"", "1.4.0", false},
		{"1.4.1-beta", "1.4.0", false}, // a suffix we cannot order is never an update
		{"1.4.0.1", "1.4.0", false},
		{" v1.4.1 ", "1.4.0", true},
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
	for _, exe := range []string{"/usr/local/bin/oflux", "oflux", ""} {
		if p := AppPathFromExe(exe); p != "" {
			t.Errorf("AppPathFromExe(%q) = %q, want empty", exe, p)
		}
	}
}
