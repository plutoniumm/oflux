package launchd

import (
	"strings"
	"testing"
)

func TestPlistContainsEssentials(t *testing.T) {
	exe := "/Applications/oflux.app/Contents/MacOS/oflux"
	p := plist(exe, "/Users/u/.oflux/logs")
	for _, want := range []string{
		"<string>" + Label + "</string>",
		"<string>" + exe + "</string>",
		"<string>menubar</string>",
		"<key>RunAtLoad</key>",
		"<key>KeepAlive</key>",
		"/Users/u/.oflux/logs/launchd.err.log",
	} {
		if !strings.Contains(p, want) {
			t.Errorf("plist missing %q\n---\n%s", want, p)
		}
	}
}

func TestPlistPath(t *testing.T) {
	t.Setenv("HOME", "/tmp/home")
	want := "/tmp/home/Library/LaunchAgents/" + Label + ".plist"
	if got := PlistPath(); got != want {
		t.Errorf("PlistPath() = %q, want %q", got, want)
	}
}
