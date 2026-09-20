// Package launchd installs/removes the per-user LaunchAgent that keeps the
// oflux menu-bar app running and relaunches it at login. macOS only.
package launchd

import (
	"encoding/xml"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

// Label is the reverse-DNS LaunchAgent label. It matches CFBundleIdentifier.
const Label = "io.github.plutoniumm.oflux"

// legacyLabels are agent labels used by earlier builds. They are booted out and
// removed on install, otherwise an old agent keeps relaunching a stale copy of
// the app and the two daemons fight over the port.
var legacyLabels = []string{"ch.manav.oflux"}

// The current user's launchd GUI domain, and one agent inside it.
func guiDomain() string             { return "gui/" + strconv.Itoa(os.Getuid()) }
func serviceTarget(l string) string { return guiDomain() + "/" + l }

// launchctl reports what the command printed: its exit status is never diagnostic.
func launchctl(args ...string) error {
	out, err := exec.Command("launchctl", args...).CombinedOutput()
	if err != nil {
		return fmt.Errorf("launchctl %s: %v: %s", args[0], err, strings.TrimSpace(string(out)))
	}
	return nil
}

// plistPathFor is where label's agent plist lives for the current user.
// PlistPath is the same for our own label.
func plistPathFor(label string) string {
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		home = os.Getenv("HOME")
	}
	return filepath.Join(home, "Library", "LaunchAgents", label+".plist")
}

func PlistPath() string { return plistPathFor(Label) }

// removeLegacy unloads and deletes any superseded LaunchAgent.
func removeLegacy() {
	for _, l := range legacyLabels {
		if l == Label {
			continue
		}
		_ = launchctl("bootout", serviceTarget(l))
		_ = os.Remove(plistPathFor(l))
	}
}

// xmlStr escapes a string for inclusion in a plist <string> element. Paths can
// legitimately contain &, <, > and quotes; unescaped they produce an invalid
// plist that launchd rejects with an opaque error.
func xmlStr(s string) string {
	var b strings.Builder
	_ = xml.EscapeText(&b, []byte(s))
	return b.String()
}

// plist renders the LaunchAgent for the given executable path. It runs
// `<exe> menubar`, relaunches at login and on crash, and captures output to the
// store's log dir.
func plist(exe, logDir string) string {
	logs := xmlStr(logDir)
	return fmt.Sprintf(`<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>              <string>%s</string>
  <key>ProgramArguments</key>
  <array>
    <string>%s</string>
    <string>menubar</string>
  </array>
  <key>RunAtLoad</key>          <true/>
  <key>KeepAlive</key>          <true/>
  <key>StandardOutPath</key>    <string>%[3]s/launchd.out.log</string>
  <key>StandardErrorPath</key>  <string>%[3]s/launchd.err.log</string>
</dict>
</plist>
`, Label, xmlStr(exe), logs)
}

// writeAgent drops any superseded agent and writes our plist, returning its path.
func writeAgent(exe, logDir string) (string, error) {
	removeLegacy()
	p := PlistPath()
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return "", err
	}
	if err := os.WriteFile(p, []byte(plist(exe, logDir)), 0o644); err != nil {
		return "", err
	}
	return p, nil
}

// Install writes the LaunchAgent plist for exe and bootstraps it so the menu-bar
// app starts now and at every login. logDir receives launchd's stdout/stderr.
func Install(exe, logDir string) error {
	p, err := writeAgent(exe, logDir)
	if err != nil {
		return err
	}
	// Reload cleanly: bootout an existing instance (ignore error), then bootstrap.
	_ = launchctl("bootout", serviceTarget(Label))
	return launchctl("bootstrap", guiDomain(), p)
}

// WritePlist writes the LaunchAgent plist for exe WITHOUT bootstrapping it, so
// it takes effect at the next login. The menu-bar app uses this for first-run
// self-setup, where bootstrapping would double-launch the already-running app.
func WritePlist(exe, logDir string) error {
	_, err := writeAgent(exe, logDir)
	return err
}

// Installed reports whether the LaunchAgent plist exists.
func Installed() bool {
	_, err := os.Stat(PlistPath())
	return err == nil
}

// Uninstall stops the agent and removes its plist.
func Uninstall() error {
	removeLegacy()
	_ = launchctl("bootout", serviceTarget(Label))
	if err := os.Remove(PlistPath()); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}
