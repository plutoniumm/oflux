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

// Label is the reverse-DNS LaunchAgent label.
const Label = "ch.manav.oflux"

// PlistPath is where the agent plist lives for the current user.
func PlistPath() string {
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		home = os.Getenv("HOME")
	}
	return filepath.Join(home, "Library", "LaunchAgents", Label+".plist")
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
  <key>StandardOutPath</key>    <string>%s/launchd.out.log</string>
  <key>StandardErrorPath</key>  <string>%s/launchd.err.log</string>
</dict>
</plist>
`, Label, xmlStr(exe), xmlStr(logDir), xmlStr(logDir))
}

// Install writes the LaunchAgent plist for exe and bootstraps it so the menu-bar
// app starts now and at every login. logDir receives launchd's stdout/stderr.
func Install(exe, logDir string) error {
	p := PlistPath()
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(p, []byte(plist(exe, logDir)), 0o644); err != nil {
		return err
	}
	// Reload cleanly: bootout an existing instance (ignore error), then bootstrap.
	target := "gui/" + strconv.Itoa(os.Getuid())
	_ = exec.Command("launchctl", "bootout", target+"/"+Label).Run()
	if out, err := exec.Command("launchctl", "bootstrap", target, p).CombinedOutput(); err != nil {
		return fmt.Errorf("launchctl bootstrap: %v: %s", err, out)
	}
	return nil
}

// WritePlist writes the LaunchAgent plist for exe WITHOUT bootstrapping it, so
// it takes effect at the next login. The menu-bar app uses this for first-run
// self-setup, where bootstrapping would double-launch the already-running app.
func WritePlist(exe, logDir string) error {
	p := PlistPath()
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}
	return os.WriteFile(p, []byte(plist(exe, logDir)), 0o644)
}

// Installed reports whether the LaunchAgent plist exists.
func Installed() bool {
	_, err := os.Stat(PlistPath())
	return err == nil
}

// Uninstall stops the agent and removes its plist.
func Uninstall() error {
	target := "gui/" + strconv.Itoa(os.Getuid())
	_ = exec.Command("launchctl", "bootout", target+"/"+Label).Run()
	if err := os.Remove(PlistPath()); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}
