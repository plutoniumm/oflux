// Package updater checks GitHub Releases for a newer oflux and installs it in
// place. It downloads the release's macOS-arm64 .app zip, extracts it with
// ditto (preserving the code signature), and swaps it over the installed
// bundle. The caller then quits so launchd (KeepAlive) relaunches the new build.
package updater

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// Repo is the GitHub repository releases are published to.
const Repo = "plutoniumm/oflux"

// An update writes to /Applications, so helpers are addressed absolutely rather
// than resolved through a caller-controlled $PATH.
const (
	dittoBin    = "/usr/bin/ditto"
	codesignBin = "/usr/bin/codesign"

	stageSuffix  = ".new"         // fully-staged replacement bundle
	backupSuffix = ".bak"         // previous bundle, kept until the swap lands
	lockSuffix   = ".update.lock" // cross-process update lock
)

// Release is the subset of a GitHub release oflux uses.
type Release struct {
	Tag     string
	Version string // Tag with any leading "v" stripped
	ZipURL  string // download URL of the macOS-arm64 .app zip
}

// Latest returns the latest published (non-prerelease) release.
func Latest(ctx context.Context) (Release, error) {
	url := "https://api.github.com/repos/" + Repo + "/releases/latest"
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	req.Header.Set("Accept", "application/vnd.github+json")
	req.Header.Set("User-Agent", "oflux-updater")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return Release{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound {
		return Release{}, fmt.Errorf("no published release found for %s yet", Repo)
	}
	if resp.StatusCode != http.StatusOK {
		return Release{}, fmt.Errorf("github releases: %s", resp.Status)
	}
	var body struct {
		TagName string `json:"tag_name"`
		Assets  []struct {
			Name string `json:"name"`
			URL  string `json:"browser_download_url"`
		} `json:"assets"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return Release{}, err
	}
	rel := Release{Tag: body.TagName, Version: strings.TrimPrefix(body.TagName, "v")}
	for _, a := range body.Assets {
		if strings.HasSuffix(a.Name, "-macos-arm64.zip") {
			rel.ZipURL = a.URL
			break
		}
		if strings.HasSuffix(a.Name, ".zip") && rel.ZipURL == "" {
			rel.ZipURL = a.URL
		}
	}
	if rel.ZipURL == "" {
		return rel, fmt.Errorf("release %s has no macOS .app zip asset", rel.Tag)
	}
	return rel, nil
}

// IsNewer reports whether release version `latest` is newer than `current`
// (numeric X.Y.Z). A non-numeric current (e.g. "dev") is never updatable.
func IsNewer(latest, current string) bool {
	lc, cc := parseVer(latest), parseVer(current)
	if lc == nil || cc == nil {
		return false
	}
	for i := range 3 {
		if lc[i] != cc[i] {
			return lc[i] > cc[i]
		}
	}
	return false
}

func parseVer(v string) []int {
	v = strings.TrimPrefix(strings.TrimSpace(v), "v")
	if v == "" {
		return nil
	}
	parts := strings.SplitN(v, ".", 3)
	out := []int{0, 0, 0}
	for i := 0; i < len(parts) && i < 3; i++ {
		n, err := strconv.Atoi(strings.TrimSpace(parts[i]))
		if err != nil {
			return nil
		}
		out[i] = n
	}
	return out
}

// Apply downloads rel and swaps its oflux.app in at appPath (the installed
// bundle). On success the caller should quit; launchd relaunches the new binary.
func Apply(ctx context.Context, rel Release, appPath string) error {
	unlock, err := lockUpdate(appPath)
	if err != nil {
		return err
	}
	defer unlock()

	tmp, err := os.MkdirTemp("", "oflux-update-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmp)

	zipPath := filepath.Join(tmp, "oflux.zip")
	if err := download(ctx, rel.ZipURL, zipPath); err != nil {
		return fmt.Errorf("download: %w", err)
	}

	ext := filepath.Join(tmp, "x")
	if out, err := exec.CommandContext(ctx, "ditto", "-x", "-k", zipPath, ext).CombinedOutput(); err != nil {
		return fmt.Errorf("unzip: %v: %s", err, out)
	}
	newApp := filepath.Join(ext, "oflux.app")

	// Stage the new bundle beside the installed one (same filesystem, so the
	// swap below is a rename rather than a long copy).
	staged := appPath + stageSuffix
	_ = os.RemoveAll(staged)
	if out, err := exec.CommandContext(ctx, dittoBin, newApp, staged).CombinedOutput(); err != nil {
		_ = os.RemoveAll(staged)
		return fmt.Errorf("stage new bundle (is %s writable?): %v: %s", filepath.Dir(appPath), err, out)
	}
	// Validate BEFORE touching the working install: a build that can't run
	// would otherwise leave the user with no usable app and no way back.
	if err := validateBundle(ctx, staged, appPath); err != nil {
		_ = os.RemoveAll(staged)
		return fmt.Errorf("rejected downloaded build: %w", err)
	}

	// Swap with two renames, so a crash leaves either the old or the new bundle
	// in place — never a half-copied one. Recover() repairs the gap between
	// them. The running process keeps its (now-unlinked) binary mapped until it
	// exits, so replacing it underneath is safe.
	backup := appPath + backupSuffix
	_ = os.RemoveAll(backup)
	if err := os.Rename(appPath, backup); err != nil {
		_ = os.RemoveAll(staged)
		return fmt.Errorf("move current bundle aside: %w", err)
	}
	if err := os.Rename(staged, appPath); err != nil {
		_ = os.Rename(backup, appPath) // put the working app back
		_ = os.RemoveAll(staged)
		return fmt.Errorf("install new bundle: %w", err)
	}
	_ = os.RemoveAll(backup)
	return nil
}

// Recover repairs an interrupted update. Called at daemon startup: if the
// installed bundle is missing or unusable but a backup or staged bundle is
// present, one of them is moved into place. Returns a description of any repair.
func Recover(appPath string) string {
	if appPath == "" {
		return ""
	}
	if _, err := os.Stat(filepath.Join(appPath, "Contents", "MacOS", "oflux")); err == nil {
		_ = os.RemoveAll(appPath + stageSuffix) // leftovers from a failed attempt
		_ = os.RemoveAll(appPath + backupSuffix)
		return ""
	}
	for _, cand := range []string{appPath + backupSuffix, appPath + stageSuffix} {
		if _, err := os.Stat(filepath.Join(cand, "Contents", "MacOS", "oflux")); err != nil {
			continue
		}
		_ = os.RemoveAll(appPath)
		if err := os.Rename(cand, appPath); err == nil {
			return "recovered oflux.app from " + filepath.Base(cand) + " after an interrupted update"
		}
	}
	return ""
}

// teamID returns the Developer ID team identifier a bundle is signed with, or
// "" for an ad-hoc/unsigned bundle.
func teamID(ctx context.Context, bundle string) string {
	out, err := exec.CommandContext(ctx, codesignBin, "-dv", "--verbose=2", bundle).CombinedOutput()
	if err != nil {
		return ""
	}
	for line := range strings.SplitSeq(string(out), "\n") {
		if v, ok := strings.CutPrefix(strings.TrimSpace(line), "TeamIdentifier="); ok {
			v = strings.TrimSpace(v)
			if v == "" || v == "not set" {
				return ""
			}
			return v
		}
	}
	return ""
}

// validateBundle checks a staged bundle is genuinely runnable, and genuinely
// ours, before it replaces a working install: the executable exists, the code
// signature verifies, the signing team matches the currently-installed build,
// and it runs and reports a version.
//
// The team check is what makes auto-update safe: without it, anyone able to
// publish a release (a stolen token, a hijacked download URL) could hand every
// user arbitrary code, because a signature that merely "verifies" says nothing
// about who produced it.
func validateBundle(ctx context.Context, bundle, installed string) error {
	exe := filepath.Join(bundle, "Contents", "MacOS", "oflux")
	fi, err := os.Stat(exe)
	if err != nil {
		return fmt.Errorf("missing executable")
	}
	if fi.Mode()&0o111 == 0 {
		return fmt.Errorf("executable bit not set")
	}
	if out, err := exec.CommandContext(ctx, codesignBin, "--verify", "--strict", bundle).CombinedOutput(); err != nil {
		return fmt.Errorf("code signature does not verify: %v: %s", err, strings.TrimSpace(string(out)))
	}
	want, got := teamID(ctx, installed), teamID(ctx, bundle)
	switch {
	case want == "":
		// The running build is ad-hoc signed (a local `make install`), so there
		// is no identity to match against. Refuse rather than accept anything.
		return fmt.Errorf("this build is not Developer-ID signed; update it with `git pull && make install` instead")
	case got != want:
		return fmt.Errorf("signed by team %q, expected %q — refusing to install", got, want)
	}
	vctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	out, err := exec.CommandContext(vctx, exe, "version").CombinedOutput()
	if err != nil {
		return fmt.Errorf("new build does not run: %v: %s", err, strings.TrimSpace(string(out)))
	}
	if !strings.HasPrefix(strings.TrimSpace(string(out)), "oflux") {
		return fmt.Errorf("unexpected `version` output: %q", strings.TrimSpace(string(out)))
	}
	return nil
}

// lockUpdate serializes updates across processes: the menu bar and a terminal
// `oflux update` are separate processes, and without this one can delete the
// other's backup mid-swap.
func lockUpdate(appPath string) (func(), error) {
	lock := appPath + lockSuffix
	f, err := os.OpenFile(lock, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		if os.IsExist(err) {
			if fi, serr := os.Stat(lock); serr == nil && time.Since(fi.ModTime()) > time.Hour {
				_ = os.Remove(lock) // stale lock from a killed updater
				return lockUpdate(appPath)
			}
			return nil, fmt.Errorf("another update is already in progress")
		}
		return nil, err
	}
	_ = f.Close()
	return func() { _ = os.Remove(lock) }, nil
}

// AppPathFromExe derives the .app bundle path from an executable inside it
// (.../oflux.app/Contents/MacOS/oflux -> .../oflux.app). Returns "" if exe is
// not inside a .app bundle.
func AppPathFromExe(exe string) string {
	app := filepath.Dir(filepath.Dir(filepath.Dir(exe)))
	if strings.HasSuffix(app, ".app") {
		return app
	}
	return ""
}

func download(ctx context.Context, url, dest string) error {
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	req.Header.Set("User-Agent", "oflux-updater")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("http %s", resp.Status)
	}
	f, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = io.Copy(f, resp.Body)
	return err
}
