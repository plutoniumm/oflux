// Package updater checks GitHub Releases for a newer oflux and installs it in
// place. It downloads the release's macOS-arm64 .app zip, extracts it with
// ditto (preserving the code signature), and swaps it over the installed
// bundle. The caller then quits so launchd (KeepAlive) relaunches the new build.
package updater

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
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

// get issues a GET carrying the updater's User-Agent. The caller closes the body.
func get(ctx context.Context, url, accept string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "oflux-updater")
	if accept != "" {
		req.Header.Set("Accept", accept)
	}
	return http.DefaultClient.Do(req)
}

// run returns a helper's trimmed combined output, and on failure wraps it: a
// ditto/codesign exit status alone says nothing about what went wrong.
func run(ctx context.Context, bin string, args ...string) (string, error) {
	out, err := exec.CommandContext(ctx, bin, args...).CombinedOutput()
	s := strings.TrimSpace(string(out))
	if err != nil {
		return s, fmt.Errorf("%v: %s", err, s)
	}
	return s, nil
}

// bundleExe is the binary inside a .app: its presence is what separates a usable
// bundle from a half-copied one, so install and Recover both gate on it.
func bundleExe(bundle string) string {
	return filepath.Join(bundle, "Contents", "MacOS", "oflux")
}

// Latest returns the latest published (non-prerelease) release.
func Latest(ctx context.Context) (Release, error) {
	resp, err := get(ctx, "https://api.github.com/repos/"+Repo+"/releases/latest", "application/vnd.github+json")
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
		if rel.ZipURL == "" && strings.HasSuffix(a.Name, ".zip") {
			rel.ZipURL = a.URL // fallback, until the arm64 asset shows up
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
	lv, lok := parseVer(latest)
	cv, cok := parseVer(current)
	return lok && cok && slices.Compare(lv[:], cv[:]) > 0
}

func parseVer(v string) (out [3]int, ok bool) {
	v = strings.TrimPrefix(strings.TrimSpace(v), "v")
	if v == "" {
		return out, false
	}
	for i, part := range strings.SplitN(v, ".", 3) {
		n, err := strconv.Atoi(strings.TrimSpace(part))
		if err != nil {
			return out, false
		}
		out[i] = n
	}
	return out, true
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
	if _, err := run(ctx, dittoBin, "-x", "-k", zipPath, ext); err != nil {
		return fmt.Errorf("unzip: %w", err)
	}

	// Stage the new bundle beside the installed one (same filesystem, so the
	// swap below is a rename rather than a long copy).
	staged := appPath + stageSuffix
	_ = os.RemoveAll(staged)
	if _, err := run(ctx, dittoBin, filepath.Join(ext, "oflux.app"), staged); err != nil {
		_ = os.RemoveAll(staged)
		return fmt.Errorf("stage new bundle (is %s writable?): %w", filepath.Dir(appPath), err)
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
	if _, err := os.Stat(bundleExe(appPath)); err == nil {
		_ = os.RemoveAll(appPath + stageSuffix) // leftovers from a failed attempt
		_ = os.RemoveAll(appPath + backupSuffix)
		return ""
	}
	for _, cand := range []string{appPath + backupSuffix, appPath + stageSuffix} {
		if _, err := os.Stat(bundleExe(cand)); err != nil {
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
	out, err := run(ctx, codesignBin, "-dv", "--verbose=2", bundle)
	if err != nil {
		return ""
	}
	for line := range strings.SplitSeq(out, "\n") {
		if v, ok := strings.CutPrefix(strings.TrimSpace(line), "TeamIdentifier="); ok {
			if v = strings.TrimSpace(v); v == "not set" {
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
	exe := bundleExe(bundle)
	fi, err := os.Stat(exe)
	if err != nil {
		return errors.New("missing executable")
	}
	if fi.Mode()&0o111 == 0 {
		return errors.New("executable bit not set")
	}
	if _, err := run(ctx, codesignBin, "--verify", "--strict", bundle); err != nil {
		return fmt.Errorf("code signature does not verify: %w", err)
	}
	switch want, got := teamID(ctx, installed), teamID(ctx, bundle); {
	case want == "":
		// The running build is ad-hoc signed (a local `make install`), so there
		// is no identity to match against. Refuse rather than accept anything.
		return errors.New("this build is not Developer-ID signed; update it with `git pull && make install` instead")
	case got != want:
		return fmt.Errorf("signed by team %q, expected %q — refusing to install", got, want)
	}
	vctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	out, err := run(vctx, exe, "version")
	if err != nil {
		return fmt.Errorf("new build does not run: %w", err)
	}
	if !strings.HasPrefix(out, "oflux") {
		return fmt.Errorf("unexpected `version` output: %q", out)
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
			return nil, errors.New("another update is already in progress")
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
	resp, err := get(ctx, url, "")
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
	if _, err := io.Copy(f, resp.Body); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
