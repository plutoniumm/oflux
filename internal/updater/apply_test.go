package updater

import (
	"archive/zip"
	"bytes"
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// redirect is a RoundTripper sending every request to one test server, so the
// hard-coded api.github.com URL is exercised without a URL knob in production.
type redirect struct{ host string }

func (r redirect) RoundTrip(req *http.Request) (*http.Response, error) {
	req.URL.Scheme, req.URL.Host = "http", r.host
	return http.DefaultTransport.RoundTrip(req)
}

func serve(t *testing.T, h http.HandlerFunc) {
	t.Helper()
	srv := httptest.NewServer(h)
	old := http.DefaultClient
	http.DefaultClient = &http.Client{Transport: redirect{strings.TrimPrefix(srv.URL, "http://")}}
	t.Cleanup(func() { http.DefaultClient = old; srv.Close() })
}

func TestLatest(t *testing.T) {
	const arm, other = "https://dl/oflux-1.4.0-macos-arm64.zip", "https://dl/other.zip"
	asset := func(name, url string) string {
		return `{"name":"` + name + `","browser_download_url":"` + url + `"}`
	}
	cases := []struct {
		name    string
		status  int
		body    string
		wantURL string
		wantErr string
	}{
		{"arm64 asset", 200, `{"tag_name":"v1.4.0","assets":[` + asset("oflux-1.4.0.dmg", "https://dl/dmg") + `,` + asset("oflux-1.4.0-macos-arm64.zip", arm) + `]}`, arm, ""},
		// The arm64 zip wins wherever it sits: an earlier generic zip is only a fallback.
		{"arm64 beats earlier zip", 200, `{"tag_name":"v1.4.0","assets":[` + asset("other.zip", other) + `,` + asset("oflux-1.4.0-macos-arm64.zip", arm) + `]}`, arm, ""},
		{"zip fallback", 200, `{"tag_name":"v1.4.0","assets":[` + asset("other.zip", other) + `]}`, other, ""},
		{"dmg only", 200, `{"tag_name":"v1.4.0","assets":[` + asset("oflux-1.4.0.dmg", "https://dl/dmg") + `]}`, "", "no macOS .app zip"},
		{"no release yet", 404, `{}`, "", "no published release"},
		{"rate limited", 403, `{}`, "", "github releases: 403"},
		{"not json", 200, `<html>`, "", "invalid character"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			serve(t, func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/repos/"+Repo+"/releases/latest" || r.Header.Get("User-Agent") == "" {
					t.Errorf("unexpected request: %s UA=%q", r.URL.Path, r.Header.Get("User-Agent"))
				}
				w.WriteHeader(c.status)
				w.Write([]byte(c.body))
			})
			rel, err := Latest(context.Background())
			if c.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), c.wantErr) {
					t.Fatalf("err = %v, want one containing %q", err, c.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if rel.ZipURL != c.wantURL || rel.Tag != "v1.4.0" || rel.Version != "1.4.0" {
				t.Errorf("got %+v, want url %q", rel, c.wantURL)
			}
		})
	}
}

// zipBundle is a release asset: oflux.app with an executable saying marker.
func zipBundle(t *testing.T, marker string) []byte {
	t.Helper()
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	h := &zip.FileHeader{Name: "oflux.app/Contents/MacOS/oflux", Method: zip.Deflate}
	h.SetMode(0o755)
	w, err := zw.CreateHeader(h)
	if err != nil {
		t.Fatal(err)
	}
	w.Write([]byte(marker))
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// The property that matters: whatever goes wrong, the working install is still
// there afterwards, and no stage/backup/lock is left to confuse the next run.
func TestApply(t *testing.T) {
	if _, err := os.Stat(dittoBin); err != nil {
		t.Skip("needs", dittoBin)
	}
	accept := func(_ context.Context, staged, installed string) error {
		// Validation must see the staged build while the install is untouched.
		if exeContent(t, staged) != "new" || exeContent(t, installed) != "old" {
			t.Errorf("validated staged=%q installed=%q", exeContent(t, staged), exeContent(t, installed))
		}
		return nil
	}
	cases := []struct {
		name     string
		status   int
		body     []byte
		validate func(context.Context, string, string) error
		locked   bool
		want     string // executable content afterwards
		wantErr  string
	}{
		{name: "swaps in the new build", status: 200, body: zipBundle(t, "new"), validate: accept, want: "new"},
		// The real check, against a build nobody signed.
		{name: "unsigned build", status: 200, body: zipBundle(t, "new"), validate: validateBundle, want: "old", wantErr: "rejected downloaded build"},
		{name: "validation fails", status: 200, body: zipBundle(t, "new"),
			validate: func(context.Context, string, string) error { return errors.New("wrong team") }, want: "old", wantErr: "wrong team"},
		{name: "download 404", status: 404, validate: accept, want: "old", wantErr: "download: http 404"},
		{name: "not a zip", status: 200, body: []byte("<html>"), validate: accept, want: "old", wantErr: "unzip"},
		{name: "zip without the app", status: 200, body: func() []byte {
			var buf bytes.Buffer
			zw := zip.NewWriter(&buf)
			w, _ := zw.Create("README.txt")
			w.Write([]byte("hi"))
			zw.Close()
			return buf.Bytes()
		}(), validate: accept, want: "old", wantErr: "stage new bundle"},
		{name: "another update running", locked: true, validate: accept, want: "old", wantErr: "already in progress"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			app := filepath.Join(t.TempDir(), "oflux.app")
			mkBundle(t, app, "old")
			if c.locked {
				unlock, err := lockUpdate(app)
				if err != nil {
					t.Fatal(err)
				}
				defer unlock()
			}
			serve(t, func(w http.ResponseWriter, r *http.Request) {
				if c.locked {
					t.Error("downloaded while another update held the lock")
				}
				w.WriteHeader(c.status)
				w.Write(c.body)
			})
			validate = c.validate
			t.Cleanup(func() { validate = validateBundle })

			err := Apply(context.Background(), Release{ZipURL: "https://dl/oflux.zip"}, app)
			if c.wantErr == "" && err != nil {
				t.Fatal(err)
			}
			if c.wantErr != "" && (err == nil || !strings.Contains(err.Error(), c.wantErr)) {
				t.Fatalf("err = %v, want one containing %q", err, c.wantErr)
			}
			if got := exeContent(t, app); got != c.want {
				t.Errorf("installed executable = %q, want %q", got, c.want)
			}
			leftovers := []string{stageSuffix, backupSuffix}
			if !c.locked {
				leftovers = append(leftovers, lockSuffix)
			}
			for _, suffix := range leftovers {
				if _, err := os.Stat(app + suffix); !os.IsNotExist(err) {
					t.Errorf("%s left behind", filepath.Base(app)+suffix)
				}
			}
		})
	}
}

func TestValidateBundle(t *testing.T) {
	if _, err := os.Stat(codesignBin); err != nil {
		t.Skip("needs", codesignBin)
	}
	ctx := context.Background()
	dir := t.TempDir()
	installed := filepath.Join(dir, "installed.app")
	mkBundle(t, installed, "old")

	missing := filepath.Join(dir, "missing.app")
	noExec := filepath.Join(dir, "noexec.app")
	mkBundle(t, noExec, "x")
	if err := os.Chmod(bundleExe(noExec), 0o644); err != nil {
		t.Fatal(err)
	}
	unsigned := filepath.Join(dir, "unsigned.app")
	mkBundle(t, unsigned, "x")
	// Ad-hoc signed, as every local `make install` is: the signature verifies,
	// which is exactly why verifying alone must not be enough.
	adhoc := filepath.Join(dir, "adhoc.app")
	mkBundle(t, adhoc, "#!/bin/sh\necho oflux test\n")
	if _, err := run(ctx, codesignBin, "-s", "-", adhoc); err != nil {
		t.Skip("cannot ad-hoc sign here:", err)
	}

	for _, c := range []struct{ bundle, wantErr string }{
		{missing, "missing executable"},
		{noExec, "executable bit not set"},
		{unsigned, "code signature does not verify"},
		{adhoc, "not Developer-ID signed"},
	} {
		err := validateBundle(ctx, c.bundle, installed)
		if err == nil || !strings.Contains(err.Error(), c.wantErr) {
			t.Errorf("%s: err = %v, want one containing %q", filepath.Base(c.bundle), err, c.wantErr)
		}
	}
	if id := teamID(ctx, adhoc); id != "" {
		t.Errorf("ad-hoc bundle reported team %q", id)
	}
}

// A lock from an updater that was killed must not block updates forever.
func TestLockUpdateBreaksStaleLock(t *testing.T) {
	app := filepath.Join(t.TempDir(), "oflux.app")
	lock := app + lockSuffix
	if err := os.WriteFile(lock, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	old := time.Now().Add(-2 * time.Hour)
	if err := os.Chtimes(lock, old, old); err != nil {
		t.Fatal(err)
	}
	unlock, err := lockUpdate(app)
	if err != nil {
		t.Fatalf("stale lock was honoured: %v", err)
	}
	unlock()
}

func TestRecoverNothingToDo(t *testing.T) {
	if msg := Recover(""); msg != "" {
		t.Errorf("Recover(\"\") = %q", msg)
	}
	// Broken install and no candidate: nothing to move, and nothing invented.
	app := filepath.Join(t.TempDir(), "oflux.app")
	if msg := Recover(app); msg != "" {
		t.Errorf("Recover with no candidates = %q", msg)
	}
	if _, err := os.Stat(app); !os.IsNotExist(err) {
		t.Error("Recover created something out of nothing")
	}
}
