package updater

import (
	"os"
	"path/filepath"
	"testing"
)

func mkBundle(t *testing.T, path, marker string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Join(path, "Contents", "MacOS"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(path, "Contents", "MacOS", "oflux"), []byte(marker), 0o755); err != nil {
		t.Fatal(err)
	}
}

func exeContent(t *testing.T, app string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(app, "Contents", "MacOS", "oflux"))
	if err != nil {
		return ""
	}
	return string(b)
}

// An update killed between the two renames leaves only the backup: startup must
// restore it rather than leave the user with no app.
func TestRecoverRestoresBackup(t *testing.T) {
	dir := t.TempDir()
	app := filepath.Join(dir, "oflux.app")
	mkBundle(t, app+backupSuffix, "old")

	if msg := Recover(app); msg == "" {
		t.Fatal("expected a recovery message")
	}
	if got := exeContent(t, app); got != "old" {
		t.Errorf("app executable = %q, want the restored backup", got)
	}
}

// If the swap never happened but a validated bundle was staged, use it.
func TestRecoverUsesStagedWhenNoBackup(t *testing.T) {
	dir := t.TempDir()
	app := filepath.Join(dir, "oflux.app")
	mkBundle(t, app+stageSuffix, "new")

	if msg := Recover(app); msg == "" {
		t.Fatal("expected a recovery message")
	}
	if got := exeContent(t, app); got != "new" {
		t.Errorf("app executable = %q, want the staged build", got)
	}
}

// A healthy install is left alone, and leftovers are cleaned up.
func TestRecoverNoopWhenHealthy(t *testing.T) {
	dir := t.TempDir()
	app := filepath.Join(dir, "oflux.app")
	mkBundle(t, app, "current")
	mkBundle(t, app+backupSuffix, "old")

	if msg := Recover(app); msg != "" {
		t.Errorf("expected no recovery, got %q", msg)
	}
	if got := exeContent(t, app); got != "current" {
		t.Errorf("healthy app was modified: %q", got)
	}
	if _, err := os.Stat(app + backupSuffix); !os.IsNotExist(err) {
		t.Error("leftover backup should have been cleaned up")
	}
}

// Only one updater may swap the bundle at a time.
func TestLockUpdateIsExclusive(t *testing.T) {
	dir := t.TempDir()
	app := filepath.Join(dir, "oflux.app")

	release, err := lockUpdate(app)
	if err != nil {
		t.Fatalf("first lock: %v", err)
	}
	if _, err := lockUpdate(app); err == nil {
		t.Error("second concurrent update should have been refused")
	}
	release()
	release2, err := lockUpdate(app)
	if err != nil {
		t.Fatalf("lock after release: %v", err)
	}
	release2()
}
