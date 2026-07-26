//go:build darwin

// Package menubar runs the oflux macOS menu-bar app: it hosts the daemon
// in-process and presents a small tray UI (status, installed models with a
// loaded-checkmark, start-at-login, quit). macOS only (cgo/Cocoa via systray).
package menubar

import (
	"context"
	_ "embed"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"fyne.io/systray"

	"oflux/internal/app"
	"oflux/internal/launchd"
	"oflux/internal/selfinstall"
	"oflux/internal/updater"
	"oflux/internal/version"
)

// iconPNG is the oflux fox (color, transparent background) shown in the menu bar.
//
//go:embed icon.png
var iconPNG []byte

// maxModelSlots is the number of reusable menu rows for installed models.
const maxModelSlots = 24

type tray struct {
	app       *app.App
	cancel    context.CancelFunc
	appPath   string // installed oflux.app path ("" if not a bundle)
	updatable bool   // running as a versioned .app (so self-update is possible)

	mStatus *systray.MenuItem
	mLogs   *systray.MenuItem
	mLogin  *systray.MenuItem
	mUpdate *systray.MenuItem
	mQuit   *systray.MenuItem

	mu      sync.Mutex
	items   []*systray.MenuItem
	models  []string         // model name shown in each slot ("" = hidden)
	pending *updater.Release // a newer release found by the update check
}

// Run hosts the daemon and blocks running the menu-bar UI until the user quits.
// It must be called on the main goroutine (systray requirement).
func Run(a *app.App) error {
	// Single instance: if a daemon is already serving, don't start a second
	// menu-bar icon (e.g. double-launch alongside the login agent).
	if daemonAlreadyServing(a.Addr()) {
		fmt.Fprintln(os.Stderr, "oflux is already running")
		return nil
	}
	exe, _ := os.Executable()
	if resolved, e := filepath.EvalSymlinks(exe); e == nil {
		exe = resolved
	}
	// First-run self-setup: register the login item (takes effect next login;
	// no bootstrap here so we don't double-launch) and link the CLI onto PATH.
	ensureSelfInstalled(a, exe)

	ctx, cancel := context.WithCancel(context.Background())
	t := &tray{app: a, cancel: cancel, appPath: updater.AppPathFromExe(exe)}
	t.updatable = t.appPath != "" && version.Version != "dev"
	go func() {
		if err := a.Serve(ctx); err != nil {
			fmt.Fprintln(os.Stderr, "oflux serve:", err)
		}
	}()
	// Reap engines on a clean stop (launchd sends SIGTERM before SIGKILL on
	// bootout); the startup reap is the backstop for hard kills. launchd only
	// grants a short grace period, so cap our own cleanup rather than risk
	// being SIGKILLed mid-shutdown with engines still alive.
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGTERM, syscall.SIGINT)
		<-sigCh
		cancel() // let the HTTP server drain in parallel
		done := make(chan struct{})
		go func() { a.Sup.Shutdown(); close(done) }()
		select {
		case <-done:
		case <-time.After(8 * time.Second):
		}
		os.Exit(0)
	}()
	systray.Run(t.onReady, func() { cancel() })
	return nil
}

// daemonAlreadyServing reports whether an oflux daemon already answers on addr.
func daemonAlreadyServing(addr string) bool {
	c := &http.Client{Timeout: 500 * time.Millisecond}
	resp, err := c.Get("http://" + addr + "/healthz")
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

// ensureSelfInstalled registers the login item + CLI symlink if not already set.
// exe must be the resolved path to the real binary (not a PATH symlink).
func ensureSelfInstalled(a *app.App, exe string) {
	if !launchd.Installed() {
		_ = launchd.WritePlist(exe, a.Store.LogsDir())
	}
	_, _, _ = selfinstall.LinkCLI(exe)
}

func (t *tray) onReady() {
	systray.SetIcon(iconPNG)
	systray.SetTooltip("oflux — local diffusion image editing")

	t.mStatus = systray.AddMenuItem("starting…", "")
	t.mStatus.Disable()
	systray.AddSeparator()

	t.items = make([]*systray.MenuItem, maxModelSlots)
	t.models = make([]string, maxModelSlots)
	for i := range t.items {
		it := systray.AddMenuItem("", "click to unload")
		it.Hide()
		t.items[i] = it
		go t.watchSlot(i)
	}

	systray.AddSeparator()
	t.mLogs = systray.AddMenuItem("Open logs folder", "")
	t.mLogin = systray.AddMenuItemCheckbox("Start at login", "", launchdInstalled())
	t.mUpdate = systray.AddMenuItem("Check for updates…", "")
	if !t.updatable {
		t.mUpdate.SetTitle("oflux " + version.Version)
		t.mUpdate.Disable()
	}
	systray.AddSeparator()
	t.mQuit = systray.AddMenuItem("Quit oflux", "")

	go t.watchControls()
	if t.updatable {
		go t.updateLoop()
	}

	t.refresh()
	go t.loop()
}

// loop refreshes the menu periodically so status/loaded state stay current.
func (t *tray) loop() {
	tick := time.NewTicker(3 * time.Second)
	defer tick.Stop()
	for range tick.C {
		t.refresh()
	}
}

func (t *tray) refresh() {
	models, _ := t.app.Store.ListManifests()
	loaded := map[string]bool{}
	for _, n := range t.app.Sup.Loaded() {
		loaded[n] = true
	}

	switch {
	case len(loaded) > 0:
		names := make([]string, 0, len(loaded))
		for n := range loaded {
			names = append(names, n)
		}
		t.mStatus.SetTitle("serving: " + strings.Join(names, ", "))
	case len(models) == 0:
		t.mStatus.SetTitle("idle — no models (pull one from the CLI)")
	default:
		t.mStatus.SetTitle(fmt.Sprintf("idle — %d model(s)", len(models)))
	}

	t.mu.Lock()
	defer t.mu.Unlock()
	for i, it := range t.items {
		if i < len(models) {
			m := models[i]
			t.models[i] = m.Name
			label := m.Name + "  (" + string(m.Mode) + ")"
			it.SetTitle(label)
			if loaded[m.Name] {
				it.Check()
			} else {
				it.Uncheck()
			}
			it.Show()
		} else {
			t.models[i] = ""
			it.Hide()
		}
	}
}

// watchSlot unloads the model in slot i when its row is clicked.
func (t *tray) watchSlot(i int) {
	for range t.items[i].ClickedCh {
		t.mu.Lock()
		name := t.models[i]
		t.mu.Unlock()
		if name != "" {
			_ = t.app.Sup.Unload(name)
			t.refresh()
		}
	}
}

func (t *tray) watchControls() {
	for {
		select {
		case <-t.mLogs.ClickedCh:
			_ = exec.Command("open", t.app.Store.LogsDir()).Run()
		case <-t.mLogin.ClickedCh:
			t.toggleLogin()
		case <-t.mUpdate.ClickedCh:
			t.onUpdateClick()
		case <-t.mQuit.ClickedCh:
			t.quit()
			return
		}
	}
}

// quit stops the engines and really exits. Two things have to happen here that
// a bare systray.Quit() would skip: the sd-server children must be killed (they
// outlive their parent and hold many GB), and the login agent must be booted
// out — its KeepAlive=true would otherwise relaunch us within a second, making
// "Quit" look broken.
func (t *tray) quit() {
	if launchdInstalled() {
		_ = exec.Command("launchctl", "bootout",
			fmt.Sprintf("gui/%d/%s", os.Getuid(), launchd.Label)).Run()
	}
	t.cancel()
	t.app.Sup.Shutdown() // waits for the engine processes to exit
	systray.Quit()
}

func (t *tray) toggleLogin() {
	if t.mLogin.Checked() {
		if err := launchd.Uninstall(); err == nil {
			t.mLogin.Uncheck()
		}
		return
	}
	exe, err := os.Executable()
	if err != nil {
		return
	}
	if err := launchd.Install(exe, t.app.Store.LogsDir()); err == nil {
		t.mLogin.Check()
	}
}

func launchdInstalled() bool {
	_, err := os.Stat(launchd.PlistPath())
	return err == nil
}

// updateLoop checks for a newer release ~30s after launch, then daily.
func (t *tray) updateLoop() {
	time.Sleep(30 * time.Second)
	t.checkUpdate(false)
	tick := time.NewTicker(24 * time.Hour)
	defer tick.Stop()
	for range tick.C {
		t.checkUpdate(false)
	}
}

// onUpdateClick applies a pending update if one is known, else checks now.
func (t *tray) onUpdateClick() {
	t.mu.Lock()
	pend := t.pending
	t.mu.Unlock()
	// Both branches run off the control loop: applyUpdate can take many minutes,
	// and systray drops clicks that arrive while nobody is receiving — a
	// synchronous call would make Quit / Open logs silently inert meanwhile.
	if pend != nil {
		go t.applyUpdate(*pend)
	} else {
		go t.checkUpdate(true)
	}
}

// checkUpdate queries GitHub for a newer release and reflects it in the menu.
func (t *tray) checkUpdate(manual bool) {
	if !t.updatable {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	rel, err := updater.Latest(ctx)
	if err != nil {
		if manual {
			t.mUpdate.SetTitle("Update check failed")
		}
		return
	}
	if updater.IsNewer(rel.Version, version.Version) {
		r := rel
		t.mu.Lock()
		t.pending = &r
		t.mu.Unlock()
		t.mUpdate.SetTitle("Install update v" + rel.Version)
	} else {
		t.mu.Lock()
		t.pending = nil
		t.mu.Unlock()
		if manual {
			t.mUpdate.SetTitle("oflux " + version.Version + " — up to date")
		}
	}
}

// applyUpdate downloads + swaps in the release, then quits so launchd
// (KeepAlive) relaunches the new build.
func (t *tray) applyUpdate(rel updater.Release) {
	t.mUpdate.SetTitle("Downloading update…")
	t.mUpdate.Disable()
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
	defer cancel()
	if err := updater.Apply(ctx, rel, t.appPath); err != nil {
		fmt.Fprintln(os.Stderr, "update failed:", err)
		t.mUpdate.SetTitle("Update failed — click to retry")
		t.mUpdate.Enable()
		return
	}
	// Exit so the new build takes over. KeepAlive relaunches us; kickstart is a
	// belt-and-braces nudge for installs whose agent was written but never
	// bootstrapped (in which case KeepAlive wouldn't fire).
	t.cancel()
	t.app.Sup.Shutdown()
	_ = exec.Command("launchctl", "kickstart", "-k",
		fmt.Sprintf("gui/%d/%s", os.Getuid(), launchd.Label)).Start()
	systray.Quit()
}
