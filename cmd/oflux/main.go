// Command oflux is the single binary: the menu-bar app (`oflux menubar`), the
// headless daemon (`oflux serve`), and the model manager (`pull`/`list`/`rm`/
// `ps`/`run`). Client commands talk to the local daemon on the configured port
// (default 11534).
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"oflux/internal/app"
	"oflux/internal/launchd"
	"oflux/internal/menubar"
	"oflux/internal/selfinstall"
	"oflux/internal/store"
	"oflux/internal/types"
	"oflux/internal/updater"
	"oflux/internal/version"
)

func main() {
	if len(os.Args) < 2 {
		// No subcommand. When launched as the .app bundle, Info.plist sets
		// OFLUX_LAUNCH=menubar (via LSEnvironment) so we open the menu-bar UI;
		// from a terminal the variable is unset and we print usage.
		if os.Getenv("OFLUX_LAUNCH") == "menubar" {
			if err := cmdMenubar(nil); err != nil {
				fmt.Fprintln(os.Stderr, "error:", err)
				os.Exit(1)
			}
			return
		}
		usage()
		os.Exit(2)
	}
	cmd, args := os.Args[1], os.Args[2:]
	var err error
	switch cmd {
	case "serve":
		err = cmdServe(args)
	case "menubar":
		err = cmdMenubar(args)
	case "install":
		err = cmdInstall(args)
	case "uninstall":
		err = cmdUninstall(args)
	case "update":
		err = cmdUpdate(args)
	case "version", "--version", "-v":
		fmt.Printf("oflux %s\n", version.Version)
	case "pull":
		err = cmdPull(args)
	case "run":
		err = cmdRun(args)
	case "list", "ls":
		err = cmdList(args)
	case "ps":
		err = cmdPS(args)
	case "rm", "delete":
		err = cmdRm(args)
	case "-h", "--help", "help":
		usage()
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n", cmd)
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `oflux — local diffusion image-editing daemon

Usage:
  oflux menubar                   run the macOS menu-bar app (hosts the daemon)
  oflux serve                     run the daemon headless
  oflux install / uninstall       add/remove the login agent + CLI symlink
  oflux update                    update to the latest GitHub release
  oflux version                   print the version
  oflux pull <name|org/repo>      install a model (curated name or Hugging Face repo)
  oflux run  <name>               install if needed, then print how to call it
  oflux list                      list installed models
  oflux ps                        show currently-loaded models
  oflux rm   <name>               remove an installed model

Flags:
  pull/run: --quant <Q8_0|Q6_K|...>   quantization preference (default from config)
`)
}

// ---- daemon / app ----

func cmdServe(_ []string) error {
	a, err := app.Setup()
	if err != nil {
		return err
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	fmt.Printf("oflux serving on http://%s\n", a.Addr())
	return a.Serve(ctx)
}

func cmdMenubar(_ []string) error {
	a, err := app.Setup()
	if err != nil {
		return err
	}
	return menubar.Run(a)
}

func cmdInstall(_ []string) error {
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	st, err := store.Open(os.Getenv("OFLUX_HOME"))
	if err != nil {
		return err
	}
	if err := launchd.Install(exe, st.LogsDir()); err != nil {
		return err
	}
	fmt.Println("installed LaunchAgent — the oflux menu-bar app starts at login (RunAtLoad + KeepAlive)")
	if target, onPath, err := selfinstall.LinkCLI(exe); err != nil {
		fmt.Fprintln(os.Stderr, "note: couldn't link the oflux CLI onto your PATH:", err)
	} else if onPath {
		fmt.Printf("linked CLI: %s (try: oflux pull qwen-image-edit)\n", target)
	} else {
		fmt.Printf("linked CLI: %s — add its directory to your PATH to use `oflux`\n", target)
	}
	return nil
}

func cmdUninstall(_ []string) error {
	if err := launchd.Uninstall(); err != nil {
		return err
	}
	selfinstall.UnlinkCLI()
	fmt.Println("removed LaunchAgent and CLI symlink")
	return nil
}

func cmdUpdate(_ []string) error {
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	// The CLI is usually invoked via a PATH symlink; resolve it to the real
	// binary inside the .app so we can find the bundle to replace.
	if resolved, e := filepath.EvalSymlinks(exe); e == nil {
		exe = resolved
	}
	appPath := updater.AppPathFromExe(exe)
	if appPath == "" {
		return errors.New("`oflux update` only works for the installed oflux.app")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
	defer cancel()
	rel, err := updater.Latest(ctx)
	if err != nil {
		return err
	}
	if !updater.IsNewer(rel.Version, version.Version) {
		fmt.Printf("oflux %s is already up to date\n", version.Version)
		return nil
	}
	fmt.Printf("updating %s → %s …\n", version.Version, rel.Version)
	if err := updater.Apply(ctx, rel, appPath); err != nil {
		return err
	}
	// Restart the running menu-bar app so the new binary takes over.
	_ = exec.Command("launchctl", "kickstart", "-k",
		fmt.Sprintf("gui/%d/%s", os.Getuid(), launchd.Label)).Run()
	fmt.Printf("updated to %s\n", rel.Version)
	return nil
}

// ---- client commands ----

func cmdPull(args []string) error {
	name, quant, err := parseNameQuant(args)
	if err != nil {
		return err
	}
	return streamPull(name, quant)
}

func cmdRun(args []string) error {
	name, quant, err := parseNameQuant(args)
	if err != nil {
		return err
	}
	installed, err := listModels()
	if err != nil {
		return err
	}
	if !containsModel(installed, name) {
		if err := streamPull(name, quant); err != nil {
			return err
		}
	}
	base, _ := daemonBaseURL()
	fmt.Printf("%s is ready. Try:\n  curl %s/v1/edit -d '{\"model\":\"%s\",\"prompt\":\"...\",\"image\":\"<base64>\"}'\n", name, base, name)
	return nil
}

func cmdList(_ []string) error {
	models, err := listModels()
	if err != nil {
		return err
	}
	if len(models) == 0 {
		fmt.Println("no models installed — try: oflux pull qwen-image-edit")
		return nil
	}
	fmt.Printf("%-28s %-18s %-10s %s\n", "NAME", "ARCH", "MODE", "LOADED")
	for _, m := range models {
		loaded := ""
		if m.Loaded {
			loaded = "yes"
		}
		fmt.Printf("%-28s %-18s %-10s %s\n", m.Name, m.Architecture, m.Mode, loaded)
	}
	return nil
}

func cmdPS(_ []string) error {
	base, err := daemonBaseURL()
	if err != nil {
		return err
	}
	var out struct {
		Loaded []string `json:"loaded"`
	}
	if err := getJSON(base+"/api/ps", &out); err != nil {
		return err
	}
	if len(out.Loaded) == 0 {
		fmt.Println("no models loaded")
		return nil
	}
	for _, n := range out.Loaded {
		fmt.Println(n)
	}
	return nil
}

func cmdRm(args []string) error {
	if len(args) < 1 {
		return errors.New("usage: oflux rm <name>")
	}
	base, err := daemonBaseURL()
	if err != nil {
		return err
	}
	body, _ := json.Marshal(map[string]string{"name": args[0]})
	resp, err := http.Post(base+"/api/delete", "application/json", bytes.NewReader(body))
	if err != nil {
		return daemonDownError(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return apiError(resp)
	}
	fmt.Printf("removed %s\n", args[0])
	return nil
}

// ---- daemon HTTP helpers ----

type modelRow struct {
	Name         string     `json:"name"`
	Architecture string     `json:"architecture"`
	Mode         types.Mode `json:"mode"`
	Loaded       bool       `json:"loaded"`
}

func listModels() ([]modelRow, error) {
	base, err := daemonBaseURL()
	if err != nil {
		return nil, err
	}
	var out struct {
		Models []modelRow `json:"models"`
	}
	if err := getJSON(base+"/api/tags", &out); err != nil {
		return nil, err
	}
	return out.Models, nil
}

func containsModel(ms []modelRow, name string) bool {
	for _, m := range ms {
		if m.Name == name {
			return true
		}
	}
	return false
}

func streamPull(name, quant string) error {
	base, err := daemonBaseURL()
	if err != nil {
		return err
	}
	body, _ := json.Marshal(map[string]string{"name": name, "quant": quant})
	resp, err := http.Post(base+"/api/pull", "application/json", bytes.NewReader(body))
	if err != nil {
		return daemonDownError(err)
	}
	defer resp.Body.Close()
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var msg map[string]string
		if err := json.Unmarshal([]byte(line), &msg); err != nil {
			fmt.Println(line)
			continue
		}
		if e, ok := msg["error"]; ok {
			return errors.New(e)
		}
		if s, ok := msg["status"]; ok {
			fmt.Println(s)
		}
	}
	return sc.Err()
}

func parseNameQuant(args []string) (name, quant string, err error) {
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--quant", "-q":
			if i+1 >= len(args) {
				return "", "", errors.New("--quant requires a value")
			}
			quant = args[i+1]
			i++
		default:
			if name == "" {
				name = args[i]
			}
		}
	}
	if name == "" {
		return "", "", errors.New("a model name is required")
	}
	return name, quant, nil
}

func daemonBaseURL() (string, error) {
	port := 11534
	if st, err := store.Open(os.Getenv("OFLUX_HOME")); err == nil {
		if cfg, err := st.LoadConfig(); err == nil && cfg.Port != 0 {
			port = cfg.Port
		}
	}
	return fmt.Sprintf("http://127.0.0.1:%d", port), nil
}

func getJSON(url string, v any) error {
	resp, err := http.Get(url)
	if err != nil {
		return daemonDownError(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return apiError(resp)
	}
	return json.NewDecoder(resp.Body).Decode(v)
}

func apiError(resp *http.Response) error {
	var e struct {
		Error string `json:"error"`
	}
	if json.NewDecoder(resp.Body).Decode(&e) == nil && e.Error != "" {
		return errors.New(e.Error)
	}
	return fmt.Errorf("daemon returned %s", resp.Status)
}

func daemonDownError(err error) error {
	return fmt.Errorf("cannot reach the oflux daemon (%v)\nstart it from the oflux menu-bar app, or run: oflux serve", err)
}
