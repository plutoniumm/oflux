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
	case "lora", "loras":
		err = cmdLora(args)
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

func usage() { fmt.Fprint(os.Stderr, usageText) }

const usageText = `oflux — local diffusion image-editing daemon

Usage:
  oflux menubar                   run the macOS menu-bar app (hosts the daemon)
  oflux serve                     run the daemon headless
  oflux install / uninstall       add/remove the login agent + CLI symlink
  oflux update                    update to the latest GitHub release
  oflux version                   print the version
  oflux pull <name|org/repo>...   install models (curated names or Hugging Face repos)
  oflux run  <name>               install if needed, then print how to call it
  oflux list                      list installed models
  oflux ps                        show currently-loaded models
  oflux rm   <name>...            remove installed models

  oflux lora ls                   list LoRA adapters (installed and available)
  oflux lora pull <name|org/repo> install a LoRA adapter
  oflux lora rm   <name>...       remove LoRA adapters

Flags:
  pull/run: --quant <Q8_0|Q6_K|...>   quantization preference (default from config)
            --file <path-in-repo>     pin exact weights in a repo with many builds
            --control-net <org/repo>  attach a ControlNet (loaded with the model)
            --control-net-file <path> pick one from a multi-file ControlNet repo
            --as <name>               install under a different name
  lora pull: --file <path-in-repo>    pick one adapter from a multi-adapter repo
             --as <name>              install under a different name

LoRAs are applied per request, not baked into a model:
  curl :11534/v1/edit -d '{"model":"qwen-image-edit","prompt":"...","image":"<b64>",
                           "loras":[{"name":"qwen-edit-lightning-4step"}]}'
`

// ---- daemon / app ----

func cmdServe(_ []string) error {
	a, err := app.Setup()
	if err != nil {
		return err
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	return a.Serve(ctx) // prints the listen address once the port is bound

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
	p, err := parseNameQuant(args)
	if err != nil {
		return err
	}
	// Keep going after a failure so one bad name in a batch does not strand the
	// rest, then report every failure at the end.
	var failed []string
	for _, name := range p.Names {
		if len(p.Names) > 1 {
			fmt.Printf("── %s\n", name)
		}
		if err := streamPull(p, name); err != nil {
			fmt.Fprintf(os.Stderr, "error: %s: %v\n", name, err)
			failed = append(failed, name)
		}
	}
	if len(failed) > 0 {
		return fmt.Errorf("%d of %d failed: %s", len(failed), len(p.Names), strings.Join(failed, ", "))
	}
	return nil
}

func cmdRun(args []string) error {
	p, err := parseNameQuant(args)
	if err != nil {
		return err
	}
	if len(p.Names) > 1 {
		return errors.New("run takes one model; use `oflux pull` to install several")
	}
	installed, err := listModels()
	if err != nil {
		return err
	}
	name := p.Names[0]
	if p.As != "" {
		name = p.As
	}
	if !containsModel(installed, name) {
		if err := streamPull(p, p.Names[0]); err != nil {
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
		return errors.New("usage: oflux rm <name> [name...]")
	}
	base, err := daemonBaseURL()
	if err != nil {
		return err
	}
	// Remove every name given, and keep going past a failure: silently stopping
	// at the first one left the rest installed while the command looked like it
	// had done its job.
	var failed []string
	for _, name := range args {
		if err := deleteModel(base, name); err != nil {
			fmt.Fprintf(os.Stderr, "error: %s: %v\n", name, err)
			failed = append(failed, name)
			continue
		}
		fmt.Printf("removed %s\n", name)
	}
	if len(failed) > 0 {
		return fmt.Errorf("%d of %d failed: %s", len(failed), len(args), strings.Join(failed, ", "))
	}
	return nil
}

func deleteModel(base, name string) error {
	body, _ := json.Marshal(map[string]string{"name": name})
	resp, err := http.Post(base+"/api/delete", "application/json", bytes.NewReader(body))
	if err != nil {
		return daemonDownError(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return apiError(resp)
	}
	return nil
}

// ---- loras ----

func cmdLora(args []string) error {
	if len(args) == 0 {
		return errors.New("usage: oflux lora <ls|pull|rm> [args]")
	}
	sub, rest := args[0], args[1:]
	switch sub {
	case "ls", "list":
		return cmdLoraList()
	case "pull", "add":
		return cmdLoraPull(rest)
	case "rm", "delete":
		return cmdLoraRm(rest)
	default:
		return fmt.Errorf("unknown lora command %q (want ls, pull or rm)", sub)
	}
}

type loraRow struct {
	Name        string   `json:"name"`
	Installed   bool     `json:"installed"`
	Size        int64    `json:"size"`
	Archs       []string `json:"archs"`
	Steps       int      `json:"steps"`
	Description string   `json:"description"`
}

func cmdLoraList() error {
	base, err := daemonBaseURL()
	if err != nil {
		return err
	}
	var out struct {
		Loras []loraRow `json:"loras"`
	}
	if err := getJSON(base+"/api/loras", &out); err != nil {
		return err
	}
	if len(out.Loras) == 0 {
		fmt.Println("no loras available")
		return nil
	}
	fmt.Printf("%-28s %-10s %-8s %-6s %s\n", "NAME", "STATE", "SIZE", "STEPS", "FOR")
	for _, l := range out.Loras {
		state := "available"
		size := ""
		if l.Installed {
			state = "installed"
			size = humanSize(l.Size)
		}
		steps := ""
		if l.Steps > 0 {
			steps = fmt.Sprint(l.Steps)
		}
		fmt.Printf("%-28s %-10s %-8s %-6s %s\n", l.Name, state, size, steps, strings.Join(l.Archs, ","))
	}
	return nil
}

func cmdLoraPull(args []string) error {
	var name, file, as string
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--file", "-f":
			if i+1 >= len(args) {
				return errors.New("--file requires a value")
			}
			file = args[i+1]
			i++
		case "--as", "-a":
			if i+1 >= len(args) {
				return errors.New("--as requires a value")
			}
			as = args[i+1]
			i++
		default:
			if name == "" {
				name = args[i]
			}
		}
	}
	if name == "" {
		return errors.New("usage: oflux lora pull <name|org/repo> [--file <path>] [--as <name>]")
	}
	base, err := daemonBaseURL()
	if err != nil {
		return err
	}
	body, _ := json.Marshal(map[string]string{"name": name, "file": file, "as": as})
	resp, err := http.Post(base+"/api/loras/pull", "application/json", bytes.NewReader(body))
	if err != nil {
		return daemonDownError(err)
	}
	defer resp.Body.Close()
	return streamNDJSON(resp)
}

func cmdLoraRm(args []string) error {
	if len(args) < 1 {
		return errors.New("usage: oflux lora rm <name> [name...]")
	}
	base, err := daemonBaseURL()
	if err != nil {
		return err
	}
	var failed []string
	for _, name := range args {
		body, _ := json.Marshal(map[string]string{"name": name})
		resp, err := http.Post(base+"/api/loras/delete", "application/json", bytes.NewReader(body))
		if err != nil {
			return daemonDownError(err)
		}
		if resp.StatusCode != http.StatusOK {
			fmt.Fprintf(os.Stderr, "error: %s: %v\n", name, apiError(resp))
			failed = append(failed, name)
		} else {
			fmt.Printf("removed lora %s\n", name)
		}
		resp.Body.Close()
	}
	if len(failed) > 0 {
		return fmt.Errorf("%d of %d failed: %s", len(failed), len(args), strings.Join(failed, ", "))
	}
	return nil
}

func humanSize(n int64) string {
	const unit = 1000
	if n < unit {
		return fmt.Sprintf("%dB", n)
	}
	div, exp := int64(unit), 0
	for m := n / unit; m >= unit; m /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f%cB", float64(n)/float64(div), "kMGT"[exp])
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

func streamPull(p pullArgs, name string) error {
	base, err := daemonBaseURL()
	if err != nil {
		return err
	}
	body, _ := json.Marshal(p.request(name))
	resp, err := http.Post(base+"/api/pull", "application/json", bytes.NewReader(body))
	if err != nil {
		return daemonDownError(err)
	}
	defer resp.Body.Close()
	return streamNDJSON(resp)
}

// streamNDJSON prints the {"status":...} lines of a progress stream and turns
// the first {"error":...} line into the command's error.
func streamNDJSON(resp *http.Response) error {
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

// pullArgs is the parsed form of a `pull`/`run` command line. Names holds every
// positional argument, so one invocation can install several models.
type pullArgs struct {
	Names          []string
	Quant          string
	File           string
	ControlNet     string
	ControlNetFile string
	As             string
}

// request is the wire form for one model, since /api/pull installs one at a time.
func (p pullArgs) request(name string) map[string]string {
	req := map[string]string{"name": name}
	for k, v := range map[string]string{
		"quant": p.Quant, "file": p.File, "control_net": p.ControlNet,
		"control_net_file": p.ControlNetFile, "as": p.As,
	} {
		if v != "" {
			req[k] = v
		}
	}
	return req
}

func parseNameQuant(args []string) (pullArgs, error) {
	var p pullArgs
	need := func(i int, flag string) (string, error) {
		if i+1 >= len(args) {
			return "", fmt.Errorf("%s requires a value", flag)
		}
		return args[i+1], nil
	}
	for i := 0; i < len(args); i++ {
		var err error
		switch a := args[i]; a {
		case "--quant", "-q":
			p.Quant, err = need(i, a)
			i++
		case "--file", "-f":
			p.File, err = need(i, a)
			i++
		case "--control-net", "--controlnet":
			p.ControlNet, err = need(i, a)
			i++
		case "--control-net-file", "--controlnet-file":
			p.ControlNetFile, err = need(i, a)
			i++
		case "--as":
			p.As, err = need(i, a)
			i++
		default:
			if strings.HasPrefix(a, "-") {
				return pullArgs{}, fmt.Errorf("unknown flag %q", a)
			}
			p.Names = append(p.Names, a)
		}
		if err != nil {
			return pullArgs{}, err
		}
	}
	if len(p.Names) == 0 {
		return pullArgs{}, errors.New("a model name is required")
	}
	if p.ControlNetFile != "" && p.ControlNet == "" {
		return pullArgs{}, errors.New("--control-net-file needs --control-net")
	}
	// These name or reshape a single install, so they cannot be spread across
	// several: --as would give every model the same name, and --file/--control-net
	// name a path inside one specific repo.
	if len(p.Names) > 1 {
		for flag, v := range map[string]string{
			"--as": p.As, "--file": p.File, "--control-net": p.ControlNet,
		} {
			if v != "" {
				return pullArgs{}, fmt.Errorf("%s applies to a single model, but %d were given", flag, len(p.Names))
			}
		}
	}
	return p, nil
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
