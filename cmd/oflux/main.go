// Command oflux is the single binary: the menu-bar app (`oflux menubar`), the
// headless daemon (`oflux serve`), and the model manager (`pull`/`list`/`rm`/
// `ps`/`run`). Client commands talk to the local daemon (default port 11534).
package main

import (
	"bufio"
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"oflux/internal/app"
	"oflux/internal/launchd"
	"oflux/internal/menubar"
	"oflux/internal/selfinstall"
	"oflux/internal/server"
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
	name, args := os.Args[1], os.Args[2:]
	c, ok := lookup(name)
	if !ok {
		fmt.Fprintf(os.Stderr, "unknown command %q\n", name)
		usage()
		os.Exit(2)
	}
	if err := c.run(args); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

// command is one CLI verb; dispatch and help are generated from the one table.
type command struct {
	name    string
	aliases []string // other accepted spellings
	args    string   // argument spelling, for help
	summary string
	run     func([]string) error
}

// A func, not a var: the table names cmdHelp, which renders it — a var cycles.
func commandTable() []command {
	return []command{
		{name: "menubar", summary: "run the macOS menu-bar app (hosts the daemon)", run: cmdMenubar},
		{name: "serve", summary: "run the daemon headless", run: cmdServe},
		{name: "install", summary: "add the login agent + CLI symlink", run: cmdInstall},
		{name: "uninstall", summary: "remove the login agent + CLI symlink", run: cmdUninstall},
		{name: "update", summary: "update to the latest GitHub release", run: cmdUpdate},
		{name: "version", aliases: []string{"--version", "-v"}, summary: "print the version", run: cmdVersion},

		{name: "pull", args: "<name|org/repo>...", summary: "install models (curated names or Hugging Face repos)", run: cmdPull},
		{name: "run", args: "<name>", summary: "install if needed, then print how to call it", run: cmdRun},
		{name: "list", aliases: []string{"ls"}, summary: "list installed models and presets", run: cmdList},
		{name: "ps", summary: "show currently-loaded models", run: cmdPS},
		{name: "rm", aliases: []string{"delete"}, args: "<name>...", summary: "remove installed models", run: cmdRm},

		{name: "lora", aliases: []string{"loras"}, args: "<ls|pull|rm>", summary: "manage LoRA adapters", run: cmdLora},
		{name: "preset", aliases: []string{"presets"}, args: "<add|ls|rm>", summary: "name a model + LoRAs + sampling as one callable model", run: cmdPreset},
		{name: "job", aliases: []string{"jobs"}, args: "[stop] <id>...", summary: "show or abort an in-flight generation", run: cmdJob},

		{name: "help", aliases: []string{"--help", "-h"}, summary: "print this message", run: cmdHelp},
	}
}

func lookup(name string) (command, bool) {
	for _, c := range commandTable() {
		if c.name == name || slices.Contains(c.aliases, name) {
			return c, true
		}
	}
	return command{}, false
}

func (c command) invocation() string {
	spellings := []string{c.name}
	for _, a := range c.aliases {
		if !strings.HasPrefix(a, "-") {
			spellings = append(spellings, a)
		}
	}
	s := strings.Join(spellings, "|")
	if c.args != "" {
		s += " " + c.args
	}
	return s
}

func cmdVersion([]string) error { fmt.Printf("oflux %s\n", version.Version); return nil }

func cmdHelp([]string) error { usage(); return nil }

func usage() { fmt.Fprint(os.Stderr, usageText()) }

func usageText() string {
	cmds := commandTable()
	width := 0
	for _, c := range cmds {
		width = max(width, len(c.invocation()))
	}
	var b strings.Builder
	b.WriteString("oflux — local diffusion image-editing daemon\n\nUsage:\n")
	for _, c := range cmds {
		fmt.Fprintf(&b, "  oflux %-*s  %s\n", width, c.invocation(), c.summary)
	}
	b.WriteString(flagHelp)
	return b.String()
}

const flagHelp = `
Flags:
  pull/run:   --quant <Q8_0|Q6_K|...>   quantization preference (default from config)
              --file <path-in-repo>     pin exact weights in a repo with many builds
              --control-net <org/repo>  attach a ControlNet (loaded with the model)
              --control-net-file <path> pick one from a multi-file ControlNet repo
              --as <name>               install under a different name
  run:        --keep-alive <dur>        how long the model stays loaded after a
                                        request ("10m"); sent as keep_alive
  lora pull:  --file <path-in-repo>     pick one adapter from a multi-adapter repo
              --as <name>               install under a different name
              --for <model-or-arch>     what the adapter is for (repeatable)
              --steps N  --cfg X        the sampling regime the adapter expects
  preset add: --model <name>            the model the preset calls (required)
              --lora <name>             adapter to apply (repeatable)
              --steps N  --cfg X        sampling overrides
              --sampler / --scheduler / --negative-prompt / --label

LoRAs are applied per request, not baked into a model — or name a combination
once with ` + "`oflux preset add`" + ` and call the preset like any other model:
  curl :11534/v1/edit -d '{"model":"qwe-2511","prompt":"...","image":"<b64>",
                           "loras":[{"name":"qwen-edit-lightning-4step"}],
                           "keep_alive":"10m"}'
`

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
		fmt.Printf("linked CLI: %s (try: oflux pull qwe-2511)\n", target)
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

func cmdPull(args []string) error {
	p, err := parseNameQuant(args)
	if err != nil {
		return err
	}
	if p.Fields["keep_alive"] != "" {
		return errors.New("--keep-alive applies to generation, not to pull; pass it to `oflux run` or in the request body")
	}
	return eachName(p.Names, func(name string) error {
		if len(p.Names) > 1 {
			fmt.Printf("── %s\n", name)
		}
		return postStream("/api/pull", p.request(name))
	})
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
	name := cmp.Or(p.Fields["as"], p.Names[0])
	if !slices.ContainsFunc(installed, func(m server.ModelRow) bool { return m.Name == name }) {
		if err := postStream("/api/pull", p.request(p.Names[0])); err != nil {
			return err
		}
	}
	// Generation happens over HTTP, so --keep-alive is shown where it is sent.
	keepAlive := ""
	if d := p.Fields["keep_alive"]; d != "" {
		keepAlive = fmt.Sprintf(",\"keep_alive\":\"%s\"", d)
	}
	fmt.Printf("%s is ready. Try:\n  curl %s/v1/edit -d '{\"model\":\"%s\",\"prompt\":\"...\",\"image\":\"<base64>\"%s}'\n",
		name, daemonBase(), name, keepAlive)
	return nil
}

func cmdList(_ []string) error {
	models, err := listModels()
	if err != nil {
		return err
	}
	if len(models) == 0 {
		fmt.Println("no models installed — try: oflux pull qwe-2511")
		return nil
	}
	rows := [][]string{{"NAME", "ARCH", "MODE", "LOADED", "LABEL"}}
	for _, m := range models {
		loaded := ""
		if m.Loaded {
			loaded = "yes"
		}
		arch, label := m.Architecture, m.Label
		if m.Preset {
			// A preset borrows its model's arch, so say which rows are presets.
			arch = cmp.Or(arch, "preset")
			label = strings.TrimSpace(label + " (preset)")
		}
		rows = append(rows, []string{m.Name, arch, string(m.Mode), loaded, label})
	}
	printTable([]int{28, 18, 10, 6}, rows)
	return nil
}

func cmdPS(_ []string) error {
	loaded, err := getList[string]("/api/ps", "loaded")
	if err != nil {
		return err
	}
	if len(loaded) == 0 {
		fmt.Println("no models loaded")
		return nil
	}
	for _, n := range loaded {
		fmt.Println(n)
	}
	return nil
}

func cmdRm(args []string) error { return removeEach("/api/delete", "", args) }

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
		return removeEach("/api/loras/delete", "lora ", rest)
	default:
		return fmt.Errorf("unknown lora command %q (want ls, pull or rm)", sub)
	}
}

func cmdLoraList() error {
	loras, err := getList[server.LoraRow]("/api/loras", "loras")
	if err != nil {
		return err
	}
	if len(loras) == 0 {
		fmt.Println("no loras available")
		return nil
	}
	rows := [][]string{{"NAME", "STATE", "SIZE", "STEPS", "CFG", "FOR"}}
	for _, l := range loras {
		state, size := "available", ""
		if l.Installed {
			state, size = "installed", humanSize(l.Size)
		}
		steps, cfgv := "", ""
		if l.Steps > 0 {
			steps = strconv.Itoa(l.Steps)
		}
		if l.CFG > 0 {
			cfgv = fmt.Sprintf("%g", l.CFG)
		}
		rows = append(rows, []string{l.Name, state, size, steps, cfgv, strings.Join(l.For, ",")})
	}
	printTable([]int{28, 10, 8, 6, 5}, rows)
	return nil
}

// --for/--steps/--cfg are provenance for the adapter's sidecar: an adapter
// outside the curated table has nowhere else to record what it is for or needs.
var loraPullFlags = map[string]string{
	"--file": "file", "-f": "file",
	"--as": "as", "-a": "as",
	"--for":   "for",
	"--steps": "steps",
	"--cfg":   "cfg", "--guidance": "cfg",
}

func cmdLoraPull(args []string) error {
	names, set, err := parseFlagsMulti(args, loraPullFlags)
	if err != nil {
		return err
	}
	if len(names) == 0 {
		return errors.New("usage: oflux lora pull <name|org/repo> [--file <path>] [--as <name>] [--for <model-or-arch>] [--steps N] [--cfg X]")
	}
	req := map[string]any{"name": names[0]}
	for _, field := range []string{"file", "as"} {
		if v := last(set[field]); v != "" {
			req[field] = v
		}
	}
	if v := set["for"]; len(v) > 0 {
		req["for"] = v
	}
	steps, err := intFlag(set, "steps")
	if err != nil {
		return err
	}
	if steps != nil {
		req["steps"] = *steps
	}
	cfg, err := floatFlag(set, "cfg")
	if err != nil {
		return err
	}
	if cfg != nil {
		req["cfg"] = *cfg
	}
	return postStream("/api/loras/pull", req)
}

func cmdPreset(args []string) error {
	if len(args) == 0 {
		return errors.New("usage: oflux preset <add|ls|rm> [args]")
	}
	sub, rest := args[0], args[1:]
	switch sub {
	case "ls", "list":
		return cmdPresetList()
	case "add", "create", "set":
		return cmdPresetAdd(rest)
	case "rm", "delete":
		return removeEach("/api/presets/delete", "preset ", rest)
	default:
		return fmt.Errorf("unknown preset command %q (want add, ls or rm)", sub)
	}
}

var presetFlags = map[string]string{
	"--model": "model", "-m": "model",
	"--lora": "lora", "-l": "lora",
	"--steps": "steps",
	"--cfg":   "cfg", "--guidance": "cfg",
	"--sampler": "sampler", "--scheduler": "scheduler",
	"--negative-prompt": "negative_prompt", "--negative": "negative_prompt",
	"--label": "label",
}

func cmdPresetAdd(args []string) error {
	names, set, err := parseFlagsMulti(args, presetFlags)
	if err != nil {
		return err
	}
	if len(names) != 1 {
		return errors.New("usage: oflux preset add <name> --model <model> [--lora <name>]... [--steps N] [--cfg X] [--label \"...\"]")
	}
	p := store.Preset{
		Name:           names[0],
		Label:          last(set["label"]),
		Model:          last(set["model"]),
		Loras:          set["lora"],
		Sampler:        last(set["sampler"]),
		Scheduler:      last(set["scheduler"]),
		NegativePrompt: last(set["negative_prompt"]),
	}
	if p.Model == "" {
		return errors.New("preset add needs --model <name>")
	}
	if p.Steps, err = intFlag(set, "steps"); err != nil {
		return err
	}
	if p.CFG, err = floatFlag(set, "cfg"); err != nil {
		return err
	}
	resp, err := post("/api/presets/create", p)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	var out struct {
		Note string `json:"note"`
	}
	_ = json.NewDecoder(resp.Body).Decode(&out)
	fmt.Printf("saved preset %s → %s\n", p.Name, p.Model)
	if out.Note != "" {
		fmt.Println(out.Note)
	}
	return nil
}

func cmdPresetList() error {
	presets, err := getList[store.Preset]("/api/presets", "presets")
	if err != nil {
		return err
	}
	if len(presets) == 0 {
		fmt.Println("no presets — try: oflux preset add fast --model qwe-2511 --lora qwen-edit-lightning-4step --steps 4 --cfg 1")
		return nil
	}
	rows := [][]string{{"NAME", "MODEL", "LORAS", "STEPS", "CFG", "LABEL"}}
	for _, p := range presets {
		rows = append(rows, []string{
			p.Name, p.Model, strings.Join(p.Loras, ","), optNum(p.Steps), optNum(p.CFG), p.Label,
		})
	}
	printTable([]int{20, 24, 26, 5, 5}, rows)
	return nil
}

// A long edit's only way out was killing the daemon, taking every other job with it.
func cmdJob(args []string) error {
	if len(args) == 0 {
		return errors.New("usage: oflux job <id> | oflux job stop <id>...")
	}
	if sub := args[0]; sub == "stop" || sub == "cancel" || sub == "rm" {
		ids := args[1:]
		if len(ids) == 0 {
			return errors.New("usage: oflux job stop <id>...")
		}
		return eachName(ids, func(id string) error {
			if err := del(jobPath(id)); err != nil {
				return err
			}
			fmt.Printf("cancelled %s\n", id)
			return nil
		})
	}
	return eachName(args, showJob)
}

func jobPath(id string) string { return "/v1/jobs/" + url.PathEscape(id) }

func showJob(id string) error {
	// A finished job answers with its images instead of a progress line.
	var j server.StreamLine
	if err := getJSON(jobPath(id), &j); err != nil {
		return err
	}
	line := cmp.Or(j.Status, "done")
	if j.Model != "" {
		line += "  " + j.Model
	}
	if j.Total > 0 {
		line += fmt.Sprintf("  step %d/%d", j.Step, j.Total)
	}
	if j.Elapsed > 0 {
		line += fmt.Sprintf("  %.0fs", j.Elapsed)
	}
	fmt.Printf("%-16s %s\n", cmp.Or(j.Job, id), line)
	return nil
}

// label names the thing removed ("" for models, "lora " for adapters).
func removeEach(path, label string, names []string) error {
	if len(names) == 0 {
		return fmt.Errorf("usage: oflux %srm <name> [name...]", label)
	}
	return eachName(names, func(name string) error {
		resp, err := post(path, map[string]string{"name": name})
		if err != nil {
			return err
		}
		resp.Body.Close()
		fmt.Printf("removed %s%s\n", label, name)
		return nil
	})
}

// eachName keeps going past a failure and reports them all at the end: stopping
// at the first stranded the rest of a batch while still looking successful.
func eachName(names []string, fn func(string) error) error {
	var failed []string
	for _, name := range names {
		if err := fn(name); err != nil {
			fmt.Fprintf(os.Stderr, "error: %s: %v\n", name, err)
			failed = append(failed, name)
		}
	}
	if len(failed) > 0 {
		return fmt.Errorf("%d of %d failed: %s", len(failed), len(names), strings.Join(failed, ", "))
	}
	return nil
}

// widths has one entry fewer than a row: the last column is free-form.
func printTable(widths []int, rows [][]string) {
	var f strings.Builder
	for _, w := range widths {
		fmt.Fprintf(&f, "%%-%ds ", w)
	}
	f.WriteString("%s")
	format := f.String()
	for _, row := range rows {
		cells := make([]any, len(row))
		for i, c := range row {
			cells[i] = c
		}
		fmt.Println(strings.TrimRight(fmt.Sprintf(format, cells...), " "))
	}
}

func optNum[T int | float64](p *T) string {
	if p == nil {
		return ""
	}
	return fmt.Sprintf("%v", *p)
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

// The server's own row types, not copies: a renamed field would decode here as
// an empty column nobody notices.
func listModels() ([]server.ModelRow, error) {
	return getList[server.ModelRow]("/api/tags", "models")
}

func getList[T any](path, key string) ([]T, error) {
	var out map[string][]T
	if err := getJSON(path, &out); err != nil {
		return nil, err
	}
	return out[key], nil
}

// OFLUX_HOST points the CLI at a daemon elsewhere, and is the seam its own tests
// aim at an httptest server; otherwise the port comes from the store's config.
func daemonBase() string {
	if h := os.Getenv("OFLUX_HOST"); h != "" {
		return normalizeHost(h)
	}
	return configuredBase()
}

var configuredBase = sync.OnceValue(func() string {
	port := types.DefaultConfig().Port
	if st, err := store.Open(os.Getenv("OFLUX_HOME")); err == nil {
		if cfg, err := st.LoadConfig(); err == nil && cfg.Port != 0 {
			port = cfg.Port
		}
	}
	return fmt.Sprintf("http://127.0.0.1:%d", port)
})

func normalizeHost(h string) string {
	h = strings.TrimSuffix(h, "/")
	switch {
	case strings.Contains(h, "://"):
		return h
	case strings.HasPrefix(h, ":"):
		return "http://127.0.0.1" + h
	default:
		return "http://" + h
	}
}

func do(method, path string, body any) (*http.Response, error) {
	var rdr io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		rdr = bytes.NewReader(b)
	}
	req, err := http.NewRequest(method, daemonBase()+path, rdr)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, daemonDownError(err)
	}
	if resp.StatusCode/100 != 2 {
		defer resp.Body.Close()
		return nil, apiError(resp)
	}
	return resp, nil
}

func post(path string, v any) (*http.Response, error) { return do(http.MethodPost, path, v) }

func getJSON(path string, v any) error {
	resp, err := do(http.MethodGet, path, nil)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	return json.NewDecoder(resp.Body).Decode(v)
}

func del(path string) error {
	resp, err := do(http.MethodDelete, path, nil)
	if err != nil {
		return err
	}
	return resp.Body.Close()
}

// Prints the NDJSON stream's {"status":...} lines; its first error ends the command.
func postStream(path string, v any) error {
	resp, err := post(path, v)
	if err != nil {
		return err
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

// parseFlagsMulti splits a command line into positional arguments and
// `--flag value` pairs, keeping every occurrence of a repeated flag. flags maps
// every accepted spelling — aliases included — to the wire field it fills;
// anything else beginning with "-" is a typo, not a name. Unset flags stay
// absent rather than being sent as "", which would override configured defaults.
func parseFlagsMulti(args []string, flags map[string]string) (names []string, set map[string][]string, err error) {
	set = map[string][]string{}
	for i := 0; i < len(args); i++ {
		a := args[i]
		if field, ok := flags[a]; ok {
			if i+1 >= len(args) {
				return nil, nil, fmt.Errorf("%s requires a value", a)
			}
			set[field] = append(set[field], args[i+1])
			i++
			continue
		}
		if strings.HasPrefix(a, "-") {
			return nil, nil, fmt.Errorf("unknown flag %q", a)
		}
		names = append(names, a)
	}
	return names, set, nil
}

func parseFlags(args []string, flags map[string]string) (names []string, set map[string]string, err error) {
	names, multi, err := parseFlagsMulti(args, flags)
	if err != nil {
		return nil, nil, err
	}
	set = make(map[string]string, len(multi))
	for field, vals := range multi {
		set[field] = last(vals)
	}
	return names, set, nil
}

func last(vals []string) string {
	if len(vals) == 0 {
		return ""
	}
	return vals[len(vals)-1]
}

// nil when absent; steps/cfg are JSON numbers, so a string would be a decode error.
func intFlag(set map[string][]string, field string) (*int, error) {
	raw, ok := set[field]
	if !ok {
		return nil, nil
	}
	n, err := strconv.Atoi(last(raw))
	if err != nil || n <= 0 {
		return nil, fmt.Errorf("--%s wants a positive whole number, got %q", wireToFlag(field), last(raw))
	}
	return &n, nil
}

func floatFlag(set map[string][]string, field string) (*float64, error) {
	raw, ok := set[field]
	if !ok {
		return nil, nil
	}
	f, err := strconv.ParseFloat(last(raw), 64)
	if err != nil || f < 0 {
		return nil, fmt.Errorf("--%s wants a number, got %q", wireToFlag(field), last(raw))
	}
	return &f, nil
}

func wireToFlag(field string) string { return strings.ReplaceAll(field, "_", "-") }

var pullFlags = map[string]string{
	"--quant": "quant", "-q": "quant",
	"--file": "file", "-f": "file",
	"--control-net": "control_net", "--controlnet": "control_net",
	"--control-net-file": "control_net_file", "--controlnet-file": "control_net_file",
	"--as":         "as",
	"--keep-alive": "keep_alive", "--keepalive": "keep_alive",
}

type pullArgs struct {
	Names  []string
	Fields map[string]string
}

// request is the wire form for one model, since /api/pull installs one at a time.
func (p pullArgs) request(name string) map[string]string {
	req := maps.Clone(p.Fields)
	req["name"] = name
	return req
}

func parseNameQuant(args []string) (pullArgs, error) {
	names, fields, err := parseFlags(args, pullFlags)
	if err != nil {
		return pullArgs{}, err
	}
	if len(names) == 0 {
		return pullArgs{}, errors.New("a model name is required")
	}
	if fields["control_net_file"] != "" && fields["control_net"] == "" {
		return pullArgs{}, errors.New("--control-net-file needs --control-net")
	}
	if d := fields["keep_alive"]; d != "" {
		if _, err := time.ParseDuration(d); err != nil {
			return pullArgs{}, fmt.Errorf("--keep-alive wants a duration like 10m or 30s, got %q", d)
		}
	}
	// These name or reshape a single install, so they cannot be spread across
	// several: --as would give every model the same name, and --file/--control-net
	// name a path inside one specific repo.
	if len(names) > 1 {
		for _, field := range []string{"as", "file", "control_net"} {
			if fields[field] != "" {
				return pullArgs{}, fmt.Errorf("--%s applies to a single model, but %d were given",
					wireToFlag(field), len(names))
			}
		}
	}
	return pullArgs{Names: names, Fields: fields}, nil
}
