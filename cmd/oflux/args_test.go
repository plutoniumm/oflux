package main

import (
	"slices"
	"strings"
	"testing"
)

// `oflux rm a b` used to act on "a" only and report success, leaving "b"
// installed. Positional arguments are now all collected.
func TestParseNameQuantCollectsEveryName(t *testing.T) {
	p, err := parseNameQuant([]string{"z-image-turbo", "qwen-image-edit", "flux.1-dev"})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"z-image-turbo", "qwen-image-edit", "flux.1-dev"}
	if !slices.Equal(p.Names, want) {
		t.Fatalf("Names = %v, want %v", p.Names, want)
	}
}

func TestParseNameQuantFlagsApplyToAll(t *testing.T) {
	p, err := parseNameQuant([]string{"a", "--quant", "Q6_K", "b"})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(p.Names, []string{"a", "b"}) {
		t.Fatalf("Names = %v", p.Names)
	}
	if p.Fields["quant"] != "Q6_K" {
		t.Fatalf("Quant = %q", p.Fields["quant"])
	}
	// A flag value must never be mistaken for a model name.
	if slices.Contains(p.Names, "Q6_K") {
		t.Fatal("flag value leaked into Names")
	}
}

// --as/--file/--control-net each name one specific install, so spreading them
// over several models is a mistake worth catching rather than guessing at.
func TestParseNameQuantRejectsSingleOnlyFlagsWithManyNames(t *testing.T) {
	for _, flag := range []string{"--as", "--file", "--control-net"} {
		if _, err := parseNameQuant([]string{"a", "b", flag, "x"}); err == nil {
			t.Errorf("%s with two names should error", flag)
		}
		// The same flag is fine with a single name.
		if _, err := parseNameQuant([]string{"a", flag, "x"}); err != nil {
			t.Errorf("%s with one name: %v", flag, err)
		}
	}
}

func TestParseNameQuantErrors(t *testing.T) {
	if _, err := parseNameQuant(nil); err == nil {
		t.Error("no names should error")
	}
	if _, err := parseNameQuant([]string{"--quant"}); err == nil {
		t.Error("a flag with no value should error")
	}
	// An unknown flag must not be silently treated as a model name — that turned
	// a typo into a confusing "unknown model" from the daemon.
	if _, err := parseNameQuant([]string{"a", "--nope"}); err == nil {
		t.Error("unknown flag should error")
	}
	if _, err := parseNameQuant([]string{"a", "--control-net-file", "x"}); err == nil {
		t.Error("--control-net-file without --control-net should error")
	}
}

// Only the fields that were actually set may reach the daemon; empty strings
// would override the configured default quant with "".
func TestPullArgsRequestOmitsEmptyFields(t *testing.T) {
	p, err := parseNameQuant([]string{"a", "b"})
	if err != nil {
		t.Fatal(err)
	}
	req := p.request("b")
	if req["name"] != "b" {
		t.Fatalf("name = %q, want the per-model name", req["name"])
	}
	if len(req) != 1 {
		t.Fatalf("unset fields should be omitted, got %v", req)
	}

	p, err = parseNameQuant([]string{"repo/x", "--quant", "Q4_K_M", "--file", "v19/w.gguf"})
	if err != nil {
		t.Fatal(err)
	}
	req = p.request("repo/x")
	for k, want := range map[string]string{"name": "repo/x", "quant": "Q4_K_M", "file": "v19/w.gguf"} {
		if req[k] != want {
			t.Errorf("%s = %q, want %q", k, req[k], want)
		}
	}
}

func TestHumanSize(t *testing.T) {
	for _, tc := range []struct {
		in   int64
		want string
	}{
		{512, "512B"},
		{849608296, "849.6MB"},
		{21750652384, "21.8GB"},
	} {
		if got := humanSize(tc.in); got != tc.want {
			t.Errorf("humanSize(%d) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestUsageMentionsMultipleNames(t *testing.T) {
	// The help text has to advertise the plural forms, or nobody discovers them.
	for _, want := range []string{"rm|delete <name>...", "pull <name|org/repo>..."} {
		if !strings.Contains(usageText(), want) {
			t.Errorf("usage missing %q", want)
		}
	}
}

// One table, so a command can never be callable but undocumented.
func TestUsageCoversEveryCommand(t *testing.T) {
	help := usageText()
	for _, c := range commandTable() {
		if !strings.Contains(help, "oflux "+c.invocation()) {
			t.Errorf("command %q missing from the help block", c.name)
		}
		for _, spelling := range append([]string{c.name}, c.aliases...) {
			got, ok := lookup(spelling)
			if !ok || got.name != c.name {
				t.Errorf("%q does not dispatch to %q", spelling, c.name)
			}
		}
	}
	if _, ok := lookup("nope"); ok {
		t.Error("an unknown command must not resolve")
	}
}

// `--lora a --lora b` keeps both; the single-valued parser takes the last.
func TestParseFlagsMultiKeepsRepeats(t *testing.T) {
	names, set, err := parseFlagsMulti([]string{"fast", "--lora", "a", "--model", "m", "--lora", "b"}, presetFlags)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(names, []string{"fast"}) {
		t.Fatalf("names = %v", names)
	}
	if !slices.Equal(set["lora"], []string{"a", "b"}) {
		t.Fatalf("lora = %v, want both", set["lora"])
	}
	_, single, err := parseFlags([]string{"--model", "m1", "--model", "m2"}, presetFlags)
	if err != nil {
		t.Fatal(err)
	}
	if single["model"] != "m2" {
		t.Fatalf("model = %q, want the last value", single["model"])
	}
}

// "unset" has to stay distinguishable from zero, or a preset pins a default.
func TestNumericFlagsAreTypedAndOptional(t *testing.T) {
	n, err := intFlag(map[string][]string{}, "steps")
	if n != nil || err != nil {
		t.Fatalf("absent steps = %v, %v; want nil, nil", n, err)
	}
	n, err = intFlag(map[string][]string{"steps": {"4"}}, "steps")
	if err != nil || n == nil || *n != 4 {
		t.Fatalf("steps = %v, %v", n, err)
	}
	if _, err := intFlag(map[string][]string{"steps": {"lots"}}, "steps"); err == nil {
		t.Error("a non-numeric --steps should error")
	}
	if _, err := intFlag(map[string][]string{"steps": {"0"}}, "steps"); err == nil {
		t.Error("--steps 0 should error rather than pin zero steps")
	}
	f, err := floatFlag(map[string][]string{"cfg": {"1.5"}}, "cfg")
	if err != nil || f == nil || *f != 1.5 {
		t.Fatalf("cfg = %v, %v", f, err)
	}
	if _, err := floatFlag(map[string][]string{"cfg": {"none"}}, "cfg"); err == nil {
		t.Error("a non-numeric --cfg should error")
	}
}

// A typo must fail here, not after a multi-minute model load.
func TestParseNameQuantValidatesKeepAlive(t *testing.T) {
	p, err := parseNameQuant([]string{"a", "--keep-alive", "10m"})
	if err != nil {
		t.Fatal(err)
	}
	if p.Fields["keep_alive"] != "10m" {
		t.Fatalf("keep_alive = %q", p.Fields["keep_alive"])
	}
	if _, err := parseNameQuant([]string{"a", "--keep-alive", "forever"}); err == nil {
		t.Error("an unparseable --keep-alive should error")
	}
	// /api/pull would ignore it, so say where it belongs rather than no-op.
	if err := cmdPull([]string{"a", "--keep-alive", "10m"}); err == nil {
		t.Error("--keep-alive on pull should error")
	}
}

// `pull`/`run` and `lora pull` share one flag parser, so every alias has to
// keep filling the same wire field, and flags that were not given must stay
// out of the request entirely.
func TestParseFlagsAliasesAndOmissions(t *testing.T) {
	names, fields, err := parseFlags([]string{"a", "-q", "Q8_0", "--controlnet", "org/cn", "-f", "w.gguf"}, pullFlags)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(names, []string{"a"}) {
		t.Fatalf("names = %v", names)
	}
	for k, want := range map[string]string{"quant": "Q8_0", "control_net": "org/cn", "file": "w.gguf"} {
		if fields[k] != want {
			t.Errorf("%s = %q, want %q", k, fields[k], want)
		}
	}
	if _, ok := fields["as"]; ok {
		t.Error("an unset flag must not appear in the request")
	}
	if _, _, err := parseFlags([]string{"a", "--as"}, pullFlags); err == nil {
		t.Error("a flag with no value should error")
	}
}
