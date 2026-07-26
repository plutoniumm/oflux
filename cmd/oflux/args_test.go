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
	if p.Quant != "Q6_K" {
		t.Fatalf("Quant = %q", p.Quant)
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
	for _, want := range []string{"oflux rm   <name>...", "oflux pull <name|org/repo>..."} {
		if !strings.Contains(usageText, want) {
			t.Errorf("usage missing %q", want)
		}
	}
}
