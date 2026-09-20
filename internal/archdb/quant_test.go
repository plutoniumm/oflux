package archdb

import (
	"slices"
	"strings"
	"testing"
)

func TestParseQuant(t *testing.T) {
	cases := map[string]Quant{
		"flux1-dev-Q8_0.gguf":                "Q8_0",
		"Qwen2.5-VL-7B-Instruct.Q4_K_M.gguf": "Q4_K_M",
		"z_image_turbo-Q4_K.gguf":            "Q4_K",
		"m-q6_k.GGUF":                        "Q6_K",
		"qwen-image-edit-2511-BF16.gguf":     "BF16",
		"flux1-dev-F16.gguf":                 "F16",
		"Q2_K":                               "Q2_K",
		"split_files/vae/ae.safetensors":     "",
	}
	for in, want := range cases {
		got, ok := ParseQuant(in)
		if want == "" {
			if ok {
				t.Errorf("ParseQuant(%q) = %q, want no label", in, got)
			}
			continue
		}
		if !ok || got != want {
			t.Errorf("ParseQuant(%q) = %q,%v; want %q", in, got, ok, want)
		}
	}
}

func TestQuantChain(t *testing.T) {
	// A bare K-quant must be able to reach its sized siblings: diffusion repos
	// publish "Q4_K" where encoder repos publish "Q4_K_M"/"Q4_K_S".
	got := QuantChain("Q4_K")
	if got[0] != "Q4_K" {
		t.Errorf("exact match must come first, got %q", got[0])
	}
	mi, si := slices.Index(got, "Q4_K_M"), slices.Index(got, "Q4_K_S")
	if mi < 0 || si < 0 {
		t.Fatalf("Q4_K should fall back to its sized siblings: %v", got)
	}
	if mi > si {
		t.Errorf("medium should be preferred over small: %v", got)
	}
	if mi > slices.Index(got, "Q8_0") {
		t.Errorf("the same bit width should be preferred over a bigger one: %v", got)
	}

	if got := QuantChain("Q5_K_M"); got[0] != "Q5_K_M" || !slices.Contains(got, "Q5_K") {
		t.Errorf("Q5_K_M should fall back to Q5_K: %v", got)
	}

	// Every chain reaches a general ladder so something always resolves, and a
	// full-precision label is never reached before a real quantization.
	for _, q := range []Quant{"Q4_K", "Q8_0", "Q4_0", "weird", ""} {
		chain := QuantChain(q)
		if !slices.Contains(chain, "Q4_K_M") {
			t.Errorf("%s chain has no general fallback: %v", q, chain)
		}
		if q.Unquantized() {
			continue
		}
		if slices.Index(chain, "F16") < slices.Index(chain, "Q2_K") {
			t.Errorf("%s chain reaches f16 before the last quant: %v", q, chain)
		}
	}
}

// unsloth's Qwen3-8B publishes no Q4_0, so a Q4_0 klein-9B has to land on the
// nearest Q4 that repo does have rather than a filename that 404s.
func TestPickQuant(t *testing.T) {
	have := []Quant{"Q8_0", "Q6_K", "Q4_K_M", "Q4_K_S"}
	cases := map[Quant]Quant{
		"Q4_0":  "Q4_K_M",
		"Q4_K":  "Q4_K_M",
		"Q8_0":  "Q8_0",
		"Q2_K":  "Q8_0",
		"weird": "Q8_0",
	}
	for pref, want := range cases {
		got, ok := PickQuant(pref, have)
		if !ok || got != want {
			t.Errorf("PickQuant(%q) = %q,%v; want %q", pref, got, ok, want)
		}
	}
	if _, ok := PickQuant("Q8_0", nil); ok {
		t.Error("PickQuant with nothing to pick from must report ok=false")
	}
}

func TestQuantUnquantized(t *testing.T) {
	for _, q := range []Quant{"F16", "f16", "BF16", "F32", "FP16"} {
		if !q.Unquantized() {
			t.Errorf("%q should be unquantized", q)
		}
	}
	for _, q := range []Quant{"Q8_0", "Q4_K_M", ""} {
		if q.Unquantized() {
			t.Errorf("%q should not be unquantized", q)
		}
	}
}

// ParseQuant scans quantOrder as substrings, so a longer label that contains a
// shorter one must come first or "Q4_K_M" gets reported as "Q4_K".
func TestQuantOrderIsScanSafe(t *testing.T) {
	for i, a := range quantOrder {
		for j, b := range quantOrder {
			if i == j {
				continue
			}
			if len(b) > len(a) && strings.Contains(string(b), string(a)) && j > i {
				t.Errorf("%q contains %q but sorts after it", b, a)
			}
		}
	}
}
