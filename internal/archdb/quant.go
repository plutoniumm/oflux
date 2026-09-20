package archdb

import (
	"path"
	"slices"
	"strings"
)

// Quant is a ggml quantization label as it appears in a GGUF filename
// ("Q8_0", "Q4_K_M", "F16"), and the one quantization vocabulary for the whole
// program: compat labels a repo's files with it, the registry lists which ones
// a source publishes, the puller turns a user's --quant into one.
type Quant string

// quantOrder is every label oflux recognizes, by DESCENDING precision. A longer
// label must precede one it contains ("Q4_K_M" before "Q4_K", "BF16" before
// "F16") or ParseQuant's substring scan reports the shorter one. It ranks
// precision, not download preference — that is what QuantChain is for.
var quantOrder = []Quant{
	"F32", "FP32", "BF16", "F16", "FP16",
	"Q8_0",
	"Q6_K",
	"Q5_K_L", "Q5_K_M", "Q5_K_S", "Q5_K", "Q5_1", "Q5_0",
	"Q4_K_L", "Q4_K_M", "Q4_K_S", "Q4_K", "Q4_1", "Q4_0",
	"Q3_K_L", "Q3_K_M", "Q3_K_S", "Q3_K",
	"Q2_K_L", "Q2_K",
}

// generalChain is the ladder used once the requested quant and its own family
// are exhausted: the labels nearly every publisher ships, best first.
var generalChain = []Quant{"Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0"}

// unquantized labels are not a quantization failure, just enormous, so they are
// reported by name and never reached by a fallback until everything else was.
var unquantized = []Quant{"F32", "FP32", "BF16", "F16", "FP16"}

// Unquantized reports whether the label is f16/bf16/f32 rather than a quant.
func (q Quant) Unquantized() bool {
	return slices.ContainsFunc(unquantized, func(u Quant) bool {
		return strings.EqualFold(string(u), string(q))
	})
}

// family is the bit-width group: "Q4" for every Q4_* label, "F" for the
// unquantized ones. Falling back within a family keeps the precision asked for
// when publishers disagree on granularity (Q4_K vs Q4_K_M/Q4_K_S vs Q4_0).
func (q Quant) family() string {
	s := strings.ToUpper(string(q))
	if len(s) >= 2 && s[0] == 'Q' && s[1] >= '0' && s[1] <= '9' {
		return s[:2]
	}
	return "F"
}

// ParseQuant extracts the label from a filename (or a bare label), matched
// case-insensitively against the basename, in its canonical spelling.
func ParseQuant(filename string) (Quant, bool) {
	base := strings.ToLower(path.Base(filename))
	for _, q := range quantOrder {
		if strings.Contains(base, strings.ToLower(string(q))) {
			return q, true
		}
	}
	return "", false
}

// QuantChain returns the labels to try for pref, best first: pref, the rest of
// its bit-width family, the general ladder, then the rest with full precision
// last. A companion must be allowed to differ slightly from the diffusion
// weights, or resolving it at their exact label 404s after gigabytes.
func QuantChain(pref Quant) []Quant {
	out := make([]Quant, 0, len(quantOrder))
	seen := make(map[Quant]bool, len(quantOrder))
	add := func(q Quant) {
		if q != "" && !seen[q] {
			seen[q] = true
			out = append(out, q)
		}
	}
	if canon, ok := ParseQuant(string(pref)); ok {
		add(canon)
		for _, q := range quantOrder {
			if q.family() == canon.family() {
				add(q)
			}
		}
	}
	for _, q := range generalChain {
		add(q)
	}
	for _, q := range quantOrder {
		if !q.Unquantized() {
			add(q)
		}
	}
	for _, q := range quantOrder {
		add(q) // whatever is left: the full-precision labels
	}
	return out
}

// PickQuant chooses the best of have for a caller who asked for pref. ok=false
// leaves the caller free to substitute pref verbatim.
func PickQuant(pref Quant, have []Quant) (Quant, bool) {
	if len(have) == 0 {
		return "", false
	}
	for _, q := range QuantChain(pref) {
		if slices.Contains(have, q) {
			return q, true
		}
	}
	return "", false
}
