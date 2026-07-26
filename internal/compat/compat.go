// Package compat is oflux's pull-time compatibility checker: given a Hugging
// Face repo, it decides whether the bundled sd-server engine can run it and, if
// so, produces a resolved manifest (which weight files to fetch, from where,
// and how to launch the engine).
//
// It is the product's novel piece. All architecture knowledge lives in
// internal/archdb; this package only classifies a concrete repo against that
// table. It performs no network I/O itself — callers pass a RepoFetcher (the
// real *hfclient.Client satisfies it structurally, so we avoid importing
// hfclient and the build dependency that would bring).
package compat

import (
	"context"
	"encoding/json"
	"path"
	"regexp"
	"strings"

	"oflux/internal/archdb"
	"oflux/internal/types"
)

// RepoFetcher is the subset of the Hugging Face client that compat needs. The
// real *hfclient.Client satisfies it structurally — do NOT import hfclient.
type RepoFetcher interface {
	Tree(ctx context.Context, repo, revision string) ([]types.HFFile, error)
	ReadFile(ctx context.Context, repo, revision, path string, maxBytes int64) ([]byte, error)
}

// maxConfigBytes caps how much of a JSON config we read; these files are tiny.
const maxConfigBytes = 1 << 20

// diffusionExcludeTokens are substrings that mark a weight file as a shared
// component (VAE / text encoder / vision projector) rather than the DiT/UNet
// diffusion weights. A candidate diffusion file must contain none of them.
var diffusionExcludeTokens = []string{
	"vae", "text_encoder", "clip", "t5", "mmproj", "llm",
	"qwen2.5-vl", "qwen_2.5", "qwen3",
}

// roleKeywords are substrings used to recognize an in-repo file for a shared
// role (best-effort; the common case is that shared components come from
// companion repos instead). Kept consistent with diffusionExcludeTokens so a
// file is never both a diffusion candidate and a role file.
var roleKeywords = map[types.Role][]string{
	types.RoleVAE:    {"vae"},
	types.RoleCLIPL:  {"clip_l", "clip-l", "clip"},
	types.RoleT5XXL:  {"t5xxl", "t5-xxl", "t5_xxl", "t5"},
	types.RoleLLM:    {"qwen2.5-vl", "qwen_2.5", "qwen3", "text_encoder", "llm"},
	types.RoleMMProj: {"mmproj"},
}

// Inspect classifies a Hugging Face repo against the sd-server capability table
// and returns a Verdict. quantPref is the ordered quantization preference, e.g.
// ["Q8_0","Q6_K","Q5_K_M","Q4_K_M"]. When compatible, Verdict.Manifest is
// populated with Components carrying Source+File (Blob empty until downloaded)
// and a full Engine spec.
func Inspect(ctx context.Context, f RepoFetcher, repo string, quantPref []string) (types.Verdict, error) {
	files, err := f.Tree(ctx, repo, "")
	if err != nil {
		return types.Verdict{}, err
	}

	arch, ok, _ := DetectArch(ctx, f, repo, files)
	if !ok {
		return types.Verdict{
			Repo:       repo,
			Compatible: false,
			Blockers: []types.Blocker{{
				Kind:   types.BlockerArchitecture,
				Detail: "architecture not recognized / not supported by sd-server",
			}},
		}, nil
	}

	// Pick the diffusion weights from the repo. Prefer a GGUF matching the
	// quant preference; a bare fp16/bf16 file (or a gguf that matches no
	// preference) is a no-quant situation.
	diffFiles := diffusionCandidates(files)
	var ggufFiles []string
	for _, c := range diffFiles {
		if strings.HasSuffix(strings.ToLower(path.Base(c)), ".gguf") {
			ggufFiles = append(ggufFiles, c)
		}
	}
	chosen, quant, qok := SelectQuant(ggufFiles, quantPref)
	if !qok {
		if len(diffFiles) > 0 {
			return types.Verdict{
				Repo:       repo,
				Compatible: false,
				Blockers: []types.Blocker{{
					Kind:    types.BlockerNoQuant,
					Role:    types.RoleDiffusion,
					Detail:  describeUnusable(diffFiles),
					Suggest: suggestedGGUFRepo(arch.Name),
				}},
			}, nil
		}
		// No diffusion weights at all — safety net; unusual for a recognized arch.
		return types.Verdict{
			Repo:       repo,
			Compatible: false,
			Blockers: []types.Blocker{{
				Kind:   types.BlockerMissingRole,
				Role:   types.RoleDiffusion,
				Detail: "no diffusion weights found in repo",
			}},
		}, nil
	}

	// Resolve quantized companions at the quant we actually matched for the
	// diffusion weights. Using quantPref[0] instead would, whenever the repo
	// lacks the top preference, request an encoder quant that may not exist in
	// its own repo — a 404 that only surfaces after the multi-GB DiT download.
	companionQuant := quant

	resolveRole := func(role types.Role) (types.Component, bool) {
		if role == types.RoleDiffusion {
			return types.Component{Role: role, File: chosen, Source: repo}, true
		}
		if inrepo, ok := findRoleFile(files, role, chosen); ok {
			return types.Component{Role: role, File: inrepo, Source: repo}, true
		}
		if comp, ok := arch.Companions[role]; ok {
			file := comp.FilePattern
			if comp.Quantized {
				file = strings.ReplaceAll(file, "{quant}", companionQuant)
			}
			return types.Component{Role: role, File: file, Source: comp.Source}, true
		}
		return types.Component{}, false
	}

	var components []types.Component
	var blockers []types.Blocker
	for _, role := range arch.Required {
		if c, ok := resolveRole(role); ok {
			components = append(components, c)
		} else {
			blockers = append(blockers, types.Blocker{
				Kind:   types.BlockerMissingRole,
				Role:   role,
				Detail: "required component not present in repo and no known companion source",
			})
		}
	}
	for _, role := range arch.Optional {
		// Optional roles are only added when we can actually resolve one.
		if c, ok := resolveRole(role); ok {
			components = append(components, c)
		}
	}

	if len(blockers) > 0 {
		return types.Verdict{Repo: repo, Compatible: false, Blockers: blockers}, nil
	}

	m := types.Manifest{
		Name:         deriveName(repo),
		Architecture: arch.Name,
		Mode:         arch.Mode,
		Components:   components,
		// EngineSpec() already carries this arch's ModelArgs and Defaults.
		Engine: arch.EngineSpec(),
	}
	notes := []string{"quant: " + quant}
	if isUnquantizedLabel(quant) {
		notes = append(notes, "diffusion weights are unquantized ("+quant+")")
	}
	return types.Verdict{
		Repo:       repo,
		Compatible: true,
		Manifest:   &m,
		Notes:      notes,
	}, nil
}

// SelectQuant picks the best filename from candidates given the preference
// order, matching the quant token case-insensitively anywhere in the filename.
// It returns the chosen filename and the matched quant label; ok=false if none
// match. Preference order wins: for the first preference that any candidate
// contains, the first such candidate (in the order given) is returned.
func SelectQuant(candidates, quantPref []string) (file, quant string, ok bool) {
	for _, pref := range quantPref {
		if pref == "" {
			continue
		}
		lp := strings.ToLower(pref)
		for _, c := range candidates {
			if strings.Contains(strings.ToLower(path.Base(c)), lp) {
				return c, pref, true
			}
		}
	}
	return "", "", false
}

// DetectArch determines the architecture of a repo: first via the diffusers
// _class_name found in model_index.json (top-level "_class_name") or config.json
// ("_class_name" or "architectures"[0]) using archdb.Lookup; falling back to
// archdb.MatchKeyword over the repo id and the repo's filenames. ok=false if
// unknown. A non-nil error is only returned when a config read failed AND no
// architecture could otherwise be determined.
func DetectArch(ctx context.Context, f RepoFetcher, repo string, files []types.HFFile) (arch archdb.Arch, ok bool, err error) {
	var lastErr error

	for _, name := range []string{"model_index.json", "config.json"} {
		p, found := findConfigFile(files, name)
		if !found {
			continue
		}
		cn, rerr := readClassName(ctx, f, repo, p)
		if rerr != nil {
			lastErr = rerr
			continue
		}
		if cn == "" {
			continue
		}
		if a, hit := archdb.Lookup(cn); hit {
			return a, true, nil
		}
	}

	// Keyword fallback: repo id first, then each filename.
	if a, hit := archdb.MatchKeyword(repo); hit {
		return a, true, nil
	}
	for _, fl := range files {
		if a, hit := archdb.MatchKeyword(path.Base(fl.Path)); hit {
			return a, true, nil
		}
	}
	return archdb.Arch{}, false, lastErr
}

// readClassName reads a diffusers config and returns its architecture class
// name: "_class_name" when set, else the first entry of "architectures". A
// malformed JSON body is treated as "no class name" (non-fatal); only an I/O
// read error is returned.
func readClassName(ctx context.Context, f RepoFetcher, repo, p string) (string, error) {
	b, err := f.ReadFile(ctx, repo, "", p, maxConfigBytes)
	if err != nil {
		return "", err
	}
	var doc struct {
		ClassName     string   `json:"_class_name"`
		Architectures []string `json:"architectures"`
	}
	if json.Unmarshal(b, &doc) != nil {
		return "", nil
	}
	if doc.ClassName != "" {
		return doc.ClassName, nil
	}
	if len(doc.Architectures) > 0 {
		return doc.Architectures[0], nil
	}
	return "", nil
}

// findConfigFile locates a config file by name, preferring a root-level match
// (path == name) over a match in any subfolder (base == name).
func findConfigFile(files []types.HFFile, name string) (string, bool) {
	for _, fl := range files {
		if fl.Path == name {
			return fl.Path, true
		}
	}
	for _, fl := range files {
		if path.Base(fl.Path) == name {
			return fl.Path, true
		}
	}
	return "", false
}

// diffusionCandidates returns the repo files that could be diffusion weights:
// .gguf/.safetensors whose path contains none of the shared-component tokens.
func diffusionCandidates(files []types.HFFile) []string {
	var out []string
	for _, fl := range files {
		if !isWeightFile(fl.Path) {
			continue
		}
		if containsAny(strings.ToLower(fl.Path), diffusionExcludeTokens) {
			continue
		}
		out = append(out, fl.Path)
	}
	return out
}

// findRoleFile looks for an in-repo weight file matching a shared role,
// skipping the already-chosen diffusion file.
func findRoleFile(files []types.HFFile, role types.Role, exclude string) (string, bool) {
	kws := roleKeywords[role]
	if kws == nil {
		return "", false
	}
	// Keywords are ordered most- to least-specific, so try every file against
	// keyword[0] before falling back to keyword[1]. Iterating files first would
	// let a broad keyword on an alphabetically earlier file win — e.g. "clip"
	// matching clip_g.safetensors and handing CLIP-G to the --clip_l flag.
	for _, kw := range kws {
		for _, fl := range files {
			if fl.Path == exclude || !isWeightFile(fl.Path) {
				continue
			}
			full := strings.ToLower(fl.Path)
			// The LLM text encoder and its vision projector share a name
			// prefix; don't mistake an mmproj file for the LLM.
			if role == types.RoleLLM && strings.Contains(full, "mmproj") {
				continue
			}
			// A more specific sibling role's file must not be claimed here
			// (clip_g is never the clip_l component).
			if role == types.RoleCLIPL && strings.Contains(full, "clip_g") {
				continue
			}
			if strings.Contains(full, kw) {
				return fl.Path, true
			}
		}
	}
	return "", false
}

// deriveName turns a repo id into a manifest name: the last path segment,
// lowercased, with a trailing "-gguf" stripped.
func deriveName(repo string) string {
	seg := repo
	if i := strings.LastIndex(repo, "/"); i >= 0 {
		seg = repo[i+1:]
	}
	seg = strings.ToLower(seg)
	return strings.TrimSuffix(seg, "-gguf")
}

// suggestedGGUFRepo maps an architecture to a known GGUF mirror to suggest when
// a repo only ships unquantized weights. Empty if there's no known mirror.
// foreignQuants are quantization formats other runtimes use that ggml cannot
// read. Naming the actual format is far more useful than calling such a file
// "unquantized", which is both wrong and sends people looking for the wrong fix.
var foreignQuants = []struct{ token, name string }{
	{"nvfp4", "NVFP4 (NVIDIA 4-bit, for TensorRT/Blackwell GPUs)"},
	{"fp8", "FP8"},
	{"int8", "INT8"},
	{"int4", "INT4"},
	{"awq", "AWQ"},
	{"gptq", "GPTQ"},
	{"bnb", "bitsandbytes"},
	{"4bit", "4-bit (bitsandbytes-style)"},
	{"8bit", "8-bit (bitsandbytes-style)"},
	{"svdq", "SVDQuant (Nunchaku)"},
	{"nunchaku", "SVDQuant (Nunchaku)"},
}

// shardRe matches HF's sharded-checkpoint naming, e.g.
// "diffusion_pytorch_model-00001-of-00003.safetensors".
var shardRe = regexp.MustCompile(`-\d{5}-of-\d{5}\.`)

// describeUnusable explains why a repo's weight files can't be used, given that
// none of them is a GGUF at an acceptable quantization.
func describeUnusable(files []string) string {
	sharded := false
	for _, f := range files {
		low := strings.ToLower(path.Base(f))
		for _, fq := range foreignQuants {
			if strings.Contains(low, fq.token) {
				return "weights are quantized as " + fq.name + ", which the sd-server engine cannot read (it needs GGUF)"
			}
		}
		if shardRe.MatchString(low) {
			sharded = true
		}
	}
	if sharded {
		return "weights are unquantized and split across shards; sd-server needs a single GGUF diffusion file"
	}
	return "only unquantized weights found"
}

func suggestedGGUFRepo(arch string) string {
	switch arch {
	case "flux":
		return "city96/FLUX.1-dev-gguf"
	case "flux-kontext":
		return "QuantStack/FLUX.1-Kontext-dev-GGUF"
	case "qwen-image":
		return "QuantStack/Qwen-Image-GGUF"
	case "qwen-image-edit":
		return "QuantStack/Qwen-Image-Edit-GGUF"
	case "z-image":
		return "leejet/Z-Image-Turbo-GGUF"
	default:
		return ""
	}
}

// isUnquantizedLabel reports whether a matched quant label denotes full
// precision rather than an actual quantization.
func isUnquantizedLabel(q string) bool {
	switch strings.ToLower(q) {
	case "f16", "f32", "bf16", "fp16", "fp32":
		return true
	default:
		return false
	}
}

func isWeightFile(p string) bool {
	b := strings.ToLower(path.Base(p))
	return strings.HasSuffix(b, ".gguf") || strings.HasSuffix(b, ".safetensors")
}

func containsAny(s string, subs []string) bool {
	for _, sub := range subs {
		if strings.Contains(s, sub) {
			return true
		}
	}
	return false
}
