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
	"fmt"
	"path"
	"regexp"
	"slices"
	"strconv"
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

const maxConfigBytes = 1 << 20

// roleKeywords recognize an in-repo file for a shared role, most- to
// least-specific. Best-effort: usually they come from companion repos instead.
var roleKeywords = map[types.Role][]string{
	types.RoleVAE:    {"vae"},
	types.RoleCLIPL:  {"clip_l", "clip-l", "clip"},
	types.RoleT5XXL:  {"t5xxl", "t5-xxl", "t5_xxl", "t5"},
	types.RoleLLM:    {"qwen2.5-vl", "qwen_2.5", "qwen3", "text_encoder", "llm"},
	types.RoleMMProj: {"mmproj"},
}

// roleExcludeTokens disqualify a file from a role that a broader keyword would
// otherwise claim: the LLM text encoder and its vision projector share a name
// prefix, and clip_g is never the clip_l component.
var roleExcludeTokens = map[types.Role][]string{
	types.RoleLLM:   {"mmproj"},
	types.RoleCLIPL: {"clip_g"},
}

// diffusionExcludeTokens mark a weight file as a shared component rather than
// the DiT/UNet weights. Derived from roleKeywords so the two views of one fact
// cannot drift: anything that identifies a file as a VAE or text encoder must
// also disqualify it as a diffusion candidate.
var diffusionExcludeTokens = func() []string {
	var out []string
	for _, kws := range roleKeywords {
		out = append(out, kws...)
	}
	slices.Sort(out)
	return slices.Compact(out)
}()

// Inspect classifies a Hugging Face repo against the sd-server capability table
// and returns a Verdict. quantPref is the ordered quantization preference, e.g.
// ["Q8_0","Q6_K","Q5_K_M","Q4_K_M"].
func Inspect(ctx context.Context, f RepoFetcher, repo string, quantPref []string) (types.Verdict, error) {
	return InspectFile(ctx, f, repo, quantPref, "")
}

// InspectFile is Inspect with the diffusion weights pinned to a file in the
// repo. Some repos publish dozens of builds — revisions, SFW and NSFW cuts,
// several quants of each — and quant preference alone resolves to whichever
// sorts first, which is rarely the one the user wanted.
func InspectFile(ctx context.Context, f RepoFetcher, repo string, quantPref []string, wantFile string) (types.Verdict, error) {
	files, err := f.Tree(ctx, repo, "")
	if err != nil {
		return types.Verdict{}, err
	}

	arch, ok, _ := detectArch(ctx, f, repo, files)
	if !ok {
		return incompatible(repo, types.Blocker{
			Kind:   types.BlockerArchitecture,
			Detail: "architecture not recognized / not supported by sd-server",
		}), nil
	}

	// A bare fp16/bf16 file, or a gguf matching no preference, is a no-quant.
	diffFiles := diffusionCandidates(files)
	var ggufFiles []string
	for _, c := range diffFiles {
		if hasAnySuffix(path.Base(c), ".gguf") {
			ggufFiles = append(ggufFiles, c)
		}
	}

	if wantFile != "" {
		chosen, ok := matchFile(files, wantFile)
		if !ok {
			return incompatible(repo, types.Blocker{
				Kind:   types.BlockerMissingRole,
				Role:   types.RoleDiffusion,
				Detail: fmt.Sprintf("no file %q in %s", wantFile, repo),
			}), nil
		}
		quant, ok := archdb.ParseQuant(chosen)
		if !ok {
			quant = "unknown"
		}
		return buildVerdict(ctx, f, repo, arch, files, chosen, quant)
	}

	chosen, quant, qok := selectQuant(ggufFiles, quantPrefs(quantPref))
	if !qok {
		if len(diffFiles) > 0 {
			return incompatible(repo, types.Blocker{
				Kind:    types.BlockerNoQuant,
				Role:    types.RoleDiffusion,
				Detail:  describeUnusable(diffFiles),
				Suggest: suggestedGGUFRepo(arch.Name),
			}), nil
		}
		// No diffusion weights at all — safety net; unusual for a recognized arch.
		return incompatible(repo, types.Blocker{
			Kind:   types.BlockerMissingRole,
			Role:   types.RoleDiffusion,
			Detail: "no diffusion weights found in repo",
		}), nil
	}

	return buildVerdict(ctx, f, repo, arch, files, chosen, quant)
}

// Blockers are what the CLI turns into the pull error, so a rejection must
// never be returned without one.
func incompatible(repo string, blockers ...types.Blocker) types.Verdict {
	return types.Verdict{Repo: repo, Compatible: false, Blockers: blockers}
}

func buildVerdict(ctx context.Context, f RepoFetcher, repo string, arch archdb.Arch, files []types.HFFile, chosen string, quant archdb.Quant) (types.Verdict, error) {
	// The repo id and the chosen filename are what tell klein-9B from klein-4B,
	// which take different text encoders.
	arch = arch.For(repo, chosen)

	trees := map[string][]types.HFFile{repo: files}
	resolve := func(role types.Role) (types.Component, bool) {
		if role == types.RoleDiffusion {
			return component(role, repo, chosen, files), true
		}
		if inrepo, ok := findRoleFile(files, role, chosen); ok {
			return component(role, repo, inrepo, files), true
		}
		comp, ok := arch.Companions[role]
		if !ok {
			return types.Component{}, false
		}
		// Resolve the companion at the quant actually matched for the diffusion
		// weights, not at quantPref[0]: whenever the repo lacks the top
		// preference, that would request an encoder quant its own repo may not
		// publish — a 404 that only surfaces after the multi-GB DiT download.
		ctree := repoTree(ctx, f, trees, comp.Source)
		return component(role, comp.Source, companionFile(comp, quant, ctree), ctree), true
	}

	notes := []string{"quant: " + string(quant)}
	if quant.Unquantized() {
		notes = append(notes, "diffusion weights are unquantized ("+string(quant)+")")
	}
	// A step-distilled checkpoint samples nothing like its base architecture, and
	// those defaults are baked into the launch flags, so they have to be decided
	// here rather than per request.
	overrides, note := fewStepDefaults(chosen)
	if note != "" {
		notes = append(notes, note)
	}

	m, blockers := archdb.Build(arch, archdb.BuildOpts{
		Name:    deriveName(repo),
		Repo:    repo,
		Hints:   []string{chosen},
		Engine:  arch.EngineSpecWith(overrides),
		Resolve: resolve,
	})
	if len(blockers) > 0 {
		return incompatible(repo, blockers...), nil
	}
	return types.Verdict{
		Repo:       repo,
		Compatible: true,
		Manifest:   &m,
		Notes:      notes,
	}, nil
}

// component fills in what the tree knows, saving the puller a second walk of
// it. Only the LFS oid is a sha256; a git oid is a sha1 and must not land here.
func component(role types.Role, source, file string, tree []types.HFFile) types.Component {
	c := types.Component{Role: role, File: file, Source: source}
	for _, fl := range tree {
		if fl.Path != file {
			continue
		}
		c.Size = fl.Size
		if fl.IsLFS {
			c.SHA256 = fl.LFSOID
		}
		break
	}
	return c
}

// repoTree fetches a listing once, remembering failures so a broken companion
// source is not refetched per role.
func repoTree(ctx context.Context, f RepoFetcher, cache map[string][]types.HFFile, repo string) []types.HFFile {
	if t, ok := cache[repo]; ok {
		return t
	}
	t, err := f.Tree(ctx, repo, "")
	if err != nil {
		t = nil
	}
	cache[repo] = t
	return t
}

// companionFile resolves a companion at the best quant its own repo publishes,
// starting from the one matched for the diffusion weights. With no tree to
// consult it uses the labels archdb recorded, then a straight substitution.
func companionFile(comp archdb.Companion, quant archdb.Quant, tree []types.HFFile) string {
	if !comp.Quantized {
		return comp.FilePattern
	}
	sub := func(q archdb.Quant) string { return strings.ReplaceAll(comp.FilePattern, "{quant}", string(q)) }
	if len(tree) == 0 {
		if q, ok := archdb.PickQuant(quant, comp.Quants); ok {
			return sub(q)
		}
		return sub(quant)
	}
	have := make(map[string]bool, 2*len(tree))
	for _, fl := range tree {
		have[strings.ToLower(fl.Path)] = true
		have[strings.ToLower(path.Base(fl.Path))] = true
	}
	for _, q := range archdb.QuantChain(quant) {
		if have[strings.ToLower(sub(q))] {
			return sub(q)
		}
	}
	return sub(quant)
}

// matchFile prefers an exact path match over a basename match in a subfolder,
// so a caller may name a file either way and a root-level config wins.
func matchFile(files []types.HFFile, want string) (string, bool) {
	for _, f := range files {
		if f.Path == want {
			return f.Path, true
		}
	}
	for _, f := range files {
		if path.Base(f.Path) == want {
			return f.Path, true
		}
	}
	return "", false
}

// stepCountRe matches an explicit step count in a filename, e.g.
// "Lightning-4steps-V1.0" or "hyper-8step-lora".
var stepCountRe = regexp.MustCompile(`(?i)(?:^|[^0-9a-z])(\d{1,2})[-_ ]?steps?(?:$|[^0-9a-z])`)

// fewStepFamilies always ship a step-distillation adapter merged in, at the
// step count they are documented to run at: Phr00t's Qwen-Image-Edit-Rapid-AIO
// ("1 CFG, 4 step", its GGUF conversions carry no count) and lightx2v's
// Lightning line. Only consulted when the filename carries no count of its own.
var fewStepFamilies = map[string]int{"rapid": 4, "lightning": 4}

// fewStepDefaults reports the sampling defaults a step-distilled checkpoint
// needs, and nil for ordinary weights. Getting it wrong is not subtle: a 4-step
// merge sampled at its base architecture's 20 steps and cfg 2.5 returns burnt
// images, and nothing in the pull output would explain why.
func fewStepDefaults(file string) (map[string]any, string) {
	base := strings.ToLower(path.Base(file))
	steps := 0
	if m := stepCountRe.FindStringSubmatch(base); m != nil {
		n, err := strconv.Atoi(m[1])
		if err == nil && n > 0 && n <= 16 {
			steps = n
		}
	}
	if steps == 0 {
		for family, n := range fewStepFamilies {
			if strings.Contains(base, family) {
				steps = n
				break
			}
		}
	}
	if steps == 0 {
		return nil, ""
	}
	return map[string]any{"cfg_scale": 1.0, "steps": steps},
		fmt.Sprintf("looks step-distilled: defaulting to %d steps at cfg 1.0 (override per request with \"steps\"/\"cfg\")", steps)
}

// quantPrefs keeps a label archdb does not know verbatim, so a user naming an
// exotic quant by hand still matches.
func quantPrefs(pref []string) []archdb.Quant {
	out := make([]archdb.Quant, 0, len(pref))
	for _, p := range pref {
		if p == "" {
			continue
		}
		if q, ok := archdb.ParseQuant(p); ok {
			out = append(out, q)
		} else {
			out = append(out, archdb.Quant(p))
		}
	}
	return out
}

// selectQuant picks a candidate for the preference order: the first preference
// any candidate matches takes the first such candidate.
func selectQuant(candidates []string, prefs []archdb.Quant) (file string, quant archdb.Quant, ok bool) {
	labels := make([]archdb.Quant, len(candidates))
	for i, c := range candidates {
		labels[i], _ = archdb.ParseQuant(c)
	}
	for _, p := range prefs {
		for i, c := range candidates {
			unknown := labels[i] == "" && strings.Contains(strings.ToLower(path.Base(c)), strings.ToLower(string(p)))
			if labels[i] == p || unknown {
				return c, p, true
			}
		}
	}
	return "", "", false
}

// detectArch determines the architecture of a repo: first via the diffusers
// _class_name found in model_index.json (top-level "_class_name") or config.json
// ("_class_name" or "architectures"[0]) using archdb.Lookup; falling back to
// archdb.MatchKeyword over the repo id and the repo's filenames. ok=false if
// unknown. A non-nil error is only returned when a config read failed AND no
// architecture could otherwise be determined.
func detectArch(ctx context.Context, f RepoFetcher, repo string, files []types.HFFile) (arch archdb.Arch, ok bool, err error) {
	var lastErr error

	for _, name := range []string{"model_index.json", "config.json"} {
		p, found := matchFile(files, name)
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

// readClassName returns "_class_name", else the first of "architectures". A
// malformed body is "no class name" (non-fatal); only an I/O error is returned.
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

func diffusionCandidates(files []types.HFFile) []string {
	var out []string
	for _, fl := range files {
		if isWeightFile(fl.Path) && !containsAny(strings.ToLower(fl.Path), diffusionExcludeTokens) {
			out = append(out, fl.Path)
		}
	}
	return out
}

func findRoleFile(files []types.HFFile, role types.Role, exclude string) (string, bool) {
	kws := roleKeywords[role]
	if kws == nil {
		return "", false
	}
	// Every file against keyword[0] before keyword[1]: iterating files first
	// would let a broad keyword on an earlier file win — "clip" matching
	// clip_g.safetensors and handing CLIP-G to --clip_l.
	for _, kw := range kws {
		for _, fl := range files {
			if fl.Path == exclude || !isWeightFile(fl.Path) {
				continue
			}
			full := strings.ToLower(fl.Path)
			if containsAny(full, roleExcludeTokens[role]) {
				continue
			}
			if strings.Contains(full, kw) {
				return fl.Path, true
			}
		}
	}
	return "", false
}

func deriveName(repo string) string { return strings.ToLower(archdb.BaseOf(repo)) }

// foreignQuants are formats ggml cannot read. Naming the format beats calling
// the file "unquantized", which sends people looking for the wrong fix.
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

// shardRe matches "diffusion_pytorch_model-00001-of-00003.safetensors".
var shardRe = regexp.MustCompile(`-\d{5}-of-\d{5}\.`)

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

// ggufMirrors maps an architecture to a known GGUF mirror to suggest when a
// repo only ships weights sd-server cannot read.
var ggufMirrors = map[string]string{
	"flux":            "city96/FLUX.1-dev-gguf",
	"flux-kontext":    "QuantStack/FLUX.1-Kontext-dev-GGUF",
	"qwen-image":      "QuantStack/Qwen-Image-GGUF",
	"qwen-image-edit": "QuantStack/Qwen-Image-Edit-GGUF",
	"z-image":         "leejet/Z-Image-Turbo-GGUF",
}

func suggestedGGUFRepo(arch string) string { return ggufMirrors[arch] }

func isWeightFile(p string) bool {
	return hasAnySuffix(path.Base(p), ".gguf", ".safetensors")
}

func hasAnySuffix(name string, suffixes ...string) bool {
	low := strings.ToLower(name)
	return slices.ContainsFunc(suffixes, func(s string) bool { return strings.HasSuffix(low, s) })
}

func containsAny(s string, subs []string) bool {
	return slices.ContainsFunc(subs, func(sub string) bool { return strings.Contains(s, sub) })
}
