package archdb

import (
	"regexp"
	"strings"

	"oflux/internal/types"
)

// BuildOpts carries everything Build cannot derive from the architecture.
type BuildOpts struct {
	Name    string   // manifest name, e.g. "qwen-image-edit"
	Repo    string   // repo the diffusion weights come from = checkpoint identity
	Hints   []string // more strings to mine for a release tag, e.g. the filename
	Engine  types.EngineSpec
	Resolve func(types.Role) (types.Component, bool) // ok=false: nobody publishes it
}

// Build walks arch.Roles() and resolves each one, turning an unresolvable
// required role into a Blocker and leaving an unresolvable optional one out.
// It is the one walk both the curated registry and the compatibility checker
// do; they differ only in how a role resolves and what they do with blockers.
func Build(a Arch, o BuildOpts) (types.Manifest, []types.Blocker) {
	roles := a.Roles()
	comps := make([]types.Component, 0, len(roles))
	var blockers []types.Blocker
	for _, role := range roles {
		c, ok := o.Resolve(role)
		switch {
		case ok:
			comps = append(comps, c)
		case a.requires(role):
			blockers = append(blockers, types.Blocker{
				Kind:   types.BlockerMissingRole,
				Role:   role,
				Detail: "required component not present in repo and no known companion source",
			})
		}
	}
	return types.Manifest{
		Name:         o.Name,
		Architecture: a.Name,
		Mode:         a.Mode,
		Base:         BaseOf(o.Repo),
		Revision:     RevisionOf(append([]string{o.Repo}, o.Hints...)...),
		Components:   comps,
		Engine:       o.Engine,
	}, blockers
}

// BaseOf turns a repo id into the upstream checkpoint identity: the last path
// segment without the GGUF-conversion suffix, as published. The manifest name
// is made for the command line ("qwen-image-edit") and hides which checkpoint
// that actually is; this says "Qwen-Image-Edit-2511".
func BaseOf(repo string) string {
	seg := repo
	if i := strings.LastIndex(repo, "/"); i >= 0 {
		seg = repo[i+1:]
	}
	for _, suf := range []string{"-gguf", "_gguf", ".gguf"} {
		if s, cut := cutSuffixFold(seg, suf); cut {
			return s
		}
	}
	return seg
}

// releaseTagRe matches a YYMM release tag, how publishers date successive
// checkpoints of one architecture (2509 and 2511 are both "qwen-image-edit").
// Narrow on purpose: four-digit numbers are everywhere in weight filenames, so
// only a plausible year and month standing alone between separators counts.
var releaseTagRe = regexp.MustCompile(`(?:^|[^0-9a-z])(2[3-9](?:0[1-9]|1[0-2]))(?:$|[^0-9a-z])`)

// RevisionOf returns the release tag from the first hint that carries one.
func RevisionOf(hints ...string) string {
	for _, h := range hints {
		if m := releaseTagRe.FindStringSubmatch(strings.ToLower(h)); m != nil {
			return m[1]
		}
	}
	return ""
}

func cutSuffixFold(s, suffix string) (string, bool) {
	if len(s) >= len(suffix) && strings.EqualFold(s[len(s)-len(suffix):], suffix) {
		return s[:len(s)-len(suffix)], true
	}
	return s, false
}
