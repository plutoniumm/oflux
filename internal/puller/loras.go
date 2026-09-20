package puller

import (
	"context"
	"fmt"
	"os"
	"path"
	"slices"
	"strings"

	"oflux/internal/registry"
	"oflux/internal/store"
	"oflux/internal/types"
)

// maxLoraCandidates bounds how many filenames an ambiguity error lists.
const maxLoraCandidates = 12

// LoraOpts carries the optional knobs for a LoRA pull.
type LoraOpts struct {
	// File pins the exact adapter in a repo that publishes several.
	File string
	// As installs under this name instead of the derived one.
	As string
	// For, Steps and CFG go into the adapter's metadata sidecar. A curated
	// LoRA seeds all three from the registry and these override it; for an
	// arbitrary repo they are all oflux will ever know.
	For   []string
	Steps int
	CFG   float64
}

// PullLora installs a LoRA adapter and returns the name it was installed under.
//
// nameOrRepo is either a curated LoRA name or a Hugging Face repo id. For a
// repo, opts.File pins the exact adapter within it; when it is empty the repo
// must contain exactly one .safetensors, otherwise the candidates are reported
// so the caller can choose.
func (p *Puller) PullLora(ctx context.Context, nameOrRepo string, opts LoraOpts, prog Progress) (string, error) {
	trees := map[string][]types.HFFile{} // shared with resolveLora: one tree fetch
	meta, err := p.resolveLora(ctx, trees, nameOrRepo, opts)
	if err != nil {
		return "", err
	}
	if err := store.ValidLoraName(meta.Name); err != nil {
		return "", err
	}

	// A LoRA is stored under its name, not its content hash, so a re-pull always
	// re-downloads. They are small (under ~2 GB) and the name is the identity.
	sha, _ := p.publishedSHA(ctx, trees, meta.Source, meta.File)
	tmpName := fmt.Sprintf("lora-%d-%s", os.Getpid(), sanitize(meta.Name))
	tmp, _, err := p.download(ctx, meta.Source, meta.File, sha, "lora "+meta.File, tmpName, prog)
	if err != nil {
		return "", err
	}
	if err := p.store.PutLora(meta.Name, tmp); err != nil {
		_ = os.Remove(tmp)
		return "", err
	}
	if err := p.store.WriteLoraMeta(meta); err != nil {
		return "", fmt.Errorf("lora %s installed, but recording its metadata failed: %w", meta.Name, err)
	}
	prog.emit(fmt.Sprintf("installed lora %s", meta.Name))
	return meta.Name, nil
}

// resolveLora decides which repo file to fetch and what to record about it,
// caching any repo tree it fetches in trees for the download that follows.
func (p *Puller) resolveLora(ctx context.Context, trees map[string][]types.HFFile, nameOrRepo string, opts LoraOpts) (store.LoraMeta, error) {
	meta := store.LoraMeta{For: opts.For, Steps: opts.Steps, CFG: opts.CFG}
	if l, ok := registry.LookupLora(nameOrRepo); ok {
		meta.Name, meta.Source, meta.File = l.Name, l.Source, l.File
		if opts.File != "" {
			meta.File = opts.File
		}
		if opts.As != "" {
			meta.Name = opts.As
		}
		// The registry pins the sampling regime a curated adapter is distilled
		// to; the caller only overrides it.
		if len(meta.For) == 0 {
			meta.For = slices.Clone(l.Archs)
		}
		if meta.Steps == 0 {
			meta.Steps = l.Steps
		}
		if meta.CFG == 0 {
			meta.CFG = l.CFG
		}
		return meta, nil
	}
	if !strings.Contains(nameOrRepo, "/") {
		return meta, fmt.Errorf("unknown lora %q; pass a Hugging Face repo as org/name, or one of: %s",
			nameOrRepo, strings.Join(registry.LoraNames(), ", "))
	}

	meta.Source = nameOrRepo
	files, terr := p.repoTree(ctx, trees, meta.Source)
	if terr != nil {
		return meta, fmt.Errorf("inspect %s: %w", meta.Source, hintGated(terr))
	}
	var candidates []types.HFFile
	for _, f := range files {
		if strings.HasSuffix(strings.ToLower(f.Path), store.LoraExt) {
			candidates = append(candidates, f)
		}
	}
	switch {
	case opts.File != "":
		f, ok := matchPath(candidates, func(f types.HFFile) string { return f.Path }, opts.File)
		if !ok {
			return meta, fmt.Errorf("%s has no file %q%s", meta.Source, opts.File, listCandidates(candidates))
		}
		meta.File = f.Path
	case len(candidates) == 0:
		return meta, fmt.Errorf("%s contains no %s file — LoRAs must be safetensors", meta.Source, store.LoraExt)
	case len(candidates) == 1:
		meta.File = candidates[0].Path
	default:
		return meta, fmt.Errorf("%s contains %d adapters; pick one with --file%s",
			meta.Source, len(candidates), listCandidates(candidates))
	}

	meta.Name = opts.As
	if meta.Name == "" {
		meta.Name = LoraNameFrom(strings.TrimSuffix(path.Base(meta.File), store.LoraExt))
	}
	return meta, nil
}

func listCandidates(candidates []types.HFFile) string {
	if len(candidates) == 0 {
		return ""
	}
	shown := candidates
	suffix := ""
	if len(shown) > maxLoraCandidates {
		shown = shown[:maxLoraCandidates]
		suffix = fmt.Sprintf("\n  … and %d more", len(candidates)-maxLoraCandidates)
	}
	var b strings.Builder
	for _, f := range shown {
		b.WriteString("\n  " + f.Path)
	}
	b.WriteString(suffix)
	return b.String()
}

// LoraNameFrom derives a valid LoRA name from an arbitrary filename stem:
// lowercased, with runs of unsupported characters collapsed to a single dash.
func LoraNameFrom(s string) string {
	var b strings.Builder
	dash := false
	for _, r := range strings.ToLower(s) {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9', r == '.', r == '_':
			b.WriteRune(r)
			dash = false
		default:
			if !dash && b.Len() > 0 {
				b.WriteByte('-')
				dash = true
			}
		}
	}
	out := strings.Trim(b.String(), "-._")
	if len(out) > 64 {
		out = strings.Trim(out[:64], "-._")
	}
	return out
}
