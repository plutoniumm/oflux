package puller

import (
	"context"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"

	"oflux/internal/registry"
	"oflux/internal/store"
	"oflux/internal/types"
)

// maxLoraCandidates bounds how many filenames an ambiguity error lists.
const maxLoraCandidates = 12

// PullLora installs a LoRA adapter and returns the name it was installed under.
//
// nameOrRepo is either a curated LoRA name or a Hugging Face repo id. For a
// repo, file pins the exact adapter within it; when file is empty the repo must
// contain exactly one .safetensors, otherwise the candidates are reported so
// the caller can choose. as overrides the installed name.
func (p *Puller) PullLora(ctx context.Context, nameOrRepo, file, as string, prog Progress) (string, error) {
	source, repoFile, name, err := p.resolveLora(ctx, nameOrRepo, file, as)
	if err != nil {
		return "", err
	}
	if err := store.ValidLoraName(name); err != nil {
		return "", err
	}

	// A LoRA is stored under its name, not its content hash, so a re-pull always
	// re-downloads. They are small (under ~2 GB) and the name is the identity.
	sha, _, verified := p.lookupFile(ctx, map[string][]types.HFFile{}, source, repoFile)
	if !verified {
		sha = ""
		prog.emit(fmt.Sprintf("! %s: no sha256 published; integrity check skipped", repoFile))
	}

	tmpDir := filepath.Join(p.store.Root(), "tmp")
	if err := os.MkdirAll(tmpDir, 0o755); err != nil {
		return "", err
	}
	tmp := filepath.Join(tmpDir, fmt.Sprintf("lora-%d-%s", os.Getpid(), sanitize(name)))

	prog.emit(fmt.Sprintf("↓ lora %s from %s", repoFile, source))
	if _, err := p.hf.Download(ctx, source, "main", repoFile, tmp, sha); err != nil {
		_ = os.Remove(tmp)
		return "", fmt.Errorf("download %s/%s: %w", source, repoFile, hintGated(err))
	}
	if err := p.store.PutLora(name, tmp); err != nil {
		_ = os.Remove(tmp)
		return "", err
	}
	prog.emit(fmt.Sprintf("installed lora %s", name))
	return name, nil
}

// resolveLora decides which repo file to fetch and what to call it.
func (p *Puller) resolveLora(ctx context.Context, nameOrRepo, file, as string) (source, repoFile, name string, err error) {
	if l, ok := registry.LookupLora(nameOrRepo); ok {
		name = l.Name
		if as != "" {
			name = as
		}
		if file != "" {
			return l.Source, file, name, nil
		}
		return l.Source, l.File, name, nil
	}
	if !strings.Contains(nameOrRepo, "/") {
		return "", "", "", fmt.Errorf("unknown lora %q; pass a Hugging Face repo as org/name, or one of: %s",
			nameOrRepo, strings.Join(registry.LoraNames(), ", "))
	}

	source = nameOrRepo
	files, terr := p.hf.Tree(ctx, source, "")
	if terr != nil {
		return "", "", "", fmt.Errorf("inspect %s: %w", source, hintGated(terr))
	}
	var candidates []string
	for _, f := range files {
		if strings.HasSuffix(strings.ToLower(f.Path), store.LoraExt) {
			candidates = append(candidates, f.Path)
		}
	}
	switch {
	case file != "":
		repoFile = matchRepoFile(candidates, file)
		if repoFile == "" {
			return "", "", "", fmt.Errorf("%s has no file %q%s", source, file, listCandidates(candidates))
		}
	case len(candidates) == 0:
		return "", "", "", fmt.Errorf("%s contains no %s file — LoRAs must be safetensors", source, store.LoraExt)
	case len(candidates) == 1:
		repoFile = candidates[0]
	default:
		return "", "", "", fmt.Errorf("%s contains %d adapters; pick one with --file%s",
			source, len(candidates), listCandidates(candidates))
	}

	name = as
	if name == "" {
		name = LoraNameFrom(strings.TrimSuffix(path.Base(repoFile), store.LoraExt))
	}
	return source, repoFile, name, nil
}

// matchRepoFile resolves a user-supplied file against the repo's adapters,
// accepting either the full path or a bare basename.
func matchRepoFile(candidates []string, file string) string {
	for _, c := range candidates {
		if c == file {
			return c
		}
	}
	for _, c := range candidates {
		if path.Base(c) == file {
			return c
		}
	}
	return ""
}

func listCandidates(candidates []string) string {
	if len(candidates) == 0 {
		return ""
	}
	shown := candidates
	suffix := ""
	if len(shown) > maxLoraCandidates {
		shown = shown[:maxLoraCandidates]
		suffix = fmt.Sprintf("\n  … and %d more", len(candidates)-maxLoraCandidates)
	}
	return "\n  " + strings.Join(shown, "\n  ") + suffix
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
