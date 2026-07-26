package compat

import (
	"context"
	"fmt"
	"path"
	"slices"
	"strings"

	"oflux/internal/archdb"
	"oflux/internal/types"
)

// controlNetExts are the weight formats sd-server will load as a ControlNet.
var controlNetExts = []string{".safetensors", ".gguf", ".pth", ".ckpt", ".bin"}

// AttachControlNet resolves a ControlNet in its Hugging Face repo and adds it to
// the manifest, both as a component and as a --control-net launch flag.
//
// sd-server takes the ControlNet only at startup and offers no endpoint to load
// or swap one, so it belongs to the installed model rather than to a request.
// That is also why this is an explicit opt-in at pull time: attaching one to
// every model would load weights nothing asked for.
//
// file pins the exact weights when the repo publishes several; with file empty
// the repo must contain exactly one candidate.
func AttachControlNet(ctx context.Context, f RepoFetcher, m *types.Manifest, repo, file string) error {
	if m == nil {
		return fmt.Errorf("no manifest to attach a control net to")
	}
	if !strings.Contains(repo, "/") {
		return fmt.Errorf("control net %q is not a Hugging Face repo (want org/name)", repo)
	}

	files, err := f.Tree(ctx, repo, "")
	if err != nil {
		return fmt.Errorf("inspect control net %s: %w", repo, err)
	}
	var candidates []string
	for _, fl := range files {
		low := strings.ToLower(path.Base(fl.Path))
		for _, ext := range controlNetExts {
			if strings.HasSuffix(low, ext) {
				candidates = append(candidates, fl.Path)
				break
			}
		}
	}

	var chosen string
	switch {
	case file != "":
		var ok bool
		if chosen, ok = matchFile(files, file); !ok {
			return fmt.Errorf("%s has no file %q%s", repo, file, listPaths(candidates))
		}
	case len(candidates) == 0:
		return fmt.Errorf("%s contains no loadable control net weights", repo)
	case len(candidates) == 1:
		chosen = candidates[0]
	default:
		return fmt.Errorf("%s contains %d candidates; pick one with --control-net-file%s",
			repo, len(candidates), listPaths(candidates))
	}

	// Replace rather than append, so re-attaching does not stack components.
	m.Components = slices.DeleteFunc(m.Components, func(c types.Component) bool {
		return c.Role == types.RoleControlNet
	})
	m.Components = append(m.Components, types.Component{
		Role:   types.RoleControlNet,
		File:   chosen,
		Source: repo,
	})

	// The flag has to be inserted before the sampling flags rather than appended,
	// so argv keeps role flags and value placeholders adjacent.
	if !slices.Contains(m.Engine.Flags, "{control_net}") {
		flags := slices.Clone(m.Engine.Flags)
		at := len(flags)
		for i, fl := range flags {
			if strings.HasPrefix(fl, "--cfg-scale") || strings.HasPrefix(fl, "--steps") ||
				strings.HasPrefix(fl, "--flow-shift") || strings.HasPrefix(fl, "--sampling-method") {
				at = i
				break
			}
		}
		m.Engine.Flags = slices.Insert(flags, at,
			archdb.FlagName(types.RoleControlNet), "{control_net}")
	}
	return nil
}

func listPaths(paths []string) string {
	if len(paths) == 0 {
		return ""
	}
	shown := paths
	suffix := ""
	if len(shown) > 12 {
		shown, suffix = shown[:12], fmt.Sprintf("\n  … and %d more", len(paths)-12)
	}
	return "\n  " + strings.Join(shown, "\n  ") + suffix
}
