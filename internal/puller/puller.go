// Package puller installs a model: it resolves a name or Hugging Face repo into
// a manifest, downloads every component into the content-addressed blob store
// (deduplicating shared encoders/VAEs), and writes the manifest. It is the glue
// between registry + compat (what to install), hfclient (fetch it) and store.
package puller

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"
	"sync"

	"oflux/internal/archdb"
	"oflux/internal/compat"
	"oflux/internal/hfclient"
	"oflux/internal/registry"
	"oflux/internal/store"
	"oflux/internal/types"
)

type Puller struct {
	hf    *hfclient.Client
	store *store.Store

	mu       sync.Mutex
	inFlight map[string]bool // models currently being pulled
}

func New(hf *hfclient.Client, st *store.Store) *Puller {
	return &Puller{hf: hf, store: st, inFlight: make(map[string]bool)}
}

// HF is the Hub client the puller was built with. Searching the Hub goes
// through it rather than a second client, so one token — and one test base
// URL — covers both.
func (p *Puller) HF() *hfclient.Client { return p.hf }

// claim refuses a second concurrent pull of the same model: two writers would
// race over the same components.
func (p *Puller) claim(name string) (release func(), err error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.inFlight[name] {
		return nil, fmt.Errorf("%s is already being pulled", name)
	}
	p.inFlight[name] = true
	return func() {
		p.mu.Lock()
		delete(p.inFlight, name)
		p.mu.Unlock()
	}, nil
}

type Progress func(msg string)

func (p Progress) emit(msg string) {
	if p != nil {
		p(msg)
	}
}

// Resolve decides what would be installed, WITHOUT downloading anything: a
// curated name through the registry, anything containing a "/" as a repo to
// inspect. file pins the diffusion weights in a repo that publishes many
// builds, and is meaningless for a curated model, which pins its own.
func (p *Puller) Resolve(ctx context.Context, nameOrRepo, quant, file string) (types.Verdict, error) {
	if m, ok := registry.Resolve(nameOrRepo, quant); ok {
		if file != "" {
			return types.Verdict{}, fmt.Errorf("%s is a curated model and already pins its weights; --file only applies to a Hugging Face repo", nameOrRepo)
		}
		return types.Verdict{Repo: nameOrRepo, Compatible: true, Manifest: &m}, nil
	}
	if strings.Contains(nameOrRepo, "/") {
		return compat.InspectFile(ctx, p.hf, nameOrRepo, quantPref(quant), file)
	}
	return types.Verdict{}, fmt.Errorf("unknown model %q; pass a Hugging Face repo as org/name to inspect it", nameOrRepo)
}

// Opts carries the optional knobs for a pull.
type Opts struct {
	// File pins the exact diffusion weights within a multi-build repo.
	File string
	// ControlNet is a Hugging Face repo whose weights are attached to the
	// installed model and loaded with it; ControlNetFile pins one of several.
	ControlNet     string
	ControlNetFile string
	// As installs under this name instead of the derived one. Useful when the
	// same base model is installed more than once with different attachments.
	As string
}

// Pull downloads every component (skipping blobs already present) and writes
// the manifest. An incompatible repo yields an error listing the blockers.
func (p *Puller) Pull(ctx context.Context, nameOrRepo, quant string, opts Opts, prog Progress) (types.Manifest, error) {
	v, err := p.Resolve(ctx, nameOrRepo, quant, opts.File)
	if err != nil {
		return types.Manifest{}, err
	}
	if !v.Compatible || v.Manifest == nil {
		return types.Manifest{}, blockerError(nameOrRepo, v)
	}
	m := *v.Manifest

	if opts.ControlNet != "" {
		if err := compat.AttachControlNet(ctx, p.hf, &m, opts.ControlNet, opts.ControlNetFile); err != nil {
			return types.Manifest{}, err
		}
	}
	if opts.As != "" {
		if err := store.ValidModelName(opts.As); err != nil {
			return types.Manifest{}, err
		}
		m.Name = opts.As
	}

	release, err := p.claim(m.Name)
	if err != nil {
		return types.Manifest{}, err
	}
	defer release()
	// Hold off garbage collection until the manifest is committed, so blobs we
	// download here can't be swept by a concurrent `oflux rm`.
	defer p.store.BeginWrite()()
	for _, note := range v.Notes {
		prog.emit(note)
	}

	trees := map[string][]types.HFFile{} // source repo -> file list, cached per pull
	for i := range m.Components {
		c := &m.Components[i]
		if c.Source == "" || c.File == "" {
			return types.Manifest{}, fmt.Errorf("component %s has no source/file", c.Role)
		}
		label := fmt.Sprintf("%s %s", c.Role, c.File)

		// Resolve the content hash up front so a blob we already have (a shared
		// encoder or VAE) is never downloaded twice. compat walked the repo tree
		// to resolve the component and carries what it saw; only one that came
		// without it costs a tree fetch here.
		sha, size := p.componentSHA(ctx, trees, *c)
		if sha == "" {
			prog.emit(fmt.Sprintf("! %s: no sha256 published; integrity check skipped", label))
		} else if blob := store.BlobName(sha); p.store.HasBlob(blob) {
			c.Blob, c.SHA256, c.Size = blob, sha, size
			prog.emit(fmt.Sprintf("✓ %s (cached) %s", c.Role, c.File))
			continue
		}

		tmpName := fmt.Sprintf("%d-%d-%s__%s", os.Getpid(), i, sanitize(c.Source), sanitize(c.File))
		tmp, got, err := p.download(ctx, c.Source, c.File, sha, label, tmpName, prog)
		if err != nil {
			return types.Manifest{}, err
		}
		blob, err := p.store.PutBlob(got, tmp)
		if err != nil {
			return types.Manifest{}, fmt.Errorf("store %s: %w", c.File, err)
		}
		c.Blob, c.SHA256 = blob, got
		if size > 0 {
			c.Size = size
		} else if fi, statErr := os.Stat(p.store.BlobPath(blob)); statErr == nil {
			c.Size = fi.Size()
		}
	}

	if err := p.store.WriteManifest(m); err != nil {
		return types.Manifest{}, err
	}
	prog.emit(fmt.Sprintf("installed %s (%s, %s)", m.Name, m.Architecture, m.Mode))
	return m, nil
}

// download returns the temp path plus the content's sha256. A non-empty sha is
// the checksum the transfer is verified against.
func (p *Puller) download(ctx context.Context, source, file, sha, label, tmpName string, prog Progress) (tmp, sum string, err error) {
	tmpDir := filepath.Join(p.store.Root(), "tmp")
	if err := os.MkdirAll(tmpDir, 0o755); err != nil {
		return "", "", err
	}
	// tmpName is unique per attempt: two concurrent pulls that share a
	// component (a VAE, an encoder) must not write into the same file.
	tmp = filepath.Join(tmpDir, tmpName)
	prog.emit(fmt.Sprintf("↓ %s from %s", label, source))
	sum, err = p.hf.Download(ctx, source, "main", file, tmp, sha)
	if err != nil {
		_ = os.Remove(tmp)
		return "", "", fmt.Errorf("download %s/%s: %w", source, file, hintGated(err))
	}
	return tmp, sum, nil
}

func (p *Puller) componentSHA(ctx context.Context, trees map[string][]types.HFFile, c types.Component) (sha string, size int64) {
	if isSHA256(c.SHA256) {
		return c.SHA256, c.Size
	}
	sha, size = p.publishedSHA(ctx, trees, c.Source, c.File)
	if size == 0 {
		size = c.Size
	}
	return sha, size
}

// isSHA256 guards against a component carrying a git blob oid instead — a
// sha1, which as an expected checksum fails every transfer it guards.
func isSHA256(s string) bool {
	if len(s) != 64 {
		return false
	}
	for i := 0; i < len(s); i++ {
		if c := s[i]; !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f' || c >= 'A' && c <= 'F') {
			return false
		}
	}
	return true
}

// publishedSHA returns the checksum source/file can be verified against, plus
// the published size (0 when there is none). Only an LFS entry carries a
// sha256; a plain git blob's oid is a sha1 of the pointer, useless as both a
// blob address and a checksum, so such a file is fetched unverified.
func (p *Puller) publishedSHA(ctx context.Context, trees map[string][]types.HFFile, source, file string) (sha string, size int64) {
	files, err := p.repoTree(ctx, trees, source)
	if err != nil {
		return "", 0
	}
	f, ok := matchPath(files, func(f types.HFFile) string { return f.Path }, file)
	if !ok {
		return "", 0
	}
	if f.IsLFS && f.LFSOID != "" {
		return f.LFSOID, f.Size
	}
	return "", f.Size
}

// repoTree fetches each repo at most once per pull: components often share a
// repo, and a LoRA pull inspects the tree it then downloads from.
func (p *Puller) repoTree(ctx context.Context, cache map[string][]types.HFFile, source string) ([]types.HFFile, error) {
	if files, ok := cache[source]; ok {
		return files, nil
	}
	files, err := p.hf.Tree(ctx, source, "")
	if err != nil {
		cache[source] = nil // don't re-ask a repo that just failed
		return nil, err
	}
	cache[source] = files
	return files, nil
}

// matchPath prefers an exact path over a basename match: a repo can hold the
// same basename in several directories (an archived copy under old/), and the
// wrong one binds a component to the wrong content.
func matchPath[T any](items []T, pathOf func(T) string, want string) (T, bool) {
	for _, it := range items {
		if pathOf(it) == want {
			return it, true
		}
	}
	for _, it := range items {
		if path.Base(pathOf(it)) == want {
			return it, true
		}
	}
	var zero T
	return zero, false
}

// quantPref orders the quantizations to try: the requested one first, then
// archdb's chain. The order lives in archdb, beside the arch table that decides
// which companion quants exist, so there is one such list and not four.
func quantPref(quant string) []string {
	want, known := archdb.ParseQuant(quant)
	var out []string
	if quant != "" && !known {
		// An unknown label is still worth trying first: the user named a build
		// they can see in the repo, and SelectQuant matches it textually.
		out = append(out, quant)
	}
	for _, q := range archdb.QuantChain(want) {
		out = append(out, string(q))
	}
	return out
}

func blockerError(repo string, v types.Verdict) error {
	if len(v.Blockers) == 0 {
		return fmt.Errorf("%s is not compatible", repo)
	}
	var b strings.Builder
	fmt.Fprintf(&b, "%s is not compatible with the sd-server engine:", repo)
	for _, bl := range v.Blockers {
		fmt.Fprintf(&b, "\n  - [%s] %s", bl.Kind, bl.Detail)
		if bl.Suggest != "" {
			fmt.Fprintf(&b, " → try %s", bl.Suggest)
		}
	}
	return errors.New(b.String())
}

func hintGated(err error) error {
	if errors.Is(err, hfclient.ErrUnauthorized) {
		return fmt.Errorf("%w — this repo is gated/private; accept its license on Hugging Face and set an HF token (config hf_token)", err)
	}
	return err
}

// tmpNameUnsafe must not reach a temp-file name.
var tmpNameUnsafe = strings.NewReplacer("/", "_", "\\", "_", ":", "_", " ", "_")

func sanitize(s string) string { return tmpNameUnsafe.Replace(s) }
