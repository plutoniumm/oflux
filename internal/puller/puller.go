// Package puller orchestrates installing a model: it resolves a friendly name
// or Hugging Face repo into a manifest (via the curated registry or the generic
// compatibility inspector), downloads every component into the content-addressed
// blob store (deduplicating shared encoders/VAEs across models), and writes the
// resolved manifest.
//
// It is the glue between internal/registry + internal/compat (what to install),
// internal/hfclient (fetch it), and internal/store (where it lives).
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

	"oflux/internal/compat"
	"oflux/internal/hfclient"
	"oflux/internal/registry"
	"oflux/internal/store"
	"oflux/internal/types"
)

// defaultQuantChain is the fallback preference appended after the user's
// requested quant when inspecting arbitrary repos.
var defaultQuantChain = []string{"Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0"}

// Puller installs models into a Store using a Hugging Face client.
type Puller struct {
	hf    *hfclient.Client
	store *store.Store

	mu       sync.Mutex
	inFlight map[string]bool // models currently being pulled
}

// New returns a Puller.
func New(hf *hfclient.Client, st *store.Store) *Puller {
	return &Puller{hf: hf, store: st, inFlight: make(map[string]bool)}
}

// claim marks name as being pulled, refusing a second concurrent pull of the
// same model (which would otherwise have two writers racing over the same
// components).
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

// Progress is an optional callback for human-readable progress lines.
type Progress func(msg string)

func (p Progress) emit(msg string) {
	if p != nil {
		p(msg)
	}
}

// IsCurated reports whether name is a curated registry model.
func IsCurated(name string) bool {
	_, ok := registry.Lookup(name)
	return ok
}

// Resolve decides what would be installed for nameOrRepo at the given quant,
// WITHOUT downloading anything. Curated names go through the registry; anything
// containing a "/" is treated as a Hugging Face repo and inspected for
// compatibility; otherwise it is an error.
func (p *Puller) Resolve(ctx context.Context, nameOrRepo, quant string) (types.Verdict, error) {
	if m, ok := registry.Resolve(nameOrRepo, quant); ok {
		return types.Verdict{Repo: nameOrRepo, Compatible: true, Manifest: &m}, nil
	}
	if strings.Contains(nameOrRepo, "/") {
		return compat.Inspect(ctx, p.hf, nameOrRepo, quantPref(quant))
	}
	return types.Verdict{}, fmt.Errorf("unknown model %q; pass a Hugging Face repo as org/name to inspect it", nameOrRepo)
}

// Pull resolves nameOrRepo, downloads every component (skipping blobs already
// present), writes the manifest, and returns it. An incompatible repo yields a
// descriptive error listing the blockers.
func (p *Puller) Pull(ctx context.Context, nameOrRepo, quant string, prog Progress) (types.Manifest, error) {
	v, err := p.Resolve(ctx, nameOrRepo, quant)
	if err != nil {
		return types.Manifest{}, err
	}
	if !v.Compatible || v.Manifest == nil {
		return types.Manifest{}, blockerError(nameOrRepo, v)
	}
	m := *v.Manifest

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

	tmpDir := filepath.Join(p.store.Root(), "tmp")
	if err := os.MkdirAll(tmpDir, 0o755); err != nil {
		return types.Manifest{}, err
	}

	trees := map[string][]types.HFFile{} // source repo -> file list, cached per pull
	for i := range m.Components {
		c := &m.Components[i]
		if c.Source == "" || c.File == "" {
			return types.Manifest{}, fmt.Errorf("component %s has no source/file", c.Role)
		}

		// Resolve the file's content hash + size up front so we can skip
		// downloading a blob we already have (shared encoders/VAEs). Only an
		// LFS entry carries a sha256; a plain git blob's oid is a sha1 and is
		// useless both as a blob address and as a download checksum.
		sha, size, verified := p.lookupFile(ctx, trees, c.Source, c.File)
		if verified {
			blob := store.BlobName(sha)
			if p.store.HasBlob(blob) {
				c.Blob, c.SHA256, c.Size = blob, sha, size
				prog.emit(fmt.Sprintf("✓ %s (cached) %s", c.Role, c.File))
				continue
			}
		} else {
			sha = "" // never pass a non-sha256 as the expected checksum
			prog.emit(fmt.Sprintf("! %s %s: no sha256 published; integrity check skipped", c.Role, c.File))
		}

		prog.emit(fmt.Sprintf("↓ %s %s from %s", c.Role, c.File, c.Source))
		// A unique temp name per attempt: two concurrent pulls that share a
		// component (a VAE, an encoder) must not write into the same file.
		tmp := filepath.Join(tmpDir, fmt.Sprintf("%d-%d-%s__%s", os.Getpid(), i, sanitize(c.Source), sanitize(c.File)))
		got, err := p.hf.Download(ctx, c.Source, "main", c.File, tmp, sha)
		if err != nil {
			return types.Manifest{}, fmt.Errorf("download %s/%s: %w", c.Source, c.File, hintGated(err))
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

// lookupFile finds a file in its source repo's tree (cached), returning its
// content sha256, size, and whether that hash is a real sha256 (LFS entries
// only). verified is false when the tree can't be fetched, the file isn't
// found, or the entry is a plain git blob — the caller then downloads without a
// pre-known checksum.
func (p *Puller) lookupFile(ctx context.Context, cache map[string][]types.HFFile, source, file string) (sha string, size int64, verified bool) {
	files, cached := cache[source]
	if !cached {
		f, err := p.hf.Tree(ctx, source, "")
		if err != nil {
			cache[source] = nil
			return "", 0, false
		}
		files, cache[source] = f, f
	}
	// Exact path wins over a basename match: a repo can hold several files with
	// the same basename in different directories (e.g. an archived copy under
	// old/), and matching the wrong one binds the component to wrong content.
	match := func(pred func(types.HFFile) bool) (types.HFFile, bool) {
		for _, f := range files {
			if pred(f) {
				return f, true
			}
		}
		return types.HFFile{}, false
	}
	f, ok := match(func(f types.HFFile) bool { return f.Path == file })
	if !ok {
		f, ok = match(func(f types.HFFile) bool { return path.Base(f.Path) == file })
	}
	if !ok {
		return "", 0, false
	}
	if !f.IsLFS || f.LFSOID == "" {
		return "", f.Size, false // git sha1 is not a content sha256
	}
	return f.LFSOID, f.Size, true
}

// quantPref builds the ordered quant preference: the requested quant first, then
// the default fallback chain, de-duplicated.
func quantPref(quant string) []string {
	seen := map[string]bool{}
	var out []string
	add := func(q string) {
		if q == "" || seen[q] {
			return
		}
		seen[q] = true
		out = append(out, q)
	}
	add(quant)
	for _, q := range defaultQuantChain {
		add(q)
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

// hintGated enriches an auth error with actionable guidance.
func hintGated(err error) error {
	if errors.Is(err, hfclient.ErrUnauthorized) {
		return fmt.Errorf("%w — this repo is gated/private; accept its license on Hugging Face and set an HF token (config hf_token)", err)
	}
	return err
}

func sanitize(s string) string {
	r := strings.NewReplacer("/", "_", "\\", "_", ":", "_", " ", "_")
	return r.Replace(s)
}
