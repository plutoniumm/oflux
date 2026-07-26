package compat

import (
	"context"
	"slices"
	"strings"
	"testing"

	"oflux/internal/archdb"
	"oflux/internal/types"
)

// fakeRepo serves a fixed file list for any repo.
type fakeRepo struct{ paths []string }

func (f fakeRepo) Tree(ctx context.Context, repo, rev string) ([]types.HFFile, error) {
	out := make([]types.HFFile, 0, len(f.paths))
	for _, p := range f.paths {
		out = append(out, types.HFFile{Path: p, IsLFS: true, LFSOID: "abc", Size: 1})
	}
	return out, nil
}

func (f fakeRepo) ReadFile(ctx context.Context, repo, rev, p string, max int64) ([]byte, error) {
	return nil, context.Canceled
}

func sd15Manifest(t *testing.T) types.Manifest {
	t.Helper()
	arch, ok := archdb.ByName("sdxl")
	if !ok {
		t.Fatal("sdxl arch missing")
	}
	return types.Manifest{
		Name:         "sd15",
		Architecture: arch.Name,
		Mode:         arch.Mode,
		Components:   []types.Component{{Role: types.RoleDiffusion, File: "sd15.safetensors", Source: "org/sd15"}},
		Engine:       arch.EngineSpec(),
	}
}

func TestAttachControlNet(t *testing.T) {
	m := sd15Manifest(t)
	repo := fakeRepo{paths: []string{"README.md", "diffusion_pytorch_model.safetensors"}}

	if err := AttachControlNet(context.Background(), repo, &m, "lllyasviel/control_v11p_sd15_canny", ""); err != nil {
		t.Fatalf("AttachControlNet: %v", err)
	}

	c, ok := m.Component(types.RoleControlNet)
	if !ok {
		t.Fatal("no control_net component was added")
	}
	if c.Source != "lllyasviel/control_v11p_sd15_canny" || c.File != "diffusion_pytorch_model.safetensors" {
		t.Errorf("component = %+v", c)
	}

	// The flag must be present with its placeholder immediately after it, or the
	// supervisor substitutes a path into the wrong argv slot.
	i := slices.Index(m.Engine.Flags, "--control-net")
	if i < 0 {
		t.Fatalf("--control-net missing from flags: %q", m.Engine.Flags)
	}
	if i+1 >= len(m.Engine.Flags) || m.Engine.Flags[i+1] != "{control_net}" {
		t.Fatalf("placeholder not adjacent to flag: %q", m.Engine.Flags)
	}
	// It must land before the sampling flags, which take bare values.
	if j := slices.Index(m.Engine.Flags, "--cfg-scale"); j >= 0 && i > j {
		t.Errorf("--control-net came after sampling flags: %q", m.Engine.Flags)
	}
}

func TestAttachControlNetIsIdempotent(t *testing.T) {
	m := sd15Manifest(t)
	repo := fakeRepo{paths: []string{"cn.safetensors"}}
	for range 3 {
		if err := AttachControlNet(context.Background(), repo, &m, "org/cn", ""); err != nil {
			t.Fatal(err)
		}
	}
	n := 0
	for _, c := range m.Components {
		if c.Role == types.RoleControlNet {
			n++
		}
	}
	if n != 1 {
		t.Errorf("got %d control_net components, want 1", n)
	}
	if got := strings.Count(strings.Join(m.Engine.Flags, " "), "--control-net"); got != 1 {
		t.Errorf("--control-net appears %d times", got)
	}
}

func TestAttachControlNetAmbiguous(t *testing.T) {
	m := sd15Manifest(t)
	repo := fakeRepo{paths: []string{"canny.safetensors", "depth.safetensors"}}

	err := AttachControlNet(context.Background(), repo, &m, "org/cn", "")
	if err == nil {
		t.Fatal("a multi-candidate repo must not silently pick one")
	}
	// The error has to name the candidates, or the user cannot act on it.
	for _, want := range []string{"canny.safetensors", "depth.safetensors", "--control-net-file"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q missing %q", err, want)
		}
	}

	// Pinning one resolves it.
	if err := AttachControlNet(context.Background(), repo, &m, "org/cn", "depth.safetensors"); err != nil {
		t.Fatalf("pinned attach: %v", err)
	}
	c, _ := m.Component(types.RoleControlNet)
	if c.File != "depth.safetensors" {
		t.Errorf("file = %q, want depth.safetensors", c.File)
	}
}

func TestAttachControlNetErrors(t *testing.T) {
	m := sd15Manifest(t)
	repo := fakeRepo{paths: []string{"cn.safetensors"}}

	if err := AttachControlNet(context.Background(), repo, &m, "not-a-repo", ""); err == nil {
		t.Error("a bare name is not a Hugging Face repo")
	}
	if err := AttachControlNet(context.Background(), repo, &m, "org/cn", "missing.safetensors"); err == nil {
		t.Error("an unknown file must error")
	}
	if err := AttachControlNet(context.Background(), fakeRepo{paths: []string{"README.md"}}, &m, "org/cn", ""); err == nil {
		t.Error("a repo with no weights must error")
	}
}

func TestQuantFallbacks(t *testing.T) {
	// A bare K-quant must be able to reach its sized siblings: diffusion repos
	// publish "Q4_K" where encoder repos publish "Q4_K_M"/"Q4_K_S".
	got := quantFallbacks("Q4_K")
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

	// And the reverse: a sized label falls back to the bare one.
	got = quantFallbacks("Q5_K_M")
	if got[0] != "Q5_K_M" || !slices.Contains(got, "Q5_K") {
		t.Errorf("Q5_K_M should fall back to Q5_K: %v", got)
	}

	// Every chain ends in a general-quality ladder so something always resolves.
	for _, q := range []string{"Q4_K", "Q8_0", "weird"} {
		if !slices.Contains(quantFallbacks(q), "Q4_K_M") {
			t.Errorf("%s chain has no general fallback: %v", q, quantFallbacks(q))
		}
	}
}

// The real failure this prevents: the diffusion repo publishes Q4_K, the
// encoder repo does not, and the 404 only surfaces after gigabytes of download.
func TestCompanionQuantFallsBackToWhatTheRepoHas(t *testing.T) {
	comp := archdb.Companion{
		Source:      "mradermacher/Qwen2.5-VL-7B-Instruct-GGUF",
		FilePattern: "Qwen2.5-VL-7B-Instruct.{quant}.gguf",
		Quantized:   true,
	}
	encoder := fakeRepo{paths: []string{
		"Qwen2.5-VL-7B-Instruct.Q4_K_M.gguf",
		"Qwen2.5-VL-7B-Instruct.Q4_K_S.gguf",
		"Qwen2.5-VL-7B-Instruct.Q8_0.gguf",
	}}
	trees := map[string][]types.HFFile{}

	got := companionFile(context.Background(), encoder, trees, comp, "Q4_K")
	if got != "Qwen2.5-VL-7B-Instruct.Q4_K_M.gguf" {
		t.Errorf("Q4_K should resolve to the Q4_K_M the repo actually has, got %q", got)
	}

	// An exact match is still preferred over any fallback.
	if got := companionFile(context.Background(), encoder, trees, comp, "Q8_0"); got != "Qwen2.5-VL-7B-Instruct.Q8_0.gguf" {
		t.Errorf("Q8_0 = %q", got)
	}

	// A non-quantized companion is passed through untouched.
	fixed := archdb.Companion{Source: "org/vae", FilePattern: "ae.safetensors"}
	if got := companionFile(context.Background(), encoder, trees, fixed, "Q4_K"); got != "ae.safetensors" {
		t.Errorf("fixed companion = %q", got)
	}
}
