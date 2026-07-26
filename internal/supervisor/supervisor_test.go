package supervisor

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"slices"
	"testing"
	"time"

	"oflux/internal/engineclient"
	"oflux/internal/types"
)

// fakeEnginePath is the compiled path to testdata/fakeengine, built in TestMain.
var fakeEnginePath string

func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "fakeengine-bin")
	if err != nil {
		fmt.Fprintln(os.Stderr, "mktemp:", err)
		os.Exit(1)
	}
	bin := filepath.Join(dir, "fakeengine")
	build := exec.Command("go", "build", "-o", bin, "./testdata/fakeengine")
	if out, err := build.CombinedOutput(); err != nil {
		fmt.Fprintf(os.Stderr, "build fakeengine: %v\n%s", err, out)
		os.RemoveAll(dir)
		os.Exit(1)
	}
	fakeEnginePath = bin

	code := m.Run()
	os.RemoveAll(dir)
	os.Exit(code)
}

// testManifest builds a minimal manifest whose flags reference one diffusion
// component. The fake engine ignores the flags, but buildArgs still needs the
// component to resolve the {diffusion} placeholder.
func testManifest(name string) types.Manifest {
	return types.Manifest{
		Name:         name,
		Architecture: "flux",
		Mode:         types.ModeGenerate,
		Components:   []types.Component{{Role: types.RoleDiffusion, Blob: "sha256-" + name}},
		Engine:       types.EngineSpec{Flags: []string{"--diffusion-model", "{diffusion}"}},
	}
}

func newTestSupervisor(t *testing.T, opts Options) *Supervisor {
	t.Helper()
	if opts.EnginePath == "" {
		opts.EnginePath = fakeEnginePath
	}
	if opts.LogDir == "" {
		opts.LogDir = t.TempDir()
	}
	if opts.BlobPath == nil {
		opts.BlobPath = func(blob string) string { return "/fake/blobs/" + blob }
	}
	s := New(opts)
	t.Cleanup(s.Shutdown)
	return s
}

func TestBuildArgs(t *testing.T) {
	m := types.Manifest{
		Name: "flux-dev",
		Components: []types.Component{
			{Role: types.RoleDiffusion, Blob: "sha256-diff"},
			{Role: types.RoleVAE, Blob: "sha256-vae"},
		},
		Engine: types.EngineSpec{
			Flags:     []string{"--diffusion-model", "{diffusion}", "--vae", "{vae}", "--diffusion-fa"},
			ModelArgs: map[string]any{"qwen_image_zero_cond_t": true},
		},
	}
	blobPath := func(blob string) string { return "/blobs/" + blob }

	got, err := buildArgs(m, blobPath, "127.0.0.1", "8080", "/store/loras")
	if err != nil {
		t.Fatalf("buildArgs: %v", err)
	}
	want := []string{
		"--diffusion-model", "/blobs/sha256-diff",
		"--vae", "/blobs/sha256-vae",
		"--diffusion-fa",
		"--model-args", "qwen_image_zero_cond_t=true",
		"--lora-model-dir", "/store/loras",
		"--listen-ip", "127.0.0.1",
		"--listen-port", "8080",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("argv mismatch\n got: %q\nwant: %q", got, want)
	}

	// With no LoRA directory configured the flag must be omitted entirely —
	// sd-server treats an empty --lora-model-dir as the current directory.
	got, err = buildArgs(m, blobPath, "127.0.0.1", "8080", "")
	if err != nil {
		t.Fatalf("buildArgs (no lora dir): %v", err)
	}
	if slices.Contains(got, "--lora-model-dir") {
		t.Fatalf("empty LoraDir should omit the flag, got: %q", got)
	}
}

// A ControlNet is a launch-time flag, so its blob path must land in argv.
func TestBuildArgsControlNet(t *testing.T) {
	m := types.Manifest{
		Name: "sd15-canny",
		Components: []types.Component{
			{Role: types.RoleDiffusion, Blob: "sha256-diff"},
			{Role: types.RoleControlNet, Blob: "sha256-cn"},
		},
		Engine: types.EngineSpec{
			Flags: []string{"--model", "{diffusion}", "--control-net", "{control_net}"},
		},
	}
	got, err := buildArgs(m, func(b string) string { return "/blobs/" + b }, "127.0.0.1", "8080", "")
	if err != nil {
		t.Fatalf("buildArgs: %v", err)
	}
	i := slices.Index(got, "--control-net")
	if i < 0 || got[i+1] != "/blobs/sha256-cn" {
		t.Fatalf("control net path not substituted: %q", got)
	}
}

func TestBuildArgsMissingComponent(t *testing.T) {
	m := types.Manifest{
		Name:   "broken",
		Engine: types.EngineSpec{Flags: []string{"--vae", "{vae}"}},
	}
	if _, err := buildArgs(m, nil, "127.0.0.1", "1", ""); err == nil {
		t.Fatal("expected error for missing vae component")
	}
}

func TestGenerateReturnsPNG(t *testing.T) {
	s := newTestSupervisor(t, Options{})

	img, err := s.Generate(context.Background(), testManifest("modelA"), engineclient.ImgGenRequest{Prompt: "a cat"})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	// PNG magic number.
	if !bytes.HasPrefix(img, []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'}) {
		t.Fatalf("result is not a PNG (len=%d, prefix=%x)", len(img), img[:min(8, len(img))])
	}
	if got := s.Loaded(); len(got) != 1 || got[0] != "modelA" {
		t.Fatalf("Loaded = %v, want [modelA]", got)
	}
}

func TestLazyLoad(t *testing.T) {
	s := newTestSupervisor(t, Options{})

	if got := s.Loaded(); len(got) != 0 {
		t.Fatalf("expected nothing loaded before Generate, got %v", got)
	}
	if _, err := s.Generate(context.Background(), testManifest("lazy"), engineclient.ImgGenRequest{Prompt: "x"}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if got := s.Loaded(); len(got) != 1 || got[0] != "lazy" {
		t.Fatalf("expected [lazy] loaded after Generate, got %v", got)
	}
}

func TestMaxLoadedEviction(t *testing.T) {
	s := newTestSupervisor(t, Options{MaxLoaded: 1})

	if _, err := s.Generate(context.Background(), testManifest("A"), engineclient.ImgGenRequest{Prompt: "x"}); err != nil {
		t.Fatalf("Generate A: %v", err)
	}
	if got := s.Loaded(); len(got) != 1 || got[0] != "A" {
		t.Fatalf("after A: Loaded = %v", got)
	}
	if _, err := s.Generate(context.Background(), testManifest("B"), engineclient.ImgGenRequest{Prompt: "y"}); err != nil {
		t.Fatalf("Generate B: %v", err)
	}
	if got := s.Loaded(); len(got) != 1 || got[0] != "B" {
		t.Fatalf("after B (MaxLoaded=1): Loaded = %v, want [B] (A evicted)", got)
	}
}

func TestIdleReap(t *testing.T) {
	s := newTestSupervisor(t, Options{IdleTTL: 250 * time.Millisecond})

	if _, err := s.Generate(context.Background(), testManifest("ephemeral"), engineclient.ImgGenRequest{Prompt: "x"}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if got := s.Loaded(); len(got) != 1 {
		t.Fatalf("expected 1 loaded right after Generate, got %v", got)
	}

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if len(s.Loaded()) == 0 {
			return // reaped
		}
		time.Sleep(25 * time.Millisecond)
	}
	t.Fatalf("model was not reaped after idle TTL; still loaded: %v", s.Loaded())
}

func TestUnload(t *testing.T) {
	s := newTestSupervisor(t, Options{})

	if _, err := s.Generate(context.Background(), testManifest("u"), engineclient.ImgGenRequest{Prompt: "x"}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if err := s.Unload("u"); err != nil {
		t.Fatalf("Unload: %v", err)
	}
	if got := s.Loaded(); len(got) != 0 {
		t.Fatalf("expected nothing loaded after Unload, got %v", got)
	}
	if err := s.Unload("nope"); err == nil {
		t.Fatal("expected error unloading unknown model")
	}
}

func TestStartTimeoutBadBinary(t *testing.T) {
	s := newTestSupervisor(t, Options{EnginePath: "/bin/true", StartTimeout: 300 * time.Millisecond})

	_, err := s.Generate(context.Background(), testManifest("dead"), engineclient.ImgGenRequest{Prompt: "x"})
	if err == nil {
		t.Fatal("expected error when engine never becomes healthy")
	}
}
