//go:build sam3

package sam3

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"image"
	"image/color"
	"image/png"
	"os"
	"strings"
	"sync"
	"testing"
)

const (
	discSize = 512
	discCX   = 256
	discCY   = 256
	discR    = 140
)

// discPNG is a red disc on white: something both a text prompt and a click can
// plausibly pick out.
func discPNG(t *testing.T, extra bool) []byte {
	t.Helper()
	img := image.NewNRGBA(image.Rect(0, 0, discSize, discSize))
	inside := func(x, y, cx, cy, r int) bool {
		dx, dy := x-cx, y-cy
		return dx*dx+dy*dy < r*r
	}
	for y := range discSize {
		for x := range discSize {
			switch {
			case inside(x, y, discCX, discCY, discR):
				img.Set(x, y, color.NRGBA{R: 220, G: 30, B: 30, A: 255})
			case extra && inside(x, y, 60, 60, 40):
				img.Set(x, y, color.NRGBA{R: 220, G: 30, B: 30, A: 255})
			default:
				img.Set(x, y, color.White)
			}
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encode png: %v", err)
	}
	return buf.Bytes()
}

func testImage(t *testing.T) ([]byte, string) {
	t.Helper()
	if p := os.Getenv("OFLUX_SAM3_TEST_IMAGE"); p != "" {
		data, err := os.ReadFile(p)
		if err != nil {
			t.Fatalf("read %s: %v", p, err)
		}
		prompt := os.Getenv("OFLUX_SAM3_TEST_PROMPT")
		if prompt == "" {
			prompt = "the main subject"
		}
		return data, prompt
	}
	return discPNG(t, false), "a red circle"
}

func requireModel(t *testing.T) {
	t.Helper()
	if !Available() {
		t.Skipf("no SAM3 checkpoint installed: %s", Status())
	}
}

// decodeMask turns a Mask back into pixels and reports how many are white.
func decodeMask(t *testing.T, m Mask, w, h int) (*image.Gray, int) {
	t.Helper()
	raw, err := base64.StdEncoding.DecodeString(m.B64)
	if err != nil {
		t.Fatalf("mask is not base64: %v", err)
	}
	img, format, err := image.Decode(bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("mask does not decode: %v", err)
	}
	if format != "png" {
		t.Fatalf("mask format = %q, want png", format)
	}
	if img.Bounds().Dx() != w || img.Bounds().Dy() != h {
		t.Fatalf("mask is %v, want %dx%d", img.Bounds(), w, h)
	}
	gray, ok := img.(*image.Gray)
	if !ok {
		t.Fatalf("mask decoded as %T, want *image.Gray", img)
	}
	var white int
	for _, p := range gray.Pix {
		if p == 255 {
			white++
		}
	}
	return gray, white
}

func TestSegmentValidatesBeforeReachingC(t *testing.T) {
	// These must fail on the Go side of the boundary, without a model and
	// without ever calling into libsam3.
	isolateModelEnv(t)
	good := discPNG(t, false)
	tests := []struct {
		name   string
		req    Request
		errHas string
	}{
		{name: "no prompt at all", req: Request{Image: good}, errHas: "no prompt"},
		{name: "empty prompt", req: Request{Image: good, Prompt: "  "}, errHas: "no prompt"},
		{name: "nul in prompt", req: Request{Image: good, Prompt: "cat\x00"}, errHas: "NUL"},
		{
			name:   "long prompt",
			req:    Request{Image: good, Prompt: strings.Repeat("x", MaxPromptBytes+1)},
			errHas: "limit is",
		},
		{name: "garbage image", req: Request{Image: []byte("nope"), Prompt: "a cat"}, errHas: "does not decode"},
		{
			name:   "tiny image",
			req:    Request{Image: solidPNG(t, 8, 8, color.White), Prompt: "a cat"},
			errHas: "minimum is",
		},
		{
			name:   "point outside the image",
			req:    Request{Image: good, Points: []Point{{X: discSize, Y: 0}}},
			errHas: "outside",
		},
		{
			name:   "box outside the image",
			req:    Request{Image: good, Boxes: []Box{{X0: 0, Y0: 0, X1: discSize + 1, Y1: 10}}},
			errHas: "outside",
		},
		{
			name:   "text mixed with a point",
			req:    Request{Image: good, Prompt: "a cat", Points: []Point{{X: 1, Y: 1}}},
			errHas: "cannot be combined",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			masks, err := Segment(t.Context(), tt.req)
			if err == nil || !strings.Contains(err.Error(), tt.errHas) {
				t.Fatalf("err = %v, want it to contain %q", err, tt.errHas)
			}
			if masks != nil {
				t.Fatalf("masks = %v, want nil on a rejected request", masks)
			}
		})
	}
}

func TestLinkedBuildIsCompiled(t *testing.T) {
	if !Compiled() {
		t.Fatal("Compiled() is false in a build with the sam3 tag")
	}
}

// With a checkpoint already installed, EnsureModel must be a local lookup.
func TestEnsureModelIsALookupWhenInstalled(t *testing.T) {
	requireModel(t)
	hub, srv := newFakeHub(t, "weights")
	t.Setenv("OFLUX_SAM3_BASE_URL", srv.URL)

	got, err := EnsureModel(t.Context(), nil)
	if err != nil {
		t.Fatalf("EnsureModel: %v", err)
	}
	if got != ModelPath() {
		t.Fatalf("EnsureModel = %s, want the installed %s", got, ModelPath())
	}
	if n := hub.trees.Load() + hub.downloads.Load(); n != 0 {
		t.Fatalf("%d requests reached the Hub although a checkpoint is installed", n)
	}
}

func TestSegmentWithoutCheckpointIsUnavailable(t *testing.T) {
	isolateModelEnv(t)
	if Available() {
		t.Fatal("Available() is true with an empty model directory")
	}
	_, err := Segment(t.Context(), Request{Image: discPNG(t, false), Prompt: "a cat"})
	if !errors.Is(err, ErrUnavailable) {
		t.Fatalf("err = %v, want ErrUnavailable", err)
	}
}

func TestSegmentTextPrompt(t *testing.T) {
	requireModel(t)
	data, prompt := testImage(t)

	masks, err := Segment(t.Context(), Request{Image: data, Prompt: prompt})
	if errors.Is(err, ErrNoMatch) {
		t.Logf("no detections for %q (model: %s)", prompt, ModelPath())
		return
	}
	if err != nil {
		t.Fatalf("Segment: %v", err)
	}
	if len(masks) != 1 {
		t.Fatalf("%d masks, want 1 merged mask", len(masks))
	}
	cfg, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	_, white := decodeMask(t, masks[0], cfg.Width, cfg.Height)
	if white == 0 {
		t.Fatal("mask has no white pixels but Segment reported a match")
	}
	if masks[0].Score <= 0 {
		t.Fatalf("score = %v, want a positive detection score", masks[0].Score)
	}
	t.Logf("%q -> %d white px, score %.3f, box %+v", prompt, white, masks[0].Score, masks[0].Box)
}

func TestSegmentPointPrompt(t *testing.T) {
	requireModel(t)
	data := discPNG(t, false)

	masks, err := Segment(t.Context(), Request{
		Image:  data,
		Points: []Point{{X: discCX, Y: discCY}},
	})
	if err != nil {
		t.Fatalf("Segment: %v", err)
	}
	if len(masks) == 0 {
		t.Fatal("a click in the middle of the disc produced no mask")
	}
	gray, white := decodeMask(t, masks[0], discSize, discSize)
	if gray.GrayAt(discCX, discCY).Y != 255 {
		t.Fatal("the clicked pixel is not inside its own mask")
	}
	// The disc covers ~23% of the frame; a mask that is nearly everything or
	// nearly nothing means the point never reached the decoder correctly.
	frac := float64(white) / float64(discSize*discSize)
	if frac < 0.05 || frac > 0.60 {
		t.Fatalf("mask covers %.1f%% of the image, want roughly the disc's 23%%", frac*100)
	}
	t.Logf("point -> %d white px (%.1f%%), score %.3f, box %+v", white, frac*100, masks[0].Score, masks[0].Box)
}

func TestSegmentBoxPrompt(t *testing.T) {
	requireModel(t)
	data := discPNG(t, false)

	masks, err := Segment(t.Context(), Request{
		Image: data,
		Boxes: []Box{{X0: discCX - discR, Y0: discCY - discR, X1: discCX + discR, Y1: discCY + discR}},
	})
	if err != nil {
		t.Fatalf("Segment: %v", err)
	}
	if len(masks) == 0 {
		t.Fatal("a box around the disc produced no mask")
	}
	gray, white := decodeMask(t, masks[0], discSize, discSize)
	if gray.GrayAt(discCX, discCY).Y != 255 {
		t.Fatal("the centre of the boxed region is not in the mask")
	}
	frac := float64(white) / float64(discSize*discSize)
	if frac < 0.05 || frac > 0.60 {
		t.Fatalf("mask covers %.1f%% of the image, want roughly the disc's 23%%", frac*100)
	}
	t.Logf("box -> %d white px (%.1f%%), score %.3f, box %+v", white, frac*100, masks[0].Score, masks[0].Box)
}

func TestSegmentNegativePointIsAccepted(t *testing.T) {
	requireModel(t)
	data := discPNG(t, false)

	masks, err := Segment(t.Context(), Request{
		Image: data,
		Points: []Point{
			{X: discCX, Y: discCY},
			{X: 10, Y: 10, Negative: true},
		},
	})
	if err != nil {
		t.Fatalf("Segment: %v", err)
	}
	if len(masks) == 0 {
		t.Fatal("positive + negative points produced no mask")
	}
	gray, white := decodeMask(t, masks[0], discSize, discSize)
	if gray.GrayAt(10, 10).Y == 255 {
		t.Error("the negative point landed inside the mask")
	}
	t.Logf("pos+neg -> %d white px, score %.3f", white, masks[0].Score)
}

func TestSegmentSeparateReturnsPerInstanceMasks(t *testing.T) {
	requireModel(t)
	data := discPNG(t, true) // two discs

	sep, err := Segment(t.Context(), Request{Image: data, Prompt: "a red circle", Separate: true})
	if errors.Is(err, ErrNoMatch) {
		t.Skip("model found no circles in the synthetic image")
	}
	if err != nil {
		t.Fatalf("Segment(separate): %v", err)
	}
	if len(sep) == 0 {
		t.Fatal("no masks")
	}
	for i := 1; i < len(sep); i++ {
		if sep[i].Score > sep[i-1].Score {
			t.Fatalf("mask %d scores %.3f above mask %d's %.3f — want best first",
				i, sep[i].Score, i-1, sep[i-1].Score)
		}
	}

	merged, err := Segment(t.Context(), Request{Image: data, Prompt: "a red circle"})
	if err != nil {
		t.Fatalf("Segment(merged): %v", err)
	}
	if len(merged) != 1 {
		t.Fatalf("%d merged masks, want 1", len(merged))
	}

	_, mergedWhite := decodeMask(t, merged[0], discSize, discSize)
	var biggest int
	for i, m := range sep {
		_, white := decodeMask(t, m, discSize, discSize)
		if white > biggest {
			biggest = white
		}
		t.Logf("instance %d: %d white px, score %.3f, box %+v", i, white, m.Score, m.Box)
	}
	// The union of the instances cannot be smaller than the largest of them.
	if mergedWhite < biggest {
		t.Fatalf("merged mask has %d white px, less than the biggest instance's %d", mergedWhite, biggest)
	}
	if merged[0].Score != sep[0].Score {
		t.Fatalf("merged score %.3f, want the best instance's %.3f", merged[0].Score, sep[0].Score)
	}
	t.Logf("%d instances, merged %d white px", len(sep), mergedWhite)
}

// Concurrent callers must serialise rather than drive one ggml backend from
// two goroutines; this is the case that kills the daemon if the gate is wrong.
func TestSegmentIsSerialised(t *testing.T) {
	requireModel(t)
	data := discPNG(t, false)

	var wg sync.WaitGroup
	errs := make([]error, 3)
	for i := range errs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := Segment(context.Background(), Request{
				Image:  data,
				Points: []Point{{X: discCX, Y: discCY}},
			})
			errs[i] = err
		}()
	}
	wg.Wait()
	for i, err := range errs {
		if err != nil && !errors.Is(err, ErrNoMatch) {
			t.Fatalf("goroutine %d: %v", i, err)
		}
	}
}

func TestSegmentHonoursCancellation(t *testing.T) {
	requireModel(t)
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	_, err := Segment(ctx, Request{Image: discPNG(t, false), Prompt: "a red circle"})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
}
