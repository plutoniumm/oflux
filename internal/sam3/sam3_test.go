package sam3

import (
	"bytes"
	"encoding/base64"
	"errors"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func solidPNG(t *testing.T, w, h int, c color.Color) []byte {
	t.Helper()
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			img.Set(x, y, c)
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encode png: %v", err)
	}
	return buf.Bytes()
}

func TestErrUnavailableIsTheFrozenSeam(t *testing.T) {
	// internal/server matches on this sentinel to answer 501.
	if got, want := ErrUnavailable.Error(), "sam3: not built into this binary"; got != want {
		t.Fatalf("ErrUnavailable = %q, want %q", got, want)
	}
}

func TestUnavailableErrorKeepsBothTheReasonAndTheSentinel(t *testing.T) {
	err := unavailable("sam3: no checkpoint in %s", "/nowhere")
	if !errors.Is(err, ErrUnavailable) {
		t.Fatal("unavailable() does not match ErrUnavailable, so /v1/segment would answer 500 instead of 501")
	}
	if got := err.Error(); got != "sam3: no checkpoint in /nowhere" {
		t.Fatalf("Error() = %q, want the actionable reason", got)
	}
	// Both builds wrap, so == never works and errors.Is always does. Callers
	// that get this wrong would behave differently tagged vs untagged.
	if err == ErrUnavailable { //nolint:errorlint // that is the point
		t.Fatal("unavailable() returned the bare sentinel")
	}
}

// Both Segment and EnsureModel must report unavailability the same way in the
// linked and stub builds; this runs in both.
func TestUnavailabilityIsReportedIdenticallyInBothBuilds(t *testing.T) {
	isolateModelEnv(t)
	t.Setenv("OFLUX_SAM3_BASE_URL", "http://127.0.0.1:1") // nothing must dial it

	_, segErr := Segment(t.Context(), Request{Image: solidPNG(t, 32, 32, color.White), Prompt: "a cat"})
	if Compiled() {
		if !errors.Is(segErr, ErrUnavailable) {
			t.Fatalf("Segment err = %v, want ErrUnavailable with no checkpoint", segErr)
		}
	} else {
		if !errors.Is(segErr, ErrUnavailable) {
			t.Fatalf("Segment err = %v, want ErrUnavailable", segErr)
		}
		if _, err := EnsureModel(t.Context(), nil); !errors.Is(err, ErrUnavailable) {
			t.Fatalf("EnsureModel err = %v, want ErrUnavailable in a stub build", err)
		}
	}
	if segErr == ErrUnavailable { //nolint:errorlint // that is the point
		t.Fatal("Segment returned the bare sentinel; errors.Is must be the only working test")
	}
}

func TestValidatePrompt(t *testing.T) {
	tests := []struct {
		name   string
		in     string
		want   string
		errHas string
	}{
		{name: "plain", in: "a cat", want: "a cat"},
		{name: "trimmed", in: "  a cat \n", want: "a cat"},
		{name: "unicode", in: "le chât", want: "le chât"},
		{name: "empty", in: "", errHas: "empty"},
		{name: "whitespace only", in: "   \t\n ", errHas: "empty"},
		{name: "too long", in: strings.Repeat("x", MaxPromptBytes+1), errHas: "limit is"},
		{name: "at limit", in: strings.Repeat("x", MaxPromptBytes), want: strings.Repeat("x", MaxPromptBytes)},
		{name: "invalid utf8", in: "cat\xff\xfe", errHas: "UTF-8"},
		{name: "nul byte", in: "cat\x00dog", errHas: "NUL"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := validatePrompt(tt.in)
			if tt.errHas != "" {
				if err == nil {
					t.Fatalf("validatePrompt(%q) = %q, want error containing %q", tt.in, got, tt.errHas)
				}
				if !strings.Contains(err.Error(), tt.errHas) {
					t.Fatalf("error = %v, want it to contain %q", err, tt.errHas)
				}
				return
			}
			if err != nil {
				t.Fatalf("validatePrompt(%q): %v", tt.in, err)
			}
			if got != tt.want {
				t.Fatalf("validatePrompt(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestDecodeRGBSolidImage(t *testing.T) {
	pix, w, h, err := decodeRGB(solidPNG(t, 32, 24, color.NRGBA{R: 255, A: 255}))
	if err != nil {
		t.Fatalf("decodeRGB: %v", err)
	}
	if w != 32 || h != 24 {
		t.Fatalf("dims = %dx%d, want 32x24", w, h)
	}
	if len(pix) != 32*24*3 {
		t.Fatalf("len(pix) = %d, want %d", len(pix), 32*24*3)
	}
	for i := 0; i < len(pix); i += 3 {
		if pix[i] != 255 || pix[i+1] != 0 || pix[i+2] != 0 {
			t.Fatalf("pixel %d = %v, want pure red", i/3, pix[i:i+3])
		}
	}
}

func TestDecodeRGBCompositesAlphaOverWhite(t *testing.T) {
	// A fully transparent black pixel must not reach the model as black.
	pix, _, _, err := decodeRGB(solidPNG(t, 16, 16, color.NRGBA{}))
	if err != nil {
		t.Fatalf("decodeRGB: %v", err)
	}
	for i := 0; i < len(pix); i += 3 {
		if pix[i] != 255 || pix[i+1] != 255 || pix[i+2] != 255 {
			t.Fatalf("pixel %d = %v, want white", i/3, pix[i:i+3])
		}
	}
}

func TestDecodeRGBAcceptsJPEG(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 32, 32))
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, nil); err != nil {
		t.Fatalf("encode jpeg: %v", err)
	}
	if _, w, h, err := decodeRGB(buf.Bytes()); err != nil || w != 32 || h != 32 {
		t.Fatalf("decodeRGB(jpeg) = %dx%d, %v", w, h, err)
	}
}

func TestDecodeRGBRejects(t *testing.T) {
	tests := []struct {
		name   string
		in     []byte
		errHas string
	}{
		{name: "empty", in: nil, errHas: "empty"},
		{name: "not an image", in: []byte("this is not a png at all, honestly"), errHas: "does not decode"},
		{name: "truncated png", in: solidPNG(t, 32, 32, color.White)[:20], errHas: "does not decode"},
		{name: "too small", in: solidPNG(t, MinSide-1, MinSide-1, color.White), errHas: "minimum is"},
		{name: "too wide", in: solidPNG(t, MaxSide+1, MinSide, color.White), errHas: "maximum is"},
		{name: "oversized payload", in: make([]byte, MaxImageBytes+1), errHas: "limit is"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, _, err := decodeRGB(tt.in)
			if err == nil {
				t.Fatalf("decodeRGB(%s) = nil error, want %q", tt.name, tt.errHas)
			}
			if !strings.Contains(err.Error(), tt.errHas) {
				t.Fatalf("error = %v, want it to contain %q", err, tt.errHas)
			}
		})
	}
}

func TestEncodeMaskPNGRoundTrip(t *testing.T) {
	const w, h = 40, 30
	mask := make([]byte, w*h)
	for y := 10; y < 20; y++ {
		for x := 5; x < 15; x++ {
			mask[y*w+x] = 255
		}
	}

	b64, err := encodeMaskPNG(mask, w, h)
	if err != nil {
		t.Fatalf("encodeMaskPNG: %v", err)
	}
	if strings.HasPrefix(b64, "data:") {
		t.Fatalf("mask is a data URL, the seam wants raw base64: %.32q", b64)
	}
	raw, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		t.Fatalf("mask is not base64: %v", err)
	}
	img, format, err := image.Decode(bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("mask does not decode: %v", err)
	}
	if format != "png" {
		t.Fatalf("format = %q, want png", format)
	}
	if got := img.Bounds(); got.Dx() != w || got.Dy() != h {
		t.Fatalf("mask is %v, want %dx%d", got, w, h)
	}
	for y := range h {
		for x := range w {
			r, _, _, _ := img.At(x, y).RGBA()
			white := r>>8 == 255
			if want := mask[y*w+x] == 255; white != want {
				t.Fatalf("pixel (%d,%d) white=%v, want %v", x, y, white, want)
			}
		}
	}
}

func TestEncodeMaskPNGRejectsWrongLength(t *testing.T) {
	if _, err := encodeMaskPNG(make([]byte, 10), 4, 4); err == nil {
		t.Fatal("encodeMaskPNG accepted a 10-byte mask for a 4x4 image")
	}
	if _, err := encodeMaskPNG(nil, 0, 0); err == nil {
		t.Fatal("encodeMaskPNG accepted a zero-sized image")
	}
}

// isolateModelEnv points model resolution at an empty directory so the
// developer's real ~/.oflux cannot influence the result.
func isolateModelEnv(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	t.Setenv("OFLUX_SAM3_MODEL", "")
	t.Setenv("OFLUX_SAM3_DIR", dir)
	t.Setenv("OFLUX_HOME", dir)
	return dir
}

func writeFile(t *testing.T, path string, size int) {
	t.Helper()
	if err := os.WriteFile(path, make([]byte, size), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func TestResolveModelPrefersSmallestQuant(t *testing.T) {
	dir := isolateModelEnv(t)
	writeFile(t, filepath.Join(dir, "sam3-f16.ggml"), 8)
	writeFile(t, filepath.Join(dir, "sam3-q8_0.ggml"), 8)
	writeFile(t, filepath.Join(dir, "sam3-q4_0.ggml"), 8)

	got, err := resolveModel()
	if err != nil {
		t.Fatalf("resolveModel: %v", err)
	}
	if want := filepath.Join(dir, "sam3-q4_0.ggml"); got != want {
		t.Fatalf("resolveModel = %s, want %s", got, want)
	}
	if ModelPath() != got {
		t.Fatalf("ModelPath = %s, want %s", ModelPath(), got)
	}
}

func TestResolveModelIgnoresVisualOnlyAndEmptyFiles(t *testing.T) {
	dir := isolateModelEnv(t)
	// sam3-visual has no text encoder, so it can never serve a text prompt.
	writeFile(t, filepath.Join(dir, "sam3-visual-q4_0.ggml"), 8)
	writeFile(t, filepath.Join(dir, "sam3-q4_0.ggml"), 0) // truncated download

	_, err := resolveModel()
	if err == nil {
		t.Fatal("resolveModel accepted a visual-only/zero-byte directory")
	}
	if !strings.Contains(err.Error(), dir) {
		t.Fatalf("error = %v, want it to name %s", err, dir)
	}
	if ModelPath() != "" {
		t.Fatalf("ModelPath = %q, want empty", ModelPath())
	}
}

func TestResolveModelExplicitOverride(t *testing.T) {
	dir := isolateModelEnv(t)
	path := filepath.Join(dir, "anything.ggml")
	writeFile(t, path, 8)
	t.Setenv("OFLUX_SAM3_MODEL", path)

	got, err := resolveModel()
	if err != nil {
		t.Fatalf("resolveModel: %v", err)
	}
	if got != path {
		t.Fatalf("resolveModel = %s, want %s", got, path)
	}

	t.Setenv("OFLUX_SAM3_MODEL", filepath.Join(dir, "missing.ggml"))
	if _, err := resolveModel(); err == nil {
		t.Fatal("resolveModel accepted OFLUX_SAM3_MODEL pointing at nothing")
	}
}

func TestEnvFloat(t *testing.T) {
	t.Setenv("OFLUX_SAM3_TEST_F", "")
	if got := envFloat("OFLUX_SAM3_TEST_F", 0.5); got != 0.5 {
		t.Fatalf("unset = %v, want 0.5", got)
	}
	t.Setenv("OFLUX_SAM3_TEST_F", "0.25")
	if got := envFloat("OFLUX_SAM3_TEST_F", 0.5); got != 0.25 {
		t.Fatalf("set = %v, want 0.25", got)
	}
	for _, bad := range []string{"nonsense", "0", "-1"} {
		t.Setenv("OFLUX_SAM3_TEST_F", bad)
		if got := envFloat("OFLUX_SAM3_TEST_F", 0.5); got != 0.5 {
			t.Fatalf("%q = %v, want the 0.5 fallback", bad, got)
		}
	}
}
