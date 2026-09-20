// Package sam3 turns a prompt — text, points or a box — into segmentation
// masks, so /v1/edit can be told which pixels to change instead of hoping the
// diffusion model infers it from prompt phrasing.
//
// It links sam3.cpp (github.com/PABannier/sam3.cpp — ggml + Metal) statically
// into the daemon via cgo. That is a deliberate trade: no IPC and no second
// process to supervise, but libsam3 aborts the process on an internal
// GGML_ASSERT, so a bug in it takes the daemon with it. This is the only place
// oflux links inference code; the sd.cpp engine stays an opaque subprocess.
//
// The cgo implementation is behind the `sam3` build tag because the static
// libraries live in third_party/, which is gitignored — an unconditional link
// would break `go build ./...` on a fresh clone:
//
//	go build ./...             stub: Available() == false, Segment() == ErrUnavailable
//	go build -tags sam3 ./...  real: links third_party/sam3-darwin-arm64
//
// Weights are fetched on demand by EnsureModel, never bundled.
package sam3

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/jpeg"
	"image/png"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"oflux/internal/hfclient"
)

// ErrUnavailable means this binary cannot segment. Both builds report it
// wrapped, never bare, so `errors.Is(err, ErrUnavailable)` is the only test
// that works — `err == ErrUnavailable` is not, in either build.
//
// It covers two different situations, which Compiled() tells apart: no library
// linked (permanent, 501) versus no checkpoint yet (fixable by EnsureModel).
var ErrUnavailable = errors.New("sam3: not built into this binary")

// ErrNoMatch means the model ran and found nothing. A normal outcome.
var ErrNoMatch = errors.New("sam3: prompt matched nothing in the image")

// unavailableError reports the actionable reason while still matching
// ErrUnavailable, which is what the HTTP layer tests. Both build variants use
// it so the two behave identically under errors.Is.
type unavailableError struct{ reason error }

func (e unavailableError) Error() string   { return e.reason.Error() }
func (e unavailableError) Unwrap() []error { return []error{e.reason, ErrUnavailable} }

func unavailable(format string, args ...any) error {
	return unavailableError{fmt.Errorf(format, args...)}
}

// Point is a click in original-image pixel coordinates. libsam3 scales it by
// orig_width/orig_height, so it must not be pre-scaled to the model's input.
type Point struct {
	X, Y     int
	Negative bool
}

// Box is x0,y0,x1,y1 in original-image pixel coordinates, top-left inclusive.
type Box struct {
	X0, Y0, X1, Y1 int
}

// Request is one segmentation. Exactly one prompt style may be used: Prompt
// drives the text detector (PCS), Points/Boxes drive the interactive decoder
// (PVS). libsam3 has no path that takes both, and Segment rejects the
// combination before anything crosses into C rather than trusting callers.
type Request struct {
	Image    []byte
	Prompt   string
	Points   []Point
	Boxes    []Box
	Separate bool
}

// Mask is one result, grayscale PNG as raw base64 at the input's dimensions,
// white being the region.
type Mask struct {
	B64   string
	Score float32
	Box   Box
}

// Bounds enforced before anything crosses into C: libsam3 validates none of
// its inputs, so a bad one is a segfault rather than an error return.
const (
	MinSide       = 16
	MaxSide       = 8192
	MaxImageBytes = 64 << 20
	MaxPoints     = 64
	// The text encoder truncates to 32 BPE tokens (sam3_hparams.text_ctx_len);
	// anything past this is already ignored by the model.
	MaxPromptBytes = 256
)

// Where EnsureModel fetches weights from. Verified against the live HF API:
// repo sha a3892b63, sam3-q4_0.ggml is 706606590 bytes with LFS sha256
// 5dafc790…1889. sam3.cpp's own .ggml container, not GGUF.
const (
	ModelRepo     = "PABannier/sam3.cpp"
	ModelRevision = "main"
	ModelFile     = "sam3-q4_0.ggml"
)

// Only full `sam3-*` checkpoints: the `sam3-visual-*` builds drop the text
// encoder, and sam3_segment_pcs refuses to run on them. Smallest first — the
// upstream benchmarks show no latency difference between q4_0 and f16.
var modelPreference = []string{
	"sam3-q4_0.ggml",
	"sam3-q4_1.ggml",
	"sam3-q8_0.ggml",
	"sam3-f16.ggml",
	"sam3-f32.ggml",
}

// Mirrors store.Open's rooting without importing the store.
func modelDir() string {
	if d := os.Getenv("OFLUX_SAM3_DIR"); d != "" {
		return d
	}
	root := os.Getenv("OFLUX_HOME")
	if root == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return ""
		}
		root = filepath.Join(home, ".oflux")
	}
	return filepath.Join(root, "sam3")
}

func resolveModel() (string, error) {
	if p := os.Getenv("OFLUX_SAM3_MODEL"); p != "" {
		if isReadableFile(p) {
			return p, nil
		}
		return "", fmt.Errorf("sam3: OFLUX_SAM3_MODEL=%s is not a readable file", p)
	}
	dir := modelDir()
	if dir == "" {
		return "", errors.New("sam3: no model directory (set OFLUX_SAM3_MODEL or OFLUX_SAM3_DIR)")
	}
	for _, name := range modelPreference {
		if p := filepath.Join(dir, name); isReadableFile(p) {
			return p, nil
		}
	}
	return "", fmt.Errorf("sam3: no checkpoint in %s (want one of %s) — call EnsureModel or run scripts/fetch-sam3.sh --model",
		dir, strings.Join(modelPreference, ", "))
}

// ModelPath reports the checkpoint Segment would load, or "" if none is.
func ModelPath() string {
	p, err := resolveModel()
	if err != nil {
		return ""
	}
	return p
}

func isReadableFile(p string) bool {
	fi, err := os.Stat(p)
	if err != nil || !fi.Mode().IsRegular() || fi.Size() == 0 {
		return false
	}
	f, err := os.Open(p)
	if err != nil {
		return false
	}
	_ = f.Close()
	return true
}

var ensureMu sync.Mutex

// ensureModel is the real downloader. Only the linked build exports it: a stub
// binary must never fetch 707 MB of weights it cannot use, least of all from
// inside `go test`.
func ensureModel(ctx context.Context, prog func(string)) (string, error) {
	// Serialised so two callers cannot write the same .part file. Re-resolving
	// under the lock is what makes the second caller cheap rather than a
	// second download.
	ensureMu.Lock()
	defer ensureMu.Unlock()

	if p, err := resolveModel(); err == nil {
		emit(prog, fmt.Sprintf("✓ %s (cached)", filepath.Base(p)))
		return p, nil
	}
	if p := os.Getenv("OFLUX_SAM3_MODEL"); p != "" {
		return "", fmt.Errorf("sam3: OFLUX_SAM3_MODEL=%s is not a readable file; refusing to download over an explicit path", p)
	}
	dir := modelDir()
	if dir == "" {
		return "", errors.New("sam3: no model directory (set OFLUX_SAM3_DIR)")
	}
	dest := filepath.Join(dir, ModelFile)

	client := hfclient.New(os.Getenv("HF_TOKEN"))
	if base := os.Getenv("OFLUX_SAM3_BASE_URL"); base != "" {
		client.SetBaseURL(base)
	}

	size, sha := int64(0), ""
	files, err := client.Tree(ctx, ModelRepo, ModelRevision)
	if err != nil {
		return "", fmt.Errorf("sam3: listing %s: %w", ModelRepo, err)
	}
	for _, f := range files {
		if f.Path == ModelFile {
			size = f.Size
			// ContentHash falls back to the git sha1 for non-LFS files, which
			// would fail Download's sha256 check. Only trust the LFS oid.
			if f.IsLFS {
				sha = f.LFSOID
			}
			break
		}
	}
	if size == 0 {
		return "", fmt.Errorf("sam3: %s has no %s", ModelRepo, ModelFile)
	}

	emit(prog, fmt.Sprintf("↓ %s from %s (%s)", ModelFile, ModelRepo, humanBytes(size)))
	stop := watchPart(dest+".part", size, prog)
	// Download stages through dest+".part" and renames only after the sha256
	// matches, so an interrupted fetch can never be picked up as a model:
	// resolveModel accepts exact .ggml names only.
	if _, err := client.Download(ctx, ModelRepo, ModelRevision, ModelFile, dest, sha); err != nil {
		stop()
		return "", fmt.Errorf("sam3: downloading %s: %w", ModelFile, err)
	}
	stop()
	if sha == "" {
		emit(prog, fmt.Sprintf("! %s: no sha256 published; integrity check skipped", ModelFile))
	}
	emit(prog, fmt.Sprintf("installed %s (%s)", ModelFile, humanBytes(size)))
	return dest, nil
}

func emit(prog func(string), msg string) {
	if prog != nil {
		prog(msg)
	}
}

// watchPart reports download progress by watching the staging file grow;
// hfclient.Download has no progress hook of its own.
func watchPart(part string, total int64, prog func(string)) (stop func()) {
	if prog == nil || total <= 0 {
		return func() {}
	}
	done := make(chan struct{})
	var once sync.Once
	name := filepath.Base(strings.TrimSuffix(part, ".part"))
	go func() {
		t := time.NewTicker(2 * time.Second)
		defer t.Stop()
		last := -1
		for {
			select {
			case <-done:
				return
			case <-t.C:
				fi, err := os.Stat(part)
				if err != nil {
					continue
				}
				if pct := int(fi.Size() * 100 / total); pct != last {
					last = pct
					emit(prog, fmt.Sprintf("  %s %d%% (%s/%s)", name, pct,
						humanBytes(fi.Size()), humanBytes(total)))
				}
			}
		}
	}()
	return func() { once.Do(func() { close(done) }) }
}

func humanBytes(n int64) string {
	switch {
	case n >= 1<<30:
		return fmt.Sprintf("%.1f GB", float64(n)/(1<<30))
	case n >= 1<<20:
		return fmt.Sprintf("%.0f MB", float64(n)/(1<<20))
	default:
		return fmt.Sprintf("%d B", n)
	}
}

func envFloat(key string, def float64) float64 {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	f, err := strconv.ParseFloat(v, 64)
	if err != nil || f <= 0 {
		return def
	}
	return f
}

// errNoPrompt is checked before the image is decoded so an empty request
// fails on what is actually wrong with it.
var errNoPrompt = errors.New("sam3: request has no prompt: set Prompt, Points or Boxes")

func requestIsEmpty(req Request) bool {
	return strings.TrimSpace(req.Prompt) == "" && len(req.Points) == 0 && len(req.Boxes) == 0
}

// plan is the validated, bounds-checked form of a Request, in the shape the C
// shim takes.
type plan struct {
	pcs    bool
	prompt string
	points []Point
	box    Box
	useBox bool
}

func (p plan) describe() string {
	if p.pcs {
		return fmt.Sprintf("%q", p.prompt)
	}
	var parts []string
	if n := len(p.points); n > 0 {
		parts = append(parts, fmt.Sprintf("%d point(s)", n))
	}
	if p.useBox {
		parts = append(parts, fmt.Sprintf("box (%d,%d)-(%d,%d)", p.box.X0, p.box.Y0, p.box.X1, p.box.Y1))
	}
	return strings.Join(parts, " + ")
}

func validatePrompt(prompt string) (string, error) {
	p := strings.TrimSpace(prompt)
	if p == "" {
		return "", errors.New("sam3: prompt is empty")
	}
	if len(p) > MaxPromptBytes {
		return "", fmt.Errorf("sam3: prompt is %d bytes, limit is %d", len(p), MaxPromptBytes)
	}
	if !utf8.ValidString(p) {
		return "", errors.New("sam3: prompt is not valid UTF-8")
	}
	// A NUL would silently truncate the prompt once it becomes a C string.
	if strings.IndexByte(p, 0) >= 0 {
		return "", errors.New("sam3: prompt contains a NUL byte")
	}
	return p, nil
}

func validateRequest(req Request, w, h int) (plan, error) {
	text := strings.TrimSpace(req.Prompt) != ""
	geom := len(req.Points) > 0 || len(req.Boxes) > 0
	switch {
	case !text && !geom:
		return plan{}, errNoPrompt
	case text && geom:
		return plan{}, errors.New("sam3: a text prompt cannot be combined with points or boxes: " +
			"text runs the PCS detector, points and boxes the PVS decoder")
	}

	if text {
		p, err := validatePrompt(req.Prompt)
		if err != nil {
			return plan{}, err
		}
		return plan{pcs: true, prompt: p}, nil
	}

	if len(req.Points) > MaxPoints {
		return plan{}, fmt.Errorf("sam3: %d points given, limit is %d", len(req.Points), MaxPoints)
	}
	// sam3_pvs_params carries a single box, not a list.
	if len(req.Boxes) > 1 {
		return plan{}, fmt.Errorf("sam3: %d boxes given, the PVS decoder takes at most one", len(req.Boxes))
	}

	var positive int
	for i, pt := range req.Points {
		if pt.X < 0 || pt.X >= w || pt.Y < 0 || pt.Y >= h {
			return plan{}, fmt.Errorf("sam3: point %d (%d,%d) is outside the %dx%d image", i, pt.X, pt.Y, w, h)
		}
		if !pt.Negative {
			positive++
		}
	}

	out := plan{points: req.Points}
	if len(req.Boxes) == 1 {
		b := req.Boxes[0]
		if b.X1 <= b.X0 || b.Y1 <= b.Y0 {
			return plan{}, fmt.Errorf("sam3: box (%d,%d)-(%d,%d) is empty", b.X0, b.Y0, b.X1, b.Y1)
		}
		if b.X0 < 0 || b.Y0 < 0 || b.X1 > w || b.Y1 > h {
			return plan{}, fmt.Errorf("sam3: box (%d,%d)-(%d,%d) is outside the %dx%d image", b.X0, b.Y0, b.X1, b.Y1, w, h)
		}
		out.box, out.useBox = b, true
	}
	// libsam3 bails out on this combination too, but silently, with an empty
	// result rather than an error.
	if positive == 0 && !out.useBox {
		return plan{}, errors.New("sam3: point prompts need at least one positive point, or a box")
	}
	return out, nil
}

func checkDims(w, h int) error {
	if w < MinSide || h < MinSide {
		return fmt.Errorf("sam3: image is %dx%d, minimum is %dx%d", w, h, MinSide, MinSide)
	}
	if w > MaxSide || h > MaxSide {
		return fmt.Errorf("sam3: image is %dx%d, maximum is %dx%d", w, h, MaxSide, MaxSide)
	}
	return nil
}

// decodeRGB produces tightly packed 8-bit RGB, the only layout sam3.cpp's
// preprocessor reads: it indexes data[(y*w+x)*3+c] regardless of what
// sam3_image.channels says.
func decodeRGB(data []byte) (pix []byte, w, h int, err error) {
	if len(data) == 0 {
		return nil, 0, 0, errors.New("sam3: image is empty")
	}
	if len(data) > MaxImageBytes {
		return nil, 0, 0, fmt.Errorf("sam3: image is %d bytes, limit is %d", len(data), MaxImageBytes)
	}
	// Header first: a decompression bomb declares its dimensions here, so it
	// is rejected before any pixel buffer is allocated.
	cfg, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return nil, 0, 0, fmt.Errorf("sam3: image does not decode: %w", err)
	}
	if err := checkDims(cfg.Width, cfg.Height); err != nil {
		return nil, 0, 0, err
	}

	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, 0, 0, fmt.Errorf("sam3: image does not decode: %w", err)
	}
	b := img.Bounds()
	w, h = b.Dx(), b.Dy()
	// The header is not trustworthy on its own; re-check what decoded.
	if err := checkDims(w, h); err != nil {
		return nil, 0, 0, err
	}

	// Composite over white so a transparent PNG does not reach the model as a
	// field of black; draw.Draw absorbs paletted/16-bit/YCbCr/premultiplied.
	flat := image.NewNRGBA(image.Rect(0, 0, w, h))
	draw.Draw(flat, flat.Bounds(), image.NewUniform(color.White), image.Point{}, draw.Src)
	draw.Draw(flat, flat.Bounds(), img, b.Min, draw.Over)

	pix = make([]byte, w*h*3)
	for i, j := 0, 0; i < len(pix); i, j = i+3, j+4 {
		pix[i] = flat.Pix[j]
		pix[i+1] = flat.Pix[j+1]
		pix[i+2] = flat.Pix[j+2]
	}
	return pix, w, h, nil
}

// encodeMaskPNG returns raw base64 with no data: prefix, matching the HTTP
// seam. White is the region, the convention sd.cpp's mask_image expects.
func encodeMaskPNG(mask []byte, w, h int) (string, error) {
	if w <= 0 || h <= 0 || len(mask) != w*h {
		return "", fmt.Errorf("sam3: mask is %d bytes, want %d for %dx%d", len(mask), w*h, w, h)
	}
	img := image.NewGray(image.Rect(0, 0, w, h))
	copy(img.Pix, mask) // image.Gray strides exactly w bytes per row
	var buf bytes.Buffer
	enc := png.Encoder{CompressionLevel: png.BestCompression}
	if err := enc.Encode(&buf, img); err != nil {
		return "", fmt.Errorf("sam3: encoding mask: %w", err)
	}
	return base64.StdEncoding.EncodeToString(buf.Bytes()), nil
}
