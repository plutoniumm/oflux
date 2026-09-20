//go:build sam3

package sam3

/*
#cgo CPPFLAGS: -I${SRCDIR} -I${SRCDIR}/../../third_party/sam3-darwin-arm64/include
#cgo CXXFLAGS: -std=c++14 -O2
#cgo LDFLAGS: -L${SRCDIR}/../../third_party/sam3-darwin-arm64/lib
#cgo LDFLAGS: -lsam3 -lggml -lggml-cpu -lggml-metal -lggml-blas -lggml-base
#cgo LDFLAGS: -framework Foundation -framework Metal -framework MetalKit -framework Accelerate -framework CoreFoundation

#include <stdlib.h>
#include "bridge.h"
*/
import "C"

import (
	"context"
	"fmt"
	"math"
	"os"
	"runtime"
	"sync"
	"time"
	"unsafe"
)

const (
	idleTTL    = 5 * time.Minute
	idleSweep  = time.Minute
	errBufSize = 512
)

// gate serialises every call into libsam3 and guards handle/lastUse. It is a
// channel rather than a Mutex so waiting for it can respect a context, and the
// send/receive pair gives the reaper goroutine the happens-before it needs.
//
// Serialising is not optional: sam3_create_state copies the model's ggml
// backend into the state (state->backend = model.backend), so two concurrent
// segmentations would drive one Metal backend from two goroutines. libsam3
// has no locking, and the failure mode is a dead daemon.
var (
	gate       = make(chan struct{}, 1)
	handle     *C.oflux_sam3
	handlePath string
	lastUse    time.Time
	reaperOnce sync.Once
)

func Compiled() bool { return true }

func Available() bool {
	_, err := resolveModel()
	return err == nil
}

// EnsureModel downloads the default checkpoint when none is installed;
// Available() flips to true once it succeeds. Segment never calls it, because
// a 707 MB download has no business happening inside an edit request.
func EnsureModel(ctx context.Context, prog func(string)) (string, error) {
	return ensureModel(ctx, prog)
}

func Status() string {
	p, err := resolveModel()
	if err != nil {
		return err.Error()
	}
	return "sam3: ready (" + p + ")"
}

func Segment(ctx context.Context, req Request) ([]Mask, error) {
	if requestIsEmpty(req) {
		return nil, errNoPrompt
	}
	rgb, w, h, err := decodeRGB(req.Image)
	if err != nil {
		return nil, err
	}
	pl, err := validateRequest(req, w, h)
	if err != nil {
		return nil, err
	}
	model, err := resolveModel()
	if err != nil {
		return nil, unavailableError{err}
	}

	select {
	case gate <- struct{}{}:
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	// The cgo call cannot be interrupted, so it runs on its own goroutine and
	// releases the gate when C actually returns. Abandoning the result on
	// cancellation stays memory-safe because this goroutine keeps rgb alive
	// and owns every C allocation involved.
	type outcome struct {
		masks []rawMask
		err   error
	}
	done := make(chan outcome, 1)
	go func() {
		defer func() { <-gate }()
		masks, err := segment(model, rgb, w, h, pl, req.Separate)
		done <- outcome{masks, err}
	}()

	var raw []rawMask
	select {
	case r := <-done:
		if r.err != nil {
			return nil, r.err
		}
		raw = r.masks
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	if len(raw) == 0 {
		return nil, fmt.Errorf("%w: %s", ErrNoMatch, pl.describe())
	}
	// PNG encoding is pure Go, so it happens after the gate is released.
	out := make([]Mask, 0, len(raw))
	for _, m := range raw {
		b64, err := encodeMaskPNG(m.pixels, w, h)
		if err != nil {
			return nil, err
		}
		out = append(out, Mask{B64: b64, Score: m.score, Box: m.box})
	}
	return out, nil
}

type rawMask struct {
	pixels []byte
	score  float32
	box    Box
}

// segment runs with the gate held.
func segment(model string, rgb []byte, w, h int, pl plan, separate bool) ([]rawMask, error) {
	sam, err := open(model)
	if err != nil {
		return nil, err
	}

	var cPrompt *C.char
	if pl.pcs {
		cPrompt = C.CString(pl.prompt)
		defer C.free(unsafe.Pointer(cPrompt))
	}

	// Point and box arrays stay in Go memory: they hold no Go pointers, and
	// the shim copies them into libsam3's own vectors before returning.
	cPoints := make([]C.oflux_sam3_point, len(pl.points))
	for i, p := range pl.points {
		cPoints[i].x = C.float(p.X)
		cPoints[i].y = C.float(p.Y)
		if p.Negative {
			cPoints[i].negative = 1
		}
	}
	var pointsPtr *C.oflux_sam3_point
	if len(cPoints) > 0 {
		pointsPtr = &cPoints[0]
	}
	var boxPtr *C.oflux_sam3_box
	var cBox C.oflux_sam3_box
	if pl.useBox {
		cBox.x0, cBox.y0 = C.float(pl.box.X0), C.float(pl.box.Y0)
		cBox.x1, cBox.y1 = C.float(pl.box.X1), C.float(pl.box.Y1)
		boxPtr = &cBox
	}

	var (
		cErr   [errBufSize]C.char
		cMasks *C.oflux_sam3_mask
		cCount C.int
		iSep   C.int
	)
	if separate {
		iSep = 1
	}

	// rgb is passed as a Go pointer, which is legal here: it holds no Go
	// pointers and libsam3 copies it into a std::vector rather than keeping
	// it. cMasks comes back calloc'd on the C heap and is freed below, after
	// GoBytes has copied every pixel buffer.
	rc := C.oflux_sam3_segment(sam,
		(*C.uchar)(unsafe.Pointer(&rgb[0])), C.int(w), C.int(h),
		cPrompt, pointsPtr, C.int(len(cPoints)), boxPtr,
		C.float(scoreThreshold()), C.float(nmsThreshold()), iSep,
		&cMasks, &cCount, &cErr[0], C.size_t(errBufSize))
	runtime.KeepAlive(rgb)
	runtime.KeepAlive(cPoints)
	lastUse = time.Now()

	if rc != 0 {
		return nil, fmt.Errorf("sam3: segmenting %s: %s", pl.describe(), C.GoString(&cErr[0]))
	}
	if cMasks == nil || cCount == 0 {
		return nil, nil
	}
	defer C.oflux_sam3_free_masks(cMasks, cCount)

	out := make([]rawMask, 0, int(cCount))
	for _, m := range unsafe.Slice(cMasks, int(cCount)) {
		if m.pixels == nil {
			continue
		}
		out = append(out, rawMask{
			pixels: C.GoBytes(unsafe.Pointer(m.pixels), C.int(w*h)),
			score:  float32(m.score),
			box:    goBox(m.box, w, h),
		})
	}
	return out, nil
}

// goBox rounds libsam3's float box to pixels and clamps it: PCS derives boxes
// from normalised centre/size, so they can land just outside the image.
func goBox(b C.oflux_sam3_box, w, h int) Box {
	clamp := func(v C.float, hi int) int {
		i := int(math.Round(float64(v)))
		if i < 0 {
			return 0
		}
		if i > hi {
			return hi
		}
		return i
	}
	return Box{
		X0: clamp(b.x0, w), Y0: clamp(b.y0, h),
		X1: clamp(b.x1, w), Y1: clamp(b.y1, h),
	}
}

// open runs with the gate held.
func open(model string) (*C.oflux_sam3, error) {
	if handle != nil {
		if handlePath == model {
			return handle, nil
		}
		C.oflux_sam3_close(handle)
		handle, handlePath = nil, ""
	}

	cPath := C.CString(model)
	defer C.free(unsafe.Pointer(cPath))
	var cErr [errBufSize]C.char

	h := C.oflux_sam3_open(cPath, C.int(threads()), C.int(gpu()), &cErr[0], C.size_t(errBufSize))
	if h == nil {
		return nil, fmt.Errorf("sam3: loading %s: %s", model, C.GoString(&cErr[0]))
	}
	handle, handlePath, lastUse = h, model, time.Now()
	startReaper()
	return h, nil
}

// startReaper drops the model after idleTTL. A loaded SAM3 holds 700MB–3.4GB
// resident in a daemon that also supervises multi-gigabyte engines.
func startReaper() {
	reaperOnce.Do(func() {
		go func() {
			for range time.Tick(idleSweep) {
				gate <- struct{}{}
				if handle != nil && time.Since(lastUse) > idleTTL {
					C.oflux_sam3_close(handle)
					handle, handlePath = nil, ""
				}
				<-gate
			}
		}()
	})
}

func threads() int {
	n := runtime.NumCPU()
	if n > 8 {
		n = 8
	}
	if n < 1 {
		n = 1
	}
	return n
}

func gpu() int {
	if os.Getenv("OFLUX_SAM3_CPU") != "" {
		return 0
	}
	return 1
}

func scoreThreshold() float64 { return envFloat("OFLUX_SAM3_SCORE", 0.5) }
func nmsThreshold() float64   { return envFloat("OFLUX_SAM3_NMS", 0.1) }
