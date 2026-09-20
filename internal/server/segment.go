package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	"image/png"
	"net/http"

	"oflux/internal/sam3"
)

type SegmentRequest struct {
	Image  ImageInput     `json:"image"`
	Prompt string         `json:"prompt,omitempty"`
	Points []SegmentPoint `json:"points,omitempty"`
	Boxes  []SegmentBox   `json:"boxes,omitempty"`
	// Separate returns one mask per instance found instead of one merged mask.
	Separate bool `json:"separate,omitempty"`
}

type SegmentPoint struct {
	X int `json:"x"`
	Y int `json:"y"`
	// Negative marks a point to exclude: "this thing, but not that part of it".
	Negative bool `json:"negative,omitempty"`
}

type SegmentBox struct {
	X0 int `json:"x0"`
	Y0 int `json:"y0"`
	X1 int `json:"x1"`
	Y1 int `json:"y1"`
}

type SegmentMask struct {
	Mask  string     `json:"mask"`
	Score float32    `json:"score,omitempty"`
	Box   SegmentBox `json:"box"`
}

type SegmentResponse struct {
	Masks []SegmentMask `json:"masks"`
}

func (s *Server) handleSegment(w http.ResponseWriter, r *http.Request) {
	var req SegmentRequest
	if err := decodeJSON(r, &req); err != nil {
		fail(w, err)
		return
	}
	if len(req.Image) == 0 {
		fail(w, badRequest("image is required"))
		return
	}
	text, clicks := req.Prompt != "", len(req.Points) > 0 || len(req.Boxes) > 0
	if !text && !clicks {
		fail(w, badRequest("say what to segment: a prompt, points or boxes"))
		return
	}
	// libsam3 has two decoders and no path taking both: text is PCS, clicks PVS.
	if text && clicks {
		fail(w, badRequest("use either a prompt or points/boxes, not both"))
		return
	}
	if err := checkPrompt("prompt", req.Prompt); err != nil {
		fail(w, err)
		return
	}
	// libsam3 validates none of its inputs, so a bad one is a segfault that takes
	// the daemon rather than an error return. Everything is checked here first.
	raw, cfg, err := decodeToPNG("image", req.Image.First())
	if err != nil {
		fail(w, err)
		return
	}
	points, boxes, err := segmentPrompts(req, cfg)
	if err != nil {
		fail(w, err)
		return
	}

	masks, err := runSegment(r.Context(), sam3.Request{
		Image: raw, Prompt: req.Prompt, Points: points, Boxes: boxes, Separate: req.Separate,
	}, nil)
	if err != nil {
		fail(w, err)
		return
	}
	out := make([]SegmentMask, 0, len(masks))
	for _, m := range masks {
		out = append(out, SegmentMask{
			Mask:  m.B64,
			Score: m.Score,
			Box:   SegmentBox{X0: m.Box.X0, Y0: m.Box.Y0, X1: m.Box.X1, Y1: m.Box.Y1},
		})
	}
	writeJSON(w, http.StatusOK, SegmentResponse{Masks: out})
}

// prog narrates the checkpoint download and may be nil.
func runSegment(ctx context.Context, req sam3.Request, prog func(string)) ([]sam3.Mask, error) {
	if prog == nil {
		prog = func(string) {}
	}
	masks, err := sam3.Segment(ctx, req)
	// Unavailable means either no library or no checkpoint. Only the second is
	// fixable, and only a linked build should ever start the download.
	if errors.Is(err, sam3.ErrUnavailable) && sam3.Compiled() {
		if _, derr := sam3.EnsureModel(ctx, prog); derr != nil {
			return nil, statusErr(http.StatusBadGateway, fmt.Errorf("sam3 checkpoint: %w", derr))
		}
		masks, err = sam3.Segment(ctx, req)
	}
	switch {
	case errors.Is(err, sam3.ErrNoMatch):
		return nil, noMatch(req.Prompt)
	case errors.Is(err, sam3.ErrUnavailable):
		return nil, statusErr(http.StatusNotImplemented, err)
	case err != nil:
		return nil, statusErr(http.StatusBadGateway, fmt.Errorf("segmentation failed: %w", err))
	case len(masks) == 0:
		return nil, noMatch(req.Prompt)
	}
	return masks, nil
}

// 422, not 404: the request was fine and the image simply holds nothing
// matching, where 404 would collide with "model not installed".
func noMatch(prompt string) error {
	if prompt == "" {
		return coded(codeNoMatch, statusErr(http.StatusUnprocessableEntity,
			errors.New("nothing matched in the image")))
	}
	return coded(codeNoMatch, statusErr(http.StatusUnprocessableEntity,
		fmt.Errorf("%q matched nothing in the image", prompt)))
}

func checkPrompt(field, p string) error {
	if len(p) > sam3.MaxPromptBytes {
		return badRequest("%s is %d bytes; the text encoder takes at most %d", field, len(p), sam3.MaxPromptBytes)
	}
	return nil
}

func segmentPrompts(req SegmentRequest, cfg image.Config) ([]sam3.Point, []sam3.Box, error) {
	if len(req.Points) > sam3.MaxPoints || len(req.Boxes) > sam3.MaxPoints {
		return nil, nil, badRequest("at most %d points and %d boxes", sam3.MaxPoints, sam3.MaxPoints)
	}
	points := make([]sam3.Point, 0, len(req.Points))
	for i, p := range req.Points {
		if p.X < 0 || p.X >= cfg.Width || p.Y < 0 || p.Y >= cfg.Height {
			return nil, nil, badRequest("points[%d] (%d,%d) is outside the %dx%d image",
				i, p.X, p.Y, cfg.Width, cfg.Height)
		}
		points = append(points, sam3.Point{X: p.X, Y: p.Y, Negative: p.Negative})
	}
	boxes := make([]sam3.Box, 0, len(req.Boxes))
	for i, b := range req.Boxes {
		if b.X0 < 0 || b.Y0 < 0 || b.X0 >= b.X1 || b.Y0 >= b.Y1 || b.X1 > cfg.Width || b.Y1 > cfg.Height {
			return nil, nil, badRequest("boxes[%d] (%d,%d)-(%d,%d) is not inside the %dx%d image",
				i, b.X0, b.Y0, b.X1, b.Y1, cfg.Width, cfg.Height)
		}
		boxes = append(boxes, sam3.Box{X0: b.X0, Y0: b.Y0, X1: b.X1, Y1: b.Y1})
	}
	return points, boxes, nil
}

// resolveMask turns mask_prompt into the mask field, so one call segments and
// edits instead of making the caller round-trip a mask. An explicit mask wins.
func resolveMask(ctx context.Context, req *ImageRequest, prog func(string)) error {
	if req.MaskPrompt == "" || req.MaskImage != "" {
		return nil
	}
	raw, _, err := decodeToPNG("image", req.Image.First())
	if err != nil {
		return err
	}
	masks, err := runSegment(ctx, sam3.Request{Image: raw, Prompt: req.MaskPrompt}, prog)
	if err != nil {
		return err
	}
	req.MaskImage = masks[0].B64
	return nil
}

func decodeToPNG(field, v string) ([]byte, image.Config, error) {
	var cfg image.Config
	b64, err := normalizeImage(field, v)
	if err != nil {
		return nil, cfg, err
	}
	raw, err := decodeB64(b64)
	if err != nil {
		return nil, cfg, badRequest("%s is not valid base64", field)
	}
	if len(raw) > sam3.MaxImageBytes {
		return nil, cfg, badRequest("%s is %d bytes, over the %d-byte limit", field, len(raw), sam3.MaxImageBytes)
	}
	// DecodeConfig reads the header only, so an absurd declared size is refused
	// before the pixels are ever allocated. The bounds are libsam3's own.
	cfg, format, err := image.DecodeConfig(bytes.NewReader(raw))
	if err != nil {
		return nil, cfg, badRequest("%s is not a PNG or JPEG image", field)
	}
	if cfg.Width < sam3.MinSide || cfg.Height < sam3.MinSide ||
		cfg.Width > sam3.MaxSide || cfg.Height > sam3.MaxSide {
		return nil, cfg, badRequest("%s is %dx%d; each side must be between %d and %d",
			field, cfg.Width, cfg.Height, sam3.MinSide, sam3.MaxSide)
	}
	if format == "png" {
		return raw, cfg, nil
	}
	img, _, err := image.Decode(bytes.NewReader(raw))
	if err != nil {
		return nil, cfg, badRequest("%s could not be decoded: %v", field, err)
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil, cfg, err
	}
	return buf.Bytes(), cfg, nil
}

func decodeB64(s string) ([]byte, error) {
	if b, err := base64.StdEncoding.DecodeString(s); err == nil {
		return b, nil
	}
	return base64.RawStdEncoding.DecodeString(s)
}
