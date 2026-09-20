package server

import (
	"encoding/base64"
	"io"
	"mime/multipart"
	"net/http"
	"strconv"
	"strings"
	"time"

	"oflux/internal/types"
)

// OpenAI-compatible image endpoints. They only translate their input into an
// ImageRequest; validation, generation and error mapping stay the native path's.
//
//   POST /v1/images/edits        multipart: image[], mask, prompt, model, size
//   POST /v1/images/generations  JSON:      {model, prompt, size}
//
// Both answer in the OpenAI shape, {created, model, data:[{b64_json}]}, and fail
// in OpenAI's — a client that speaks this dialect parses the error too.

func (s *Server) handleOpenAIEdit(w http.ResponseWriter, r *http.Request) {
	req, err := openAIEditRequest(r)
	if err != nil {
		failOpenAI(w, err)
		return
	}
	s.serveOpenAI(w, r, req, types.ModeEdit)
}

func (s *Server) handleOpenAIGenerate(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Model  string `json:"model"`
		Prompt string `json:"prompt"`
		Size   string `json:"size"`
	}
	if err := decodeJSON(r, &body); err != nil {
		failOpenAI(w, err)
		return
	}
	req := &ImageRequest{Model: body.Model, Prompt: body.Prompt}
	req.Width, req.Height = parseSize(body.Size)
	s.serveOpenAI(w, r, req, types.ModeGenerate)
}

func (s *Server) serveOpenAI(w http.ResponseWriter, r *http.Request, req *ImageRequest, mode types.Mode) {
	res, err := s.generate(r.Context(), req, mode)
	if err != nil {
		failOpenAI(w, err)
		return
	}
	data := make([]map[string]string, 0, len(res.Images))
	for _, img := range res.Images {
		data = append(data, map[string]string{"b64_json": img})
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"created": time.Now().Unix(),
		"model":   res.Model,
		"data":    data,
	})
}

func openAIEditRequest(r *http.Request) (*ImageRequest, error) {
	if err := r.ParseMultipartForm(multipartMemory); err != nil {
		return nil, badRequest("invalid multipart form: %w", err)
	}
	req := &ImageRequest{Model: r.FormValue("model"), Prompt: r.FormValue("prompt")}
	req.Width, req.Height = parseSize(r.FormValue("size"))

	var files []*multipart.FileHeader
	if r.MultipartForm != nil {
		files = append(files, r.MultipartForm.File["image"]...)
		files = append(files, r.MultipartForm.File["image[]"]...)
	}
	if len(files) == 0 {
		return nil, badRequest("at least one image file is required")
	}
	for i, fh := range files {
		b, err := fileToBase64(fh)
		if err != nil {
			return nil, badRequest("reading image %d: %w", i, err)
		}
		req.Image = append(req.Image, b)
	}
	if masks := r.MultipartForm.File["mask"]; len(masks) > 0 {
		b, err := fileToBase64(masks[0])
		if err != nil {
			return nil, badRequest("reading mask: %w", err)
		}
		req.MaskImage = b
	}
	return req, nil
}

// Nils for empty or malformed input, so the model's default size applies.
func parseSize(s string) (*int, *int) {
	s = strings.TrimSpace(strings.ToLower(s))
	if s == "" {
		return nil, nil
	}
	parts := strings.SplitN(s, "x", 2)
	if len(parts) != 2 {
		return nil, nil
	}
	wv, err1 := strconv.Atoi(strings.TrimSpace(parts[0]))
	hv, err2 := strconv.Atoi(strings.TrimSpace(parts[1]))
	if err1 != nil || err2 != nil || wv <= 0 || hv <= 0 {
		return nil, nil
	}
	return &wv, &hv
}

// Streams rather than reading the file into a []byte first: an upload is
// megabytes, and the raw copy would sit alongside the encoded one for nothing.
func fileToBase64(fh *multipart.FileHeader) (string, error) {
	f, err := fh.Open()
	if err != nil {
		return "", err
	}
	defer f.Close()
	var b strings.Builder
	b.Grow(base64.StdEncoding.EncodedLen(int(fh.Size)))
	enc := base64.NewEncoder(base64.StdEncoding, &b)
	if _, err := io.Copy(enc, f); err != nil {
		return "", err
	}
	if err := enc.Close(); err != nil {
		return "", err
	}
	return b.String(), nil
}
