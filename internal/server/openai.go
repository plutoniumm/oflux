package server

import (
	"encoding/base64"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
	"strconv"
	"strings"
	"time"

	"oflux/internal/types"
)

// OpenAI-compatible image endpoints, so existing OpenAI image clients can point
// at oflux. They translate onto the same generate() path as the native API.
//
//   POST /v1/images/edits        multipart: image[], mask, prompt, model, size
//   POST /v1/images/generations  JSON:      {model, prompt, size}
//
// Both respond with the OpenAI shape: {created, model, data:[{b64_json}]}.

func (s *Server) handleOpenAIEdit(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseMultipartForm(64 << 20); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid multipart form: "+err.Error())
		return
	}
	model := r.FormValue("model")
	if model == "" {
		writeErr(w, http.StatusBadRequest, "model is required")
		return
	}
	req := ImageRequest{Model: model, Prompt: r.FormValue("prompt")}
	req.Width, req.Height = parseSize(r.FormValue("size"))

	var files []*multipart.FileHeader
	if r.MultipartForm != nil {
		files = append(files, r.MultipartForm.File["image"]...)
		files = append(files, r.MultipartForm.File["image[]"]...)
	}
	if len(files) == 0 {
		writeErr(w, http.StatusBadRequest, "at least one image file is required")
		return
	}
	// First image is the one being edited; any extras are reference images.
	b, err := fileToBase64(files[0])
	if err != nil {
		writeErr(w, http.StatusBadRequest, "reading image: "+err.Error())
		return
	}
	req.Image = b
	for _, fh := range files[1:] {
		if rb, err := fileToBase64(fh); err == nil {
			req.RefImages = append(req.RefImages, rb)
		}
	}
	if masks := r.MultipartForm.File["mask"]; len(masks) > 0 {
		if mb, err := fileToBase64(masks[0]); err == nil {
			req.MaskImage = mb
		}
	}

	name, img, code, err := s.generate(r.Context(), model, req, types.ModeEdit)
	if err != nil {
		writeErr(w, code, err.Error())
		return
	}
	writeOpenAIImages(w, name, img)
}

func (s *Server) handleOpenAIGenerate(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Model  string `json:"model"`
		Prompt string `json:"prompt"`
		Size   string `json:"size"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}
	if body.Model == "" {
		writeErr(w, http.StatusBadRequest, "model is required")
		return
	}
	req := ImageRequest{Model: body.Model, Prompt: body.Prompt}
	req.Width, req.Height = parseSize(body.Size)

	name, img, code, err := s.generate(r.Context(), body.Model, req, types.ModeGenerate)
	if err != nil {
		writeErr(w, code, err.Error())
		return
	}
	writeOpenAIImages(w, name, img)
}

func writeOpenAIImages(w http.ResponseWriter, model string, imgs ...[]byte) {
	data := make([]map[string]string, 0, len(imgs))
	for _, b := range imgs {
		data = append(data, map[string]string{"b64_json": base64.StdEncoding.EncodeToString(b)})
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"created": time.Now().Unix(),
		"model":   model,
		"data":    data,
	})
}

// parseSize turns "1024x768" into width/height pointers; returns nils for empty
// or malformed input (so the model's default size applies).
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

func fileToBase64(fh *multipart.FileHeader) (string, error) {
	f, err := fh.Open()
	if err != nil {
		return "", err
	}
	defer f.Close()
	b, err := io.ReadAll(f)
	if err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(b), nil
}
