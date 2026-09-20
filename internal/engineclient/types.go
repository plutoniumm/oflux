package engineclient

import (
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
)

type SLG struct {
	Layers     []int    `json:"layers,omitempty"`
	LayerStart *float64 `json:"layer_start,omitempty"`
	LayerEnd   *float64 `json:"layer_end,omitempty"`
	Scale      *float64 `json:"scale,omitempty"`
}

type Guidance struct {
	TxtCFG            *float64 `json:"txt_cfg,omitempty"`
	ImgCFG            *float64 `json:"img_cfg,omitempty"`
	DistilledGuidance *float64 `json:"distilled_guidance,omitempty"`
	SLG               *SLG     `json:"slg,omitempty"`
}

type SampleParams struct {
	Scheduler    string    `json:"scheduler,omitempty"`
	SampleMethod string    `json:"sample_method,omitempty"`
	SampleSteps  *int      `json:"sample_steps,omitempty"`
	Guidance     *Guidance `json:"guidance,omitempty"`
}

// Path is resolved against --lora-model-dir and must include the extension. The
// engine does NOT parse "<lora:name:scale>" prompt tags (verified: tokenized as
// literal text), so this struct is the only way to apply a LoRA over HTTP.
type Lora struct {
	Path       string  `json:"path"`
	Multiplier float64 `json:"multiplier"`
}

// Image fields carry raw base64, no data: prefix. Every optional numeric knob is
// a pointer with omitempty on purpose: the engine distinguishes absent (keep the
// value baked into the launch flags at pull time) from present-and-zero, so a
// plain int would silently reset the model's defaults on every request.
type ImgGenRequest struct {
	Prompt         string `json:"prompt"`
	NegativePrompt string `json:"negative_prompt,omitempty"`

	// Loras marshals to "lora" (singular), the engine's field name.
	Loras []Lora `json:"lora,omitempty"`

	Width  *int `json:"width,omitempty"`
	Height *int `json:"height,omitempty"`

	Strength *float64 `json:"strength,omitempty"`
	Seed     *int64   `json:"seed,omitempty"`

	// oflux never sends InitImage: edit models take the input image as RefImages,
	// because init_image denoises the input away at the default strength.
	InitImage string   `json:"init_image,omitempty"`
	RefImages []string `json:"ref_images,omitempty"`
	MaskImage string   `json:"mask_image,omitempty"`

	ControlImage    string   `json:"control_image,omitempty"`
	ControlStrength *float64 `json:"control_strength,omitempty"`

	SampleParams *SampleParams `json:"sample_params,omitempty"`
	OutputFormat string        `json:"output_format,omitempty"`
}

// Result is left raw so ImagesB64 can tolerate shape differences across engine
// builds. The engine's "poll_url" is not modelled: it is always exactly the
// /sdcpp/v1/jobs/{id} path Poll builds from the id, and cancel has to be built
// from the id anyway, so honouring it for one call in two would be a half-promise.
type Job struct {
	ID     string          `json:"id"`
	Status string          `json:"status"`
	Error  string          `json:"error"`
	Result json.RawMessage `json:"result"`
}

// Reason is the engine's raw failure text, which for a model that cannot run is
// the whole sd.cpp log for the attempt — hundreds of lines. Callers are expected
// to show a short message and truncate Reason into a detail field.
type JobError struct{ Op, Status, Reason string }

func (e *JobError) Error() string {
	op := cmp.Or(e.Op, "job")
	status := cmp.Or(e.Status, "failed")
	if e.Reason == "" {
		return fmt.Sprintf("engine: %s %s", op, status)
	}
	return fmt.Sprintf("engine: %s %s: %s", op, status, e.Reason)
}

func isTerminal(status string) bool {
	switch status {
	case "completed", "failed", "cancelled":
		return true
	default:
		return false
	}
}

// Engine builds and OpenAI-compatible shims key the payload as either "b64_json"
// or "data", so both are accepted.
type b64Item struct {
	B64JSON string `json:"b64_json"`
	Data    string `json:"data"`
}

// Native sd-server is {"images":[...]}, the OpenAI-like shape {"data":[...]}.
type imgResult struct {
	Images []b64Item `json:"images"`
	Data   []b64Item `json:"data"`
}

// ImagesB64 returns the result images as RAW base64, deliberately undecoded: the
// engine hands us base64 and every oflux response hands base64 back out, so a
// decode here would cost a decode plus a re-encode upstream (and ~2x the image in
// garbage) per request for a string we already had.
func (j Job) ImagesB64() ([]string, error) {
	if len(j.Result) == 0 || string(j.Result) == "null" {
		return nil, errors.New("engine: job has no result")
	}
	var r imgResult
	if err := json.Unmarshal(j.Result, &r); err != nil {
		return nil, fmt.Errorf("engine: decode result: %w", err)
	}
	items := r.Images
	if len(items) == 0 {
		items = r.Data
	}
	if len(items) == 0 {
		return nil, errors.New("engine: result carries no images")
	}
	out := make([]string, 0, len(items))
	for i, it := range items {
		enc := cmp.Or(it.B64JSON, it.Data)
		if enc == "" {
			return nil, fmt.Errorf("engine: image %d has no base64 payload", i)
		}
		out = append(out, padB64(enc))
	}
	return out, nil
}

// padB64 restores the padding some producers omit — the one normalisation
// ImagesB64 still does, because it is string arithmetic rather than a decode.
func padB64(enc string) string {
	switch len(enc) % 4 {
	case 2:
		return enc + "=="
	case 3:
		return enc + "="
	}
	// 1 is not valid base64 in any encoding; let the consumer report it.
	return enc
}
