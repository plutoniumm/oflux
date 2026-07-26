package engineclient

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
)

// SLG holds skip-layer-guidance parameters.
type SLG struct {
	Layers     []int    `json:"layers,omitempty"`
	LayerStart *float64 `json:"layer_start,omitempty"`
	LayerEnd   *float64 `json:"layer_end,omitempty"`
	Scale      *float64 `json:"scale,omitempty"`
}

// Guidance groups the various guidance knobs the engine understands.
type Guidance struct {
	TxtCFG            *float64 `json:"txt_cfg,omitempty"`
	ImgCFG            *float64 `json:"img_cfg,omitempty"`
	DistilledGuidance *float64 `json:"distilled_guidance,omitempty"`
	SLG               *SLG     `json:"slg,omitempty"`
}

// SampleParams controls the sampler used for a generation.
type SampleParams struct {
	Scheduler    string    `json:"scheduler,omitempty"`
	SampleMethod string    `json:"sample_method,omitempty"`
	SampleSteps  *int      `json:"sample_steps,omitempty"`
	Eta          *float64  `json:"eta,omitempty"`
	Guidance     *Guidance `json:"guidance,omitempty"`
}

// Lora is one LoRA to apply to a generation. Path is resolved relative to the
// engine's --lora-model-dir and must include the file extension.
//
// The engine deliberately does NOT parse "<lora:name:scale>" tags out of the
// prompt (verified against sd-server: such a tag is tokenized as literal prompt
// text). Structured entries here are the only way to apply a LoRA over HTTP.
type Lora struct {
	Path       string  `json:"path"`
	Multiplier float64 `json:"multiplier"`
}

// ImgGenRequest is the body of POST /sdcpp/v1/img_gen. All base64-image fields
// carry raw base64 (no data: prefix), matching the sd-server native API.
type ImgGenRequest struct {
	Prompt         string `json:"prompt"`
	NegativePrompt string `json:"negative_prompt,omitempty"`

	// Loras marshals to "lora" (singular), the engine's field name.
	Loras []Lora `json:"lora,omitempty"`

	Width  *int `json:"width,omitempty"`
	Height *int `json:"height,omitempty"`

	Strength   *float64 `json:"strength,omitempty"`
	Seed       *int64   `json:"seed,omitempty"`
	BatchCount *int     `json:"batch_count,omitempty"`

	InitImage string   `json:"init_image,omitempty"`
	RefImages []string `json:"ref_images,omitempty"`
	MaskImage string   `json:"mask_image,omitempty"`

	ControlImage    string   `json:"control_image,omitempty"`
	ControlStrength *float64 `json:"control_strength,omitempty"`

	SampleParams *SampleParams `json:"sample_params,omitempty"`
	OutputFormat string        `json:"output_format,omitempty"`
}

// Job is the engine's view of a submitted generation. PollURL is only populated
// by the img_gen (Submit) response; Result and Error are populated once the job
// reaches a terminal state. Result is left raw so callers decode it via
// ImagesPNG, tolerating small shape differences across engine builds.
type Job struct {
	ID      string          `json:"id"`
	Status  string          `json:"status"`
	PollURL string          `json:"poll_url"`
	Error   string          `json:"error"`
	Result  json.RawMessage `json:"result"`
}

// isTerminal reports whether a job status will not change further.
func isTerminal(status string) bool {
	switch status {
	case "completed", "failed", "cancelled":
		return true
	default:
		return false
	}
}

// b64Item is one image entry inside a completed job's result. Different engine
// builds (and OpenAI-compatible shims) key the payload as either "b64_json" or
// "data", so we accept both.
type b64Item struct {
	B64JSON string `json:"b64_json"`
	Data    string `json:"data"`
}

// imgResult is the completed-job result envelope. The native sd-server shape is
// {"images":[{"b64_json":...}]}; the OpenAI-like shape is
// {"data":[{"b64_json":...}]}. Both are handled.
type imgResult struct {
	Images []b64Item `json:"images"`
	Data   []b64Item `json:"data"`
}

// ImagesPNG decodes every base64 image carried in a completed job's result.
// It tolerates {"images":[{"b64_json":...}]}, {"images":[{"data":...}]} and
// the OpenAI-like {"data":[{"b64_json":...}]} shapes.
func (j Job) ImagesPNG() ([][]byte, error) {
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
	out := make([][]byte, 0, len(items))
	for i, it := range items {
		enc := it.B64JSON
		if enc == "" {
			enc = it.Data
		}
		if enc == "" {
			return nil, fmt.Errorf("engine: image %d has no base64 payload", i)
		}
		raw, err := base64.StdEncoding.DecodeString(enc)
		if err != nil {
			// Some producers omit padding.
			raw, err = base64.RawStdEncoding.DecodeString(enc)
			if err != nil {
				return nil, fmt.Errorf("engine: decode image %d: %w", i, err)
			}
		}
		out = append(out, raw)
	}
	return out, nil
}
