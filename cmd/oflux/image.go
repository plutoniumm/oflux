package main

import (
	"cmp"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"slices"
	"strconv"
	"time"

	"oflux/internal/server"
	"oflux/internal/types"
)

var imageFlags = map[string]string{
	"--output": "output", "-o": "output",
	"--steps": "steps",
	"--cfg":   "cfg", "--guidance": "cfg",
	"--seed":  "seed",
	"--width": "width", "--height": "height",
	"--lora": "lora", "-l": "lora",
	"--negative-prompt": "negative_prompt", "--negative": "negative_prompt",
	"--sampler": "sampler", "--scheduler": "scheduler",
	"--keep-alive": "keep_alive", "--keepalive": "keep_alive",
	"--mask-prompt": "mask_prompt",
	"--ref":         "ref",
}

// The server's own request type, so a renamed field cannot silently stop being
// sent. keep_alive is the one field shadowed: server.Duration only knows how to
// decode, so it would marshal as nanoseconds and be read back as seconds.
type imageBody struct {
	server.ImageRequest
	KeepAlive string `json:"keep_alive,omitempty"`
}

func cmdEdit(args []string) error     { return runImage(types.ModeEdit, args) }
func cmdGenerate(args []string) error { return runImage(types.ModeGenerate, args) }

func runImage(mode types.Mode, args []string) error {
	body, out, err := parseImageArgs(mode, args)
	if err != nil {
		return err
	}
	// Known before the minutes of sampling, not after: with a fixed seed the
	// default filename is already decided.
	if out == "" && body.Seed != nil {
		if _, err := os.Stat(defaultOutput(*body.Seed)); err == nil {
			return existsError(defaultOutput(*body.Seed))
		}
	}
	path := "/v1/generate"
	if mode == types.ModeEdit {
		path = "/v1/edit"
	}
	res, err := streamImage(path, body)
	if err != nil {
		return err
	}
	png, err := base64.StdEncoding.DecodeString(res.Images[0])
	if err != nil {
		return fmt.Errorf("the daemon returned an undecodable image: %w", err)
	}
	// An explicit -o overwrites; the derived name never does.
	flags := os.O_WRONLY | os.O_CREATE | os.O_TRUNC
	if out == "" {
		out, flags = defaultOutput(res.Seed), flags|os.O_EXCL
	}
	f, err := os.OpenFile(out, flags, 0o644)
	if errors.Is(err, os.ErrExist) {
		return existsError(out)
	}
	if err != nil {
		return err
	}
	if _, err := f.Write(png); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	fmt.Printf("%s  seed %d\n", out, res.Seed)
	return nil
}

func defaultOutput(seed int64) string { return fmt.Sprintf("oflux-%d.png", seed) }

func existsError(path string) error {
	return fmt.Errorf("%s already exists — pass -o to name the output (an explicit -o overwrites)", path)
}

// edit: <model> <image>... "<prompt>" — the prompt is last, so a shell glob of
// images can sit in the middle. generate: <model> "<prompt>".
func parseImageArgs(mode types.Mode, args []string) (body imageBody, out string, err error) {
	pos, set, err := parseFlagsMulti(args, imageFlags)
	if err != nil {
		return body, "", err
	}
	if mode == types.ModeEdit {
		if len(pos) < 3 {
			return body, "", errors.New(`usage: oflux edit <model> <image>... "<prompt>" [-o out.png]`)
		}
		for _, p := range pos[1 : len(pos)-1] {
			img, err := readImage(p)
			if err != nil {
				return body, "", err
			}
			body.Image = append(body.Image, img)
		}
	} else if len(pos) != 2 {
		return body, "", errors.New(`usage: oflux generate <model> "<prompt>" [-o out.png]`)
	}
	body.Model, body.Prompt = pos[0], pos[len(pos)-1]
	body.Stream = true
	body.NegativePrompt = last(set["negative_prompt"])
	body.Sampler, body.Scheduler = last(set["sampler"]), last(set["scheduler"])
	body.MaskPrompt = last(set["mask_prompt"])
	for _, n := range set["lora"] {
		body.Loras = append(body.Loras, server.LoraRef{Name: n})
	}
	for _, p := range set["ref"] {
		img, err := readImage(p)
		if err != nil {
			return body, "", err
		}
		body.RefImages = append(body.RefImages, img)
	}
	if body.Steps, err = intFlag(set, "steps"); err != nil {
		return body, "", err
	}
	if body.Width, err = intFlag(set, "width"); err != nil {
		return body, "", err
	}
	if body.Height, err = intFlag(set, "height"); err != nil {
		return body, "", err
	}
	if body.CFG, err = floatFlag(set, "cfg"); err != nil {
		return body, "", err
	}
	// Not intFlag: 0 is a perfectly good seed.
	if raw, ok := set["seed"]; ok {
		seed, err := strconv.ParseInt(last(raw), 10, 64)
		if err != nil || seed < 0 {
			return body, "", fmt.Errorf("--seed wants a whole number >= 0, got %q", last(raw))
		}
		body.Seed = &seed
	}
	if d := last(set["keep_alive"]); d != "" {
		if _, err := time.ParseDuration(d); err != nil {
			return body, "", fmt.Errorf("--keep-alive wants a duration like 10m or 30s, got %q", d)
		}
		body.KeepAlive = d
	}
	return body, last(set["output"]), nil
}

func readImage(path string) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(b), nil
}

// streamImage follows the NDJSON stream to its terminal line: the ImageResponse
// a blocking call would have returned, or an error object. A json.Decoder, not
// a line scanner: that last line carries the whole PNG and has no useful cap.
func streamImage(path string, body imageBody) (server.ImageResponse, error) {
	resp, err := post(path, body)
	if err != nil {
		return server.ImageResponse{}, err
	}
	defer resp.Body.Close()
	dec := json.NewDecoder(resp.Body)
	progress := false
	defer func() {
		if progress {
			fmt.Fprintln(os.Stderr)
		}
	}()
	for {
		var raw json.RawMessage
		if err := dec.Decode(&raw); err != nil {
			if errors.Is(err, io.EOF) {
				return server.ImageResponse{}, errors.New("the stream ended without an image — check: oflux job <id>")
			}
			return server.ImageResponse{}, err
		}
		var end struct {
			server.ImageResponse
			Error string `json:"error"`
		}
		if err := json.Unmarshal(raw, &end); err != nil {
			return server.ImageResponse{}, err
		}
		switch {
		case end.Error != "":
			return server.ImageResponse{}, errors.New(end.Error)
		case len(end.Images) > 0:
			return end.ImageResponse, nil
		}
		var l server.StreamLine
		_ = json.Unmarshal(raw, &l)
		progress = true
		fmt.Fprintf(os.Stderr, "\r%-60s", jobLine(l))
	}
}

func jobLine(j server.StreamLine) string {
	line := cmp.Or(j.Status, "done")
	if j.Model != "" {
		line += "  " + j.Model
	}
	if j.Total > 0 {
		line += fmt.Sprintf("  step %d/%d", j.Step, j.Total)
	}
	if j.Elapsed > 0 {
		line += fmt.Sprintf("  %.0fs", j.Elapsed)
	}
	return line
}

func findModel(name string) (server.ModelRow, bool, error) {
	models, err := listModels()
	if err != nil {
		return server.ModelRow{}, false, err
	}
	i := slices.IndexFunc(models, func(m server.ModelRow) bool { return m.Name == name })
	if i < 0 {
		return server.ModelRow{}, false, nil
	}
	return models[i], true, nil
}
