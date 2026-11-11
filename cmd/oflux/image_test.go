package main

import (
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"oflux/internal/types"
)

func writeFile(t *testing.T, dir, name, content string) string {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestParseEditArgsPromptLastImagesBetween(t *testing.T) {
	dir := t.TempDir()
	a, b, ref := writeFile(t, dir, "a.png", "AAA"), writeFile(t, dir, "b.png", "BBB"), writeFile(t, dir, "r.png", "RRR")
	body, out, err := parseImageArgs(types.ModeEdit, []string{
		"qwe-2511", a, b, "make it night", "-o", "x.png", "--seed", "0", "--steps", "4",
		"--lora", "l1", "--lora", "l2", "--ref", ref, "--keep-alive", "10m",
	})
	if err != nil {
		t.Fatal(err)
	}
	if body.Model != "qwe-2511" || body.Prompt != "make it night" || out != "x.png" {
		t.Fatalf("model=%q prompt=%q out=%q", body.Model, body.Prompt, out)
	}
	want := base64.StdEncoding.EncodeToString([]byte("BBB"))
	if len(body.Image) != 2 || body.Image[1] != want || len(body.RefImages) != 1 || len(body.Loras) != 2 {
		t.Fatalf("images=%v refs=%v loras=%v", body.Image, body.RefImages, body.Loras)
	}
	if body.Seed == nil || *body.Seed != 0 {
		t.Fatalf("seed 0 must survive, got %v", body.Seed)
	}
	// The wire form: keep_alive as the string the server parses, and streamed.
	raw, _ := json.Marshal(body)
	var wire map[string]any
	json.Unmarshal(raw, &wire)
	if wire["keep_alive"] != "10m" || wire["stream"] != true || wire["steps"] != float64(4) {
		t.Fatalf("wire = %s", raw)
	}
	if _, ok := wire["width"]; ok {
		t.Error("an unset flag must not reach the daemon")
	}

	for _, bad := range [][]string{
		{"qwe-2511", "just a prompt"},             // no image
		{"qwe-2511", a, "p", "--seed", "-1"},      // negative seed
		{"qwe-2511", "missing.png", "p"},          // unreadable image
		{"qwe-2511", a, "p", "--keep-alive", "x"}, // not a duration
	} {
		if _, _, err := parseImageArgs(types.ModeEdit, bad); err == nil {
			t.Errorf("%v should be rejected", bad)
		}
	}
	if _, _, err := parseImageArgs(types.ModeGenerate, []string{"zim-1-turbo", a, "p"}); err == nil {
		t.Error("generate takes no image positional")
	}
}

func TestGenerateStreamsToAFile(t *testing.T) {
	t.Chdir(t.TempDir())
	png := base64.StdEncoding.EncodeToString([]byte("\x89PNG fake"))
	var path string
	var body map[string]any
	daemon(t, func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		json.NewDecoder(r.Body).Decode(&body)
		io.WriteString(w, `{"status":"queued","model":"zim-1-turbo","job":"j1"}`+"\n"+
			`{"status":"running","model":"zim-1-turbo","job":"j1","step":2,"total":8,"elapsed_seconds":3}`+"\n"+
			`{"model":"zim-1-turbo","images":["`+png+`"],"seed":42}`+"\n")
	})
	out, err := capture(t, func() error { return cmdGenerate([]string{"zim-1-turbo", "a fox"}) })
	if err != nil {
		t.Fatal(err)
	}
	if path != "/v1/generate" || body["prompt"] != "a fox" || body["stream"] != true {
		t.Fatalf("%s %+v", path, body)
	}
	if got, _ := os.ReadFile("oflux-42.png"); string(got) != "\x89PNG fake" {
		t.Fatalf("file = %q", got)
	}
	if !strings.Contains(out, "oflux-42.png") || !strings.Contains(out, "seed 42") {
		t.Errorf("out = %q", out)
	}

	// The derived name never overwrites, and with a fixed seed says so up front.
	hits := 0
	daemon(t, func(w http.ResponseWriter, r *http.Request) { hits++ })
	_, err = capture(t, func() error { return cmdGenerate([]string{"zim-1-turbo", "a fox", "--seed", "42"}) })
	if err == nil || !strings.Contains(err.Error(), "already exists") || hits != 0 {
		t.Fatalf("err = %v, daemon hits = %d", err, hits)
	}
}

func TestEditStreamErrorLineEndsTheCommand(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	img := writeFile(t, dir, "in.png", "x")
	var path string
	daemon(t, func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		io.WriteString(w, `{"status":"queued","job":"j1"}`+"\n"+`{"error":"engine job failed: out of memory","code":"engine_failed"}`+"\n")
	})
	_, err := capture(t, func() error { return cmdEdit([]string{"qwe-2511", img, "night"}) })
	if err == nil || !strings.Contains(err.Error(), "out of memory") || path != "/v1/edit" {
		t.Fatalf("err = %v, path = %s", err, path)
	}
	if m, _ := filepath.Glob("oflux-*.png"); len(m) != 0 {
		t.Errorf("a failed edit wrote %v", m)
	}
}

func TestRunHintFollowsTheModelsMode(t *testing.T) {
	daemon(t, func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"models":[{"name":"zim-1-turbo","mode":"generate"},{"name":"flx-1-kontext","mode":"edit"},{"name":"qwe-2511","mode":"both"}]}`)
	})
	for name, want := range map[string][2]bool{ // {edit hint, generate hint}
		"zim-1-turbo": {false, true}, "flx-1-kontext": {true, false}, "qwe-2511": {true, true},
	} {
		out, err := capture(t, func() error { return cmdRun([]string{name}) })
		if err != nil {
			t.Fatal(err)
		}
		got := [2]bool{strings.Contains(out, "oflux edit "+name), strings.Contains(out, "oflux generate "+name)}
		if got != want {
			t.Errorf("%s: hints (edit, generate) = %v, want %v\n%s", name, got, want, out)
		}
	}
}
