package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

func daemon(t *testing.T, h http.HandlerFunc) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	t.Setenv("OFLUX_HOST", srv.URL)
	return srv
}

func capture(t *testing.T, fn func() error) (string, error) {
	t.Helper()
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	orig := os.Stdout
	os.Stdout = w
	out := make(chan string, 1)
	go func() {
		b, _ := io.ReadAll(r)
		out <- string(b)
	}()
	defer func() {
		os.Stdout = orig
		r.Close()
	}()
	fnErr := fn()
	w.Close()
	return <-out, fnErr
}

func TestNormalizeHost(t *testing.T) {
	for in, want := range map[string]string{
		"http://127.0.0.1:11534/": "http://127.0.0.1:11534",
		"https://box.local":       "https://box.local",
		"box.local:9000":          "http://box.local:9000",
		":9000":                   "http://127.0.0.1:9000",
	} {
		if got := normalizeHost(in); got != want {
			t.Errorf("normalizeHost(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestDaemonBaseHonoursOFLUXHost(t *testing.T) {
	t.Setenv("OFLUX_HOST", "box.local:9000")
	if got := daemonBase(); got != "http://box.local:9000" {
		t.Fatalf("daemonBase = %q", got)
	}
}

func TestGetListDecodesEnvelopes(t *testing.T) {
	daemon(t, func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			io.WriteString(w, `{"models":[{"name":"qwen-image-edit","architecture":"qwen_image","mode":"both","loaded":true}]}`)
		case "/api/ps":
			io.WriteString(w, `{"loaded":["qwen-image-edit"]}`)
		default:
			http.NotFound(w, r)
		}
	})
	models, err := listModels()
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 1 || models[0].Name != "qwen-image-edit" || !models[0].Loaded {
		t.Fatalf("models = %+v", models)
	}
	loaded, err := getList[string]("/api/ps", "loaded")
	if err != nil || len(loaded) != 1 {
		t.Fatalf("loaded = %v, %v", loaded, err)
	}
}

func TestListPrintsPresetsWithTheirLabel(t *testing.T) {
	daemon(t, func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"models":[
			{"name":"qwen-image-edit","architecture":"qwen_image","mode":"both"},
			{"name":"fast","preset":true,"label":"4-step lightning","mode":"edit"}]}`)
	})
	out, err := capture(t, func() error { return cmdList(nil) })
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "4-step lightning") {
		t.Errorf("preset label missing from list output:\n%s", out)
	}
	if !strings.Contains(out, "preset") {
		t.Errorf("preset row not marked as one:\n%s", out)
	}
}

func TestLoraListShowsProvenanceColumns(t *testing.T) {
	daemon(t, func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"loras":[{"name":"hand-dropped","installed":true,"size":849608296,
			"for":["qwen_image"],"steps":4,"cfg":1}]}`)
	})
	out, err := capture(t, cmdLoraList)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"hand-dropped", "installed", "849.6MB", "4", "1", "qwen_image"} {
		if !strings.Contains(out, want) {
			t.Errorf("lora ls missing %q:\n%s", want, out)
		}
	}
}

// Provenance has to arrive with its JSON types intact, not as strings.
func TestLoraPullSendsTypedProvenance(t *testing.T) {
	var body map[string]any
	daemon(t, func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&body)
		io.WriteString(w, `{"status":"pulled"}`+"\n")
	})
	if _, err := capture(t, func() error {
		return cmdLoraPull([]string{"org/repo", "--as", "hand-dropped", "--for", "qwen_image", "--for", "flux", "--steps", "4", "--cfg", "1"})
	}); err != nil {
		t.Fatal(err)
	}
	if body["name"] != "org/repo" || body["as"] != "hand-dropped" {
		t.Fatalf("body = %+v", body)
	}
	if steps, ok := body["steps"].(float64); !ok || steps != 4 {
		t.Errorf("steps = %#v, want the number 4", body["steps"])
	}
	if cfg, ok := body["cfg"].(float64); !ok || cfg != 1 {
		t.Errorf("cfg = %#v, want the number 1", body["cfg"])
	}
	archs, ok := body["for"].([]any)
	if !ok || len(archs) != 2 {
		t.Errorf("for = %#v, want both values", body["for"])
	}
	if _, ok := body["file"]; ok {
		t.Error("an unset flag must not reach the daemon")
	}
}

func TestPresetAddSendsPresetShape(t *testing.T) {
	var body map[string]any
	daemon(t, func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&body)
		io.WriteString(w, `{"status":"created","note":"model \"qwen-image-edit\" is not installed yet"}`)
	})
	out, err := capture(t, func() error {
		return cmdPreset([]string{"add", "fast", "--model", "qwen-image-edit",
			"--lora", "a", "--lora", "b", "--steps", "4", "--cfg", "1",
			"--sampler", "euler", "--label", "4-step lightning"})
	})
	if err != nil {
		t.Fatal(err)
	}
	// A preset for a model that is not installed is legal but worth saying.
	if !strings.Contains(out, "not installed yet") {
		t.Errorf("the daemon's note was dropped: %q", out)
	}
	if body["name"] != "fast" || body["model"] != "qwen-image-edit" || body["label"] != "4-step lightning" {
		t.Fatalf("body = %+v", body)
	}
	if loras, ok := body["loras"].([]any); !ok || len(loras) != 2 {
		t.Errorf("loras = %#v, want both adapters", body["loras"])
	}
	if _, ok := body["steps"].(float64); !ok {
		t.Errorf("steps = %#v, want a number", body["steps"])
	}
	// An omitted override must not be sent, or it would pin a default to zero.
	if _, ok := body["scheduler"]; ok {
		t.Error("an unset override must not reach the daemon")
	}
}

func TestPresetAddNeedsModel(t *testing.T) {
	if err := cmdPresetAdd([]string{"fast"}); err == nil {
		t.Error("preset add without --model should error")
	}
	if err := cmdPresetAdd([]string{"--model", "m"}); err == nil {
		t.Error("preset add without a name should error")
	}
}

func TestPresetListPrintsRows(t *testing.T) {
	daemon(t, func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"presets":[{"name":"fast","model":"qwen-image-edit",
			"loras":["qwen-edit-lightning-4step"],"steps":4,"cfg":1,"label":"4-step lightning"}]}`)
	})
	out, err := capture(t, cmdPresetList)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"fast", "qwen-image-edit", "qwen-edit-lightning-4step", "4", "1", "4-step lightning"} {
		if !strings.Contains(out, want) {
			t.Errorf("preset ls missing %q:\n%s", want, out)
		}
	}
}

func TestJobShowAndStop(t *testing.T) {
	var method, path string
	daemon(t, func(w http.ResponseWriter, r *http.Request) {
		method, path = r.Method, r.URL.Path
		io.WriteString(w, `{"job":"j1","status":"running","model":"qwen-image-edit","step":7,"total":20}`)
	})
	out, err := capture(t, func() error { return cmdJob([]string{"j1"}) })
	if err != nil {
		t.Fatal(err)
	}
	if method != http.MethodGet || path != "/v1/jobs/j1" {
		t.Fatalf("%s %s", method, path)
	}
	if !strings.Contains(out, "running") || !strings.Contains(out, "step 7/20") {
		t.Errorf("job output = %q", out)
	}

	if _, err := capture(t, func() error { return cmdJob([]string{"stop", "j1"}) }); err != nil {
		t.Fatal(err)
	}
	if method != http.MethodDelete {
		t.Fatalf("stop issued %s, want DELETE", method)
	}
}

func TestApiErrorPrefersTheDaemonMessage(t *testing.T) {
	daemon(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		io.WriteString(w, `{"error":"model \"nope\" not installed"}`)
	})
	err := getJSON("/api/tags", &struct{}{})
	if err == nil || !strings.Contains(err.Error(), `not installed`) {
		t.Fatalf("err = %v", err)
	}

	daemon(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		io.WriteString(w, "<html>gateway</html>")
	})
	if err := getJSON("/api/tags", &struct{}{}); err == nil || !strings.Contains(err.Error(), "500") {
		t.Fatalf("err = %v", err)
	}
}

func TestPostStreamPrintsStatusAndStopsAtError(t *testing.T) {
	daemon(t, func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, "{\"status\":\"downloading\"}\n\n{\"status\":\"verifying\"}\n{\"error\":\"disk full\"}\n{\"status\":\"never\"}\n")
	})
	out, err := capture(t, func() error { return postStream("/api/pull", map[string]string{"name": "x"}) })
	if err == nil || err.Error() != "disk full" {
		t.Fatalf("err = %v, want the streamed error", err)
	}
	if !strings.Contains(out, "downloading") || !strings.Contains(out, "verifying") {
		t.Errorf("progress lines missing: %q", out)
	}
	if strings.Contains(out, "never") {
		t.Error("printing continued past the error line")
	}
}

func TestDaemonDownIsExplained(t *testing.T) {
	srv := httptest.NewServer(http.NotFoundHandler())
	url := srv.URL
	srv.Close()
	t.Setenv("OFLUX_HOST", url)

	err := getJSON("/api/tags", &struct{}{})
	if err == nil || !strings.Contains(err.Error(), "cannot reach the oflux daemon") ||
		!strings.Contains(err.Error(), "oflux serve") {
		t.Fatalf("err = %v", err)
	}
}

func TestRemoveEachKeepsGoing(t *testing.T) {
	var seen []string
	daemon(t, func(w http.ResponseWriter, r *http.Request) {
		var req map[string]string
		json.NewDecoder(r.Body).Decode(&req)
		seen = append(seen, req["name"])
		if req["name"] == "b" {
			w.WriteHeader(http.StatusNotFound)
			io.WriteString(w, `{"error":"not installed"}`)
			return
		}
		io.WriteString(w, `{"status":"deleted"}`)
	})
	_, err := capture(t, func() error { return removeEach("/api/delete", "", []string{"a", "b", "c"}) })
	if err == nil || !strings.Contains(err.Error(), "1 of 3 failed") {
		t.Fatalf("err = %v", err)
	}
	if strings.Join(seen, ",") != "a,b,c" {
		t.Fatalf("visited %v, want every name", seen)
	}
}

func TestPrintTablePadsAllButTheLastColumn(t *testing.T) {
	out, err := capture(t, func() error {
		printTable([]int{6, 4}, [][]string{{"NAME", "ARCH", "MODE"}, {"a", "b", "c"}, {"a", "b", ""}})
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	// An empty last column must not leave the row padded with trailing spaces.
	want := "NAME   ARCH MODE\na      b    c\na      b\n"
	if out != want {
		t.Fatalf("printTable =\n%q\nwant\n%q", out, want)
	}
}
