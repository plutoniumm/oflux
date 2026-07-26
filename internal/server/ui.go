package server

import (
	"bytes"
	_ "embed"
	"net/http"
	"time"
)

// indexHTML is the built-in web UI, embedded so the daemon serves it with no
// external assets. It drives the ordinary HTTP API (/api/tags, /v1/edit,
// /v1/generate) over fetch from the same origin — there is no private UI
// endpoint, so anything the page can do, a script can do too.
//
//go:embed ui/index.html
var indexHTML []byte

// startTime seeds the Last-Modified header so browsers revalidate the page
// after the daemon is upgraded.
var startTime = time.Now()

func (s *Server) handleUI(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	// The page is inert apart from its own inline script: no external origins,
	// no framing, no plugins.
	w.Header().Set("Content-Security-Policy",
		"default-src 'none'; img-src 'self' data:; style-src 'unsafe-inline'; "+
			"script-src 'unsafe-inline'; connect-src 'self'; form-action 'none'; frame-ancestors 'none'")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	http.ServeContent(w, r, "index.html", startTime, bytes.NewReader(indexHTML))
}
