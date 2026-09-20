package server

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"sync"
	"time"

	"oflux/internal/supervisor"
	"oflux/internal/types"
)

// StreamLine is one progress line. The terminal line is never one of these: it
// is the ImageResponse — or the error object — a blocking call would return.
type StreamLine struct {
	Status  string  `json:"status"` // queued | running, or a free-text step like a pull's
	Model   string  `json:"model,omitempty"`
	Job     string  `json:"job,omitempty"`
	Step    int     `json:"step,omitempty"`
	Total   int     `json:"total,omitempty"`
	Elapsed float64 `json:"elapsed_seconds,omitempty"`
}

const (
	jobPoll   = 250 * time.Millisecond
	heartbeat = 2 * time.Second
)

func (s *Server) streamImage(w http.ResponseWriter, r *http.Request, req *ImageRequest, mode types.Mode) {
	m, err := s.validate(req, mode)
	if err != nil {
		fail(w, withModel(err, req.Model))
		return
	}
	out := newNDJSON(w)
	// Segmenting can mean downloading a checkpoint, narrated like a pull.
	if err := resolveMask(r.Context(), req, func(msg string) {
		out.line(StreamLine{Status: msg, Model: m.Name})
	}); err != nil {
		streamFail(w, out, withModel(err, m.Name))
		return
	}
	// The job deliberately outlives this request: a client that drops the stream
	// reattaches with GET /v1/jobs/{id} rather than losing a multi-minute run.
	id, err := s.sup.SubmitJob(context.WithoutCancel(r.Context()), m,
		buildImgGen(m, *req, mode), s.genOpts(req))
	if err != nil {
		streamFail(w, out, withModel(generationErr(err), m.Name))
		return
	}
	seed := *req.Seed
	s.jobs.remember(id, m.Name, seed)

	last := StreamLine{Status: "queued", Model: m.Name, Job: id}
	out.line(last)

	tick := time.NewTicker(jobPoll)
	defer tick.Stop()
	started, beat := time.Now(), time.Now()
	for {
		select {
		case <-r.Context().Done():
			return
		case <-tick.C:
		}
		st, ok := s.sup.JobState(id)
		if !ok {
			out.line(errorLine(coded(codeJobNotFound, notFound("job %q is no longer known", id))))
			return
		}
		switch st.Status {
		case "done":
			out.line(ImageResponse{Model: m.Name, Images: st.Images, Seed: seed})
			return
		case "failed", "cancelled":
			out.line(errorLine(withModel(jobError(st), m.Name)))
			return
		}
		// Transitions and elapsed time: inventing step numbers would be worse.
		line := StreamLine{Status: st.Status, Model: m.Name, Job: id, Step: st.Step, Total: st.Total}
		if line == last && time.Since(beat) < heartbeat {
			continue
		}
		last, beat = line, time.Now()
		line.Elapsed = time.Since(started).Round(time.Second).Seconds()
		out.line(line)
	}
}

// A real status code is still possible until the first line goes out, so an
// early failure is an ordinary HTTP error and a late one a terminal line.
func streamFail(w http.ResponseWriter, out *ndjson, err error) {
	if out.wrote {
		out.line(errorLine(err))
		return
	}
	fail(w, err)
}

func (s *Server) handleJob(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	st, ok := s.sup.JobState(id)
	if !ok {
		fail(w, coded(codeJobNotFound, notFound("job %q not found", id)))
		return
	}
	note := s.jobs.lookup(id)
	switch st.Status {
	case "done":
		writeJSON(w, http.StatusOK, ImageResponse{Model: note.model, Images: st.Images, Seed: note.seed})
	case "failed", "cancelled":
		// A finished-badly job answers with the error a blocking call would have
		// returned, so a client's failure path is the same either way.
		fail(w, withModel(jobError(st), note.model))
	default:
		writeJSON(w, http.StatusOK, StreamLine{
			Status: st.Status, Model: note.model, Job: id, Step: st.Step, Total: st.Total,
		})
	}
}

func (s *Server) handleJobCancel(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if !s.sup.CancelJob(id) {
		fail(w, coded(codeJobNotFound, notFound("job %q not found", id)))
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"status": "cancelled", "job": id})
}

func jobError(st supervisor.JobState) error {
	switch {
	case st.Err != nil:
		return generationErr(st.Err)
	case st.Status == "cancelled":
		return coded(codeCancelled, statusErr(http.StatusConflict, errors.New("job was cancelled")))
	default:
		return statusErr(http.StatusBadGateway, fmt.Errorf("engine job ended with status %q", st.Status))
	}
}

func errorLine(err error) errorBody {
	_, body := apiError(err)
	return body
}

// The supervisor's registry carries neither the model name nor the resolved
// seed, and a client reattaching to a job needs the same object as a blocking call.
type jobNotes struct {
	mu sync.Mutex
	m  map[string]jobNote
}

type jobNote struct {
	model string
	seed  int64
	at    time.Time
}

const maxJobNotes = 256

func newJobNotes() *jobNotes { return &jobNotes{m: map[string]jobNote{}} }

func (j *jobNotes) remember(id, model string, seed int64) {
	j.mu.Lock()
	defer j.mu.Unlock()
	if len(j.m) >= maxJobNotes {
		oldest, at := "", time.Now()
		for k, v := range j.m {
			if !v.at.After(at) {
				oldest, at = k, v.at
			}
		}
		delete(j.m, oldest)
	}
	j.m[id] = jobNote{model: model, seed: seed, at: time.Now()}
}

func (j *jobNotes) lookup(id string) jobNote {
	j.mu.Lock()
	defer j.mu.Unlock()
	return j.m[id]
}
