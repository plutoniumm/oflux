package supervisor

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"slices"
	"time"

	"oflux/internal/engineclient"
	"oflux/internal/types"
)

const (
	statusQueued    = "queued"
	statusRunning   = "running"
	statusDone      = "done"
	statusFailed    = "failed"
	statusCancelled = "cancelled"
)

const (
	pollInterval = 250 * time.Millisecond
	// Finished jobs hold their images, megabytes of base64 each, so the
	// registry is capped by count as well as age.
	jobRetention = 10 * time.Minute
	maxKeptJobs  = 32
)

type JobState struct {
	Status      string   // "queued" | "running" | "done" | "failed" | "cancelled"
	Images      []string // raw base64, set when Status=="done"
	Err         error
	Step, Total int
}

// Every field is guarded by Supervisor.jobsMu.
type job struct {
	id       string
	model    string
	status   string
	images   []string
	err      error
	step     int
	total    int
	engineID string
	client   *engineclient.Client
	cancel   context.CancelFunc
	canceled bool
	ended    time.Time
}

// SubmitJob returns as soon as the request has a place in the model's queue, so
// a caller is never held for the minutes an edit takes. The job deliberately
// outlives ctx — its values are kept, its cancellation is not — because the
// request that submitted it returns immediately. CancelJob is how one is stopped.
func (s *Supervisor) SubmitJob(ctx context.Context, m types.Manifest, req engineclient.ImgGenRequest, opts GenOpts) (id string, err error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	t, err := s.enqueue(m.Name)
	if err != nil {
		return "", err
	}
	jobCtx, cancel := context.WithCancel(context.WithoutCancel(ctx))
	j := &job{id: newJobID(), model: m.Name, status: statusQueued, cancel: cancel}

	s.jobsMu.Lock()
	s.sweepJobsLocked()
	s.jobs[j.id] = j
	s.jobsMu.Unlock()

	// Wrapping the caller's slot is what gives JobState step/total even when
	// the caller wanted no callback of its own.
	onStep := opts.OnStep
	opts.OnStep = func(step, total int) {
		s.jobsMu.Lock()
		j.step, j.total = step, total
		s.jobsMu.Unlock()
		if onStep != nil {
			onStep(step, total)
		}
	}

	go func() {
		defer cancel()
		// Not "err": that is the named result, and this outlives the call.
		imgs, runErr := s.run(jobCtx, m, req, opts, j, t)

		s.jobsMu.Lock()
		defer s.jobsMu.Unlock()
		j.ended = time.Now()
		switch {
		case j.canceled || errors.Is(runErr, context.Canceled):
			j.status = statusCancelled
			if j.err == nil {
				j.err = context.Canceled
			}
		case runErr != nil:
			j.status, j.err = statusFailed, runErr
		default:
			j.status, j.images = statusDone, imgs
		}
	}()
	return j.id, nil
}

func (s *Supervisor) JobState(id string) (JobState, bool) {
	s.jobsMu.Lock()
	defer s.jobsMu.Unlock()
	j, ok := s.jobs[id]
	if !ok {
		return JobState{}, false
	}
	return JobState{
		Status: j.status,
		Images: slices.Clone(j.images),
		Err:    j.err,
		Step:   j.step,
		Total:  j.total,
	}, true
}

// CancelJob reports whether there was a live job to stop. One still waiting in
// the queue is dropped from it and never reaches the engine at all.
func (s *Supervisor) CancelJob(id string) bool {
	s.jobsMu.Lock()
	j, ok := s.jobs[id]
	if !ok || isTerminalStatus(j.status) {
		s.jobsMu.Unlock()
		return false
	}
	j.canceled = true
	j.status = statusCancelled
	j.err = context.Canceled
	j.ended = time.Now()
	cancel, client, engineID := j.cancel, j.client, j.engineID
	s.jobsMu.Unlock()

	// Cancelling the context unwinds Wait, which cancels best-effort itself;
	// hitting the engine too covers the window before Wait starts.
	cancel()
	if client != nil && engineID != "" {
		go func() {
			ctx, stop := context.WithTimeout(context.Background(), 5*time.Second)
			defer stop()
			_ = client.Cancel(ctx, engineID)
		}()
	}
	return true
}

func (s *Supervisor) cancelAllJobs() {
	s.jobsMu.Lock()
	defer s.jobsMu.Unlock()
	for _, j := range s.jobs {
		if !isTerminalStatus(j.status) {
			j.canceled = true
			j.cancel()
		}
	}
}

// Live jobs are never swept.
func (s *Supervisor) sweepJobsLocked() {
	cutoff := time.Now().Add(-jobRetention)
	done := make([]*job, 0, len(s.jobs))
	for id, j := range s.jobs {
		switch {
		case !isTerminalStatus(j.status):
		case j.ended.Before(cutoff):
			delete(s.jobs, id)
		default:
			done = append(done, j)
		}
	}
	if len(done) <= maxKeptJobs {
		return
	}
	slices.SortFunc(done, func(a, b *job) int { return a.ended.Compare(b.ended) })
	for _, j := range done[:len(done)-maxKeptJobs] {
		delete(s.jobs, j.id)
	}
}

func isTerminalStatus(status string) bool {
	switch status {
	case statusDone, statusFailed, statusCancelled:
		return true
	}
	return false
}

func newJobID() string {
	var b [12]byte
	if _, err := rand.Read(b[:]); err != nil {
		return fmt.Sprintf("job-%d", time.Now().UnixNano())
	}
	return "job-" + hex.EncodeToString(b[:])
}
