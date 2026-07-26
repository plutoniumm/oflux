// Package engineclient is a small HTTP client for the sd-server native async
// image-generation API from leejet/stable-diffusion.cpp.
//
// The API is asynchronous: POST /sdcpp/v1/img_gen returns 202 with a job id,
// and callers poll GET /sdcpp/v1/jobs/{id} until the job reaches a terminal
// status. The exact endpoint paths and payload shapes mirror api.md:
//
//	POST /sdcpp/v1/img_gen            -> 202 {id,kind,status,created,poll_url}
//	GET  /sdcpp/v1/jobs/{id}          -> job incl. status + result
//	POST /sdcpp/v1/jobs/{id}/cancel   -> cancel an accepted job
//	GET  /sdcpp/v1/capabilities       -> capability metadata (used as a health probe)
//
// A completed img_gen job carries base64 images at result.images[].b64_json.
package engineclient

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"
)

// ErrQueueFull is returned when the engine rejects a submission with HTTP 429
// because its job queue is full.
var ErrQueueFull = errors.New("engine: queue full")

// Client talks to a single sd-server instance over HTTP.
type Client struct {
	baseURL string
	http    *http.Client
}

// New returns a Client targeting baseURL (e.g. "http://127.0.0.1:8080"). Any
// trailing slash on baseURL is trimmed so path joins stay well-formed.
func New(baseURL string) *Client {
	for len(baseURL) > 0 && baseURL[len(baseURL)-1] == '/' {
		baseURL = baseURL[:len(baseURL)-1]
	}
	return &Client{baseURL: baseURL, http: &http.Client{}}
}

// SetHTTPClient overrides the HTTP client used for all requests. A nil client
// is ignored.
func (c *Client) SetHTTPClient(h *http.Client) {
	if h != nil {
		c.http = h
	}
}

func (c *Client) do(ctx context.Context, method, path string, body []byte) (*http.Response, error) {
	var rdr io.Reader
	if body != nil {
		rdr = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, rdr)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	return c.http.Do(req)
}

func drain(resp *http.Response) {
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<20))
	_ = resp.Body.Close()
}

// Capabilities probes GET /sdcpp/v1/capabilities. It returns nil when the
// server answers with any 2xx status, making it a convenient readiness check.
func (c *Client) Capabilities(ctx context.Context) error {
	resp, err := c.do(ctx, http.MethodGet, "/sdcpp/v1/capabilities", nil)
	if err != nil {
		return err
	}
	defer drain(resp)
	if resp.StatusCode/100 != 2 {
		return fmt.Errorf("engine: capabilities status %d", resp.StatusCode)
	}
	return nil
}

// Submit posts an image-generation request. It expects HTTP 202 and returns the
// accepted Job (id + queued status + poll_url). A 429 maps to ErrQueueFull.
func (c *Client) Submit(ctx context.Context, req ImgGenRequest) (Job, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return Job{}, fmt.Errorf("engine: encode img_gen: %w", err)
	}
	resp, err := c.do(ctx, http.MethodPost, "/sdcpp/v1/img_gen", body)
	if err != nil {
		return Job{}, err
	}
	defer drain(resp)
	if resp.StatusCode == http.StatusTooManyRequests {
		return Job{}, ErrQueueFull
	}
	if resp.StatusCode != http.StatusAccepted {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return Job{}, fmt.Errorf("engine: img_gen status %d: %s", resp.StatusCode, bytes.TrimSpace(snippet))
	}
	var job Job
	if err := json.NewDecoder(resp.Body).Decode(&job); err != nil {
		return Job{}, fmt.Errorf("engine: decode img_gen response: %w", err)
	}
	return job, nil
}

// Poll fetches the current state of job id from GET /sdcpp/v1/jobs/{id}.
func (c *Client) Poll(ctx context.Context, id string) (Job, error) {
	resp, err := c.do(ctx, http.MethodGet, "/sdcpp/v1/jobs/"+id, nil)
	if err != nil {
		return Job{}, err
	}
	defer drain(resp)
	if resp.StatusCode == http.StatusTooManyRequests {
		return Job{}, ErrQueueFull
	}
	if resp.StatusCode/100 != 2 {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return Job{}, fmt.Errorf("engine: poll status %d: %s", resp.StatusCode, bytes.TrimSpace(snippet))
	}
	var job Job
	if err := json.NewDecoder(resp.Body).Decode(&job); err != nil {
		return Job{}, fmt.Errorf("engine: decode job: %w", err)
	}
	return job, nil
}

// Cancel requests cancellation of job id. Any 2xx response is treated as
// success.
func (c *Client) Cancel(ctx context.Context, id string) error {
	resp, err := c.do(ctx, http.MethodPost, "/sdcpp/v1/jobs/"+id+"/cancel", nil)
	if err != nil {
		return err
	}
	defer drain(resp)
	if resp.StatusCode/100 == 2 {
		return nil
	}
	return fmt.Errorf("engine: cancel status %d", resp.StatusCode)
}

// bestEffortCancel cancels a job using a fresh, bounded context so it still runs
// even when the caller's context has already been cancelled.
func (c *Client) bestEffortCancel(id string) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = c.Cancel(ctx, id)
}

// Wait polls job id every poll interval until it reaches a terminal status or
// ctx is done. If ctx is cancelled first, Wait makes a best-effort Cancel call
// and returns ctx.Err().
func (c *Client) Wait(ctx context.Context, id string, poll time.Duration) (Job, error) {
	if poll <= 0 {
		poll = 250 * time.Millisecond
	}
	timer := time.NewTimer(poll)
	defer timer.Stop()
	for {
		if err := ctx.Err(); err != nil {
			c.bestEffortCancel(id)
			return Job{}, err
		}
		job, err := c.Poll(ctx, id)
		if err != nil {
			if ctx.Err() != nil {
				c.bestEffortCancel(id)
				return Job{}, ctx.Err()
			}
			return Job{}, err
		}
		if isTerminal(job.Status) {
			return job, nil
		}
		if !timer.Stop() {
			select {
			case <-timer.C:
			default:
			}
		}
		timer.Reset(poll)
		select {
		case <-ctx.Done():
			c.bestEffortCancel(id)
			return Job{}, ctx.Err()
		case <-timer.C:
		}
	}
}
