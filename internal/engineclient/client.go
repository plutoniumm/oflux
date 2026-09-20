// Package engineclient is an HTTP client for the async sd-server image API from
// leejet/stable-diffusion.cpp. Paths and payloads mirror api.md:
//
//	POST /sdcpp/v1/img_gen            -> 202 {id,kind,status,created,poll_url}
//	GET  /sdcpp/v1/jobs/{id}          -> job incl. status + result
//	POST /sdcpp/v1/jobs/{id}/cancel   -> cancel an accepted job
//	GET  /sdcpp/v1/capabilities       -> capability metadata (used as a health probe)
package engineclient

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"time"
)

// ErrQueueFull is the engine's own 429: its job queue is full.
var ErrQueueFull = errors.New("engine: queue full")

// anyStatus2xx accepts any success, for endpoints the engine does not pin down.
const anyStatus2xx = 0

// None of these bound a generation: every call is short, so the minutes a
// generation takes are spent between requests. They exist for a wedged engine,
// which otherwise hangs a call until the OS gives up on the socket.
const (
	requestTimeout        = 2 * time.Minute
	responseHeaderTimeout = 60 * time.Second
	dialTimeout           = 5 * time.Second
)

// maxPollFailures is how many *consecutive* failed polls Wait absorbs. One
// dropped poll must not throw away minutes of GPU time.
const maxPollFailures = 3

type Client struct {
	baseURL string
	http    *http.Client
}

func New(baseURL string) *Client {
	return &Client{baseURL: strings.TrimRight(baseURL, "/"), http: newHTTPClient()}
}

func newHTTPClient() *http.Client {
	return &http.Client{
		Timeout: requestTimeout,
		Transport: &http.Transport{
			DialContext:           (&net.Dialer{Timeout: dialTimeout}).DialContext,
			ResponseHeaderTimeout: responseHeaderTimeout,
			ExpectContinueTimeout: time.Second,
			// The poll loop hits one host every 250ms for minutes.
			MaxIdleConns:        8,
			MaxIdleConnsPerHost: 4,
			IdleConnTimeout:     90 * time.Second,
			// cpp-httplib is HTTP/1.1 only, and gzip on multi-MB base64 over
			// loopback costs more than the bytes it saves.
			ForceAttemptHTTP2:  false,
			DisableCompression: true,
		},
	}
}

// SetHTTPClient ignores a nil client.
func (c *Client) SetHTTPClient(h *http.Client) {
	if h != nil {
		c.http = h
	}
}

// want is the exact status the endpoint promises, or anyStatus2xx. On success
// the caller owns the body and must drain it; on failure do already has.
func (c *Client) do(ctx context.Context, method, path, op string, want int, body any) (*http.Response, error) {
	var payload io.Reader
	if body != nil {
		enc, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("engine: encode %s: %w", op, err)
		}
		payload = bytes.NewReader(enc)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, payload)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode == http.StatusTooManyRequests {
		drain(resp)
		return nil, ErrQueueFull
	}
	if resp.StatusCode != want && !(want == anyStatus2xx && resp.StatusCode/100 == 2) {
		// Carry the engine's own message forward: its failures are otherwise
		// opaque, and the body is where it explains itself.
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		drain(resp)
		return nil, fmt.Errorf("engine: %s status %d: %s", op, resp.StatusCode, bytes.TrimSpace(snippet))
	}
	return resp, nil
}

func fetch[T any](ctx context.Context, c *Client, method, path, op string, want int, body any) (T, error) {
	var out T
	resp, err := c.do(ctx, method, path, op, want, body)
	if err != nil {
		return out, err
	}
	defer drain(resp)
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return out, fmt.Errorf("engine: decode %s response: %w", op, err)
	}
	return out, nil
}

func (c *Client) discard(ctx context.Context, method, path, op string) error {
	resp, err := c.do(ctx, method, path, op, anyStatus2xx, nil)
	if err != nil {
		return err
	}
	drain(resp)
	return nil
}

func drain(resp *http.Response) {
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<20))
	_ = resp.Body.Close()
}

// Capabilities answers 2xx whenever the engine is up: it doubles as the probe.
func (c *Client) Capabilities(ctx context.Context) error {
	return c.discard(ctx, http.MethodGet, "/sdcpp/v1/capabilities", "capabilities")
}

func (c *Client) Cancel(ctx context.Context, id string) error {
	return c.discard(ctx, http.MethodPost, "/sdcpp/v1/jobs/"+id+"/cancel", "cancel")
}

// Submit expects HTTP 202; a 429 maps to ErrQueueFull.
func (c *Client) Submit(ctx context.Context, req ImgGenRequest) (Job, error) {
	return fetch[Job](ctx, c, http.MethodPost, "/sdcpp/v1/img_gen", "img_gen", http.StatusAccepted, req)
}

func (c *Client) Poll(ctx context.Context, id string) (Job, error) {
	return fetch[Job](ctx, c, http.MethodGet, "/sdcpp/v1/jobs/"+id, "poll", anyStatus2xx, nil)
}

// bestEffortCancel uses a fresh context so it still runs when the caller's is done.
func (c *Client) bestEffortCancel(id string) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = c.Cancel(ctx, id)
}

// Wait polls until job id is terminal or ctx is done, cancelling the job
// best-effort if ctx goes first. A single failed poll is retried rather than
// fatal — see maxPollFailures.
func (c *Client) Wait(ctx context.Context, id string, poll time.Duration) (Job, error) {
	if poll <= 0 {
		poll = 250 * time.Millisecond
	}
	// One timer, reset per iteration: since Go 1.23 timer channels are
	// unbuffered, so Reset can never deliver a stale tick and needs no draining.
	timer := time.NewTimer(poll)
	defer timer.Stop()
	fails := 0
	for {
		if err := ctx.Err(); err != nil {
			c.bestEffortCancel(id)
			return Job{}, err
		}
		job, err := c.Poll(ctx, id)
		switch {
		case err == nil:
			fails = 0
			if isTerminal(job.Status) {
				return job, nil
			}
		case ctx.Err() != nil:
			c.bestEffortCancel(id)
			return Job{}, ctx.Err()
		default:
			if fails++; fails >= maxPollFailures {
				return Job{}, err
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
