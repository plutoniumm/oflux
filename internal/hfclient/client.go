// Package hfclient is a minimal, dependency-free Hugging Face Hub client used by
// oflux to inspect and download model repositories.
//
// It speaks only to the public Hub HTTP API and the resolve (download) endpoint;
// it never shells out to git or git-lfs. All methods honour the caller's
// context. The zero value is not usable; construct a Client with New.
package hfclient

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"oflux/internal/types"
)

// Exported sentinel errors. Callers should test with errors.Is.
var (
	// ErrNotFound is returned for HTTP 404 responses.
	ErrNotFound = errors.New("hf: not found")
	// ErrUnauthorized is returned for HTTP 401/403 responses (gated or private repo).
	ErrUnauthorized = errors.New("hf: unauthorized (gated or private; HF token required)")
	// ErrRateLimited is returned for HTTP 429 responses.
	ErrRateLimited = errors.New("hf: rate limited")
)

// defaultBaseURL is the public Hugging Face Hub origin.
const defaultBaseURL = "https://huggingface.co"

// apiTimeout bounds metadata/API calls (Tree, ReadFile). Downloads are not
// bounded by a client-level timeout because model weights can be large; they are
// governed solely by the caller's context.
const apiTimeout = 60 * time.Second

// Client talks to the Hugging Face Hub over HTTP.
type Client struct {
	baseURL    string
	token      string
	httpClient *http.Client
}

// New returns a Client. A token of "" means anonymous access to public repos.
func New(token string) *Client {
	return &Client{
		baseURL: defaultBaseURL,
		token:   token,
		// No client-level Timeout: it would abort large Downloads mid-stream.
		// API methods apply their own deadline via context instead.
		httpClient: &http.Client{},
	}
}

// SetBaseURL overrides the Hub origin (default https://huggingface.co). Any
// trailing slash is trimmed. Intended for tests.
func (c *Client) SetBaseURL(u string) {
	c.baseURL = strings.TrimRight(u, "/")
}

// SetHTTPClient overrides the underlying *http.Client. Intended for tests.
func (c *Client) SetHTTPClient(h *http.Client) {
	c.httpClient = h
}

// treeEntry is one raw entry from the Hub tree listing.
type treeEntry struct {
	Type string `json:"type"` // "file" or "directory"
	OID  string `json:"oid"`  // git blob sha1
	Size int64  `json:"size"`
	Path string `json:"path"`
	LFS  *struct {
		OID         string `json:"oid"`  // sha256 of the real content
		Size        int64  `json:"size"` // true content size
		PointerSize int    `json:"pointerSize"`
	} `json:"lfs"`
}

// Tree lists every file in repo at revision (recursively). An empty revision
// defaults to "main". Directory entries are skipped. Pagination via the Link
// response header is followed transparently.
func (c *Client) Tree(ctx context.Context, repo, revision string) ([]types.HFFile, error) {
	revision = normRevision(revision)
	ctx, cancel := context.WithTimeout(ctx, apiTimeout)
	defer cancel()

	url := fmt.Sprintf("%s/api/models/%s/tree/%s?recursive=true", c.baseURL, repo, revision)

	var out []types.HFFile
	for url != "" {
		entries, next, err := c.treePage(ctx, url)
		if err != nil {
			return nil, err
		}
		for _, e := range entries {
			if e.Type == "directory" {
				continue
			}
			f := types.HFFile{Path: e.Path, OID: e.OID, Size: e.Size}
			if e.LFS != nil {
				f.IsLFS = true
				f.LFSOID = e.LFS.OID
				f.Size = e.LFS.Size
			}
			out = append(out, f)
		}
		url = next
	}
	return out, nil
}

// treePage fetches and decodes one page of a tree listing, returning the decoded
// entries and the absolute URL of the next page ("" when there is none).
func (c *Client) treePage(ctx context.Context, url string) ([]treeEntry, string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, "", err
	}
	resp, err := c.do(req)
	if err != nil {
		return nil, "", err
	}
	defer resp.Body.Close()

	var entries []treeEntry
	if err := json.NewDecoder(resp.Body).Decode(&entries); err != nil {
		return nil, "", fmt.Errorf("hf: decode tree page: %w", err)
	}
	return entries, parseNextLink(resp.Header.Get("Link")), nil
}

// ReadFile fetches the whole content of a single file at repo@revision/path. An
// empty revision defaults to "main". When maxBytes > 0 the read is capped to
// that many bytes; maxBytes <= 0 reads the full body.
func (c *Client) ReadFile(ctx context.Context, repo, revision, path string, maxBytes int64) ([]byte, error) {
	revision = normRevision(revision)
	ctx, cancel := context.WithTimeout(ctx, apiTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.resolveURL(repo, revision, path), nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var r io.Reader = resp.Body
	if maxBytes > 0 {
		r = io.LimitReader(resp.Body, maxBytes)
	}
	return io.ReadAll(r)
}

// Download streams repo@revision/path to destPath and returns the lowercase hex
// sha256 of the content. An empty revision defaults to "main". Parent
// directories are created as needed. The body is streamed to destPath+".part",
// hashed on the fly, fsync'd, then atomically renamed into place. When
// expectSHA256 is non-empty it is compared case-insensitively against the
// computed digest; on mismatch the partial file is removed and an error
// returned. Download is not subject to a client timeout; cancel via ctx.
func (c *Client) Download(ctx context.Context, repo, revision, path, destPath, expectSHA256 string) (string, error) {
	revision = normRevision(revision)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.resolveURL(repo, revision, path), nil)
	if err != nil {
		return "", err
	}
	resp, err := c.do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if err := os.MkdirAll(filepath.Dir(destPath), 0o755); err != nil {
		return "", err
	}
	partPath := destPath + ".part"
	f, err := os.Create(partPath)
	if err != nil {
		return "", err
	}

	h := sha256.New()
	if _, err := io.Copy(f, io.TeeReader(resp.Body, h)); err != nil {
		f.Close()
		os.Remove(partPath)
		return "", err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		os.Remove(partPath)
		return "", err
	}
	if err := f.Close(); err != nil {
		os.Remove(partPath)
		return "", err
	}

	sum := hex.EncodeToString(h.Sum(nil))
	if expectSHA256 != "" && !strings.EqualFold(sum, expectSHA256) {
		os.Remove(partPath)
		return "", fmt.Errorf("hf: sha256 mismatch for %s: got %s want %s", path, sum, strings.ToLower(expectSHA256))
	}
	if err := os.Rename(partPath, destPath); err != nil {
		os.Remove(partPath)
		return "", err
	}
	return sum, nil
}

// resolveURL builds the download/resolve URL for a file.
func (c *Client) resolveURL(repo, revision, path string) string {
	return fmt.Sprintf("%s/%s/resolve/%s/%s", c.baseURL, repo, revision, path)
}

// do sends req with auth applied and maps non-2xx statuses to errors. On success
// the caller owns resp.Body and must close it; on error the body is drained and
// closed here.
func (c *Client) do(req *http.Request) (*http.Response, error) {
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode/100 != 2 {
		defer resp.Body.Close()
		return nil, statusError(resp)
	}
	return resp, nil
}

// statusError maps an HTTP status to a sentinel error, or a formatted error with
// a short body snippet for unexpected statuses.
func statusError(resp *http.Response) error {
	switch resp.StatusCode {
	case http.StatusNotFound:
		return ErrNotFound
	case http.StatusUnauthorized, http.StatusForbidden:
		return ErrUnauthorized
	case http.StatusTooManyRequests:
		return ErrRateLimited
	}
	snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
	msg := strings.TrimSpace(string(snippet))
	if msg == "" {
		return fmt.Errorf("hf: unexpected status %s", resp.Status)
	}
	return fmt.Errorf("hf: unexpected status %s: %s", resp.Status, msg)
}

// normRevision defaults an empty revision to "main".
func normRevision(revision string) string {
	if revision == "" {
		return "main"
	}
	return revision
}

// parseNextLink extracts the rel="next" URL from an RFC 5988 Link header,
// returning "" when there is no next link.
func parseNextLink(header string) string {
	if header == "" {
		return ""
	}
	for _, part := range strings.Split(header, ",") {
		part = strings.TrimSpace(part)
		lt := strings.IndexByte(part, '<')
		gt := strings.IndexByte(part, '>')
		if lt != 0 || gt < 0 {
			continue
		}
		target := part[lt+1 : gt]
		params := part[gt+1:]
		for _, p := range strings.Split(params, ";") {
			p = strings.TrimSpace(p)
			if !strings.HasPrefix(p, "rel=") {
				continue
			}
			rel := strings.Trim(strings.TrimPrefix(p, "rel="), `"`)
			if rel == "next" {
				return target
			}
		}
	}
	return ""
}
