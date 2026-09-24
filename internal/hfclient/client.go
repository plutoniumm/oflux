// Package hfclient is a minimal, dependency-free Hugging Face Hub client. It
// speaks only to the public Hub HTTP API and the resolve (download) endpoint,
// never shelling out to git or git-lfs. Construct a Client with New.
package hfclient

import (
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strconv"
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

const defaultBaseURL = "https://huggingface.co"

// apiTimeout bounds metadata calls only: a download of multi-gigabyte weights
// is governed solely by the caller's context.
const apiTimeout = 60 * time.Second

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

// Tree lists every file in repo at revision, recursively, following Link-header
// pagination. An empty revision means "main".
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

// treePage returns one page of entries plus the next page's URL ("" if last).
func (c *Client) treePage(ctx context.Context, url string) ([]treeEntry, string, error) {
	resp, err := c.get(ctx, url)
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

// Search returns Hub models matching query, most-downloaded first. limit <= 0
// leaves the Hub's own page size.
func (c *Client) Search(ctx context.Context, query string, limit int) ([]types.HFModel, error) {
	ctx, cancel := context.WithTimeout(ctx, apiTimeout)
	defer cancel()

	// expand[] REPLACES the default field set, so every field we read must be
	// listed — including downloads and likes, which the bare endpoint returns
	// but an expanded one does not. Without this the list endpoint omits
	// "gated" entirely and every repo decodes as ungated, which only shows up
	// as a failed download of a repo the user had to accept terms for.
	// ("pipeline_tag" is requested but the Hub does not return it for model
	// lists; the field stays empty rather than being faked.)
	q := url.Values{
		"search": {query}, "sort": {"downloads"}, "direction": {"-1"},
		"expand[]": {"downloads", "likes", "gated", "lastModified", "pipeline_tag"},
	}
	if limit > 0 {
		q.Set("limit", strconv.Itoa(limit))
	}
	resp, err := c.get(ctx, c.baseURL+"/api/models?"+q.Encode())
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	// "gated" is false or one of "auto"/"manual", so it cannot decode as a bool.
	var hits []struct {
		ID           string          `json:"id"`
		Downloads    int             `json:"downloads"`
		Likes        int             `json:"likes"`
		PipelineTag  string          `json:"pipeline_tag"`
		Gated        json.RawMessage `json:"gated"`
		LastModified string          `json:"lastModified"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&hits); err != nil {
		return nil, fmt.Errorf("hf: decode search: %w", err)
	}
	out := make([]types.HFModel, 0, len(hits))
	for _, h := range hits {
		g := string(h.Gated)
		out = append(out, types.HFModel{
			ID:          h.ID,
			Downloads:   h.Downloads,
			Likes:       h.Likes,
			PipelineTag: h.PipelineTag,
			Gated:       g != "" && g != "false" && g != "null",
			Updated:     h.LastModified,
		})
	}
	// The Hub honours sort=downloads, but that ranking is a wire contract here.
	slices.SortStableFunc(out, func(a, b types.HFModel) int { return cmp.Compare(b.Downloads, a.Downloads) })
	return out, nil
}

// ReadFile reads repo@revision/path, capped to maxBytes when that is positive.
func (c *Client) ReadFile(ctx context.Context, repo, revision, path string, maxBytes int64) ([]byte, error) {
	ctx, cancel := context.WithTimeout(ctx, apiTimeout)
	defer cancel()

	resp, err := c.get(ctx, c.resolveURL(repo, revision, path))
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

// Download streams repo@revision/path to destPath+".part", hashing on the fly,
// and renames it into place only once a non-empty expectSHA256 matches. It has
// no client timeout: cancel via ctx. Returns the lowercase hex sha256.
func (c *Client) Download(ctx context.Context, repo, revision, path, destPath, expectSHA256 string) (sum string, err error) {
	resp, err := c.get(ctx, c.resolveURL(repo, revision, path))
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
	// Past this point every failure must take the partial file with it: bytes
	// left at a name a caller trusts are worse than no download at all.
	defer func() {
		if err != nil {
			os.Remove(partPath)
		}
	}()

	h := sha256.New()
	if _, err = io.Copy(f, io.TeeReader(resp.Body, h)); err == nil {
		err = f.Sync()
	}
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return "", err
	}

	sum = hex.EncodeToString(h.Sum(nil))
	if expectSHA256 != "" && !strings.EqualFold(sum, expectSHA256) {
		err = fmt.Errorf("hf: sha256 mismatch for %s: got %s want %s", path, sum, strings.ToLower(expectSHA256))
		return "", err
	}
	if err = os.Rename(partPath, destPath); err != nil {
		return "", err
	}
	return sum, nil
}

func (c *Client) resolveURL(repo, revision, path string) string {
	return fmt.Sprintf("%s/%s/resolve/%s/%s", c.baseURL, repo, normRevision(revision), path)
}

// get applies the token and maps non-2xx to errors. On success the caller owns
// resp.Body; on error it is read for a snippet and closed here.
func (c *Client) get(ctx context.Context, url string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
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

func normRevision(revision string) string {
	if revision == "" {
		return "main"
	}
	return revision
}

// parseNextLink reads the rel="next" URL out of an RFC 5988 Link header.
func parseNextLink(header string) string {
	for _, part := range strings.Split(header, ",") {
		target, params, ok := strings.Cut(strings.TrimSpace(part), ">")
		if !ok || !strings.HasPrefix(target, "<") {
			continue
		}
		for _, p := range strings.Split(params, ";") {
			rel, ok := strings.CutPrefix(strings.TrimSpace(p), "rel=")
			if ok && strings.Trim(rel, `"`) == "next" {
				return strings.TrimPrefix(target, "<")
			}
		}
	}
	return ""
}
