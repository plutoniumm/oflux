package hfclient

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// newTestClient wires a Client to point at the given test server.
func newTestClient(token, baseURL string) *Client {
	c := New(token)
	c.SetBaseURL(baseURL)
	return c
}

func TestTree(t *testing.T) {
	// (a) an LFS file, (b) a small git file, (c) a directory entry.
	const body = `[
		{"type":"file","oid":"gitsha_small","size":1234,"path":"config.json"},
		{"type":"directory","oid":"dirsha","size":0,"path":"subdir"},
		{"type":"file","oid":"gitsha_ptr","size":135,"path":"model.safetensors",
			"lfs":{"oid":"abc123sha256","size":9999999,"pointerSize":135}}
	]`

	var gotPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		if r.URL.Query().Get("recursive") != "true" {
			t.Errorf("expected recursive=true, got query %q", r.URL.RawQuery)
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, body)
	}))
	defer srv.Close()

	c := newTestClient("", srv.URL)
	files, err := c.Tree(context.Background(), "org/model", "")
	if err != nil {
		t.Fatalf("Tree: %v", err)
	}

	// revision "" must default to "main".
	if want := "/api/models/org/model/tree/main"; gotPath != want {
		t.Errorf("request path = %q, want %q", gotPath, want)
	}

	if len(files) != 2 {
		t.Fatalf("got %d files, want 2 (directory must be skipped): %+v", len(files), files)
	}

	small := files[0]
	if small.Path != "config.json" || small.IsLFS || small.OID != "gitsha_small" || small.Size != 1234 {
		t.Errorf("small file mapped wrong: %+v", small)
	}
	if small.LFSOID != "" {
		t.Errorf("small file should have empty LFSOID, got %q", small.LFSOID)
	}
	if got := small.ContentHash(); got != "gitsha_small" {
		t.Errorf("small ContentHash = %q, want git oid", got)
	}

	lfs := files[1]
	if lfs.Path != "model.safetensors" || !lfs.IsLFS {
		t.Errorf("lfs file mapped wrong: %+v", lfs)
	}
	if lfs.LFSOID != "abc123sha256" {
		t.Errorf("lfs LFSOID = %q, want lfs.oid", lfs.LFSOID)
	}
	if lfs.Size != 9999999 {
		t.Errorf("lfs Size = %d, want lfs.size (true content size)", lfs.Size)
	}
	if lfs.OID != "gitsha_ptr" {
		t.Errorf("lfs OID = %q, want git pointer oid", lfs.OID)
	}
	if got := lfs.ContentHash(); got != "abc123sha256" {
		t.Errorf("lfs ContentHash = %q, want lfs sha256", got)
	}
}

func TestTreePagination(t *testing.T) {
	page1 := `[{"type":"file","oid":"o1","size":1,"path":"a.txt"}]`
	page2 := `[{"type":"file","oid":"o2","size":2,"path":"b.txt"}]`

	var pagesServed int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Query().Get("cursor") == "" {
			pagesServed++
			// Point rel="next" at an absolute URL back to this server.
			next := "http://" + r.Host + r.URL.Path + "?recursive=true&cursor=PAGE2"
			w.Header().Set("Link", fmt.Sprintf(`<%s>; rel="next"`, next))
			fmt.Fprint(w, page1)
			return
		}
		pagesServed++
		// No Link header => last page.
		fmt.Fprint(w, page2)
	}))
	defer srv.Close()

	c := newTestClient("", srv.URL)
	files, err := c.Tree(context.Background(), "org/model", "main")
	if err != nil {
		t.Fatalf("Tree: %v", err)
	}
	if pagesServed != 2 {
		t.Errorf("served %d pages, want 2", pagesServed)
	}
	if len(files) != 2 {
		t.Fatalf("got %d files across pages, want 2: %+v", len(files), files)
	}
	if files[0].Path != "a.txt" || files[1].Path != "b.txt" {
		t.Errorf("pages not accumulated in order: %+v", files)
	}
}

func TestReadFile(t *testing.T) {
	const content = "hello model_index"

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "/notfound.json"):
			w.WriteHeader(http.StatusNotFound)
		case strings.HasSuffix(r.URL.Path, "/gated.json"):
			w.WriteHeader(http.StatusUnauthorized)
		default:
			fmt.Fprint(w, content)
		}
	}))
	defer srv.Close()

	c := newTestClient("", srv.URL)
	ctx := context.Background()

	t.Run("full read", func(t *testing.T) {
		b, err := c.ReadFile(ctx, "org/model", "main", "model_index.json", 0)
		if err != nil {
			t.Fatalf("ReadFile: %v", err)
		}
		if string(b) != content {
			t.Errorf("got %q, want %q", b, content)
		}
	})

	t.Run("maxBytes caps", func(t *testing.T) {
		b, err := c.ReadFile(ctx, "org/model", "main", "model_index.json", 5)
		if err != nil {
			t.Fatalf("ReadFile: %v", err)
		}
		if string(b) != content[:5] {
			t.Errorf("capped read = %q, want %q", b, content[:5])
		}
	})

	t.Run("404 -> ErrNotFound", func(t *testing.T) {
		_, err := c.ReadFile(ctx, "org/model", "main", "notfound.json", 0)
		if !errors.Is(err, ErrNotFound) {
			t.Errorf("err = %v, want ErrNotFound", err)
		}
	})

	t.Run("401 -> ErrUnauthorized", func(t *testing.T) {
		_, err := c.ReadFile(ctx, "org/model", "main", "gated.json", 0)
		if !errors.Is(err, ErrUnauthorized) {
			t.Errorf("err = %v, want ErrUnauthorized", err)
		}
	})
}

func TestReadFileStatusMapping(t *testing.T) {
	tests := []struct {
		name    string
		status  int
		wantErr error
	}{
		{"403 forbidden", http.StatusForbidden, ErrUnauthorized},
		{"429 rate limited", http.StatusTooManyRequests, ErrRateLimited},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.status)
			}))
			defer srv.Close()
			c := newTestClient("", srv.URL)
			_, err := c.ReadFile(context.Background(), "org/model", "main", "f", 0)
			if !errors.Is(err, tt.wantErr) {
				t.Errorf("status %d: err = %v, want %v", tt.status, err, tt.wantErr)
			}
		})
	}
}

func TestDownload(t *testing.T) {
	const content = "the actual weights bytes"
	sum := sha256.Sum256([]byte(content))
	wantHex := hex.EncodeToString(sum[:])

	// Track the Authorization header seen by the server for the auth assertions.
	var lastAuth string
	makeServer := func() *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			lastAuth = r.Header.Get("Authorization")
			if want := "/org/model/resolve/main/weights.bin"; r.URL.Path != want {
				t.Errorf("download path = %q, want %q", r.URL.Path, want)
			}
			fmt.Fprint(w, content)
		}))
	}

	t.Run("writes file and returns sha256", func(t *testing.T) {
		srv := makeServer()
		defer srv.Close()
		c := newTestClient("", srv.URL)
		dest := filepath.Join(t.TempDir(), "nested", "weights.bin")

		got, err := c.Download(context.Background(), "org/model", "main", "weights.bin", dest, "")
		if err != nil {
			t.Fatalf("Download: %v", err)
		}
		if got != wantHex {
			t.Errorf("sha256 = %q, want %q", got, wantHex)
		}
		data, err := os.ReadFile(dest)
		if err != nil {
			t.Fatalf("read dest: %v", err)
		}
		if string(data) != content {
			t.Errorf("dest content = %q, want %q", data, content)
		}
		// No leftover .part file.
		if _, err := os.Stat(dest + ".part"); !os.IsNotExist(err) {
			t.Errorf(".part file should not exist after success")
		}
	})

	t.Run("expectSHA256 match passes", func(t *testing.T) {
		srv := makeServer()
		defer srv.Close()
		c := newTestClient("", srv.URL)
		dest := filepath.Join(t.TempDir(), "weights.bin")
		// Uppercase expected value to exercise case-insensitive compare.
		got, err := c.Download(context.Background(), "org/model", "main", "weights.bin", dest, strings.ToUpper(wantHex))
		if err != nil {
			t.Fatalf("Download with matching expect: %v", err)
		}
		if got != wantHex {
			t.Errorf("sha256 = %q, want %q (lowercase)", got, wantHex)
		}
	})

	t.Run("expectSHA256 mismatch errors and leaves no partial", func(t *testing.T) {
		srv := makeServer()
		defer srv.Close()
		c := newTestClient("", srv.URL)
		dest := filepath.Join(t.TempDir(), "weights.bin")
		_, err := c.Download(context.Background(), "org/model", "main", "weights.bin", dest,
			"0000000000000000000000000000000000000000000000000000000000000000")
		if err == nil {
			t.Fatal("expected mismatch error, got nil")
		}
		if _, err := os.Stat(dest); !os.IsNotExist(err) {
			t.Errorf("dest file should not exist after mismatch")
		}
		if _, err := os.Stat(dest + ".part"); !os.IsNotExist(err) {
			t.Errorf(".part file should not exist after mismatch")
		}
	})

	t.Run("Authorization header present with token", func(t *testing.T) {
		srv := makeServer()
		defer srv.Close()
		c := newTestClient("tok", srv.URL)
		dest := filepath.Join(t.TempDir(), "weights.bin")
		if _, err := c.Download(context.Background(), "org/model", "main", "weights.bin", dest, ""); err != nil {
			t.Fatalf("Download: %v", err)
		}
		if lastAuth != "Bearer tok" {
			t.Errorf("Authorization = %q, want %q", lastAuth, "Bearer tok")
		}
	})

	t.Run("Authorization header absent without token", func(t *testing.T) {
		srv := makeServer()
		defer srv.Close()
		c := newTestClient("", srv.URL)
		dest := filepath.Join(t.TempDir(), "weights.bin")
		if _, err := c.Download(context.Background(), "org/model", "main", "weights.bin", dest, ""); err != nil {
			t.Fatalf("Download: %v", err)
		}
		if lastAuth != "" {
			t.Errorf("Authorization = %q, want empty (anonymous)", lastAuth)
		}
	})
}

func TestParseNextLink(t *testing.T) {
	tests := []struct {
		name   string
		header string
		want   string
	}{
		{"empty", "", ""},
		{"next only", `<https://example.com/p2>; rel="next"`, "https://example.com/p2"},
		{
			"next among many",
			`<https://example.com/p1>; rel="prev", <https://example.com/p3>; rel="next"`,
			"https://example.com/p3",
		},
		{"no next rel", `<https://example.com/p1>; rel="prev"`, ""},
		{"unquoted rel", `<https://example.com/p2>; rel=next`, "https://example.com/p2"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := parseNextLink(tt.header); got != tt.want {
				t.Errorf("parseNextLink(%q) = %q, want %q", tt.header, got, tt.want)
			}
		})
	}
}
