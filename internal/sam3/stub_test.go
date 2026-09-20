//go:build !sam3

package sam3

import (
	"context"
	"errors"
	"image/color"
	"strings"
	"testing"
)

func TestStubIsUnavailable(t *testing.T) {
	if Compiled() {
		t.Fatal("Compiled() is true in a build without the sam3 tag")
	}
	if Available() {
		t.Fatal("Available() is true in a build without the sam3 tag")
	}
	if !strings.Contains(Status(), "-tags sam3") {
		t.Fatalf("Status() = %q, want it to name the build tag", Status())
	}
}

func TestStubSegmentAlwaysReportsUnavailable(t *testing.T) {
	good := solidPNG(t, 32, 32, color.White)
	tests := []struct {
		name string
		req  Request
	}{
		{name: "text prompt", req: Request{Image: good, Prompt: "a cat"}},
		{name: "point prompt", req: Request{Image: good, Points: []Point{{X: 4, Y: 4}}}},
		{name: "box prompt", req: Request{Image: good, Boxes: []Box{{X1: 10, Y1: 10}}}},
		{name: "separate masks", req: Request{Image: good, Prompt: "a cat", Separate: true}},
		{name: "empty request", req: Request{}},
		{name: "garbage image", req: Request{Image: []byte("not an image"), Prompt: "a cat"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			masks, err := Segment(t.Context(), tt.req)
			if !errors.Is(err, ErrUnavailable) {
				t.Fatalf("err = %v, want ErrUnavailable", err)
			}
			if masks != nil {
				t.Fatalf("masks = %v, want nil", masks)
			}
		})
	}
}

// The stub must not download weights it has no library to load — least of all
// during `go test`, which is offline by contract.
func TestStubEnsureModelTouchesNoNetwork(t *testing.T) {
	hub, srv := newFakeHub(t, "weights")
	useFakeHub(t, srv)

	path, err := EnsureModel(t.Context(), func(string) {
		t.Fatal("EnsureModel reported progress in a stub build")
	})
	if !errors.Is(err, ErrUnavailable) {
		t.Fatalf("err = %v, want ErrUnavailable", err)
	}
	if path != "" {
		t.Fatalf("path = %q, want empty", path)
	}
	if n := hub.trees.Load() + hub.downloads.Load(); n != 0 {
		t.Fatalf("%d requests reached the Hub from a stub build", n)
	}
}

func TestStubSegmentIgnoresCancellation(t *testing.T) {
	// Unavailability outranks the context: the caller gets the actionable
	// answer rather than a confusing "context canceled".
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := Segment(ctx, Request{Prompt: "a cat"}); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("err = %v, want ErrUnavailable", err)
	}
}
