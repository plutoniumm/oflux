//go:build !sam3

package sam3

import "context"

func Compiled() bool { return false }

func Available() bool { return false }

func Status() string {
	return "sam3: not built in — rebuild with `go build -tags sam3` (see internal/sam3/README.md)"
}

// Segment reports ErrUnavailable without validating the request: nothing here
// can segment, so there is no input worth rejecting more specifically.
func Segment(ctx context.Context, req Request) ([]Mask, error) {
	return nil, unavailable("%s", Status())
}

// EnsureModel refuses rather than downloading: the weights would be 707 MB
// this binary has no library to load them with.
func EnsureModel(ctx context.Context, prog func(string)) (string, error) {
	return "", unavailable("%s", Status())
}
