package sam3

import (
	"errors"
	"strings"
	"testing"
)

func TestValidateRequestRouting(t *testing.T) {
	const w, h = 100, 80
	tests := []struct {
		name   string
		req    Request
		want   plan
		errHas string
	}{
		{
			name: "text runs PCS",
			req:  Request{Prompt: "  a cat "},
			want: plan{pcs: true, prompt: "a cat"},
		},
		{
			name: "points run PVS",
			req:  Request{Points: []Point{{X: 10, Y: 20}}},
			want: plan{points: []Point{{X: 10, Y: 20}}},
		},
		{
			name: "box alone runs PVS",
			req:  Request{Boxes: []Box{{X0: 1, Y0: 2, X1: 50, Y1: 60}}},
			want: plan{box: Box{X0: 1, Y0: 2, X1: 50, Y1: 60}, useBox: true},
		},
		{
			name: "negative point with a box is fine",
			req: Request{
				Points: []Point{{X: 5, Y: 5, Negative: true}},
				Boxes:  []Box{{X0: 0, Y0: 0, X1: 99, Y1: 79}},
			},
			want: plan{
				points: []Point{{X: 5, Y: 5, Negative: true}},
				box:    Box{X0: 0, Y0: 0, X1: 99, Y1: 79},
				useBox: true,
			},
		},
		{name: "nothing set", req: Request{}, errHas: "no prompt"},
		{
			name:   "text plus points is not a thing libsam3 has",
			req:    Request{Prompt: "a cat", Points: []Point{{X: 1, Y: 1}}},
			errHas: "cannot be combined",
		},
		{
			name:   "text plus box is not a thing libsam3 has",
			req:    Request{Prompt: "a cat", Boxes: []Box{{X1: 5, Y1: 5}}},
			errHas: "cannot be combined",
		},
		{
			name:   "point outside the image",
			req:    Request{Points: []Point{{X: 10, Y: 20}, {X: 100, Y: 20}}},
			errHas: "point 1 (100,20) is outside the 100x80 image",
		},
		{
			name:   "negative point coordinate",
			req:    Request{Points: []Point{{X: -1, Y: 0}}},
			errHas: "outside",
		},
		{
			name:   "only negative points",
			req:    Request{Points: []Point{{X: 5, Y: 5, Negative: true}}},
			errHas: "at least one positive point",
		},
		{
			name:   "two boxes",
			req:    Request{Boxes: []Box{{X1: 5, Y1: 5}, {X1: 6, Y1: 6}}},
			errHas: "at most one",
		},
		{
			name:   "inverted box",
			req:    Request{Boxes: []Box{{X0: 50, Y0: 50, X1: 10, Y1: 60}}},
			errHas: "is empty",
		},
		{
			name:   "box past the right edge",
			req:    Request{Boxes: []Box{{X0: 0, Y0: 0, X1: 101, Y1: 10}}},
			errHas: "outside",
		},
		{
			name:   "too many points",
			req:    Request{Points: make([]Point, MaxPoints+1)},
			errHas: "limit is",
		},
		{
			name:   "prompt too long",
			req:    Request{Prompt: strings.Repeat("x", MaxPromptBytes+1)},
			errHas: "limit is",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := validateRequest(tt.req, w, h)
			if tt.errHas != "" {
				if err == nil {
					t.Fatalf("validateRequest = %+v, want error containing %q", got, tt.errHas)
				}
				if !strings.Contains(err.Error(), tt.errHas) {
					t.Fatalf("error = %v, want it to contain %q", err, tt.errHas)
				}
				return
			}
			if err != nil {
				t.Fatalf("validateRequest: %v", err)
			}
			if got.pcs != tt.want.pcs || got.prompt != tt.want.prompt ||
				got.useBox != tt.want.useBox || got.box != tt.want.box ||
				len(got.points) != len(tt.want.points) {
				t.Fatalf("plan = %+v, want %+v", got, tt.want)
			}
			for i := range got.points {
				if got.points[i] != tt.want.points[i] {
					t.Fatalf("point %d = %+v, want %+v", i, got.points[i], tt.want.points[i])
				}
			}
		})
	}
}

func TestRequestIsEmptyMatchesValidate(t *testing.T) {
	empty := Request{Prompt: "   "}
	if !requestIsEmpty(empty) {
		t.Fatal("requestIsEmpty says a whitespace-only prompt is a prompt")
	}
	if _, err := validateRequest(empty, 100, 100); !errors.Is(err, errNoPrompt) {
		t.Fatalf("validateRequest = %v, want errNoPrompt — the two checks must agree", err)
	}
}

func TestPlanDescribe(t *testing.T) {
	pcs, err := validateRequest(Request{Prompt: "a cat"}, 100, 100)
	if err != nil {
		t.Fatal(err)
	}
	if got := pcs.describe(); got != `"a cat"` {
		t.Fatalf("describe = %s", got)
	}
	pvs, err := validateRequest(Request{
		Points: []Point{{X: 1, Y: 2}},
		Boxes:  []Box{{X0: 0, Y0: 0, X1: 10, Y1: 10}},
	}, 100, 100)
	if err != nil {
		t.Fatal(err)
	}
	if got := pvs.describe(); got != "1 point(s) + box (0,0)-(10,10)" {
		t.Fatalf("describe = %s", got)
	}
}
