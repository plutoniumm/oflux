package supervisor

import (
	"context"
	"errors"
	"maps"
	"slices"
	"sync"
)

// ErrBusy is oflux's queue at QueueDepth; engineclient.ErrQueueFull is the
// engine rejecting a submit with its own 429.
var ErrBusy = errors.New("supervisor: queue full")

// One generation at a time is not timidity: sd-server takes no parallelism flag,
// reports a single max_queue_size and has no worker/slot symbols at all, so a
// second submit would only queue inside the engine, where oflux can no longer
// cancel or count it.
type gate struct {
	mu      sync.Mutex
	running int
	max     int
	depth   int
	waiting []chan struct{}
}

func newGate(max, depth int) *gate {
	if max <= 0 {
		max = 1
	}
	return &gate{max: max, depth: depth}
}

func (g *gate) enqueue() (chan struct{}, error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.running < g.max && len(g.waiting) == 0 {
		g.running++
		return nil, nil
	}
	if g.depth > 0 && len(g.waiting) >= g.depth {
		return nil, ErrBusy
	}
	ch := make(chan struct{}, 1)
	g.waiting = append(g.waiting, ch)
	return ch, nil
}

func (g *gate) wait(ctx context.Context, ch chan struct{}) error {
	select {
	case <-ch:
		return nil
	case <-ctx.Done():
	}
	g.mu.Lock()
	if i := slices.Index(g.waiting, ch); i >= 0 {
		g.waiting = slices.Delete(g.waiting, i, i+1)
		g.mu.Unlock()
		return ctx.Err()
	}
	g.mu.Unlock()
	// Handed the slot as we gave up on it: ours to pass on, not to drop.
	<-ch
	g.release()
	return ctx.Err()
}

func (g *gate) release() {
	g.mu.Lock()
	defer g.mu.Unlock()
	if len(g.waiting) > 0 {
		ch := g.waiting[0]
		g.waiting = g.waiting[1:]
		ch <- struct{}{}
		return
	}
	g.running--
}

// A ticket is a place in a model's queue plus the pending count that keeps the
// model from being evicted underneath it.
type ticket struct {
	name string
	gate *gate
	ch   chan struct{}
	held bool
}

// enqueue returns as soon as it holds the slot or a place in line, so a full
// queue is ErrBusy now rather than a wait that never ends.
func (s *Supervisor) enqueue(name string) (*ticket, error) {
	s.mu.Lock()
	s.pending[name]++
	g, ok := s.gates[name]
	if !ok {
		g = newGate(s.opts.MaxConcurrent, s.opts.QueueDepth)
		s.gates[name] = g
	}
	s.mu.Unlock()

	ch, err := g.enqueue()
	if err != nil {
		s.leave(name)
		return nil, err
	}
	return &ticket{name: name, gate: g, ch: ch, held: ch == nil}, nil
}

func (t *ticket) wait(ctx context.Context) error {
	if t.held {
		return nil
	}
	if err := t.gate.wait(ctx, t.ch); err != nil {
		return err
	}
	t.held = true
	return nil
}

func (s *Supervisor) release(t *ticket) {
	if t.held {
		t.gate.release()
		t.held = false
	}
	s.leave(t.name)
}

func (s *Supervisor) leave(name string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.pending[name]--; s.pending[name] <= 0 {
		delete(s.pending, name)
		delete(s.gates, name)
	}
}

// Pending reports requests queued or running per model, including models that
// are not loaded yet. An absent model has none.
func (s *Supervisor) Pending() map[string]int {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.pending) == 0 {
		return nil
	}
	return maps.Clone(s.pending)
}
