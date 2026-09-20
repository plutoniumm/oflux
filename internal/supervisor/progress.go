package supervisor

import (
	"context"
	"io"
	"os"
	"regexp"
	"strconv"
	"time"
)

// The engine's job payload carries no progress. Verified against the bundled
// sd-server: its own web UI polls the four endpoints oflux does and renders
// id, kind, status, queue_position, created, started, completed, result, error
// — no step field exists. Progress only ever reaches the log the supervisor
// already captures, as "\r  |=====>  | 3/28 - 6.69s/it\x1b[K". Tensor-loading
// bars share the N/M shape but report MB/s, which is what the rate unit below
// discriminates on.
var samplingBar = regexp.MustCompile(`\|\s*(\d+)/(\d+) - [0-9.]+(?:s/it|it/s)`)

const (
	progressPoll = 250 * time.Millisecond
	// Only the newest bar matters, so a backlog is skipped, not scanned.
	maxProgressCatchUp = 64 << 10
)

// A log tailProgress cannot read means no progress, never a failed request. One
// engine writes one log and runs one job at a time, so while MaxConcurrent is 1
// a bar can only belong to the request being tailed.
func tailProgress(ctx context.Context, path string, onStep func(step, total int)) {
	f, err := os.Open(path)
	if err != nil {
		return
	}
	defer f.Close()

	// Bars already in the log belong to earlier requests.
	off, err := f.Seek(0, io.SeekEnd)
	if err != nil {
		return
	}
	buf := make([]byte, maxProgressCatchUp)
	lastStep, lastTotal := 0, 0

	tick := time.NewTicker(progressPoll)
	defer tick.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-tick.C:
		}
		fi, err := f.Stat()
		if err != nil {
			return
		}
		size := fi.Size()
		if size < off {
			off = 0 // the engine restarted and truncated its log
		}
		if size <= off {
			continue
		}
		if size-off > int64(len(buf)) {
			off = size - int64(len(buf))
		}
		n, _ := f.ReadAt(buf[:size-off], off)
		if n <= 0 {
			continue
		}
		off += int64(n)
		step, total, ok := lastSamplingBar(buf[:n])
		if !ok || (step == lastStep && total == lastTotal) {
			continue
		}
		lastStep, lastTotal = step, total
		onStep(step, total)
	}
}

func lastSamplingBar(chunk []byte) (step, total int, ok bool) {
	m := samplingBar.FindAllSubmatchIndex(chunk, -1)
	if len(m) == 0 {
		return 0, 0, false
	}
	last := m[len(m)-1]
	step, err1 := strconv.Atoi(string(chunk[last[2]:last[3]]))
	total, err2 := strconv.Atoi(string(chunk[last[4]:last[5]]))
	if err1 != nil || err2 != nil || total <= 0 {
		return 0, 0, false
	}
	return step, total, true
}
