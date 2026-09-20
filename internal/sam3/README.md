# internal/sam3 — prompted segmentation

Turns `"the cat"`, a click, or a drag into masks, so `/v1/edit` can be told
which pixels to change instead of depending on the diffusion model inferring it
from prompt phrasing.

Upstream is [PABannier/sam3.cpp](https://github.com/PABannier/sam3.cpp) — Meta's
SAM 3 on ggml with a Metal backend. Unlike the sd.cpp engine, which oflux
supervises as an opaque subprocess, **libsam3 is linked statically into the
daemon via cgo**. That was a deliberate choice: no IPC, no second process, no
second port — at the price that an internal `GGML_ASSERT` in libsam3 aborts the
whole daemon rather than one job.

## API

```go
sam3.Compiled()  bool    // was the library linked into this binary?
sam3.Available() bool    // ... and is a checkpoint installed, so Segment can run now?
sam3.Status()    string  // human-readable reason when it cannot
sam3.ModelPath() string  // which checkpoint Segment would load

sam3.EnsureModel(ctx, prog) (string, error)  // download the weights if absent
sam3.Segment(ctx, sam3.Request) ([]sam3.Mask, error)
```

`Compiled()` and `Available()` answer different questions, and the caller needs
both: `!Compiled()` is permanent and means 501; `Compiled() && !Available()`
just means the weights have not been fetched yet, which `EnsureModel` fixes.

Errors are always **wrapped**, in both build variants, so
`errors.Is(err, sam3.ErrUnavailable)` is the only test that works —
`err == sam3.ErrUnavailable` is never true in either build, deliberately, so
tagged and untagged binaries cannot diverge.

`ErrNoMatch` means the model ran and found nothing. That is a normal answer,
not a failure.

### Prompt styles

A `Request` carries exactly one of them. libsam3 has no path that takes both,
and `Segment` rejects the combination before anything crosses into C.

| set | path | notes |
|---|---|---|
| `Prompt` | PCS — text detector | finds every instance; needs the full `sam3-*` checkpoint |
| `Points` and/or `Boxes` | PVS — interactive decoder | at least one positive point, or a box |

Points and boxes are in **original-image pixel coordinates**; libsam3 scales
them by the encoded image's dimensions itself, so they must not be pre-scaled.
Everything is bounds-checked against the decoded image first — libsam3
validates nothing, so a bad coordinate is a segfault rather than an error.

`Separate: false` (the default) merges every detection above the score
threshold into one mask, carrying the best score and the union of the boxes.
`Separate: true` returns them individually, **best score first**.

Exemplar boxes on the PCS path are deliberately **not** exposed: upstream's
`sam3_segment_pcs` feeds `pos_exemplars` straight into math that expects
normalised `[0,1]` coordinates (sam3.cpp line ~7842) while its own GUI passes
raw image pixels, so the feature is inconsistent upstream. Boxes go to PVS,
where the scaling is explicit and verified.

## The build tag

`third_party/` is gitignored, so an unconditional cgo link would break
`go build ./...` and `go test ./...` on a fresh clone and in CI. Everything
cgo lives behind the `sam3` tag:

| build | `Compiled()` | `Segment()` |
|---|---|---|
| `go build ./...` | `false` | wrapped `ErrUnavailable` |
| `go build -tags sam3 ./...` | `true` | real masks |

The stub's `EnsureModel` refuses instead of downloading: 707 MB of weights a
stub binary has no library to load would be pure waste, and `go test` in this
repo never touches the network.

## Getting the library

```bash
./scripts/fetch-sam3.sh            # -> third_party/sam3-darwin-arm64/{include,lib,bin}
```

It takes the prebuilt `sam3-darwin-arm64.tar.gz` from the latest sam3.cpp
GitHub release and falls back to a CMake source build. If neither produces a
`libsam3.a`, it fails loudly and installs nothing. Re-running is a no-op once
the libraries are in place.

## Getting the weights

`EnsureModel` is the path that matters — it is what lets the feature ship on by
default instead of 501ing until someone finds a make target. It downloads into
`${OFLUX_HOME:-~/.oflux}/sam3`, emits puller-shaped progress lines, and is safe
to call from several goroutines at once (the second caller waits and then finds
the file already there). `scripts/fetch-sam3.sh --model [name]` does the same
thing from a shell.

Source: **`https://huggingface.co/PABannier/sam3.cpp`** (verified against the
live HF API). The format is sam3.cpp's own `.ggml` container, not GGUF: the
loader checks a private magic and carries the BPE tokenizer inside the file.

| file | size | notes |
|---|---|---|
| `sam3-q4_0.ggml` | 707 MB | the default; sha256 `5dafc790…1889` |
| `sam3-q4_1.ggml` | 756 MB | |
| `sam3-q8_0.ggml` | 1.10 GB | |
| `sam3-f16.ggml` | 1.84 GB | no measured latency gain over q4_0 |
| `sam3-f32.ggml` | 3.45 GB | |

`hfclient.Download` stages through `<name>.ggml.part` and renames only after
the sha256 matches, and checkpoint resolution accepts exact `.ggml` names only
— so an interrupted fetch can never be picked up as a corrupt model.

The `sam3-visual-*` checkpoints are **not usable here**: they drop the text
encoder and detector, so `sam3_segment_pcs` refuses to run. The binding
rejects them at load with an explicit message rather than returning empty
masks forever, and the fetch script refuses to download them.

Resolution order: `$OFLUX_SAM3_MODEL` (exact path), else the first of the table
above found in `$OFLUX_SAM3_DIR`, else `${OFLUX_HOME:-~/.oflux}/sam3`.

| env | default | |
|---|---|---|
| `OFLUX_SAM3_MODEL` | — | exact checkpoint path; EnsureModel refuses to download over it |
| `OFLUX_SAM3_DIR` | `${OFLUX_HOME:-~/.oflux}/sam3` | where to look and download |
| `OFLUX_SAM3_CPU` | unset | any value disables Metal |
| `OFLUX_SAM3_SCORE` | `0.5` | detection score threshold |
| `OFLUX_SAM3_NMS` | `0.1` | NMS IoU threshold |
| `HF_TOKEN` | unset | passed to hfclient |

---

# Wiring the build (to apply by hand)

## 1. `Makefile`

Add the switch near the top, after `BIN := dist/oflux`. It defaults to **on**,
and the library is a build prerequisite so a fresh clone still works:

```make
# SAM3 segmentation is linked statically into the binary. On by default; the
# libs are a prerequisite because third_party/ is gitignored, so a fresh clone
# has to fetch them before the tagged build can link.
SAM3   ?= 1
SAM3LIB := third_party/sam3-darwin-arm64/lib/libsam3.a
ifeq ($(SAM3),1)
GOTAGS := -tags sam3
SAM3DEP := $(SAM3LIB)
endif

$(SAM3LIB):
	./scripts/fetch-sam3.sh
```

Add the two targets to `.PHONY` and to the target list:

```make
.PHONY: build test race live-test smoke engine sam3 sam3-model icons app dmg sign notarize release install clean

sam3: ## fetch the SAM3 static libs into third_party/
	./scripts/fetch-sam3.sh

sam3-model: ## download the default SAM3 checkpoint (707MB) into ~/.oflux/sam3
	./scripts/fetch-sam3.sh --model
```

Thread the tag and the prerequisite through the Go invocations:

```make
build: $(SAM3DEP) ## build the CLI/daemon binary
	go build $(GOTAGS) -o $(BIN) ./cmd/oflux

test: $(SAM3DEP) ## run all tests
	go test $(GOTAGS) ./...

race: $(SAM3DEP) ## run all tests with the race detector
	go test -race $(GOTAGS) ./...

app: $(SAM3DEP) ## build dist/oflux.app
	VERSION=$(VERSION) SAM3=$(SAM3) ./scripts/build-app.sh
```

**One tradeoff to decide.** With `SAM3=1` the default, `make test` on a machine
with no `third_party/` fetches ~3.5 MB from GitHub once, which breaks the
"`make test` is fully offline" promise in CLAUDE.md for that first run. The
tests themselves stay offline — the tagged ones skip cleanly with no checkpoint
and the download tests use `httptest`. `SAM3=0 make test` restores a completely
network-free run and is the right thing for CI.

## 2. `scripts/build-app.sh`

Replace the single `go build` line with:

```sh
# SAM3=1 links libsam3 + ggml statically into the binary (see internal/sam3).
# Plain string, not a bash array: macOS ships bash 3.2, where "${A[@]}" on an
# empty array trips `set -u`.
TAGS=""
if [ "${SAM3:-1}" = "1" ]; then
  TAGS="-tags sam3"
  echo "==> linking SAM3 segmentation (third_party/sam3-darwin-arm64)"
fi
CGO_ENABLED=1 go build -trimpath $TAGS -ldflags "-X oflux/internal/version.Version=$VERSION" -o "$MACOS/oflux" ./cmd/oflux
```

Nothing else in that script changes: the libraries are static, so there is no
new file to copy into `Contents/Resources`.

## 3. cgo flags

The package already carries working `${SRCDIR}`-relative flags, so no
environment is needed when `third_party/sam3-darwin-arm64` is in place:

```
CPPFLAGS  -I${SRCDIR} -I${SRCDIR}/../../third_party/sam3-darwin-arm64/include
CXXFLAGS  -std=c++14 -O2
LDFLAGS   -L${SRCDIR}/../../third_party/sam3-darwin-arm64/lib
          -lsam3 -lggml -lggml-cpu -lggml-metal -lggml-blas -lggml-base -lobjc
          -framework Foundation -framework Metal -framework MetalKit
          -framework Accelerate -framework CoreFoundation
```

`CGO_CFLAGS` / `CGO_LDFLAGS` from the environment are appended to these, so
they can still point at a library tree somewhere else. `-lc++` is deliberately
absent: the Go toolchain adds the C++ runtime itself for packages containing
`.cpp`, and naming it again only produces a duplicate-library warning from
`ld`.

## 4. Signing and notarization

- **No new signing steps.** The libraries are `.a` archives linked into
  `Contents/MacOS/oflux`, so nothing new appears in `Contents/Resources` and
  the inside-out loop in `sign-app.sh` (which signs `*.dylib` and `sd-server`)
  is unchanged. One Mach-O in, one Mach-O out — notarization sees no new
  executable.
- **No new entitlements.** `ggml-metal` builds its pipeline from an embedded
  Metal shader library at runtime; the existing `allow-jit`,
  `allow-unsigned-executable-memory` and `disable-library-validation`
  entitlements (already required by sd-server) cover it.
- The binary grows ~4–5 MB and gains `Metal`, `MetalKit`, `Accelerate`,
  `Foundation`, `CoreFoundation` and `libobjc` in its load commands.
- Shipping with SAM3 on means every user runs a daemon that a libsam3 assert
  can abort — the accepted cost of linking rather than supervising. It also
  requires `third_party/sam3-darwin-arm64` on the release machine, which the
  `$(SAM3LIB)` prerequisite guarantees. Weights are never bundled (707 MB);
  `EnsureModel` fetches them on the user's machine.

---

## Runtime behaviour worth knowing

- **Calls are serialised.** `sam3_create_state` copies the model's ggml backend
  into the state (`state->backend = model.backend`), so two concurrent
  segmentations would drive one Metal backend from two goroutines. A gate
  channel allows exactly one in flight; waiting on it respects the context.
- **The image is encoded once per call** and the state is thrown away
  afterwards. Only the model is cached.
- **The model unloads after 5 minutes idle** — it is 700 MB to 3.4 GB resident
  inside a daemon that already supervises multi-gigabyte engines.
- First call after a cold start pays the model load plus ~8 s of Metal shader
  library init. After that an encode is ~3 s and a decode well under 1 s on an
  M4 Max.
- `Segment` never downloads. A 707 MB fetch inside an edit request would be a
  bad surprise, so the daemon should call `EnsureModel` at startup or behind an
  explicit action.
- libsam3 logs progress to **stderr**, which lands in the daemon log.
