<img src="packaging/oflux.svg" width="88" align="right" alt="">

# oflux

Local diffusion **image editing** on macOS — an Ollama-shaped daemon around
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp). Send an
image and a prompt, get an edited image back. Generation falls out of the same
endpoint.

Apple Silicon only. Models run on your GPU; nothing leaves the machine.

## Install

Grab the `.dmg` from [Releases](../../releases) — signed and notarized, so it
opens normally — and drag **oflux.app** to Applications. The fox appears in your
menu bar, the `oflux` CLI is linked onto your `PATH`, and it starts at login.

From source: `make engine && make install` (needs Go and `brew install librsvg`).

## Use

```bash
oflux pull qwe-2511 flx-2-klein-4b  # curated names or Hugging Face repos
oflux list / ps
oflux rm <name>...                        # several at a time
```

Then open <http://localhost:11534> for the UI, or:

```bash
curl localhost:11534/v1/edit -d '{
  "model": "qwe-2511",
  "prompt": "make it night time",
  "image": "data:image/png;base64,..."
}'
```

Drop `"image"` and hit `/v1/generate` to generate instead. Both return
`{"images": ["<base64 png>"], "seed": 12345}` — the seed is echoed even when you
did not pass one, so a good result is always reproducible.

`"image"` also takes an array: the first entry is the subject, the rest are
reference images. That is how multi-image editing works — style refs, identity
refs, "put this in that".

```bash
curl localhost:11534/v1/edit -d '{
  "model": "qwe-2511",
  "prompt": "put the person from the second image on this bench",
  "image": ["data:image/png;base64,...", "data:image/png;base64,..."]
}'
```

Optional fields: `loras[]`, `ref_images[]`, `mask` (white = edit),
`mask_prompt` (name what to change and SAM3 masks it for you),
`control_image` + `control_strength`, `negative_prompt`, `strength`, `steps`,
`seed`, `cfg`, `sampler`, `scheduler`, `transparent`, `keep_alive`, `stream`,
`guidance{txt_cfg,img_cfg,distilled_guidance,slg}`.

`transparent: true` asks for an RGBA result on a model whose weights can produce
one — today that is `qwe-2.1`, which `/api/tags` marks with `"alpha":
true`. There is no engine switch for it: the model decides from the wording of
the prompt, so oflux supplies the phrasing Qwen prescribes. Asking a model that
cannot is a 400 rather than a silently opaque image.

**Known upstream bug:** on Metal the subject decodes correctly but much of the
background comes back opaque white instead of transparent, so the cut-out needs
cleaning up by hand. That is [stable-diffusion.cpp#2024][sd2024], not something
oflux can fix from here — raising steps does not help. The alpha channel itself
is real, so a cleanup pass on the result works.

[sd2024]: https://github.com/leejet/stable-diffusion.cpp/issues/2024

An edit returns the same size it was given: with no `width`/`height`, the canvas
follows the input image (rounded to a multiple of 64, capped at 2048). Pass them
explicitly to override — a smaller canvas is much faster, since sampling cost
scales with the pixel count.

`keep_alive` overrides the idle unload for that model — `"10m"` to hold it, `-1`
to pin it resident. Chained edits otherwise pay a full reload between turns.

On a step-distilled model (cfg 1.0) a large-area edit can drift into
regenerating the image instead of editing it. Either say what to keep — "keep
the fox exactly as it is, replace only the background" — or raise
`guidance.img_cfg` to re-anchor it to the input.

### Long jobs

An edit can take minutes. `"stream": true` returns NDJSON instead of one
response: the first line carries a job id, then status and step progress, then
the same object a normal call would return. Disconnecting does not cancel the
run — reattach with `GET /v1/jobs/{id}`, or abort with `DELETE /v1/jobs/{id}`.

Requests queue one at a time per model (sd-server takes no parallelism flag), so
a burst is serialised rather than thrashing the GPU. Past `queue_depth` the
daemon answers `429` with `Retry-After` instead of blocking.

### Presets

A distilled LoRA is only correct at its trained steps and cfg, so bake the
combination once and call it by name:

```bash
oflux preset add qwen8 --model qwe-2511 \
  --lora qwen-edit-lightning-8step --steps 8 --cfg 1 --label "Qwen 2511 (8-step)"
```

`"model": "qwen8"` then resolves to all of it; anything you pass explicitly wins.
Presets appear in `/api/tags` as callable names.

### Errors

Failures return `{error, code, model, detail}`. `error` is the one line worth
reading, `code` is a stable slug to branch on, and `detail` holds the tail of the
engine log — sd.cpp failures are hundreds of lines and used to arrive as one
opaque string.

**OpenAI-compatible:** `POST /v1/images/edits` (multipart) and
`/v1/images/generations` (JSON), including OpenAI's error shape.
**Management:** `/api/pull`, `/api/tags`, `/api/delete`, `/api/ps`, `/api/loras`,
`/api/presets`.
**Segmentation:** `POST /v1/segment` takes `{image, prompt}` — or point/box
prompts — and returns masks. Usually you do not need it directly: pass
`mask_prompt` to `/v1/edit` and the mask is generated and applied in one call.

## LoRAs

Adapters are applied per request and need no reload, so one loaded model can
serve different behaviours. The big win is step distillation: a ~0.9 GB adapter
turns a 20-step model into a 4-step one, instead of downloading a separately
merged 13–22 GB checkpoint.

```bash
oflux lora ls                             # installed + available
oflux lora pull qwen-edit-lightning-4step # ~0.9 GB
oflux lora pull <org>/<repo> --file <path-in-repo> --as <name>
```

```bash
curl localhost:11534/v1/edit -d '{
  "model": "qwe-2511",
  "prompt": "make it night time",
  "image": "data:image/png;base64,...",
  "loras": [{"name": "qwen-edit-lightning-4step", "scale": 1.0}]
}'
```

| Name | For | Steps |
|------|-----|-------|
| `qwen-2.1-turbo-6step` | `qwen-image-2.1` | 6 |
| `qwen-edit-lightning-4step` / `-8step` | `qwen-image-edit` | 4 / 8 |
| `qwen-image-lightning-4step` | `qwen-image` | 4 |
| `flux-turbo-8step`, `flux-hyper-8step` | `flux` | 8 |

A curated step-distillation adapter also supplies its sampling regime (steps and
cfg), because running one at the base model's defaults produces burnt output.
Passing `steps` or `cfg` explicitly overrides it.

## ControlNet

The engine can only load a ControlNet at startup and offers no way to switch one
over HTTP, so it is attached to a model at install time rather than chosen per
request:

```bash
oflux pull <org>/<sd15-repo> \
  --control-net lllyasviel/control_v11p_sd15_canny --as sd15-canny
```

Then `control_image` and `control_strength` work on that model. Sending
`control_image` to a model installed without one is a 400 rather than a silently
ignored field. Note stable-diffusion.cpp documents ControlNet for **SD 1.5**
only — none of the curated models above support it.

## Models

| Name | Task |
|------|------|
| `qwe-2.1` | **both** — newest; 10 reference images, transparent output |
| `qwe-2.1-uc` | **both** — the same weights, abliterated |
| `qwe-2511` | **both** — best instruction following |
| `flx-2-klein-4b` | **both** — 4-step, fast |
| `flx-2-klein-9b` | **both** — the larger klein |
| `flx-1-kontext` | **edit** |
| `zim-1-turbo` | generate — fast |
| `flx-1-krea`, `flx-1-dev`, `flx-1-schnell`, `qwi-1` | generate |

Curated names changed in 1.3.1 to `<line><version>[-modifier]`, so successive
checkpoints of one line coexist (`qwe-2511` and `qwe-2.1` are both installable)
instead of one name quietly meaning whichever is newest. There are no aliases:
`oflux pull qwen-image-edit` now fails. Models you already installed keep their
old names until you run `./scripts/migrate-names.sh` once per machine — it
renames the manifests and re-points any presets, and `DRY=1` shows the plan
without touching anything.

"both" is a hybrid: the same weights edit when given an image and generate from
text alone. Quantized weights (Q8_0 by default) are preferred and pulled from
GGUF mirrors.

Any other Hugging Face repo works too — `oflux pull <org>/<repo>` inspects it and
either installs it or tells you exactly what makes it incompatible. When a repo
publishes many builds, pin one with `--file`:

```bash
oflux pull Novice25/Qwen-Image-Edit-Rapid-AIO-GGUF \
  --file v23/Qwen-Rapid-NSFW-v23_Q4_K.gguf
```

Weights whose name marks them as step-distilled (`…-4steps…`, `lightning`,
`rapid`) are installed with few-step sampling defaults rather than the base
architecture's.

## How it works

```
menu bar ── hosts ──► daemon :11534
                        │
   HTTP API ─► supervisor ── spawns/reaps ─► sd-server (Metal, one per model)
                        │
   store ~/.oflux/{blobs,manifests}  ◄── puller ── Hugging Face
```

The supervisor loads a model on first use and unloads it after 2 minutes idle.
oflux never reimplements inference — `sd-server` is a bundled binary it talks to
over HTTP.

## Config `~/.oflux/config.json`

```json
{ "port": 11534, "idle_ttl": "2m", "max_loaded": 1, "max_concurrent": 1,
  "queue_depth": 8, "default_quant": "Q8_0", "hf_token": "" }
```

`hf_token` is only needed for gated repos; the mirrors oflux prefers are open.

## Develop

```bash
make build      # binary
make test       # unit tests
make live-test  # drives the running daemon on real GPU work
make app        # dist/oflux.app
make release    # signed + notarized .dmg/.zip (needs a Developer ID cert)
```

SAM3 is compiled into the binary rather than supervised as a subprocess, and is
on by default — the static libs are fetched automatically on first build, and
the checkpoint downloads on first use. `make build SAM3=0` opts out, and
`/v1/segment` then answers 501.

Releases are Developer-ID signed and notarized; `verify.sh` (untracked, see
`verify.sh.example`) holds the credentials and runs the whole flow. Local builds
are ad-hoc signed and refuse to auto-update — updates there are `git pull &&
make install`.

`make release` needs notary credentials once:

```bash
xcrun notarytool store-credentials oflux-notary \
  --apple-id "you@example.com" --team-id "TEAMID" --password "<app-specific-password>"
```

Or export `APPLE_ID`, `TEAM_ID` and `APP_PASSWORD` for a single run.
`./scripts/notarize.sh --check` verifies credentials without submitting; the
release runs it first so a missing one fails immediately rather than after a
full build. The version comes from the `VERSION` file — bump it there only.

## License

MIT
