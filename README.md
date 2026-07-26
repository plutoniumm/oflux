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
oflux pull qwen-image-edit flux.2-klein   # curated names or Hugging Face repos
oflux list / ps
oflux rm <name>...                        # several at a time
```

Then open <http://localhost:11534> for the UI, or:

```bash
curl localhost:11534/v1/edit -d '{
  "model": "qwen-image-edit",
  "prompt": "make it night time",
  "image": "data:image/png;base64,..."
}'
```

Drop `"image"` and hit `/v1/generate` to generate instead. Both return
`{"images": ["<base64 png>"]}`.

Optional fields: `loras[]`, `ref_images[]`, `mask_image`, `control_image` +
`control_strength`, `negative_prompt`, `strength`, `steps`, `seed`, `cfg`,
`sampler`, `scheduler`, `guidance{txt_cfg,img_cfg,distilled_guidance,slg}`.

On a step-distilled model (cfg 1.0) a large-area edit can drift into
regenerating the image instead of editing it. Either say what to keep — "keep
the fox exactly as it is, replace only the background" — or raise
`guidance.img_cfg` to re-anchor it to the input.

**OpenAI-compatible:** `POST /v1/images/edits` (multipart) and
`/v1/images/generations` (JSON).
**Management:** `/api/pull`, `/api/tags`, `/api/delete`, `/api/ps`, `/api/loras`.

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
  "model": "qwen-image-edit",
  "prompt": "make it night time",
  "image": "data:image/png;base64,...",
  "loras": [{"name": "qwen-edit-lightning-4step", "scale": 1.0}]
}'
```

| Name | For | Steps |
|------|-----|-------|
| `qwen-edit-lightning-4step` / `-8step` | `qwen-image-edit` | 4 / 8 |
| `qwen-image-lightning-4step` | `qwen-image` | 4 |
| `flux-turbo-8step`, `flux-hyper-8step` | `flux.1-dev`, `flux.1-krea` | 8 |

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
| `qwen-image-edit` | **both** — best instruction following |
| `flux.2-klein` | **both** — 4-step, fast |
| `flux.1-kontext` | **edit** |
| `z-image-turbo` | generate — fast |
| `flux.1-krea`, `flux.1-dev`, `flux.1-schnell`, `qwen-image` | generate |

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
{ "port": 11534, "idle_ttl": "2m", "max_loaded": 1, "default_quant": "Q8_0", "hf_token": "" }
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

Releases are Developer-ID signed and notarized; `verify.sh` (untracked, see
`verify.sh.example`) holds the credentials and runs the whole flow. Local builds
are ad-hoc signed and refuse to auto-update — updates there are `git pull &&
make install`.

## License

MIT
