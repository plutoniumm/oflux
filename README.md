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
oflux pull qwen-image-edit     # or any Hugging Face repo
oflux list / ps / rm <name>
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

Optional fields: `ref_images[]`, `mask_image`, `control_image` +
`control_strength`, `negative_prompt`, `strength`, `steps`, `seed`, `cfg`.

**OpenAI-compatible:** `POST /v1/images/edits` (multipart) and
`/v1/images/generations` (JSON).
**Management:** `/api/pull`, `/api/tags`, `/api/delete`, `/api/ps`.

## Models

| Name | Task |
|------|------|
| `qwen-image-edit` | **edit** — best instruction following |
| `flux.1-kontext` | **edit** |
| `z-image-turbo` | generate — fast |
| `flux.1-krea`, `flux.1-dev`, `flux.1-schnell`, `qwen-image` | generate |

Editing needs Qwen-Image-Edit or Flux-Kontext; the rest generate. Quantized
weights (Q8_0 by default) are preferred and pulled from GGUF mirrors.

Any other Hugging Face repo works too — `oflux pull <org>/<repo>` inspects it and
either installs it or tells you exactly what makes it incompatible.

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
