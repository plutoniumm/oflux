# The oflux catalog

Everything `oflux` can install: the curated models, the curated LoRA adapters,
and the exact commands that pull them.

A curated name is pinned to a hand-verified Hugging Face repo and filename, so
`oflux pull <name>` just works. Any other compatible Hugging Face repo works
too — oflux inspects it and either installs it or tells you exactly what makes
it unusable.

**Jump to:** [Models](#models) · [Quantization](#quantization) · [LoRAs](#loras)
· [Pulling](#pulling) · [ControlNet](#controlnet) · [Calling it](#calling-it)

> Apple Silicon only. The daemon listens on `http://localhost:11534`.

---

## Models

```bash
oflux pull qwe-2511
```

| Name | Does | Architecture | Pulled from | What it is |
|------|------|--------------|-------------|------------|
| `qwe-2.1` | **both** | `qwen-image-2.1` | `leejet/Qwen-Image-2.1-GGUF` | Qwen-Image 2.1 — unified text-to-image and editing, up to 10 reference images, native RGBA (Qwen3-VL-8B encoder). Non-commercial licence. |
| `qwe-2.1-uc` | **both** | `qwen-image-2.1` | `abenzerps/Qwen-Image-2.1-Uncensored-GGUF` | Qwen-Image 2.1, abliterated — as above with the refusal direction removed. Third-party weights. |
| `qwe-2511` | **both** | `qwen-image-edit` | `unsloth/Qwen-Image-Edit-2511-GGUF` | Qwen-Image-Edit 2511 — instruction image editing (Qwen2.5-VL encoder). |
| `flx-1-kontext` | edit | `flux-kontext` | `QuantStack/FLUX.1-Kontext-dev-GGUF` | FLUX.1 Kontext [dev] — in-context image editing. |
| `flx-2-klein-4b` | **both** | `flux2-klein` | `leejet/FLUX.2-klein-4B-GGUF` | FLUX.2 klein 4B — few-step text-to-image and editing (Qwen3-4B encoder). |
| `flx-2-klein-9b` | **both** | `flux2-klein` | `leejet/FLUX.2-klein-9B-GGUF` | FLUX.2 klein 9B — few-step text-to-image and editing (Qwen3-8B encoder). |
| `zim-1-turbo` | generate | `z-image` | `leejet/Z-Image-Turbo-GGUF` | Z-Image Turbo — fast few-step text-to-image (Qwen3-4B encoder). |
| `flx-1-krea` | generate | `flux` | `QuantStack/FLUX.1-Krea-dev-GGUF` | FLUX.1 Krea [dev] — photographic text-to-image. |
| `flx-1-dev` | generate | `flux` | `city96/FLUX.1-dev-gguf` | FLUX.1 [dev] — guidance-distilled text-to-image. |
| `flx-1-schnell` | generate | `flux` | `city96/FLUX.1-schnell-gguf` | FLUX.1 [schnell] — fast few-step text-to-image (Apache-2.0). |
| `qwi-1` | generate | `qwen-image` | `QuantStack/Qwen-Image-GGUF` | Qwen-Image — text-to-image (Qwen2.5-VL encoder). |

Which quant labels each of these publishes is
[its own table below](#what-each-model-publishes).

**`both` is a hybrid, not a pair of models.** The same weights edit when given
an image and generate from text alone. The route you call — `/v1/edit` or
`/v1/generate` — decides how the input is used, not the model.

The **architecture** column is what LoRAs match on: an adapter trained for
`flux` applies to every model in the `flux` row.

A friendly name hides which checkpoint it actually is, so `oflux list` and
`GET /api/tags` report the identity alongside it: `base` (for `qwe-2511`,
`Qwen-Image-Edit-2511`) and, when the publisher dates its releases, `revision`
(`2511`). That is the answer to "which checkpoint is this really".

### What a pull actually downloads

A GGUF diffusion repo ships only the diffusion weights. The VAE and text
encoders are shared across every model of an architecture, so oflux fetches
them from their own pinned repos and caches them once:

| Architecture | Shared components pulled alongside the weights |
|--------------|------------------------------------------------|
| `flux`, `flux-kontext` | VAE `ffxvs/vae-flux` · CLIP-L `comfyanonymous/flux_text_encoders` · T5-XXL `city96/t5-v1_1-xxl-encoder-gguf` (quantized) |
| `flux2-klein` — 4B | VAE `Comfy-Org/vae-text-encorder-for-flux-klein-4b` · LLM encoder `unsloth/Qwen3-4B-GGUF` (quantized) |
| `flux2-klein` — 9B | VAE `Comfy-Org/vae-text-encorder-for-flux-klein-9b` · LLM encoder `unsloth/Qwen3-8B-GGUF` (quantized) |
| `qwen-image`, `qwen-image-edit` | VAE `Comfy-Org/Qwen-Image_ComfyUI` · LLM encoder `mradermacher/Qwen2.5-VL-7B-Instruct-GGUF` (quantized) |
| `qwen-image-2.1` | VAE `Comfy-Org/Qwen-Image-2.1` · LLM encoder + vision tower `Qwen/Qwen3-VL-8B-Instruct-GGUF` (quantized) |

`qwe-2.1-uc` overrides the encoder with the matching abliterated
build, `pottokao/Qwen-Image-2.1-Text-Encoder-Heretic-GGUF` — the prompt is read
by Qwen3-VL, so ablating only the diffusion weights does half the job.

| `z-image` | VAE `ffxvs/vae-flux` · LLM encoder `unsloth/Qwen3-4B-Instruct-2507-GGUF` (quantized) |

Pull a second FLUX model and only its diffusion weights are new — the VAE and
encoders are already on disk.

Both klein sizes launch through the same `flux2-klein` architecture but take
**different encoders**, picked from the `klein-9b` in the repo id and filename.
This matters: handing a 9B the 4B encoder does not fail cleanly, it kills the
engine with a tensor-shape mismatch that reads like a corrupt download.

> The `encorder` in those two VAE repo ids is upstream's own spelling, not a
> typo here — `Comfy-Org/flux2-klein` now only 307-redirects to them.

## Quantization

Quantized weights are preferred and come from GGUF mirrors. The default is
`Q8_0` (`default_quant` in `~/.oflux/config.json`); override it per pull:

```bash
oflux pull qwe-2511 --quant Q4_K_M      # or -q
```

### What each model publishes

These are the labels each diffusion repo really ships, best first:

| Model | Quant labels |
|-------|--------------|
| `qwe-2.1` | `Q8_0` `Q6_K` `Q5_0` `Q4_K` `Q4_0` `Q3_K` `Q2_K` |
| `qwe-2.1-uc` | `BF16` `Q8_0` `Q6_K` `Q5_K_M` `Q4_K_M` `Q4_0` |
| `qwe-2511` | `Q8_0` `Q6_K` `Q5_K_M` `Q5_K_S` `Q5_1` `Q5_0` `Q4_K_M` `Q4_K_S` `Q4_1` `Q4_0` `Q3_K_L` `Q3_K_M` `Q3_K_S` `Q2_K` |
| `flx-1-kontext` | `Q8_0` `Q6_K` `Q5_K_M` `Q5_K_S` `Q5_1` `Q5_0` `Q4_K_M` `Q4_K_S` `Q4_1` `Q4_0` `Q3_K_M` `Q3_K_S` `Q2_K` |
| `flx-2-klein-4b` | `Q8_0` `Q4_0` |
| `flx-2-klein-9b` | `Q8_0` `Q4_0` |
| `zim-1-turbo` | `Q8_0` `Q6_K` `Q5_0` `Q4_K` `Q4_0` `Q3_K` `Q2_K` |
| `flx-1-krea` | `Q8_0` `Q6_K` `Q5_K_M` `Q5_K_S` `Q5_1` `Q5_0` `Q4_K_M` `Q4_K_S` `Q4_1` `Q4_0` `Q3_K_M` `Q3_K_S` `Q2_K` |
| `flx-1-dev` | `Q8_0` `Q6_K` `Q5_K_S` `Q5_1` `Q5_0` `Q4_K_S` `Q4_1` `Q4_0` `Q3_K_S` `Q2_K` |
| `flx-1-schnell` | `Q8_0` `Q6_K` `Q5_K_S` `Q5_1` `Q5_0` `Q4_K_S` `Q4_1` `Q4_0` `Q3_K_S` `Q2_K` |
| `qwi-1` | `Q8_0` `Q6_K` `Q5_K_M` `Q5_K_S` `Q5_1` `Q5_0` `Q4_K_M` `Q4_K_S` `Q4_1` `Q4_0` `Q3_K_M` `Q3_K_S` `Q2_K` |

The list describes the **diffusion repo alone**. Nothing is hand-intersected
with the encoders any more.

### Asking for a label a repo does not have

Publishers disagree about granularity — city96's FLUX repos ship the `_S`
K-quants and no `_M`, unsloth's Qwen3-8B has no `Q4_0` at all — so every
component resolves independently against the labels **its own** repo publishes,
walking one shared fallback chain:

1. the label you asked for,
2. the rest of its bit-width family (`Q4_K_M` → `Q4_K_S` → `Q4_K` → `Q4_1` → `Q4_0`),
3. the general ladder nearly everyone ships: `Q8_0` → `Q6_K` → `Q5_K_M` → `Q4_K_M` → `Q4_0`,
4. whatever else is quantized, with full precision (`F16`/`BF16`/`F32`) last —
   those are enormous rather than wrong.

So `oflux pull flx-2-klein-9b --quant Q4_0` takes the `Q4_0` diffusion weights
it asked for and pairs them with the nearest Q4 the encoder repo does publish,
`Q4_K_M`. The old behaviour was a 404 on a filename that never existed, after
gigabytes had already downloaded.

## LoRAs

```bash
oflux lora ls                              # installed + available, with size and steps
oflux lora pull qwen-edit-lightning-4step
```

| Name | Applies to | Steps | CFG | What it is |
|------|-----------|-------|-----|------------|
| `qwen-2.1-turbo-6step` | `qwen-image-2.1` | 6 | 1.0 | 6-step distillation for Qwen-Image 2.1 (~5.6x faster). Preview quality; sd.cpp binds 326 of its 454 tensors. |
| `qwen-edit-lightning-4step` | `qwen-image-edit` | 4 | 1.0 | 4-step distillation for Qwen-Image-Edit 2511 (~5x faster edits). |
| `qwen-edit-lightning-8step` | `qwen-image-edit` | 8 | 1.0 | 8-step distillation for Qwen-Image-Edit 2511 (better fidelity than 4-step). |
| `qwen-image-lightning-4step` | `qwen-image` | 4 | 1.0 | 4-step distillation for Qwen-Image text-to-image. |
| `flux-turbo-8step` | `flux` | 8 | 1.0 | 8-step turbo distillation for FLUX.1 dev/Krea. |
| `flux-hyper-8step` | `flux` | 8 | 1.0 | ByteDance Hyper-SD 8-step distillation for FLUX.1 dev/Krea. |

"Applies to" is an **architecture**, so `qwen-image-edit` covers the
`qwe-2511` model and `flux` covers `flx-1-dev`, `flx-1-krea` and
`flx-1-schnell`. Both `flux` adapters are trained on FLUX.1-dev; Krea is a dev
finetune sharing its weights layout, which is why the same adapter applies.

### Why step distillation is the win

A ~0.9 GB adapter turns a 20-step model into a 4-step one. The alternative is
downloading a separately merged 13–22 GB checkpoint for the same speed-up.

Adapters are applied **per request** and need no reload, so one loaded model can
serve several behaviours — plain edits on one call, 4-step edits on the next.

A curated step-distillation adapter also supplies its own sampling regime (the
Steps and CFG columns above), because running one at the base model's defaults
produces burnt, over-saturated output. Passing `steps` or `cfg` explicitly in
the request overrides it.

## Pulling

### Models

```bash
oflux pull qwe-2511                  # a curated name
oflux pull qwe-2511 flux.2-klein     # several at a time
oflux pull city96/FLUX.1-dev-gguf           # any Hugging Face repo
```

| Flag | Alias | Does |
|------|-------|------|
| `--quant <label>` | `-q` | quantization preference, e.g. `Q4_K_M` |
| `--file <path-in-repo>` | `-f` | pin exact weights in a repo that publishes many builds |
| `--as <name>` | | install under a different name |
| `--control-net <org/repo>` | `--controlnet` | attach a ControlNet, loaded with the model |
| `--control-net-file <path>` | `--controlnet-file` | pick one from a multi-file ControlNet repo |

`--file` pins one build out of a repo with dozens:

```bash
oflux pull Novice25/Qwen-Image-Edit-Rapid-AIO-GGUF \
  --file v23/Qwen-Rapid-NSFW-v23_Q4_K.gguf --as qwen-rapid
```

`--as`, `--file` and `--control-net` each name or reshape **one** install, so
they are rejected when several models are given on one command line.
`--control-net-file` needs `--control-net`. Curated models ignore `--file`;
their weights are already pinned.

Weights whose filename marks them step-distilled — an explicit count like
`…-4steps…`, or the `lightning` and `rapid` families — are installed with
few-step sampling defaults (cfg 1.0 at the detected step count) instead of the
base architecture's, and the pull output says so. Sampling defaults are baked
into the launch flags at install time, which is why this is decided here and not
per request.

`oflux run <name>` is `pull` plus a printed example call, and installs only if
the model is missing. It takes the same flags, but one model at a time.

### LoRAs

```bash
oflux lora pull qwen-edit-lightning-4step             # a curated name
oflux lora pull <org>/<repo>                          # any repo with one .safetensors
oflux lora pull <org>/<repo> --file <path-in-repo> --as my-lora
oflux lora pull <org>/<repo> --for qwen-image-edit --steps 4 --cfg 1.0
```

| Flag | Alias | Does |
|------|-------|------|
| `--file <path-in-repo>` | `-f` | pick one adapter from a multi-adapter repo |
| `--as <name>` | `-a` | install under a different name |
| `--for <model-or-arch>` | | which model or architecture it is for; repeatable |
| `--steps <n>` | | the step count it is distilled for |
| `--cfg <x>` | `--guidance` | the cfg scale to sample it at |

A repo with exactly one `.safetensors` needs no `--file`; with more than one,
oflux lists the candidates and asks you to pick. Without `--as`, the name is
derived from the filename. Adapters must be safetensors.

`--for`, `--steps` and `--cfg` are recorded in the adapter's sidecar. A curated
LoRA already knows all three; for an arbitrary repo they are the only thing
oflux will ever know about what the adapter is for and the regime it needs.

Remove things with `oflux rm <name>...` and `oflux lora rm <name>...`.

## ControlNet

sd-server can only load a ControlNet **at engine startup** and offers no way to
swap one over HTTP, so it belongs to the installed model rather than to a
request:

```bash
oflux pull <org>/<sd15-repo> \
  --control-net lllyasviel/control_v11p_sd15_canny --as sd15-canny
```

`control_image` and `control_strength` then work on that model. Sending
`control_image` to a model installed without one is a 400, not a silently
ignored field.

> Caveat: stable-diffusion.cpp documents ControlNet for **SD 1.5 only** — none
> of the curated models above support it.

## Calling it

```bash
curl localhost:11534/v1/edit -d '{
  "model": "qwe-2511",
  "prompt": "make it night time",
  "image": "data:image/png;base64,...",
  "loras": [{"name": "qwen-edit-lightning-4step", "scale": 1.0}]
}'
```

Drop `"image"` and POST to `/v1/generate` to generate from text alone. Both
return `{"images": ["<base64 png>"]}`.

`loras` is an array of `{"name": ..., "scale": ...}`; `scale` is the adapter
multiplier and defaults to 1.0 when omitted. The adapter has to be installed
first — an unknown name is a 404 telling you which `oflux lora pull` to run.

There is also an OpenAI-compatible surface — `POST /v1/images/edits`
(multipart) and `POST /v1/images/generations` (JSON) — and the management API
`/api/pull`, `/api/tags`, `/api/delete`, `/api/ps`, `/api/loras`,
`/api/loras/pull`, `/api/loras/delete`. See the project README for the full
request surface.
