macOS (Apple Silicon) menu-bar build with the bundled Metal sd-server engine. Signed with a Developer ID and notarized by Apple, so it opens normally. Install: open the .dmg and drag oflux.app to Applications; the fox appears in your menu bar and models download on demand. Verify downloads against SHA256SUMS.

## Model manager in the web UI

The UI at `http://127.0.0.1:11534` now has a **Manage** tab beside Create, so
models no longer need the CLI:

- **Installed** — what you have, with architecture, mode and a delete.
- **Add a model** — search Hugging Face, **Inspect** a repo to see whether it can
  actually run (architecture, mode, available quants, or the specific blockers
  that stop it) *before* downloading anything, then pull with live progress.
- **Adapters** — pull and remove LoRAs.
- **Presets** — create, rename and delete.

New endpoints behind it: `GET /api/search`, `POST /api/inspect`,
`POST /api/presets/rename`.

## Fixes

- Hugging Face search now reports **gated** repos correctly. The Hub omits
  `gated`, `likes` and `lastModified` from a model list unless they are
  expanded, so every repo previously looked ungated and a gated pull failed
  mid-download instead of being flagged up front.
- An adapter name coming off the wire was interpolated into HTML in the LoRA
  chip. It is text now.
- `go test ./...` no longer replaces the machine's `/opt/homebrew/bin/oflux`
  symlink. `selfinstall.LinkCLI` tries the system PATH directories before
  anything under `$HOME`, so sandboxing `HOME` did not contain it; tests now set
  `OFLUX_BIN_DIR`.

## Upgrading from 1.3.0 or earlier

Curated model names changed in 1.3.1 to `<line><version>[-modifier]` —
`qwe-2511`, `qwe-2.1`, `qwe-2.1-uc`, `qwi-1`, `flx-1-dev`, `flx-2-klein-9b`,
`zim-1-turbo`. There are no aliases, so `oflux pull qwen-image-edit` fails.
Models already installed keep their old names until you run, once per machine:

```
DRY=1 ./scripts/migrate-names.sh   # preview
./scripts/migrate-names.sh          # apply, then restart the daemon
```
