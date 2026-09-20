#!/usr/bin/env bash
# Acquire the SAM3 segmentation library into third_party/sam3-darwin-arm64.
# Tries the prebuilt macOS-arm64 release first; falls back to a source build.
#
#   ./scripts/fetch-sam3.sh                       libs + headers only
#   ./scripts/fetch-sam3.sh --model               ... plus the default checkpoint
#   ./scripts/fetch-sam3.sh --model sam3-q8_0     ... a specific one
#
# The daemon fetches weights itself through sam3.EnsureModel; --model is the
# shell equivalent, for setting a machine up ahead of time. It is opt-in
# because the smallest usable checkpoint is 707 MB.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party"
OUT="$DEST/sam3-darwin-arm64"
REPO="PABannier/sam3.cpp"
HF_REPO="PABannier/sam3.cpp"
DEFAULT_MODEL="sam3-q4_0"

WANT_MODEL=""
case "${1:-}" in
  --model) WANT_MODEL="${2:-$DEFAULT_MODEL}" ;;
  "") ;;
  *) echo "usage: $0 [--model [name]]" >&2; exit 2 ;;
esac

[ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ] || {
  echo "error: oflux is Apple Silicon only; this is $(uname -s)/$(uname -m)" >&2
  exit 1
}

libs_present() {
  local f
  for f in include/sam3.h lib/libsam3.a lib/libggml-base.a lib/libggml-metal.a; do
    [ -f "$OUT/$f" ] || return 1
  done
  return 0
}

fetch_libs() {
  if libs_present; then
    echo "==> sam3 libs already in $OUT"
    return 0
  fi
  mkdir -p "$DEST"
  echo "==> resolving latest $REPO release"
  local api url tmp
  api="https://api.github.com/repos/$REPO/releases/latest"
  url="$(curl -fsSL "$api" \
    | grep browser_download_url \
    | grep -iE 'darwin|macos' | grep -i arm64 \
    | head -1 | cut -d'"' -f4 || true)"

  if [ -n "${url:-}" ]; then
    echo "==> downloading prebuilt: $url"
    tmp="$(mktemp -d)"
    if curl -fsSL -o "$tmp/sam3.tar.gz" "$url" && tar xzf "$tmp/sam3.tar.gz" -C "$tmp"; then
      # The archive already carries the sam3-darwin-arm64/{include,lib,bin}
      # layout the cgo build's ${SRCDIR}-relative flags expect.
      local src
      src="$(find "$tmp" -maxdepth 3 -type d -name 'include' | head -1 || true)"
      if [ -n "$src" ]; then
        rm -rf "$OUT"
        mkdir -p "$OUT"
        cp -R "$(dirname "$src")"/* "$OUT"/
        rm -rf "$tmp"
        return 0
      fi
      echo "==> prebuilt archive has no include/ directory; building from source"
    fi
    rm -rf "$tmp"
  fi

  echo "==> building sam3.cpp from source with Metal"
  command -v cmake >/dev/null || { echo "error: cmake not found" >&2; exit 1; }
  local src="$DEST/sam3.cpp"
  [ -d "$src/.git" ] || git clone --depth 1 --recursive "https://github.com/$REPO" "$src"
  cmake -S "$src" -B "$src/build" -DCMAKE_BUILD_TYPE=Release \
    -DSAM3_METAL=ON -DSAM3_BUILD_EXAMPLES=OFF >/dev/null
  cmake --build "$src/build" --config Release -j --target sam3

  rm -rf "$OUT"
  mkdir -p "$OUT/lib" "$OUT/include"
  local found=0 a
  while IFS= read -r a; do
    cp "$a" "$OUT/lib/"; found=1
  done < <(find "$src/build" -type f \( -name 'libsam3.a' -o -name 'libggml*.a' \))
  [ "$found" = 1 ] || { echo "error: source build produced no static libraries" >&2; exit 1; }
  cp "$src/sam3.h" "$OUT/include/"
  cp "$src"/ggml/include/*.h "$OUT/include/"
}

verify_libs() {
  if ! libs_present; then
    echo "error: $OUT is incomplete — neither the prebuilt release nor a source" >&2
    echo "       build produced a usable libsam3. Nothing was installed." >&2
    exit 1
  fi
  echo "==> sam3 libs -> $OUT"
  echo "    build oflux against them with: go build -tags sam3 ./..."
}

fetch_model() {
  # Full sam3-* only. The sam3-visual-* builds drop the text encoder, so the
  # text-prompt path (sam3_segment_pcs) refuses to run on them.
  local name="${1%.ggml}"
  case "$name" in
    sam3-f32|sam3-f16|sam3-q8_0|sam3-q4_1|sam3-q4_0) ;;
    sam3-visual-*)
      echo "error: $name has no text encoder; oflux segments from a text prompt." >&2
      echo "       Use one of: sam3-q4_0 sam3-q4_1 sam3-q8_0 sam3-f16 sam3-f32" >&2
      exit 1 ;;
    *)
      echo "error: unknown checkpoint '$name'." >&2
      echo "       Known: sam3-q4_0 sam3-q4_1 sam3-q8_0 sam3-f16 sam3-f32" >&2
      exit 1 ;;
  esac

  local dir="${OFLUX_SAM3_DIR:-${OFLUX_HOME:-$HOME/.oflux}/sam3}"
  mkdir -p "$dir"
  local out="$dir/$name.ggml"
  if [ -s "$out" ]; then
    echo "==> $out already present"
    return 0
  fi
  local url="https://huggingface.co/$HF_REPO/resolve/main/$name.ggml"
  echo "==> downloading $name.ggml (this is hundreds of MB)"
  # .part so an interrupted download is never mistaken for a usable model.
  curl -fL --retry 3 --progress-bar -o "$out.part" "$url"
  mv "$out.part" "$out"
  echo "==> installed checkpoint -> $out"
}

fetch_libs
verify_libs
# Must stay an if: a bare `[ … ] && cmd` exits non-zero under `set -e` when the
# test fails, which would make a libs-only run look like a failure.
if [ -n "$WANT_MODEL" ]; then
  fetch_model "$WANT_MODEL"
fi
