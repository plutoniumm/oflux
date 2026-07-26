#!/usr/bin/env bash
# Acquire a Metal-capable sd-server engine into third_party/sd-server.
# Tries the prebuilt macOS-arm64 release first; falls back to a source build
# with -DSD_METAL=ON.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party"
OUT="$DEST/sd-server"
mkdir -p "$DEST"

echo "==> resolving latest stable-diffusion.cpp release"
API="https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/latest"
URL="$(curl -fsSL "$API" \
  | grep browser_download_url \
  | grep -iE 'darwin|macos' | grep -i arm64 \
  | head -1 | cut -d'"' -f4 || true)"

if [ -n "${URL:-}" ]; then
  echo "==> downloading prebuilt: $URL"
  tmp="$(mktemp -d)"
  if curl -fsSL -o "$tmp/sd.zip" "$URL" && unzip -oq "$tmp/sd.zip" -d "$tmp/x"; then
    bin="$(find "$tmp/x" -type f -name sd-server | head -1 || true)"
    if [ -n "$bin" ]; then
      cp "$bin" "$OUT"; chmod +x "$OUT"
      # colocate any shared libraries the binary needs
      dir="$(dirname "$bin")"
      cp "$dir"/*.dylib "$DEST"/ 2>/dev/null || true
      echo "==> installed prebuilt sd-server -> $OUT"
      "$OUT" --help >/dev/null 2>&1 && echo "==> sd-server runs" || echo "==> WARNING: sd-server --help failed (may need a source build)"
      exit 0
    fi
    echo "==> prebuilt archive has no 'sd-server' target; building from source"
  fi
fi

echo "==> building sd-server from source with Metal (-DSD_METAL=ON)"
SRC="$DEST/stable-diffusion.cpp"
[ -d "$SRC/.git" ] || git clone --depth 1 --recursive https://github.com/leejet/stable-diffusion.cpp "$SRC"
cmake -S "$SRC" -B "$SRC/build" -DSD_METAL=ON -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$SRC/build" --config Release -j --target sd-server
found="$(find "$SRC/build" -type f -name sd-server | head -1)"
cp "$found" "$OUT"; chmod +x "$OUT"
echo "==> built sd-server (Metal) -> $OUT"
