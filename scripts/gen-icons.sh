#!/usr/bin/env bash
# Rasterize packaging/oflux.svg into the app icon (oflux.icns) and the menu-bar
# PNG (internal/menubar/icon.png). Needs rsvg-convert (brew install librsvg).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PKG="$ROOT/packaging"
SVG="$PKG/oflux.svg"           # full logo (beige rounded-rect bg) -> app icon
MSVG="$PKG/oflux-menubar.svg"  # fox only, transparent -> menu bar

command -v rsvg-convert >/dev/null || { echo "error: need rsvg-convert (brew install librsvg)" >&2; exit 1; }

# --- app icon (.icns) ---
WORK="$(mktemp -d)"; ICONSET="$WORK/oflux.iconset"; mkdir -p "$ICONSET"
gen() { rsvg-convert -w "$1" -h "$1" "$SVG" -o "$ICONSET/$2"; }
gen 16   icon_16x16.png
gen 32   icon_16x16@2x.png
gen 32   icon_32x32.png
gen 64   icon_32x32@2x.png
gen 128  icon_128x128.png
gen 256  icon_128x128@2x.png
gen 256  icon_256x256.png
gen 512  icon_256x256@2x.png
gen 512  icon_512x512.png
gen 1024 icon_512x512@2x.png
iconutil -c icns "$ICONSET" -o "$PKG/oflux.icns"
rm -rf "$WORK"
echo "==> wrote $PKG/oflux.icns"

# --- menu-bar icon (color fox on transparent; 44px for retina menu bar) ---
rsvg-convert -w 44 -h 44 "$MSVG" -o "$ROOT/internal/menubar/icon.png"
echo "==> wrote internal/menubar/icon.png"

# --- a 512 PNG for READMEs / general use ---
rsvg-convert -w 512 -h 512 "$SVG" -o "$PKG/oflux-512.png"
echo "==> wrote $PKG/oflux-512.png"
