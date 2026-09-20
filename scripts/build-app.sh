#!/usr/bin/env bash
# Build dist/oflux.app — menu-bar (LSUIElement) bundle + the Metal engine.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="${VERSION:-$(cat "$ROOT/VERSION")}"
APP="$ROOT/dist/oflux.app"
MACOS="$APP/Contents/MacOS"
RES="$APP/Contents/Resources"

echo "==> building oflux.app $VERSION"
rm -rf "$APP"
mkdir -p "$MACOS" "$RES"

# SAM3=1 links the SAM3 segmenter into the binary. Plain string, not a bash
# array: macOS ships bash 3.2, where "${A[@]}" on an empty array trips set -u.
TAGS=""
if [ "${SAM3:-1}" = "1" ]; then
  TAGS="-tags sam3"
  echo "==> linking SAM3 (static, from third_party/sam3-darwin-arm64)"
fi

# One binary for both: Info.plist sets OFLUX_LAUNCH=menubar, so double-clicking
# opens the menu bar while a terminal invocation is the CLI.
CGO_ENABLED=1 go build -trimpath $TAGS -ldflags "-X oflux/internal/version.Version=$VERSION" -o "$MACOS/oflux" ./cmd/oflux

sed "s/__VERSION__/$VERSION/g" "$ROOT/packaging/Info.plist" > "$APP/Contents/Info.plist"

# Regenerate the icon from the SVG if missing (needs rsvg-convert).
[ -f "$ROOT/packaging/oflux.icns" ] || "$ROOT/scripts/gen-icons.sh" || true
[ -f "$ROOT/packaging/oflux.icns" ] && cp "$ROOT/packaging/oflux.icns" "$RES/oflux.icns" && echo "==> bundled app icon"

# sd-server links @rpath/libstable-diffusion.dylib, resolved via
# @executable_path, so its colocated dylibs must come along.
ENGINE="${OFLUX_ENGINE:-$ROOT/third_party/sd-server}"
if [ -x "$ENGINE" ]; then
  cp "$ENGINE" "$RES/sd-server"
  cp "$(dirname "$ENGINE")"/*.dylib "$RES"/ 2>/dev/null && echo "==> bundled engine libs" || true
  echo "==> bundled engine: $ENGINE"
else
  echo "==> WARNING: no sd-server engine bundled"
  echo "    run 'make engine' (or set OFLUX_ENGINE), then rebuild."
fi

echo "==> built $APP"
echo "    run: open '$APP'    (or: '$MACOS/oflux' serve)"
