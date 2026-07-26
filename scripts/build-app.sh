#!/usr/bin/env bash
# Build dist/oflux.app — a macOS menu-bar (LSUIElement) app bundle wrapping the
# oflux binary and, when available, the Metal sd-server engine.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="${VERSION:-$(cat "$ROOT/VERSION")}"
APP="$ROOT/dist/oflux.app"
MACOS="$APP/Contents/MacOS"
RES="$APP/Contents/Resources"

echo "==> building oflux.app $VERSION"
rm -rf "$APP"
mkdir -p "$MACOS" "$RES"

# The CLI/daemon/menu-bar binary. Double-clicking the .app opens the menu-bar
# UI because Info.plist sets OFLUX_LAUNCH=menubar; from a terminal it's the CLI.
CGO_ENABLED=1 go build -trimpath -ldflags "-X oflux/internal/version.Version=$VERSION" -o "$MACOS/oflux" ./cmd/oflux

# Info.plist (version substituted).
sed "s/__VERSION__/$VERSION/g" "$ROOT/packaging/Info.plist" > "$APP/Contents/Info.plist"

# App icon. Regenerate from the SVG if missing (needs rsvg-convert).
[ -f "$ROOT/packaging/oflux.icns" ] || "$ROOT/scripts/gen-icons.sh" || true
[ -f "$ROOT/packaging/oflux.icns" ] && cp "$ROOT/packaging/oflux.icns" "$RES/oflux.icns" && echo "==> bundled app icon"

# Bundle the engine if we can find it, plus its colocated shared libraries
# (sd-server links @rpath/libstable-diffusion.dylib, resolved via @executable_path).
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
