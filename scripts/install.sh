#!/usr/bin/env bash
# Build + sign oflux.app, install it to /Applications (or ~/Applications), and
# register the login LaunchAgent — which also starts the menu-bar app now.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

# Stop the login agent first so KeepAlive doesn't relaunch the app mid-reinstall
# (which would hold the bundle open and race the swap), then stop any strays.
launchctl bootout "gui/$(id -u)/ch.manav.oflux" 2>/dev/null || true
pkill -f "oflux serve"   2>/dev/null || true
pkill -f "oflux menubar" 2>/dev/null || true
sleep 1

VERSION="${VERSION:-1.0.0}" ./scripts/build-app.sh
./scripts/sign-app.sh

# Choose an install dir we can write without sudo.
if [ -w /Applications ]; then DEST="/Applications/oflux.app"; else
  mkdir -p "$HOME/Applications"; DEST="$HOME/Applications/oflux.app"
fi
echo "==> installing to $DEST"
rm -rf "$DEST"
ditto "$ROOT/dist/oflux.app" "$DEST"
xattr -c "$DEST" 2>/dev/null || true
codesign --verify "$DEST" 2>/dev/null && echo "    signature OK at $DEST"

echo "==> registering LaunchAgent + launching menu-bar app"
"$DEST/Contents/MacOS/oflux" install

# The agent starts `oflux menubar`, which serves on :11534.
if curl -s --retry 40 --retry-connrefused --retry-delay 1 http://127.0.0.1:11534/healthz >/dev/null; then
  echo "==> oflux is running — fox is in your menu bar, daemon on http://127.0.0.1:11534"
else
  echo "==> installed, but the daemon isn't answering yet; check: launchctl print gui/$(id -u)/ch.manav.oflux"
fi
