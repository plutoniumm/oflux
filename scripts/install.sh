#!/usr/bin/env bash
# Build + sign oflux.app, install it, and register the login LaunchAgent.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

# Stop the agent first, or KeepAlive relaunches the app mid-swap and holds the
# bundle open.
for L in io.github.plutoniumm.oflux ch.manav.oflux; do
  launchctl bootout "gui/$(id -u)/$L" 2>/dev/null || true
done
pkill -f "oflux serve"   2>/dev/null || true
pkill -f "oflux menubar" 2>/dev/null || true
sleep 1

VERSION="${VERSION:-$(cat VERSION)}" ./scripts/build-app.sh
./scripts/sign-app.sh

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

if curl -s --retry 40 --retry-connrefused --retry-delay 1 http://127.0.0.1:11534/healthz >/dev/null; then
  echo "==> oflux is running — fox is in your menu bar, daemon on http://127.0.0.1:11534"
else
  echo "==> installed, but the daemon isn't answering yet; check: launchctl print gui/$(id -u)/io.github.plutoniumm.oflux"
fi
