#!/usr/bin/env bash
# Package dist/oflux.app into a compressed .dmg. Run build-app.sh first.
set -euo pipefail

VERSION="${VERSION:-1.0.0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$ROOT/dist/oflux.app"
DMG="$ROOT/dist/oflux-$VERSION.dmg"

[ -d "$APP" ] || { echo "error: $APP not found — run scripts/build-app.sh first" >&2; exit 1; }

echo "==> creating $DMG"
rm -f "$DMG"
STAGE="$(mktemp -d)"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
hdiutil create -volname "oflux" -srcfolder "$STAGE" -ov -format UDZO "$DMG" >/dev/null
rm -rf "$STAGE"
echo "==> built $DMG"
echo "    (unsigned — first launch needs right-click > Open, or a Developer ID signature + notarization)"
