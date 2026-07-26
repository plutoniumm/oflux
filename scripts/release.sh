#!/usr/bin/env bash
# Build a properly signed + notarized release: oflux.app -> .dmg + .zip with a
# SHA256SUMS file. With --publish, create the GitHub release and upload them.
#
# Requires a "Developer ID Application" certificate and notary credentials
# (see README "Signing"). Refuses to build an ad-hoc release, because the
# updater verifies the publisher's Team ID before installing a new build.
#
#   VERSION=1.0.0 ./scripts/release.sh
#   VERSION=1.0.0 ./scripts/release.sh --publish
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
VERSION="${VERSION:-1.0.0}"
PUBLISH=0; [ "${1:-}" = "--publish" ] && PUBLISH=1

# `|| true`: with `set -e`+pipefail a no-match grep would abort the script
# before the explanatory message below could be printed.
IDENTITY="${SIGN_IDENTITY:-$(security find-identity -v -p codesigning 2>/dev/null \
  | grep 'Developer ID Application' | head -1 | sed -E 's/.*"(.*)".*/\1/' || true)}"
if [ -z "$IDENTITY" ]; then
  echo "error: no 'Developer ID Application' identity found." >&2
  echo "       A release must be Developer-ID signed and notarized — see README 'Signing'." >&2
  echo "       (For a local build instead, use: make install)" >&2
  exit 1
fi

[ -x third_party/sd-server ] || ./scripts/fetch-engine.sh

VERSION="$VERSION" ./scripts/build-app.sh
SIGN_IDENTITY="$IDENTITY" ./scripts/sign-app.sh

# Staple the .app itself before packaging, so both artifacts carry the ticket.
VERSION="$VERSION" ./scripts/build-dmg.sh
DMG="dist/oflux-$VERSION.dmg"
./scripts/notarize.sh "$DMG"

# The stapled ticket lives on the .app inside the disk image; re-stapling the
# app directly lets the .zip carry it too.
xcrun stapler staple dist/oflux.app || true
ZIP="dist/oflux-$VERSION-macos-arm64.zip"
rm -f "$ZIP"
( cd dist && ditto -c -k --keepParent oflux.app "$(basename "$ZIP")" )

# Checksums let anyone verify a download independently of Apple's ticket.
( cd dist && shasum -a 256 "$(basename "$DMG")" "$(basename "$ZIP")" > SHA256SUMS )

echo "==> artifacts:"
echo "    $DMG"
echo "    $ZIP"
echo "    dist/SHA256SUMS"
echo "==> signed by: $IDENTITY"

if [ "$PUBLISH" = 1 ]; then
  command -v gh >/dev/null || { echo "error: gh CLI required to publish" >&2; exit 1; }
  notes="macOS (Apple Silicon) menu-bar build with the bundled Metal sd-server engine. Signed with a Developer ID and notarized by Apple, so it opens normally. Install: open the .dmg and drag oflux.app to Applications; the fox appears in your menu bar and models download on demand. Verify downloads against SHA256SUMS."
  gh release create "v$VERSION" "$DMG" "$ZIP" dist/SHA256SUMS \
    --title "oflux v$VERSION" --notes "$notes"
  echo "==> published v$VERSION"
fi
