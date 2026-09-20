#!/usr/bin/env bash
# Cut a release: build, sign, notarize, package, publish to GitHub.
# RELEASE_NAME is an optional codename for the release title.
#
#   VERSION=1.1.0 ./scripts/release.sh [--no-publish]
#
# Developer ID only — the updater checks the publisher's Team ID, so an ad-hoc
# build would strand everyone who installed it.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
VERSION="${VERSION:-$(cat "$ROOT/VERSION")}"
RELEASE_NAME="${RELEASE_NAME:-}"
TAG="v$VERSION"
PUBLISH=1; [ "${1:-}" = "--no-publish" ] && PUBLISH=0

# `|| true`: a no-match grep would otherwise abort before the message below.
IDENTITY="${SIGN_IDENTITY:-$(security find-identity -v -p codesigning 2>/dev/null \
  | grep 'Developer ID Application' | head -1 | sed -E 's/.*"(.*)".*/\1/' || true)}"
if [ -z "$IDENTITY" ]; then
  echo "error: no 'Developer ID Application' identity found." >&2
  echo "       A release must be Developer-ID signed and notarized — see README 'Signing'." >&2
  echo "       (For a local build instead, use: make install)" >&2
  exit 1
fi

# Everything that can fail is checked before the first byte is built:
# notarization takes minutes and publishing comes after it, so a missing
# credential or an already-taken tag must not surface at the very end.
./scripts/notarize.sh --check

if [ "$PUBLISH" = 1 ]; then
  command -v gh >/dev/null || { echo "error: gh CLI required to publish" >&2; exit 1; }
  gh auth status >/dev/null 2>&1 || { echo "error: gh is not authenticated — run: gh auth login" >&2; exit 1; }

  if gh release view "$TAG" >/dev/null 2>&1; then
    echo "error: a GitHub release for $TAG already exists." >&2
    echo "       Bump VERSION, or delete it: gh release delete $TAG" >&2
    exit 1
  fi
  git rev-parse -q --verify "refs/tags/$TAG" >/dev/null || {
    echo "error: tag $TAG does not exist locally — create it: git tag -a $TAG -m \"$RELEASE_NAME\"" >&2
    exit 1
  }
  # gh would otherwise create the tag from whatever the remote's default branch
  # points at, which is not necessarily the commit being released.
  BRANCH="$(git rev-parse --abbrev-ref HEAD)"
  if ! git merge-base --is-ancestor "$TAG" "@{upstream}" 2>/dev/null; then
    echo "error: $TAG is not on the remote yet." >&2
    echo "       Push it first: git push origin $BRANCH && git push origin $TAG" >&2
    exit 1
  fi
  echo "==> publishing as: $TAG"
fi

[ -x third_party/sd-server ] || ./scripts/fetch-engine.sh
[ -f third_party/sam3-darwin-arm64/lib/libsam3.a ] || ./scripts/fetch-sam3.sh

VERSION="$VERSION" ./scripts/build-app.sh
SIGN_IDENTITY="$IDENTITY" ./scripts/sign-app.sh

VERSION="$VERSION" ./scripts/build-dmg.sh
DMG="dist/oflux-$VERSION.dmg"
./scripts/notarize.sh "$DMG"

# Staple the .app too, so the .zip carries a ticket as well as the .dmg.
xcrun stapler staple dist/oflux.app || true
ZIP="dist/oflux-$VERSION-macos-arm64.zip"
rm -f "$ZIP"
( cd dist && ditto -c -k --keepParent oflux.app "$(basename "$ZIP")" )

( cd dist && shasum -a 256 "$(basename "$DMG")" "$(basename "$ZIP")" > SHA256SUMS )

echo "==> artifacts:"
echo "    $DMG"
echo "    $ZIP"
echo "    dist/SHA256SUMS"
echo "==> signed by: $IDENTITY"

if [ "$PUBLISH" = 1 ]; then
  notes="macOS (Apple Silicon) menu-bar build with the bundled Metal sd-server engine. Signed with a Developer ID and notarized by Apple, so it opens normally. Install: open the .dmg and drag oflux.app to Applications; the fox appears in your menu bar and models download on demand. Verify downloads against SHA256SUMS."
  title="oflux $TAG"
  [ -n "$RELEASE_NAME" ] && title="$title — $RELEASE_NAME"
  gh release create "$TAG" "$DMG" "$ZIP" dist/SHA256SUMS \
    --title "$title" --notes "$notes"
  echo "==> published $TAG"
else
  echo "==> not published (--no-publish)"
fi
