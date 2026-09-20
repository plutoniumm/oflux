#!/usr/bin/env bash
# Sign dist/oflux.app. Identity: $SIGN_IDENTITY, else an installed Developer ID
# / Apple Development cert, else ad-hoc.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$ROOT/dist/oflux.app"
ENT="$ROOT/packaging/entitlements.plist"
[ -d "$APP" ] || { echo "error: $APP not found — run 'make app' first" >&2; exit 1; }

IDENTITY="${SIGN_IDENTITY:-}"
if [ -z "$IDENTITY" ]; then
  IDENTITY="$(security find-identity -v -p codesigning \
    | grep -E 'Developer ID Application|Apple Development' \
    | head -1 | sed -E 's/.*"(.*)".*/\1/' || true)"
fi
[ -z "$IDENTITY" ] && IDENTITY="-"
echo "==> signing with identity: $IDENTITY"

# codesign rejects bundles with com.apple.FinderInfo, which iCloud re-stamps on
# the .app faster than we can clear it. Sign a stripped temp copy instead.
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
STAGE="$WORK/oflux.app"
ditto --norsrc --noextattr --noqtn "$APP" "$STAGE"

if [ "$IDENTITY" = "-" ]; then
  codesign --force --deep --sign - "$STAGE"
else
  # Inside-out, hardened runtime + entitlements.
  while IFS= read -r -d '' f; do
    echo "    sign $f"
    codesign --force --options runtime --timestamp --sign "$IDENTITY" "$f"
  done < <(find "$STAGE/Contents/Resources" -type f \( -name '*.dylib' -o -name 'sd-server' \) -print0)
  codesign --force --options runtime --timestamp --entitlements "$ENT" --sign "$IDENTITY" "$STAGE/Contents/MacOS/oflux"
  codesign --force --options runtime --timestamp --entitlements "$ENT" --sign "$IDENTITY" "$STAGE"
fi
codesign --verify --deep --strict "$STAGE"

# Basic (not --strict) verify after the swap: iCloud re-adds the cosmetic
# FinderInfo xattr, which --strict calls detritus. The seal was checked above.
rm -rf "$APP"
ditto "$STAGE" "$APP"
xattr -c "$APP" 2>/dev/null || true
codesign -v "$APP"
echo "==> signed and verified."
if [ "$IDENTITY" = "-" ]; then
  # Must stay an if/else: a bare `[ … ] && echo` exits non-zero with a real
  # cert and aborts the caller under `set -e`.
  echo "    (ad-hoc; for other Macs, right-click > Open the first time.)"
else
  echo "    Developer ID signed. Notarize before distributing: make notarize ART=<file>"
fi
