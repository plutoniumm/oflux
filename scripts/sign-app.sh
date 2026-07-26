#!/usr/bin/env bash
# Sign dist/oflux.app for LOCAL use — no notarization, no Apple round-trip.
#
# Identity selection (first that applies):
#   1. $SIGN_IDENTITY if set
#   2. an installed "Developer ID Application" or "Apple Development" cert
#   3. ad-hoc ("-")  <- needs nothing; runs on the machine that built it
#
# Locally-built apps have no quarantine attribute, so ad-hoc is enough to launch
# them here. To hand the .app to another Mac, that user right-clicks > Open once.
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

# codesign rejects bundles carrying com.apple.FinderInfo, which iCloud-managed
# folders (this repo may be one) keep re-stamping on the .app directory faster
# than we can clear it. So sign a clean temp copy (xattrs stripped via ditto),
# then copy the signed bundle back — the code seal covers Contents/, so a dir
# xattr re-added by iCloud afterward is harmless to verification.
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
STAGE="$WORK/oflux.app"
ditto --norsrc --noextattr --noqtn "$APP" "$STAGE"

if [ "$IDENTITY" = "-" ]; then
  # Ad-hoc: a single deep sign is simplest and runs on this machine as-is.
  codesign --force --deep --sign - "$STAGE"
else
  # Real cert: sign inside-out with hardened runtime + entitlements.
  while IFS= read -r -d '' f; do
    echo "    sign $f"
    codesign --force --options runtime --timestamp --sign "$IDENTITY" "$f"
  done < <(find "$STAGE/Contents/Resources" -type f \( -name '*.dylib' -o -name 'sd-server' \) -print0)
  codesign --force --options runtime --timestamp --entitlements "$ENT" --sign "$IDENTITY" "$STAGE/Contents/MacOS/oflux"
  codesign --force --options runtime --timestamp --entitlements "$ENT" --sign "$IDENTITY" "$STAGE"
fi
codesign --verify --deep --strict "$STAGE"

# Swap the signed bundle back into place. The strict seal was already verified
# on the clean temp copy above; here we use a basic verify because iCloud keeps
# re-stamping a (cosmetic) FinderInfo xattr on the .app directory that --strict
# treats as detritus. Basic verify confirms the code seal is intact regardless.
rm -rf "$APP"
ditto "$STAGE" "$APP"
xattr -c "$APP" 2>/dev/null || true
codesign -v "$APP"
echo "==> signed and verified."
if [ "$IDENTITY" = "-" ]; then
  # NOTE: this must not be a bare `[ … ] && echo`, whose non-zero exit under
  # `set -e` would abort the caller (install.sh) whenever a real cert is used.
  echo "    (ad-hoc; for other Macs, right-click > Open the first time.)"
else
  echo "    Developer ID signed. Notarize before distributing: make notarize ART=<file>"
fi
