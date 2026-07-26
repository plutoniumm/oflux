#!/usr/bin/env bash
# Notarize and staple a built artifact (.dmg or .zip) with Apple.
#
# One-time setup (see README "Signing"):
#   xcrun notarytool store-credentials oflux-notary \
#     --apple-id "you@example.com" --team-id "TEAMID" --password "<app-specific-password>"
#
# Usage: ./scripts/notarize.sh dist/oflux-1.0.0.dmg
# Env:   NOTARY_PROFILE (default "oflux-notary"), or NOTARY_APPLE_ID +
#        NOTARY_TEAM_ID + NOTARY_PASSWORD for CI (no keychain profile).
set -euo pipefail

ART="${1:-}"
[ -f "$ART" ] || { echo "usage: $0 <path to .dmg or .zip>" >&2; exit 1; }
PROFILE="${NOTARY_PROFILE:-oflux-notary}"

if [ -n "${NOTARY_APPLE_ID:-}" ] && [ -n "${NOTARY_TEAM_ID:-}" ] && [ -n "${NOTARY_PASSWORD:-}" ]; then
  AUTH=(--apple-id "$NOTARY_APPLE_ID" --team-id "$NOTARY_TEAM_ID" --password "$NOTARY_PASSWORD")
else
  AUTH=(--keychain-profile "$PROFILE")
fi

echo "==> submitting $ART to Apple (this usually takes 1-5 minutes)"
xcrun notarytool submit "$ART" "${AUTH[@]}" --wait

# Stapling attaches the ticket so the artifact validates offline. Only disk
# images and app bundles can be stapled — a .zip cannot, but the .app inside it
# is validated by the ticket Apple issued for it.
case "$ART" in
  *.dmg)
    xcrun stapler staple "$ART"
    xcrun stapler validate "$ART"
    echo "==> notarized + stapled: $ART"
    ;;
  *)
    echo "==> notarized: $ART (zip archives cannot be stapled; staple the .app before zipping)"
    ;;
esac
