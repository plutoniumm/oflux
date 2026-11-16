#!/usr/bin/env bash
# Notarize and staple a .dmg/.zip. `--check` verifies credentials only.
#
#   ./scripts/notarize.sh dist/oflux-1.1.0.dmg
#
# Creds: keychain profile $NOTARY_PROFILE, else APPLE_ID + TEAM_ID + APP_PASSWORD
# (NOTARY_-prefixed names also work).
set -euo pipefail

PROFILE="${NOTARY_PROFILE:-oflux-notary}"
APPLE="${NOTARY_APPLE_ID:-${APPLE_ID:-}}"
TEAM="${NOTARY_TEAM_ID:-${TEAM_ID:-}}"
PASS="${NOTARY_PASSWORD:-${APP_PASSWORD:-}}"

auth_args() {
  if [ -n "$APPLE" ] && [ -n "$TEAM" ] && [ -n "$PASS" ]; then
    printf '%s\n' --apple-id "$APPLE" --team-id "$TEAM" --password "$PASS"
  else
    printf '%s\n' --keychain-profile "$PROFILE"
  fi
}

# Asks Apple rather than reading the keychain: notarytool stores saved creds
# where the `security` CLI cannot see them, so a keychain lookup reports
# "missing" for a profile that works.
have_creds() {
  local auth=()
  while IFS= read -r a; do auth+=("$a"); done < <(auth_args)
  APPLE_SAYS="$(xcrun notarytool history "${auth[@]}" 2>&1 >/dev/null)"
}

no_creds_help() {
  cat >&2 <<EOF
error: notarization credentials are missing or not accepted by Apple.

  Store them once (an app-specific password from appleid.apple.com):

    xcrun notarytool store-credentials $PROFILE \\
      --apple-id "you@example.com" --team-id "TEAMID" --password "<app-specific-password>"

  Or export APPLE_ID, TEAM_ID and APP_PASSWORD for this run.
EOF
}

if [ "${1:-}" = "--check" ]; then
  # Apple's own words first: a lapsed developer agreement is a 403 with valid
  # credentials, and the store-credentials advice alone sent us re-saving a
  # password that was fine (1.5.1).
  have_creds || { echo "${APPLE_SAYS:-}" >&2; no_creds_help; exit 1; }
  echo "==> notarization credentials OK"
  exit 0
fi

ART="${1:-}"
[ -f "$ART" ] || { echo "usage: $0 <path to .dmg or .zip> | --check" >&2; exit 1; }

AUTH=()
while IFS= read -r a; do AUTH+=("$a"); done < <(auth_args)

echo "==> submitting $ART to Apple (this usually takes 1-5 minutes)"
xcrun notarytool submit "$ART" "${AUTH[@]}" --wait

case "$ART" in
  *.dmg)
    xcrun stapler staple "$ART"
    xcrun stapler validate "$ART"
    echo "==> notarized + stapled: $ART"
    ;;
  *)
    # A .zip cannot be stapled; the .app inside carries the ticket.
    echo "==> notarized: $ART (staple the .app before zipping)"
    ;;
esac
