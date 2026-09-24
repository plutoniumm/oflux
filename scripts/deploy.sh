#!/usr/bin/env bash
# One command to cut a release: choose the version, gate on the test suite,
# build + sign + notarize, publish to GitHub, and push the docs site.
#
# VERSION is the single source of truth for the stamped version; the tag and the
# GitHub release are derived from it, so this script is the only place the three
# can get out of step.
#
#   ./scripts/deploy.sh            # interactive
#   NEXT=1.4.0 ./scripts/deploy.sh # scripted
#   DRY=1 ./scripts/deploy.sh      # print the plan, change nothing
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DRY="${DRY:-0}"
RELEASE_NAME="${RELEASE_NAME:-$(grep -m1 '^RELEASE_NAME' Makefile | sed 's/.*?= *//')}"

say()  { printf '\033[1m==>\033[0m %s\n' "$*"; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }
run()  { if [ "$DRY" = 1 ]; then printf '   would run: %s\n' "$*"; else "$@"; fi; }

# ---- where are we -----------------------------------------------------------
cur="$(cat VERSION)"
branch="$(git rev-parse --abbrev-ref HEAD)"
last_tag="$(git describe --tags --abbrev=0 2>/dev/null || echo 'none')"
last_rel="$(gh release view --json tagName -q .tagName 2>/dev/null || echo 'none')"

say "current state"
printf '    VERSION file : %s\n' "$cur"
printf '    last git tag : %s\n' "$last_tag"
printf '    last release : %s\n' "$last_rel"
printf '    branch       : %s\n' "$branch"
changed=$(git status --porcelain | wc -l | tr -d ' ')
printf '    uncommitted  : %s file(s)\n' "$changed"

# ---- pick the next version --------------------------------------------------
IFS=. read -r MA MI PA <<<"$cur"
case "$MA$MI$PA" in *[!0-9]*) die "VERSION $cur is not x.y.z" ;; esac
patch="$MA.$MI.$((PA + 1))"
minor="$MA.$((MI + 1)).0"
major="$((MA + 1)).0.0"

next="${NEXT:-}"
if [ -z "$next" ]; then
  echo
  printf '    1) patch  %s\n    2) minor  %s\n    3) major  %s\n    4) keep   %s (re-publish)\n' \
    "$patch" "$minor" "$major" "$cur"
  printf 'next version [1-4 or x.y.z]: '
  read -r pick
  case "$pick" in
    1|"") next="$patch" ;;
    2) next="$minor" ;;
    3) next="$major" ;;
    4) next="$cur" ;;
    *) next="$pick" ;;
  esac
fi
case "$next" in [0-9]*.[0-9]*.[0-9]*) ;; *) die "'$next' is not x.y.z" ;; esac
tag="v$next"

# ---- refuse early, not after a 5-minute notarization ------------------------
say "pre-flight"
[ "$branch" = main ] || die "on branch '$branch'; releases are cut from main"
command -v gh >/dev/null || die "gh CLI required"
gh auth status >/dev/null 2>&1 || die "gh is not authenticated — run: gh auth login"
gh release view "$tag" >/dev/null 2>&1 && die "a GitHub release for $tag already exists (gh release delete $tag)"
# Credentials are only needed if we actually have to notarize. A previous run
# may have left a stapled artifact for this exact version — reuse it rather than
# demanding an unlocked keychain. (notarytool reports a locked login keychain as
# "no password item found", indistinguishable from creds never being stored.)
if [ -f "dist/oflux-$next.dmg" ] && xcrun stapler validate "dist/oflux-$next.dmg" >/dev/null 2>&1; then
  echo "    dist/oflux-$next.dmg is already notarized — skipping the credential check"
elif [ "$DRY" = 1 ]; then
  ./scripts/notarize.sh --check || echo "    (dry run: notary credentials unavailable — a real run would stop here)"
else
  ./scripts/notarize.sh --check || die "notarization credentials unavailable. If they were working earlier, the login keychain has locked — unlock it (open Keychain Access, or log in again) and retry."
fi
echo "    $tag is free"

say "releasing $cur -> $next as $tag ($RELEASE_NAME)"
if [ "$DRY" != 1 ] && [ -z "${NEXT:-}" ]; then
  printf 'proceed? [y/N]: '
  read -r yes
  case "$yes" in y | Y | yes) ;; *) die "aborted" ;; esac
fi

# ---- gate: a broken build must never reach a tag ----------------------------
say "tests (both build modes)"
if [ "$DRY" = 1 ]; then
  echo "   would run: SAM3=0 go test ./... && go test -tags sam3 ./..."
else
  gofmt -l . | grep -v third_party && die "gofmt: files need formatting"
  go vet ./... || die "go vet failed"
  SAM3=0 go test ./... >/dev/null || die "tests failed (SAM3=0)"
  go test -tags sam3 ./... >/dev/null || die "tests failed (-tags sam3)"
  ./scripts/docs.sh build
  echo "    all green"
fi

# ---- stamp and build BEFORE anything becomes public --------------------------
# Order matters: notarization takes minutes and needs an unlocked keychain, so
# it must fail while the tag is still local. Pushing first left v1.3.1 tagged
# on the remote with no release attached.
say "stamping VERSION"
[ "$DRY" = 1 ] || echo "$next" > VERSION

dmg="dist/oflux-$next.dmg"
zip="dist/oflux-$next-macos-arm64.zip"
if [ -f "$dmg" ] && [ -f "$zip" ] && xcrun stapler validate "$dmg" >/dev/null 2>&1; then
  say "reusing already-notarized artifacts for $next"
else
  say "build + notarize (several minutes)"
  run env VERSION="$next" RELEASE_NAME="$RELEASE_NAME" ./scripts/release.sh --no-publish
fi

say "committing and pushing"
run git add -A
if [ "$DRY" = 1 ] || ! git diff --cached --quiet; then
  run git commit -m "oflux $next"
else
  echo "    nothing to commit"
fi
run git tag -a "$tag" -m "$RELEASE_NAME"
run git push origin "$branch"
run git push origin "$tag"

say "publishing $tag"
title="oflux $tag"
[ -n "$RELEASE_NAME" ] && title="$title — $RELEASE_NAME"
# NOTES_FILE, else RELEASE_NOTES.md if present, else the generic blurb. One
# commit per release makes --generate-notes useless here.
notes_file="${NOTES_FILE:-RELEASE_NOTES.md}"
if [ -f "$notes_file" ]; then
  say "using release notes from $notes_file"
  run gh release create "$tag" "$dmg" "$zip" dist/SHA256SUMS --title "$title" --notes-file "$notes_file"
  [ "$DRY" = 1 ] || rm -f "$notes_file"
else
  run gh release create "$tag" "$dmg" "$zip" dist/SHA256SUMS --title "$title" \
    --notes "macOS (Apple Silicon) menu-bar build with the bundled Metal sd-server engine. Signed with a Developer ID and notarized by Apple, so it opens normally. Install: open the .dmg and drag oflux.app to Applications; the fox appears in your menu bar and models download on demand. Verify downloads against SHA256SUMS."
fi

say "docs -> gh-pages"
run ./scripts/docs.sh deploy

say "done: $tag published"
[ "$DRY" = 1 ] && echo "    (dry run — nothing changed)"
exit 0
