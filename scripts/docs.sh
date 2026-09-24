#!/usr/bin/env bash
# docsify is static, so "build" is a consistency check rather than a compile:
# it catches the docs drifting from the curated registry, which is exactly what
# happens after a rename. "deploy" publishes docs/ to the gh-pages branch.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

build() {
  local fail=0
  for f in docs/index.html docs/README.md docs/.nojekyll; do
    [ -f "$f" ] || { echo "missing $f"; fail=1; }
  done

  # Every curated model must appear in the docs table, and every handle in the
  # table must still exist in the registry.
  local reg docs_names
  reg=$(grep -oE 'Name:[[:space:]]+"[^"]+"' internal/registry/registry.go | sed 's/.*"\(.*\)"/\1/' | sort)
  docs_names=$(grep -oE '^\| `[a-z0-9.-]+` \|' docs/README.md | tr -d '|` ' | sort -u)

  while read -r n; do
    [ -z "$n" ] && continue
    grep -qx "$n" <<<"$docs_names" || { echo "registry model '$n' is not documented"; fail=1; }
  done <<<"$reg"

  while read -r n; do
    [ -z "$n" ] && continue
    case "$n" in qwe-*|qwi-*|flx-*|zim-*) ;; *) continue ;; esac
    grep -qx "$n" <<<"$reg" || { echo "docs reference unknown model '$n'"; fail=1; }
  done <<<"$docs_names"

  [ "$fail" = 0 ] && echo "docs ok: $(wc -l < docs/README.md) lines, $(wc -l <<<"$reg") curated models"
  return "$fail"
}

deploy() {
  build
  local remote tmp
  remote=$(git remote get-url origin)
  tmp=$(mktemp -d)
  trap 'rm -rf "${tmp:-}"' EXIT  # function-local under set -u: the trap fires after it goes out of scope
  cp -R docs/. "$tmp/"
  # A throwaway repo, so gh-pages carries only the site and never this history.
  git -C "$tmp" init -q
  git -C "$tmp" add -A
  git -C "$tmp" -c user.email=docs@oflux -c user.name=oflux \
      commit -qm "docs $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  git -C "$tmp" push -q --force "$remote" HEAD:gh-pages
  echo "pushed docs/ -> gh-pages on $remote"
}

case "${1:-build}" in
  build) build ;;
  deploy) deploy ;;
  *) echo "usage: $0 build|deploy" >&2; exit 2 ;;
esac
