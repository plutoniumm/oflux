#!/usr/bin/env bash
# One-off migration for oflux 1.3.1: curated models were renamed to the
# <line><version>[-modifier] scheme. Installed manifests are keyed by filename
# AND carry the name inside, and presets reference the model by name, so all
# three have to move together. Idempotent: re-running is a no-op.
set -euo pipefail
H="${OFLUX_HOME:-$HOME/.oflux}"
DRY="${DRY:-0}"
map="qwen-image-2.1-uncensored:qwe-2.1-uc
qwen-image-2.1:qwe-2.1
qwen-image-edit:qwe-2511
qwen-image:qwi-1
flux.1-kontext:flx-1-kontext
flux.2-klein-9b:flx-2-klein-9b
flux.2-klein:flx-2-klein-4b
z-image-turbo:zim-1-turbo
flux.1-krea:flx-1-krea
flux.1-dev:flx-1-dev
flux.1-schnell:flx-1-schnell"

while IFS=: read -r old new; do
  src="$H/manifests/$old.json"; dst="$H/manifests/$new.json"
  [ -f "$src" ] || continue
  if [ -f "$dst" ]; then echo "skip $old (target $new already exists)"; continue; fi
  echo "manifest $old -> $new"
  [ "$DRY" = 1 ] && continue
  python3 - "$src" "$new" <<'PY'
import json,sys
p,new=sys.argv[1],sys.argv[2]
d=json.load(open(p)); d["name"]=new
json.dump(d,open(p,"w"),indent=2)
PY
  mv "$src" "$dst"
done <<<"$map"

# Presets point at a model by name.
for p in "$H"/presets/*.json; do
  [ -f "$p" ] || continue
  python3 - "$p" "$DRY" <<'PY'
import json,sys
path,dry=sys.argv[1],sys.argv[2]
m={"qwen-image-2.1-uncensored":"qwe-2.1-uc","qwen-image-2.1":"qwe-2.1","qwen-image-edit":"qwe-2511",
   "qwen-image":"qwi-1","flux.1-kontext":"flx-1-kontext","flux.2-klein-9b":"flx-2-klein-9b",
   "flux.2-klein":"flx-2-klein-4b","z-image-turbo":"zim-1-turbo","flux.1-krea":"flx-1-krea",
   "flux.1-dev":"flx-1-dev","flux.1-schnell":"flx-1-schnell"}
d=json.load(open(path))
old=d.get("model")
if old in m:
    print(f"preset {d.get('name',path)}: model {old} -> {m[old]}")
    if dry!="1":
        d["model"]=m[old]; json.dump(d,open(path,"w"),indent=2)
PY
done
echo "done — restart the daemon so it re-reads the store"
