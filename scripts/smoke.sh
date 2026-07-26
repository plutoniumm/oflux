#!/usr/bin/env bash
# Real end-to-end Metal smoke test: pull models through oflux and run genuine
# generation + edit requests against the bundled sd-server on the GPU.
# Continues past individual failures; everything is logged; PNGs land in dist/smoke.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
BIN="$ROOT/dist/oflux"
OUT="$ROOT/dist/smoke"
mkdir -p "$OUT"
export OFLUX_ENGINE="$ROOT/third_party/sd-server"

log() { echo "[$(date +%H:%M:%S)] $*"; }
secs() { date +%s; }

log "building oflux"
go build -o "$BIN" ./cmd/oflux || { log "BUILD FAILED"; exit 1; }

log "starting daemon (engine: $OFLUX_ENGINE)"
"$BIN" serve >"$OUT/daemon.log" 2>&1 &
DPID=$!
trap 'kill $DPID 2>/dev/null' EXIT
curl -s --retry 60 --retry-connrefused --retry-delay 1 http://127.0.0.1:11534/healthz >/dev/null \
  && log "daemon up (pid $DPID)" || { log "DAEMON FAILED"; cat "$OUT/daemon.log"; exit 1; }

pull() { # name quant
  local t0 t1; t0=$(secs)
  log "PULL $1 ($2) ..."
  if "$BIN" pull "$1" --quant "$2" 2>&1 | sed 's/^/    /'; then
    t1=$(secs); log "  pulled $1 in $((t1-t0))s"
  else
    log "  PULL FAILED: $1"; return 1
  fi
}

gen_json() { # endpoint json outpng label
  local t0 t1 code; t0=$(secs)
  log "$4: POST $1"
  code=$(curl -s -o "$OUT/_resp.json" -w '%{http_code}' --max-time 1800 \
    -X POST "http://127.0.0.1:11534$1" -H 'Content-Type: application/json' -d "$2")
  t1=$(secs)
  if [ "$code" != "200" ]; then log "  HTTP $code in $((t1-t0))s: $(head -c 300 "$OUT/_resp.json")"; return 1; fi
  if python3 - "$OUT/_resp.json" "$3" <<'PY'
import json,base64,sys
d=json.load(open(sys.argv[1]))
imgs=d.get("images") or []
if not imgs: print("  no image in response:", json.dumps(d)[:200]); sys.exit(1)
open(sys.argv[2],"wb").write(base64.b64decode(imgs[0]))
PY
  then log "  OK in $((t1-t0))s -> $3 ($(du -h "$3"|cut -f1))"; else log "  DECODE FAILED"; return 1; fi
}

edit_body() { # model prompt imgfile outbody
  python3 - "$@" <<'PY'
import json,base64,sys
model,prompt,img,out=sys.argv[1:5]
b=base64.b64encode(open(img,'rb').read()).decode()
json.dump({"model":model,"prompt":prompt,"image":b}, open(out,'w'))
PY
}

# ---------- 1) z-image-turbo: generation ----------
if pull z-image-turbo Q8_0; then
  gen_json /v1/generate '{"model":"z-image-turbo","prompt":"a red fox sitting in a green forest, sharp photograph","steps":8}' "$OUT/fox.png" "z-image generate"
fi

# ---------- 2) flux.1-kontext: edit ----------
if pull flux.1-kontext Q4_K_M; then
  if [ -f "$OUT/fox.png" ]; then
    edit_body flux.1-kontext "make it night time with a full moon" "$OUT/fox.png" "$OUT/_kontext.json"
    gen_json /v1/edit "@$OUT/_kontext.json" "$OUT/fox-night.png" "flux-kontext edit"
  else
    log "skip kontext edit: no input image"
  fi
fi

# ---------- 3) qwen-image-edit: edit ----------
if pull qwen-image-edit Q4_K_M; then
  if [ -f "$OUT/fox.png" ]; then
    edit_body qwen-image-edit "add a small red hat on the fox" "$OUT/fox.png" "$OUT/_qwen.json"
    gen_json /v1/edit "@$OUT/_qwen.json" "$OUT/fox-hat.png" "qwen-image-edit edit"
  else
    log "skip qwen edit: no input image"
  fi
fi

log "=== oflux ps ==="; "$BIN" ps 2>&1 | sed 's/^/    /'
log "=== oflux list ==="; "$BIN" list 2>&1 | sed 's/^/    /'
log "=== outputs ==="; ls -la "$OUT"/*.png 2>/dev/null | sed 's/^/    /' || log "  (no PNGs produced)"
log "SMOKE DONE"
