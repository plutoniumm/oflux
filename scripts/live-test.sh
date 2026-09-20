#!/usr/bin/env bash
# Integration tests against the RUNNING installed oflux daemon (:11534) — the
# launchd-managed /Applications build with the real Metal engine. Exercises the
# HTTP API, the CLI, OpenAI-compat, and a real generation + edit on the GPU.
set -uo pipefail
BASE="http://127.0.0.1:11534"
OUT="dist/smoke/live"; mkdir -p "$OUT"
pass=0; fail=0
ok(){ echo "  PASS  $1"; pass=$((pass+1)); }
no(){ echo "  FAIL  $1"; fail=$((fail+1)); }
ispng(){ python3 -c "import sys;sys.exit(0 if open('$1','rb').read(8)==b'\x89PNG\r\n\x1a\n' else 1)"; }

echo "== metadata / health =="
v=$(oflux version 2>/dev/null)
case "$v" in "oflux "[0-9]*) ok "oflux version ($v)" ;; *) no "version: ${v:-<none>}" ;; esac
[ "$(curl -s $BASE/healthz)" = "ok" ] && ok "/healthz" || no "/healthz"
curl -s $BASE/api/tags | grep -q '"models"'      && ok "/api/tags"   || no "/api/tags"
curl -s $BASE/api/ps   | grep -q '"loaded"'       && ok "/api/ps"     || no "/api/ps"
curl -s $BASE/v1/models| grep -q '"object":"list"'&& ok "/v1/models"  || no "/v1/models"
oflux list 2>/dev/null | grep -q qwen-image-edit  && ok "CLI: oflux list" || no "oflux list"

echo "== live generation (native JSON, z-image on Metal) =="
c=$(curl -s -o "$OUT/gen.json" -w '%{http_code}' --max-time 600 -X POST $BASE/v1/generate \
  -H 'Content-Type: application/json' -d '{"model":"z-image-turbo","prompt":"a mossy stone bridge over a river, photograph","steps":8}')
if [ "$c" = 200 ]; then
  python3 -c "import json,base64;d=json.load(open('$OUT/gen.json'));open('$OUT/gen.png','wb').write(base64.b64decode(d['images'][0]))"
  ispng "$OUT/gen.png" && ok "generate -> valid PNG ($(du -h "$OUT/gen.png"|cut -f1))" || no "generate: not a PNG"
else no "generate HTTP $c: $(head -c200 "$OUT/gen.json")"; fi

echo "== OpenAI-compat generation (/v1/images/generations) =="
c=$(curl -s -o "$OUT/oai.json" -w '%{http_code}' --max-time 600 -X POST $BASE/v1/images/generations \
  -H 'Content-Type: application/json' -d '{"model":"z-image-turbo","prompt":"a red maple leaf on snow","size":"512x512"}')
if [ "$c" = 200 ] && python3 -c "import json;assert json.load(open('$OUT/oai.json'))['data'][0]['b64_json']" 2>/dev/null; then
  ok "/v1/images/generations (OpenAI shape)"
else no "openai generate HTTP $c"; fi

echo "== live edit (ref-image path, flux-kontext on Metal) =="
if [ -f "$OUT/gen.png" ]; then
  python3 -c "import json,base64;b=base64.b64encode(open('$OUT/gen.png','rb').read()).decode();json.dump({'model':'flux.1-kontext','prompt':'turn it into a watercolor painting','image':b},open('$OUT/editbody.json','w'))"
  c=$(curl -s -o "$OUT/edit.json" -w '%{http_code}' --max-time 900 -X POST $BASE/v1/edit \
    -H 'Content-Type: application/json' -d @"$OUT/editbody.json")
  if [ "$c" = 200 ]; then
    python3 -c "import json,base64;d=json.load(open('$OUT/edit.json'));open('$OUT/edit.png','wb').write(base64.b64decode(d['images'][0]))"
    ispng "$OUT/edit.png" && ok "edit -> valid PNG ($(du -h "$OUT/edit.png"|cut -f1))" || no "edit: not a PNG"
  else no "edit HTTP $c: $(head -c200 "$OUT/edit.json")"; fi
else no "edit skipped (no input image)"; fi

echo "== error handling =="
c=$(curl -s -o /dev/null -w '%{http_code}' -X POST $BASE/v1/edit -H 'Content-Type: application/json' -d '{"model":"nope","prompt":"x","image":"x"}')
[ "$c" = 404 ] && ok "unknown model -> 404" || no "unknown model -> $c"
c=$(curl -s -o /dev/null -w '%{http_code}' -X POST $BASE/v1/edit -H 'Content-Type: application/json' -d '{"model":"z-image-turbo","prompt":"x"}')
[ "$c" = 400 ] && ok "edit without image -> 400" || no "edit w/o image -> $c"

echo "== segmentation + masked edit (SAM3) =="
# Only meaningful in a SAM3=1 build; a SAM3=0 daemon answers 501 and we say so
# rather than failing. The first call may download the 707MB checkpoint.
seg=$(curl -s -o "$OUT/seg.json" -w '%{http_code}' --max-time 900 -X POST $BASE/v1/segment \
  -H 'Content-Type: application/json' \
  -d "{\"prompt\":\"the bridge\",\"image\":\"$(python3 -c "import base64;print(base64.b64encode(open('$OUT/gen.png','rb').read()).decode())" 2>/dev/null)\"}")
case "$seg" in
  200) python3 -c "import json;d=json.load(open('$OUT/seg.json'));assert d['masks'][0]['mask']" 2>/dev/null \
         && ok "/v1/segment -> mask" || no "/v1/segment: no mask in response" ;;
  501) echo "  SKIP  /v1/segment (built without SAM3)" ;;
  *)   no "/v1/segment HTTP $seg: $(head -c200 "$OUT/seg.json")" ;;
esac

if [ "$seg" = 200 ] && [ -f "$OUT/gen.png" ]; then
  python3 -c "import json,base64;b=base64.b64encode(open('$OUT/gen.png','rb').read()).decode();json.dump({'model':'qwen-image-edit','prompt':'make it a stone arch','mask_prompt':'the bridge','image':b},open('$OUT/maskbody.json','w'))"
  c=$(curl -s -o "$OUT/maskedit.json" -w '%{http_code}' --max-time 900 -X POST $BASE/v1/edit \
    -H 'Content-Type: application/json' -d @"$OUT/maskbody.json")
  if [ "$c" = 200 ]; then
    python3 -c "import json,base64;d=json.load(open('$OUT/maskedit.json'));open('$OUT/maskedit.png','wb').write(base64.b64decode(d['images'][0]))"
    ispng "$OUT/maskedit.png" && ok "mask_prompt edit -> valid PNG ($(du -h "$OUT/maskedit.png"|cut -f1))" || no "mask_prompt edit: not a PNG"
  else no "mask_prompt edit HTTP $c: $(head -c200 "$OUT/maskedit.json")"; fi

  # A prompt that matches nothing must 422, never a silent whole-image edit.
  c=$(curl -s -o /dev/null -w '%{http_code}' --max-time 600 -X POST $BASE/v1/edit \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"qwen-image-edit\",\"prompt\":\"x\",\"mask_prompt\":\"a purple giraffe wearing a top hat\",\"image\":\"$(python3 -c "import base64;print(base64.b64encode(open('$OUT/gen.png','rb').read()).decode())")\"}")
  [ "$c" = 422 ] && ok "mask_prompt with no match -> 422" || no "mask_prompt no-match -> $c"
fi

echo "== seed echo =="
python3 -c "import json;d=json.load(open('$OUT/gen.json'));assert isinstance(d.get('seed'),int)" 2>/dev/null \
  && ok "generate echoes the seed used" || no "generate response carries no seed"

echo ""; echo "RESULT: $pass passed, $fail failed"
[ "$fail" -eq 0 ] && echo "ALL LIVE TESTS PASSED" || echo "SOME LIVE TESTS FAILED"
