#!/usr/bin/env bash
# Fire a few requests at a running routerai server and print routing + metrics.
# Usage:
#   uvicorn app.main:app --port 8000   # in one terminal
#   ./scripts/demo_curl.sh             # in another
set -euo pipefail
BASE="${ROUTERAI_BASE:-http://127.0.0.1:8000}"

echo "== cheap tier =="
curl -s "$BASE/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d '{"model":"cheap","messages":[{"role":"user","content":"classify: I love this!"}]}' \
  | python3 -m json.tool

echo "== premium tier =="
curl -s "$BASE/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d '{"model":"premium","messages":[{"role":"user","content":"prove step by step and analyze"}]}' \
  | python3 -m json.tool

echo "== metrics =="
curl -s "$BASE/metrics" | python3 -m json.tool
