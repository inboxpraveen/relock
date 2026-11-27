#!/usr/bin/env sh
# relock from any language: the whole loop with curl.
#   relock serve &        (export RELOCK_TOKEN=... first to require a token)
#   sh examples/http/curl.sh
set -eu
URL="${RELOCK_URL:-http://127.0.0.1:8765}/v1"
AUTH="Authorization: Bearer ${RELOCK_TOKEN:-}"
post() { curl -sS --fail-with-body -X POST "$URL$1" -H "$AUTH" -H 'Content-Type: application/json' -d "$2"; }
PY="${PYTHON:-$(command -v python3 || command -v python)}"
field() { "$PY" -c "import json,sys; print(json.load(sys.stdin)$1)"; }

post /items '{"title": "Fetch orders by date", "key": "orders.fetch", "tools": ["get_orders"], "exist_ok": true,
  "clauses": ["Authenticate with the orders API key before calling get_orders.",
              "Pass the date to get_orders as MM/DD/YYYY."]}' > /dev/null

for attempt in 1 2 3; do
  EP=$(post /episodes '{"task": "orders for March 14"}' | field '["episode_id"]')
  CONTEXT=$(post "/episodes/$EP/recall" '{"query": "fetch the orders placed on a date"}' | field '["context"]')
  case "$CONTEXT" in
    *YYYY-MM-DD*) OUT='{"tool": "get_orders", "ok": true}' ;;
    *) OUT='{"tool": "get_orders", "ok": false, "error": {"message": "invalid date", "status": 422,
             "correction": "Pass the date to get_orders as YYYY-MM-DD."}}' ;;
  esac
  VERDICT=$(post "/episodes/$EP/tool-result" "$OUT" | field '["observations"][0]["verdict"]')
  post "/episodes/$EP/close" '{}' > /dev/null
  echo "attempt $attempt: $VERDICT"
done

curl -sS --fail-with-body "$URL/items/orders.fetch" -H "$AUTH" | field '["clauses"]'
