#!/usr/bin/env bash
#
# Comprehensive end-to-end smoke. Run this on the MBP before sending the library
# Mac off on a multi-hour run. Burns ~5–10 min and asserts every known failure
# mode from the 2026-05-17 library session is fixed.
#
# Checks (in order):
#   1. Pre-flight: .env has NVIDIA_API_KEY, ollama is up, opencode is on PATH
#   2. Proxy starts and /healthz responds
#   3. One full_system issue runs through end-to-end
#   4. Proxy log shows the *issue ID* as session — NOT "default" (session-ID fix)
#   5. results/smoke/predictions.jsonl exists and is non-empty
#   6. The recorded patch is non-empty (local model can drive opencode)
#   7. Both `local` and `frontier` were hit (heuristic escalation fires)
#   8. No NIM 429 retry storm: if 429s appear, each call retried ≤ 2x
#
# Exit code: 0 if every check passes, 1 otherwise. Diagnostics print inline.
#
# Env:
#   LOCAL_MODEL          override the model name the proxy sends to Ollama
#                        (defaults to qwen2.5-coder-14b-16k)
#   OPENCODE_TIMEOUT     per-issue cap in seconds (default: 600)
#   SMOKE_ISSUE_IDX      which issue from issues_subset20.json to use (default: 0)

set -uo pipefail   # NOT -e: we want to keep collecting failures and report at the end

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PATH="$HOME/.local/bin:$HOME/.opencode/bin:$PATH"

PROXY_PORT="${SMOKE_PROXY_PORT:-8765}"   # off the default so we can run concurrently with a real run
PROXY_URL="http://127.0.0.1:${PROXY_PORT}/v1"
PROXY_LOG="logs/smoke.proxy.log"
OUTPUT_DIR="results/smoke"
SMOKE_ISSUE_IDX="${SMOKE_ISSUE_IDX:-0}"
ISSUES_FILE="benchmark/issues_smoke1.json"

mkdir -p logs results

fail=0
check() {
    # check "name" status_code "diagnostic on failure"
    local name="$1" rc="$2" diag="${3:-}"
    if [[ "$rc" == "0" ]]; then
        echo "  ✓ $name"
    else
        echo "  ✗ $name"
        [[ -n "$diag" ]] && echo "      $diag"
        fail=1
    fi
}

# ---------------------------------------------------------------------------
# 1. Pre-flight
# ---------------------------------------------------------------------------
echo "=== 1. Pre-flight ==="

if grep -qE "^NVIDIA_API_KEY=nvapi-" .env 2>/dev/null; then
    check ".env has NVIDIA_API_KEY" 0
else
    check ".env has NVIDIA_API_KEY" 1 "set NVIDIA_API_KEY=nvapi-... in $REPO_ROOT/.env"
fi

if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    check "ollama server reachable" 0
else
    check "ollama server reachable" 1 "run scripts/setup.sh first (or 'ollama serve &')"
fi

: "${LOCAL_MODEL:=qwen2.5-coder-14b-16k}"
if ollama list 2>/dev/null | awk '{print $1}' | grep -qx "${LOCAL_MODEL}:latest\|${LOCAL_MODEL}"; then
    check "local model $LOCAL_MODEL present" 0
else
    check "local model $LOCAL_MODEL present" 1 "run scripts/setup.sh to pull + create the 16K variant"
fi

if command -v opencode >/dev/null 2>&1; then
    check "opencode on PATH" 0
else
    check "opencode on PATH" 1 "run scripts/setup.sh"
fi

if [[ "$fail" == "1" ]]; then
    echo
    echo "SMOKE FAILED at pre-flight. Fix the above and re-run." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Carve a 1-issue file from subset20
# ---------------------------------------------------------------------------
echo
echo "=== 2. Prepare 1-issue subset ==="

if [[ ! -f benchmark/issues_subset20.json ]]; then
    .venv/bin/python -m benchmark.sample_issues --seed 42
fi

.venv/bin/python -c "
import json, sys
src = json.load(open('benchmark/issues_subset20.json'))
idx = int('$SMOKE_ISSUE_IDX')
json.dump([src[idx]], open('$ISSUES_FILE','w'), indent=2)
print('  picked', src[idx]['instance_id'])
"
SMOKE_IID="$(.venv/bin/python -c "import json; print(json.load(open('$ISSUES_FILE'))[0]['instance_id'])")"

# Clean any prior smoke output so checks operate on fresh artifacts
rm -rf "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# 3. Start proxy, /healthz, run one issue
# ---------------------------------------------------------------------------
echo
echo "=== 3. Run one full_system issue ($SMOKE_IID) ==="

LOCAL_MODEL="$LOCAL_MODEL" ROUTER_MODE=full \
    .venv/bin/uvicorn proxy.server:app \
    --host 127.0.0.1 --port "$PROXY_PORT" \
    >"$PROXY_LOG" 2>&1 &
proxy_pid=$!
trap 'kill $proxy_pid 2>/dev/null; wait $proxy_pid 2>/dev/null' EXIT

healthy=0
for _ in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:${PROXY_PORT}/healthz" >/dev/null 2>&1; then
        healthy=1
        break
    fi
    sleep 0.5
done
if [[ "$healthy" == "1" ]]; then
    check "proxy /healthz responds" 0
else
    check "proxy /healthz responds" 1 "see $PROXY_LOG — most often model name doesn't match"
    exit 1
fi

OPENCODE_TIMEOUT="${OPENCODE_TIMEOUT:-600}" \
    .venv/bin/python -m benchmark.run_swe_bench \
    --issues "$ISSUES_FILE" \
    --run-name smoke \
    --output "$OUTPUT_DIR" \
    --proxy-url "$PROXY_URL" \
    --max-steps 30 \
    --parallel 1 \
    --no-eval 2>&1 | tail -20

kill $proxy_pid 2>/dev/null || true
wait $proxy_pid 2>/dev/null || true
trap - EXIT

# ---------------------------------------------------------------------------
# 4. Validate artifacts
# ---------------------------------------------------------------------------
echo
echo "=== 4. Validate artifacts ==="

# 4a. Session ID is the issue ID, not "default" (the bug from the library run)
if grep -qE "traj=smoke/${SMOKE_IID}" "$PROXY_LOG"; then
    check "proxy log shows per-issue session ID" 0
else
    check "proxy log shows per-issue session ID" 1 \
        "expected 'traj=smoke/${SMOKE_IID}' in $PROXY_LOG — opencode isn't propagating the session"
fi

if grep -qE "traj=default" "$PROXY_LOG"; then
    check "no requests fell back to traj=default" 1 \
        "$(grep -c 'traj=default' "$PROXY_LOG") requests collapsed to traj=default"
else
    check "no requests fell back to traj=default" 0
fi

# 4b. predictions.jsonl exists and contains our issue
if [[ -s "$OUTPUT_DIR/predictions.jsonl" ]]; then
    check "predictions.jsonl exists + non-empty" 0
else
    check "predictions.jsonl exists + non-empty" 1 "harness didn't write predictions"
fi

# 4c. The patch field is non-empty
patch_chars="$(.venv/bin/python -c "
import json
try:
    rec = json.load(open('$OUTPUT_DIR/trajectories/${SMOKE_IID}.json'))
    print(len(rec.get('patch','') or ''))
except Exception:
    print(-1)
")"
if [[ "$patch_chars" -gt 50 ]]; then
    check "patch is non-empty (${patch_chars} chars)" 0
else
    check "patch is non-empty" 1 \
        "patch length=${patch_chars} — local model isn't producing edits. Try LOCAL_MODEL_PULL=qwen2.5-coder:14b or check opencode_stdout in trajectories/${SMOKE_IID}.json"
fi

# 4d. Both backends were exercised (full_system should escalate at least once)
n_local="$(grep -cE '-> local ' "$PROXY_LOG" || true)"
n_frontier="$(grep -cE '-> frontier ' "$PROXY_LOG" || true)"
if [[ "$n_local" -gt 0 && "$n_frontier" -gt 0 ]]; then
    check "both local + frontier called (local=$n_local, frontier=$n_frontier)" 0
else
    check "both local + frontier called" 1 \
        "local=$n_local frontier=$n_frontier — heuristic escalation isn't firing"
fi

# 4e. No NIM 429 retry storm. Count "NIM 429 → failing soft" events; if > 0,
# fail-soft is working. Then count raw 429s in proxy log — should be modest.
n_429_observed="$(grep -cE 'NIM 429' "$PROXY_LOG" || true)"
if [[ "$n_429_observed" == "0" ]]; then
    check "no NIM 429s observed" 0
else
    # 429s happen sometimes. The check is whether they're handled (fail-soft, not retry-storm).
    n_fallback="$(grep -cE 'failing soft to local' "$PROXY_LOG" || true)"
    if [[ "$n_429_observed" -le 5 && "$n_fallback" -ge 1 ]]; then
        check "NIM 429s handled via fail-soft ($n_429_observed seen, $n_fallback fallbacks)" 0
    else
        check "NIM 429s handled via fail-soft" 1 \
            "$n_429_observed 429s seen, $n_fallback fallbacks — retry budget may be too large"
    fi
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo
if [[ "$fail" == "0" ]]; then
    echo "================================================"
    echo "  SMOKE PASS — pipeline is wired correctly."
    echo "  Safe to launch ./scripts/run_all.sh"
    echo "================================================"
    exit 0
else
    echo "================================================"
    echo "  SMOKE FAIL — DO NOT send him to the library."
    echo "  Inspect: $PROXY_LOG"
    echo "           $OUTPUT_DIR/trajectories/${SMOKE_IID}.json"
    echo "================================================"
    exit 1
fi
