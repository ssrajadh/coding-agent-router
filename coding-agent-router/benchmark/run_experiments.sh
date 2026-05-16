#!/usr/bin/env bash
#
# Phase 4 experiment driver.
#
# Loops over each config in configs/, spawns the proxy with the matching
# ROUTER_MODE, runs the SWE-bench harness against that config's issue subset,
# and tears the proxy down before moving on.
#
# Usage:
#     ./benchmark/run_experiments.sh                   # all configs
#     ./benchmark/run_experiments.sh full_system       # one config
#     CONFIGS="all_local full_system" ./benchmark/run_experiments.sh
#
# Env vars:
#     SMOKE=1            Use a 3-issue subset and skip the expensive configs.
#                        For wiring validation on a 16 GB MBP.
#     NO_EVAL=1          Don't run the swebench Docker evaluator. Predictions
#                        JSONL is still written for later grading. Required on
#                        locked-down machines without Docker.
#     CONFIGS="..."      Whitespace-separated subset of configs to run.
#
# Outputs land under results/<config_name>/.
# Re-running with the same output directory resumes from per-issue result files.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Run order: cheap baselines first so we have *something* if we run out of time.
# all_frontier last because it costs money / hits NIM rate limits hardest.
if [[ "${SMOKE:-0}" == "1" ]]; then
    # Pick the three most informative configs for wiring validation: the local
    # baseline, the cheapest router (format_check), and the full system.
    DEFAULT_CONFIGS=(all_local format_check full_system)
else
    DEFAULT_CONFIGS=(all_local random format_check full_system all_frontier)
fi

if [[ $# -gt 0 ]]; then
    CONFIGS_TO_RUN=("$@")
elif [[ -n "${CONFIGS:-}" ]]; then
    read -r -a CONFIGS_TO_RUN <<<"$CONFIGS"
else
    CONFIGS_TO_RUN=("${DEFAULT_CONFIGS[@]}")
fi

PROXY_PORT="${PROXY_PORT:-8000}"
PROXY_URL="http://127.0.0.1:${PROXY_PORT}/v1"

mkdir -p results logs

# Smoke mode: carve a 3-issue file from issues_subset20.json and force every
# config to use it. Lets MBP wiring tests finish in ~15 min instead of hours.
SMOKE_ISSUES_FILE="benchmark/issues_smoke3.json"
if [[ "${SMOKE:-0}" == "1" ]]; then
    if [[ ! -f "$SMOKE_ISSUES_FILE" ]]; then
        echo "→ Creating smoke 3-issue file at $SMOKE_ISSUES_FILE"
        .venv/bin/python -c "
import json
src = json.load(open('benchmark/issues_subset20.json'))
json.dump(src[:3], open('$SMOKE_ISSUES_FILE', 'w'), indent=2)
"
    fi
fi

extra_args=()
if [[ "${NO_EVAL:-0}" == "1" ]]; then
    extra_args+=(--no-eval)
    echo "ℹ NO_EVAL=1: predictions.jsonl will be written but no swebench Docker eval will run"
fi

parse_yaml_field() {
    # cheap YAML reader: grep for "key:" at line start, strip key + whitespace + quotes
    local file="$1" key="$2"
    grep -E "^${key}:" "$file" | head -n1 | sed -E "s/^${key}:\s*//; s/^['\"]//; s/['\"]$//"
}

start_proxy() {
    local mode="$1" logfile="$2"
    ROUTER_MODE="$mode" .venv/bin/uvicorn proxy.server:app \
        --host 127.0.0.1 --port "$PROXY_PORT" \
        >"$logfile" 2>&1 &
    echo $!
}

wait_for_proxy() {
    for _ in $(seq 1 30); do
        if curl -sf "http://127.0.0.1:${PROXY_PORT}/healthz" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.5
    done
    echo "proxy did not become healthy" >&2
    return 1
}

stop_proxy() {
    local pid="$1"
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

for cfg_name in "${CONFIGS_TO_RUN[@]}"; do
    cfg_path="configs/${cfg_name}.yaml"
    if [[ ! -f "$cfg_path" ]]; then
        echo "ERROR: $cfg_path not found" >&2
        exit 1
    fi

    router_mode="$(parse_yaml_field "$cfg_path" router_mode)"
    issues="$(parse_yaml_field "$cfg_path" issues)"
    parallel="$(parse_yaml_field "$cfg_path" parallel)"
    max_steps="$(parse_yaml_field "$cfg_path" max_steps)"

    # Smoke override: ignore each config's issues field and use the tiny set.
    if [[ "${SMOKE:-0}" == "1" ]]; then
        issues="$SMOKE_ISSUES_FILE"
        max_steps=20  # cap step budget to keep smoke fast
    fi

    if [[ ! -f "$issues" ]]; then
        echo "SKIP: issues file $issues missing — run benchmark/sample_issues.py first" >&2
        continue
    fi

    output_dir="results/${cfg_name}"
    proxy_log="logs/${cfg_name}.proxy.log"

    echo "============================================================"
    echo "  config:    $cfg_name (mode=$router_mode)"
    echo "  issues:    $issues"
    echo "  parallel:  $parallel"
    echo "  output:    $output_dir"
    echo "============================================================"

    proxy_pid="$(start_proxy "$router_mode" "$proxy_log")"
    trap 'stop_proxy "$proxy_pid"' EXIT
    wait_for_proxy

    set +e
    .venv/bin/python -m benchmark.run_swe_bench \
        --issues "$issues" \
        --run-name "$cfg_name" \
        --output "$output_dir" \
        --proxy-url "$PROXY_URL" \
        --max-steps "$max_steps" \
        --parallel "$parallel" \
        --resume \
        ${extra_args[@]+"${extra_args[@]}"}
    rc=$?
    set -e

    stop_proxy "$proxy_pid"
    trap - EXIT
    sleep 2

    if [[ $rc -ne 0 ]]; then
        echo "WARN: $cfg_name run exited $rc — continuing to next config" >&2
    fi
done

echo
echo "All configs complete. Summaries:"
for cfg_name in "${CONFIGS_TO_RUN[@]}"; do
    summary="results/${cfg_name}/summary.json"
    if [[ -f "$summary" ]]; then
        echo "  $cfg_name:"
        .venv/bin/python -c "
import json, sys
s = json.load(open('$summary'))
print(f\"    passed={s['passed']}/{s['total']} ({100*s['success_rate']:.1f}%)  errors={s['errors']}\")
"
    fi
done
