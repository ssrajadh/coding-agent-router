#!/usr/bin/env bash
#
# THE one script. After you've put NVIDIA_API_KEY in .env, run this and walk away.
#
# Steps:
#   1. Bootstrap dependencies (Python venv, ollama, opencode, models) via setup.sh
#   2. Sample 50 issues from SWE-bench Lite if not already done
#   3. Run the experiment sweep across all 5 configs
#   4. Tar up results for the report
#
# Env vars:
#   SMOKE=1       — 3 issues × 3 configs × small model. ~15 min on M1 Pro 16GB.
#                   Use this on your MBP first to validate the pipeline.
#   NO_EVAL=1     — Skip the swebench Docker evaluator (auto-detected if Docker
#                   is not on PATH). Predictions JSONL still gets written.
#   LOCAL_MODEL=… — Override the local model name. Default is the 16K variant
#                   that setup.sh creates.
#
# Output:
#   results/<config>/summary.json          — pass/fail counts
#   results/<config>/predictions.jsonl     — for later swebench grading
#   results/<config>/trajectories/         — per-issue results + patches
#   runs/trajectories/<session>.json       — per-step proxy traces
#   results-bundle-<timestamp>.tar.gz      — everything zipped for the report

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Pull in any user-side bins setup.sh installed (ollama, opencode) so subsequent
# steps see them even if the shell hasn't been re-sourced.
export PATH="$HOME/.local/bin:$HOME/.opencode/bin:$PATH"

# ---------------------------------------------------------------------------
# 1. Bootstrap
# ---------------------------------------------------------------------------

echo "============================================================"
echo "  STEP 1/4  Setup"
echo "============================================================"
./scripts/setup.sh

# Pick up LOCAL_MODEL setup.sh just printed, if the caller didn't already set one.
if [[ -z "${LOCAL_MODEL:-}" ]]; then
    if [[ "${SMOKE:-0}" == "1" ]]; then
        export LOCAL_MODEL="qwen2.5-coder-3b-16k"
    else
        export LOCAL_MODEL="qwen3-coder-8b-16k"
    fi
fi
echo "Using LOCAL_MODEL=$LOCAL_MODEL"

# Auto-detect Docker — if missing, set NO_EVAL=1 so the run still completes.
if ! command -v docker >/dev/null 2>&1; then
    if [[ "${NO_EVAL:-}" != "0" ]]; then
        echo "ℹ Docker not found → setting NO_EVAL=1 (predictions will be graded later)"
        export NO_EVAL=1
    fi
fi

# ---------------------------------------------------------------------------
# 2. Sample issues
# ---------------------------------------------------------------------------

echo
echo "============================================================"
echo "  STEP 2/4  Sample issues"
echo "============================================================"
if [[ -f benchmark/issues.json && -f benchmark/issues_subset20.json ]]; then
    echo "✓ benchmark/issues.json + issues_subset20.json already exist"
else
    .venv/bin/python -m benchmark.sample_issues --seed 42
fi

# ---------------------------------------------------------------------------
# 3. Run experiments
# ---------------------------------------------------------------------------

echo
echo "============================================================"
echo "  STEP 3/4  Run experiments"
echo "============================================================"
START_TS="$(date +%s)"
SMOKE="${SMOKE:-0}" NO_EVAL="${NO_EVAL:-0}" ./benchmark/run_experiments.sh
END_TS="$(date +%s)"
ELAPSED_MIN=$(( (END_TS - START_TS) / 60 ))
echo "Experiments finished in ~${ELAPSED_MIN} min"

# ---------------------------------------------------------------------------
# 4. Bundle results
# ---------------------------------------------------------------------------

echo
echo "============================================================"
echo "  STEP 4/4  Bundle results"
echo "============================================================"
STAMP="$(date +%Y%m%d-%H%M%S)"
BUNDLE="results-bundle-${STAMP}.tar.gz"

# Include only the small/portable artifacts — never the workdirs.
tar -czf "$BUNDLE" \
    results/*/summary.json \
    results/*/predictions.jsonl \
    results/*/trajectories \
    runs/trajectories \
    benchmark/issues*.json \
    configs \
    logs/*.log \
    2>/dev/null || true

SIZE_MB="$(du -m "$BUNDLE" 2>/dev/null | cut -f1 || echo '?')"
echo "✓ Bundle written: $BUNDLE  (${SIZE_MB} MB)"
echo
echo "Next: copy that tarball to a Docker-having machine and grade with:"
echo "  for p in results/*/predictions.jsonl; do"
echo "    python -m swebench.harness.run_evaluation --predictions_path \"\$p\" --run_id grade-\$(basename \$(dirname \$p))"
echo "  done"
