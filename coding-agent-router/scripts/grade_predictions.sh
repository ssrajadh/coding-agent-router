#!/usr/bin/env bash
#
# Grade one or more predictions.jsonl files with the SWE-bench evaluator.
# Requires Docker. Use on the Linux laptop / any Docker-having machine
# after copying back the Mac's results/ directory.
#
# Usage:
#     ./scripts/grade_predictions.sh                    # grade every results/*/predictions.jsonl
#     ./scripts/grade_predictions.sh results/all_local  # grade just one config
#     ./scripts/grade_predictions.sh path/to/preds.jsonl  # single file
#
# Re-running is safe — swebench's evaluator caches per-instance results.
# After this finishes, each results/<config>/summary.json is updated in place
# with real passed/failed counts.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v docker >/dev/null 2>&1; then
    echo "✗ Docker is required to grade SWE-bench predictions." >&2
    echo "  Install Docker Desktop or run this on a machine that has it." >&2
    exit 1
fi
if ! docker info >/dev/null 2>&1; then
    echo "✗ Docker is installed but not running. Start it and try again." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Collect predictions paths
# ---------------------------------------------------------------------------

PRED_PATHS=()
if [[ $# -eq 0 ]]; then
    while IFS= read -r p; do PRED_PATHS+=("$p"); done < <(
        find results -mindepth 2 -maxdepth 2 -name predictions.jsonl 2>/dev/null
    )
else
    for arg in "$@"; do
        if [[ -d "$arg" ]]; then
            if [[ -f "$arg/predictions.jsonl" ]]; then
                PRED_PATHS+=("$arg/predictions.jsonl")
            fi
        elif [[ -f "$arg" ]]; then
            PRED_PATHS+=("$arg")
        else
            echo "skipping unknown path: $arg" >&2
        fi
    done
fi

if [[ ${#PRED_PATHS[@]} -eq 0 ]]; then
    echo "no predictions.jsonl files found under results/" >&2
    exit 1
fi

echo "Will grade ${#PRED_PATHS[@]} prediction file(s):"
for p in "${PRED_PATHS[@]}"; do echo "  - $p"; done
echo

# ---------------------------------------------------------------------------
# Grade each
# ---------------------------------------------------------------------------

for preds in "${PRED_PATHS[@]}"; do
    config_dir="$(dirname "$preds")"
    config_name="$(basename "$config_dir")"
    run_id="grade-${config_name}-$(date +%s)"
    eval_dir="${config_dir}/evals/${run_id}"
    mkdir -p "$eval_dir"

    echo "============================================================"
    echo "  grading $config_name"
    echo "  preds:   $preds"
    echo "  eval:    $eval_dir"
    echo "============================================================"

    set +e
    .venv/bin/python -m swebench.harness.run_evaluation \
        --predictions_path "$(cd "$(dirname "$preds")" && pwd)/$(basename "$preds")" \
        --max_workers 1 \
        --run_id "$run_id" \
        --cache_level instance \
        --dataset_name princeton-nlp/SWE-bench_Lite
    rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
        echo "⚠ $config_name evaluator exited $rc — partial results may still exist" >&2
    fi

    # The evaluator writes a results JSON named "<model>.<run_id>.json" in cwd
    # by default. Move whatever it produced into the per-config eval dir.
    mv -- *.${run_id}.json "$eval_dir/" 2>/dev/null || true

    # Refresh summary.json with real pass/fail counts
    .venv/bin/python <<PY
import json
from pathlib import Path

config_dir = Path("$config_dir")
summary_path = config_dir / "summary.json"
if not summary_path.exists():
    print(f"no summary.json at {summary_path}, skipping")
    raise SystemExit
summary = json.loads(summary_path.read_text())

# The evaluator's report file lists resolved instance ids
resolved = set()
for f in (config_dir / "evals").rglob("*.json"):
    try:
        data = json.loads(f.read_text())
    except json.JSONDecodeError:
        continue
    resolved.update(data.get("resolved_ids", data.get("resolved", [])))

passed = 0
for r in summary.get("results", []):
    if r["instance_id"] in resolved:
        r["passed"] = True
        r["eval_skipped"] = False
        passed += 1
    else:
        r["passed"] = False
total = summary.get("total", len(summary.get("results", [])))
summary["passed"] = passed
summary["failed"] = total - passed - summary.get("errors", 0)
summary["success_rate"] = round(passed / total, 4) if total else 0.0
summary_path.write_text(json.dumps(summary, indent=2))
print(f"  → $config_name: {passed}/{total} resolved ({100*summary['success_rate']:.1f}%)")
PY
done

echo
echo "All grading runs complete. Final summaries:"
for preds in "${PRED_PATHS[@]}"; do
    summary="$(dirname "$preds")/summary.json"
    cfg="$(basename "$(dirname "$preds")")"
    if [[ -f "$summary" ]]; then
        .venv/bin/python -c "
import json
s = json.load(open('$summary'))
print(f\"  {'$cfg':20s}  {s['passed']:3d}/{s['total']:3d}  ({100*s['success_rate']:.1f}%)\")
"
    fi
done
