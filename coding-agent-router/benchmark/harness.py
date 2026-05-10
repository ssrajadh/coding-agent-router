"""
Thin wrapper around the SWE-bench evaluation harness.

Responsibilities:
- Clone repos and check out base commits into isolated sandboxes.
- Extract git diffs as patches.
- Write prediction files and invoke the SWE-bench evaluator subprocess.
- Parse evaluation results JSON back to a Python dict.
"""

import json
import subprocess
import uuid
from pathlib import Path


# ---------------------------------------------------------------------------
# Repo setup
# ---------------------------------------------------------------------------

def setup_repo(instance: dict, workdir: Path) -> None:
    """Clone the repo at base_commit into workdir."""
    workdir.mkdir(parents=True, exist_ok=True)
    repo = instance.get("repo", "")
    base_commit = instance["base_commit"]

    # SWE-bench instances carry repo as "owner/name"; construct the GitHub URL.
    repo_url = instance.get("repo_url") or f"https://github.com/{repo}.git"

    subprocess.run(
        ["git", "clone", "--depth=50", repo_url, str(workdir)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "fetch", "--depth=50", "origin", base_commit],
        cwd=workdir,
        check=False,  # shallow clone may not need this
        capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", base_commit],
        cwd=workdir,
        check=True,
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

def extract_patch(workdir: Path) -> str:
    """Return the git diff against HEAD (i.e. uncommitted edits made by the agent)."""
    return subprocess.check_output(
        ["git", "diff"],
        cwd=workdir,
        text=True,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_patch(
    instance: dict,
    patch: str,
    run_id: str | None = None,
    eval_dir: Path | None = None,
) -> dict:
    """
    Write a predictions JSONL and invoke the SWE-bench evaluator.
    Returns {"passed": bool, "instance_id": str, "raw": dict}.
    """
    if run_id is None:
        run_id = "eval-" + str(uuid.uuid4())[:8]
    if eval_dir is None:
        eval_dir = Path("runs") / "evals" / run_id
    eval_dir.mkdir(parents=True, exist_ok=True)

    pred = {
        "instance_id": instance["instance_id"],
        "model_patch": patch,
        "model_name_or_path": "hybrid-router",
    }
    preds_path = eval_dir / "preds.jsonl"
    preds_path.write_text(json.dumps(pred) + "\n")

    subprocess.run(
        [
            "python", "-m", "swebench.harness.run_evaluation",
            "--predictions_path", str(preds_path),
            "--max_workers", "1",
            "--run_id", run_id,
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=eval_dir,
    )

    return parse_eval_result(instance["instance_id"], run_id, eval_dir)


def parse_eval_result(instance_id: str, run_id: str, eval_dir: Path) -> dict:
    """
    The SWE-bench evaluator writes results under:
      <eval_dir>/<run_id>.<model_name>/results/results.json

    We search broadly so minor naming variations don't break us.
    """
    # Search for any results.json under eval_dir
    candidates = list(eval_dir.rglob("results.json"))
    for path in candidates:
        try:
            data = json.loads(path.read_text())
            # Results JSON format: {"resolved": [...instance_ids...], ...}
            resolved = set(data.get("resolved", []))
            return {
                "instance_id": instance_id,
                "passed": instance_id in resolved,
                "raw": data,
            }
        except (json.JSONDecodeError, KeyError):
            continue

    # Fallback: evaluator didn't produce output (e.g. not installed)
    return {"instance_id": instance_id, "passed": False, "raw": {}}
