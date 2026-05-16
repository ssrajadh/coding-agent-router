"""
SWE-bench benchmark driver.

Usage examples:
    # Single issue, all-local mode
    python -m benchmark.run_swe_bench \\
        --issues benchmark/issues.json \\
        --run-name all_local \\
        --output results/all_local \\
        --max-steps 60 \\
        --parallel 2

    # Resume an interrupted run (skips already-finished issues)
    python -m benchmark.run_swe_bench \\
        --issues benchmark/issues.json \\
        --run-name all_local \\
        --output results/all_local \\
        --resume

Multi-machine parallelism:
    Each machine runs this script against the same NFS-mounted results directory.
    Issues are claimed via atomic mv-rename of .todo → .wip files in
    results/<run_name>/queue/.  Completed issues write a .done file.
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from .harness import evaluate_patch, extract_patch, setup_repo

log = logging.getLogger("benchmark")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)


# ---------------------------------------------------------------------------
# Work queue (mv-rename for atomic claiming across machines)
# ---------------------------------------------------------------------------

class WorkQueue:
    def __init__(self, queue_dir: Path):
        self.queue_dir = queue_dir
        self.queue_dir.mkdir(parents=True, exist_ok=True)

    def populate(self, instance_ids: list[str]) -> None:
        for iid in instance_ids:
            todo = self.queue_dir / f"{iid}.todo"
            done = self.queue_dir / f"{iid}.done"
            if not todo.exists() and not done.exists():
                todo.write_text(iid)

    def claim(self) -> str | None:
        """Atomically claim one todo item. Returns instance_id or None."""
        for todo in sorted(self.queue_dir.glob("*.todo")):
            wip = todo.with_suffix(".wip")
            try:
                todo.rename(wip)
                return todo.stem
            except (FileNotFoundError, OSError):
                continue
        return None

    def complete(self, instance_id: str) -> None:
        wip = self.queue_dir / f"{instance_id}.wip"
        done = self.queue_dir / f"{instance_id}.done"
        try:
            wip.rename(done)
        except FileNotFoundError:
            done.write_text(instance_id)

    def requeue(self, instance_id: str) -> None:
        """Return a failed/abandoned wip back to todo."""
        wip = self.queue_dir / f"{instance_id}.wip"
        todo = self.queue_dir / f"{instance_id}.todo"
        try:
            wip.rename(todo)
        except FileNotFoundError:
            todo.write_text(instance_id)

    def remaining(self) -> int:
        return len(list(self.queue_dir.glob("*.todo")))

    def in_progress(self) -> int:
        return len(list(self.queue_dir.glob("*.wip")))

    def done_count(self) -> int:
        return len(list(self.queue_dir.glob("*.done")))


# ---------------------------------------------------------------------------
# OpenCode invocation
# ---------------------------------------------------------------------------

def run_opencode(
    instance: dict,
    workdir: Path,
    session_id: str,
    proxy_url: str,
    max_steps: int = 60,
    timeout: int = 1800,
) -> subprocess.CompletedProcess:
    import subprocess as sp
    prompt = (
        f"Resolve this GitHub issue:\n\n{instance['problem_statement']}\n\n"
        "Make the minimal code changes required to fix the issue. "
        "Run the relevant tests to verify your fix before stopping."
    )
    env = os.environ.copy()
    env["OPENCODE_PROVIDER"] = "hybrid"
    env["X_SESSION_ID"] = session_id
    # Some opencode versions read OPENAI_BASE_URL for the provider URL
    env["OPENAI_BASE_URL"] = proxy_url
    env["OPENAI_API_KEY"] = "not-needed"

    return sp.run(
        ["opencode", "run", "--max-steps", str(max_steps), prompt],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Single-issue runner
# ---------------------------------------------------------------------------

def run_issue(
    instance: dict,
    run_name: str,
    output_dir: Path,
    proxy_url: str,
    max_steps: int,
    keep_workdir: bool,
    skip_eval: bool = False,
) -> dict:
    iid = instance["instance_id"]
    session_id = f"{run_name}/{iid}"
    workdir = output_dir / "workdirs" / iid
    traj_dir = output_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    result: dict = {"instance_id": iid, "passed": False, "error": None}
    t0 = time.time()

    try:
        log.info("[%s] setting up repo", iid)
        setup_repo(instance, workdir)

        log.info("[%s] running opencode (session=%s)", iid, session_id)
        oc_result = run_opencode(
            instance, workdir, session_id, proxy_url, max_steps=max_steps
        )
        if oc_result.returncode != 0:
            log.warning("[%s] opencode exited %d", iid, oc_result.returncode)

        patch = extract_patch(workdir)
        if not patch.strip():
            log.warning("[%s] empty patch — no changes made", iid)

        result["patch"] = patch
        result["opencode_stdout"] = oc_result.stdout[-4000:] if oc_result.stdout else ""
        result["opencode_returncode"] = oc_result.returncode

        if skip_eval:
            # No Docker available (e.g. locked-down library Mac).
            # Persist the patch into predictions.jsonl; grade later on a Docker box.
            result["eval_skipped"] = True
        else:
            run_id = f"{run_name}-{iid[:20]}-{str(uuid.uuid4())[:6]}"
            eval_dir = output_dir / "evals" / iid
            eval_result = evaluate_patch(instance, patch, run_id=run_id, eval_dir=eval_dir)
            result.update(eval_result)

    except subprocess.TimeoutExpired:
        result["error"] = "timeout"
        log.error("[%s] timed out", iid)
    except Exception as exc:
        result["error"] = str(exc)
        log.exception("[%s] unexpected error", iid)
    finally:
        result["wall_seconds"] = round(time.time() - t0, 1)
        # Write per-issue result
        (traj_dir / f"{iid}.json").write_text(json.dumps(result, indent=2))
        if not keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)

    status = "PASS" if result["passed"] else ("ERROR" if result["error"] else "FAIL")
    log.info("[%s] %s  (%.0fs)", iid, status, result["wall_seconds"])
    return result


# ---------------------------------------------------------------------------
# Parallel runner
# ---------------------------------------------------------------------------

async def run_parallel(
    instances_by_id: dict[str, dict],
    queue: WorkQueue,
    run_name: str,
    output_dir: Path,
    proxy_url: str,
    max_steps: int,
    parallel: int,
    keep_workdir: bool,
    skip_eval: bool = False,
) -> list[dict]:
    sem = asyncio.Semaphore(parallel)
    results: list[dict] = []
    loop = asyncio.get_event_loop()

    async def worker():
        while True:
            iid = queue.claim()
            if iid is None:
                return
            instance = instances_by_id.get(iid)
            if instance is None:
                log.warning("Unknown instance_id %s in queue — skipping", iid)
                queue.complete(iid)
                continue
            async with sem:
                try:
                    result = await loop.run_in_executor(
                        None,
                        run_issue,
                        instance,
                        run_name,
                        output_dir,
                        proxy_url,
                        max_steps,
                        keep_workdir,
                        skip_eval,
                    )
                    results.append(result)
                    queue.complete(iid)
                except Exception:
                    log.exception("[%s] worker crashed — requeuing", iid)
                    queue.requeue(iid)

    workers = [asyncio.create_task(worker()) for _ in range(parallel)]
    await asyncio.gather(*workers)
    return results


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def write_summary(results: list[dict], output_dir: Path, run_name: str) -> None:
    passed = sum(1 for r in results if r.get("passed"))
    failed = sum(1 for r in results if not r.get("passed") and not r.get("error"))
    errors = sum(1 for r in results if r.get("error"))
    total = len(results)
    summary = {
        "run_name": run_name,
        "total": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "success_rate": round(passed / total, 4) if total else 0.0,
        "results": results,
    }
    out = output_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    log.info(
        "Summary: %d/%d passed (%.1f%%)  errors=%d  → %s",
        passed, total, 100 * summary["success_rate"], errors, out,
    )


def write_predictions_jsonl(results: list[dict], output_dir: Path, run_name: str) -> None:
    """
    Write SWE-bench-compatible predictions JSONL.
    Lets you grade later on a machine that has Docker, even if eval was skipped here.
    """
    path = output_dir / "predictions.jsonl"
    with path.open("w") as f:
        for r in results:
            patch = r.get("patch", "")
            if patch is None:
                patch = ""
            f.write(json.dumps({
                "instance_id": r["instance_id"],
                "model_patch": patch,
                "model_name_or_path": run_name,
            }) + "\n")
    log.info("Wrote %d predictions → %s", len(results), path)


def load_existing_results(output_dir: Path) -> dict[str, dict]:
    """Load already-finished per-issue results for --resume."""
    traj_dir = output_dir / "trajectories"
    existing: dict[str, dict] = {}
    if traj_dir.exists():
        for f in traj_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                existing[data["instance_id"]] = data
            except (json.JSONDecodeError, KeyError):
                pass
    return existing


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run SWE-bench issues through the proxy")
    parser.add_argument("--issues", required=True, help="Path to issues.json")
    parser.add_argument("--run-name", required=True, help="Identifier for this run")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--proxy-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--parallel", type=int, default=1,
                        help="Max concurrent issues (stay within NIM 40 RPM)")
    parser.add_argument("--keep-workdirs", action="store_true",
                        help="Don't delete repo workdirs after eval")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed issues")
    parser.add_argument("--no-eval", action="store_true",
                        help="Don't run swebench evaluator (requires Docker). "
                             "Still writes predictions.jsonl for later grading.")
    args = parser.parse_args()

    issues_path = Path(args.issues)
    instances: list[dict] = json.loads(issues_path.read_text())
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    instances_by_id = {inst["instance_id"]: inst for inst in instances}

    # Resume: skip instances that already have a result
    if args.resume:
        existing = load_existing_results(output_dir)
        log.info("Resume: %d already done, %d remaining",
                 len(existing), len(instances) - len(existing))
        instances = [i for i in instances if i["instance_id"] not in existing]
    else:
        existing = {}

    queue = WorkQueue(output_dir / "queue")
    queue.populate([i["instance_id"] for i in instances])

    log.info(
        "Run=%s  issues=%d  parallel=%d  proxy=%s",
        args.run_name, len(instances), args.parallel, args.proxy_url,
    )

    new_results = asyncio.run(
        run_parallel(
            instances_by_id=instances_by_id,
            queue=queue,
            run_name=args.run_name,
            output_dir=output_dir,
            proxy_url=args.proxy_url,
            max_steps=args.max_steps,
            parallel=args.parallel,
            keep_workdir=args.keep_workdirs,
            skip_eval=args.no_eval,
        )
    )

    all_results = list(existing.values()) + new_results
    write_summary(all_results, output_dir, args.run_name)
    write_predictions_jsonl(all_results, output_dir, args.run_name)


if __name__ == "__main__":
    main()
