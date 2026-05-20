"""
Turn the contents of results/ + runs/trajectories/ into paper-ready artifacts:

    figures/pareto.pdf    — cost-vs-success scatter, one point per config
    figures/latency_cdf.pdf — per-step latency CDF, one line per config
    analysis/summary.csv  — per-config table (success, cost, escalation rate)
    analysis/per_step.csv — flattened per-step trace for ad-hoc plots

Usage:
    .venv/bin/python -m analysis.analyze

Cost rates come from PRICE_PER_1K below. Local-model traffic is $0; NIM rates
follow the published Qwen3-coder-480b-a35b list price as of 2026-05 — adjust
if NVIDIA changes its pricing before you submit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # plotting is optional; CSV still works


def wilson_ci(passed: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI for a binomial proportion. Returns (low, high) in [0, 1]."""
    if total == 0:
        return (0.0, 0.0)
    p = passed / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


PRICE_PER_1K = {
    "local":    {"prompt": 0.0,    "completion": 0.0},
    # NVIDIA NIM published rate for qwen/qwen3-coder-480b-a35b-instruct.
    # Source: build.nvidia.com pricing page (record date in commit msg).
    "frontier": {"prompt": 0.0006, "completion": 0.0024},
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_config_summary(results_dir: Path, skip_prefixes: tuple[str, ...] = ("smoke",)) -> list[dict]:
    """One row per config from results/<cfg>/summary.json. Skips smoke dirs by default."""
    rows = []
    for cfg_dir in sorted(results_dir.iterdir()):
        if any(cfg_dir.name.startswith(p) for p in skip_prefixes):
            continue
        summary = cfg_dir / "summary.json"
        if not summary.exists():
            continue
        data = json.loads(summary.read_text())
        # Non-empty patch counts come from per-issue results; the inline-eval
        # column ("passed") can't be trusted because the harness's swebench
        # invocation used bare `python` (silent ModuleNotFoundError) on some
        # machines. Real pass counts come from load_grade_reports below.
        nonempty = sum(
            1 for r in data["results"]
            if len((r.get("patch") or "")) > 50
        )
        rows.append({
            "config": cfg_dir.name,
            "total": data["total"],
            "passed_inline": data["passed"],  # potentially zero for legitimate runs
            "nonempty_patches": nonempty,
            "errors": data.get("errors", 0),
            "results": data["results"],
        })
    return rows


def load_grade_reports(root: Path) -> dict[str, dict]:
    """Read each `<config>.grade-<run_id>.json` swebench report. Keyed by config name."""
    out: dict[str, dict] = {}
    for f in sorted(root.glob("*.grade-*.json")):
        config = f.name.split(".grade-", 1)[0]
        data = json.loads(f.read_text())
        out[config] = {
            "resolved": data.get("resolved_instances", 0),
            "completed": data.get("completed_instances", 0),
            "submitted": data.get("submitted_instances", 0),
            "empty": data.get("empty_patch_instances", 0),
            "errors": data.get("error_instances", 0),
            "resolved_ids": data.get("resolved_ids", []),
            "completed_ids": data.get("completed_ids", []),
            "error_ids": data.get("error_ids", []),
        }
    return out


def load_trajectories(traj_dir: Path) -> list[dict]:
    """Flatten runs/trajectories/*.json into one row per request step."""
    rows = []
    for f in sorted(traj_dir.glob("*.json")):
        try:
            traj = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        # session id encodes "<config>/<instance_id>"; recover both
        sid = traj.get("id", f.stem)
        config = sid.split("/", 1)[0] if "/" in sid else "unknown"
        for step in traj.get("steps", []):
            usage = step.get("response_usage") or {}
            rows.append({
                "config": config,
                "session": sid,
                "backend": step.get("backend"),
                "reason": step.get("reason"),
                "latency_s": step.get("latency_s", 0.0),
                "local_failed": bool(step.get("local_failed")),
                "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
                "completion_tokens": usage.get("completion_tokens", 0) or 0,
            })
    return rows


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------

def step_cost(row: dict) -> float:
    price = PRICE_PER_1K.get(row["backend"], PRICE_PER_1K["local"])
    return (row["prompt_tokens"] / 1000) * price["prompt"] + \
           (row["completion_tokens"] / 1000) * price["completion"]


def per_config_metrics(
    config_rows: list[dict],
    step_rows: list[dict],
    grades: dict[str, dict],
) -> list[dict]:
    by_config: dict[str, list[dict]] = {}
    for s in step_rows:
        by_config.setdefault(s["config"], []).append(s)

    out = []
    for cfg in config_rows:
        cfg_name = cfg["config"]
        steps = by_config.get(cfg_name, [])
        total_cost = sum(step_cost(s) for s in steps)
        n_trajectories = max(1, cfg["total"])
        frontier_steps = sum(1 for s in steps if s["backend"] == "frontier")
        escalated_steps = sum(1 for s in steps if (s["reason"] or "").startswith("escalated_"))
        latencies = [s["latency_s"] for s in steps if s["latency_s"]]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        g = grades.get(cfg_name, {})
        resolved = g.get("resolved", 0)
        n = cfg["total"]
        ci_low, ci_high = wilson_ci(resolved, n)

        out.append({
            "config": cfg_name,
            "n_issues": n,
            "resolved": resolved,
            "pass_rate": resolved / n if n else 0.0,
            "pass_rate_ci_low": round(ci_low, 4),
            "pass_rate_ci_high": round(ci_high, 4),
            "graded": g.get("completed", 0),
            "empty_patches": g.get("empty", cfg["total"] - cfg["nonempty_patches"]),
            "grade_errors": g.get("errors", 0),
            "nonempty_patches": cfg["nonempty_patches"],
            "n_steps": len(steps),
            "frontier_step_pct": round(100 * frontier_steps / max(1, len(steps)), 1),
            "escalation_step_pct": round(100 * escalated_steps / max(1, len(steps)), 1),
            "avg_step_latency_s": round(avg_latency, 2),
            "total_cost_usd": round(total_cost, 4),
            "cost_per_trajectory_usd": round(total_cost / n_trajectories, 4),
        })
    return out


def reason_histogram(step_rows: list[dict]) -> dict[str, Counter]:
    """For each config, count routing-decision reasons across all steps."""
    by_config: dict[str, Counter] = {}
    for s in step_rows:
        cfg = s["config"]
        reason = s.get("reason") or "unknown"
        by_config.setdefault(cfg, Counter())[reason] += 1
    return by_config


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pareto_plot(metrics: list[dict], out_path: Path) -> None:
    if plt is None:
        print("matplotlib not installed — skipping pareto plot")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    # Hard-coded offsets per config to avoid label collisions at (~0, 0)
    label_offsets = {
        "all_local":    (8, 12),
        "full_system":  (8, -24),
        "all_frontier": (-10, 6),
    }
    for m in metrics:
        cfg = m["config"]
        x = m["cost_per_trajectory_usd"]
        y = 100 * m["pass_rate"]
        y_low = 100 * m["pass_rate_ci_low"]
        y_high = 100 * m["pass_rate_ci_high"]
        yerr = [[y - y_low], [y_high - y]]
        ax.errorbar(
            x, y, yerr=yerr,
            fmt="o", markersize=11, capsize=6, capthick=1.5,
            label=cfg,
        )
        dx, dy = label_offsets.get(cfg, (8, 6))
        ha = "right" if dx < 0 else "left"
        ax.annotate(
            f"{cfg}\n({m['nonempty_patches']}/{m['n_issues']} non-empty, "
            f"{m['resolved']}/{m['n_issues']} resolved)",
            (x, y),
            textcoords="offset points", xytext=(dx, dy),
            fontsize=8, ha=ha,
        )
    ax.set_xlabel("Cost per trajectory (USD, NIM list rates)")
    ax.set_ylabel("Task success rate (%) — Wilson 95% CI")
    ax.set_title("Cost–success trade-off on SWE-bench Lite (n=20 stratified)")
    ax.set_ylim(-3, 35)  # leave headroom for the all_frontier CI upper bound
    ax.set_xlim(-0.05, 0.85)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def latency_cdf_plot(step_rows: list[dict], out_path: Path) -> None:
    if plt is None:
        return
    by_config: dict[str, list[float]] = {}
    for s in step_rows:
        if s["latency_s"]:
            by_config.setdefault(s["config"], []).append(s["latency_s"])
    if not by_config:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for cfg, lats in sorted(by_config.items()):
        lats = sorted(lats)
        ys = [(i + 1) / len(lats) for i in range(len(lats))]
        ax.plot(lats, ys, label=cfg, linewidth=1.8)
    ax.set_xlabel("Per-step latency (s)")
    ax.set_ylabel("CDF")
    ax.set_xscale("log")
    ax.set_title("Per-step latency distribution by config")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results", help="Directory of per-config results")
    parser.add_argument("--trajectories", default="runs/trajectories", help="Per-step traces dir")
    parser.add_argument("--out", default="analysis", help="Output dir for CSVs")
    parser.add_argument("--figures", default="figures", help="Output dir for plots")
    args = parser.parse_args()

    results_dir = Path(args.results)
    traj_dir = Path(args.trajectories)
    out_dir = Path(args.out)
    fig_dir = Path(args.figures)

    if not results_dir.exists():
        raise SystemExit(f"no results dir at {results_dir}")

    config_rows = load_config_summary(results_dir)
    step_rows = load_trajectories(traj_dir) if traj_dir.exists() else []
    grades = load_grade_reports(Path("."))

    if not config_rows:
        raise SystemExit("no per-config summaries found — was the experiment graded?")

    print(f"Loaded {len(config_rows)} configs, {len(step_rows)} steps, "
          f"{len(grades)} grade reports")

    metrics = per_config_metrics(config_rows, step_rows, grades)
    write_csv(metrics, out_dir / "summary.csv")
    write_csv(step_rows, out_dir / "per_step.csv")
    print(f"  → {out_dir / 'summary.csv'}")
    print(f"  → {out_dir / 'per_step.csv'}")

    pareto_plot(metrics, fig_dir / "pareto.pdf")
    latency_cdf_plot(step_rows, fig_dir / "latency_cdf.pdf")
    if plt:
        print(f"  → {fig_dir / 'pareto.pdf'}")
        print(f"  → {fig_dir / 'latency_cdf.pdf'}")

    # Pretty print to stdout — headline table
    print()
    print(f"{'config':14s}  {'n':>3s}  {'resv':>4s}  {'rate':>5s}  {'95% CI':>14s}  "
          f"{'nonemp':>6s}  {'$/traj':>8s}  {'front%':>6s}  {'esc%':>5s}")
    print("-" * 90)
    for m in metrics:
        ci_str = f"[{100*m['pass_rate_ci_low']:4.1f}, {100*m['pass_rate_ci_high']:4.1f}]"
        print(
            f"{m['config']:14s}  "
            f"{m['n_issues']:3d}  "
            f"{m['resolved']:4d}  "
            f"{100 * m['pass_rate']:4.1f}%  "
            f"{ci_str:>14s}  "
            f"{m['nonempty_patches']:6d}  "
            f"${m['cost_per_trajectory_usd']:7.4f}  "
            f"{m['frontier_step_pct']:5.1f}%  "
            f"{m['escalation_step_pct']:4.1f}%"
        )

    # Routing-decision histogram per config (only interesting for full_system)
    print()
    print("Decision-reason histogram per config (only `full` mode has variety):")
    hist = reason_histogram(step_rows)
    for cfg in sorted(hist):
        total = sum(hist[cfg].values())
        print(f"\n  {cfg}  ({total} steps):")
        for reason, count in hist[cfg].most_common():
            pct = 100 * count / total if total else 0
            print(f"    {reason:35s}  {count:4d}  ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
