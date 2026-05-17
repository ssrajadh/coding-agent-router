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
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # plotting is optional; CSV still works


PRICE_PER_1K = {
    "local":    {"prompt": 0.0,    "completion": 0.0},
    # NVIDIA NIM published rate for qwen/qwen3-coder-480b-a35b-instruct.
    # Source: build.nvidia.com pricing page (record date in commit msg).
    "frontier": {"prompt": 0.0006, "completion": 0.0024},
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_config_summary(results_dir: Path) -> list[dict]:
    """One row per config from results/<cfg>/summary.json."""
    rows = []
    for cfg_dir in sorted(results_dir.iterdir()):
        summary = cfg_dir / "summary.json"
        if not summary.exists():
            continue
        data = json.loads(summary.read_text())
        rows.append({
            "config": cfg_dir.name,
            "total": data["total"],
            "passed": data["passed"],
            "errors": data.get("errors", 0),
            "success_rate": data["success_rate"],
            "results": data["results"],
        })
    return rows


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


def per_config_metrics(config_rows: list[dict], step_rows: list[dict]) -> list[dict]:
    by_config: dict[str, list[dict]] = {}
    for s in step_rows:
        by_config.setdefault(s["config"], []).append(s)

    out = []
    for cfg in config_rows:
        steps = by_config.get(cfg["config"], [])
        total_cost = sum(step_cost(s) for s in steps)
        n_trajectories = max(1, cfg["total"])
        frontier_steps = sum(1 for s in steps if s["backend"] == "frontier")
        escalated_steps = sum(1 for s in steps if s["reason"].startswith("escalated_"))
        latencies = [s["latency_s"] for s in steps if s["latency_s"]]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        out.append({
            "config": cfg["config"],
            "n_issues": cfg["total"],
            "passed": cfg["passed"],
            "success_rate": cfg["success_rate"],
            "n_steps": len(steps),
            "frontier_step_pct": round(100 * frontier_steps / max(1, len(steps)), 1),
            "escalation_step_pct": round(100 * escalated_steps / max(1, len(steps)), 1),
            "avg_step_latency_s": round(avg_latency, 2),
            "total_cost_usd": round(total_cost, 4),
            "cost_per_trajectory_usd": round(total_cost / n_trajectories, 4),
        })
    return out


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
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for m in metrics:
        ax.scatter(
            m["cost_per_trajectory_usd"],
            100 * m["success_rate"],
            s=120, label=m["config"],
        )
        ax.annotate(
            m["config"],
            (m["cost_per_trajectory_usd"], 100 * m["success_rate"]),
            textcoords="offset points", xytext=(6, 4), fontsize=9,
        )
    ax.set_xlabel("Cost per trajectory (USD)")
    ax.set_ylabel("Task success rate (%)")
    ax.set_title("Cost–success trade-off on SWE-bench Lite subset")
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

    if not config_rows:
        raise SystemExit("no per-config summaries found — was the experiment graded?")

    print(f"Loaded {len(config_rows)} configs, {len(step_rows)} steps")

    metrics = per_config_metrics(config_rows, step_rows)
    write_csv(metrics, out_dir / "summary.csv")
    write_csv(step_rows, out_dir / "per_step.csv")
    print(f"  → {out_dir / 'summary.csv'}")
    print(f"  → {out_dir / 'per_step.csv'}")

    pareto_plot(metrics, fig_dir / "pareto.pdf")
    latency_cdf_plot(step_rows, fig_dir / "latency_cdf.pdf")
    if plt:
        print(f"  → {fig_dir / 'pareto.pdf'}")
        print(f"  → {fig_dir / 'latency_cdf.pdf'}")

    # Pretty print to stdout
    print()
    print(f"{'config':18s}  {'n':>3s}  {'pass':>4s}  {'rate':>5s}  {'$/traj':>8s}  {'front%':>6s}  {'esc%':>5s}")
    print("-" * 70)
    for m in metrics:
        print(
            f"{m['config']:18s}  "
            f"{m['n_issues']:3d}  "
            f"{m['passed']:4d}  "
            f"{100 * m['success_rate']:4.1f}%  "
            f"${m['cost_per_trajectory_usd']:7.4f}  "
            f"{m['frontier_step_pct']:5.1f}%  "
            f"{m['escalation_step_pct']:4.1f}%"
        )


if __name__ == "__main__":
    main()
