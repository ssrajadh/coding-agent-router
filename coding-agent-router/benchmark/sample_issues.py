"""
Stratified sampling from SWE-bench Lite.

Selects 50 issues across >= 5 repos, with a mix of:
- fail-to-pass test counts (easy / medium / hard)
- issue body lengths (short / medium / long)

Writes benchmark/issues.json and benchmark/issues_subset20.json (all_frontier subset).

Usage:
    python -m benchmark.sample_issues [--seed 42] [--out benchmark/issues.json]
"""

import argparse
import json
import random
from pathlib import Path


REPOS_OF_INTEREST = [
    "sympy/sympy",
    "django/django",
    "scikit-learn/scikit-learn",
    "pallets/flask",
    "psf/requests",
    "matplotlib/matplotlib",
    "sphinx-doc/sphinx",
    "astropy/astropy",
    "pytest-dev/pytest",
    "pydata/xarray",
]

TARGET_TOTAL = 50
MIN_REPOS = 5


def _bucket(value: int, thresholds: list[int]) -> int:
    for i, t in enumerate(thresholds):
        if value <= t:
            return i
    return len(thresholds)


def stratified_sample(instances: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)

    # Assign buckets
    ftp_thresholds = [1, 3]       # fail-to-pass: 1 / 2-3 / 4+
    len_thresholds = [500, 1500]  # chars: short / medium / long

    for inst in instances:
        ftp = len(inst.get("FAIL_TO_PASS", inst.get("fail_to_pass", "[]")))
        if isinstance(ftp, str):
            ftp = len(json.loads(ftp))
        body = inst.get("problem_statement", "")
        inst["_ftp_bucket"] = _bucket(ftp, ftp_thresholds)
        inst["_len_bucket"] = _bucket(len(body), len_thresholds)

    # Group by (repo, ftp_bucket, len_bucket)
    groups: dict[tuple, list] = {}
    for inst in instances:
        key = (
            inst.get("repo", inst.get("instance_id", "").rsplit("-", 2)[0]),
            inst["_ftp_bucket"],
            inst["_len_bucket"],
        )
        groups.setdefault(key, []).append(inst)

    for g in groups.values():
        rng.shuffle(g)

    selected: list[dict] = []
    repos_seen: set[str] = set()

    # Round-robin across groups until we have n
    group_keys = sorted(groups.keys())
    idx = 0
    while len(selected) < n:
        if idx >= len(group_keys):
            idx = 0
            # All groups exhausted before reaching n — stop
            all_empty = all(not groups[k] for k in group_keys)
            if all_empty:
                break
        key = group_keys[idx]
        if groups[key]:
            inst = groups[key].pop(0)
            selected.append(inst)
            repos_seen.add(key[0])
        idx += 1

    if len(repos_seen) < MIN_REPOS:
        raise ValueError(
            f"Only {len(repos_seen)} repos in sample — need >= {MIN_REPOS}. "
            "Broaden REPOS_OF_INTEREST or reduce MIN_REPOS."
        )

    # Clean up helper keys
    for inst in selected:
        inst.pop("_ftp_bucket", None)
        inst.pop("_len_bucket", None)

    return selected


def load_swebench_lite() -> list[dict]:
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        return list(ds)
    except Exception as e:
        raise RuntimeError(
            "Could not load SWE-bench Lite. Install with:\n"
            "  pip install datasets swebench\n"
            f"Original error: {e}"
        )


def main():
    parser = argparse.ArgumentParser(description="Sample 50 issues from SWE-bench Lite")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="benchmark/issues.json")
    parser.add_argument("--subset-out", default="benchmark/issues_subset20.json")
    parser.add_argument("--subset-size", type=int, default=20)
    args = parser.parse_args()

    print("Loading SWE-bench Lite …")
    instances = load_swebench_lite()
    print(f"  {len(instances)} total instances")

    sample = stratified_sample(instances, TARGET_TOTAL, seed=args.seed)
    print(f"  selected {len(sample)} instances across "
          f"{len({i.get('repo', '') for i in sample})} repos")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sample, indent=2))
    print(f"  wrote {out}")

    subset = sample[: args.subset_size]
    sub_out = Path(args.subset_out)
    sub_out.write_text(json.dumps(subset, indent=2))
    print(f"  wrote subset ({args.subset_size} issues) → {sub_out}")


if __name__ == "__main__":
    main()
