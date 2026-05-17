# Evaluation Plan — Linux Laptop

**Run this after the Mac finishes.** See `EVALUATION_PLAN_MAC.md` first.

| | |
|---|---|
| **Role** | Run the network-only config + grade everything + produce paper artifacts |
| **Configs** | `all_frontier` (here); grades the 4 Mac configs |
| **Wall time** | ~40 min run + ~30 min grade + ~30 sec analyze ≈ 1 hour 10 min |
| **Hardware** | Linux laptop with Docker (admin-level access) |
| **Output** | `analysis/summary.csv`, `figures/pareto.pdf`, `figures/latency_cdf.pdf` |

---

## Pre-flight (~1 min)

```bash
cd ~/Documents/coding-agent-router/coding-agent-router

# Confirm the API key is still in .env
grep -q '^NVIDIA_API_KEY=nvapi-' .env && echo "key OK" || echo "missing key in .env"

# Make sure Docker is running (grading depends on it)
docker info >/dev/null 2>&1 && echo "Docker OK" || echo "start Docker Desktop"

# Pull latest if you've been editing
git status && git pull
```

---

## Step 1 — Frontier run (~40 min runtime + ~10 min inline grading)

```bash
FRONTIER_ONLY=1 ./scripts/run_all.sh
```

What `FRONTIER_ONLY=1` does:
- Skips every Ollama step in `setup.sh` (no local model needed for this config)
- Forces `CONFIGS=all_frontier`
- Keeps the Docker-based grader **on** (laptop has Docker), so this config is fully graded inline

When it finishes:

```
results/all_frontier/summary.json        # passed/total, real graded numbers
results/all_frontier/predictions.jsonl
results/all_frontier/trajectories/*.json
```

---

## Step 2 — Receive the Mac's results (~2 min)

If the Mac sent a tarball:

```bash
tar -xzf results-bundle-*.tar.gz
```

If you used scp of the directories directly, they should already be in place under `results/`. Confirm:

```bash
ls results/   # should list all_frontier (yours) + all_local, random, format_check, full_system (Mac's)
ls runs/trajectories/ | head   # per-step proxy traces from the Mac
```

If `runs/trajectories/` is missing, the per-step cost/escalation numbers in the final analysis will be empty — go scp them from the Mac.

---

## Step 3 — Grade the Mac's predictions (~30 min)

The Mac wrote patches but couldn't grade them. Your laptop has Docker:

```bash
./scripts/grade_predictions.sh
```

Walks every `results/*/predictions.jsonl` and runs the official SWE-bench evaluator subprocess. For each, it:

1. Pulls the per-instance Docker image (first time only — cached afterward)
2. Applies the patch, runs the hidden test command
3. Writes a results JSON
4. **Updates `results/<config>/summary.json` in place** with the real passed/failed counts

To grade a single config only: `./scripts/grade_predictions.sh results/full_system`.

Re-running is safe — the evaluator caches successful instances. The first few grades will be slow (~5–15 min per instance) while Docker pulls the per-repo image; subsequent grades on the same repo are fast.

---

## Step 4 — Generate paper-ready artifacts (~30 sec)

```bash
.venv/bin/python -m analysis.analyze
```

Output:

- **stdout:** per-config table (`config | n | pass | rate | $/traj | front% | esc%`)
- `analysis/summary.csv` — the table for the paper
- `analysis/per_step.csv` — raw per-step data for custom plots
- `figures/pareto.pdf` — **the headline figure** (cost vs success, one point per config)
- `figures/latency_cdf.pdf` — per-step latency by config

If you need a different cost model (NVIDIA changes pricing, or you want DeepSeek-V3 list prices instead), edit `PRICE_PER_1K` at the top of `analysis/analyze.py` and re-run — it reads existing data, no re-running experiments needed.

---

## At the end you have

```
analysis/summary.csv          <- paper table
analysis/per_step.csv         <- raw data for custom plots
figures/pareto.pdf            <- headline figure
figures/latency_cdf.pdf       <- secondary figure
results/<config>/summary.json <- graded per-config stats (all 5 configs)
results/<config>/predictions.jsonl
results/<config>/trajectories/*.json
runs/trajectories/*.json
```

---

## Realistic expectations (n=20 caveat)

With 20 issues per config:
- Pass rates for SWE-bench Lite at this model class typically land in the **15–40%** range; a 10-point gap between configs corresponds to ~2 issues. Visible on the Pareto plot but **inside CI overlap** at n=20.
- State this in the paper's limitations section: *"primary evaluation on a 20-issue stratified subset due to compute budget; we report point estimates and note that 95% Wilson CIs span ±20pp."*
- The Pareto plot's *shape* (full_system below all_frontier in cost, above all_local in success) is the load-bearing claim — that holds with much less data than significance does.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `grade_predictions.sh` says Docker not running | Start Docker Desktop, retry |
| First grade is very slow (15+ min) | Normal — pulling the per-repo Docker image; subsequent grades on same repo are fast |
| `analyze.py` says no per-config summaries found | Grading didn't run or didn't update summaries. Re-run `grade_predictions.sh` |
| `summary.csv` shows passed=0 for the Mac configs | Means the Mac's predictions never got graded. Run step 3. |
| Cost numbers look wrong | Update `PRICE_PER_1K` in `analysis/analyze.py` |
| No `runs/trajectories/` data | You didn't copy the per-step traces from the Mac → `frontier_step_pct` etc. will be 0/inaccurate |
| NIM returns 404 mid-run | Retired model slug. Update `FRONTIER_MODEL` in `.env` (current: `qwen/qwen3-coder-480b-a35b-instruct`) |

---

## Tests

The Python tests verify the proxy, router, and confidence-gate code. They have nothing to do with this end-to-end run — they're a separate sanity check. Don't run them as part of this plan. If you want to:

```bash
.venv/bin/python -m pytest tests/ -v
```

Expect 46 passing.

---

## Future work (not in scope for the first run)

Worth doing later for a stronger paper:

1. **Bump n to 50** — re-run `full_system` and `format_check` on the Mac with the full 50-issue set overnight; re-grade here; re-run `analyze.py`. Replaces the n=20 Pareto with a tighter one.
2. **Ablations** (GUIDE §5.3) — heuristic-only, confidence-only, failure-recovery-only, threshold sweep. Each is another `CONFIGS=<name>` Mac run with a new config file under `configs/`.
3. **Local-model swap** — re-run `full_system` with DeepSeek-Coder 6.7B as the local model, report the delta against Qwen3-Coder 8B.

Each ablation is ~1 hour with the same infrastructure (sample, run, grade, analyze).
