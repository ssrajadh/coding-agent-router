# Evaluation Plan — End-to-End Runbook

A single document covering everything from "I just sat down at the Mac" to "I have figures for the paper." Roughly **3 hours of wall time** if you start the Mac and laptop sides in parallel; **30 minutes of your attention** spread across that window.

## At a glance

| Machine | Configs | Role | Wall time |
|---|---|---|---|
| **Mac Studio (M1 Max 32 GB, no admin)** | `all_local`, `random`, `format_check`, `full_system` | Runs everything that needs the local model | ~3 hr |
| **Linux laptop (with Docker)** | `all_frontier` | Network-bound config + grades everything | ~40 min runtime, ~30 min grading |

End deliverables:
- `analysis/summary.csv` — per-config success rate, cost, escalation rate
- `figures/pareto.pdf` — the headline cost-vs-success plot
- `figures/latency_cdf.pdf` — per-step latency by config
- `results/<config>/predictions.jsonl` — graded patches (in case you re-grade later)

---

## Pre-flight (5 min, do once)

### On the Mac

```bash
# 1. Clone
git clone <repo-url>
cd <repo>/coding-agent-router

# 2. Add the API key
cp ../.env.example .env
# Edit .env in any text editor and replace the NVIDIA_API_KEY=nvapi-... line
#   with your real key from build.nvidia.com.
```

That's it. Don't run `setup.sh` yourself — `run_all.sh` will.

### On the Linux laptop

The laptop is already set up (it's where this code was developed). Just confirm:

```bash
cd ~/Documents/coding-agent-router/coding-agent-router
grep -q '^NVIDIA_API_KEY=nvapi-' .env && echo "key OK" || echo "missing key in .env"
git pull   # if you've been editing on the Mac
```

---

## Step 1 — Start the Mac run

On the Mac, in a terminal you can leave open for ~3 hours:

```bash
cd <repo>/coding-agent-router
CONFIGS="all_local random format_check full_system" ./scripts/run_all.sh
```

What happens automatically:
- `setup.sh` installs Python venv, downloads the ollama binary into `~/.local/bin`, pulls `qwen3-coder:8b`, creates the 16K context variant, installs opencode user-side.
- `sample_issues.py` writes `benchmark/issues.json` (50 stratified issues) and `issues_subset20.json` (20).
- Each config runs in turn: proxy spins up in the right mode, the 20-issue subset is processed in parallel, proxy is killed, next config starts.
- Because the Mac has no Docker, the script auto-detects this and sets `NO_EVAL=1`. Each config produces `results/<config>/predictions.jsonl` instead of in-place grading.
- A `results-bundle-<timestamp>.tar.gz` is written at the repo root when everything finishes.

You can `tail -f logs/<config>.proxy.log` from another shell to watch progress.

### What to do if something breaks

The Mac has a `CLAUDE.md` runbook in the repo root. If you're using Claude Code on the Mac as a co-pilot, just say *"the script is failing, fix it"* — it'll consult that file and likely resolve the issue without you intervening.

Common ones to know about anyway:
- Out of memory mid-trajectory → re-run with `LOCAL_MODEL_PULL=qwen2.5-coder:3b ./scripts/setup.sh` then re-run `run_all.sh` with the new model. The runs you already completed are skipped via `--resume`.
- NIM 401 → key wrong in `.env`. Restart the run after fixing.
- An issue hangs → it'll timeout at 1800 s automatically and the next issue continues.

---

## Step 2 — Start the laptop run (in parallel)

While the Mac is grinding, on the laptop:

```bash
cd ~/Documents/coding-agent-router/coding-agent-router
FRONTIER_ONLY=1 ./scripts/run_all.sh
```

The `FRONTIER_ONLY=1` flag:
- skips every Ollama step in `setup.sh` (no model needed)
- forces `CONFIGS=all_frontier`
- keeps Docker-based grading **on** (because the laptop has Docker), so this config gets fully graded inline

Expected: ~40 min runtime + ~10 min grading. When done you'll have:

```
results/all_frontier/summary.json        # passed/total counts
results/all_frontier/predictions.jsonl
results/all_frontier/trajectories/*.json
```

---

## Step 3 — Copy Mac results back to the laptop (~5 min)

Once the Mac finishes (or at any point — you can incrementally grade what's done):

```bash
# On the laptop, replace <mac-host> with hostname or IP
scp -r '<mac-user>@<mac-host>:<repo>/coding-agent-router/results/{all_local,random,format_check,full_system}' results/
scp -r '<mac-user>@<mac-host>:<repo>/coding-agent-router/runs/trajectories' runs/
```

If SSH isn't set up, the tarball `results-bundle-<timestamp>.tar.gz` from the Mac contains everything; transfer it (USB, AirDrop, whatever) and:

```bash
tar -xzf results-bundle-*.tar.gz
```

---

## Step 4 — Grade the Mac's predictions on the laptop (~30 min)

The Mac wrote patches but couldn't grade them. The laptop has Docker, so:

```bash
./scripts/grade_predictions.sh
```

This walks every `results/*/predictions.jsonl` and runs the official SWE-bench evaluator subprocess. For each, it:
1. Pulls the per-instance Docker image (first time only — cached afterward)
2. Applies the patch, runs the hidden test command
3. Writes a results JSON
4. **Updates `results/<config>/summary.json` in place** with the real passed/failed counts

Watch progress in the terminal. Re-running is safe — the evaluator caches successful instances.

If you only want one config: `./scripts/grade_predictions.sh results/full_system`.

---

## Step 5 — Generate paper-ready artifacts (~30 sec)

```bash
.venv/bin/python -m analysis.analyze
```

Output:
- Prints a per-config table to stdout (`config | n | pass | rate | $/traj | front% | esc%`)
- Writes `analysis/summary.csv` and `analysis/per_step.csv`
- Writes `figures/pareto.pdf` (the headline) and `figures/latency_cdf.pdf`

The Pareto plot has one point per config in (cost per trajectory, success rate) space. That's the figure you put on the first page of the paper.

If you need a different cost model (NVIDIA changes pricing, you want to use DeepSeek-V3 prices, etc.), edit `PRICE_PER_1K` at the top of `analysis/analyze.py` and re-run — it reads existing data, no re-running experiments.

---

## What you have at the end

```
analysis/summary.csv         <- table for the paper
analysis/per_step.csv        <- raw per-step data if you want custom plots
figures/pareto.pdf           <- headline figure
figures/latency_cdf.pdf      <- secondary figure
results/<config>/summary.json   <- graded per-config stats
results/<config>/predictions.jsonl   <- patches (for re-grading or sharing)
results/<config>/trajectories/*.json <- per-issue details
runs/trajectories/*.json     <- per-step proxy traces (full request/response)
```

---

## Realistic expectations (n=20 caveat)

With 20 issues per config:
- Pass rates for SWE-bench Lite at this model class typically land in the **15–40%** range; a 10-point gap between configs corresponds to ~2 issues. That gap is **visible on the Pareto plot but inside CI overlap** at n=20.
- You should explicitly say in the limitations section: *"primary evaluation on a 20-issue stratified subset due to compute budget; we report point estimates and note that 95% Wilson CIs span ±20pp."*
- The Pareto plot's *shape* (full_system below all_frontier in cost, above all_local in success) is the load-bearing claim — that holds with much less data than significance does.

If you want n=50 later: re-run with `CONFIGS=full_system` and the default issue file on the Mac overnight, then re-grade. Existing 20-issue results auto-resume.

---

## Troubleshooting cheat-sheet

| Problem | Where to look | Fix |
|---|---|---|
| Mac: `setup.sh` fails on ollama binary | `logs/setup.log` (if you teed it) | Try the GitHub URL in `CLAUDE.md` |
| Mac: all issues fail with `error: timeout` | `results/<cfg>/trajectories/*.json` | Switch to a smaller model — see CLAUDE.md |
| Laptop: `grade_predictions.sh` says Docker not running | terminal output | Start Docker Desktop |
| Laptop: first grade is slow (15+ min per instance) | normal | Pulling the per-repo Docker image; subsequent grades on the same repo are fast |
| `analyze.py` says no per-config summaries found | `ls results/*/summary.json` | Grading didn't run or didn't update summaries. Re-run grading. |
| Cost numbers look wrong | `analysis/analyze.py` PRICE_PER_1K | Update rates if NVIDIA's pricing changed |
| No `runs/trajectories/` on the laptop | you didn't copy them from Mac | Step 3 includes them; without them, `frontier_step_pct` etc. will be inaccurate |

---

## Future work (not in scope for the first ~3-hour run)

These take more time and are worth doing later for the final report:

1. **Bump n to 50** — re-run `full_system` and the strongest baseline (`format_check`) on the full 50-issue set overnight. Re-grade. Pareto plot replaces the n=20 version.
2. **Ablations** (GUIDE §5.3) — heuristic-only, confidence-only, failure-recovery-only, threshold sweep. Each is another `CONFIGS=<name>` run with a new config file.
3. **Local-model swap** — re-run `full_system` with DeepSeek-Coder 6.7B as the local model, report the delta.

Each ablation is ~1 hour with the same infrastructure.
