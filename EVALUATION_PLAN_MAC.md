# Evaluation Plan — Mac Studio

**Run this first.** When it finishes, see `EVALUATION_PLAN_LINUX.md`.

| | |
|---|---|
| **Role** | Run the 2 configs that need the local model |
| **Configs** | `all_local`, `full_system` (n=20 stratified subset each) |
| **Local model** | `qwen2.5-coder:14b` (16K-context variant created by setup.sh) |
| **Wall time** | ~1.5–2 hr |
| **Your attention** | ~10 min smoke test, then walk away |
| **Hardware** | M1 Max 32 GB, no admin |
| **Output** | `results-bundle-<timestamp>.tar.gz` to copy to the laptop |

`random` and `format_check` are deliberately dropped — see "Why only 2 configs" at the bottom.

---

## Step 1 — Set up the repo (~5 min, do once)

```bash
git clone <repo-url>
cd <repo>/coding-agent-router

# Add the API key
cp ../.env.example .env
# Edit .env: replace NVIDIA_API_KEY=nvapi-... with your real key from build.nvidia.com.
```

---

## Step 2 — Run setup, then **smoke test before walking away**

```bash
./scripts/setup.sh               # ~15 min: venv, ollama, qwen2.5-coder:14b pull
./scripts/smoke_test.sh          # ~5–10 min: 1 real issue end-to-end
```

`smoke_test.sh` is the load-bearing step. It asserts every failure mode from the
2026-05-17 library run is fixed:

- per-issue session IDs (was collapsing to `traj=default`)
- non-empty patch (was empty across all 50)
- both backends called (heuristic escalation fires)
- NIM 429s get fail-soft to local (was retry-storming for 30 min)

**If smoke prints `SMOKE FAIL`, stop and read the diagnostics.** Do not start
the real run — you'll waste another library session.

| smoke says | fix |
|---|---|
| `local model … not present` | `LOCAL_MODEL_PULL=qwen2.5-coder:14b ./scripts/setup.sh` |
| `patch is non-empty: length=0` | local model didn't produce edits → check `~/.config/opencode/opencode.jsonc` exists with `"hybrid"` key |
| `no requests fell back to traj=default` failed | session-ID propagation regressed; check `XDG_CONFIG_HOME` plumbing in `run_swe_bench.py` |
| `proxy /healthz` failed | model name mismatch → `ollama list` should show `qwen2.5-coder-14b-16k:latest` |
| `both local + frontier called` failed | NVIDIA_API_KEY missing/invalid, or heuristic thresholds wrong for this issue — re-run with `SMOKE_ISSUE_IDX=3` |

---

## Step 3 — Start the real run

Only after `SMOKE PASS`:

```bash
./scripts/run_all.sh             # default CONFIGS="all_local full_system"
```

What this does:

1. Confirms setup is idempotent (no-op if already done)
2. Samples 50 issues + carves the 20-issue subset (idempotent)
3. Runs `all_local` (~30 min): pure local, 20 issues, parallel=2
4. Runs `full_system` (~60–90 min): heuristic router, 20 issues, parallel=2
5. Bundles `results-bundle-<timestamp>.tar.gz` at the repo root

Watch progress: `tail -f logs/all_local.proxy.log` then `logs/full_system.proxy.log`.

---

## Step 4 — Transfer to the laptop (~2 min)

```bash
ls -lh results-bundle-*.tar.gz   # confirm
scp results-bundle-*.tar.gz user@<laptop>:~/Documents/coding-agent-router/coding-agent-router/
```

Move to `EVALUATION_PLAN_LINUX.md`.

---

## Why only 2 configs

The Pareto plot needs **(cheap baseline, full system, expensive baseline)** to make
its claim. That's `all_local` (Mac), `full_system` (Mac), `all_frontier` (laptop).

Dropped:
- **`random`** — sanity check ("does *any* frontier traffic help"). The Pareto already
  answers that by comparing all_local vs full_system. Not load-bearing.
- **`format_check`** — its confidence gate didn't fire on the 2026-05-17 run, suggesting
  Ollama isn't returning logprobs for this model. Diagnosable but not worth burning Mac
  hours on; lives in the paper's "future work" section.

Adding them back later is one config-file edit + one config rerun each, ~1 hr Mac time.

---

## If something breaks

The repo has a `CLAUDE.md` runbook. If you're using Claude Code on the Mac, just
say *"the script is failing, fix it"* — Claude will consult that file.

| Symptom | Quick fix |
|---|---|
| `setup.sh` 404 on ollama download | The script auto-falls back to the .zip. If both fail, you're behind a corporate proxy; download manually and place the binary at `~/.local/bin/ollama` |
| Out of memory mid-trajectory | `LOCAL_MODEL_PULL=qwen2.5-coder:7b ./scripts/setup.sh` then re-run smoke. 14b is borderline on a busy machine |
| All issues fail in ~0.3 s with `opencode exited 1` | `hybrid/proxy-default` provider not registered. Check `~/.config/opencode/opencode.jsonc` exists and has a `"hybrid"` key |
| One issue hangs | Hard 30-min timeout fires automatically and the harness moves on |
| Library kicks you off / Mac sleeps mid-run | Re-running `run_all.sh` is safe — `--resume` skips finished issues |

---

## Tests

The Python tests verify the proxy, router, and confidence-gate code. Separate from
this run. Don't invoke as part of the plan. If you want to:

```bash
.venv/bin/python -m pytest tests/ -v
```

Expect 46 passing.
