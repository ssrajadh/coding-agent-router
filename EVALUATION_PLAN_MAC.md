# Evaluation Plan — Mac Studio

**Run this first.** When it finishes, see `EVALUATION_PLAN_LINUX.md`.

| | |
|---|---|
| **Role** | Run the 4 configs that need the local model |
| **Configs** | `all_local`, `random`, `format_check`, `full_system` |
| **Wall time** | ~3 hours |
| **Your attention** | ~10 min at the start, then walk away |
| **Hardware** | M1 Max 32 GB, no admin |
| **Output** | `results-bundle-<timestamp>.tar.gz` to copy to the laptop |

---

## Step 1 — Set up the repo (~5 min, do once)

```bash
git clone <repo-url>
cd <repo>/coding-agent-router

# Add the API key
cp ../.env.example .env
# Edit .env in any text editor: replace the NVIDIA_API_KEY=nvapi-... line
# with your real key from build.nvidia.com.
```

Don't run `setup.sh` yourself — `run_all.sh` will.

---

## Step 2 — Start the run

```bash
CONFIGS="all_local random format_check full_system" ./scripts/run_all.sh
```

What this does automatically:

1. **Bootstrap (~10 min)** — `setup.sh`:
   - Creates Python venv from `requirements.txt`
   - Downloads the ollama binary into `~/.local/bin` (falls back to extracting from `Ollama-darwin.zip` if the direct binary URL 404s)
   - Pulls `qwen3-coder:8b` and creates the 16K-context variant
   - User-installs opencode and writes its `hybrid/proxy-default` provider config
2. **Sample issues** — writes `benchmark/issues.json` (50 stratified) and `issues_subset20.json` (20)
3. **Run 4 configs** — for each: spin up proxy in the right mode, process the 20-issue subset in parallel, kill proxy, next config. Detects "no Docker on this Mac" and sets `NO_EVAL=1` so each config produces `predictions.jsonl` instead of trying to grade in place.
4. **Bundle** — `results-bundle-<timestamp>.tar.gz` at the repo root with everything portable

Watch progress in another shell: `tail -f logs/<config>.proxy.log`.

---

## Step 3 — Transfer to the laptop (~2 min)

When the Mac finishes:

```bash
ls -lh results-bundle-*.tar.gz   # confirm the bundle exists
```

Get it to the laptop by whichever channel works (scp, AirDrop, USB stick):

```bash
# Option A: scp directly (replace <laptop>)
scp results-bundle-*.tar.gz user@<laptop>:~/Documents/coding-agent-router/coding-agent-router/

# Option B: copy the four results dirs + trajectories (smaller, more selective)
scp -r results/{all_local,random,format_check,full_system} \
       runs/trajectories \
       user@<laptop>:~/Documents/coding-agent-router/coding-agent-router/results-from-mac/
```

After this point, the Mac is done. Move to `EVALUATION_PLAN_LINUX.md`.

---

## If something breaks

The repo has a `CLAUDE.md` runbook. If you're using Claude Code on the Mac, just say *"the script is failing, fix it"* — Claude will consult that file and resolve most things on its own.

The most common issues and the quick fixes are:

| Symptom | Quick fix |
|---|---|
| `setup.sh` 404 on ollama download | The script auto-falls back to the .zip. If both fail, you're behind a corporate proxy; download manually and place the binary at `~/.local/bin/ollama` |
| Out of memory mid-trajectory | `LOCAL_MODEL_PULL=qwen2.5-coder:3b ./scripts/setup.sh` then re-run. `--resume` skips finished issues |
| All issues fail in ~0.3 s with `opencode exited 1` | The `hybrid/proxy-default` provider isn't registered. Check `~/.config/opencode/opencode.jsonc` exists and has a `"hybrid"` key. Re-run `setup.sh` if not |
| Proxy not healthy after 30 s | `cat logs/<cfg>.proxy.log` — usually the local model name doesn't match. Confirm `ollama list` shows `qwen3-coder-8b-16k` |
| One issue hangs | Hard timeout fires automatically after 30 min and the harness moves on |

Re-running `./scripts/run_all.sh` is always safe — every step is idempotent and the harness resumes from per-issue result files.

---

## Tests

The Python tests verify the proxy, router, and confidence-gate code. They have nothing to do with this end-to-end run — they're a separate sanity check. Don't run them as part of this plan. If you want to:

```bash
.venv/bin/python -m pytest tests/ -v
```

Expect 46 passing.
