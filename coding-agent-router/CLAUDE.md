# Hybrid router — Claude Code runbook

This file is loaded automatically by Claude Code when it opens this repo.
It exists so Claude can recover from common Mac-side failures without the user
having to step in.

## Project in one paragraph

A FastAPI proxy that sits between OpenCode and two LLM backends (local Ollama,
frontier NVIDIA NIM). Per request, a router decides which backend to call.
The benchmark runs SWE-bench Lite issues through OpenCode → proxy → backend(s),
producing per-issue patches that get graded by the SWE-bench evaluator.
Full design is in `../GUIDE.md`.

## The one command the user runs

```
./scripts/run_all.sh
```

Honors `SMOKE=1` (3 issues × 3 configs, ~15 min) and `NO_EVAL=1`
(skip the Docker evaluator).

## Failure → fix table

If `run_all.sh` errors, look at the last ~50 lines of its output to identify
which step failed, then apply the matching fix below. Re-run the same command —
all scripts are idempotent.

### Setup step (`scripts/setup.sh`)

| Symptom | Likely cause | Fix |
|---|---|---|
| `curl: (22) ... 403` on ollama download | macOS Gatekeeper / proxy blocking GitHub | Try the alternative URL: `curl -fsSL https://github.com/ollama/ollama/releases/latest/download/ollama-darwin -o ~/.local/bin/ollama` |
| `ollama: command not found` after install | `~/.local/bin` not on PATH | `export PATH="$HOME/.local/bin:$PATH"` (already done in run_all.sh, but check the user's interactive shell) |
| `port 11434 already in use` | ollama already running, possibly an old version | `lsof -i :11434` to find PID, kill it, retry |
| pip install fails on `swebench` | macOS Xcode CLI tools missing | `xcode-select --install` (user prompt, no admin needed). If still failing, install without swebench: `.venv/bin/pip install -r requirements.txt --no-deps` is NOT safe — instead, remove swebench from requirements.txt for this Mac (we don't need its evaluator) |
| opencode installer prompts for input | non-interactive install issue | `yes "" \| curl -fsSL https://opencode.ai/install \| bash` |

### Run step (`benchmark/run_experiments.sh`)

| Symptom | Likely cause | Fix |
|---|---|---|
| `proxy did not become healthy` | ollama down, or wrong model name | Check `logs/<config>.proxy.log`. Common error: `model not found` → confirm `ollama list` shows the 16K variant; if not, re-run `scripts/setup.sh` |
| All issues fail with `error: timeout` | Local model is too slow on this hardware | Switch to a smaller model: `LOCAL_MODEL_PULL=qwen2.5-coder:3b ./scripts/setup.sh` then re-run. Update `LOCAL_MODEL` env to match. |
| NIM returns 401 | bad / missing API key | `grep NVIDIA_API_KEY .env`. Key should start with `nvapi-`. Re-source the env or restart the proxy. |
| NIM returns 404 page-not-found | retired model slug | Verify with `curl https://integrate.api.nvidia.com/v1/models -H "Authorization: Bearer $NVIDIA_API_KEY" \| grep qwen`. Update `FRONTIER_MODEL` in `.env` accordingly. Current default is `qwen/qwen3-coder-480b-a35b-instruct`. |
| `opencode: command not found` mid-run | PATH lost between subprocess and runner | The benchmark runner inherits the env from the shell. Re-run `run_all.sh` from a shell that has `~/.opencode/bin` and `~/.local/bin` on PATH. |
| `git clone … Permission denied` for a repo | shallow clone hit GitHub rate limits | Wait 10 min and re-run with `--resume`; the queue will only retry failed issues |
| Empty `predictions.jsonl` for a config | opencode produced no diff for any issue | Check `results/<config>/trajectories/*.json` `opencode_returncode` — if non-zero, look at `opencode_stdout`. Often the local model is failing to call tools at all (too small or wrong format); switch model. |

### Disk / RAM pressure

| Symptom | Fix |
|---|---|
| Out-of-memory on 16 GB MBP | Use `SMOKE=1` (forces 3B model) — the 8B default is too tight |
| `runs/` dir grows unbounded | Workdirs get cleaned automatically unless `--keep-workdirs` is passed. Check that none are leaking; `du -sh runs/* results/*/workdirs` |

## What "done" looks like

After a successful `run_all.sh` you should have:

- `results/<config>/summary.json` for each config (passed/total)
- `results/<config>/predictions.jsonl` (one line per issue)
- A tarball `results-bundle-<timestamp>.tar.gz` at the repo root

If `NO_EVAL=1` was set, `passed=0` in every summary — that's expected.
Grading happens later on a Docker-having machine.

## Don't

- **Don't** modify the proxy or router code on the Mac mid-run. If something
  looks broken, fix it in the next iteration.
- **Don't** delete `runs/` while the experiment is running — it's where
  trajectory state lives.
- **Don't** try to install Docker. The user explicitly does not have admin.
  Use the `NO_EVAL=1` path.
- **Don't** add new dependencies to `requirements.txt` casually — the
  user-level pip install on a library machine is fragile.

## Memory

Useful context lives in `~/.claude/projects/-…/memory/` — read those when the
user references "the M1 Max Mac Studio" or asks about hardware.
