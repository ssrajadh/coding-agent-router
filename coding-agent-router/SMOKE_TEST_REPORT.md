# Smoke Test Report — `scripts/run_all.sh`

**Date:** 2026-05-16
**Host:** macOS (Darwin 24.6.0, arm64)
**Mode:** `SMOKE=1` (3 issues × 3 configs, `NO_EVAL=1` auto-set — no Docker on box)

## TL;DR

The smoke run now completes end-to-end (~8 min) across `all_local`, `format_check`, and `full_system`. Three independent bugs blocked the pipeline; all are fixed. `passed=0` is expected with `NO_EVAL=1` — predictions JSONL is written for later Docker grading. The `full_system` config produces real NIM-side errors that are pre-existing (model slug / rate-limit) and not introduced by the smoke wiring.

Final summary the run printed:

```
all_local:     passed=0/3 (0.0%)  errors=0
format_check:  passed=0/3 (0.0%)  errors=0
full_system:   passed=0/3 (0.0%)  errors=2   (frontier-backend 400/429 from NIM)
Bundle:        results-bundle-20260516-182207.tar.gz
```

---

## Issues found and fixed

### 1. Ollama download URL returned 404

**Symptom:**

```
→ Downloading ollama binary to /Users/.../ollama
curl: (56) The requested URL returned error: 404
```

**Root cause:** `scripts/setup.sh` curls `https://ollama.com/download/ollama-darwin`, which now 307-redirects to `Ollama-darwin.zip` (the GUI bundle). The bare-binary URL no longer exists.

**Fix:** installed ollama via Homebrew (`brew install ollama`). `setup.sh` is idempotent and short-circuits on `command -v ollama`, so subsequent runs skip the broken download. No code change made — this is an environmental fix for this Mac. A more durable fix would be to update `step_ollama_binary` to fall back to `brew install ollama` (or extract the binary from the `.zip`) when the direct URL 404s.

### 2. `opencode run --max-steps` flag doesn't exist

**Symptom:** every issue printed `opencode exited 1` in ~0.3 s with an empty patch. `opencode run --help` confirms `--max-steps` is not an option in v1.15.3 — passing it causes opencode to dump usage and exit 1 without ever calling the proxy.

**Fix:** `benchmark/run_swe_bench.py` — dropped `--max-steps`, added `--model hybrid/proxy-default` and `--dangerously-skip-permissions` so the run is non-interactive.

```python
return sp.run(
    ["opencode", "run",
     "--model", "hybrid/proxy-default",
     "--dangerously-skip-permissions",
     prompt],
    ...
)
```

### 3. Opencode had no `hybrid` provider configured

**Symptom:** even with the args fixed, opencode had no way to route through the local proxy — `opencode models` listed only the bundled `opencode/*` free models. `OPENCODE_PROVIDER=hybrid` env var alone does nothing without a matching config entry.

**Fix:** created `~/.config/opencode/opencode.jsonc` with an `@ai-sdk/openai-compatible`-backed `hybrid` provider exposing `proxy-default`:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "hybrid": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Hybrid Proxy",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "not-needed"
      },
      "models": { "proxy-default": { "name": "Hybrid Default" } }
    }
  }
}
```

After this, `opencode models` lists `hybrid/proxy-default` and the harness can target it.

### 4. BSD `sed` ate the YAML value, breaking router mode

**Symptom:** after issues 1–3 were fixed, the proxy raised `NotImplementedError:  all_local` (note the leading space) on every chat-completions request. All 3 issues then timed out at 30 min each (~30 min per issue × 3 = ~90 min wasted before I caught this).

**Root cause:** `benchmark/run_experiments.sh::parse_yaml_field` used `sed -E "s/^${key}:\s*//"`. BSD `sed` (macOS default) does not interpret `\s`, so the leading whitespace after `router_mode:` survived. The header `mode= all_local` in the log gave it away.

**Fix:** swapped `\s` for the POSIX class `[[:space:]]`:

```bash
grep -E "^${key}:" "$file" | head -n1 \
  | sed -E "s/^${key}:[[:space:]]*//; s/^['\"]//; s/['\"]$//"
```

### 5. 30-minute per-issue timeout is too long for smoke

**Symptom:** without a step cap, the small 3B model loops; smoke runs hit the hard-coded 1800 s timeout per issue. Three issues × three configs = up to 4.5 h on a smoke run.

**Fix:**
- `benchmark/run_swe_bench.py`: `run_opencode` now reads `OPENCODE_TIMEOUT` (default 1800 s).
- `benchmark/run_experiments.sh`: in `SMOKE=1` mode, defaults `OPENCODE_TIMEOUT=300`. Total smoke time dropped from "hours" to ~8 min.

---

## Pre-existing issues observed (not fixed in this pass)

### `full_system` frontier backend returns 400 / 429 from NIM

`logs/full_system.proxy.log` shows repeated:

```
POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 400 Bad Request"
POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
ERROR:proxy:backend failure: RetryError[... HTTPStatusError ...]
```

The two `full_system` errors in the summary are the two issues that timed out waiting on this backend (the third happened to land while local was the chosen backend). This matches the failure modes documented in `CLAUDE.md` ("NIM returns 404 page-not-found — retired model slug"). Likely fix: refresh `FRONTIER_MODEL` in `.env` and confirm with `curl https://integrate.api.nvidia.com/v1/models -H "Authorization: Bearer $NVIDIA_API_KEY"`. Out of scope for a smoke run.

### Empty patches across all configs

The 3B-coder model in smoke mode finished each issue quickly but produced no diff. Expected — these are real SWE-bench Lite issues; a 3B model with no step budgeting isn't going to solve `astropy` or `django` bugs. Wiring is validated by the trajectories landing in `runs/trajectories/` and the proxy log showing routed requests. Production runs would use the 8B variant.

---

## Files changed

- `benchmark/run_swe_bench.py` — opencode CLI args; `OPENCODE_TIMEOUT` env hook.
- `benchmark/run_experiments.sh` — POSIX-portable YAML sed; `OPENCODE_TIMEOUT=300` default in smoke.
- `~/.config/opencode/opencode.jsonc` — registered `hybrid/proxy-default` provider (machine-local, not in repo).

## Environment fixes (not in repo)

- `brew install ollama` (worked around the 404 from `ollama.com/download/ollama-darwin`).
- `.env` was bootstrapped from `../.env.example` automatically on first run.

## Artifacts

- `results/all_local/`, `results/format_check/`, `results/full_system/` — `summary.json`, `predictions.jsonl`, `trajectories/*.json` each.
- `runs/trajectories/<config>__<instance>.json` — per-step proxy traces.
- `logs/<config>.proxy.log` — full proxy stdout per config.
- `results-bundle-20260516-182207.tar.gz` — portable bundle for grading on a Docker host.

## Re-running

```
SMOKE=1 ./scripts/run_all.sh
```

is now idempotent end-to-end on this Mac (~8 min). For a full (non-smoke) run, you'd want to (a) update `FRONTIER_MODEL` to a current NIM slug, and (b) drop the `OPENCODE_TIMEOUT=300` override (or raise it).
