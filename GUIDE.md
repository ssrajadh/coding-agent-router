# Per-Step Hybrid Routing for Coding Agents — Technical Build Guide

A practical, end-to-end implementation guide for building the routing proxy system over a 5-week project window. Written assuming three teammates working in parallel with Claude Code as a coding assistant.

---

## 0. Project shape at a glance

```
[OpenCode]  --(OpenAI-compatible HTTP)-->  [Routing Proxy (FastAPI)]
                                                   |
                                       Routing decision per request
                                                   |
                          +------------------------+------------------------+
                          v                                                 v
              [Local: Ollama / Qwen3-Coder 8B]                [Frontier: NVIDIA NIM /
                                                              Qwen3-Coder 32B or DeepSeek-V3]
```

Repo layout we'll build toward:

```
hybrid-router/
├── proxy/
│   ├── server.py            # FastAPI app, OpenAI-compatible endpoints
│   ├── backends.py          # Ollama, NIM, Anthropic adapters
│   ├── router.py            # Routing decision logic
│   ├── trajectory.py        # Per-trajectory state tracker
│   ├── features.py          # Feature extraction from requests
│   ├── confidence.py        # Confidence signal extraction
│   └── config.py            # Settings, env vars
├── benchmark/
│   ├── run_swe_bench.py     # Driver: spin up OpenCode against an SWE-bench issue
│   ├── harness.py           # SWE-bench eval wrapper (patch + test)
│   └── issues.json          # The stratified subset we use
├── analysis/
│   ├── parse_logs.py        # Convert proxy logs to dataframes
│   ├── plots.py             # Pareto, ablations, latency CDFs
│   └── notebooks/
├── configs/
│   ├── all_frontier.yaml
│   ├── all_local.yaml
│   ├── random.yaml
│   ├── format_check.yaml
│   └── full_system.yaml
├── tests/
└── README.md
```

---

## 1. Environment setup (Day 1)

### 1.1 Local machines

Each teammate needs:
- Python 3.11+
- Docker (for SWE-bench evaluation containers)
- Ollama installed and running
- A GPU capable of running Qwen3-Coder 8B at 4-bit quantization. RTX 3090/4090, A4000, or M-series Mac with 16GB+ unified memory all work.

Install Ollama and pull the local model:

```bash
# macOS
brew install ollama
ollama serve  # leave this running in a terminal

# Pull and tune the local model
ollama pull qwen3-coder:8b

# Increase the default context window — Ollama defaults are tiny
ollama run qwen3-coder:8b
>>> /set parameter num_ctx 16384
>>> /save qwen3-coder-16k
>>> /bye
```

**Heads up:** the default 4K context will silently truncate prompts and break tool calling. Always use the 16K-saved variant. This bites everyone on day one.

### 1.2 NVIDIA NIM API key

1. Sign up at `build.nvidia.com` with the NVIDIA Developer Program (free, no card).
2. Generate an API key — it'll be prefixed `nvapi-`.
3. Confirm rate limits in the dashboard top-right (~40 RPM per model).

Test it:

```bash
export NVIDIA_API_KEY="nvapi-..."
curl https://integrate.api.nvidia.com/v1/chat/completions \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen/qwen3-coder-32b-instruct",
    "messages": [{"role": "user", "content": "say hi"}],
    "max_tokens": 20
  }'
```

If this works, you're done with backend setup.

### 1.3 OpenCode

```bash
brew install opencode      # macOS
# or: curl -fsSL https://opencode.ai/install | bash
opencode auth login        # at minimum, set up an Anthropic key for sanity-check runs later
```

Don't try to wire OpenCode to your proxy yet — first we'll build a passthrough proxy and then point OpenCode at it.

### 1.4 Project scaffolding

```bash
mkdir hybrid-router && cd hybrid-router
python -m venv .venv && source .venv/bin/activate
pip install fastapi uvicorn[standard] httpx pydantic pyyaml \
            tenacity python-dotenv pandas matplotlib pytest
```

Create `.env`:

```
NVIDIA_API_KEY=nvapi-...
ANTHROPIC_API_KEY=sk-ant-...   # only for the validation runs in week 4
OLLAMA_URL=http://localhost:11434
NIM_URL=https://integrate.api.nvidia.com/v1
PROXY_PORT=8000
```

---

## 2. Phase 1: Passthrough proxy (Days 1-3)

The first milestone: an OpenAI-compatible HTTP server that forwards requests to a single fixed backend. No routing yet. The point is to confirm OpenCode can talk to your proxy at all.

### 2.1 Minimal FastAPI server

`proxy/server.py`:

```python
import os
import uuid
import time
import json
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx

from .backends import OllamaBackend, NIMBackend
from .router import Router
from .trajectory import TrajectoryStore
from .config import settings

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("proxy")

app = FastAPI()
trajectory_store = TrajectoryStore()
backends = {
    "local": OllamaBackend(settings.ollama_url, model="qwen3-coder-16k"),
    "frontier": NIMBackend(settings.nim_url, settings.nvidia_api_key,
                           model="qwen/qwen3-coder-32b-instruct"),
}
router = Router(mode=settings.router_mode)

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    request_id = str(uuid.uuid4())
    trajectory_id = body.get("user") or request.headers.get("x-session-id", "default")
    traj = trajectory_store.get_or_create(trajectory_id)

    decision = router.decide(body, traj)
    log.info(f"req={request_id} traj={trajectory_id} -> {decision.backend}")

    backend = backends[decision.backend]
    t0 = time.time()
    try:
        response = await backend.chat_completion(body)
    except Exception as e:
        log.exception(f"backend failure: {e}")
        return JSONResponse({"error": str(e)}, status_code=502)
    elapsed = time.time() - t0

    traj.record_step(
        request=body,
        response=response,
        backend=decision.backend,
        decision_reason=decision.reason,
        latency_s=elapsed,
    )

    return JSONResponse(response)

@app.get("/healthz")
def healthz():
    return {"ok": True}
```

### 2.2 Backend adapters

`proxy/backends.py`:

```python
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

class OllamaBackend:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = httpx.AsyncClient(timeout=300.0)

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=8))
    async def chat_completion(self, body: dict) -> dict:
        # Ollama exposes an OpenAI-compatible endpoint at /v1
        payload = {**body, "model": self.model}
        # Force non-streaming for now; we'll handle streaming separately
        payload["stream"] = False
        r = await self.client.post(
            f"{self.base_url}/v1/chat/completions", json=payload
        )
        r.raise_for_status()
        return r.json()


class NIMBackend:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.client = httpx.AsyncClient(timeout=300.0, headers=self.headers)

    @retry(stop=stop_after_attempt(4),
           wait=wait_exponential(min=2, max=30))  # 40 RPM means real backoff
    async def chat_completion(self, body: dict) -> dict:
        payload = {**body, "model": self.model, "stream": False}
        r = await self.client.post(f"{self.base_url}/chat/completions", json=payload)
        if r.status_code == 429:
            raise httpx.HTTPStatusError("rate limited", request=r.request, response=r)
        r.raise_for_status()
        return r.json()
```

**Important details:**
- Always force `stream=False` initially. Streaming complicates logging, retry, and confidence extraction. Add streaming back only if OpenCode requires it.
- Long timeouts (300s) — long-context generations can take a while.
- NIM rate-limits aggressively at 40 RPM. Build retry-with-backoff from day one or you'll lose hours debugging "ghost" failures.

### 2.3 Stub router

`proxy/router.py` (passthrough version):

```python
from dataclasses import dataclass

@dataclass
class Decision:
    backend: str
    reason: str

class Router:
    def __init__(self, mode: str = "all_local"):
        self.mode = mode

    def decide(self, body: dict, trajectory) -> Decision:
        if self.mode == "all_local":
            return Decision("local", "static_all_local")
        if self.mode == "all_frontier":
            return Decision("frontier", "static_all_frontier")
        if self.mode == "random":
            import random
            b = random.choice(["local", "frontier"])
            return Decision(b, f"random:{b}")
        raise NotImplementedError(self.mode)
```

We'll add the full router in week 2.

### 2.4 Trajectory store stub

`proxy/trajectory.py`:

```python
import time
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Trajectory:
    id: str
    steps: list = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def record_step(self, request, response, backend, decision_reason, latency_s):
        self.steps.append({
            "ts": time.time(),
            "backend": backend,
            "reason": decision_reason,
            "latency_s": latency_s,
            "request_messages": request.get("messages", []),
            "request_tools": request.get("tools", []),
            "response_choice": response.get("choices", [{}])[0],
            "response_usage": response.get("usage", {}),
        })

class TrajectoryStore:
    def __init__(self):
        self._store = {}

    def get_or_create(self, trajectory_id: str) -> Trajectory:
        if trajectory_id not in self._store:
            self._store[trajectory_id] = Trajectory(id=trajectory_id)
        return self._store[trajectory_id]

    def dump(self, path: str):
        import json
        with open(path, "w") as f:
            json.dump({k: v.__dict__ for k, v in self._store.items()}, f, default=str)
```

### 2.5 Wire OpenCode to the proxy

Run the proxy: `uvicorn proxy.server:app --host 127.0.0.1 --port 8000`.

In OpenCode's config (`~/.config/opencode/opencode.json`):

```json
{
  "providers": {
    "hybrid": {
      "name": "hybrid-proxy",
      "baseURL": "http://127.0.0.1:8000/v1",
      "apiKey": "not-needed",
      "models": ["proxy-default"]
    }
  },
  "agents": {
    "coder": { "model": "hybrid.proxy-default" }
  }
}
```

Run OpenCode against a trivial prompt: "what's in this directory?". Confirm in proxy logs that requests are flowing through.

**Common gotchas:**
- OpenCode may complain about missing `tools` field — the proxy must forward it as-is.
- Ollama's OpenAI compatibility layer doesn't always populate `usage` token counts. Track them yourself by tokenizing the input/output if needed.
- Tool calls in responses must use the exact OpenAI format (`tool_calls` array with `id`, `type`, `function.name`, `function.arguments`). Both Ollama and NIM mostly do this correctly, but watch for edge cases.

By end of day 3 you should have OpenCode successfully completing one trivial task entirely through the proxy in `all_local` mode and entirely through `all_frontier` mode.

---

## 3. Phase 2: SWE-bench harness (Days 4-7)

### 3.1 Pick the issue subset

Clone SWE-bench Lite locally for the issue list:

```bash
pip install swebench
python -c "from swebench.harness.test_spec import make_test_spec; from datasets import load_dataset; \
           ds = load_dataset('princeton-nlp/SWE-bench_Lite', split='test'); \
           print(ds.column_names); print(len(ds))"
```

Stratified sampling — pick 50 issues across:
- 5 repos minimum (sympy, django, sklearn, flask, requests are common)
- Mix of "fail-to-pass" counts (number of tests the patch must turn green)
- Mix of issue body lengths

Save the chosen `instance_id`s into `benchmark/issues.json`. Document your sampling code so it's reproducible.

### 3.2 Run an issue end-to-end

This is the most painful integration step. The flow per issue:

1. SWE-bench gives you: a repo URL, a base commit, an issue description, and a hidden test command.
2. Clone the repo at the base commit into a sandbox.
3. Hand the issue description to OpenCode pointed at that workdir.
4. Let OpenCode run until it stops (or hits a step limit).
5. Diff the workdir against the base commit to extract the patch.
6. Run SWE-bench's official evaluator on the patch to get pass/fail.

`benchmark/run_swe_bench.py` skeleton:

```python
import subprocess, json, shutil, os, uuid, time
from pathlib import Path
from datasets import load_dataset

def setup_repo(instance, workdir):
    workdir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", instance["repo_url"], str(workdir)], check=True)
    subprocess.run(["git", "checkout", instance["base_commit"]], cwd=workdir, check=True)

def run_opencode(instance, workdir, session_id, max_steps=60):
    prompt = (
        f"Resolve this issue:\n\n{instance['problem_statement']}\n\n"
        "Make minimal code changes to fix the issue. Run tests to verify."
    )
    env = os.environ.copy()
    env["OPENCODE_PROVIDER"] = "hybrid"
    env["X_SESSION_ID"] = session_id  # forwarded as header
    # Run opencode in non-interactive mode
    result = subprocess.run(
        ["opencode", "run", "--max-steps", str(max_steps), prompt],
        cwd=workdir, env=env, capture_output=True, text=True, timeout=1800
    )
    return result

def extract_patch(workdir):
    return subprocess.check_output(["git", "diff"], cwd=workdir, text=True)

def evaluate_patch(instance, patch):
    # Use the SWE-bench harness: write patch to a predictions file and call the evaluator
    pred = {"instance_id": instance["instance_id"], "model_patch": patch,
            "model_name_or_path": "hybrid-router"}
    Path("preds.jsonl").write_text(json.dumps(pred) + "\n")
    subprocess.run([
        "python", "-m", "swebench.harness.run_evaluation",
        "--predictions_path", "preds.jsonl",
        "--max_workers", "1",
        "--run_id", "eval-" + str(uuid.uuid4())[:8],
    ], check=False)
    # Parse the results JSON the harness writes
    # ... (see SWE-bench docs for exact path)
    return parse_eval_result(instance["instance_id"])

def main(instances, run_name):
    for instance in instances:
        session_id = f"{run_name}/{instance['instance_id']}"
        workdir = Path(f"runs/{run_name}/{instance['instance_id']}")
        try:
            setup_repo(instance, workdir)
            run_opencode(instance, workdir, session_id)
            patch = extract_patch(workdir)
            result = evaluate_patch(instance, patch)
            print(json.dumps({"instance_id": instance["instance_id"],
                              "passed": result["passed"]}))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
```

**Don't underestimate this step.** SWE-bench harness setup, OpenCode's non-interactive mode quirks, and patch extraction edge cases will eat 2-3 days. Get one issue passing end-to-end before scaling up.

### 3.3 Session-ID plumbing

OpenCode → Proxy must communicate which trajectory each request belongs to so the proxy can keep per-trajectory state. The cleanest way: pass `X-Session-Id` as an HTTP header. Some OpenCode versions support custom headers via config; if not, use a unique `user` field in the request body, which the OpenAI spec passes through.

Verify by hitting the proxy with two parallel sessions and confirming the trajectory store separates them.

### 3.4 First baseline run

Day 7 deliverable: run all 50 issues in `all_local` and `all_frontier` modes. Even if numbers are messy, you have a baseline and a working pipeline.

Expected totals: ~250-500 trajectories' worth of wall-clock by end of project. Each issue takes 5-30 minutes. Parallelize across teammate machines using a shared work queue (`runs/queue/<instance_id>.todo`, claimed via mv-rename).

---

## 4. Phase 3: The router (Days 8-14)

### 4.1 Feature extraction

`proxy/features.py`:

```python
import re
from dataclasses import dataclass
from typing import Optional

ERROR_REGEX = re.compile(
    r"\b(error|traceback|exception|fail(?:ed|ure)?|exit code [1-9])\b",
    re.IGNORECASE,
)
PLAN_REGEX = re.compile(
    r"\b(plan|approach|let'?s think|step by step|first.*then)\b",
    re.IGNORECASE,
)

CHEAP_TOOLS = {"read_file", "list_files", "list_directory", "view", "cat", "ls", "grep"}
EXPENSIVE_TOOLS = {"write_file", "str_replace", "edit_file", "patch", "bash"}

@dataclass
class StepFeatures:
    step_index: int
    msg_count: int
    last_user_msg_len: int
    last_tool_name: Optional[str]
    last_tool_failed: bool
    contains_error_keywords: bool
    contains_plan_keywords: bool
    available_tool_count: int
    is_repeated_action: bool
    recent_failure_count: int

def extract(body: dict, traj) -> StepFeatures:
    messages = body.get("messages", [])
    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    last_user_text = (last_user or {}).get("content", "") if isinstance(
        (last_user or {}).get("content"), str) else ""

    # Look at the last 3 messages for error indicators
    recent_text = " ".join(
        m.get("content", "") if isinstance(m.get("content"), str) else ""
        for m in messages[-3:]
    )

    # Extract last tool call from prior messages (if any)
    last_tool_name = None
    last_tool_failed = False
    for m in reversed(messages):
        if m.get("role") == "tool":
            last_tool_name = m.get("name")
            content = m.get("content", "")
            last_tool_failed = bool(ERROR_REGEX.search(content if isinstance(content, str) else ""))
            break

    return StepFeatures(
        step_index=len(traj.steps),
        msg_count=len(messages),
        last_user_msg_len=len(last_user_text),
        last_tool_name=last_tool_name,
        last_tool_failed=last_tool_failed,
        contains_error_keywords=bool(ERROR_REGEX.search(recent_text)),
        contains_plan_keywords=bool(PLAN_REGEX.search(recent_text)),
        available_tool_count=len(body.get("tools", [])),
        is_repeated_action=traj.last_action_repeated() if hasattr(traj, "last_action_repeated") else False,
        recent_failure_count=sum(1 for s in traj.steps[-5:] if s.get("local_failed")),
    )
```

### 4.2 Heuristic router

`proxy/router.py` (full version):

```python
from dataclasses import dataclass
from .features import extract, CHEAP_TOOLS, EXPENSIVE_TOOLS

@dataclass
class Decision:
    backend: str
    reason: str
    confidence_check: bool = False  # whether to verify with confidence signal

class Router:
    def __init__(self, mode: str, config: dict | None = None):
        self.mode = mode
        self.config = config or {}

    def decide(self, body, traj) -> Decision:
        if self.mode == "all_local":
            return Decision("local", "static")
        if self.mode == "all_frontier":
            return Decision("frontier", "static")
        if self.mode == "random":
            import random
            return Decision(random.choice(["local", "frontier"]), "random")
        if self.mode == "format_check":
            # Try local; escalation happens in server.py based on parse failure
            return Decision("local", "format_check_first", confidence_check=True)
        if self.mode == "full":
            return self._full_decide(body, traj)
        raise NotImplementedError(self.mode)

    def _full_decide(self, body, traj) -> Decision:
        f = extract(body, traj)

        # Hard escalations: if anything is going wrong, go frontier
        if f.last_tool_failed:
            return Decision("frontier", "tool_failed")
        if f.is_repeated_action:
            return Decision("frontier", "repeated_action")
        if f.recent_failure_count >= 2:
            return Decision("frontier", "trajectory_struggling")
        if f.contains_error_keywords and f.step_index > 3:
            return Decision("frontier", "error_in_recent_context")

        # Hard "this is hard" signals
        if f.contains_plan_keywords and f.step_index < 3:
            return Decision("frontier", "initial_planning")
        if f.step_index > self.config.get("depth_threshold", 20):
            return Decision("frontier", "deep_trajectory")

        # Cheap-step signals
        if f.last_tool_name in CHEAP_TOOLS:
            return Decision("local", "post_cheap_tool", confidence_check=True)

        # Default: try local, verify confidence
        return Decision("local", "default_local", confidence_check=True)
```

### 4.3 Confidence and escalation

`proxy/confidence.py`:

```python
import json
from typing import Optional

def parse_response_quality(response: dict) -> dict:
    """Extract confidence signals from a chat completion response."""
    choice = response.get("choices", [{}])[0]
    msg = choice.get("message", {})
    finish_reason = choice.get("finish_reason")

    signals = {
        "finish_reason_ok": finish_reason in ("stop", "tool_calls"),
        "tool_calls_valid": True,
        "json_args_parseable": True,
        "min_logprob": None,
    }

    # Check tool call validity
    for tc in msg.get("tool_calls", []) or []:
        fn = tc.get("function", {})
        if not fn.get("name"):
            signals["tool_calls_valid"] = False
        try:
            json.loads(fn.get("arguments", "{}"))
        except json.JSONDecodeError:
            signals["json_args_parseable"] = False

    # Logprobs (Ollama may or may not return these)
    lp = choice.get("logprobs")
    if lp and lp.get("content"):
        token_lps = [t.get("logprob", 0.0) for t in lp["content"]]
        if token_lps:
            signals["min_logprob"] = min(token_lps)

    return signals

def should_escalate(signals: dict, threshold: float = -3.0) -> tuple[bool, str]:
    if not signals["finish_reason_ok"]:
        return True, "bad_finish_reason"
    if not signals["tool_calls_valid"]:
        return True, "invalid_tool_call"
    if not signals["json_args_parseable"]:
        return True, "unparseable_json"
    if signals["min_logprob"] is not None and signals["min_logprob"] < threshold:
        return True, f"low_confidence_logprob"
    return False, "ok"
```

In `server.py`, the escalation flow:

```python
decision = router.decide(body, traj)
backend = backends[decision.backend]
response = await backend.chat_completion(body)

if decision.confidence_check and decision.backend == "local":
    signals = parse_response_quality(response)
    escalate, reason = should_escalate(signals)
    if escalate:
        log.info(f"escalating: {reason}")
        traj.mark_local_failure()
        response = await backends["frontier"].chat_completion(body)
        decision = Decision("frontier", f"escalated_{reason}")

traj.record_step(...)
return JSONResponse(response)
```

### 4.4 Self-consistency fallback

If logprobs aren't reliably available from Ollama, use self-consistency: send the request 3 times with `temperature=0.7`, hash the resulting tool calls, and escalate if they don't agree. More expensive but more portable. Implementation:

```python
async def self_consistent_local(backend, body, n=3):
    bodies = [{**body, "temperature": 0.7} for _ in range(n)]
    responses = await asyncio.gather(*(backend.chat_completion(b) for b in bodies))
    # Hash tool calls + content
    fingerprints = [_fingerprint(r) for r in responses]
    most_common = max(set(fingerprints), key=fingerprints.count)
    agreement = fingerprints.count(most_common) / n
    return responses[fingerprints.index(most_common)], agreement
```

Default to logprob-based confidence for the main system, document self-consistency as an ablation.

---

## 5. Phase 4: Experiments and ablations (Days 15-21)

### 5.1 Experiment matrix

Five primary configurations × 50 issues = 250 trajectories. Run order matters — start with baselines, end with the full system, so if you hit time pressure you sacrifice ablations not baselines.

| Config           | Mode             | Notes                          |
| ---------------- | ---------------- | ------------------------------ |
| `all_frontier`   | `all_frontier`   | Run on 20-issue subset only    |
| `all_local`      | `all_local`      | Full 50 issues                 |
| `random`         | `random`         | Full 50 issues                 |
| `format_check`   | `format_check`   | Full 50 issues                 |
| `full_system`    | `full`           | Full 50 issues                 |

### 5.2 Driver script

`benchmark/run_experiments.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

for config in all_local random format_check full_system; do
    echo "=== $config ==="
    ROUTER_MODE="$config" uvicorn proxy.server:app --port 8000 &
    PROXY_PID=$!
    sleep 3
    python -m benchmark.run_swe_bench \
        --issues benchmark/issues.json \
        --run-name "$config" \
        --output "results/$config"
    kill $PROXY_PID
    sleep 5
done

# all_frontier on the smaller subset
ROUTER_MODE="all_frontier" uvicorn proxy.server:app --port 8000 &
PROXY_PID=$!
python -m benchmark.run_swe_bench \
    --issues benchmark/issues_subset20.json \
    --run-name all_frontier --output results/all_frontier
kill $PROXY_PID
```

### 5.3 Ablations

Once the main experiments are done, run ablations on the same 50-issue subset:

1. **Heuristic only** — disable confidence and failure-recovery.
2. **Confidence only** — heuristic always picks local; confidence drives all escalation.
3. **Failure-recovery only** — heuristic always picks local; only escalate on observed prior failures.
4. **Threshold sweep** — vary the logprob threshold across {-2, -3, -4, -5}.
5. **Local model swap** — re-run full system with DeepSeek-Coder 6.7B in place of Qwen3-Coder 8B.

Each ablation reuses the same harness with a different config file.

### 5.4 Parallelization

Run 2-3 issues in parallel per machine. Each uses its own `session_id` so trajectory state stays separated. Limit concurrency to keep within NIM's 40 RPM. A simple `asyncio.Semaphore(3)` in the driver is enough.

---

## 6. Phase 5: Analysis and writing (Days 22-30)

### 6.1 Log parsing

The proxy writes per-step JSON logs (or you can tail them from the trajectory store). Convert to a single dataframe:

```python
# analysis/parse_logs.py
import json, pandas as pd
from pathlib import Path

def load_runs(results_dir):
    rows = []
    for run in Path(results_dir).glob("*"):
        for f in (run / "trajectories").glob("*.json"):
            traj = json.loads(f.read_text())
            for step in traj["steps"]:
                rows.append({
                    "config": run.name,
                    "instance_id": traj["id"],
                    "step_idx": step.get("step_idx"),
                    "backend": step["backend"],
                    "reason": step["reason"],
                    "latency_s": step["latency_s"],
                    "prompt_tokens": step["response_usage"].get("prompt_tokens", 0),
                    "completion_tokens": step["response_usage"].get("completion_tokens", 0),
                })
    return pd.DataFrame(rows)
```

Augment with per-issue success from the SWE-bench evaluator.

### 6.2 Cost model

For the paper, report cost in dollars even though we used free tiers, using publicly listed prices:

- Local Qwen3-Coder 8B: $0 (we run it ourselves).
- Frontier (Qwen3-Coder 32B via NIM): use the published per-token rate or a representative comparable rate (DeepSeek-V3 paid pricing is a fair proxy).

```python
PRICE_PER_1K = {
    "local": {"prompt": 0.0, "completion": 0.0},
    "frontier": {"prompt": 0.0006, "completion": 0.0024},  # adjust to current rates
}

def cost(row):
    p = PRICE_PER_1K[row["backend"]]
    return (row["prompt_tokens"]/1000) * p["prompt"] + \
           (row["completion_tokens"]/1000) * p["completion"]
```

### 6.3 Headline figure: Pareto frontier

Cost-vs-success scatter, one point per (config, sample). Each config gets a marker shape. Connect non-dominated points with a Pareto-frontier line.

```python
# analysis/plots.py
import matplotlib.pyplot as plt

def pareto_plot(df_summary):
    fig, ax = plt.subplots(figsize=(7, 5))
    for cfg, grp in df_summary.groupby("config"):
        ax.scatter(grp["cost"], grp["success_rate"], label=cfg, s=80)
    ax.set_xlabel("Cost per trajectory ($)")
    ax.set_ylabel("Task success rate")
    ax.set_title("Cost vs. success on SWE-bench Lite (n=50)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/pareto.pdf")
```

Other figures to produce:
- Latency CDF per config.
- Escalation rate over trajectory depth (bar chart).
- Per-issue-type breakdown (heatmap: configs × repos).
- Ablation table (pandas → LaTeX via `df.to_latex`).

### 6.4 Paper structure (8 pages)

Match the dLoRA paper style. Suggested allocation:

1. **Introduction** (~1 p) — problem, contributions list, headline result preview.
2. **Background and motivation** (~1 p) — coding agents, frontier-tier overspend, prior routing.
3. **System design** (~1.5 p) — architecture figure, three signals, design choices.
4. **Implementation** (~0.5 p) — proxy in Python/FastAPI, integration with OpenCode, models used.
5. **Evaluation** (~2.5 p) — setup, headline Pareto, ablations, per-category breakdown, sensitivity analysis.
6. **Discussion and limitations** (~0.5 p) — calibration sensitivity, local-model dependence, generalization beyond SWE-bench.
7. **Related work** (~0.5 p) — concise; reference RouteLLM, FrugalGPT, BudgetMLAgent, Budget-Aware Agentic Routing.
8. **Conclusion + references** (~0.5 p).

---

## 7. Risk register and mitigations

| Risk                                        | Likelihood | Impact | Mitigation |
| ------------------------------------------- | ---------- | ------ | ---------- |
| OpenCode integration takes longer than planned | High    | Medium | Build passthrough first, lock by end of week 1. If blocked, fall back to a minimal custom ReAct loop. |
| Local model has poor tool-calling reliability | Medium  | High   | Fix with 16K context, JSON-mode where supported, format-check escalation. Worst case: switch to DeepSeek-Coder. |
| NIM rate limits throttle experiment throughput | Medium  | Medium | Aggressive retry with backoff, parallelize across multiple model endpoints (NIM rate limits are per-model). |
| SWE-bench harness flakiness                  | High    | Medium | Pin SWE-bench version, document Docker image versions, retry failed evals once. |
| Logprobs unavailable from Ollama             | Medium  | Low    | Fall back to self-consistency or output-validity-only signals. |
| Scope creep (extra features, second benchmark) | High    | High   | Lock scope at end of week 1. Park ideas in `FUTURE.md`. |

---

## 8. Per-week deliverables

**Week 1** — passthrough proxy works, OpenCode runs through it, one SWE-bench issue passes end-to-end.

**Week 2** — full SWE-bench harness works, all-local and all-frontier baselines collected on 50 issues, heuristic router implemented.

**Week 3** — confidence and failure-recovery added, full system runs on 50 issues, format-check and random baselines collected.

**Week 4** — all ablations run, dataframes built, headline figures drafted, paper sections 1-4 drafted.

**Week 5** — sensitivity runs, paper polish, public repo cleanup, submission.

---

## 9. Things to avoid

- **Don't fork OpenCode.** Stay outside it. The proxy is your contribution; modifying OpenCode multiplies the surface area you have to defend.
- **Don't add a learned router.** It needs labeled traces, training time, and a fairness debate ("did you train on the test set?"). Training-free is your differentiator.
- **Don't add a second benchmark.** SWE-bench Lite is enough. Save HumanEval, LiveCodeBench, etc. for "future work" in the discussion.
- **Don't optimize prematurely.** v1 of the heuristic router can be 30 lines. Add complexity only if ablations show the simple version misses something.
- **Don't run experiments before logging is solid.** A re-run because of a missing field is a wasted day.

---

## 10. Final checklist before submission

- [ ] All 5 main configs run on the 50-issue subset, with at least 2 trials each on the full system.
- [ ] Results dataframe checked into the repo (or a script that regenerates it).
- [ ] Pareto frontier figure is the paper's centerpiece.
- [ ] At least 4 ablations reported.
- [ ] Limitations section is honest about what wasn't tested.
- [ ] README explains how to reproduce a single trajectory in under 10 minutes.
- [ ] Repo has an MIT or Apache license.
- [ ] Paper PDF compiles cleanly under the 8-page limit.

Good luck. Most of the pain in this project is in week 1 (integration) and the first 2 days of week 2 (SWE-bench harness). After that it's mostly running experiments and writing.