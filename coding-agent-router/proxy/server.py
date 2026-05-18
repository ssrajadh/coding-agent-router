import json
import logging
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .backends import OllamaBackend, NIMBackend, RateLimitError
from .confidence import parse_response_quality, should_escalate
from .router import Decision, Router
from .trajectory import TrajectoryStore
from .config import settings

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("proxy")

app = FastAPI()
trajectory_store = TrajectoryStore()
backends = {
    "local": OllamaBackend(settings.ollama_url, model=settings.local_model),
    "frontier": NIMBackend(
        settings.nim_url,
        settings.nvidia_api_key,
        model=settings.frontier_model,
    ),
}
router = Router(mode=settings.router_mode)

_traj_dir: Path | None = Path(settings.trajectory_dir) if settings.trajectory_dir else None
if _traj_dir:
    _traj_dir.mkdir(parents=True, exist_ok=True)


def _flush_trajectory(traj) -> None:
    if _traj_dir is None:
        return
    path = _traj_dir / f"{traj.id.replace('/', '__')}.json"
    path.write_text(json.dumps({"id": traj.id, "steps": traj.steps}, default=str))


async def _handle(body: dict, trajectory_id: str) -> JSONResponse:
    request_id = str(uuid.uuid4())
    traj = trajectory_store.get_or_create(trajectory_id)
    decision = router.decide(body, traj)
    log.info(
        "req=%s traj=%s -> %s (%s)  step=%d",
        request_id, trajectory_id, decision.backend, decision.reason, len(traj.steps),
    )

    backend = backends[decision.backend]
    t0 = time.time()
    local_failed = False
    try:
        response = await backend.chat_completion(body)
    except RateLimitError as e:
        # Frontier throttled — fail-soft to local so opencode keeps making progress
        # instead of seeing a 502 and giving up.
        if decision.backend == "frontier":
            log.warning("req=%s NIM 429 → failing soft to local", request_id)
            try:
                response = await backends["local"].chat_completion(body)
                decision = Decision("local", "frontier_rate_limited_fallback")
            except Exception as e2:
                log.exception("local fallback after 429 also failed: %s", e2)
                return JSONResponse({"error": f"both backends failed: {e}; {e2}"}, status_code=502)
        else:
            return JSONResponse({"error": str(e)}, status_code=502)
    except Exception as e:
        log.exception("backend failure: %s", e)
        return JSONResponse({"error": str(e)}, status_code=502)

    if decision.confidence_check and decision.backend == "local":
        signals = parse_response_quality(response)
        escalate, esc_reason = should_escalate(signals)
        if escalate:
            log.info("escalating traj=%s reason=%s", trajectory_id, esc_reason)
            local_failed = True
            try:
                response = await backends["frontier"].chat_completion(body)
                decision = Decision("frontier", f"escalated_{esc_reason}")
            except RateLimitError:
                log.warning("escalation 429'd — keeping local response")
            except Exception as e:
                log.exception("escalation to frontier failed: %s", e)

    elapsed = time.time() - t0
    traj.record_step(
        request=body,
        response=response,
        backend=decision.backend,
        decision_reason=decision.reason,
        latency_s=elapsed,
        local_failed=local_failed,
    )
    _flush_trajectory(traj)
    return JSONResponse(response)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    trajectory_id = (
        request.headers.get("x-session-id")
        or body.get("user")
        or "default"
    )
    return await _handle(body, trajectory_id)


@app.post("/sess/{session_id:path}/v1/chat/completions")
async def session_chat_completions(session_id: str, request: Request):
    # URL-path session ID — the load-bearing fix for per-issue trajectory isolation.
    # Each opencode invocation gets its own baseURL with the issue ID baked into the
    # path; the proxy reads it from here. Beats opencode.jsonc rewriting because it
    # avoids races between parallel issues without per-issue config dirs.
    body = await request.json()
    return await _handle(body, session_id)


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/trajectories/{trajectory_id:path}")
def get_trajectory(trajectory_id: str):
    traj = trajectory_store._store.get(trajectory_id)
    if traj is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse({"id": traj.id, "steps": traj.steps}, media_type="application/json")
