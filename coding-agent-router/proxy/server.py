import json
import logging
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .backends import OllamaBackend, NIMBackend
from .confidence import parse_response_quality, should_escalate
from .router import Decision, Router
from .trajectory import TrajectoryStore
from .config import settings

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("proxy")

app = FastAPI()
trajectory_store = TrajectoryStore()
backends = {
    "local": OllamaBackend(settings.ollama_url, model="qwen3-coder-16k"),
    "frontier": NIMBackend(
        settings.nim_url,
        settings.nvidia_api_key,
        model="qwen/qwen3-coder-32b-instruct",
    ),
}
router = Router(mode=settings.router_mode)

# trajectory persistence directory (empty string disables)
_traj_dir: Path | None = Path(settings.trajectory_dir) if settings.trajectory_dir else None
if _traj_dir:
    _traj_dir.mkdir(parents=True, exist_ok=True)


def _flush_trajectory(traj) -> None:
    if _traj_dir is None:
        return
    path = _traj_dir / f"{traj.id.replace('/', '__')}.json"
    path.write_text(json.dumps({"id": traj.id, "steps": traj.steps}, default=str))


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    request_id = str(uuid.uuid4())

    # prefer explicit header, fall back to body "user" field
    trajectory_id = (
        request.headers.get("x-session-id")
        or body.get("user")
        or "default"
    )
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
            except Exception as e:
                # keep original local response if frontier also fails
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


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/trajectories/{trajectory_id:path}")
def get_trajectory(trajectory_id: str):
    traj = trajectory_store._store.get(trajectory_id)
    if traj is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse({"id": traj.id, "steps": traj.steps}, media_type="application/json")
