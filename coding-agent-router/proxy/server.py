import os
import uuid
import time
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

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
    "frontier": NIMBackend(
        settings.nim_url,
        settings.nvidia_api_key,
        model="qwen/qwen3-coder-32b-instruct",
    ),
}
router = Router(mode=settings.router_mode)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    request_id = str(uuid.uuid4())
    trajectory_id = body.get("user") or request.headers.get("x-session-id", "default")
    traj = trajectory_store.get_or_create(trajectory_id)

    decision = router.decide(body, traj)
    log.info("req=%s traj=%s -> %s (%s)", request_id, trajectory_id, decision.backend, decision.reason)

    backend = backends[decision.backend]
    t0 = time.time()
    try:
        response = await backend.chat_completion(body)
    except Exception as e:
        log.exception("backend failure: %s", e)
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
