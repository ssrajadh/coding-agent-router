import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport

from proxy.server import app


FAKE_RESPONSE = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
}


@pytest.mark.anyio
async def test_healthz():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


@pytest.mark.anyio
async def test_chat_completions_routes_to_local():
    with patch("proxy.server.backends") as mock_backends:
        mock_local = AsyncMock()
        mock_local.chat_completion = AsyncMock(return_value=FAKE_RESPONSE)
        mock_backends.__getitem__ = lambda self, k: mock_local

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hello"}]},
            )

    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "hi"


def test_router_all_local():
    from proxy.router import Router
    from proxy.trajectory import Trajectory

    router = Router(mode="all_local")
    traj = Trajectory(id="t1")
    decision = router.decide({}, traj)
    assert decision.backend == "local"


def test_router_all_frontier():
    from proxy.router import Router
    from proxy.trajectory import Trajectory

    router = Router(mode="all_frontier")
    traj = Trajectory(id="t2")
    decision = router.decide({}, traj)
    assert decision.backend == "frontier"


def test_trajectory_record_step():
    from proxy.trajectory import Trajectory

    traj = Trajectory(id="t3")
    traj.record_step(
        request={"messages": [{"role": "user", "content": "hi"}]},
        response={"choices": [{}], "usage": {}},
        backend="local",
        decision_reason="static_all_local",
        latency_s=0.5,
    )
    assert len(traj.steps) == 1
    assert traj.steps[0]["backend"] == "local"
    assert traj.steps[0]["latency_s"] == 0.5
