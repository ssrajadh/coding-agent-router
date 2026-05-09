import time
from dataclasses import dataclass, field


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

    def mark_local_failure(self):
        if self.steps:
            self.steps[-1]["local_failed"] = True


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
