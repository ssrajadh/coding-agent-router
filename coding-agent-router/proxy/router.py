import random
from dataclasses import dataclass, field

from .features import extract, CHEAP_TOOLS


@dataclass
class Decision:
    backend: str
    reason: str
    confidence_check: bool = field(default=False)


class Router:
    def __init__(self, mode: str = "all_local", config: dict | None = None):
        self.mode = mode
        self.config = config or {}

    def decide(self, body: dict, trajectory) -> Decision:
        if self.mode == "all_local":
            return Decision("local", "static_all_local")
        if self.mode == "all_frontier":
            return Decision("frontier", "static_all_frontier")
        if self.mode == "random":
            b = random.choice(["local", "frontier"])
            return Decision(b, f"random:{b}")
        if self.mode == "format_check":
            # try local first; escalation fires in server if response fails validation
            return Decision("local", "format_check_first", confidence_check=True)
        if self.mode == "full":
            return self._full_decide(body, trajectory)
        raise NotImplementedError(self.mode)

    def _full_decide(self, body: dict, traj) -> Decision:
        f = extract(body, traj)

        # hard escalation: something is actively wrong
        if f.last_tool_failed:
            return Decision("frontier", "tool_failed")
        if f.is_repeated_action:
            return Decision("frontier", "repeated_action")
        if f.recent_failure_count >= 2:
            return Decision("frontier", "trajectory_struggling")
        if f.contains_error_keywords and f.step_index > 3:
            return Decision("frontier", "error_in_recent_context")

        # hard escalation: task looks hard from the start
        if f.contains_plan_keywords and f.step_index < 3:
            return Decision("frontier", "initial_planning")
        if f.step_index > self.config.get("depth_threshold", 20):
            return Decision("frontier", "deep_trajectory")

        # cheap read-only step: try local with confidence gate
        if f.last_tool_name in CHEAP_TOOLS:
            return Decision("local", "post_cheap_tool", confidence_check=True)

        # default: try local, verify before committing
        return Decision("local", "default_local", confidence_check=True)
