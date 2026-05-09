import random
from dataclasses import dataclass, field


@dataclass
class Decision:
    backend: str
    reason: str
    confidence_check: bool = field(default=False)


class Router:
    def __init__(self, mode: str = "all_local"):
        self.mode = mode

    def decide(self, body: dict, trajectory) -> Decision:
        if self.mode == "all_local":
            return Decision("local", "static_all_local")
        if self.mode == "all_frontier":
            return Decision("frontier", "static_all_frontier")
        if self.mode == "random":
            b = random.choice(["local", "frontier"])
            return Decision(b, f"random:{b}")
        raise NotImplementedError(self.mode)
