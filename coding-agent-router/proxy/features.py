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
    last_user_text = (
        (last_user or {}).get("content", "")
        if isinstance((last_user or {}).get("content"), str)
        else ""
    )

    # scan last 3 messages for keyword signals
    recent_text = " ".join(
        m.get("content", "") if isinstance(m.get("content"), str) else ""
        for m in messages[-3:]
    )

    last_tool_name = None
    last_tool_failed = False
    for m in reversed(messages):
        if m.get("role") == "tool":
            last_tool_name = m.get("name")
            content = m.get("content", "")
            last_tool_failed = bool(
                ERROR_REGEX.search(content if isinstance(content, str) else "")
            )
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
        is_repeated_action=(
            traj.last_action_repeated() if hasattr(traj, "last_action_repeated") else False
        ),
        recent_failure_count=sum(1 for s in traj.steps[-5:] if s.get("local_failed")),
    )
