"""
Branch coverage for Router._full_decide and the static modes.

The router consults `features.extract(body, traj)`, which inspects:
- body.messages (user/tool text, error/plan keywords)
- body.tools (count)
- traj.steps (length → step_index, last_action_repeated, recent_failure_count)

So each test crafts a (body, traj) pair that triggers exactly one branch.
The decision order in _full_decide matters — prior branches must NOT fire.
"""

from proxy.router import Router
from proxy.trajectory import Trajectory


# ---------------------------------------------------------------------------
# Helpers — minimal trajectories shaped like the real ones
# ---------------------------------------------------------------------------

def _empty_traj(n_prior_steps: int = 0, with_failures: int = 0) -> Trajectory:
    """A trajectory with `n_prior_steps` benign steps, `with_failures` of them marked local_failed."""
    traj = Trajectory(id="t")
    for i in range(n_prior_steps):
        traj.steps.append({
            "ts": 0.0,
            "backend": "local",
            "reason": "default_local",
            "latency_s": 0.0,
            "local_failed": i < with_failures,
            "request_messages": [],
            "request_tools": [],
            "response_choice": {"message": {"content": "ok"}},
            "response_usage": {},
        })
    return traj


def _traj_with_repeated_tool() -> Trajectory:
    """Two recent steps calling the same tool with the same args."""
    traj = Trajectory(id="t")
    step = {
        "ts": 0.0,
        "backend": "local",
        "reason": "default_local",
        "latency_s": 0.0,
        "local_failed": False,
        "request_messages": [],
        "request_tools": [],
        "response_choice": {
            "message": {
                "tool_calls": [
                    {"function": {"name": "bash", "arguments": '{"cmd": "ls"}'}}
                ]
            }
        },
        "response_usage": {},
    }
    # last_action_repeated() checks steps[-1] vs steps[-4:-1] — need >= 2 entries
    traj.steps.append(dict(step))
    traj.steps.append(dict(step))
    return traj


def _body(messages=None, tools=None):
    return {"messages": messages or [], "tools": tools or []}


# ---------------------------------------------------------------------------
# Static modes
# ---------------------------------------------------------------------------

def test_mode_all_local():
    r = Router(mode="all_local")
    d = r.decide(_body(), _empty_traj())
    assert d.backend == "local"
    assert d.reason == "static_all_local"
    assert d.confidence_check is False


def test_mode_all_frontier():
    r = Router(mode="all_frontier")
    d = r.decide(_body(), _empty_traj())
    assert d.backend == "frontier"


def test_mode_random_returns_valid_backend():
    r = Router(mode="random")
    for _ in range(20):
        d = r.decide(_body(), _empty_traj())
        assert d.backend in ("local", "frontier")
        assert d.reason.startswith("random:")


def test_mode_format_check_routes_local_with_confidence_gate():
    r = Router(mode="format_check")
    d = r.decide(_body(), _empty_traj())
    assert d.backend == "local"
    assert d.reason == "format_check_first"
    assert d.confidence_check is True


def test_unknown_mode_raises():
    r = Router(mode="nonsense")
    try:
        r.decide(_body(), _empty_traj())
    except NotImplementedError:
        return
    assert False, "expected NotImplementedError"


# ---------------------------------------------------------------------------
# _full_decide — branch 1: last_tool_failed
# ---------------------------------------------------------------------------

def test_full_tool_failed_escalates():
    r = Router(mode="full")
    body = _body(messages=[
        {"role": "user", "content": "fix it"},
        {"role": "tool", "name": "bash", "content": "Error: command failed with exit code 1"},
    ])
    d = r.decide(body, _empty_traj())
    assert d.backend == "frontier"
    assert d.reason == "tool_failed"


# ---------------------------------------------------------------------------
# _full_decide — branch 2: repeated_action
# ---------------------------------------------------------------------------

def test_full_repeated_action_escalates():
    r = Router(mode="full")
    # body has no failing tool result, no error keywords
    body = _body(messages=[{"role": "user", "content": "go"}])
    d = r.decide(body, _traj_with_repeated_tool())
    assert d.backend == "frontier"
    assert d.reason == "repeated_action"


# ---------------------------------------------------------------------------
# _full_decide — branch 3: trajectory_struggling (>=2 recent failures)
# ---------------------------------------------------------------------------

def test_full_trajectory_struggling_escalates():
    r = Router(mode="full")
    body = _body(messages=[{"role": "user", "content": "next step"}])
    traj = _empty_traj(n_prior_steps=4, with_failures=2)
    d = r.decide(body, traj)
    assert d.backend == "frontier"
    assert d.reason == "trajectory_struggling"


def test_full_one_failure_is_not_struggling():
    r = Router(mode="full")
    body = _body(messages=[{"role": "user", "content": "next step"}])
    traj = _empty_traj(n_prior_steps=4, with_failures=1)
    d = r.decide(body, traj)
    assert d.reason != "trajectory_struggling"


# ---------------------------------------------------------------------------
# _full_decide — branch 4: error_in_recent_context (step_index > 3)
# ---------------------------------------------------------------------------

def test_full_error_keywords_late_escalates():
    r = Router(mode="full")
    body = _body(messages=[
        {"role": "assistant", "content": "I see a Traceback in the output"},
        {"role": "user", "content": "keep going"},
    ])
    traj = _empty_traj(n_prior_steps=5)  # step_index=5 > 3
    d = r.decide(body, traj)
    assert d.backend == "frontier"
    assert d.reason == "error_in_recent_context"


def test_full_error_keywords_early_does_not_match_this_branch():
    """At step_index <= 3 the error_in_recent_context branch must NOT fire."""
    r = Router(mode="full")
    body = _body(messages=[
        {"role": "assistant", "content": "Got an exception earlier"},
        {"role": "user", "content": "keep going"},
    ])
    traj = _empty_traj(n_prior_steps=2)
    d = r.decide(body, traj)
    assert d.reason != "error_in_recent_context"


# ---------------------------------------------------------------------------
# _full_decide — branch 5: initial_planning (step_index < 3)
# ---------------------------------------------------------------------------

def test_full_initial_planning_escalates():
    r = Router(mode="full")
    body = _body(messages=[
        {"role": "user", "content": "Let's think step by step about this problem"},
    ])
    traj = _empty_traj(n_prior_steps=0)  # step_index=0 < 3
    d = r.decide(body, traj)
    assert d.backend == "frontier"
    assert d.reason == "initial_planning"


# ---------------------------------------------------------------------------
# _full_decide — branch 6: deep_trajectory
# ---------------------------------------------------------------------------

def test_full_deep_trajectory_escalates():
    r = Router(mode="full")
    body = _body(messages=[{"role": "user", "content": "continue"}])
    traj = _empty_traj(n_prior_steps=25)  # > default 20
    d = r.decide(body, traj)
    assert d.backend == "frontier"
    assert d.reason == "deep_trajectory"


def test_full_deep_trajectory_threshold_configurable():
    r = Router(mode="full", config={"depth_threshold": 5})
    body = _body(messages=[{"role": "user", "content": "continue"}])
    traj = _empty_traj(n_prior_steps=6)
    d = r.decide(body, traj)
    assert d.reason == "deep_trajectory"


# ---------------------------------------------------------------------------
# _full_decide — branch 7: post_cheap_tool
# ---------------------------------------------------------------------------

def test_full_post_cheap_tool_goes_local_with_confidence_check():
    r = Router(mode="full")
    body = _body(messages=[
        {"role": "user", "content": "show me the file"},
        # benign tool result — no error keywords, name is in CHEAP_TOOLS
        {"role": "tool", "name": "read_file", "content": "def foo(): pass"},
    ])
    traj = _empty_traj(n_prior_steps=5)  # past step_index<3 window so plan kw doesn't matter
    d = r.decide(body, traj)
    assert d.backend == "local"
    assert d.reason == "post_cheap_tool"
    assert d.confidence_check is True


# ---------------------------------------------------------------------------
# _full_decide — branch 8: default_local
# ---------------------------------------------------------------------------

def test_full_default_local():
    r = Router(mode="full")
    body = _body(messages=[
        {"role": "user", "content": "what does this function return"},
    ])
    traj = _empty_traj(n_prior_steps=5)  # past planning window, before depth threshold
    d = r.decide(body, traj)
    assert d.backend == "local"
    assert d.reason == "default_local"
    assert d.confidence_check is True


# ---------------------------------------------------------------------------
# Priority: tool_failed dominates other signals
# ---------------------------------------------------------------------------

def test_full_tool_failed_dominates_cheap_tool():
    """A cheap tool returning an error must still escalate (branch 1, not 7)."""
    r = Router(mode="full")
    body = _body(messages=[
        {"role": "tool", "name": "read_file", "content": "Error: file not found"},
    ])
    d = r.decide(body, _empty_traj(n_prior_steps=5))
    assert d.reason == "tool_failed"
