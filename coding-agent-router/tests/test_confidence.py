"""
Coverage for parse_response_quality + should_escalate.

The two functions together form the confidence gate that decides whether a
local response should be retried on the frontier backend.
"""

import json

from proxy.confidence import parse_response_quality, should_escalate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resp(
    finish_reason="stop",
    content="ok",
    tool_calls=None,
    logprobs=None,
):
    msg: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    choice: dict = {"message": msg, "finish_reason": finish_reason}
    if logprobs is not None:
        choice["logprobs"] = logprobs
    return {"choices": [choice]}


# ---------------------------------------------------------------------------
# parse_response_quality
# ---------------------------------------------------------------------------

def test_parse_clean_response():
    signals = parse_response_quality(_resp())
    assert signals["finish_reason_ok"] is True
    assert signals["tool_calls_valid"] is True
    assert signals["json_args_parseable"] is True
    assert signals["min_logprob"] is None


def test_parse_finish_reason_length_not_ok():
    signals = parse_response_quality(_resp(finish_reason="length"))
    assert signals["finish_reason_ok"] is False


def test_parse_tool_call_missing_name():
    tc = [{"function": {"name": "", "arguments": "{}"}}]
    signals = parse_response_quality(_resp(finish_reason="tool_calls", tool_calls=tc))
    assert signals["tool_calls_valid"] is False


def test_parse_tool_call_unparseable_args():
    tc = [{"function": {"name": "bash", "arguments": "{not valid json"}}]
    signals = parse_response_quality(_resp(finish_reason="tool_calls", tool_calls=tc))
    assert signals["json_args_parseable"] is False


def test_parse_logprobs_minimum_extracted():
    lp = {"content": [
        {"token": "a", "logprob": -0.1},
        {"token": "b", "logprob": -4.5},
        {"token": "c", "logprob": -1.2},
    ]}
    signals = parse_response_quality(_resp(logprobs=lp))
    assert signals["min_logprob"] == -4.5


# ---------------------------------------------------------------------------
# should_escalate — each escalation reason in priority order
# ---------------------------------------------------------------------------

def test_escalate_bad_finish_reason():
    signals = parse_response_quality(_resp(finish_reason="length"))
    esc, reason = should_escalate(signals)
    assert esc is True
    assert reason == "bad_finish_reason"


def test_escalate_invalid_tool_call():
    tc = [{"function": {"name": "", "arguments": "{}"}}]
    signals = parse_response_quality(_resp(finish_reason="tool_calls", tool_calls=tc))
    esc, reason = should_escalate(signals)
    assert esc is True
    assert reason == "invalid_tool_call"


def test_escalate_unparseable_json():
    tc = [{"function": {"name": "bash", "arguments": "{broken"}}]
    signals = parse_response_quality(_resp(finish_reason="tool_calls", tool_calls=tc))
    esc, reason = should_escalate(signals)
    assert esc is True
    assert reason == "unparseable_json"


def test_escalate_low_confidence_logprob():
    lp = {"content": [{"token": "x", "logprob": -5.0}]}
    signals = parse_response_quality(_resp(logprobs=lp))
    esc, reason = should_escalate(signals, threshold=-3.0)
    assert esc is True
    assert reason == "low_confidence_logprob"


def test_no_escalate_high_confidence_logprob():
    lp = {"content": [{"token": "x", "logprob": -0.5}]}
    signals = parse_response_quality(_resp(logprobs=lp))
    esc, reason = should_escalate(signals, threshold=-3.0)
    assert esc is False
    assert reason == "ok"


def test_no_escalate_no_logprobs_available():
    """Missing logprobs (Ollama default) must not be treated as low confidence."""
    signals = parse_response_quality(_resp())
    esc, _ = should_escalate(signals)
    assert esc is False


def test_no_escalate_clean_tool_call():
    tc = [{"function": {"name": "read_file", "arguments": json.dumps({"path": "x"})}}]
    signals = parse_response_quality(_resp(finish_reason="tool_calls", tool_calls=tc))
    esc, reason = should_escalate(signals)
    assert esc is False
    assert reason == "ok"


def test_threshold_is_strict():
    """Logprob equal to threshold should NOT escalate (uses < not <=)."""
    lp = {"content": [{"token": "x", "logprob": -3.0}]}
    signals = parse_response_quality(_resp(logprobs=lp))
    esc, _ = should_escalate(signals, threshold=-3.0)
    assert esc is False
