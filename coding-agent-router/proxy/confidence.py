import json

# two-stage gate: parse signals from a response, then decide whether to escalate.
# used by server.py after any local call that has confidence_check=True.


def parse_response_quality(response: dict) -> dict:
    choice = response.get("choices", [{}])[0]
    msg = choice.get("message", {})
    finish_reason = choice.get("finish_reason")

    signals = {
        "finish_reason_ok": finish_reason in ("stop", "tool_calls"),
        "tool_calls_valid": True,
        "json_args_parseable": True,
        "min_logprob": None,
    }

    # check every tool call in the response for structural validity
    for tc in msg.get("tool_calls", []) or []:
        fn = tc.get("function", {})
        if not fn.get("name"):
            signals["tool_calls_valid"] = False
        try:
            json.loads(fn.get("arguments", "{}"))
        except json.JSONDecodeError:
            signals["json_args_parseable"] = False

    # logprobs are optional; Ollama may omit them entirely
    lp = choice.get("logprobs")
    if lp and lp.get("content"):
        token_lps = [t.get("logprob", 0.0) for t in lp["content"]]
        if token_lps:
            # minimum (most uncertain) token drives the escalation decision
            signals["min_logprob"] = min(token_lps)

    return signals


def should_escalate(signals: dict, threshold: float = -3.0) -> tuple[bool, str]:
    # checks are ordered: structural failures first, then probabilistic confidence.
    # returns (escalate, reason_string) so the caller can log why it escalated.
    if not signals["finish_reason_ok"]:
        return True, "bad_finish_reason"
    if not signals["tool_calls_valid"]:
        return True, "invalid_tool_call"
    if not signals["json_args_parseable"]:
        return True, "unparseable_json"
    # threshold of -3.0 ~ top-5% uncertainty; tune via ablation (see GUIDE section 5.3)
    if signals["min_logprob"] is not None and signals["min_logprob"] < threshold:
        return True, "low_confidence_logprob"
    return False, "ok"
