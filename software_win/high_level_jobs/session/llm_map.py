"""Map free-text operator requests to session commands via LLM or keywords."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from high_level_jobs.session.commands import ALIASES, COMMANDS, normalize_command

_PROMPT = """You map a robot operator utterance to exactly one session command.

Allowed commands (return one of these strings as JSON {"command":"..."}):
- grab_sensor: pick up sensor and return to work; saves put-back motion
- put_sensor: put the sensor back where grab_sensor picked it from
- power_measure: place held sensor at a clicked target and stay (record return traj)
- power_optimization: run optical-power search/climb from current insert pose
- return: return to work using the recorded power_measure trajectory
- standby: replan from current pose back to work
- manual: hand off to keyboard teleop (manual_control)
- stop: emergency stop the robot immediately (!STOP)
- epoxy_apply: epoxy application (may be unavailable)
- help: show help
- quit: end session

If unclear or unsafe, return {"command":"help"}.
Output ONLY JSON, no markdown.
"""


def _keyword_fallback(text: str) -> str:
    t = text.lower()
    rules = [
        (r"\bput\b|\bplace back\b|\breturn sensor\b", "put_sensor"),
        (r"\bgrab\b|\bpick\b|\bpick up\b|\bgrasp\b", "grab_sensor"),
        (r"\boptim", "power_optimization"),
        (r"\bpower\b|\bmeasure\b|\binsert\b", "power_measure"),
        (r"\breturn\b|\bgo back\b", "return"),
        (r"\bstandby\b|\bwork position\b|\bgo to work\b", "standby"),
        (r"\bmanual\b|\bteleop\b", "manual"),
        (r"\bquit\b|\bexit\b|\bstop session\b", "quit"),
        (r"\bestop\b|\be-?stop\b|\bemergency stop\b|\bstop( the)?( robot)?\b", "stop"),
        (r"\bepoxy\b", "epoxy_apply"),
        (r"\bhelp\b", "help"),
    ]
    for pat, cmd in rules:
        if re.search(pat, t):
            return cmd
    return "help"


def _make_llm_client(backend: str, model: str | None) -> Any:
    software = Path(__file__).resolve().parents[3] / "software"
    if str(software) not in sys.path:
        sys.path.insert(0, str(software))
    from high_level_jobs.llm_client import OllamaClient, OpenAICompatClient

    if backend == "openai":
        return OpenAICompatClient(model=model) if model else OpenAICompatClient()
    return OllamaClient(model=model) if model else OllamaClient()


def map_utterance_to_command(
    text: str,
    *,
    backend: str = "ollama",
    model: str | None = None,
    use_llm: bool = True,
) -> str:
    """Return a normalized session command name."""
    direct = normalize_command(text)
    if direct:
        return direct

    body = text.strip()
    if body.lower().startswith("llm "):
        body = body[4:].strip()
        use_llm = True

    if not use_llm:
        return _keyword_fallback(body)

    try:
        client = _make_llm_client(backend, model)
        raw = client.complete(_PROMPT, body)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
        cmd = str(data.get("command") or "").strip().lower().replace("-", "_")
        if cmd in COMMANDS:
            return cmd
        if cmd in ALIASES:
            return ALIASES[cmd]
        print(f"[session] LLM returned unknown command {cmd!r}; using keyword fallback.")
    except Exception as exc:  # noqa: BLE001 — operator loop must stay up
        print(f"[session] LLM map failed ({exc}); using keyword fallback.")
    return _keyword_fallback(body)
