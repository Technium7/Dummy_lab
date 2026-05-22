from __future__ import annotations

import logging
import time
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)


def log_message(params: Mapping[str, Any], ctx: Mapping[str, Any]) -> None:
    text = str(params.get("text", ""))
    logger.info("[subtask:log_message] %s", text)
    if ctx.get("verbose"):
        print(f"[log_message] {text}")


def wait_seconds(params: Mapping[str, Any], ctx: Mapping[str, Any]) -> None:
    sec = float(params.get("seconds", 0.0))
    logger.info("[subtask:wait_seconds] %.4f", sec)
    if ctx.get("dry_run"):
        return
    time.sleep(max(0.0, sec))


# Register names -> (callable,). Kept minimal; add your own in registry.py pattern.
BUILTIN_SUBTASKS: dict[str, Callable[[Mapping[str, Any], Mapping[str, Any]], None]] = {
    "log_message": log_message,
    "wait_seconds": wait_seconds,
}
