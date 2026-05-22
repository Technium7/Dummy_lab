from __future__ import annotations

from typing import Any, Callable, Mapping

from high_level_jobs.subtasks.builtins import BUILTIN_SUBTASKS
from high_level_jobs.subtasks.robot_actions import ROBOT_SUBTASKS

# Add your real robot / vision / IL hooks here, e.g.:
# CUSTOM_SUBTASKS["my_skill"] = lambda p, ctx: ...

CUSTOM_SUBTASKS: dict[str, Callable[[Mapping[str, Any], Mapping[str, Any]], None]] = {}


def default_registry() -> dict[str, Callable[[Mapping[str, Any], Mapping[str, Any]], None]]:
    merged = dict(BUILTIN_SUBTASKS)
    merged.update(ROBOT_SUBTASKS)
    merged.update(CUSTOM_SUBTASKS)
    return merged
