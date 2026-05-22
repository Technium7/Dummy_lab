from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Mapping

from high_level_jobs.schema import JobSpec, load_job_yaml

logger = logging.getLogger(__name__)


def _transition_map(spec: JobSpec) -> dict[tuple[str, str], str]:
    return {(t.from_state, t.event): t.to_state for t in spec.transitions}


def run_job(
    spec: JobSpec,
    registry: Mapping[str, Callable[[Mapping[str, Any], Mapping[str, Any]], None]],
    *,
    ctx: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Execute states in order: run all steps in a state, then if the state is terminal,
    stop; else follow transition for event ``done`` if present (otherwise stop).

    Returns the mutable context dict (same object passed in, or a new dict), including
    any keys subtasks set (e.g. ``robot``).
    """
    ctx = dict(ctx or {})
    ctx.setdefault("dry_run", True)
    ctx.setdefault("verbose", True)

    trans = _transition_map(spec)
    state = spec.initial_state
    terminals = set(spec.terminal_states)

    while True:
        st = spec.states.get(state)
        if st is None:
            raise KeyError(f"unknown state {state!r}")

        logger.info("Enter state %r (%d steps)", state, len(st.steps))
        for step in st.steps:
            fn = registry.get(step.action)
            if fn is None:
                raise KeyError(
                    f"unregistered action {step.action!r} in state {state!r}. "
                    f"Register it in high_level_jobs/subtasks/registry.py"
                )
            fn(step.params, ctx)

        if state in terminals:
            logger.info("Finished terminal state %r", state)
            break

        nxt = trans.get((state, "done"))
        if nxt is None:
            logger.info("No transition from %r on event 'done'; stopping.", state)
            break
        state = nxt

    return ctx


def run_job_file(
    path: Path,
    registry: Mapping[str, Callable[[Mapping[str, Any], Mapping[str, Any]], None]],
    **ctx: Any,
) -> tuple[JobSpec, dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    spec = load_job_yaml(text)
    ctx_out = run_job(spec, registry, ctx=ctx)
    return spec, ctx_out
