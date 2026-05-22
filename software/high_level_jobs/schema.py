from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import yaml


@dataclass
class StepSpec:
    action: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class StateSpec:
    steps: list[StepSpec] = field(default_factory=list)
    description: str = ""


@dataclass
class TransitionSpec:
    from_state: str
    event: str
    to_state: str


@dataclass
class JobSpec:
    job_id: str
    initial_state: str
    states: dict[str, StateSpec]
    transitions: list[TransitionSpec]
    terminal_states: list[str] = field(default_factory=list)

    def state_names(self) -> set[str]:
        return set(self.states.keys())


def _parse_step(raw: Mapping[str, Any]) -> StepSpec:
    if "action" not in raw:
        raise ValueError(f"step missing 'action': {raw!r}")
    action = str(raw["action"])
    params = raw.get("params") or {}
    if not isinstance(params, Mapping):
        raise ValueError(f"step.params must be a mapping, got {type(params)}")
    return StepSpec(action=action, params=dict(params))


def _parse_state(name: str, raw: Mapping[str, Any]) -> StateSpec:
    desc = str(raw.get("description", ""))
    steps_raw = raw.get("steps") or []
    if not isinstance(steps_raw, list):
        raise ValueError(f"state {name!r}: 'steps' must be a list")
    steps = [_parse_step(s) for s in steps_raw]
    return StateSpec(steps=steps, description=desc)


def _parse_transition(
    raw: Mapping[str, Any],
    *,
    state_names: set[str],
    terminal_states: list[str],
) -> TransitionSpec:
    if "from" not in raw or "event" not in raw:
        raise ValueError(f"transition missing 'from' or 'event': {raw!r}")
    from_state = str(raw["from"])
    event = str(raw["event"])
    if "to" in raw and raw["to"] is not None and str(raw["to"]).strip() != "":
        to_state = str(raw["to"])
    else:
        # LLMs often omit the final "to" when pointing at a terminal; infer a sink.
        if len(terminal_states) == 1 and terminal_states[0] in state_names:
            to_state = terminal_states[0]
        elif "END" in state_names:
            to_state = "END"
        else:
            raise ValueError(
                f"transition missing 'to': {raw!r}. "
                "Add an explicit \"to\" target state, or define exactly one name in terminal_states "
                "that exists in states."
            )
    return TransitionSpec(from_state=from_state, event=event, to_state=to_state)


def _inject_missing_go_home_chain(spec: JobSpec) -> None:
    """
    Repair common LLM omissions for the CONNECT → ENABLE → MOVE_HOME → END layout.

    - Adds each ``done`` edge whenever **both** endpoint states exist (does not require
      all four states to be present before adding partial edges).
    - If ``END`` exists and looks like this flow (any of CONNECT/ENABLE/MOVE_HOME present)
      but ``END`` was left out of ``terminal_states``, append it so validation passes.
    """
    names = spec.state_names()
    existing = {(t.from_state, t.event) for t in spec.transitions}
    chain = (
        ("CONNECT", "done", "ENABLE"),
        ("ENABLE", "done", "MOVE_HOME"),
        ("MOVE_HOME", "done", "END"),
    )
    for frm, ev, to in chain:
        if frm in names and to in names and (frm, ev) not in existing:
            spec.transitions.append(TransitionSpec(frm, ev, to))
            existing.add((frm, ev))

    if "END" in names and "END" not in spec.terminal_states:
        if names & {"CONNECT", "ENABLE", "MOVE_HOME"}:
            spec.terminal_states.append("END")


def load_job_yaml(text: str) -> JobSpec:
    data = yaml.safe_load(text)
    if not isinstance(data, Mapping):
        raise ValueError("job YAML must be a mapping at the top level")

    job_id = str(data.get("job_id", "unnamed_job"))
    initial = data.get("initial_state")
    if not initial:
        raise ValueError("job requires 'initial_state'")
    initial_state = str(initial)

    states_raw = data.get("states") or {}
    if not isinstance(states_raw, Mapping):
        raise ValueError("'states' must be a mapping of state_name -> state body")

    states: dict[str, StateSpec] = {}
    for name, body in states_raw.items():
        if not isinstance(body, Mapping):
            raise ValueError(f"state {name!r} body must be a mapping")
        states[str(name)] = _parse_state(str(name), body)

    term = data.get("terminal_states") or []
    if not isinstance(term, list):
        raise ValueError("'terminal_states' must be a list")
    terminal_states = [str(x) for x in term]

    transitions_raw = data.get("transitions") or []
    if not isinstance(transitions_raw, list):
        raise ValueError("'transitions' must be a list")
    names = set(states.keys())
    transitions = [
        _parse_transition(t, state_names=names, terminal_states=terminal_states)
        for t in transitions_raw
    ]

    spec = JobSpec(
        job_id=job_id,
        initial_state=initial_state,
        states=states,
        transitions=transitions,
        terminal_states=terminal_states,
    )
    _inject_missing_go_home_chain(spec)
    validate_job(spec)
    return spec


def validate_job(spec: JobSpec) -> None:
    names = spec.state_names()
    if spec.initial_state not in names:
        raise ValueError(f"initial_state {spec.initial_state!r} not found in states")

    for t in spec.transitions:
        if t.from_state not in names:
            raise ValueError(f"transition from unknown state {t.from_state!r}")
        if t.to_state not in names:
            raise ValueError(f"transition to unknown state {t.to_state!r}")

    for term in spec.terminal_states:
        if term not in names:
            raise ValueError(f"terminal_states contains unknown state {term!r}")

    terminals = set(spec.terminal_states)
    has_done_out = {t.from_state for t in spec.transitions if t.event == "done"}
    for name in names:
        if name in terminals:
            continue
        if name not in has_done_out:
            raise ValueError(
                f"state {name!r} is not listed in terminal_states but has no outgoing transition "
                f"with event 'done'. The runner only advances on (state, \"done\"). "
                f"Add e.g. {{\"from\": {name!r}, \"event\": \"done\", \"to\": \"NEXT_STATE\"}}."
            )


def job_to_yaml_dict(spec: JobSpec) -> dict[str, Any]:
    """Round-trip friendly dict for saving edited plans."""
    return {
        "job_id": spec.job_id,
        "initial_state": spec.initial_state,
        "states": {
            name: {
                "description": st.description,
                "steps": [{"action": s.action, "params": dict(s.params)} for s in st.steps],
            }
            for name, st in spec.states.items()
        },
        "transitions": [
            {"from": t.from_state, "event": t.event, "to": t.to_state} for t in spec.transitions
        ],
        "terminal_states": list(spec.terminal_states),
    }
