"""Map free-text operator requests to session commands or multi-step plans."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from high_level_jobs.llm_client import OllamaClient, OpenAICompatClient
from high_level_jobs.session.commands import ALIASES, COMMANDS, normalize_command

_PROMPT = """You are a planner for a Dummy robot lab session. Compose a plan from
subtask capabilities below. Reason from what each action DOES (effects / state),
not from fixed example patterns. Use the fewest steps that satisfy the request.
Do not invent goals the user did not ask for (no wait, no optimize, no extra returns).

World model (keep this consistent while planning):
- Poses: WORK vs INSERT. Holding vs empty are different.
- grab_sensor ends WORK + HOLDING. put_sensor RELEASES the sensor (WORK + EMPTY).
- return moves INSERT -> WORK but STILL HOLDING. return is not put_sensor.
- After put_sensor the arm is empty, so a later place/measure needs grab_sensor first.
- Trajectory library: named joint-space motions. Replay needs a saved name.
  A new click requires power_measure (optionally save_as=name).

Subtasks (action name -> function):
- grab_sensor: Detect/pick sensor, end at WORK holding it. Writes grab record for put_sensor.
- put_sensor: Reverse grab record, RELEASE sensor into the holder, end at WORK empty.
  Must start at WORK holding (return first if currently at INSERT).
- power_measure: Operator clicks a NEW image target; move WORK->INSERT; STAY at insert;
  record outbound+return traj. Params: optional save_as=<name> also stores that traj in
  the library under <name>. Use when the place is not already in the library / is new.
  Does NOT replace record_power for a settled averaged reading.
- traj_replay: Load library name; move WORK->INSERT along saved traj; STAY. Params:
  required name. Use when revisiting a place already saved (same physical spot / same traj).
- traj_save: Copy last power_measure record into library. Params: required name.
- traj_list: List library names (no motion).
- record_power: At CURRENT pose (typically INSERT after place/replay): wait until
  joints are quiet, pin stream target, sleep {"seconds":1} (default), then average
  PM100. Use when the user asks to record / log / sample power.
- return: From INSERT, stream saved return traj to WORK, STILL HOLDING.
  Params: optional name (library); else last power_measure record.
- standby: Fresh plan from current joints to WORK (no library). Use if lost / no record.
- power_optimization: From current INSERT, PM100 search/climb; STAY. Needs prior place
  extras. Appends the final peak reading to the same JSONL as record_power
  (source=power_optimization). Only if user asks to optimize/search power.
- wait: Hold CURRENT pose for N seconds. Params: required seconds (min->sec). ONLY if
  the user asked to wait/pause/dwell. It does not move the arm and does not put/release.
- manual: Hand off to teleop. stop: emergency stop. help / quit: meta.
- epoxy_apply: stub / usually unavailable.

Composition guidance (derive from functions, do not copy recipes):
- Need a new place? -> power_measure (add save_as if that traj will be reused).
- Need the same place again? -> traj_replay(name), not another power_measure.
- Need a settled power reading at insert? -> record_power after place/replay (before return).
- Need WORK while still holding? -> return.
- Need the sensor in the holder? -> put_sensor (after return if at INSERT).
- wait only if requested, and only after the pose/holding state already matches the request.
- Repeating a suffix K more times? -> then_repeat for that suffix.
- then_repeat must not contain power_measure when the place is already saved.

Output a single JSON object only (no markdown), with this schema:
{"steps":[{"action":"<action>","params":{...}}, ...],
 "then_repeat":{"times":<int>, "steps":[{"action":"...","params":{...}}, ...]}}
Omit then_repeat when unused. Single action may be {"command":"<action>"}.
then_repeat is a TOP-LEVEL key next to steps — never nest it inside the steps array.
Unclear/unsafe -> {"command":"help"}.
"""

# Params each subtask actually understands (capability surface for sanitize/validate).
_ACTION_PARAMS: dict[str, frozenset[str]] = {
    "grab_sensor": frozenset(),
    "put_sensor": frozenset(),
    "power_measure": frozenset({"save_as", "name"}),  # name accepted as save_as alias
    "power_optimization": frozenset({"name"}),
    "record_power": frozenset({"seconds", "settle_s", "s", "n_reads", "samples", "name"}),
    "return": frozenset({"name"}),
    "standby": frozenset(),
    "wait": frozenset({"seconds", "sec", "s"}),
    "traj_list": frozenset(),
    "traj_save": frozenset({"name", "source"}),
    "traj_replay": frozenset({"name"}),
    "manual": frozenset(),
    "stop": frozenset(),
    "epoxy_apply": frozenset(),
    "help": frozenset(),
    "quit": frozenset(),
}


@dataclass
class PlanStep:
    action: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    steps: list[PlanStep] = field(default_factory=list)
    from_llm: bool = False

    def is_empty(self) -> bool:
        return not self.steps

    def summary(self) -> str:
        parts: list[str] = []
        for s in self.steps:
            if s.params:
                kv = ",".join(f"{k}={v}" for k, v in s.params.items())
                parts.append(f"{s.action}({kv})")
            else:
                parts.append(s.action)
        return " -> ".join(parts) if parts else "(empty)"

    def format_review(self) -> str:
        """Multi-line plan listing for terminal review before execute."""
        lines = [f"[session] LLM plan ({len(self.steps)} steps):"]
        for i, s in enumerate(self.steps, start=1):
            if s.params:
                kv = ", ".join(f"{k}={v}" for k, v in s.params.items())
                lines.append(f"  {i:3d}. {s.action}  ({kv})")
            else:
                lines.append(f"  {i:3d}. {s.action}")
        lines.append(f"[session] summary: {self.summary()}")
        return "\n".join(lines)


def _normalize_action(raw: str) -> str:
    cmd = str(raw or "").strip().lower().replace("-", "_")
    if cmd in COMMANDS:
        return cmd
    if cmd in ALIASES:
        return ALIASES[cmd]
    return ""


def _parse_step(obj: Any) -> PlanStep | None:
    if isinstance(obj, str):
        action = _normalize_action(obj)
        return PlanStep(action=action) if action else None
    if not isinstance(obj, dict):
        return None
    action = _normalize_action(str(obj.get("action") or obj.get("command") or ""))
    if not action:
        return None
    params = obj.get("params") or {}
    if not isinstance(params, dict):
        params = {}
    flat = {
        k: v
        for k, v in obj.items()
        if k not in ("action", "command", "params")
    }
    merged = {**flat, **params}
    return PlanStep(action=action, params=dict(merged))


def expand_plan_document(data: dict[str, Any]) -> Plan:
    """Flatten steps + optional then_repeat into a linear Plan."""
    steps: list[PlanStep] = []

    raw_steps = data.get("steps")
    if raw_steps is None and data.get("command"):
        raw_steps = [{"action": data["command"]}]
    if isinstance(raw_steps, list):
        for item in raw_steps:
            step = _parse_step(item)
            if step is None:
                raise ValueError(f"invalid plan step: {item!r}")
            steps.append(step)

    then = data.get("then_repeat") or data.get("repeat_steps")
    if isinstance(then, dict):
        times = int(then.get("times") or then.get("count") or then.get("n") or 0)
        body = then.get("steps") or []
        if times < 0:
            raise ValueError("then_repeat.times must be >= 0")
        if not isinstance(body, list):
            raise ValueError("then_repeat.steps must be a list")
        parsed_body: list[PlanStep] = []
        for item in body:
            step = _parse_step(item)
            if step is None:
                raise ValueError(f"invalid then_repeat step: {item!r}")
            parsed_body.append(step)
        for _ in range(times):
            steps.extend(
                PlanStep(action=s.action, params=dict(s.params)) for s in parsed_body
            )

    if "repeat" in data and not isinstance(then, dict):
        times = int(data.get("repeat") or 1)
        if times > 1 and steps:
            base = [PlanStep(action=s.action, params=dict(s.params)) for s in steps]
            steps = []
            for _ in range(times):
                steps.extend(
                    PlanStep(action=s.action, params=dict(s.params)) for s in base
                )

    if not steps:
        raise ValueError("plan has no steps")
    return Plan(steps=steps)


def _make_llm_client(backend: str, model: str | None) -> Any:
    if backend == "openai":
        return OpenAICompatClient(model=model) if model else OpenAICompatClient()
    return OllamaClient(model=model) if model else OllamaClient()


def _strip_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_json_object(raw: str) -> dict[str, Any]:
    """Parse a JSON object from model text; tolerate fences / leading junk."""
    text = _strip_fences(raw)

    def _try(s: str) -> dict[str, Any] | None:
        try:
            data = json.loads(s)
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    data = _try(text)
    if data is not None:
        return data

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        blob = text[start : end + 1]
        data = _try(blob)
        if data is not None:
            return data
        # Common LLM glitches: trailing commas before } or ]
        repaired = re.sub(r",\s*([}\]])", r"\1", blob)
        data = _try(repaired)
        if data is not None:
            return data

    raise ValueError(f"could not parse LLM JSON: {text[:240]!r}")


def _coerce(raw: str) -> Any:
    s = raw.strip().strip("'\"")
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        return s


def _utterance_wait_seconds(text: str) -> float | None:
    t = text.lower()
    m = re.search(
        r"\b(?:wait|pause|dwell)\s+(\d+(?:\.\d+)?)\s*(min|minutes|s|sec|seconds)?\b",
        t,
    )
    if m is None:
        m = re.search(
            r"(\d+(?:\.\d+)?)\s*(min|minutes|s|sec|seconds)\b.{0,24}\b(wait|pause|dwell)\b",
            t,
        )
    if m is None:
        return None
    value = float(m.group(1))
    unit = (m.group(2) or "s").lower()
    if unit.startswith("min"):
        value *= 60.0
    return value


def _utterance_asks_wait(text: str) -> bool:
    return bool(re.search(r"\b(wait|pause|dwell)\b", text.lower()))


def _utterance_asks_put_sensor(text: str) -> bool:
    t = text.lower()
    return bool(
        re.search(
            r"\bput\b.{0,48}\bsensor\b|\bsensor\b.{0,48}\bput(\s+back)?\b|"
            r"\bput\s+it\s+(back|away)\b|\brelease\b.{0,24}\bsensor\b",
            t,
        )
    )


def _utterance_repeat_count(text: str) -> int | None:
    t = text.lower()
    m = re.search(r"(\d+)\s*(?:x|times?)\b", t)
    if m:
        return max(1, int(m.group(1)))
    return None


def _sanitize_plan(plan: Plan, *, utterance: str) -> Plan:
    """Strip unknown params / invented waits using subtask capability surfaces."""
    allow_wait = _utterance_asks_wait(utterance)
    out: list[PlanStep] = []
    last_lib: str | None = None
    for step in plan.steps:
        allowed = _ACTION_PARAMS.get(step.action, frozenset())
        params = {k: v for k, v in step.params.items() if k in allowed}
        if step.action == "power_measure" and "name" in params and "save_as" not in params:
            params["save_as"] = params.pop("name")
        if step.action == "power_measure":
            save_as = str(params.get("save_as") or "").strip()
            if save_as:
                last_lib = save_as
        if step.action in ("traj_replay", "traj_save", "return"):
            if not str(params.get("name") or "").strip() and last_lib:
                params["name"] = last_lib
            name = str(params.get("name") or "").strip()
            if name:
                last_lib = name
        if step.action == "record_power":
            # Default settle is 1s when user asked to record power without a duration.
            if "seconds" not in params and "settle_s" not in params and "s" not in params:
                params["seconds"] = 1.0
            elif "seconds" not in params:
                for alt in ("settle_s", "s"):
                    if alt in params:
                        params["seconds"] = params.pop(alt)
                        break
        if step.action == "wait" and not allow_wait:
            continue
        if step.action == "wait" and "seconds" not in params:
            for alt in ("sec", "s"):
                if alt in params:
                    params["seconds"] = params.pop(alt)
                    break
            if "seconds" not in params:
                inferred = _utterance_wait_seconds(utterance)
                if inferred is not None:
                    params["seconds"] = inferred
        out.append(PlanStep(step.action, params))
    return _rewrite_same_spot_repeats(Plan(steps=out), utterance=utterance)


def _rewrite_same_spot_repeats(plan: Plan, *, utterance: str) -> Plan:
    """Same-spot N× visits: one click (power_measure), later visits traj_replay."""
    n_rep = _utterance_repeat_count(utterance)
    t = utterance.lower()
    if n_rep is None or n_rep < 2:
        return plan
    if not re.search(r"\b(measure|power|same\s+spot|same\s+place)\b", t):
        return plan
    n_pm = sum(1 for s in plan.steps if s.action == "power_measure")
    if n_pm <= 1:
        return plan

    lib = ""
    for s in plan.steps:
        if s.action == "power_measure":
            lib = str(s.params.get("save_as") or s.params.get("name") or "").strip()
            if lib:
                break
        if s.action in ("traj_replay", "traj_save"):
            lib = str(s.params.get("name") or "").strip()
            if lib:
                break
    lib = lib or "spot"

    out: list[PlanStep] = []
    seen_pm = False
    for s in plan.steps:
        params = dict(s.params)
        if s.action == "power_measure":
            if not seen_pm:
                seen_pm = True
                params["save_as"] = lib
                out.append(PlanStep("power_measure", params))
            else:
                out.append(PlanStep("traj_replay", {"name": lib}))
            continue
        if s.action in ("traj_replay", "return") and not str(params.get("name") or "").strip():
            params["name"] = lib
        out.append(PlanStep(s.action, params))
    return Plan(steps=out)


def _plan_is_capability_coherent(plan: Plan, *, utterance: str) -> bool:
    """Reject LLM plans that contradict subtask functions for this utterance."""
    if plan.is_empty():
        return False
    actions = [s.action for s in plan.steps]
    n_rep = _utterance_repeat_count(utterance)
    n_pm = actions.count("power_measure")
    n_replay = actions.count("traj_replay")
    n_record = actions.count("record_power")
    t = utterance.lower()

    # Repeating a measure at one place must reuse traj_replay after the first click.
    if n_rep is not None and n_rep >= 2 and re.search(
        r"\b(measure|power|same\s+spot|same\s+place)\b", t
    ):
        if n_pm > 1:
            return False
        if n_pm == 1 and n_replay < n_rep - 1:
            return False
        if n_pm == 0 and n_replay < n_rep:
            return False

    # Explicit record/log power requests need record_power (once per visit when N given).
    if re.search(r"\b(record|log|sample)\b.*\bpower\b|\bpower\b.*\b(record|log|sample)\b", t):
        if n_record < 1:
            return False
        if n_rep is not None and n_record < n_rep:
            return False

    n_put = actions.count("put_sensor")
    if _utterance_asks_put_sensor(utterance) and n_put < 1:
        return False

    # traj_replay / traj_save need a name.
    for s in plan.steps:
        if s.action in ("traj_replay", "traj_save") and not str(s.params.get("name") or "").strip():
            return False
        if s.action == "wait" and s.params.get("seconds") is None:
            return False

    return True


def _parse_inline_command(text: str, command: str) -> Plan:
    """Parse `wait 60`, `traj_save spot`, `return name=spot` style lines."""
    params: dict[str, Any] = {}
    rest = text.strip()
    lowered = rest.lower().replace("-", "_")
    matched = False
    for alias, canon in list(ALIASES.items()) + [(c, c) for c in COMMANDS]:
        if canon != command:
            continue
        for form in (alias.replace("_", " "), alias):
            form_u = form.replace(" ", "_")
            if lowered.startswith(form) or lowered.startswith(form_u):
                # Use length of the matched form in the lowered string
                prefix = form if lowered.startswith(form) else form_u
                rest = rest[len(prefix) :].strip()
                matched = True
                break
        if matched:
            break
    for m in re.finditer(r"([A-Za-z_][\w]*)\s*=\s*([^\s]+)", rest):
        params[m.group(1)] = _coerce(m.group(2))
    leftover = re.sub(r"([A-Za-z_][\w]*)\s*=\s*([^\s]+)", "", rest).strip()
    if leftover:
        if command == "wait":
            try:
                params.setdefault("seconds", float(leftover.split()[0]))
            except ValueError:
                pass
        elif command in ("traj_save", "traj_replay"):
            params.setdefault("name", leftover.split()[0])
        elif command == "return":
            token = leftover.split()[0]
            # Ignore English glue: "return to work" is bare return, not name=to.
            if token.lower() not in {"to", "back", "home", "work", "the"}:
                params.setdefault("name", token)
        elif command == "power_measure":
            params.setdefault("save_as", leftover.split()[0])
        elif command == "record_power":
            try:
                params.setdefault("seconds", float(leftover.split()[0]))
            except ValueError:
                pass
    return Plan(steps=[PlanStep(command, params)])


def map_utterance_to_plan(
    text: str,
    *,
    backend: str = "ollama",
    model: str | None = None,
    use_llm: bool = True,
) -> Plan:
    """Return a (possibly multi-step) session plan from exact tokens or the LLM."""
    direct = normalize_command(text)
    if direct and not text.strip().lower().startswith("llm "):
        first = text.strip().split()[0].lower().replace("-", "_")
        if first in COMMANDS or first in ALIASES or normalize_command(text) == direct:
            spaced = " ".join(text.strip().lower().replace("-", "_").replace("_", " ").split())
            cmd_forms = {direct.replace("_", " "), direct}
            if any(spaced == f or spaced.startswith(f + " ") for f in cmd_forms):
                return _parse_inline_command(text, direct)

    body = text.strip()
    if body.lower().startswith("llm "):
        body = body[4:].strip()
        use_llm = True

    if not use_llm:
        raise RuntimeError(
            "Free-text planning requires the LLM (keyword fallback removed). "
            "Use an exact command token, or omit --no-llm / enable LLM."
        )

    client = _make_llm_client(backend, model)
    raw = client.complete(_PROMPT, body)
    data = _parse_json_object(raw)
    plan = _sanitize_plan(expand_plan_document(data), utterance=body)
    if not _plan_is_capability_coherent(plan, utterance=body):
        raise RuntimeError(
            "LLM plan failed capability check "
            f"(actions={[s.action for s in plan.steps]!r}). Raw={raw[:400]!r}"
        )
    plan.from_llm = True
    return plan


def map_utterance_to_command(
    text: str,
    *,
    backend: str = "ollama",
    model: str | None = None,
    use_llm: bool = True,
) -> str:
    """Compatibility: return the first action of the mapped plan."""
    plan = map_utterance_to_plan(
        text, backend=backend, model=model, use_llm=use_llm
    )
    if plan.is_empty():
        return "help"
    return plan.steps[0].action
