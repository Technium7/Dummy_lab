"""Save and load physical robot run logs (``last_run_results.npz``)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_RUN_DIR = Path(__file__).resolve().parent


def default_run_results_path() -> Path:
    return _RUN_DIR / "last_run_results.npz"


def _cycle_records_to_arrays(records: List[dict]) -> Dict[str, np.ndarray]:
    n = len(records)
    at_target = np.full((n, 6), np.nan, dtype=np.float64)
    at_home = np.full((n, 6), np.nan, dtype=np.float64)
    goal_max_err = np.full(n, np.nan, dtype=np.float64)
    home_max_err = np.full(n, np.nan, dtype=np.float64)
    goal_joint_err = np.full((n, 6), np.nan, dtype=np.float64)
    home_joint_err = np.full((n, 6), np.nan, dtype=np.float64)
    cycles = np.zeros(n, dtype=np.int32)

    for i, rec in enumerate(records):
        cycles[i] = int(rec.get("cycle", i + 1))
        if rec.get("at_target") is not None:
            at_target[i] = np.asarray(rec["at_target"][:6], dtype=np.float64)
        if rec.get("at_home") is not None:
            at_home[i] = np.asarray(rec["at_home"][:6], dtype=np.float64)
        if rec.get("goal_max_err") is not None:
            goal_max_err[i] = float(rec["goal_max_err"])
        if rec.get("home_max_err") is not None:
            home_max_err[i] = float(rec["home_max_err"])
        if rec.get("goal_joint_err") is not None:
            goal_joint_err[i] = np.asarray(rec["goal_joint_err"][:6], dtype=np.float64)
        if rec.get("home_joint_err") is not None:
            home_joint_err[i] = np.asarray(rec["home_joint_err"][:6], dtype=np.float64)

    return {
        "cycles": cycles,
        "at_target_deg": at_target,
        "at_home_deg": at_home,
        "goal_max_err_deg": goal_max_err,
        "home_max_err_deg": home_max_err,
        "goal_joint_err_deg": goal_joint_err,
        "home_joint_err_deg": home_joint_err,
    }


def save_run_results(
    path: Path,
    *,
    cycle_records: List[dict],
    start_deg: Sequence[float],
    goal_deg: Sequence[float],
    start_q: Sequence[float],
    goal_q: Sequence[float],
    target_pos: Optional[Sequence[float]] = None,
    target_rpy: Optional[Sequence[float]] = None,
    plan_path: Optional[Path] = None,
    initial_joints_deg: Optional[Sequence[float]] = None,
    config: Optional[dict] = None,
) -> Path:
    """Write run log to ``path`` (``.npz``). Returns resolved output path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    arrays = _cycle_records_to_arrays(cycle_records)
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_cycles": len(cycle_records),
        "plan_path": str(plan_path) if plan_path else "",
        "config": config or {},
    }

    payload: Dict[str, Any] = {
        **arrays,
        "start_deg": np.asarray(start_deg[:6], dtype=np.float64),
        "goal_deg": np.asarray(goal_deg[:6], dtype=np.float64),
        "start_q": np.asarray(start_q[:6], dtype=np.float64),
        "goal_q": np.asarray(goal_q[:6], dtype=np.float64),
        "meta_json": np.array(json.dumps(meta)),
    }
    if target_pos is not None:
        payload["target_pos"] = np.asarray(target_pos[:3], dtype=np.float64)
    if target_rpy is not None:
        payload["target_rpy"] = np.asarray(target_rpy[:3], dtype=np.float64)
    if initial_joints_deg is not None:
        payload["initial_joints_deg"] = np.asarray(initial_joints_deg[:6], dtype=np.float64)

    np.savez_compressed(out, **payload)
    return out.resolve()


def load_run_results(path: Path | None = None) -> Dict[str, Any]:
    """Load run log; returns arrays plus parsed ``meta`` dict."""
    p = Path(path) if path else default_run_results_path()
    if not p.is_file():
        raise FileNotFoundError(p)

    data = np.load(p, allow_pickle=False)
    out: Dict[str, Any] = {k: data[k] for k in data.files if k != "meta_json"}
    out["meta"] = json.loads(str(data["meta_json"]))
    out["path"] = p.resolve()
    return out
