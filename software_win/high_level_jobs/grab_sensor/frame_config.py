"""Load canonical robot↔table frame and work pose from physical_env/robot_in_table.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .paths import DEFAULT_ROBOT_IN_TABLE_YAML, REPO_ROOT

if TYPE_CHECKING:
    from .models import RobotInTableFrame

# Firmware rest [0, -75, 180, 0, 0, 0] vs HOME [0, 0, 90, 0, 0, 0] midpoint
# used if YAML omits work_position_deg.
_DEFAULT_WORK_POSITION_DEG: tuple[float, ...] = (0.0, -37.5, 135.0, 0.0, 0.0, 0.0)
_DEFAULT_REST_POSITION_DEG: tuple[float, ...] = (0.0, -75.0, 180.0, 0.0, 0.0, 0.0)


def _resolve_yaml_path(path: Path | str | None = None) -> Path:
    yaml_path = Path(path) if path is not None else DEFAULT_ROBOT_IN_TABLE_YAML
    if not yaml_path.is_absolute():
        yaml_path = (REPO_ROOT / yaml_path).resolve()
    return yaml_path


def load_robot_station_yaml(path: Path | str | None = None) -> dict[str, Any]:
    """Load full ``physical_env/robot_in_table.yaml`` mapping."""
    yaml_path = _resolve_yaml_path(path)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"robot_in_table yaml not found: {yaml_path}")
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML required to load physical_env/robot_in_table.yaml") from exc
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Expected mapping in {yaml_path}")
    return dict(data)


def load_robot_in_table_yaml(path: Path | None = None) -> dict[str, Any]:
    """Return the ``robot_in_table_frame`` mapping from the shared YAML."""
    data = load_robot_station_yaml(path)
    block = data.get("robot_in_table_frame")
    if not isinstance(block, Mapping):
        raise ValueError(f"robot_in_table_frame missing in {_resolve_yaml_path(path)}")
    return dict(block)


def resolve_work_position_deg(
    *,
    shared_yaml: Path | str | None = None,
    override: Sequence[float] | None = None,
) -> list[float]:
    """Work pose (firmware deg) for grab_sensor / power_measure start & return.

    Prefer explicit ``override``, else ``work_position_deg`` from the shared YAML,
    else midpoint between rest [0, -75, 180, ...] and HOME [0, 0, 90, ...].
    """
    if override is not None:
        vals = [float(v) for v in override]
    else:
        try:
            data = load_robot_station_yaml(shared_yaml)
            raw = data.get("work_position_deg")
        except FileNotFoundError:
            raw = None
        if raw is None:
            vals = list(_DEFAULT_WORK_POSITION_DEG)
        else:
            vals = [float(v) for v in raw]
    while len(vals) < 6:
        vals.append(0.0)
    return vals[:6]


def resolve_rest_position_deg(
    *,
    shared_yaml: Path | str | None = None,
    override: Sequence[float] | None = None,
) -> list[float]:
    """Folded rest pose (firmware deg). Default [0, -75, 180, 0, 0, 0]."""
    if override is not None:
        vals = [float(v) for v in override]
    else:
        try:
            data = load_robot_station_yaml(shared_yaml)
            raw = data.get("rest_position_deg")
        except FileNotFoundError:
            raw = None
        if raw is None:
            vals = list(_DEFAULT_REST_POSITION_DEG)
        else:
            vals = [float(v) for v in raw]
    while len(vals) < 6:
        vals.append(0.0)
    return vals[:6]


def resolve_robot_in_table_frame(
    *,
    shared_yaml: Path | str | None = None,
    override: Mapping[str, Any] | None = None,
) -> "RobotInTableFrame":
    """Load shared physical_env pose; optional job-yaml keys override fields."""
    from .models import RobotInTableFrame

    try:
        raw = load_robot_in_table_yaml(Path(shared_yaml) if shared_yaml else None)
    except FileNotFoundError:
        raw = {}
    if override:
        # Only override keys that are explicitly present in the job yaml block.
        raw = {**raw, **dict(override)}
    ti = raw.get("translation_in", [10.5, -5.4, 0.5])
    rr = raw.get("rpy_deg", [0.0, 0.0, 0.0])
    return RobotInTableFrame(
        translation_in=tuple(float(v) for v in ti)[:3],  # type: ignore[arg-type]
        rpy_deg=tuple(float(v) for v in rr)[:3],  # type: ignore[arg-type]
    )
