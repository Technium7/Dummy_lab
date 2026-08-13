"""Config models for epoxy pre_apply / xy_optimize / z_creep phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Tuple

from high_level_jobs.grab_sensor.models import (
    ApproachConfig,
    CameraToRobot,
    ExecuteConfig,
    GraspMotionConfig,
    GrabSensorConfig,
    RobotInTableFrame,
)
from robot_config import DEFAULT_ROBOT_NAME

from .paths import DEBUG_DIR, REPO_ROOT

_INCH_TO_M = 0.0254

EpoxyPhase = Literal["pre_apply", "xy_optimize", "z_creep"]
ALL_PHASES: tuple[EpoxyPhase, ...] = ("pre_apply", "xy_optimize", "z_creep")


@dataclass(frozen=True)
class PreApplyConfig:
    """Phase 1: single motion WORK → rough TCP (safety-offset pose); stay."""

    orientation_tol_deg: float = 90.0
    plan_return_home: bool = False

    def to_grab_approach(self) -> ApproachConfig:
        """Grab-sensor ApproachConfig: no mid-air segment, no lift/return."""
        return ApproachConfig(
            pre_grasp_offset_mm=0.0,
            pre_grasp_along_xy=False,
            plan_pre_grasp=False,
            pre_grasp_orientation_tol_deg=float(self.orientation_tol_deg),
            post_grasp_lift_mm=0.0,
            plan_return_home=bool(self.plan_return_home),
            retreat_after_close_mm=0.0,
        )


@dataclass(frozen=True)
class XyOptimizeConfig:
    """Phase 2: camera XY refine at fixed Z (placeholder until camera wired)."""

    enabled: bool = True
    placeholder: bool = True
    camera: str = "zelux"


@dataclass(frozen=True)
class ZCreepConfig:
    """Phase 3: vision Z creeps (placeholder; full FSM in epoxy_apply)."""

    enabled: bool = True
    placeholder: bool = True


@dataclass(frozen=True)
class EpoxyApproachConfig:
    """Fixed table-frame rough TCP + phased apply pipeline config."""

    pre_apply: PreApplyConfig = field(default_factory=PreApplyConfig)
    xy_optimize: XyOptimizeConfig = field(default_factory=XyOptimizeConfig)
    z_creep: ZCreepConfig = field(default_factory=ZCreepConfig)
    nominal_tcp_table_xyz_in: Tuple[float, float, float] = (10.75, 8.25, 5.5)
    safety_offset_y_in: float = 0.25
    safety_offset_z_in: float = 0.25
    tcp_table_xyz_in: Tuple[float, float, float] | None = None
    # UJ tilt via A (roll), B=0: A=-30 → 30° from table-parallel toward tip-down.
    # Do not put this tilt in B (that pitches the gripper about Y).
    fixed_grasp_ab_deg: Tuple[float, float] = (-30.0, 0.0)
    uj_tilt_from_parallel_deg: float = 30.0
    grasp_c_offset_deg: float = 0.0
    grasp_rpy_yaw_base_deg: float = 180.0
    grasp_rpy_roll_offset_deg: float = -15.64
    # C not taught: IK enforces A/B only; yaw floats from the solution.
    ik_ignore_yaw: bool = True
    yaw_from_sensor_axis: bool = False
    load_environment: bool = True
    env_path: Path = field(default_factory=lambda: Path("physical_env/packaging_station_edit.obj"))
    mesh_scale: float = 1.0
    robot_name: str = DEFAULT_ROBOT_NAME
    end_link: str = "link6_1"
    tcp_link: str = "grasp_tcp_link"
    enable_collision: bool = False
    plan_seed: int = 42
    save_debug_dir: Path | None = None
    grasp_orientation_tol_deg: float = 50.0
    grasp_position_tol_mm: float = 20.0
    ik_ori_vs_pos: float = 1.0
    robot_in_table_frame: RobotInTableFrame = field(default_factory=RobotInTableFrame)
    work_position_deg: Tuple[float, float, float, float, float, float] = (
        0.0,
        -37.5,
        135.0,
        0.0,
        0.0,
        0.0,
    )
    execute: ExecuteConfig = field(default_factory=ExecuteConfig)
    camera_to_robot: CameraToRobot = field(
        default_factory=lambda: CameraToRobot(
            translation_m=(0.0, 0.0, 0.0),
            rpy_deg=(0.0, 0.0, 0.0),
        )
    )

    @property
    def approach(self) -> ApproachConfig:
        """Grab-sensor adaptor view of pre_apply."""
        return self.pre_apply.to_grab_approach()

    def effective_tcp_table_xyz_in(self) -> Tuple[float, float, float]:
        if self.tcp_table_xyz_in is not None:
            return (
                float(self.tcp_table_xyz_in[0]),
                float(self.tcp_table_xyz_in[1]),
                float(self.tcp_table_xyz_in[2]),
            )
        nx, ny, nz = (float(v) for v in self.nominal_tcp_table_xyz_in)
        return (
            nx,
            ny - float(self.safety_offset_y_in),
            nz + float(self.safety_offset_z_in),
        )

    def effective_tcp_table_xyz_m(self) -> Tuple[float, float, float]:
        xin, yin, zin = self.effective_tcp_table_xyz_in()
        return (xin * _INCH_TO_M, yin * _INCH_TO_M, zin * _INCH_TO_M)

    def to_grab_sensor_config(self) -> GrabSensorConfig:
        """Adaptor so we reuse grab_sensor plan_motion / execute / ik_debug."""
        _, _, tz_in = self.effective_tcp_table_xyz_in()
        tcp_z_mm = float(tz_in) * 25.4
        return GrabSensorConfig(
            camera_to_robot=self.camera_to_robot,
            approach=self.approach,
            vision_mode="cascade",
            camera_serial="",
            fixed_grasp_ab_deg=self.fixed_grasp_ab_deg,
            yaw_from_sensor_axis=self.yaw_from_sensor_axis,
            grasp_c_offset_deg=self.grasp_c_offset_deg,
            grasp_rpy_yaw_base_deg=self.grasp_rpy_yaw_base_deg,
            grasp_rpy_roll_offset_deg=self.grasp_rpy_roll_offset_deg,
            grasp_offset_from_holder_mm=0.0,
            load_environment=self.load_environment,
            env_path=self.env_path,
            mesh_scale=self.mesh_scale,
            robot_name=self.robot_name,
            end_link=self.end_link,
            tcp_link=self.tcp_link,
            enable_collision=self.enable_collision,
            plan_seed=self.plan_seed,
            save_debug_dir=self.save_debug_dir,
            table_calib_path=None,
            grasp_position_from_table=True,
            grasp_motion=GraspMotionConfig(grasp_height_above_table_mm=tcp_z_mm),
            grasp_orientation_tol_deg=self.grasp_orientation_tol_deg,
            grasp_position_tol_mm=self.grasp_position_tol_mm,
            ik_ori_vs_pos=self.ik_ori_vs_pos,
            prefer_nearby_wrist=True,
            ik_ignore_yaw=bool(self.ik_ignore_yaw),
            robot_in_table_frame=self.robot_in_table_frame,
            work_position_deg=self.work_position_deg,
            execute=self.execute,
            fixed_grasp_abc_input_deg=(
                float(self.fixed_grasp_ab_deg[0]),
                float(self.fixed_grasp_ab_deg[1]),
                0.0,
            ),
        )

    @staticmethod
    def from_mapping(data: Mapping[str, Any], base_dir: Path | None = None) -> "EpoxyApproachConfig":
        base = base_dir or REPO_ROOT

        def _resolve(raw: str | Path | None, default: Path) -> Path:
            if raw is None or str(raw).strip() == "":
                p = default
            else:
                p = Path(str(raw))
            if not p.is_absolute():
                p = (base / p).resolve()
            return p

        # Prefer pre_apply block; fall back to legacy approach: / pre_grasp_* keys.
        pre_raw = data.get("pre_apply") or data.get("approach") or {}
        if not isinstance(pre_raw, Mapping):
            pre_raw = {}
        xy_raw = data.get("xy_optimize") or {}
        if not isinstance(xy_raw, Mapping):
            xy_raw = {}
        z_raw = data.get("z_creep") or {}
        if not isinstance(z_raw, Mapping):
            z_raw = {}
        execute_raw = data.get("execute") or {}
        paths_raw = data.get("paths") or {}
        shared_frames = data.get("robot_in_table_yaml")
        robot_override = data.get("robot_in_table_frame")
        if isinstance(robot_override, Mapping) and not robot_override:
            robot_override = None

        tilt = float(
            data.get(
                "uj_tilt_from_parallel_deg",
                data.get("surface_tilt_from_parallel_deg", 30.0),
            )
        )
        # Prefer explicit A/B; else A=-(tilt), B=0 (UJ roll tilt, no Y pitch).
        if "fixed_grasp_ab_deg" in data:
            ab = data.get("fixed_grasp_ab_deg", [-tilt, 0.0])
            fixed_ab = (float(ab[0]), float(ab[1]))
        else:
            fixed_ab = (-tilt, 0.0)

        from high_level_jobs.grab_sensor.frame_config import (
            resolve_robot_in_table_frame,
            resolve_work_position_deg,
        )

        robot_in_table = resolve_robot_in_table_frame(
            shared_yaml=shared_frames,
            override=robot_override if isinstance(robot_override, Mapping) else None,
        )
        work_override = data.get("work_position_deg")
        work_deg = resolve_work_position_deg(
            shared_yaml=shared_frames,
            override=work_override if isinstance(work_override, (list, tuple)) else None,
        )

        debug_raw = paths_raw.get("save_debug_dir")
        save_debug = _resolve(debug_raw, DEBUG_DIR) if debug_raw is not None else DEBUG_DIR
        env_path = _resolve(data.get("env_path"), Path("physical_env/packaging_station_edit.obj"))

        tcp_override = data.get("tcp_table_xyz_in")
        tcp_xyz: Tuple[float, float, float] | None = None
        if isinstance(tcp_override, (list, tuple)) and len(tcp_override) >= 3:
            tcp_xyz = (float(tcp_override[0]), float(tcp_override[1]), float(tcp_override[2]))

        nominal = data.get("nominal_tcp_table_xyz_in", [10.75, 8.25, 5.5])
        nominal_xyz = (float(nominal[0]), float(nominal[1]), float(nominal[2]))
        cam_raw = data.get("camera_to_robot") or {}

        def _pre_key(*names: str, default: Any) -> Any:
            for name in names:
                if name in pre_raw:
                    return pre_raw[name]
            return default

        return EpoxyApproachConfig(
            pre_apply=PreApplyConfig(
                orientation_tol_deg=float(
                    _pre_key(
                        "orientation_tol_deg",
                        "pre_apply_orientation_tol_deg",
                        "pre_grasp_orientation_tol_deg",
                        default=90.0,
                    )
                ),
                plan_return_home=bool(pre_raw.get("plan_return_home", False)),
            ),
            xy_optimize=XyOptimizeConfig(
                enabled=bool(xy_raw.get("enabled", True)),
                placeholder=bool(xy_raw.get("placeholder", True)),
                camera=str(xy_raw.get("camera", "zelux") or "zelux"),
            ),
            z_creep=ZCreepConfig(
                enabled=bool(z_raw.get("enabled", True)),
                placeholder=bool(z_raw.get("placeholder", True)),
            ),
            nominal_tcp_table_xyz_in=nominal_xyz,
            safety_offset_y_in=float(data.get("safety_offset_y_in", 0.25)),
            safety_offset_z_in=float(data.get("safety_offset_z_in", 0.25)),
            tcp_table_xyz_in=tcp_xyz,
            fixed_grasp_ab_deg=fixed_ab,
            uj_tilt_from_parallel_deg=tilt,
            grasp_c_offset_deg=float(data.get("grasp_c_offset_deg", 0.0)),
            grasp_rpy_yaw_base_deg=float(data.get("grasp_rpy_yaw_base_deg", 180.0)),
            grasp_rpy_roll_offset_deg=float(data.get("grasp_rpy_roll_offset_deg", -15.64)),
            ik_ignore_yaw=bool(data.get("ik_ignore_yaw", True)),
            yaw_from_sensor_axis=bool(data.get("yaw_from_sensor_axis", False)),
            load_environment=bool(data.get("load_environment", True)),
            env_path=env_path,
            mesh_scale=float(data.get("mesh_scale", 1.0)),
            robot_name=str(
                paths_raw.get("robot")
                or data.get("robot_name")
                or "robot_design/Dummy_ros2_gripper_description/urdf/Dummy_ros2_gripper_grasp.urdf"
            ),
            end_link=str(paths_raw.get("end_link", "link6_1")),
            tcp_link=str(paths_raw.get("tcp_link", "grasp_tcp_link")),
            enable_collision=bool(data.get("enable_collision", False)),
            plan_seed=int(data.get("plan_seed", 42)),
            save_debug_dir=save_debug,
            grasp_orientation_tol_deg=float(data.get("grasp_orientation_tol_deg", 50.0)),
            grasp_position_tol_mm=float(data.get("grasp_position_tol_mm", 20.0)),
            ik_ori_vs_pos=float(data.get("ik_ori_vs_pos", 1.0)),
            robot_in_table_frame=robot_in_table,
            work_position_deg=tuple(float(v) for v in work_deg)[:6],  # type: ignore[arg-type]
            execute=ExecuteConfig(
                joint_current_limit_a=float(execute_raw.get("joint_current_limit_a", 0.0)),
                home_confirm_attempts=int(execute_raw.get("home_confirm_attempts", 3)),
            ),
            camera_to_robot=CameraToRobot(
                translation_m=tuple(float(v) for v in cam_raw.get("translation_m", [0, 0, 0]))[:3],  # type: ignore[arg-type]
                rpy_deg=tuple(float(v) for v in cam_raw.get("rpy_deg", [0, 0, 0]))[:3],  # type: ignore[arg-type]
            ),
        )
