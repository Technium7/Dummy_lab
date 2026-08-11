"""Config models for power_measure (click-to-insert)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Tuple

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


@dataclass(frozen=True)
class PowerMeterConfig:
    """Thorlabs PM100(D) via ThorlabsPM100 + PyVISA."""

    enabled: bool = True
    wavelength_nm: float = 400.0
    visa_resource: str = ""
    serial: str = ""
    settle_s: float = 0.15
    samples_per_read: int = 3
    timeout_ms: int = 5000


@dataclass(frozen=True)
class PowerScanConfig:
    """Search / maximize in plane ⊥ robot→target (left × world +Z).

    Region is a ``side_mm`` × ``side_mm`` square. Slow serpentine scan streams
    between grid points without post-move settle waits.
    """

    side_mm: float = 20.0
    sensor_radius_mm: float = 10.0
    # Lateral sample spacing (mm). Default 5 mm — denser than sensor radius so
    # the peak is not missed; still far fewer points than 1 mm hill-climb.
    grid_step_mm: float = 5.0
    detect_threshold_w: float = 1.0e-6
    contrast_ratio: float = 2.0
    contrast_delta_w: float = 1.0e-6
    insert_samples: int = 8
    # Kept for YAML compat; directed climb replaces dense neighbor search.
    optimize_step_mm: float = 2.5
    optimize_min_step_mm: float = 1.25
    optimize_max_iters: int = 16
    optimize_min_rel_improve: float = 0.02
    optimize_use_diagonals: bool = False
    move_speed_scale: float = 0.12
    search_speed_scale: float = 0.12
    climb_speed_scale: float = 0.10
    send_interval_ms: float = 220.0
    settle_timeout_sec: float = 0.0
    # Brief pause only for the meter (no robot settle wait).
    meter_pause_s: float = 0.25
    # Final approach to peak: slower move, then hold and average power.
    final_speed_scale: float = 0.08
    final_send_interval_ms: float = 280.0
    final_hold_s: float = 0.7
    final_meter_samples: int = 5

    def sample_step_mm(self) -> float:
        return max(float(self.grid_step_mm), 0.5)

    def opt_step_mm(self) -> float:
        return max(float(self.optimize_step_mm), 0.5)

    def opt_min_step_mm(self) -> float:
        return max(float(self.optimize_min_step_mm), 0.5)
@dataclass(frozen=True)
class PowerMeasureConfig:
    """Insert TCP relative to a clicked measured point on the table."""

    camera_to_robot: CameraToRobot
    approach: ApproachConfig
    camera_serial: str = ""
    measured_height_in: float = 4.0
    tcp_above_measured_in: float = 4.25
    tcp_left_of_radial_in: float = 1.25
    yaw_cw_from_radial_deg: float = 90.0
    # A=-90 + B=+90: link local X (ik-gui long red) || table +Z; tip out in front
    fixed_grasp_ab_deg: Tuple[float, float] = (-90.0, 90.0)
    grasp_c_offset_deg: float = -90.0
    grasp_rpy_yaw_base_deg: float = 180.0
    grasp_rpy_roll_offset_deg: float = -18.23
    load_environment: bool = False
    env_path: Path = field(default_factory=lambda: Path("physical_env/packaging_station_edit.obj"))
    mesh_scale: float = 1.0
    robot_name: str = DEFAULT_ROBOT_NAME
    end_link: str = "link6_1"
    tcp_link: str = "grasp_tcp_link"
    enable_collision: bool = True
    plan_seed: int = 42
    save_debug_dir: Path | None = None
    table_calib_path: Path | None = None
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
    # Min radial length (m) before falling back to table -X as "left"
    min_radial_m: float = 1e-3
    power_meter: PowerMeterConfig = field(default_factory=PowerMeterConfig)
    power_scan: PowerScanConfig = field(default_factory=PowerScanConfig)

    def to_grab_sensor_config(self) -> GrabSensorConfig:
        """Adaptor so we can reuse grab_sensor plan_motion / execute / tcp_ik."""
        tcp_z_mm = (float(self.measured_height_in) + float(self.tcp_above_measured_in)) * 25.4
        return GrabSensorConfig(
            camera_to_robot=self.camera_to_robot,
            approach=self.approach,
            vision_mode="cascade",
            camera_serial=self.camera_serial,
            fixed_grasp_ab_deg=self.fixed_grasp_ab_deg,
            # finalize_grasp_targets uses sensor_axis_deg = radial - yaw_cw
            yaw_from_sensor_axis=True,
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
            table_calib_path=self.table_calib_path,
            grasp_position_from_table=True,
            grasp_motion=GraspMotionConfig(grasp_height_above_table_mm=tcp_z_mm),
            grasp_orientation_tol_deg=self.grasp_orientation_tol_deg,
            grasp_position_tol_mm=self.grasp_position_tol_mm,
            ik_ori_vs_pos=self.ik_ori_vs_pos,
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
    def from_mapping(data: Mapping[str, Any], base_dir: Path | None = None) -> "PowerMeasureConfig":
        base = base_dir or REPO_ROOT

        def _resolve(raw: str | Path | None, default: Path) -> Path:
            if raw is None or str(raw).strip() == "":
                p = default
            else:
                p = Path(str(raw))
            if not p.is_absolute():
                p = (base / p).resolve()
            return p

        cam_raw = data.get("camera_to_robot") or {}
        approach_raw = data.get("approach") or {}
        execute_raw = data.get("execute") or {}
        paths_raw = data.get("paths") or {}
        shared_frames = data.get("robot_in_table_yaml")
        robot_override = data.get("robot_in_table_frame")
        if isinstance(robot_override, Mapping) and not robot_override:
            robot_override = None

        ab = data.get("fixed_grasp_ab_deg", [-90.0, 90.0])
        fixed_ab = (float(ab[0]), float(ab[1]))

        meter_raw = data.get("power_meter") or {}
        scan_raw = data.get("power_scan") or {}
        if not isinstance(meter_raw, Mapping):
            meter_raw = {}
        if not isinstance(scan_raw, Mapping):
            scan_raw = {}

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

        table_raw = data.get("table_calib_path")
        table_path = _resolve(table_raw, Path(".")) if table_raw else None
        if table_path is not None and table_raw is None:
            table_path = None

        env_path = _resolve(data.get("env_path"), Path("physical_env/packaging_station_edit.obj"))

        return PowerMeasureConfig(
            camera_to_robot=CameraToRobot(
                translation_m=tuple(float(v) for v in cam_raw.get("translation_m", [0, 0, 0]))[:3],  # type: ignore[arg-type]
                rpy_deg=tuple(float(v) for v in cam_raw.get("rpy_deg", [0, 0, 0]))[:3],  # type: ignore[arg-type]
            ),
            approach=ApproachConfig(
                pre_grasp_offset_mm=float(approach_raw.get("pre_grasp_offset_mm", 50.0)),
                pre_grasp_along_xy=bool(approach_raw.get("pre_grasp_along_xy", False)),
                plan_pre_grasp=bool(approach_raw.get("plan_pre_grasp", True)),
                pre_grasp_orientation_tol_deg=float(
                    approach_raw.get("pre_grasp_orientation_tol_deg", 90.0)
                ),
                post_grasp_lift_mm=float(approach_raw.get("post_grasp_lift_mm", 80.0)),
                plan_return_home=bool(approach_raw.get("plan_return_home", True)),
                retreat_after_close_mm=float(approach_raw.get("retreat_after_close_mm", 0.0)),
            ),
            camera_serial=str(data.get("camera_serial", "") or ""),
            measured_height_in=float(data.get("measured_height_in", 4.0)),
            tcp_above_measured_in=float(data.get("tcp_above_measured_in", 4.25)),
            tcp_left_of_radial_in=float(data.get("tcp_left_of_radial_in", 1.25)),
            yaw_cw_from_radial_deg=float(data.get("yaw_cw_from_radial_deg", 90.0)),
            fixed_grasp_ab_deg=fixed_ab,
            grasp_c_offset_deg=float(data.get("grasp_c_offset_deg", -90.0)),
            grasp_rpy_yaw_base_deg=float(data.get("grasp_rpy_yaw_base_deg", 180.0)),
            grasp_rpy_roll_offset_deg=float(data.get("grasp_rpy_roll_offset_deg", -18.23)),
            load_environment=bool(data.get("load_environment", False)),
            env_path=env_path,
            mesh_scale=float(data.get("mesh_scale", 1.0)),
            robot_name=str(
                paths_raw.get("robot")
                or data.get("robot_name")
                or "robot_design/Dummy_ros2_gripper_description/urdf/Dummy_ros2_gripper_grasp.urdf"
            ),
            end_link=str(paths_raw.get("end_link", "link6_1")),
            tcp_link=str(paths_raw.get("tcp_link", "grasp_tcp_link")),
            enable_collision=bool(data.get("enable_collision", True)),
            plan_seed=int(data.get("plan_seed", 42)),
            save_debug_dir=save_debug,
            table_calib_path=table_path,
            grasp_orientation_tol_deg=float(data.get("grasp_orientation_tol_deg", 50.0)),
            grasp_position_tol_mm=float(data.get("grasp_position_tol_mm", 20.0)),
            ik_ori_vs_pos=float(data.get("ik_ori_vs_pos", 1.0)),
            robot_in_table_frame=robot_in_table,
            work_position_deg=tuple(float(v) for v in work_deg)[:6],  # type: ignore[arg-type]
            execute=ExecuteConfig(
                joint_current_limit_a=float(execute_raw.get("joint_current_limit_a", 0.0)),
                home_confirm_attempts=int(execute_raw.get("home_confirm_attempts", 3)),
            ),
            min_radial_m=float(data.get("min_radial_m", 1e-3)),
            power_meter=PowerMeterConfig(
                enabled=bool(meter_raw.get("enabled", True)),
                wavelength_nm=float(meter_raw.get("wavelength_nm", 400.0)),
                visa_resource=str(meter_raw.get("visa_resource", "") or ""),
                serial=str(meter_raw.get("serial", "") or ""),
                settle_s=float(meter_raw.get("settle_s", 0.15)),
                samples_per_read=int(meter_raw.get("samples_per_read", 3)),
                timeout_ms=int(meter_raw.get("timeout_ms", 5000)),
            ),
            power_scan=PowerScanConfig(
                side_mm=float(
                    scan_raw.get(
                        "side_mm",
                        scan_raw.get("range_mm", scan_raw.get("radius_mm", 20.0)),
                    )
                ),
                sensor_radius_mm=float(scan_raw.get("sensor_radius_mm", 10.0)),
                grid_step_mm=float(scan_raw.get("grid_step_mm", 5.0)),
                detect_threshold_w=float(scan_raw.get("detect_threshold_w", 1.0e-6)),
                contrast_ratio=float(scan_raw.get("contrast_ratio", 2.0)),
                contrast_delta_w=float(scan_raw.get("contrast_delta_w", 1.0e-6)),
                insert_samples=int(scan_raw.get("insert_samples", 8)),
                optimize_step_mm=float(scan_raw.get("optimize_step_mm", 2.5)),
                optimize_min_step_mm=float(scan_raw.get("optimize_min_step_mm", 1.25)),
                optimize_max_iters=int(scan_raw.get("optimize_max_iters", 16)),
                optimize_min_rel_improve=float(scan_raw.get("optimize_min_rel_improve", 0.02)),
                optimize_use_diagonals=bool(scan_raw.get("optimize_use_diagonals", False)),
                move_speed_scale=float(scan_raw.get("move_speed_scale", 0.12)),
                search_speed_scale=float(
                    scan_raw.get("search_speed_scale", scan_raw.get("move_speed_scale", 0.12))
                ),
                climb_speed_scale=float(scan_raw.get("climb_speed_scale", 0.10)),
                send_interval_ms=float(scan_raw.get("send_interval_ms", 220.0)),
                settle_timeout_sec=float(scan_raw.get("settle_timeout_sec", 0.0)),
                meter_pause_s=float(scan_raw.get("meter_pause_s", 0.25)),
                final_speed_scale=float(scan_raw.get("final_speed_scale", 0.08)),
                final_send_interval_ms=float(scan_raw.get("final_send_interval_ms", 280.0)),
                final_hold_s=float(scan_raw.get("final_hold_s", 0.7)),
                final_meter_samples=int(scan_raw.get("final_meter_samples", 5)),
            ),
        )
