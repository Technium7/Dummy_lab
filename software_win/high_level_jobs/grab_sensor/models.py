from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Tuple

import numpy as np

from robot_config import DEFAULT_ROBOT_NAME

from .paths import DEBUG_DIR, MODEL_DIR, REPO_ROOT

VisionMode = Literal["cascade", "yolo"]  # yolo reserved for later
DetectorBackend = Literal["cascade", "yolo"]
ObjectClass = Literal["sensor", "post", "other"]


@dataclass(frozen=True)
class ExecuteConfig:
    joint_current_limit_a: float = 0.0
    home_confirm_attempts: int = 3


@dataclass(frozen=True)
class ApproachConfig:
    pre_grasp_offset_mm: float = 80.0
    pre_grasp_along_xy: bool = False  # approach from above (world +Z)
    plan_pre_grasp: bool = True
    pre_grasp_orientation_tol_deg: float = 90.0
    post_grasp_lift_mm: float = 120.0
    plan_return_home: bool = True
    retreat_after_close_mm: float = 40.0


@dataclass(frozen=True)
class GraspMotionConfig:
    grasp_height_above_table_mm: float = 0.0


@dataclass(frozen=True)
class CameraToRobot:
    translation_m: Tuple[float, float, float]
    rpy_deg: Tuple[float, float, float]


@dataclass(frozen=True)
class RobotInTableFrame:
    translation_in: Tuple[float, float, float] = (10.5, -5.4, 0.5)
    rpy_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def translation_m(self) -> Tuple[float, float, float]:
        inch = 0.0254
        return (
            float(self.translation_in[0]) * inch,
            float(self.translation_in[1]) * inch,
            float(self.translation_in[2]) * inch,
        )


@dataclass(frozen=True)
class VisionModelsConfig:
    detector_backend: DetectorBackend = "yolo"
    proposal_model_path: Path = field(default_factory=lambda: MODEL_DIR / "proposal.pt")
    class_model_path: Path = field(default_factory=lambda: MODEL_DIR / "class_net.pt")
    sensor_head_path: Path = field(default_factory=lambda: MODEL_DIR / "sensor_head.pt")
    center_head_path: Path = field(default_factory=lambda: MODEL_DIR / "center_head.pt")
    confidence: float = 0.35
    crop_padding_px: int = 12
    max_proposals: int = 16
    yolo_model_path: Path | None = field(default_factory=lambda: MODEL_DIR / "sensor_yolo.pt")


@dataclass(frozen=True)
class PostObstacleConfig:
    diameter_in: float = 1.0
    min_height_mm: float = 10.0
    max_height_mm: float = 80.0
    depth_patch_radius_px: int = 3


@dataclass
class GraspTargets:
    pre_grasp_pos_m: np.ndarray
    pre_grasp_rpy_rad: np.ndarray
    grasp_pos_m: np.ndarray
    grasp_rpy_rad: np.ndarray
    object_pos_m: np.ndarray
    object_rpy_rad: np.ndarray
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GrabSensorConfig:
    camera_to_robot: CameraToRobot
    approach: ApproachConfig
    vision_mode: VisionMode = "cascade"
    camera_serial: str = ""
    vision: VisionModelsConfig = field(default_factory=VisionModelsConfig)
    post_obstacle: PostObstacleConfig = field(default_factory=PostObstacleConfig)
    # End-link perpendicular to table: A=-90 → world roll=+90 (TCP -Y down); C from sensor axis
    fixed_grasp_ab_deg: Tuple[float, float] = (-90.0, 0.0)
    yaw_from_sensor_axis: bool = True
    grasp_c_offset_deg: float = 0.0
    grasp_rpy_yaw_base_deg: float = 180.0
    # Contact + along holder on table XY. Default 0.75 inch = 19.05 mm
    grasp_offset_from_holder_mm: float = 19.05
    # Preview offset in image pixels (label/detect overlays)
    grasp_offset_from_holder_px: float = 18.0
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
    grasp_position_from_table: bool = True
    grasp_motion: GraspMotionConfig = field(default_factory=GraspMotionConfig)
    grasp_orientation_tol_deg: float = 50.0
    grasp_position_tol_mm: float = 20.0
    # Simultaneous IK: cost=(pos/pos_tol)^2 + ik_ori_vs_pos*(ori/ori_tol)^2
    # 1.0 = equal at tolerance boundary; <1 favors TCP; >1 favors orientation
    ik_ori_vs_pos: float = 1.0
    robot_in_table_frame: RobotInTableFrame = field(default_factory=RobotInTableFrame)
    # Start/return joints (firmware deg); from physical_env/robot_in_table.yaml work_position_deg
    work_position_deg: Tuple[float, float, float, float, float, float] = (
        0.0,
        -37.5,
        135.0,
        0.0,
        0.0,
        0.0,
    )
    execute: ExecuteConfig = field(default_factory=ExecuteConfig)
    # Compat aliases used by ported plan_motion / tcp_ik
    grasp_orientation_mode: str = "fixed_robot"
    grasp_c_from_xy: bool = False
    fixed_grasp_abc_input_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    grasp_rpy_roll_offset_deg: float = -15.64

    @staticmethod
    def from_mapping(data: Mapping[str, Any], base_dir: Path | None = None) -> "GrabSensorConfig":
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
        motion_raw = data.get("motion_planning") or {}
        vision_raw = data.get("vision") or {}
        post_raw = data.get("post_obstacle") or {}
        paths_raw = data.get("paths") or {}
        shared_frames = data.get("robot_in_table_yaml")
        robot_override = data.get("robot_in_table_frame")
        # Empty override block should not wipe shared yaml — only use if keys present.
        if isinstance(robot_override, Mapping) and not robot_override:
            robot_override = None

        ab = data.get("fixed_grasp_ab_deg", [-90.0, 0.0])
        fixed_ab = (float(ab[0]), float(ab[1]))

        from .frame_config import resolve_robot_in_table_frame, resolve_work_position_deg

        robot_in_table = resolve_robot_in_table_frame(
            shared_yaml=shared_frames,
            override=robot_override if isinstance(robot_override, Mapping) else None,
        )
        work_override = data.get("work_position_deg")
        work_deg = resolve_work_position_deg(
            shared_yaml=shared_frames,
            override=work_override if isinstance(work_override, (list, tuple)) else None,
        )

        vision = VisionModelsConfig(
            detector_backend=str(vision_raw.get("detector_backend", "yolo")).lower(),  # type: ignore[arg-type]
            proposal_model_path=_resolve(
                vision_raw.get("proposal_model_path"), MODEL_DIR / "proposal.pt"
            ),
            class_model_path=_resolve(vision_raw.get("class_model_path"), MODEL_DIR / "class_net.pt"),
            sensor_head_path=_resolve(vision_raw.get("sensor_head_path"), MODEL_DIR / "sensor_head.pt"),
            center_head_path=_resolve(vision_raw.get("center_head_path"), MODEL_DIR / "center_head.pt"),
            confidence=float(vision_raw.get("confidence", 0.35)),
            crop_padding_px=int(vision_raw.get("crop_padding_px", 12)),
            max_proposals=int(vision_raw.get("max_proposals", 16)),
            yolo_model_path=_resolve(
                vision_raw.get("yolo_model_path"), MODEL_DIR / "sensor_yolo.pt"
            ),
        )
        backend = str(vision.detector_backend).lower()
        if backend not in ("cascade", "yolo"):
            raise ValueError(f"detector_backend must be cascade|yolo, got {backend!r}")
        vision = VisionModelsConfig(
            detector_backend=backend,  # type: ignore[arg-type]
            proposal_model_path=vision.proposal_model_path,
            class_model_path=vision.class_model_path,
            sensor_head_path=vision.sensor_head_path,
            center_head_path=vision.center_head_path,
            confidence=vision.confidence,
            crop_padding_px=vision.crop_padding_px,
            max_proposals=vision.max_proposals,
            yolo_model_path=vision.yolo_model_path,
        )

        debug_raw = paths_raw.get("save_debug_dir")
        save_debug = _resolve(debug_raw, DEBUG_DIR) if debug_raw is not None else DEBUG_DIR

        table_raw = data.get("table_calib_path")
        table_path = _resolve(table_raw, Path(".")) if table_raw else None
        if table_path is not None and table_raw is None:
            table_path = None

        env_path = _resolve(data.get("env_path"), Path("physical_env/packaging_station_edit.obj"))

        return GrabSensorConfig(
            camera_to_robot=CameraToRobot(
                translation_m=tuple(float(v) for v in cam_raw.get("translation_m", [0, 0, 0]))[:3],  # type: ignore[arg-type]
                rpy_deg=tuple(float(v) for v in cam_raw.get("rpy_deg", [0, 0, 0]))[:3],  # type: ignore[arg-type]
            ),
            approach=ApproachConfig(
                pre_grasp_offset_mm=float(approach_raw.get("pre_grasp_offset_mm", 80.0)),
                pre_grasp_along_xy=bool(approach_raw.get("pre_grasp_along_xy", False)),
                plan_pre_grasp=bool(approach_raw.get("plan_pre_grasp", True)),
                pre_grasp_orientation_tol_deg=float(
                    approach_raw.get("pre_grasp_orientation_tol_deg", 90.0)
                ),
                post_grasp_lift_mm=float(approach_raw.get("post_grasp_lift_mm", 120.0)),
                plan_return_home=bool(approach_raw.get("plan_return_home", True)),
                retreat_after_close_mm=float(approach_raw.get("retreat_after_close_mm", 40.0)),
            ),
            vision_mode=str(data.get("vision_mode", "cascade")).lower(),  # type: ignore[arg-type]
            camera_serial=str(data.get("camera_serial", "") or ""),
            vision=vision,
            post_obstacle=PostObstacleConfig(
                diameter_in=float(post_raw.get("diameter_in", 1.0)),
                min_height_mm=float(post_raw.get("min_height_mm", 10.0)),
                max_height_mm=float(post_raw.get("max_height_mm", 80.0)),
                depth_patch_radius_px=int(post_raw.get("depth_patch_radius_px", 3)),
            ),
            fixed_grasp_ab_deg=fixed_ab,
            yaw_from_sensor_axis=bool(data.get("yaw_from_sensor_axis", True)),
            grasp_c_offset_deg=float(data.get("grasp_c_offset_deg", 0.0)),
            grasp_rpy_yaw_base_deg=float(data.get("grasp_rpy_yaw_base_deg", 180.0)),
            grasp_rpy_roll_offset_deg=float(data.get("grasp_rpy_roll_offset_deg", -15.64)),
            grasp_offset_from_holder_mm=float(data.get("grasp_offset_from_holder_mm", 19.05)),
            grasp_offset_from_holder_px=float(data.get("grasp_offset_from_holder_px", 18.0)),
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
            grasp_position_from_table=bool(data.get("grasp_position_from_table", True)),
            grasp_motion=GraspMotionConfig(
                grasp_height_above_table_mm=float(
                    motion_raw.get("grasp_height_above_table_mm", 0.0)
                ),
            ),
            grasp_orientation_tol_deg=float(data.get("grasp_orientation_tol_deg", 50.0)),
            grasp_position_tol_mm=float(data.get("grasp_position_tol_mm", 20.0)),
            ik_ori_vs_pos=float(data.get("ik_ori_vs_pos", 1.0)),
            robot_in_table_frame=robot_in_table,
            work_position_deg=tuple(float(v) for v in work_deg)[:6],  # type: ignore[arg-type]
            execute=ExecuteConfig(
                joint_current_limit_a=float(execute_raw.get("joint_current_limit_a", 0.0)),
                home_confirm_attempts=int(execute_raw.get("home_confirm_attempts", 3)),
            ),
            fixed_grasp_abc_input_deg=(fixed_ab[0], fixed_ab[1], 0.0),
        )
