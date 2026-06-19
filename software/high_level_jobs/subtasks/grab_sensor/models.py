from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Tuple

import numpy as np

_SOFTWARE_DIR = Path(__file__).resolve().parents[3]
if str(_SOFTWARE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_DIR))

from robot_config import DEFAULT_ROBOT_NAME  # noqa: E402

GraspOrientationMode = Literal["follow_object", "fixed_robot"]
VisionMode = Literal["depth_segment", "yolo_grasp"]


@dataclass(frozen=True)
class GraspOnObjectFrame:
    """
    Gripper target relative to the vision-estimated TABLE OBJECT pose.

    This is not the camera. The camera only locates the object; this block
    encodes how the gripper should relate to that object (edge, tilt, offset).
    """

    position_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    approach_offset_mm: float = 0.0

    def position_m(self) -> np.ndarray:
        return np.asarray(self.position_mm, dtype=np.float64) * 1e-3


@dataclass(frozen=True)
class ApproachConfig:
    pre_grasp_offset_mm: float = 80.0
    pre_grasp_along_xy: bool = True
    plan_pre_grasp: bool = False
    pre_grasp_orientation_tol_deg: float = 90.0
    post_grasp_lift_mm: float = 100.0
    plan_return_home: bool = True
    retreat_after_close_mm: float = 40.0


@dataclass(frozen=True)
class GraspMotionConfig:
    """Grasp height on the sensor (TCP is ``grasp_tcp_link`` in the grasp URDF)."""

    grasp_height_above_table_mm: float = 65.0


@dataclass(frozen=True)
class SegmentationConfig:
    min_depth_m: float = 0.25
    max_depth_m: float = 0.55
    min_area_px: int = 400
    morph_kernel_px: int = 5


@dataclass(frozen=True)
class YoloConfig:
    model_path: Path
    class_name: str = "sensor"
    confidence: float = 0.35
    iou: float = 0.45
    crop_padding_px: int = 16


@dataclass(frozen=True)
class GraspNetConfig:
    model_path: Path
    base_model_path: Path | None = None
    depth_patch_radius_px: int = 3
    predict_on: Literal["full_frame", "crop"] = "crop"
    bbox_margin_px: int = 12
    depth_min_m: float = 0.25
    depth_max_m: float = 0.85


@dataclass(frozen=True)
class CameraToRobot:
    """
    Fixed D415 mount: camera optical frame -> robot base (planner world) frame.

    Calibrate once. The table object is NOT configured here — vision finds it.
    """

    translation_m: Tuple[float, float, float]
    rpy_deg: Tuple[float, float, float]


@dataclass(frozen=True)
class RobotInTableFrame:
    """Robot base origin pose expressed in the optical table frame."""

    translation_in: Tuple[float, float, float] = (-6.0, -5.0, 0.5)
    rpy_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def translation_m(self) -> Tuple[float, float, float]:
        inch = 0.0254
        return (
            float(self.translation_in[0]) * inch,
            float(self.translation_in[1]) * inch,
            float(self.translation_in[2]) * inch,
        )


@dataclass(frozen=True)
class GrabSensorConfig:
    camera_to_robot: CameraToRobot
    grasp_on_object: GraspOnObjectFrame
    approach: ApproachConfig
    segmentation: SegmentationConfig
    vision_mode: VisionMode = "depth_segment"
    yolo: YoloConfig | None = None
    grasp_net: GraspNetConfig | None = None
    grasp_orientation_mode: GraspOrientationMode = "follow_object"
    fixed_grasp_abc_input_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    grasp_rpy_roll_offset_deg: float = 15.0
    grasp_rpy_yaw_base_deg: float = 180.0
    grasp_c_from_xy: bool = True
    load_environment: bool = False
    env_path: Path = field(default_factory=lambda: Path("physical_env/packaging_station_edit.obj"))
    mesh_scale: float = 1.0
    robot_name: str = DEFAULT_ROBOT_NAME
    end_link: str = "link6_1"
    tcp_link: str = "grasp_tcp_link"
    enable_collision: bool = False
    plan_seed: int = 42
    save_debug_dir: Path | None = None
    table_calib_path: Path | None = None
    grasp_position_from_table_base: bool = False
    grasp_motion: GraspMotionConfig = field(default_factory=GraspMotionConfig)
    grasp_orientation_tol_deg: float = 45.0
    grasp_position_tol_mm: float = 80.0
    robot_in_table_frame: RobotInTableFrame = field(default_factory=RobotInTableFrame)

    @property
    def grasp_height_above_table_mm(self) -> float:
        """Alias for ``grasp_motion.grasp_height_above_table_mm``."""
        return self.grasp_motion.grasp_height_above_table_mm

    @staticmethod
    def from_mapping(data: Mapping[str, Any], base_dir: Path | None = None) -> GrabSensorConfig:
        base = base_dir or Path.cwd()

        grasp_raw = data.get("grasp_on_object") or data.get("grasp_in_sensor_frame") or {}
        approach_raw = data.get("approach", {})
        motion_raw = data.get("motion_planning", {})
        grasp_motion = GraspMotionConfig(
            grasp_height_above_table_mm=float(
                motion_raw.get(
                    "grasp_height_above_table_mm",
                    data.get("grasp_height_above_table_mm", 65.0),
                )
            ),
        )
        seg_raw = data.get("segmentation", {})
        yolo_raw = data.get("yolo", {})
        grasp_net_raw = data.get("grasp_net", {})
        cam_raw = data.get("camera_to_robot") or data.get("camera_extrinsics") or {}
        paths_raw = data.get("paths", {})

        package_dir = Path(__file__).resolve().parent
        default_model_dir = package_dir / "detection" / "dataset" / "grasp_keypoints" / "model"

        def _resolve_model_path(raw_path: str | None, default_name: str) -> Path:
            if raw_path:
                p = Path(str(raw_path))
            else:
                p = default_model_dir / default_name
            if not p.is_absolute():
                p = (base / p).resolve()
            return p

        vision_raw = str(data.get("vision_mode", "depth_segment")).strip().lower()
        if vision_raw not in ("depth_segment", "yolo_grasp"):
            raise ValueError(
                f"vision_mode must be 'depth_segment' or 'yolo_grasp', got {vision_raw!r}"
            )

        env_rel = str(paths_raw.get("env", "physical_env/packaging_station_edit.obj"))
        env_path = Path(env_rel)
        if not env_path.is_absolute():
            env_path = (base / env_rel).resolve()

        debug = paths_raw.get("save_debug_dir")
        debug_path = None
        if debug:
            debug_path = Path(str(debug))
            if not debug_path.is_absolute():
                debug_path = (base / debug_path).resolve()

        table_calib_raw = data.get("table_calib_path") or paths_raw.get("table_calib")
        table_calib_path = None
        if table_calib_raw:
            table_calib_path = Path(str(table_calib_raw))
            if not table_calib_path.is_absolute():
                table_calib_path = (base / table_calib_path).resolve()

        grasp_pos_mm = tuple(float(v) for v in grasp_raw.get("position_mm", [0.0, 0.0, 0.0])[:3])
        grasp_rpy_deg = tuple(float(v) for v in grasp_raw.get("rpy_deg", [0.0, 0.0, 0.0])[:3])
        mode_raw = str(data.get("grasp_orientation_mode", "follow_object")).strip().lower()
        if mode_raw not in ("follow_object", "fixed_robot"):
            raise ValueError(
                f"grasp_orientation_mode must be 'follow_object' or 'fixed_robot', got {mode_raw!r}"
            )

        abc_raw = data.get("fixed_grasp_abc_input_deg")
        if abc_raw is None:
            abc_raw = data.get(
                "fixed_grasp_abc_semantic_deg",
                data.get("fixed_grasp_abc_deg", [0.0, 0.0, 0.0]),
            )
        abc_input = tuple(float(v) for v in abc_raw[:3])
        roll_offset = float(data.get("grasp_rpy_roll_offset_deg", 15.0))
        yaw_base = float(data.get("grasp_rpy_yaw_base_deg", 180.0))
        grasp_c_from_xy = bool(data.get("grasp_c_from_xy", True))
        load_environment = bool(data.get("load_environment", False))

        robot_table_raw = data.get("robot_in_table_frame", {})
        trans_in = tuple(
            float(v) for v in robot_table_raw.get("translation_in", [-6.0, -5.0, 0.5])[:3]
        )
        robot_rpy = tuple(float(v) for v in robot_table_raw.get("rpy_deg", [0.0, 0.0, 0.0])[:3])

        predict_on_raw = str(grasp_net_raw.get("predict_on", "full_frame")).strip().lower()
        if predict_on_raw not in ("full_frame", "crop"):
            raise ValueError(
                f"grasp_net.predict_on must be 'full_frame' or 'crop', got {predict_on_raw!r}"
            )

        return GrabSensorConfig(
            camera_to_robot=CameraToRobot(
                translation_m=tuple(float(v) for v in cam_raw.get("translation_m", [0.0, 0.0, 0.0])[:3]),  # type: ignore
                rpy_deg=tuple(float(v) for v in cam_raw.get("rpy_deg", [0.0, 0.0, 0.0])[:3]),  # type: ignore
            ),
            grasp_on_object=GraspOnObjectFrame(
                position_mm=grasp_pos_mm,  # type: ignore[arg-type]
                rpy_deg=grasp_rpy_deg,  # type: ignore[arg-type]
                approach_offset_mm=float(grasp_raw.get("approach_offset_mm", 0.0)),
            ),
            approach=ApproachConfig(
                pre_grasp_offset_mm=float(approach_raw.get("pre_grasp_offset_mm", 80.0)),
                pre_grasp_along_xy=bool(approach_raw.get("pre_grasp_along_xy", True)),
                plan_pre_grasp=bool(approach_raw.get("plan_pre_grasp", False)),
                pre_grasp_orientation_tol_deg=float(
                    approach_raw.get("pre_grasp_orientation_tol_deg", 90.0)
                ),
                post_grasp_lift_mm=float(approach_raw.get("post_grasp_lift_mm", 100.0)),
                plan_return_home=bool(approach_raw.get("plan_return_home", True)),
                retreat_after_close_mm=float(approach_raw.get("retreat_after_close_mm", 40.0)),
            ),
            segmentation=SegmentationConfig(
                min_depth_m=float(seg_raw.get("min_depth_m", 0.25)),
                max_depth_m=float(seg_raw.get("max_depth_m", 0.55)),
                min_area_px=int(seg_raw.get("min_area_px", 400)),
                morph_kernel_px=int(seg_raw.get("morph_kernel_px", 5)),
            ),
            vision_mode=vision_raw,  # type: ignore[arg-type]
            yolo=YoloConfig(
                model_path=_resolve_model_path(yolo_raw.get("model_path"), "sensor_yolo.pt"),
                class_name=str(yolo_raw.get("class_name", "sensor")),
                confidence=float(yolo_raw.get("confidence", 0.35)),
                iou=float(yolo_raw.get("iou", 0.45)),
                crop_padding_px=int(yolo_raw.get("crop_padding_px", 16)),
            ),
            grasp_net=GraspNetConfig(
                model_path=_resolve_model_path(grasp_net_raw.get("model_path"), "grasp_point.pt"),
                base_model_path=_resolve_model_path(
                    grasp_net_raw.get("base_model_path"),
                    "base_point.pt",
                ),
                depth_patch_radius_px=int(grasp_net_raw.get("depth_patch_radius_px", 3)),
                predict_on=predict_on_raw,  # type: ignore[arg-type]
                bbox_margin_px=int(grasp_net_raw.get("bbox_margin_px", 12)),
                depth_min_m=float(grasp_net_raw.get("depth_min_m", 0.25)),
                depth_max_m=float(grasp_net_raw.get("depth_max_m", 0.85)),
            ),
            grasp_orientation_mode=mode_raw,  # type: ignore[arg-type]
            fixed_grasp_abc_input_deg=abc_input,  # type: ignore[arg-type]
            grasp_rpy_roll_offset_deg=roll_offset,
            grasp_rpy_yaw_base_deg=yaw_base,
            grasp_c_from_xy=grasp_c_from_xy,
            load_environment=load_environment,
            env_path=env_path,
            mesh_scale=float(paths_raw.get("mesh_scale", 1.0)),
            robot_name=str(paths_raw.get("robot", DEFAULT_ROBOT_NAME)),
            end_link=str(paths_raw.get("end_link", "link6_1")),
            tcp_link=str(paths_raw.get("tcp_link", "grasp_tcp_link")),
            enable_collision=bool(data.get("enable_collision", False)),
            plan_seed=int(data.get("plan_seed", 42)),
            save_debug_dir=debug_path,
            table_calib_path=table_calib_path,
            grasp_position_from_table_base=bool(data.get("grasp_position_from_table_base", False)),
            grasp_motion=grasp_motion,
            grasp_orientation_tol_deg=float(data.get("grasp_orientation_tol_deg", 45.0)),
            grasp_position_tol_mm=float(data.get("grasp_position_tol_mm", 80.0)),
            robot_in_table_frame=RobotInTableFrame(
                translation_in=trans_in,  # type: ignore[arg-type]
                rpy_deg=robot_rpy,  # type: ignore[arg-type]
            ),
        )


@dataclass
class TableObjectDetection:
    """Vision estimate of the pick target (table object) in the camera frame."""

    centroid_cam_m: np.ndarray
    yaw_rad: float
    object_rpy_cam_rad: np.ndarray
    mask: np.ndarray
    n_points: int
    bbox_xy: Tuple[int, int, int, int]


@dataclass
class GraspTargets:
    """Planner targets in robot base / PyBullet world frame (``grasp_pos_m`` = TCP tip)."""

    pre_grasp_pos_m: np.ndarray
    pre_grasp_rpy_rad: np.ndarray
    grasp_pos_m: np.ndarray
    grasp_rpy_rad: np.ndarray
    object_pos_m: np.ndarray
    object_rpy_rad: np.ndarray
    debug: dict[str, Any] = field(default_factory=dict)
