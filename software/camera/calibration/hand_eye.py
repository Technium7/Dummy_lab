"""
Hand-eye (eye-to-hand) camera calibration — reference / fallback workflow.

Estimates ``T_robot_cam`` from multiple arm poses with a checkerboard rigidly
attached to the gripper. All rotations in ``CameraMountConstraint`` and output
``rpy_deg`` are expressed in the **robot base frame** (PyBullet XYZ).

For a fixed overhead camera viewing the optical table, prefer
``camera.calibration_table`` instead (table-frame ``T_table_cam``).

Solver: joint nonlinear fit over captures satisfying
``T_base_board ≈ T_base_cam @ T_cam_board`` for each pose.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.spatial.transform import Rotation as SciRot

from high_level_jobs.subtasks.grab_sensor.transforms import decompose_matrix

from .chessboard import BoardObservation, rvec_tvec_to_matrix
from .card_config import CalibrationCardConfig
from .fk import rotation_translation_from_matrix


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required for hand-eye calibration. "
            "Install with: pip install -r requirements-vision.txt"
        ) from exc
    return cv2


@dataclass
class CalibrationSample:
    index: int
    joints_deg: list[float]
    board_found: bool
    reprojection_error_px: float
    rvec: list[float] | None
    tvec: list[float] | None
    r_base_target: list[list[float]]
    t_base_target: list[float]
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CameraMountConstraint:
    """
    Optional mount model in the **robot base frame** (not table frame).

    Default: full 6-DOF rotation + translation fit.

    Legacy constrained mode (``fit_full_rotation=False``): fix roll/pitch about
    base X/Y, fit translation + yaw about base Z.
    Rotation order: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """

    fixed_roll_deg: float = -45.0
    fixed_pitch_deg: float = 0.0
    fit_yaw: bool = True
    fit_full_rotation: bool = True


@dataclass
class CalibrationEstimate:
    sample_count: int
    translation_m: np.ndarray
    rpy_deg: np.ndarray
    distance_m: float
    angle_deg: float
    transform_robot_cam: np.ndarray
    translation_spread_m: float
    angle_spread_deg: float
    camera_origin_m: np.ndarray
    board_pose_error_m: float
    fitted_yaw_deg: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "translation_m": self.translation_m.tolist(),
            "rpy_deg": self.rpy_deg.tolist(),
            "distance_m": float(self.distance_m),
            "angle_deg": float(self.angle_deg),
            "translation_spread_m": float(self.translation_spread_m),
            "angle_spread_deg": float(self.angle_spread_deg),
            "camera_origin_m": self.camera_origin_m.tolist(),
            "board_pose_error_m": float(self.board_pose_error_m),
            "fitted_yaw_deg": self.fitted_yaw_deg,
            "transform_robot_cam": self.transform_robot_cam.tolist(),
        }


def _rotation_angle_deg(rot: np.ndarray) -> float:
    trace = float(np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(trace)))


def _homogeneous(rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = np.asarray(rot, dtype=np.float64)
    mat[:3, 3] = np.asarray(trans, dtype=np.float64).reshape(3)
    return mat


def _se3_to_xi(transform: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            SciRot.from_matrix(transform[:3, :3]).as_rotvec(),
            transform[:3, 3],
        ]
    )


def _xi_to_se3(xi: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = SciRot.from_rotvec(xi[:3]).as_matrix()
    transform[:3, 3] = xi[3:6]
    return transform


def _board_poses_from_measurements(
    r_base_target: Sequence[np.ndarray],
    t_base_target: Sequence[np.ndarray],
    r_target_cam: Sequence[np.ndarray],
    t_target_cam: Sequence[np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (T_base_board, T_cam_board) for each capture."""
    poses: list[tuple[np.ndarray, np.ndarray]] = []
    for r_bb, t_bb, r_cb, t_cb in zip(r_base_target, t_base_target, r_target_cam, t_target_cam):
        poses.append(
            (
                _homogeneous(r_bb, t_bb),
                _homogeneous(r_cb, t_cb),
            )
        )
    return poses


def _base_cam_rotation(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll) in the robot base frame."""
    return SciRot.from_euler("XYZ", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_matrix()


def _base_cam_from_params(
    translation_m: Sequence[float],
    *,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> np.ndarray:
    return _homogeneous(_base_cam_rotation(roll_deg, pitch_deg, yaw_deg), translation_m)


def _solve_base_cam_constrained(
    poses: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    roll_deg: float,
    pitch_deg: float,
) -> tuple[np.ndarray, float]:
    """
    Fit camera translation (3) + yaw about base Z (1) with roll/pitch fixed.

    Uses all captures jointly: T_base_board ≈ T_base_cam @ T_cam_board.
    """
    if len(poses) < _MIN_HAND_EYE_SAMPLES:
        raise ValueError("Need at least three valid board poses.")

    per_pose = [t_bb @ np.linalg.inv(t_cb) for t_bb, t_cb in poses]
    t0 = np.mean([t[:3, 3] for t in per_pose], axis=0)
    _, rpy0 = decompose_matrix(per_pose[len(per_pose) // 2])
    yaw0_deg = float(np.degrees(rpy0[2]))
    x0 = np.array([t0[0], t0[1], t0[2], np.radians(yaw0_deg)], dtype=np.float64)

    def residuals(x: np.ndarray) -> np.ndarray:
        trans = x[:3]
        yaw_deg = float(np.degrees(x[3]))
        t_base_cam = _base_cam_from_params(
            trans,
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
        )
        out: list[float] = []
        for t_base_board, t_cam_board in poses:
            t_pred = t_base_cam @ t_cam_board
            delta = np.linalg.inv(t_base_board) @ t_pred
            out.extend(SciRot.from_matrix(delta[:3, :3]).as_rotvec())
            out.extend(delta[:3, 3])
        return np.asarray(out, dtype=np.float64)

    result = least_squares(residuals, x0, method="lm")
    yaw_deg = float(np.degrees(result.x[3]))
    t_base_cam = _base_cam_from_params(
        result.x[:3],
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        yaw_deg=yaw_deg,
    )
    return t_base_cam, yaw_deg


def _yaw_spread_deg(
    poses: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    roll_deg: float,
    pitch_deg: float,
    translation_m: np.ndarray,
) -> float:
    """Per-pose yaw scatter when translation is fixed to the joint solution."""
    yaws: list[float] = []
    trans = np.asarray(translation_m, dtype=np.float64).reshape(3)

    for t_base_board, t_cam_board in poses:

        def cost(yaw_rad: float) -> float:
            t_base_cam = _base_cam_from_params(
                trans,
                roll_deg=roll_deg,
                pitch_deg=pitch_deg,
                yaw_deg=float(np.degrees(yaw_rad)),
            )
            delta = np.linalg.inv(t_base_board) @ (t_base_cam @ t_cam_board)
            rot_err = SciRot.from_matrix(delta[:3, :3]).as_rotvec()
            return float(np.dot(rot_err, rot_err) + np.dot(delta[:3, 3], delta[:3, 3]))

        res = minimize_scalar(cost, bounds=(-np.pi, np.pi), method="bounded")
        yaws.append(float(np.degrees(res.x)))

    if len(yaws) < 2:
        return 0.0
    return float(np.std(yaws))


def _solve_base_cam_from_poses(
    poses: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    mount: CameraMountConstraint,
) -> tuple[np.ndarray, float | None]:
    """
    Joint eye-to-hand fit over all poses.

    Finds a single fixed T_base_cam such that, for every capture:
        T_base_board ≈ T_base_cam @ T_cam_board

    With ``fit_full_rotation=True`` (default), all rotation and translation are free.
    With constrained mount, only translation and base-Z yaw are free.
    """
    if mount.fit_full_rotation:
        if len(poses) < _MIN_HAND_EYE_SAMPLES:
            raise ValueError("Need at least three valid board poses.")

        per_pose = [t_bb @ np.linalg.inv(t_cb) for t_bb, t_cb in poses]
        xi0 = _se3_to_xi(per_pose[len(per_pose) // 2])

        def residuals(xi: np.ndarray) -> np.ndarray:
            t_base_cam = _xi_to_se3(xi)
            out: list[float] = []
            for t_base_board, t_cam_board in poses:
                t_pred = t_base_cam @ t_cam_board
                delta = np.linalg.inv(t_base_board) @ t_pred
                out.extend(SciRot.from_matrix(delta[:3, :3]).as_rotvec())
                out.extend(delta[:3, 3])
            return np.asarray(out, dtype=np.float64)

        result = least_squares(residuals, xi0, method="lm")
        return _xi_to_se3(result.x), None

    return _solve_base_cam_constrained(
        poses,
        roll_deg=mount.fixed_roll_deg,
        pitch_deg=mount.fixed_pitch_deg,
    )


def _board_pose_errors(
    t_base_cam: np.ndarray,
    poses: Sequence[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Board-origin error (m) for each pose against the joint solution."""
    errors: list[float] = []
    for t_base_board, t_cam_board in poses:
        t_pred = t_base_cam @ t_cam_board
        errors.append(float(np.linalg.norm(t_pred[:3, 3] - t_base_board[:3, 3])))
    return np.asarray(errors, dtype=np.float64)


def _spread_around_solution(
    t_base_cam: np.ndarray,
    poses: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float]:
    """Per-pose camera-origin scatter around the joint fit (convergence metric)."""
    per_pose = np.array([((t_bb @ np.linalg.inv(t_cb))[:3, 3]) for t_bb, t_cb in poses])
    origin_spread = float(np.linalg.norm(np.std(per_pose, axis=0)))

    r_ref = t_base_cam[:3, :3]
    angle_deltas = [
        _rotation_angle_deg(r_ref.T @ ((t_bb @ np.linalg.inv(t_cb))[:3, :3]))
        for t_bb, t_cb in poses
    ]
    angle_spread = float(np.std(angle_deltas))
    return origin_spread, angle_spread


_MIN_HAND_EYE_SAMPLES = 3


class HandEyeCalibrator:
    """Eye-to-hand solver: fixed camera observes a checkerboard on the moving gripper."""

    def __init__(
        self,
        out_dir: Path,
        *,
        mount: CameraMountConstraint | None = None,
        card_cfg: CalibrationCardConfig | None = None,
    ) -> None:
        self.out_dir = out_dir.resolve()
        self._mount = mount or CameraMountConstraint()
        self._card_cfg = card_cfg
        self.samples_dir = self.out_dir / "samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self._samples: list[CalibrationSample] = []
        self._valid_r_base_gripper: list[np.ndarray] = []
        self._valid_t_base_gripper: list[np.ndarray] = []
        self._valid_r_target_cam: list[np.ndarray] = []
        self._valid_t_target_cam: list[np.ndarray] = []
        self._history: list[CalibrationEstimate] = []

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    @property
    def valid_count(self) -> int:
        return len(self._valid_r_base_gripper)

    @property
    def history(self) -> list[CalibrationEstimate]:
        return list(self._history)

    def add_sample(
        self,
        *,
        joints_deg: Sequence[float],
        board: BoardObservation,
        t_base_target: np.ndarray,
        color_bgr: np.ndarray,
        stem: str,
    ) -> CalibrationSample:
        idx = len(self._samples) + 1
        r_bt, t_bt = rotation_translation_from_matrix(t_base_target)
        sample = CalibrationSample(
            index=idx,
            joints_deg=[float(v) for v in joints_deg[:6]],
            board_found=board.found,
            reprojection_error_px=float(board.reprojection_error_px),
            rvec=board.rvec.tolist() if board.rvec is not None else None,
            tvec=board.tvec.tolist() if board.tvec is not None else None,
            r_base_target=r_bt.tolist(),
            t_base_target=t_bt.tolist(),
        )
        self._samples.append(sample)

        sample_dir = self.samples_dir / stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        meta_path = sample_dir / f"{stem}.meta.json"
        meta_path.write_text(json.dumps(sample.to_dict(), indent=2), encoding="utf-8")

        try:
            import cv2

            cv2.imwrite(str(sample_dir / f"{stem}.color.png"), color_bgr)
            if board.annotated_bgr is not None:
                cv2.imwrite(str(sample_dir / f"{stem}.annotated.png"), board.annotated_bgr)
        except Exception:
            pass

        if board.found and board.rvec is not None and board.tvec is not None:
            t_target_cam = rvec_tvec_to_matrix(board.rvec, board.tvec)
            r_tc, t_tc = rotation_translation_from_matrix(t_target_cam)
            self._valid_r_base_gripper.append(r_bt)
            self._valid_t_base_gripper.append(t_bt)
            self._valid_r_target_cam.append(r_tc)
            self._valid_t_target_cam.append(t_tc)
            self._append_estimate()

        return sample

    def _append_estimate(self) -> None:
        estimate = self.estimate()
        if estimate is not None:
            self._history.append(estimate)

    def samples_needed(self) -> int:
        return max(0, _MIN_HAND_EYE_SAMPLES - self.valid_count)

    def estimate(self) -> CalibrationEstimate | None:
        n = len(self._valid_r_base_gripper)
        if n < _MIN_HAND_EYE_SAMPLES:
            return None

        poses = _board_poses_from_measurements(
            self._valid_r_base_gripper,
            self._valid_t_base_gripper,
            self._valid_r_target_cam,
            self._valid_t_target_cam,
        )
        try:
            t_robot_cam, fitted_yaw_deg = _solve_base_cam_from_poses(poses, mount=self._mount)
        except ValueError:
            return None

        trans, rpy_rad = decompose_matrix(t_robot_cam)
        rpy_deg = np.degrees(rpy_rad)
        camera_origin = t_robot_cam[:3, 3].copy()
        board_errors = _board_pose_errors(t_robot_cam, poses)
        trans_spread, angle_spread = _spread_around_solution(t_robot_cam, poses)
        if fitted_yaw_deg is not None:
            angle_spread = _yaw_spread_deg(
                poses,
                roll_deg=self._mount.fixed_roll_deg,
                pitch_deg=self._mount.fixed_pitch_deg,
                translation_m=camera_origin,
            )

        return CalibrationEstimate(
            sample_count=n,
            translation_m=trans,
            rpy_deg=rpy_deg,
            distance_m=float(np.linalg.norm(camera_origin)),
            angle_deg=_rotation_angle_deg(t_robot_cam[:3, :3]),
            transform_robot_cam=t_robot_cam,
            translation_spread_m=trans_spread,
            angle_spread_deg=angle_spread,
            camera_origin_m=camera_origin,
            board_pose_error_m=float(np.mean(board_errors)),
            fitted_yaw_deg=fitted_yaw_deg,
        )

    def save_result(self, path: Path | None = None) -> Path | None:
        estimate = self.estimate()
        if estimate is None:
            return None
        out = path or (self.out_dir / "camera_to_robot.json")
        method = "eye_to_hand_joint_se3" if self._mount.fit_full_rotation else "eye_to_hand_fixed_tilt_fit_yaw"
        description = (
            "Joint nonlinear fit over all captures: T_base_board ≈ T_base_cam @ T_cam_board."
        )
        if self._mount.fit_full_rotation:
            description += " Full 6-DOF rotation and translation jointly fit."
        else:
            description += (
                f" Rotation fixed to roll={self._mount.fixed_roll_deg:g}°, "
                f"pitch={self._mount.fixed_pitch_deg:g}° about base X/Y; "
                f"only translation + yaw (base Z) were fit."
            )
        card_block: dict[str, Any] | None = None
        if self._card_cfg is not None:
            cfg = self._card_cfg
            card_block = {
                "squares_x": cfg.pattern_size[0] + 1,
                "squares_y": cfg.pattern_size[1] + 1,
                "square_size_m": cfg.square_size_m,
                "board_mount": cfg.board_mount,
                "grasp_point": cfg.grasp_point,
                "end_link_to_tcp_xyz_m": list(cfg.end_link_to_tcp_xyz_m),
                "end_link_to_tcp_distance_m": float(np.linalg.norm(cfg.end_link_to_tcp_xyz_m)),
                "end_link_to_tcp_rpy_deg": list(cfg.end_link_to_tcp_rpy_deg),
                "gripper_to_board_xyz_m": list(cfg.gripper_to_board_xyz_m),
                "gripper_to_board_rpy_deg": list(cfg.gripper_to_board_rpy_deg),
            }
        payload = {
            "method": method,
            "description": description,
            "camera_mount_constraint": {
                "fixed_roll_deg": self._mount.fixed_roll_deg,
                "fixed_pitch_deg": self._mount.fixed_pitch_deg,
                "fit_yaw": self._mount.fit_yaw,
                "fit_full_rotation": self._mount.fit_full_rotation,
            },
            "sample_count": estimate.sample_count,
            "camera_to_robot": {
                "translation_m": estimate.translation_m.tolist(),
                "rpy_deg": estimate.rpy_deg.tolist(),
            },
            "camera_origin_in_base_m": estimate.camera_origin_m.tolist(),
            "fitted_yaw_deg": estimate.fitted_yaw_deg,
            "transform_robot_cam": estimate.transform_robot_cam.tolist(),
            "metrics": {
                "distance_m": estimate.distance_m,
                "angle_deg": estimate.angle_deg,
                "translation_spread_m": estimate.translation_spread_m,
                "angle_spread_deg": estimate.angle_spread_deg,
                "board_pose_error_m": estimate.board_pose_error_m,
            },
            "samples": [s.to_dict() for s in self._samples],
        }
        if card_block is not None:
            payload["card_config"] = card_block
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out
