from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .grid_config import TableGridConfig
from .grid_detect import GridObservation, detect_table_grid, solve_pose_from_holes
from .grid_extend import draw_extended_grid
from .grid_types import DetectedHole, GridSlot
from .table_calib import TableCalibrationResult, build_calibration_result


@dataclass
class MultiShotMetrics:
    shot_count: int
    min_votes: int
    shots_with_pose: int
    fused_hole_count: int
    per_shot_hole_counts: list[int]
    per_shot_reproj_px: list[float | None]
    translation_spread_m: float = 0.0
    angle_spread_deg: float = 0.0
    camera_translation_std_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    camera_rpy_std_deg: np.ndarray = field(default_factory=lambda: np.zeros(3))
    fusion_method: str = "fused_holes"

    def to_dict(self) -> dict[str, Any]:
        return {
            "shot_count": self.shot_count,
            "min_votes": self.min_votes,
            "shots_with_pose": self.shots_with_pose,
            "fused_hole_count": self.fused_hole_count,
            "per_shot_hole_counts": self.per_shot_hole_counts,
            "per_shot_reproj_px": self.per_shot_reproj_px,
            "translation_spread_m": float(self.translation_spread_m),
            "angle_spread_deg": float(self.angle_spread_deg),
            "camera_translation_std_m": self.camera_translation_std_m.tolist(),
            "camera_rpy_std_deg": self.camera_rpy_std_deg.tolist(),
            "fusion_method": self.fusion_method,
        }


def _rotation_angle_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    rel = rot_a.T @ rot_b
    trace = float(np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(trace)))


def fuse_holes_from_observations(
    observations: list[GridObservation],
    *,
    min_votes: int,
    origin_lock_px: tuple[float, float] | None = None,
    origin_radius_px: float = 10.0,
) -> tuple[list[DetectedHole], float, DetectedHole | None]:
    """Average hole centers per grid cell across shots that saw the cell."""
    cell_samples: dict[tuple[int, int], list[DetectedHole]] = {}
    pitches: list[float] = []

    for obs in observations:
        if np.isfinite(obs.pitch_px):
            pitches.append(float(obs.pitch_px))
        for hole in obs.holes:
            if hole.status == "rejected":
                continue
            key = (hole.grid_col, hole.grid_row)
            if origin_lock_px is not None and key == (0, 0):
                continue
            cell_samples.setdefault(key, []).append(hole)

    pitch_px = float(np.median(pitches)) if pitches else float("nan")
    fused: list[DetectedHole] = []
    for (col, row), samples in cell_samples.items():
        if len(samples) < min_votes:
            continue
        cx = float(np.mean([h.center_px[0] for h in samples]))
        cy = float(np.mean([h.center_px[1] for h in samples]))
        statuses = [h.status for h in samples]
        status = "detected" if statuses.count("detected") >= statuses.count("recovered") else "recovered"
        fused.append(
            DetectedHole(
                center_px=(cx, cy),
                radius_px=float(np.mean([h.radius_px for h in samples])),
                circularity=float(np.mean([h.circularity for h in samples])),
                nn_score=float(np.mean([h.nn_score for h in samples])),
                grid_col=col,
                grid_row=row,
                source=f"fused_{len(samples)}",
                status=status,
            )
        )

    if origin_lock_px is not None:
        origin = DetectedHole(
            center_px=origin_lock_px,
            radius_px=origin_radius_px,
            circularity=1.0,
            nn_score=1.0,
            grid_col=0,
            grid_row=0,
            is_origin=True,
            source="origin_lock",
            status="locked",
        )
        fused.append(origin)
        return fused, pitch_px, origin

    origin = None
    for hole in fused:
        if hole.grid_col == 0 and hole.grid_row == 0:
            hole.is_origin = True
            origin = hole
            break
    if origin is None and fused:
        origin = min(fused, key=lambda h: h.grid_col * h.grid_col + h.grid_row * h.grid_row)
        origin.is_origin = True
    return fused, pitch_px, origin


def _pose_spread_from_results(results: list[TableCalibrationResult]) -> tuple[float, float, np.ndarray, np.ndarray]:
    if len(results) < 2:
        z = np.zeros(3)
        return 0.0, 0.0, z, z
    translations = np.array([r.camera_in_table_translation_m for r in results])
    rpys = np.array([r.camera_in_table_rpy_deg for r in results])
    t_std = translations.std(axis=0)
    rpy_std = rpys.std(axis=0)
    t_spread = float(np.linalg.norm(translations - translations.mean(axis=0), axis=1).max())
    rot_a = results[0].transform_table_cam[:3, :3]
    angles = [_rotation_angle_deg(rot_a, r.transform_table_cam[:3, :3]) for r in results[1:]]
    angle_spread = float(max(angles)) if angles else 0.0
    return t_spread, angle_spread, t_std, rpy_std


def _best_shot_observation(
    observations: list[GridObservation],
    per_shot_reproj: list[float | None],
) -> GridObservation | None:
    best: GridObservation | None = None
    best_err = float("inf")
    for i, obs in enumerate(observations):
        if not obs.found or obs.rvec is None or obs.tvec is None:
            continue
        err = obs.reprojection_error_px
        if i < len(per_shot_reproj) and per_shot_reproj[i] is not None:
            err = float(per_shot_reproj[i])
        if err < best_err:
            best_err = err
            best = obs
    return best


def _observation_from_shot(
    obs: GridObservation,
    color_bgr: np.ndarray,
    *,
    n_shots: int,
    min_votes: int,
    fusion_method: str,
    t_spread: float,
    angle_spread: float,
) -> GridObservation:
    annotated = color_bgr.copy()
    slots = [GridSlot(h.grid_col, h.grid_row, h.center_px, h.status, h) for h in obs.holes]
    draw_extended_grid(annotated, slots, obs.origin_hole)
    msg = (
        f"{fusion_method}: {len(obs.holes)} holes, reproj {obs.reprojection_error_px:.2f}px, "
        f"spread {t_spread * 1000:.1f}mm / {angle_spread:.2f}deg"
    )
    return GridObservation(
        found=True,
        holes=list(obs.holes),
        rvec=obs.rvec.copy() if obs.rvec is not None else None,
        tvec=obs.tvec.copy() if obs.tvec is not None else None,
        reprojection_error_px=float(obs.reprojection_error_px),
        pitch_px=float(obs.pitch_px),
        origin_hole=obs.origin_hole,
        origin_model_cell=obs.origin_model_cell,
        origin_lock_px=obs.origin_lock_px,
        annotated_bgr=annotated,
        message=msg,
        grid_slots=slots,
        holes_found=obs.holes_found,
        holes_recovered=obs.holes_recovered,
        holes_missing=obs.holes_missing,
        candidates_before_filter=obs.candidates_before_filter,
    )


def calibrate_from_multi_shot(
    color_frames_bgr: list[np.ndarray],
    cfg: TableGridConfig,
    intrinsics: Any,
    *,
    origin_click_px: tuple[float, float] | None = None,
    origin_grid_cell: tuple[int, int] | None = None,
    origin_lock_px: tuple[float, float] | None = None,
    min_votes: int | None = None,
) -> tuple[GridObservation | None, MultiShotMetrics | None]:
    if not color_frames_bgr:
        return None, None

    n_shots = len(color_frames_bgr)
    min_votes = min_votes or max(1, (n_shots + 1) // 2)

    observations: list[GridObservation] = []
    for i, color_bgr in enumerate(color_frames_bgr):
        observations.append(
            detect_table_grid(
                color_bgr,
                cfg,
                intrinsics,
                origin_click_px=origin_click_px if origin_lock_px is None and i == 0 else None,
                origin_grid_cell=origin_grid_cell,
                origin_lock_px=origin_lock_px,
            )
        )

    origin_radius_px = cfg.expected_radius_px
    if observations and observations[0].origin_hole is not None:
        origin_radius_px = observations[0].origin_hole.radius_px

    fused_holes, pitch_px, origin = fuse_holes_from_observations(
        observations,
        min_votes=min_votes,
        origin_lock_px=origin_lock_px,
        origin_radius_px=origin_radius_px,
    )

    per_shot_results: list[TableCalibrationResult] = []
    per_shot_reproj: list[float | None] = []
    for obs in observations:
        if obs.found:
            try:
                per_shot_results.append(build_calibration_result(obs, cfg))
                per_shot_reproj.append(float(obs.reprojection_error_px))
            except ValueError:
                per_shot_reproj.append(None)
        else:
            per_shot_reproj.append(None)

    t_spread, angle_spread, t_std, rpy_std = _pose_spread_from_results(per_shot_results)

    base_metrics = dict(
        shot_count=n_shots,
        min_votes=min_votes,
        shots_with_pose=len(per_shot_results),
        fused_hole_count=len(fused_holes),
        per_shot_hole_counts=[len(o.holes) for o in observations],
        per_shot_reproj_px=per_shot_reproj,
    )

    fusion_method = "fused_holes"
    final_obs: GridObservation | None = None

    if len(fused_holes) >= cfg.min_holes_for_pose:
        rvec, tvec, err = solve_pose_from_holes(fused_holes, cfg, intrinsics, pitch_px)
        valid_shot_errs = [e for e in per_shot_reproj if e is not None and np.isfinite(e)]
        median_shot_err = float(np.median(valid_shot_errs)) if valid_shot_errs else float("inf")
        if (
            rvec is not None
            and valid_shot_errs
            and err > max(2.0 * median_shot_err, median_shot_err + 3.0)
        ):
            best = _best_shot_observation(observations, per_shot_reproj)
            if best is not None:
                fusion_method = "best_shot"
                final_obs = _observation_from_shot(
                    best,
                    color_frames_bgr[-1],
                    n_shots=n_shots,
                    min_votes=min_votes,
                    fusion_method=fusion_method,
                    t_spread=t_spread,
                    angle_spread=angle_spread,
                )
        elif rvec is not None and tvec is not None:
            annotated = color_frames_bgr[-1].copy()
            slots = [GridSlot(h.grid_col, h.grid_row, h.center_px, h.status, h) for h in fused_holes]
            draw_extended_grid(annotated, slots, origin)
            final_obs = GridObservation(
                found=True,
                holes=fused_holes,
                rvec=rvec.reshape(3),
                tvec=tvec.reshape(3),
                reprojection_error_px=err,
                pitch_px=pitch_px,
                origin_hole=origin,
                origin_model_cell=origin_grid_cell or observations[0].origin_model_cell,
                origin_lock_px=origin_lock_px or observations[0].origin_lock_px,
                annotated_bgr=annotated,
                message=(
                    f"fused {len(fused_holes)} holes from {n_shots} shots (>={min_votes} votes), "
                    f"reproj {err:.2f}px, spread {t_spread * 1000:.1f}mm / {angle_spread:.2f}deg"
                ),
                grid_slots=slots,
                holes_found=sum(1 for h in fused_holes if h.status == "detected"),
                holes_recovered=sum(1 for h in fused_holes if h.status == "recovered"),
                holes_missing=0,
                candidates_before_filter=sum(o.candidates_before_filter for o in observations),
            )

    if final_obs is None:
        best = _best_shot_observation(observations, per_shot_reproj)
        if best is not None:
            fusion_method = "best_shot_only"
            final_obs = _observation_from_shot(
                best,
                color_frames_bgr[-1],
                n_shots=n_shots,
                min_votes=min_votes,
                fusion_method=fusion_method,
                t_spread=t_spread,
                angle_spread=angle_spread,
            )
            base_metrics["fused_hole_count"] = len(best.holes)

    if final_obs is None:
        return None, MultiShotMetrics(**base_metrics)

    metrics = MultiShotMetrics(
        **base_metrics,
        translation_spread_m=t_spread,
        angle_spread_deg=angle_spread,
        camera_translation_std_m=t_std,
        camera_rpy_std_deg=rpy_std,
        fusion_method=fusion_method,
    )
    return final_obs, metrics
