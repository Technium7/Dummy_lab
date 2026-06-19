#!/usr/bin/env python3
"""Detect optical table hole grid and calibrate camera pose relative to the central hole."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_TABLE_CALIB_DIR = Path(__file__).resolve().parent
_SOFTWARE_DIR = _TABLE_CALIB_DIR.parents[1]
if str(_SOFTWARE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_DIR))

from camera.calibration_table.grid_config import TableGridConfig
from camera.calibration_table.grid_detect import (
    apply_region_polygon,
    detect_table_grid_grow,
    detect_table_grid_preview,
    set_calibration_origin,
)
from camera.calibration_table.grid_grow import (
    assign_seed_click,
    draw_seed_picking_overlay,
    prepare_origin_pool,
)
from camera.calibration_table.hole_detect import find_hole_candidates
from camera.calibration_table.region_select import draw_final_origin_prompt, draw_polygon_draft
from camera.calibration_table.table_calib import build_calibration_result, save_calibration_result


def _real_sense_streamer():
    from high_level_jobs.subtasks.grab_sensor.detection.capture import RealSenseStreamer

    return RealSenseStreamer


def _default_run_dir(base: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    return base / stamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect optical table hole grid and calibrate camera pose relative to the central hole."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "runs",
        help="Base directory for run outputs (default: calibration_table/runs)",
    )
    parser.add_argument("--image", type=Path, default=None, help="Process a saved color image instead of live camera")
    parser.add_argument("--meta", type=Path, default=None, help="Meta JSON with color intrinsics (for --image mode)")
    parser.add_argument("--camera-serial", type=str, default="", help="RealSense serial (default: first device)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--pitch-in", type=float, default=1.0, help="Hole spacing in inches (default: 1.0)")
    parser.add_argument(
        "--hole-diameter-in",
        type=float,
        default=0.25,
        help="Expected hole diameter in inches (default: 0.25)",
    )
    parser.add_argument(
        "--min-holes",
        type=int,
        default=6,
        help="Minimum detected holes required for pose solve (default: 6)",
    )
    parser.add_argument(
        "--expected-radius-px",
        type=float,
        default=5.0,
        help="Typical hole radius in px (search-window / Hough scale hint only, not a hard filter)",
    )
    parser.add_argument(
        "--max-radius-px",
        type=float,
        default=None,
        help="Optional hard cap on hole radius (default: off — size varies with distance)",
    )
    parser.add_argument(
        "--min-circularity",
        type=float,
        default=0.65,
        help="Minimum contour circularity for hole candidates (default: 0.65)",
    )
    parser.add_argument(
        "--edge-margin-px",
        type=float,
        default=15.0,
        help="Ignore circular detections within this many px of the image border",
    )
    parser.add_argument(
        "--num-shots",
        type=int,
        default=5,
        help="Number of frames to capture when saving calibration (default: 5)",
    )
    parser.add_argument(
        "--min-votes",
        type=int,
        default=0,
        help="Min shots that must see a hole to include it (default: majority of num-shots)",
    )
    parser.add_argument(
        "--detect-interval-ms",
        type=int,
        default=400,
        help="Detection refresh interval in live mode (default: 400)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=15,
        help="Frames to discard while auto-exposure settles before locking (default: 15)",
    )
    parser.add_argument(
        "--no-lock-exposure",
        action="store_true",
        help="Leave RealSense auto exposure on (default: lock after warmup)",
    )
    parser.add_argument(
        "--exposure-us",
        type=float,
        default=None,
        help="Fixed color exposure in microseconds (default: snapshot value after warmup)",
    )
    parser.add_argument("--no-gui", action="store_true", help="Single-shot detect/save without preview window")
    return parser.parse_args()


class _SimpleIntrinsics:
    def __init__(self, fx: float, fy: float, ppx: float, ppy: float, coeffs: list[float] | None = None) -> None:
        self.fx = fx
        self.fy = fy
        self.ppx = ppx
        self.ppy = ppy
        self.coeffs = coeffs or []


def _intrinsics_from_meta(meta_path: Path) -> _SimpleIntrinsics:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    color = meta["intrinsics"]["color"]
    return _SimpleIntrinsics(
        fx=float(color["fx"]),
        fy=float(color["fy"]),
        ppx=float(color["ppx"]),
        ppy=float(color["ppy"]),
        coeffs=list(color.get("coeffs", [])),
    )


def _load_image_bgr(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def _print_result_summary(result_path: Path) -> None:
    data = json.loads(result_path.read_text(encoding="utf-8"))
    cam = data["camera_in_table_frame"]
    t = cam["translation_m"]
    rpy = cam["rpy_deg"]
    err = data["metrics"]["reprojection_error_px"]
    print(f"Saved {result_path}")
    print(
        f"  reproj {err:.2f} px  "
        f"cam in table frame: pos [{t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f}] m, "
        f"rpy [{rpy[0]:+.2f}, {rpy[1]:+.2f}, {rpy[2]:+.2f}] deg"
    )


def _save_failed_capture(
    run_dir: Path,
    color_bgr: np.ndarray,
    observation,
) -> None:
    import cv2

    run_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(run_dir / "failed.color.png"), color_bgr)
    if observation.annotated_bgr is not None:
        cv2.imwrite(str(run_dir / "failed.overlay.png"), observation.annotated_bgr)


def run_image_mode(args: argparse.Namespace, grid_cfg: TableGridConfig) -> int:
    if args.meta is None:
        raise SystemExit("--meta is required with --image (path to *.meta.json with intrinsics)")
    color_bgr = _load_image_bgr(args.image)
    intrinsics = _intrinsics_from_meta(args.meta)
    h, w = color_bgr.shape[:2]
    observation = detect_table_grid_grow(color_bgr, grid_cfg, intrinsics, (w * 0.5, h * 0.5))
    run_dir = _default_run_dir(args.out_dir.resolve())
    if not observation.found:
        print(f"Detection failed: {observation.message}", file=sys.stderr)
        _save_failed_capture(run_dir, color_bgr, observation)
        print(f"Saved debug images to {run_dir}")
        return 1

    result = build_calibration_result(observation, grid_cfg)
    result_path = save_calibration_result(
        result,
        run_dir,
        color_bgr=color_bgr,
        annotated_bgr=observation.annotated_bgr,
    )
    _print_result_summary(result_path)
    return 0


def _capture_shot_frames(streamer, n: int, pause_s: float = 0.06) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for _ in range(n):
        frame = streamer.read()
        frames.append(frame.color_rgb[:, :, ::-1].copy())
        if pause_s > 0:
            time.sleep(pause_s)
    return frames


def _save_shot_frames(run_dir: Path, frames: list[np.ndarray]) -> None:
    import cv2

    shots_dir = run_dir / "shots"
    shots_dir.mkdir(parents=True, exist_ok=True)
    for i, color_bgr in enumerate(frames, start=1):
        cv2.imwrite(str(shots_dir / f"shot_{i:02d}.png"), color_bgr)


def run_live_mode(args: argparse.Namespace, grid_cfg: TableGridConfig) -> int:
    import cv2

    RealSenseStreamer = _real_sense_streamer()
    run_dir = _default_run_dir(args.out_dir.resolve())
    streamer = RealSenseStreamer(
        width=args.width,
        height=args.height,
        fps=args.fps,
        serial=args.camera_serial.strip(),
        warmup=args.warmup,
        lock_exposure=not args.no_lock_exposure,
        exposure_us=args.exposure_us,
    )
    try:
        streamer.start()
    except RuntimeError as exc:
        print(f"Failed to start camera: {exc}", file=sys.stderr)
        return 1

    camera_settings: dict | None = None
    if streamer.color_exposure_locked:
        camera_settings = {
            "lock_exposure": True,
            **streamer.color_exposure_locked,
        }
        exp = streamer.color_exposure_locked.get("exposure_us")
        print(f"[Table calib] Locked color exposure: {exp:.0f} us")
    elif not args.no_lock_exposure:
        print("[Table calib] Warning: could not lock color exposure on this device")

    click_px: tuple[float, float] | None = None
    locked_origin_px: tuple[float, float] | None = None
    frozen_bgr: np.ndarray | None = None
    frozen_intrinsics: Any = None
    pending_origin_click = False
    picking_seeds = False
    manual_seed_clicks: list[tuple[float, float]] = []
    seed_assigned: dict[tuple[int, int], Any] = {}
    seed_pool: Any = None
    seed_origin: Any = None
    seed_origin_i: int = -1
    seed_preview_bgr: np.ndarray | None = None
    show_edges = False
    drawing_region = False
    picking_final_origin = False
    region_polygon: list[tuple[float, float]] = []
    stored_region_polygon: list[tuple[float, float]] = []
    region_outside_holes: list[Any] = []
    region_ready = False
    mouse_cursor: tuple[float, float] | None = None

    def _begin_final_origin_picking() -> None:
        nonlocal picking_final_origin, region_ready, status, cached_observation
        if cached_observation is None or len(cached_observation.holes) < grid_cfg.min_holes_for_pose:
            return
        picking_final_origin = True
        region_ready = False
        if frozen_bgr is not None:
            preview = frozen_bgr.copy()
            draw_final_origin_prompt(
                preview,
                cached_observation.holes,
                stored_region_polygon or None,
                cached_observation.pitch_px,
            )
            cached_observation.annotated_bgr = preview
        status = "click final origin hole (table 0,0) in green region"

    def _pick_final_origin(click: tuple[float, float]) -> None:
        nonlocal cached_observation, status, picking_final_origin, region_ready, locked_origin_px
        if (
            frozen_bgr is None
            or frozen_intrinsics is None
            or cached_observation is None
        ):
            return
        cached_observation = set_calibration_origin(
            frozen_bgr,
            grid_cfg,
            frozen_intrinsics,
            region_holes=list(cached_observation.holes),
            click_px=click,
            pitch_px=cached_observation.pitch_px,
            polygon=stored_region_polygon or None,
            grow_rejected=list(cached_observation.detection_rejected or []),
            all_grown_holes=cached_observation.all_grown_holes,
            outside_holes=list(region_outside_holes),
            edges_bgr=cached_observation.debug_edges_bgr,
            n_raw=cached_observation.candidates_before_filter,
        )
        status = cached_observation.message
        if cached_observation.found and cached_observation.origin_hole is not None:
            picking_final_origin = False
            region_ready = True
            locked_origin_px = cached_observation.origin_lock_px
            print(f"[Table calib] Final origin set — {status}")
        else:
            print(f"[Table calib] {status}", file=sys.stderr)

    def _begin_region_drawing() -> None:
        nonlocal drawing_region, region_polygon, region_ready, status, cached_observation
        nonlocal picking_final_origin
        if cached_observation is None or not cached_observation.all_grown_holes:
            return
        if (
            frozen_bgr is not None
            and cached_observation.all_grown_holes
            and cached_observation.origin_hole is not None
        ):
            from camera.calibration_table.grid_grow import draw_grown_grid

            preview = frozen_bgr.copy()
            known = {
                (h.grid_col, h.grid_row): h for h in cached_observation.all_grown_holes
            }
            draw_grown_grid(
                preview,
                known,
                cached_observation.detection_rejected or [],
                cached_observation.origin_hole,
                cached_observation.pitch_px,
            )
            cached_observation.annotated_bgr = preview
        drawing_region = True
        region_polygon = []
        region_ready = False
        picking_final_origin = False
        status = "draw region — click corners, Enter=apply, A=all holes, U=undo"

    def _apply_region_polygon() -> None:
        nonlocal cached_observation, status, drawing_region, stored_region_polygon, region_outside_holes
        if (
            frozen_bgr is None
            or frozen_intrinsics is None
            or cached_observation is None
            or len(region_polygon) < 3
        ):
            status = "need 3+ points — click corners of region"
            print(f"[Table calib] {status}")
            return
        all_holes = cached_observation.all_grown_holes or cached_observation.holes
        rejected = cached_observation.detection_rejected or []
        cached_observation = apply_region_polygon(
            frozen_bgr,
            grid_cfg,
            frozen_intrinsics,
            all_grown_holes=list(all_holes),
            grow_rejected=list(rejected),
            pitch_px=cached_observation.pitch_px,
            polygon=list(region_polygon),
            edges_bgr=cached_observation.debug_edges_bgr,
            n_raw=cached_observation.candidates_before_filter,
        )
        drawing_region = False
        stored_region_polygon = list(region_polygon)
        region_keys = {(h.grid_col, h.grid_row) for h in cached_observation.holes}
        region_outside_holes = [
            h for h in all_holes if (h.grid_col, h.grid_row) not in region_keys
        ]
        status = cached_observation.message
        if len(cached_observation.holes) >= grid_cfg.min_holes_for_pose:
            print(f"[Table calib] Region applied — {status}")
            _begin_final_origin_picking()
        else:
            print(f"[Table calib] Region failed — {status}", file=sys.stderr)

    def _accept_all_holes() -> None:
        nonlocal drawing_region, region_ready, region_polygon, stored_region_polygon
        nonlocal region_outside_holes, status, cached_observation
        drawing_region = False
        region_polygon = []
        stored_region_polygon = []
        region_outside_holes = []
        if cached_observation is not None and cached_observation.all_grown_holes:
            cached_observation.holes = list(cached_observation.all_grown_holes)
            if frozen_bgr is not None:
                preview = frozen_bgr.copy()
                draw_final_origin_prompt(
                    preview,
                    cached_observation.holes,
                    None,
                    cached_observation.pitch_px,
                )
                cached_observation.annotated_bgr = preview
            status = "using all grown holes — click final origin"
            print("[Table calib] Using all grown holes — click final origin")
            _begin_final_origin_picking()

    def _on_mouse(event: int, x: int, y: int, _flags: int, _userdata) -> None:
        nonlocal click_px, locked_origin_px, pending_origin_click, picking_seeds
        nonlocal manual_seed_clicks, seed_assigned, seed_pool, seed_origin, seed_origin_i
        nonlocal seed_preview_bgr, cached_observation, status
        nonlocal drawing_region, region_polygon, mouse_cursor, picking_final_origin
        mouse_cursor = (float(x), float(y))
        if event == cv2.EVENT_LBUTTONDOWN and drawing_region:
            region_polygon.append((float(x), float(y)))
            status = f"draw region ({len(region_polygon)} pts) — Enter=apply, U=undo, A=all"
            return
        if event == cv2.EVENT_LBUTTONDOWN and picking_final_origin:
            _pick_final_origin((float(x), float(y)))
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        click_px = (float(x), float(y))
        if frozen_bgr is None:
            pending_origin_click = True
            return
        if picking_seeds:
            if seed_pool is None or seed_origin is None:
                return
            result = assign_seed_click(
                seed_origin.center_px,
                seed_pool,
                click_px,
                origin_i=seed_origin_i,
                assigned=seed_assigned,
            )
            if result is None:
                status = f"ignored click — pick unused neighbor ({len(seed_assigned)}/4)"
                print(f"[Table calib] {status}")
                return
            cell, idx = result
            manual_seed_clicks.append(click_px)
            seed_assigned[cell] = idx
            hole = seed_pool.holes[idx]
            label = {(1, 0): "E", (-1, 0): "W", (0, 1): "S", (0, -1): "N"}.get(cell, "?")
            print(f"[Table calib] Seed {label} at ({hole.center_px[0]:.0f}, {hole.center_px[1]:.0f})")
            preview = seed_preview_bgr.copy() if seed_preview_bgr is not None else frozen_bgr.copy()
            assigned_holes = {cell: seed_pool.holes[i] for cell, i in seed_assigned.items()}
            draw_seed_picking_overlay(preview, seed_pool, seed_origin, assigned_holes)
            cached_observation = detect_table_grid_preview(frozen_bgr, grid_cfg)
            cached_observation.annotated_bgr = preview
            status = f"pick neighbor holes ({len(seed_assigned)}/4)"
            if len(seed_assigned) >= 4:
                picking_seeds = False
                _grow_on_frozen(
                    frozen_bgr,
                    frozen_intrinsics,
                    locked_origin_px or click_px,
                    manual_seed_clicks=list(manual_seed_clicks),
                )
        else:
            pending_origin_click = True

    window = "Table calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, args.width, args.height)
    cv2.setMouseCallback(window, _on_mouse)

    detect_interval_s = max(0.05, args.detect_interval_ms / 1000.0)
    last_detect_s = 0.0
    cached_observation = None
    status = "detecting holes — click grow-origin, then 4 neighbors"
    saved_path: Path | None = None

    print(
        "[Table calib] Live preview. "
        "Click=origin/neighbors/region  Enter=apply region  S=save  R=reset  Q=quit"
    )

    def _start_seed_picking(frame_bgr: np.ndarray, origin_px: tuple[float, float]) -> None:
        nonlocal picking_seeds, manual_seed_clicks, seed_assigned, seed_pool
        nonlocal seed_origin, seed_origin_i, seed_preview_bgr, cached_observation, status
        nonlocal locked_origin_px
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        candidates, _edges = find_hole_candidates(gray, grid_cfg)
        seed_origin, seed_pool, _pitch, seed_origin_i = prepare_origin_pool(
            candidates, origin_px, grid_cfg
        )
        locked_origin_px = seed_origin.center_px
        manual_seed_clicks = []
        seed_assigned = {}
        picking_seeds = True
        seed_preview_bgr = frame_bgr.copy()
        preview = seed_preview_bgr.copy()
        draw_seed_picking_overlay(preview, seed_pool, seed_origin, {})
        cached_observation = detect_table_grid_preview(frame_bgr, grid_cfg)
        cached_observation.annotated_bgr = preview
        status = "pick 4 neighbor holes (0/4) — grow-origin already set"
        print("[Table calib] Frame frozen — click 4 neighbor holes (grow-origin set)")

    def _grow_on_frozen(
        frame_bgr: np.ndarray,
        intr,
        origin_px: tuple[float, float],
        *,
        manual_seed_clicks: list[tuple[float, float]] | None = None,
    ):
        nonlocal cached_observation, status, locked_origin_px
        try:
            cached_observation = detect_table_grid_grow(
                frame_bgr,
                grid_cfg,
                intr,
                origin_px,
                manual_seed_clicks=manual_seed_clicks,
            )
        except cv2.error as exc:
            print(f"[Table calib] Grid grow failed: {exc}", file=sys.stderr)
            cached_observation = detect_table_grid_preview(frame_bgr, grid_cfg)
            cached_observation.message = f"grow failed: {exc}"
        if cached_observation.origin_lock_px is not None:
            locked_origin_px = cached_observation.origin_lock_px
        status = cached_observation.message
        if cached_observation.found:
            print(f"[Table calib] Grid grown — {status}")
            _begin_region_drawing()
        elif frozen_bgr is not None:
            print(f"[Table calib] Frame frozen — {status}", file=sys.stderr)

    try:
        while True:
            now = time.perf_counter()

            if frozen_bgr is None:
                frame = streamer.read()
                color_bgr = frame.color_rgb[:, :, ::-1].copy()
                intrinsics = frame.color_intrinsics

                force_detect = pending_origin_click
                if cached_observation is None or force_detect or (now - last_detect_s) >= detect_interval_s:
                    if pending_origin_click and click_px is not None:
                        frozen_bgr = color_bgr.copy()
                        frozen_intrinsics = intrinsics
                        _start_seed_picking(frozen_bgr, click_px)
                        pending_origin_click = False
                    else:
                        cached_observation = detect_table_grid_preview(color_bgr, grid_cfg)
                        status = cached_observation.message
                        pending_origin_click = False
                    last_detect_s = now
            else:
                color_bgr = frozen_bgr
                intrinsics = frozen_intrinsics
                if pending_origin_click and click_px is not None:
                    if picking_seeds:
                        pending_origin_click = False
                    else:
                        _start_seed_picking(frozen_bgr, click_px)
                        pending_origin_click = False

            preview = (
                cached_observation.annotated_bgr.copy()
                if cached_observation is not None and cached_observation.annotated_bgr is not None
                else color_bgr
            )
            if frozen_bgr is not None:
                cv2.putText(
                    preview,
                    "FRAME FROZEN",
                    (12, 52),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
            if picking_seeds:
                cv2.putText(
                    preview,
                    "PICK 4 NEIGHBORS",
                    (12, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                )
            elif drawing_region:
                cv2.putText(
                    preview,
                    "DRAW REGION",
                    (12, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 200, 255),
                    2,
                )
                draw_polygon_draft(preview, region_polygon, cursor_px=mouse_cursor)
            elif picking_final_origin:
                cv2.putText(
                    preview,
                    "PICK FINAL ORIGIN",
                    (12, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 180),
                    2,
                )
            elif region_ready:
                cv2.putText(
                    preview,
                    "REGION SET",
                    (12, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 180),
                    2,
                )
            if (
                show_edges
                and cached_observation is not None
                and cached_observation.debug_edges_bgr is not None
            ):
                preview = cached_observation.debug_edges_bgr.copy()
            cv2.putText(
                preview,
                status,
                (12, preview.shape[0] - 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                preview,
                "grow-origin -> seeds -> region -> final-origin -> S=save  R=reset",
                (12, preview.shape[0] - 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (200, 200, 200),
                1,
            )
            if camera_settings and "exposure_us" in camera_settings:
                cv2.putText(
                    preview,
                    f"exposure locked {camera_settings['exposure_us']:.0f} us",
                    (12, preview.shape[0] - 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (180, 180, 255),
                    1,
                )
            if cached_observation is not None and cached_observation.found and region_ready:
                result = build_calibration_result(cached_observation, grid_cfg)
                t = result.camera_in_table_translation_m
                rpy = result.camera_in_table_rpy_deg
                cv2.putText(
                    preview,
                    f"cam pos [{t[0]:+.3f},{t[1]:+.3f},{t[2]:+.3f}]m rpy [{rpy[0]:+.1f},{rpy[1]:+.1f},{rpy[2]:+.1f}]",
                    (12, preview.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (180, 255, 180),
                    1,
                )

            cv2.imshow(window, preview)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                click_px = None
                locked_origin_px = None
                frozen_bgr = None
                frozen_intrinsics = None
                cached_observation = None
                last_detect_s = 0.0
                picking_seeds = False
                manual_seed_clicks = []
                seed_assigned = {}
                seed_pool = None
                seed_origin = None
                seed_origin_i = -1
                seed_preview_bgr = None
                drawing_region = False
                picking_final_origin = False
                region_polygon = []
                stored_region_polygon = []
                region_outside_holes = []
                region_ready = False
                mouse_cursor = None
                status = "detecting holes — click grow-origin, then 4 neighbors"
            if key in (ord("e"), ord("E")):
                show_edges = not show_edges
            if key in (13, 10):  # Enter
                if drawing_region:
                    _apply_region_polygon()
            if key in (ord("a"), ord("A")):
                if drawing_region:
                    _accept_all_holes()
            if key in (ord("u"), ord("U")):
                if drawing_region and region_polygon:
                    region_polygon.pop()
                    status = f"draw region ({len(region_polygon)} pts) — Enter=apply, U=undo, A=all"
            if key in (ord("d"), ord("D")):
                if (
                    cached_observation is not None
                    and cached_observation.all_grown_holes
                    and not picking_seeds
                ):
                    _begin_region_drawing()
            if key in (ord("s"), ord("S")):
                if frozen_bgr is None or locked_origin_px is None or picking_seeds:
                    print(
                        "[Table calib] Complete grow-origin, seeds, region, and final origin first",
                        file=sys.stderr,
                    )
                    continue
                if picking_final_origin:
                    print("[Table calib] Click final origin hole before save", file=sys.stderr)
                    continue
                if not region_ready:
                    print(
                        "[Table calib] Draw region (Enter) or A=all, then click final origin",
                        file=sys.stderr,
                    )
                    continue
                if cached_observation is None or not cached_observation.found:
                    print("[Table calib] Grow grid has no valid pose yet", file=sys.stderr)
                    continue
                print("[Table calib] Saving calibration from frozen frame...")
                result = build_calibration_result(cached_observation, grid_cfg)
                saved_path = save_calibration_result(
                    result,
                    run_dir,
                    color_bgr=frozen_bgr,
                    annotated_bgr=cached_observation.annotated_bgr,
                    camera_settings=camera_settings,
                )
                _print_result_summary(saved_path)
    finally:
        streamer.stop()
        cv2.destroyAllWindows()

    if saved_path is None:
        print(f"[Table calib] No save — outputs would go to {run_dir}")
        return 0
    return 0


def main() -> int:
    args = parse_args()
    grid_cfg = TableGridConfig.from_cli(
        pitch_in=args.pitch_in,
        hole_diameter_in=args.hole_diameter_in,
        min_holes=args.min_holes,
        expected_radius_px=args.expected_radius_px,
        max_radius_px=args.max_radius_px,
        edge_margin_px=args.edge_margin_px,
        min_circularity=args.min_circularity,
    )

    if args.image is not None:
        return run_image_mode(args, grid_cfg)
    return run_live_mode(args, grid_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
