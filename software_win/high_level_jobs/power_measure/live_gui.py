"""Live top-camera GUI: hover depth/table XY, click to set insert target."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from high_level_jobs.grab_sensor.table_position import (
    format_table_xyz_display,
    infer_pixel_table_pose,
)

from .models import PowerMeasureConfig
from .target import InsertTarget, build_insert_target_from_uv, draw_insert_overlay

PANEL_H = 108


@dataclass
class LiveGuiResult:
    """Outcome of the live click GUI."""

    insert: InsertTarget | None = None
    confirmed: bool = False


@dataclass
class GuiState:
    hover_uv: tuple[int, int] | None = None
    hover_depth_m: float | None = None
    hover_table_xy_in: tuple[float, float] | None = None
    insert: InsertTarget | None = None
    status: str = "Click a point. Enter/E confirms (then IK/plan/execute). C clear. Q quit."
    last_error: str = ""
    confirmed: bool = False
    depth_frame: np.ndarray | None = None
    params: dict[str, Any] = field(default_factory=dict)


def _require_cv2():
    import cv2

    return cv2


def _sample_depth_m(depth_m: np.ndarray | None, u: int, v: int) -> float | None:
    if depth_m is None:
        return None
    h, w = depth_m.shape[:2]
    if not (0 <= u < w and 0 <= v < h):
        return None
    d = float(depth_m[v, u])
    if d <= 1e-4 or not np.isfinite(d):
        return None
    return d


def _hover_table_xy_in(
    cfg: PowerMeasureConfig,
    uv: tuple[int, int],
) -> tuple[float, float] | None:
    if cfg.table_calib_path is None:
        return None
    pose = infer_pixel_table_pose(uv, cfg.table_calib_path, z_m=0.0)
    if pose is None:
        return None
    xy_in = pose.get("table_xy_in")
    if not xy_in or len(xy_in) < 2:
        return None
    return float(xy_in[0]), float(xy_in[1])


def _mode_hint(params: dict[str, Any]) -> str:
    bits: list[str] = []
    if params.get("ik_gui"):
        bits.append("ik-gui")
    if params.get("plan_gui"):
        bits.append("plan-gui")
    if params.get("execute"):
        bits.append("execute")
    if not bits:
        bits.append("plan only")
    return "+".join(bits)


def _draw_panel(cv2, state: GuiState, width: int) -> np.ndarray:
    panel = np.zeros((PANEL_H, width, 3), dtype=np.uint8)
    panel[:] = (36, 36, 36)
    mode = _mode_hint(state.params)
    lines = [
        f"power_measure  |  click=target  Enter/E=go ({mode})  C=clear  Q=quit",
        state.status[:110],
    ]
    if state.hover_uv is not None:
        u, v = state.hover_uv
        depth_s = (
            f"{state.hover_depth_m:.4f} m"
            if state.hover_depth_m is not None
            else "n/a"
        )
        if state.hover_table_xy_in is not None:
            tx, ty = state.hover_table_xy_in
            table_s = f"table XY=({tx:+.2f}, {ty:+.2f}) in"
        else:
            table_s = "table XY=n/a"
        lines.append(f"hover UV=({u},{v})  depth={depth_s}  {table_s}")
    else:
        lines.append("hover: move mouse over image")

    if state.insert is not None:
        m = state.insert.measured_table_xyz_m
        t = state.insert.tcp_table_xyz_m
        lines.append(
            f"measured {format_table_xyz_display(m)}  |  "
            f"TCP {format_table_xyz_display(t)}  yaw={state.insert.radial_deg:.1f} deg radial"
        )
    elif state.last_error:
        lines.append(state.last_error[:110])

    for i, text in enumerate(lines[:5]):
        color = (180, 180, 180) if i == 0 else (220, 220, 220)
        if state.last_error and i == len(lines) - 1 and state.insert is None:
            color = (80, 80, 255)
        cv2.putText(panel, text, (8, 20 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
    return panel


def _compose_display(cv2, color_rgb: np.ndarray, state: GuiState) -> np.ndarray:
    bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    if state.insert is not None:
        draw_insert_overlay(bgr, state.insert, panel_h=0)
    if state.hover_uv is not None:
        u, v = state.hover_uv
        cv2.drawMarker(bgr, (u, v), (200, 200, 200), cv2.MARKER_CROSS, 12, 1)
    panel = _draw_panel(cv2, state, bgr.shape[1])
    return np.vstack([panel, bgr])


def run_live_gui(
    cfg: PowerMeasureConfig,
    params: dict[str, Any],
) -> LiveGuiResult:
    """Open live D415 window for click targeting.

    On Enter/E with a target: close the camera/OpenCV session and return
    ``confirmed=True`` so the caller can open PyBullet ``--ik-gui`` / ``--plan-gui``
    the same way as grab_sensor (no OpenCV conflict).
    """
    cv2 = _require_cv2()
    from camera.realsense import RealSenseStreamer

    serial = str(params.get("camera_serial") or cfg.camera_serial or "").strip() or None
    width = int(params.get("width", 640))
    height = int(params.get("height", 480))
    fps = int(params.get("fps", 30))

    state = GuiState(params=dict(params))
    window = "power_measure"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, _flags, _param):
        if y < PANEL_H:
            return
        uy, ux = int(y - PANEL_H), int(x)
        if event == cv2.EVENT_MOUSEMOVE:
            state.hover_uv = (ux, uy)
            state.hover_depth_m = _sample_depth_m(state.depth_frame, ux, uy)
            state.hover_table_xy_in = _hover_table_xy_in(cfg, (ux, uy))
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        try:
            state.insert = build_insert_target_from_uv(cfg, (ux, uy))
            state.last_error = ""
            m_in = state.insert.debug.get("measured_table_xyz_in", [])
            t_in = state.insert.debug.get("tcp_table_xyz_in", [])
            state.status = (
                f"Target set measured=({m_in[0]:+.2f},{m_in[1]:+.2f},{m_in[2]:+.2f})in "
                f"TCP=({t_in[0]:+.2f},{t_in[1]:+.2f},{t_in[2]:+.2f})in - Enter/E to go"
            )
            print(
                f"[power_measure] click UV=({ux},{uy}) "
                f"measured_in={m_in} tcp_in={t_in} "
                f"rpy_deg={state.insert.debug.get('insert_rpy_world_deg')}"
            )
        except Exception as exc:  # noqa: BLE001 - show in GUI
            state.insert = None
            state.last_error = str(exc)
            state.status = "Click failed - see error"
            print(f"[power_measure] click error: {exc}")

    cv2.setMouseCallback(window, on_mouse)

    streamer_kw: dict[str, Any] = {"width": width, "height": height, "fps": fps}
    if serial:
        streamer_kw["serial"] = serial

    with RealSenseStreamer(**streamer_kw) as streamer:
        print(
            f"[power_measure] live GUI (serial={serial or 'any'})  "
            f"mode={_mode_hint(params)}"
        )
        while True:
            frame = streamer.read(with_depth=True)
            state.depth_frame = frame.depth_m
            if state.hover_uv is not None:
                u, v = state.hover_uv
                state.hover_depth_m = _sample_depth_m(state.depth_frame, u, v)
                state.hover_table_xy_in = _hover_table_xy_in(cfg, (u, v))

            display = _compose_display(cv2, frame.color_rgb, state)
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("c"), ord("C")):
                state.insert = None
                state.last_error = ""
                state.status = "Cleared. Click a point."
                continue
            if key in (13, ord("e"), ord("E")):  # Enter or E
                if state.insert is None:
                    state.status = "No target - click first"
                    continue
                # Exit live session before PyBullet / execute (grab_sensor-style).
                state.confirmed = True
                state.status = "Confirmed - closing camera for IK/plan..."
                print("[power_measure] target confirmed; closing live camera...")
                break

    cv2.destroyWindow(window)
    try:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    except Exception:  # noqa: BLE001
        pass

    return LiveGuiResult(insert=state.insert, confirmed=state.confirmed)
