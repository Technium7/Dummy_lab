"""OpenCV trackbars for manual exposure (ms) and gain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from camera.controls import ControlLimits


@dataclass
class TrackbarState:
    exposure_ms: float
    gain: float
    dirty: bool = False


def np_clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def attach_exposure_gain_trackbars(
    cv2,
    window: str,
    cam: Any,
    *,
    exposure_ms: Optional[float] = None,
    gain: Optional[float] = None,
) -> TrackbarState:
    """
    Add Exposure and Gain trackbars under ``window``.

    Trackbar units:
      Exposure_0.1ms : value 10 => 1.0 ms
      Gain_0.1       : value 100 => 10.0
    """
    limits = cam.get_control_limits() if hasattr(cam, "get_control_limits") else ControlLimits()
    if hasattr(cam, "disable_auto_exposure_gain"):
        try:
            cam.disable_auto_exposure_gain()
        except Exception:
            pass

    cur_exp = float(exposure_ms) if exposure_ms is not None else (
        float(cam.get_exposure_ms()) if hasattr(cam, "get_exposure_ms") else 10.0
    )
    cur_gain = float(gain) if gain is not None else (
        float(cam.get_gain()) if hasattr(cam, "get_gain") else 0.0
    )
    cur_exp = float(np_clip(cur_exp, limits.exposure_ms_min, limits.exposure_ms_max))
    cur_gain = float(np_clip(cur_gain, limits.gain_min, limits.gain_max))

    if hasattr(cam, "set_exposure_ms"):
        try:
            cam.set_exposure_ms(cur_exp)
        except Exception:
            pass
    if hasattr(cam, "set_gain"):
        try:
            cam.set_gain(cur_gain)
        except Exception:
            pass

    state = TrackbarState(exposure_ms=cur_exp, gain=cur_gain, dirty=False)

    exp_max_ticks = max(1, int(round(limits.exposure_ms_max * 10.0)))
    gain_max_ticks = max(1, int(round(limits.gain_max * 10.0)))
    exp_min_ticks = max(1, int(round(limits.exposure_ms_min * 10.0)))
    exp_init = int(round(cur_exp * 10.0))
    gain_init = int(round(cur_gain * 10.0))

    def on_exposure(pos: int) -> None:
        ms = max(limits.exposure_ms_min, max(exp_min_ticks, pos) / 10.0)
        if abs(ms - state.exposure_ms) < 1e-6:
            return
        state.exposure_ms = ms
        state.dirty = True
        if hasattr(cam, "set_exposure_ms"):
            try:
                cam.set_exposure_ms(ms)
            except Exception:
                pass

    def on_gain(pos: int) -> None:
        g = float(np_clip(pos / 10.0, limits.gain_min, limits.gain_max))
        if abs(g - state.gain) < 1e-6:
            return
        state.gain = g
        state.dirty = True
        if hasattr(cam, "set_gain"):
            try:
                cam.set_gain(g)
            except Exception:
                pass

    cv2.createTrackbar("Exposure_0.1ms", window, exp_init, exp_max_ticks, on_exposure)
    cv2.createTrackbar("Gain_0.1", window, gain_init, gain_max_ticks, on_gain)
    on_exposure(exp_init)
    on_gain(gain_init)
    state.dirty = False
    return state


def status_text(cam: Any, state: TrackbarState) -> str:
    label = cam.info.label() if hasattr(cam, "info") else "camera"
    return f"{label}  exp={state.exposure_ms:.1f}ms  gain={state.gain:.1f}"
