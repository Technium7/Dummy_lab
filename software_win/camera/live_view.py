#!/usr/bin/env python3
"""Open live windows: Thorlabs Zelux front + side + Infinity 2-1RM.

Each window has manual Exposure / Gain trackbars (auto disabled when possible).

Usage::

  python software_win/camera/live_view.py
  python software_win/camera/live_view.py --demo
  python software_win/camera/live_view.py --front-serial SN1 --side-serial SN2
  python software_win/camera/live_view.py --list

Keys: Q / Esc quits (focus either window).
Trackbars: Exposure_0.1ms (10 => 1.0 ms), Gain_0.1 (100 => 10.0)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_SOFTWARE_WIN = Path(__file__).resolve().parents[1]
if str(_SOFTWARE_WIN) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_WIN))

from camera.base import CameraBackend
from camera.display import (
    WindowSizeState,
    controls_title_for,
    create_controls_window,
    create_resizable_window,
    label_frame,
    require_cv2,
    show_resizable,
    wait_quit_key,
)
from camera.infinity import DemoInfinityCamera, InfinityCamera, list_infinity_devices
from camera.trackbars import attach_exposure_gain_trackbars, status_text
from camera.zelux import DemoZeluxCamera, ZeluxCamera, list_zelux_devices


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Live view: Zelux front + Zelux side + Infinity 2-1RM"
    )
    p.add_argument("--list", action="store_true", help="List detected cameras and exit")
    p.add_argument("--demo", action="store_true", help="Synthetic frames (no hardware)")

    p.add_argument("--front-serial", default="38862", help="Thorlabs serial for front Zelux")
    p.add_argument("--front-index", type=int, default=1, help="Front Zelux index if serial omitted")
    p.add_argument("--front-exposure-ms", type=float, default=10.0, help="Initial front exposure (ms)")
    p.add_argument("--front-gain", type=float, default=0.0, help="Initial front gain")

    p.add_argument("--side-serial", default="35968", help="Thorlabs serial for side Zelux")
    p.add_argument("--side-index", type=int, default=0, help="Side Zelux index if serial omitted")
    p.add_argument("--side-exposure-ms", type=float, default=10.0, help="Initial side exposure (ms)")
    p.add_argument("--side-gain", type=float, default=0.0, help="Initial side gain")

    # Backward-compatible aliases (map to front)
    p.add_argument("--zelux-serial", default="", help=argparse.SUPPRESS)
    p.add_argument("--zelux-index", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--zelux-exposure-ms", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--zelux-gain", type=float, default=None, help=argparse.SUPPRESS)

    p.add_argument(
        "--infinity-number",
        type=int,
        default=1,
        help="lucam camera number (1-indexed, default 1)",
    )
    p.add_argument("--infinity-exposure-ms", type=float, default=25.0, help="Initial Infinity exposure (ms)")
    p.add_argument("--infinity-gain", type=float, default=1.0, help="Initial Infinity gain")
    p.add_argument(
        "--infinity-bayer",
        default="BG",
        choices=["BG", "GB", "RG", "GR"],
        help="Bayer pattern for Infinity raw demosaic (default BG)",
    )
    p.add_argument(
        "--infinity-opencv-index",
        type=int,
        default=None,
        help="Force OpenCV DirectShow index for Infinity (skips lucam)",
    )
    p.add_argument("--prefer-opencv-infinity", action="store_true", help="Prefer OpenCV for Infinity")
    p.add_argument("--max-width", type=int, default=960, help="Max display width per window")
    p.add_argument("--zelux-only", action="store_true", help="Only open front + side Zelux windows")
    p.add_argument("--front-only", action="store_true", help="Only open front Zelux")
    p.add_argument("--side-only", action="store_true", help="Only open side Zelux")
    p.add_argument("--infinity-only", action="store_true", help="Only open Infinity window")
    return p.parse_args(argv)


def _apply_legacy_aliases(args: argparse.Namespace) -> None:
    if args.zelux_serial and not args.front_serial:
        args.front_serial = args.zelux_serial
    if args.zelux_index is not None:
        args.front_index = args.zelux_index
    if args.zelux_exposure_ms is not None:
        args.front_exposure_ms = args.zelux_exposure_ms
    if args.zelux_gain is not None:
        args.front_gain = args.zelux_gain


def _print_list() -> None:
    print("Zelux / TLCamera:")
    zelux = list_zelux_devices()
    if not zelux:
        print("  (none - install ThorCam + pylablib, and plug in the camera)")
    for info in zelux:
        print(f"  [{info.index}] {info.label()}")
    print("Infinity / OpenCV:")
    infinity = list_infinity_devices()
    if not infinity:
        print("  (none - install Infinity Capture + lucam, or check DirectShow devices)")
    for info in infinity:
        print(f"  [{info.index}] {info.label()}")


def _want_front(args: argparse.Namespace) -> bool:
    if args.infinity_only or args.side_only:
        return False
    return True


def _want_side(args: argparse.Namespace) -> bool:
    if args.infinity_only or args.front_only:
        return False
    return True


def _want_infinity(args: argparse.Namespace) -> bool:
    if args.zelux_only or args.front_only or args.side_only:
        return False
    return True


def _open_zelux(
    *,
    title: str,
    serial: str,
    index: int,
    exposure_ms: float,
    gain: float,
    demo: bool,
) -> tuple[str, CameraBackend, float, float]:
    if demo:
        return (title, DemoZeluxCamera(), exposure_ms, gain)
    return (
        title,
        ZeluxCamera(
            serial=serial or None,
            exposure_s=max(exposure_ms, 0.1) / 1000.0,
            gain=gain,
            index=index,
        ),
        exposure_ms,
        gain,
    )


def _build_cameras(args: argparse.Namespace) -> list[tuple[str, CameraBackend, float, float]]:
    """Return (title, camera, initial_exposure_ms, initial_gain)."""
    cams: list[tuple[str, CameraBackend, float, float]] = []

    if _want_front(args):
        cams.append(
            _open_zelux(
                title="Zelux front",
                serial=args.front_serial,
                index=args.front_index,
                exposure_ms=args.front_exposure_ms,
                gain=args.front_gain,
                demo=args.demo,
            )
        )
    if _want_side(args):
        cams.append(
            _open_zelux(
                title="Zelux side",
                serial=args.side_serial,
                index=args.side_index,
                exposure_ms=args.side_exposure_ms,
                gain=args.side_gain,
                demo=args.demo,
            )
        )
    if _want_infinity(args):
        if args.demo:
            cams.append(
                ("Infinity 2-1RM", DemoInfinityCamera(), args.infinity_exposure_ms, args.infinity_gain)
            )
        else:
            cams.append(
                (
                    "Infinity 2-1RM",
                    InfinityCamera(
                        camera_number=args.infinity_number,
                        opencv_index=args.infinity_opencv_index,
                        exposure_ms=args.infinity_exposure_ms,
                        gain=args.infinity_gain,
                        prefer_opencv=args.prefer_opencv_infinity
                        or args.infinity_opencv_index is not None,
                        bayer=args.infinity_bayer,
                    ),
                    args.infinity_exposure_ms,
                    args.infinity_gain,
                )
            )
    return cams


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _apply_legacy_aliases(args)
    if args.list:
        _print_list()
        return 0

    cv2 = require_cv2()
    cameras = _build_cameras(args)
    if not cameras:
        print("No cameras selected.")
        return 1

    opened: list[tuple[str, CameraBackend, object]] = []
    size_states: dict[str, WindowSizeState] = {}
    try:
        for title, cam, exp_ms, gain in cameras:
            print(f"Opening {title}...")
            cam.open()
            print(f"  -> {cam.info.label()}")
            create_resizable_window(cv2, title)
            ctrl = controls_title_for(title)
            create_controls_window(cv2, ctrl)
            state = attach_exposure_gain_trackbars(
                cv2, ctrl, cam, exposure_ms=exp_ms, gain=gain
            )
            opened.append((title, cam, state))
            size_states[title] = WindowSizeState()

        print("Live view running.")
        print("  Windows: " + ", ".join(t for t, _, _ in opened))
        print("  Exposure/gain trackbars are in the separate '… controls' windows.")
        print("  Drag image window edges to resize; Q/Esc quits.")
        while True:
            for title, cam, state in opened:
                frame = cam.grab_bgr()
                if frame is None:
                    placeholder = getattr(cam, "_last_frame", None)
                    if placeholder is None:
                        import numpy as np

                        placeholder = np.zeros((240, 320, 3), dtype=np.uint8)
                        frame = label_frame(cv2, placeholder, f"{status_text(cam, state)} (no frame)")
                    else:
                        frame = label_frame(cv2, placeholder, status_text(cam, state))
                else:
                    cam._last_frame = frame  # type: ignore[attr-defined]
                    frame = label_frame(cv2, frame, status_text(cam, state))
                show_resizable(
                    cv2,
                    title,
                    frame,
                    size_states[title],
                    max_width=args.max_width,
                )
            if wait_quit_key(cv2, 1):
                break
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        for _, cam, _ in opened:
            try:
                cam.close()
            except Exception:
                pass
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
