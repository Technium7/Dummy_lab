#!/usr/bin/env python3
"""Open two live windows: Thorlabs Zelux + Infinity 2-1RM.

Each window has manual Exposure / Gain trackbars (auto disabled when possible).

Usage::

  python software_win/camera/dual_live_view.py
  python software_win/camera/dual_live_view.py --demo
  python software_win/camera/dual_live_view.py --zelux-exposure-ms 5 --zelux-gain 6

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
from camera.display import label_frame, require_cv2, resize_max, wait_quit_key
from camera.infinity import DemoInfinityCamera, InfinityCamera, list_infinity_devices
from camera.trackbars import attach_exposure_gain_trackbars, status_text
from camera.zelux import DemoZeluxCamera, ZeluxCamera, list_zelux_devices


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dual live view: Zelux + Infinity 2-1RM")
    p.add_argument("--list", action="store_true", help="List detected cameras and exit")
    p.add_argument("--demo", action="store_true", help="Synthetic frames (no hardware)")
    p.add_argument("--zelux-serial", default="", help="Thorlabs TLCamera serial (optional)")
    p.add_argument("--zelux-index", type=int, default=0, help="Zelux index if serial omitted")
    p.add_argument("--zelux-exposure-ms", type=float, default=10.0, help="Initial Zelux exposure (ms)")
    p.add_argument("--zelux-gain", type=float, default=0.0, help="Initial Zelux gain")
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
    p.add_argument("--zelux-only", action="store_true", help="Only open Zelux window")
    p.add_argument("--infinity-only", action="store_true", help="Only open Infinity window")
    return p.parse_args(argv)


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


def _build_cameras(args: argparse.Namespace) -> list[tuple[str, CameraBackend, float, float]]:
    """Return (title, camera, initial_exposure_ms, initial_gain)."""
    cams: list[tuple[str, CameraBackend, float, float]] = []
    if args.demo:
        if not args.infinity_only:
            cams.append(("Zelux", DemoZeluxCamera(), args.zelux_exposure_ms, args.zelux_gain))
        if not args.zelux_only:
            cams.append(
                ("Infinity 2-1RM", DemoInfinityCamera(), args.infinity_exposure_ms, args.infinity_gain)
            )
        return cams

    if not args.infinity_only:
        cams.append(
            (
                "Zelux",
                ZeluxCamera(
                    serial=args.zelux_serial or None,
                    exposure_s=max(args.zelux_exposure_ms, 0.1) / 1000.0,
                    gain=args.zelux_gain,
                    index=args.zelux_index,
                ),
                args.zelux_exposure_ms,
                args.zelux_gain,
            )
        )
    if not args.zelux_only:
        cams.append(
            (
                "Infinity 2-1RM",
                InfinityCamera(
                    camera_number=args.infinity_number,
                    opencv_index=args.infinity_opencv_index,
                    exposure_ms=args.infinity_exposure_ms,
                    gain=args.infinity_gain,
                    prefer_opencv=args.prefer_opencv_infinity or args.infinity_opencv_index is not None,
                    bayer=args.infinity_bayer,
                ),
                args.infinity_exposure_ms,
                args.infinity_gain,
            )
        )
    return cams


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        _print_list()
        return 0

    cv2 = require_cv2()
    cameras = _build_cameras(args)
    opened: list[tuple[str, CameraBackend, object]] = []
    try:
        for title, cam, exp_ms, gain in cameras:
            print(f"Opening {title}...")
            cam.open()
            print(f"  -> {cam.info.label()}")
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            state = attach_exposure_gain_trackbars(
                cv2, title, cam, exposure_ms=exp_ms, gain=gain
            )
            opened.append((title, cam, state))

        print("Live view running.")
        print("  Trackbars: Exposure_0.1ms (10=1ms), Gain_0.1 (100=10)")
        print("  Press Q or Esc in a camera window to quit.")
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
                frame = resize_max(cv2, frame, args.max_width)
                cv2.imshow(title, frame)
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
