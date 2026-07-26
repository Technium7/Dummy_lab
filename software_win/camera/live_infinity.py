#!/usr/bin/env python3
"""Live view for Infinity 2-1RM (manual exposure / gain trackbars).

  python software_win/camera/live_infinity.py
  python software_win/camera/live_infinity.py --demo
  python software_win/camera/live_infinity.py --opencv-index 0 --exposure-ms 8 --gain 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SOFTWARE_WIN = Path(__file__).resolve().parents[1]
if str(_SOFTWARE_WIN) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_WIN))

from camera.display import label_frame, require_cv2, resize_max, wait_quit_key
from camera.infinity import DemoInfinityCamera, InfinityCamera, list_infinity_devices
from camera.trackbars import attach_exposure_gain_trackbars, status_text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Infinity 2-1RM live view")
    p.add_argument("--list", action="store_true", help="List devices and exit")
    p.add_argument("--demo", action="store_true", help="Synthetic frames")
    p.add_argument("--camera-number", type=int, default=1, help="lucam 1-indexed camera number")
    p.add_argument("--opencv-index", type=int, default=None, help="Force OpenCV DirectShow index")
    p.add_argument("--prefer-opencv", action="store_true")
    p.add_argument("--exposure-ms", type=float, default=25.0, help="Initial exposure (ms)")
    p.add_argument("--gain", type=float, default=1.0, help="Initial gain")
    p.add_argument("--bayer", default="BG", choices=["BG", "GB", "RG", "GR"])
    p.add_argument("--max-width", type=int, default=960)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        devices = list_infinity_devices()
        if not devices:
            print("No Infinity / OpenCV cameras found.")
            return 1
        for info in devices:
            print(info.label())
        return 0

    cv2 = require_cv2()
    cam = (
        DemoInfinityCamera()
        if args.demo
        else InfinityCamera(
            camera_number=args.camera_number,
            opencv_index=args.opencv_index,
            exposure_ms=args.exposure_ms,
            gain=args.gain,
            prefer_opencv=args.prefer_opencv or args.opencv_index is not None,
            bayer=args.bayer,
        )
    )
    title = "Infinity 2-1RM"
    try:
        cam.open()
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        state = attach_exposure_gain_trackbars(
            cv2, title, cam, exposure_ms=args.exposure_ms, gain=args.gain
        )
        print(f"Live: {cam.info.label()}  (Q/Esc to quit; use trackbars for exp/gain)")
        while True:
            frame = cam.grab_bgr()
            if frame is None:
                continue
            frame = label_frame(cv2, frame, status_text(cam, state))
            frame = resize_max(cv2, frame, args.max_width)
            cv2.imshow(title, frame)
            if wait_quit_key(cv2, 1):
                break
    finally:
        cam.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
