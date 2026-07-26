#!/usr/bin/env python3
"""Live view for Thorlabs Zelux only (manual exposure / gain trackbars).

  python software_win/camera/live_zelux.py
  python software_win/camera/live_zelux.py --demo
  python software_win/camera/live_zelux.py --serial 12345 --exposure-ms 5 --gain 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SOFTWARE_WIN = Path(__file__).resolve().parents[1]
if str(_SOFTWARE_WIN) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_WIN))

from camera.display import label_frame, require_cv2, resize_max, wait_quit_key
from camera.trackbars import attach_exposure_gain_trackbars, status_text
from camera.zelux import DemoZeluxCamera, ZeluxCamera, list_zelux_devices


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Thorlabs Zelux live view")
    p.add_argument("--list", action="store_true", help="List TLCamera serials and exit")
    p.add_argument("--demo", action="store_true", help="Synthetic frames")
    p.add_argument("--serial", default="", help="Camera serial (optional)")
    p.add_argument("--index", type=int, default=0, help="Camera index if serial omitted")
    p.add_argument("--exposure-ms", type=float, default=10.0, help="Initial exposure (ms)")
    p.add_argument("--gain", type=float, default=0.0, help="Initial gain")
    p.add_argument("--max-width", type=int, default=960)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        devices = list_zelux_devices()
        if not devices:
            print("No Zelux/TLCamera found.")
            return 1
        for info in devices:
            print(info.label())
        return 0

    cv2 = require_cv2()
    cam = (
        DemoZeluxCamera()
        if args.demo
        else ZeluxCamera(
            serial=args.serial or None,
            exposure_s=max(args.exposure_ms, 0.1) / 1000.0,
            gain=args.gain,
            index=args.index,
        )
    )
    title = "Zelux"
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
