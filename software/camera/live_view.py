#!/usr/bin/env python3
"""Live mosaic of all connected RealSense cameras (for aiming / identity).

Usage (from repo root or software/):

  python software/camera/live_view.py
  python software/camera/live_view.py --serial 104122062828 --serial 024122062290

Keys:
  Q / Esc   quit
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

_SOFTWARE_DIR = Path(__file__).resolve().parents[1]
if str(_SOFTWARE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_DIR))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Live tiled view of all (or selected) RealSense cameras.",
    )
    p.add_argument(
        "--serial",
        action="append",
        default=[],
        help="Limit to this serial (repeatable). Default: all connected devices.",
    )
    p.add_argument("--width", type=int, default=640, help="Per-camera stream width")
    p.add_argument("--height", type=int, default=480, help="Per-camera stream height")
    p.add_argument("--fps", type=int, default=30, help="Per-camera stream FPS")
    p.add_argument(
        "--tile-width",
        type=int,
        default=480,
        help="Display width of each tile in the mosaic (default: 480)",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List connected RealSense serials and exit",
    )
    return p.parse_args(argv)


def _require_cv2():
    from high_level_jobs.subtasks.grab_sensor.detection.cv2_gui import require_cv2_gui

    return require_cv2_gui()


def _list_serials() -> list[str]:
    from high_level_jobs.subtasks.grab_sensor.detection.capture import _list_realsense_devices

    return list(_list_realsense_devices())


def _grid_shape(n: int) -> tuple[int, int]:
    """Rows, cols for a roughly square mosaic."""
    if n <= 0:
        return 0, 0
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def _label_bgr(cv2, img_bgr: np.ndarray, text: str) -> np.ndarray:
    out = img_bgr.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (0, 0, 0), thickness=-1)
    cv2.putText(
        out,
        text,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def _tile_mosaic(
    cv2,
    tiles_bgr: list[np.ndarray],
    *,
    tile_w: int,
    tile_h: int,
) -> np.ndarray:
    n = len(tiles_bgr)
    rows, cols = _grid_shape(n)
    canvas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles_bgr):
        r, c = divmod(i, cols)
        resized = cv2.resize(tile, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        y0, x0 = r * tile_h, c * tile_w
        canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = resized
    return canvas


def run_live_view(
    *,
    serials: list[str] | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    tile_width: int = 480,
) -> int:
    from high_level_jobs.subtasks.grab_sensor.detection.capture import RealSenseStreamer

    cv2 = _require_cv2()
    devices = list(serials) if serials else _list_serials()
    if not devices:
        print("No RealSense devices found.", file=sys.stderr)
        return 1

    print(f"[live_view] Opening {len(devices)} camera(s): {devices}", flush=True)
    streamers: list[tuple[str, Any]] = []
    try:
        for sn in devices:
            streamer = RealSenseStreamer(
                width=width,
                height=height,
                fps=fps,
                serial=sn,
                warmup=3,
                lock_exposure=False,
            )
            streamer.frame_timeout_ms = 4000
            streamer.frame_retries = 2
            streamer.start_retries = 1
            streamer.reset_on_start_retry = False
            streamer.start()
            streamers.append((sn, streamer))
            print(f"[live_view] Ready: {sn}", flush=True)

        tile_h = max(1, int(round(tile_width * height / max(width, 1))))
        window = "RealSense live view (all cameras)"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        print("[live_view] Streaming. Press Q or Esc to quit.", flush=True)

        while True:
            tiles: list[np.ndarray] = []
            for sn, streamer in streamers:
                try:
                    frame = streamer.read(with_depth=False)
                    bgr = frame.color_rgb[:, :, ::-1]
                    label = sn or frame.device_serial or "?"
                    tiles.append(_label_bgr(cv2, bgr, label))
                except Exception as exc:
                    blank = np.zeros((height, width, 3), dtype=np.uint8)
                    tiles.append(_label_bgr(cv2, blank, f"{sn}: {exc}"))

            mosaic = _tile_mosaic(cv2, tiles, tile_w=tile_width, tile_h=tile_h)
            cv2.imshow(window, mosaic)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
    finally:
        for sn, streamer in streamers:
            try:
                streamer.stop()
            except Exception:
                pass
            print(f"[live_view] Released: {sn}", flush=True)
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        devices = _list_serials()
        if not devices:
            print("No RealSense devices found.")
            return 1
        print("Connected RealSense serials:")
        for i, sn in enumerate(devices):
            print(f"  [{i}] {sn}")
        return 0

    serials = [s.strip() for s in args.serial if s.strip()] or None
    return run_live_view(
        serials=serials,
        width=args.width,
        height=args.height,
        fps=args.fps,
        tile_width=args.tile_width,
    )


if __name__ == "__main__":
    raise SystemExit(main())
