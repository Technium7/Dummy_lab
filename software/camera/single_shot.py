#!/usr/bin/env python3
"""Capture one color image and depth frame from an Intel RealSense camera (e.g. D415)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import pyrealsense2 as rs
except ImportError as exc:
    raise SystemExit(
        "pyrealsense2 is required. Install with: pip install pyrealsense2"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture a single color photo and depth frame from a RealSense camera.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("captures"),
        help="Directory for output files (default: captures)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Base filename (default: timestamp, e.g. shot_20260519_143022)",
    )
    parser.add_argument("--width", type=int, default=640, help="Stream width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Stream height (default: 480)")
    parser.add_argument("--fps", type=int, default=30, help="Stream frame rate (default: 30)")
    parser.add_argument(
        "--warmup",
        type=int,
        default=15,
        help="Frames to discard before capture (default: 15)",
    )
    parser.add_argument(
        "--serial",
        type=str,
        default="",
        help="Device serial number (default: first connected camera)",
    )
    parser.add_argument(
        "--no-depth-vis",
        action="store_true",
        help="Skip writing an 8-bit depth preview PNG",
    )
    return parser.parse_args()


def _default_name() -> str:
    return datetime.now(timezone.utc).strftime("shot_%Y%m%d_%H%M%S")


def _list_devices() -> list[str]:
    ctx = rs.context()
    return [d.get_info(rs.camera_info.serial_number) for d in ctx.query_devices()]


def _depth_stats_mm(depth_mm: np.ndarray) -> dict[str, float | int]:
    valid = depth_mm[depth_mm > 0]
    if valid.size == 0:
        return {"valid_pixels": 0}
    h, w = depth_mm.shape
    cy, cx = h // 2, w // 2
    center = int(depth_mm[cy, cx])
    return {
        "valid_pixels": int(valid.size),
        "total_pixels": int(depth_mm.size),
        "min_mm": int(valid.min()),
        "max_mm": int(valid.max()),
        "mean_mm": round(float(valid.mean()), 2),
        "center_mm": center,
        "center_m": round(center / 1000.0, 4) if center else 0.0,
    }


def _intrinsics_to_dict(intr: rs.intrinsics) -> dict:
    return {
        "width": intr.width,
        "height": intr.height,
        "fx": intr.fx,
        "fy": intr.fy,
        "ppx": intr.ppx,
        "ppy": intr.ppy,
        "model": str(intr.model),
        "coeffs": list(intr.coeffs),
    }


def capture_single_shot(
    out_dir: Path,
    name: str,
    width: int,
    height: int,
    fps: int,
    warmup: int,
    serial: str,
    write_depth_vis: bool,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / name

    pipeline = rs.pipeline()
    config = rs.config()
    if serial:
        config.enable_device(serial)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    try:
        profile = pipeline.start(config)
    except RuntimeError as exc:
        devices = _list_devices()
        hint = f" Connected serials: {devices}" if devices else " No RealSense devices found."
        raise RuntimeError(f"Failed to start camera: {exc}.{hint}") from exc

    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())

    try:
        for _ in range(max(0, warmup)):
            pipeline.wait_for_frames()

        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("Incomplete frame set (missing depth or color).")

        depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.uint16)
        color_bgr = np.asanyarray(color_frame.get_data())
        color_rgb = color_bgr[:, :, ::-1]

        depth_mm = (depth_raw.astype(np.float32) * depth_scale * 1000.0).astype(np.uint16)

        color_path = base.with_suffix(".color.png")
        depth_path = base.with_suffix(".depth.png")
        depth_npy_path = base.with_suffix(".depth.npy")
        meta_path = base.with_suffix(".meta.json")

        Image.fromarray(color_rgb).save(color_path)
        Image.fromarray(depth_raw, mode="I;16").save(depth_path)
        np.save(depth_npy_path, depth_raw)

        depth_vis_path: Path | None = None
        if write_depth_vis:
            valid = depth_mm > 0
            vis = np.zeros(depth_mm.shape, dtype=np.uint8)
            if np.any(valid):
                lo, hi = int(depth_mm[valid].min()), int(depth_mm[valid].max())
                span = max(hi - lo, 1)
                vis[valid] = ((depth_mm[valid] - lo) * 255 / span).astype(np.uint8)
            depth_vis_path = base.with_suffix(".depth_vis.png")
            Image.fromarray(vis, mode="L").save(depth_vis_path)

        depth_intr = depth_frame.profile.as_video_stream_profile().intrinsics
        color_intr = color_frame.profile.as_video_stream_profile().intrinsics

        meta = {
            "captured_utc": datetime.now(timezone.utc).isoformat(),
            "device": {
                "name": device.get_info(rs.camera_info.name),
                "serial": device.get_info(rs.camera_info.serial_number),
                "firmware": device.get_info(rs.camera_info.firmware_version),
            },
            "streams": {
                "width": width,
                "height": height,
                "fps": fps,
            },
            "depth_scale_m_per_unit": depth_scale,
            "depth_stats_mm": _depth_stats_mm(depth_mm),
            "intrinsics": {
                "depth": _intrinsics_to_dict(depth_intr),
                "color": _intrinsics_to_dict(color_intr),
            },
            "files": {
                "color_png": color_path.name,
                "depth_png": depth_path.name,
                "depth_npy": depth_npy_path.name,
            },
        }
        if depth_vis_path is not None:
            meta["files"]["depth_vis_png"] = depth_vis_path.name

        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return base
    finally:
        pipeline.stop()


def main() -> None:
    args = parse_args()
    name = args.name.strip() or _default_name()
    t0 = time.perf_counter()

    try:
        base = capture_single_shot(
            out_dir=args.out_dir.resolve(),
            name=name,
            width=args.width,
            height=args.height,
            fps=args.fps,
            warmup=args.warmup,
            serial=args.serial.strip(),
            write_depth_vis=not args.no_depth_vis,
        )
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    out_dir = args.out_dir.resolve()
    print(f"Saved capture '{name}' to {out_dir}")
    print(f"  {name}.color.png")
    print(f"  {name}.depth.png   (16-bit raw depth units)")
    print(f"  {name}.depth.npy   (uint16 array, same as PNG)")
    if not args.no_depth_vis:
        print(f"  {name}.depth_vis.png")
    print(f"  {name}.meta.json")
    print(f"Done in {elapsed_ms:.0f} ms")


if __name__ == "__main__":
    main()
