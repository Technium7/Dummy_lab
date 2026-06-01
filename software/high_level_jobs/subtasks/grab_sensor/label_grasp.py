#!/usr/bin/env python3
"""
Interactive grasp-point labeling for D415 RGB (live preview).

Click the grasp location on the live image, press S to save, adjust the scene, repeat.

Usage (from software/):
  python -m high_level_jobs.subtasks.grab_sensor.label_grasp
  python -m high_level_jobs.subtasks.grab_sensor.label_grasp --out-dir dataset/grasp_keypoints

Keys:
  Left click   set grasp point
  S / Enter    save frame + label (requires a click first)
  C            clear grasp point
  D            delete last saved sample
  Q / Esc      quit

Requires: pyrealsense2, opencv-python (with GUI — not headless)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .capture import RealSenseStreamer, next_sample_id, save_labeled_sample


def _require_cv2_gui():
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "opencv-python is required for the labeling GUI.\n"
            "Install with: pip uninstall opencv-python-headless -y && pip install opencv-python"
        ) from exc

    import re

    build = cv2.getBuildInformation()
    gui_match = re.search(r"^\s*GUI:\s*(\S+)", build, re.MULTILINE)
    if gui_match and gui_match.group(1).upper() == "NONE":
        raise SystemExit(
            "OpenCV was built without GUI support (opencv-python-headless).\n"
            "Windows needs the full package:\n"
            "  pip uninstall opencv-python-headless -y\n"
            "  pip install opencv-python"
        )

    try:
        cv2.namedWindow("__label_grasp_test__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__label_grasp_test__")
    except cv2.error as exc:
        raise SystemExit(
            "OpenCV cannot open windows on this machine.\n"
            "Try: pip uninstall opencv-python-headless -y && pip install opencv-python"
        ) from exc
    return cv2


PANEL_H = 110


def _draw_ui(
    cv2,
    frame_rgb: np.ndarray,
    grasp_uv: tuple[int, int] | None,
    *,
    saved_count: int,
    status: str,
) -> np.ndarray:
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    if grasp_uv is not None:
        u, v = grasp_uv
        cv2.drawMarker(bgr, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
        cv2.circle(bgr, (u, v), 12, (0, 0, 255), 2)
        cv2.circle(bgr, (u, v), 3, (0, 255, 255), -1)
        cv2.putText(
            bgr,
            f"grasp ({u}, {v})",
            (min(u + 14, w - 180), max(v - 14, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    panel_h = PANEL_H
    panel = np.zeros((panel_h, w, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)
    lines = [
        "Click grasp point  |  S or Enter = save  |  C = clear  |  D = undo last  |  Q = quit",
        f"Saved samples: {saved_count}  |  {status}",
        "Move/adjust the object or scene, click a new grasp point, save again.",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            panel,
            line,
            (10, 28 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55 if i == 0 else 0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    return np.vstack([panel, bgr])


def _delete_last_sample(out_dir: Path) -> str:
    labels_dir = out_dir / "labels"
    if not labels_dir.is_dir():
        return "Nothing to delete"
    labels = sorted(labels_dir.glob("*.json"))
    if not labels:
        return "Nothing to delete"

    last = labels[-1]
    data = json.loads(last.read_text(encoding="utf-8"))
    sid = data.get("id", last.stem)

    for key in ("image", "depth"):
        rel = data.get("files", {}).get(key)
        if rel:
            p = out_dir / rel
            if p.is_file():
                p.unlink()
    preview = out_dir / "previews" / f"{sid}.png"
    if preview.is_file():
        preview.unlink()
    last.unlink()

    manifest = out_dir / "manifest.jsonl"
    if manifest.is_file():
        lines = [ln for ln in manifest.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if lines:
            try:
                if json.loads(lines[-1]).get("id") == sid:
                    lines.pop()
            except json.JSONDecodeError:
                pass
        manifest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    return f"Deleted sample {sid}"


def run_label_gui(
    *,
    out_dir: Path,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    serial: str = "",
) -> int:
    cv2 = _require_cv2_gui()
    out_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "grasp_uv": None,
        "saved_count": len(list((out_dir / "labels").glob("*.json"))) if (out_dir / "labels").is_dir() else 0,
        "status": "Click on the grasp point",
        "frozen_rgb": None,
    }

    def on_mouse(event, x, y, _flags, _param):
        if y < PANEL_H:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            state["grasp_uv"] = (int(x), int(y - PANEL_H))
            state["status"] = f"Grasp at {state['grasp_uv']} — press S to save"

    window = "Grasp point labeling (live D415)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, width, height + PANEL_H)
    cv2.setMouseCallback(window, on_mouse)

    streamer = RealSenseStreamer(width=width, height=height, fps=fps, serial=serial, warmup=15)
    print(f"Opening camera… Saving to {out_dir.resolve()}")
    streamer.start()

    try:
        while True:
            frame = streamer.read()
            display = _draw_ui(
                cv2,
                frame.color_rgb,
                state["grasp_uv"],
                saved_count=state["saved_count"],
                status=state["status"],
            )
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("c"), ord("C")):
                state["grasp_uv"] = None
                state["status"] = "Click cleared"
            if key in (ord("d"), ord("D")):
                msg = _delete_last_sample(out_dir)
                state["saved_count"] = len(list((out_dir / "labels").glob("*.json")))
                state["status"] = msg
                print(msg)
            if key in (ord("s"), ord("S"), 13):
                if state["grasp_uv"] is None:
                    state["status"] = "Click a grasp point before saving"
                    continue
                sid = next_sample_id(out_dir)
                label_path = save_labeled_sample(
                    out_dir,
                    sid,
                    frame,
                    state["grasp_uv"],
                )
                state["saved_count"] += 1
                state["grasp_uv"] = None
                state["status"] = f"Saved {sid} — adjust scene and click again"
                print(f"Saved {label_path}")

    finally:
        streamer.stop()
        cv2.destroyAllWindows()

    print(f"Done. {state['saved_count']} samples in {out_dir.resolve()}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_out = Path(__file__).resolve().parent / "dataset" / "grasp_keypoints"
    p = argparse.ArgumentParser(description="Live D415 GUI: click grasp point and save training samples.")
    p.add_argument("--out-dir", type=Path, default=default_out, help="Dataset output directory")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--serial", type=str, default="", help="RealSense serial (default: first device)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_label_gui(
        out_dir=args.out_dir.resolve(),
        width=args.width,
        height=args.height,
        fps=args.fps,
        serial=args.serial.strip(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
