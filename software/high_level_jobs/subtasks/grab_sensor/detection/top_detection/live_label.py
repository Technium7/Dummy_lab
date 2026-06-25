#!/usr/bin/env python3
"""
Interactive base-point labeling for top/overhead D415 view.

Usage (from software/):
  python -m high_level_jobs.subtasks.grab_sensor.detection.top_detection.live_label

Keys:
  Left-click        set base point (green cross)
  S / Enter         save frame + label
  C                 clear point
  D                 delete last saved sample
  Q / Esc           quit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..base_ellipse import draw_label_base
from ..capture import RealSenseStreamer, next_sample_id, save_base_labeled_sample
from ..cv2_gui import PANEL_H, init_cv2_window, require_cv2_gui
from .paths import DEFAULT_DATA_DIR


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


def _draw_ui(
    cv2,
    frame_rgb: np.ndarray,
    base_uv: tuple[int, int] | None,
    *,
    saved_count: int,
    status: str,
) -> np.ndarray:
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    _, w = bgr.shape[:2]

    if base_uv is not None:
        draw_label_base(bgr, base_uv[0], base_uv[1])

    panel = np.zeros((PANEL_H, w, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)
    lines = [
        "Left-click=base point  S=save  C=clear  D=undo  Q=quit",
        f"Saved: {saved_count}  |  {status}",
        "Top/overhead view — single base point (grasp XY uses this center).",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            panel,
            line,
            (10, 28 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46 if i == 0 else 0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
    return np.vstack([panel, bgr])


def run_live_label_gui(
    *,
    out_dir: Path,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    serial: str = "",
) -> int:
    cv2 = require_cv2_gui()
    out_dir.mkdir(parents=True, exist_ok=True)

    state: dict = {
        "base_uv": None,
        "saved_count": len(list((out_dir / "labels").glob("*.json")))
        if (out_dir / "labels").is_dir()
        else 0,
        "status": "Left-click sensor center / base point on table",
    }

    def on_mouse(event, x, y, _flags, _param):
        if y < PANEL_H:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            uv = (int(x), int(y - PANEL_H))
            state["base_uv"] = uv
            state["status"] = f"Base ({uv[0]}, {uv[1]}) — S to save or click again"

    window = "Top label (base point)"
    init_cv2_window(cv2, window, width, height + PANEL_H, on_mouse)

    streamer = RealSenseStreamer(width=width, height=height, fps=fps, serial=serial, warmup=5)
    print(f"Opening camera… Saving to {out_dir.resolve()}", flush=True)
    streamer.start()
    print("Camera ready.", flush=True)

    try:
        while True:
            frame = streamer.read(with_depth=False)
            display = _draw_ui(
                cv2,
                frame.color_rgb,
                state["base_uv"],
                saved_count=state["saved_count"],
                status=state["status"],
            )
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("c"), ord("C")):
                state["base_uv"] = None
                state["status"] = "Cleared — left-click base point"
            if key in (ord("d"), ord("D")):
                msg = _delete_last_sample(out_dir)
                state["saved_count"] = len(list((out_dir / "labels").glob("*.json")))
                state["status"] = msg
                print(msg)
            if key in (ord("s"), ord("S"), 13):
                if state["base_uv"] is None:
                    state["status"] = "Left-click base point before saving"
                    continue
                state["status"] = "Saving (capturing depth)…"
                cv2.imshow(
                    window,
                    _draw_ui(
                        cv2,
                        frame.color_rgb,
                        state["base_uv"],
                        saved_count=state["saved_count"],
                        status=state["status"],
                    ),
                )
                cv2.waitKey(1)
                save_frame = streamer.read(with_depth=True)
                sid = next_sample_id(out_dir)
                label_path = save_base_labeled_sample(
                    out_dir,
                    sid,
                    save_frame,
                    state["base_uv"],
                )
                state["saved_count"] += 1
                state["base_uv"] = None
                state["status"] = f"Saved {sid} — click next base point"
                print(f"Saved {label_path}")

    finally:
        streamer.stop()
        cv2.destroyAllWindows()

    print(f"Done. {state['saved_count']} samples in {out_dir.resolve()}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live top-view GUI: label base point and save training samples.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--serial", type=str, default="")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_live_label_gui(
        out_dir=args.out_dir.resolve(),
        width=args.width,
        height=args.height,
        fps=args.fps,
        serial=args.serial.strip(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
