#!/usr/bin/env python3
"""
Offline relabel base points on saved top-view images (no camera).

Usage (from software/):
  python -m high_level_jobs.subtasks.grab_sensor.detection.top_detection.relabel

Keys:
  Left-click        set / move base point
  S / Enter         save and advance
  N / P             next / previous image
  C                 clear point
  Q / Esc           quit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from ..base_ellipse import draw_label_base
from ..capture import update_sample_base_point
from ..cv2_gui import PANEL_H, init_cv2_window, require_cv2_gui
from ..grasp_point_net import list_labeled_samples
from .paths import DEFAULT_DATA_DIR


def _load_sample_entries(data_dir: Path) -> list[dict]:
    entries: list[dict] = []
    for sample in list_labeled_samples(data_dir, require_base=True, base_only=True):
        label_path = Path(sample.get("label_path") or (data_dir / "labels" / f"{sample['id']}.json"))
        rgb = np.asarray(Image.open(sample["image_path"]).convert("RGB"))
        entries.append(
            {
                "id": sample["id"],
                "label_path": label_path,
                "image_rgb": rgb,
                "base_uv": sample["base_uv"],
            }
        )
    return entries


def _draw_ui(
    cv2,
    frame_rgb: np.ndarray,
    base_uv: tuple[int, int] | None,
    *,
    sample_id: str,
    index: int,
    total: int,
    status: str,
) -> np.ndarray:
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    _, w = bgr.shape[:2]

    if base_uv is not None:
        draw_label_base(bgr, base_uv[0], base_uv[1])

    panel = np.zeros((PANEL_H, w, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)
    lines = [
        "Left-click=base point  S=save  N/P=nav  C=clear  Q=quit",
        f"{sample_id}  [{index + 1}/{total}]  |  {status}",
        "Offline top-view relabel — green cross = base center.",
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


def run_relabel_gui(*, data_dir: Path, start_index: int = 0) -> int:
    cv2 = require_cv2_gui()
    data_dir = data_dir.resolve()

    entries = _load_sample_entries(data_dir)
    if not entries:
        print(f"No base point labels found in {data_dir}")
        print("Collect first: python -m high_level_jobs.subtasks.grab_sensor.detection.top_detection.live_label")
        return 1

    state = {
        "idx": max(0, min(start_index, len(entries) - 1)),
        "base_uv": entries[max(0, min(start_index, len(entries) - 1))]["base_uv"],
        "status": "Left-click to move base point, S to save",
    }

    def on_mouse(event, x, y, _flags, _param):
        if y < PANEL_H:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            uv = (int(x), int(y - PANEL_H))
            state["base_uv"] = uv
            state["status"] = f"Base ({uv[0]}, {uv[1]}) — S to save"

    window = "Top relabel (offline base point)"
    h, w = entries[0]["image_rgb"].shape[:2]
    init_cv2_window(cv2, window, w, h + PANEL_H, on_mouse)

    print(f"Offline top relabel: {data_dir}")
    print(f"  {len(entries)} samples")

    try:
        while True:
            entry = entries[state["idx"]]
            display = _draw_ui(
                cv2,
                entry["image_rgb"],
                state["base_uv"],
                sample_id=entry["id"],
                index=state["idx"],
                total=len(entries),
                status=state["status"],
            )
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("c"), ord("C")):
                state["base_uv"] = None
                state["status"] = "Cleared — left-click base point"
            if key in (ord("n"), ord("N")):
                state["idx"] = min(len(entries) - 1, state["idx"] + 1)
                state["base_uv"] = entries[state["idx"]]["base_uv"]
                state["status"] = f"Sample {entries[state['idx']]['id']}"
            if key in (ord("p"), ord("P")):
                state["idx"] = max(0, state["idx"] - 1)
                state["base_uv"] = entries[state["idx"]]["base_uv"]
                state["status"] = f"Sample {entries[state['idx']]['id']}"
            if key in (ord("s"), ord("S"), 13):
                if state["base_uv"] is None:
                    state["status"] = "Left-click base point before saving"
                    continue
                label_path = update_sample_base_point(
                    entry["label_path"],
                    state["base_uv"],
                    data_dir=data_dir,
                )
                entry["base_uv"] = state["base_uv"]
                state["status"] = f"Saved {entry['id']}"
                print(f"Saved {label_path}")
                if state["idx"] < len(entries) - 1:
                    state["idx"] += 1
                    state["base_uv"] = entries[state["idx"]]["base_uv"]
                    state["status"] = "Relabel next image"
    finally:
        cv2.destroyAllWindows()

    print(f"Done. {len(entries)} samples in {data_dir}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline relabel: base point on saved top-view images.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--start-index", type=int, default=0, help="Start at this sample index (0-based)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_relabel_gui(data_dir=args.data_dir.resolve(), start_index=args.start_index)


if __name__ == "__main__":
    raise SystemExit(main())
