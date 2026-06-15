#!/usr/bin/env python3
"""
Label sensor base points on existing grasp_keypoints dataset images.

The grasp point is already labeled (shown in red). Click the sensor base
(the part that sits on the table) in green, then save.

Usage (from software/):
  python -m high_level_jobs.subtasks.grab_sensor.label_base
  python -m high_level_jobs.subtasks.grab_sensor.label_base --data-dir dataset/grasp_keypoints

Keys:
  Left click   set sensor base point
  S / Enter    save base label and advance
  N            next image (without saving)
  P            previous image
  C            clear base click
  U            jump to first unlabeled sample
  Q / Esc      quit

Requires: opencv-python (with GUI — not headless)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .capture import update_sample_base_label
from .grasp_point_net import list_labeled_samples
from .label_grasp import PANEL_H, _require_cv2_gui


def _has_base_label(label_path: Path) -> bool:
    data = json.loads(label_path.read_text(encoding="utf-8"))
    return "base_u" in data and "base_v" in data


def _load_sample_entries(data_dir: Path) -> list[dict]:
    entries: list[dict] = []
    for sample in list_labeled_samples(data_dir):
        label_path = sample.get("label_path") or (data_dir / "labels" / f"{sample['id']}.json")
        rgb = np.asarray(Image.open(sample["image_path"]).convert("RGB"))
        entries.append(
            {
                "id": sample["id"],
                "label_path": Path(label_path),
                "image_rgb": rgb,
                "grasp_uv": sample["grasp_uv"],
                "base_uv": sample.get("base_uv"),
                "has_base": sample.get("base_uv") is not None,
            }
        )
    return entries


def _draw_label_ui(
    cv2,
    frame_rgb: np.ndarray,
    grasp_uv: tuple[int, int],
    base_uv: tuple[int, int] | None,
    *,
    sample_id: str,
    index: int,
    total: int,
    labeled_count: int,
    status: str,
) -> np.ndarray:
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    gu, gv = grasp_uv
    cv2.drawMarker(bgr, (gu, gv), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
    cv2.circle(bgr, (gu, gv), 12, (0, 0, 255), 2)
    cv2.putText(
        bgr,
        f"grasp ({gu}, {gv})",
        (min(gu + 14, w - 200), max(gv - 14, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )

    if base_uv is not None:
        bu, bv = base_uv
        cv2.drawMarker(bgr, (bu, bv), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 22, 2)
        cv2.circle(bgr, (bu, bv), 12, (0, 255, 0), 2)
        cv2.putText(
            bgr,
            f"base ({bu}, {bv})",
            (min(bu + 14, w - 200), min(bv + 28, h - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    panel = np.zeros((PANEL_H, w, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)
    lines = [
        "Click sensor BASE (green)  |  S/Enter=save  |  N/P=nav  |  U=unlabeled  |  Q=quit",
        f"{sample_id}  [{index + 1}/{total}]  base labeled: {labeled_count}/{total}  |  {status}",
        "Red = existing grasp point. Green = sensor base on the table.",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            panel,
            line,
            (10, 28 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52 if i == 0 else 0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
    return np.vstack([panel, bgr])


def run_label_base_gui(
    *,
    data_dir: Path,
    start_unlabeled: bool = True,
) -> int:
    cv2 = _require_cv2_gui()
    data_dir = data_dir.resolve()

    entries = _load_sample_entries(data_dir)
    if not entries:
        print(f"No labeled grasp samples found in {data_dir}")
        print("Collect grasp labels first: python -m high_level_jobs.subtasks.grab_sensor.label_grasp")
        return 1

    idx = 0
    if start_unlabeled:
        for i, entry in enumerate(entries):
            if not entry["has_base"]:
                idx = i
                break

    state = {
        "idx": idx,
        "base_uv": entries[idx].get("base_uv"),
        "status": "Click the sensor base (table contact point)",
    }

    def labeled_count() -> int:
        return sum(1 for e in entries if e["has_base"])

    def refresh_base_from_entry() -> None:
        entry = entries[state["idx"]]
        state["base_uv"] = entry.get("base_uv")

    def on_mouse(event, x, y, _flags, _param):
        if y < PANEL_H:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            state["base_uv"] = (int(x), int(y - PANEL_H))
            state["status"] = f"Base at {state['base_uv']} — press S to save"

    window = "Sensor base labeling (dataset)"
    entry0 = entries[idx]
    h, w = entry0["image_rgb"].shape[:2]
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, w, h + PANEL_H)
    cv2.setMouseCallback(window, on_mouse)

    print(f"Labeling sensor bases in {data_dir}")
    print(f"  {len(entries)} grasp samples, {labeled_count()} already have base labels")

    try:
        while True:
            entry = entries[state["idx"]]
            display = _draw_label_ui(
                cv2,
                entry["image_rgb"],
                entry["grasp_uv"],
                state["base_uv"],
                sample_id=entry["id"],
                index=state["idx"],
                total=len(entries),
                labeled_count=labeled_count(),
                status=state["status"],
            )
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            if key == 255:
                continue
            if key in (ord("c"), ord("C")):
                state["base_uv"] = None
                state["status"] = "Base click cleared"
            if key in (ord("n"), ord("N")):
                state["idx"] = min(len(entries) - 1, state["idx"] + 1)
                refresh_base_from_entry()
                state["status"] = f"Sample {entries[state['idx']]['id']}"
            if key in (ord("p"), ord("P")):
                state["idx"] = max(0, state["idx"] - 1)
                refresh_base_from_entry()
                state["status"] = f"Sample {entries[state['idx']]['id']}"
            if key in (ord("u"), ord("U")):
                for i, e in enumerate(entries):
                    if not e["has_base"]:
                        state["idx"] = i
                        refresh_base_from_entry()
                        state["status"] = f"Jumped to unlabeled {e['id']}"
                        break
                else:
                    state["status"] = "All samples already have base labels"
            if key in (ord("s"), ord("S"), 13):
                if state["base_uv"] is None:
                    state["status"] = "Click the sensor base before saving"
                    continue
                label_path = update_sample_base_label(
                    entry["label_path"],
                    state["base_uv"],
                    data_dir=data_dir,
                )
                entry["base_uv"] = state["base_uv"]
                entry["has_base"] = True
                state["status"] = f"Saved base for {entry['id']}"
                print(f"Saved {label_path}")
                if state["idx"] < len(entries) - 1:
                    state["idx"] += 1
                    refresh_base_from_entry()
                    if state["base_uv"] is None:
                        state["status"] = "Click sensor base on next image"
    finally:
        cv2.destroyAllWindows()

    print(f"Done. {labeled_count()}/{len(entries)} samples have base labels.")
    return 0 if labeled_count() >= 2 else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_data = Path(__file__).resolve().parent / "dataset" / "grasp_keypoints"
    p = argparse.ArgumentParser(description="Label sensor base points on existing grasp dataset images.")
    p.add_argument("--data-dir", type=Path, default=default_data, help="grasp_keypoints dataset directory")
    p.add_argument(
        "--start-first",
        action="store_true",
        help="Start at first sample instead of first unlabeled",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_label_base_gui(
        data_dir=args.data_dir.resolve(),
        start_unlabeled=not args.start_first,
    )


if __name__ == "__main__":
    raise SystemExit(main())
