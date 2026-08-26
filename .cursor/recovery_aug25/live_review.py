#!/usr/bin/env python3
"""Quick browse of saved labels (view-only). For edits use ``vision.relabel``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_SOFTWARE_WIN = Path(__file__).resolve().parents[3]
if str(_SOFTWARE_WIN) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_WIN))

from high_level_jobs.grab_sensor.paths import DATASET_DIR  # noqa: E402
from high_level_jobs.grab_sensor.vision.dataset_io import iter_labels, load_label  # noqa: E402
from high_level_jobs.grab_sensor.vision.live_label import PANEL_H, _draw_object  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    import cv2

    p = argparse.ArgumentParser(
        description="Browse grab_sensor labels (view-only). Use vision.relabel to edit."
    )
    p.add_argument("--out", type=Path, default=DATASET_DIR)
    p.add_argument("--start", type=int, default=0)
    args = p.parse_args(argv)
    labels = iter_labels(args.out)
    if not labels:
        print(f"No labels in {args.out / 'labels'}")
        return 1

    idx = int(np.clip(args.start, 0, len(labels) - 1))
    window = "grab_sensor live_review"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print(f"Viewing {args.out.resolve()}  |  A/D navigate  |  E opens tip: use vision.relabel to edit")

    while True:
        path = labels[idx]
        data = load_label(path)
        img_rel = (data.get("files") or {}).get("image")
        if not img_rel:
            idx = (idx + 1) % len(labels)
            continue
        rgb = np.asarray(Image.open(args.out / img_rel).convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for obj in data.get("objects") or []:
            old = obj.get("contact_uv_before_shift")
            if old and len(old) >= 2:
                cv2.drawMarker(
                    bgr,
                    (int(old[0]), int(old[1])),
                    (160, 160, 160),
                    cv2.MARKER_TILTED_CROSS,
                    14,
                    1,
                )
            _draw_object(cv2, bgr, obj, show_holder_pt=False, show_grasp_pt=False)
        h, w = bgr.shape[:2]
        panel = np.zeros((PANEL_H, w, 3), dtype=np.uint8)
        panel[:] = (35, 35, 35)
        cv2.putText(
            panel,
            f"[{idx + 1}/{len(labels)}] {data.get('id')}  objects={len(data.get('objects') or [])}  "
            "gray=old contact  A/D=prev/next  Q=quit  (edit: vision.relabel)",
            (8, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
        )
        display = np.vstack([panel, bgr])
        cv2.imshow(window, display)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("a"), ord("A"), 81):
            idx = (idx - 1) % len(labels)
        if key in (ord("d"), ord("D"), 83):
            idx = (idx + 1) % len(labels)

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
