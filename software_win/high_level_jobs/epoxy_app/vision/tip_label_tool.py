"""Click-label needle tip (x, y) into episode labels.json.

Mouse click = tip · a/d prev/next · [/] ±10 · x clear · w save · q quit

::

    python -m high_level_jobs.epoxy_app.vision.tip_label_tool --episode 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

_SOFTWARE_WIN = Path(__file__).resolve().parents[3]
if str(_SOFTWARE_WIN) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_WIN))

from high_level_jobs.epoxy_app.config_loader import load_epoxy_config
from high_level_jobs.epoxy_app.view_episode import resolve_episode
from high_level_jobs.epoxy_app.vision.dataset import list_zelux_stems, load_labels, save_labels


def _ensure_row(labels: dict, stem: str) -> dict:
    row = labels.get(stem)
    if row is None:
        row = {}
        labels[stem] = row
    return row


def set_tip(labels: dict, stem: str, x: float, y: float) -> None:
    row = _ensure_row(labels, stem)
    row["tip"] = {"x": float(x), "y": float(y)}


def clear_tip(labels: dict, stem: str) -> None:
    row = labels.get(stem)
    if row and "tip" in row:
        del row["tip"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Label needle tip on Zelux frames")
    parser.add_argument("--episode", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--results",
        action="store_true",
        help="Use dataset/record_result (apply runs) instead of training episodes/",
    )
    args = parser.parse_args(argv)

    epoxy_cfg = load_epoxy_config(args.config)
    dataset_dir = epoxy_cfg.record_result_dir if args.results else epoxy_cfg.dataset_dir
    episode_dir = resolve_episode(
        dataset_dir, args.episode, prefer_prefix="result" if args.results else "episode"
    )
    if episode_dir is None:
        print(f"Episode not found under {dataset_dir}: {args.episode}", file=sys.stderr)
        return 1

    stems = list_zelux_stems(episode_dir)
    if not stems:
        print(f"No *_zelux.jpg in {episode_dir}", file=sys.stderr)
        return 1

    labels = load_labels(episode_dir)
    idx = 0
    dirty = False
    win = "epoxy_tip_label"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    click = {"xy": None}

    def on_mouse(event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            click["xy"] = (float(x), float(y))

    cv2.setMouseCallback(win, on_mouse)

    print(f"Tip-labeling {episode_dir.name} ({len(stems)} Zelux frames)")
    print("click=tip  a/d nav  [/] ±10  x clear  w save  q quit")
    print("Sparse tips OK — training can interpolate between marked frames.")

    while True:
        stem = stems[idx]
        path = episode_dir / f"{stem}_zelux.jpg"
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            idx = min(idx + 1, len(stems) - 1)
            continue

        if click["xy"] is not None:
            set_tip(labels, stem, click["xy"][0], click["xy"][1])
            click["xy"] = None
            dirty = True

        row = labels.get(stem) or {}
        tip = row.get("tip")
        display = img.copy()
        if tip and "x" in tip and "y" in tip:
            tx, ty = int(round(float(tip["x"]))), int(round(float(tip["y"])))
            cv2.drawMarker(
                display, (tx, ty), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2
            )
            cv2.circle(display, (tx, ty), 8, (0, 255, 255), 2)

        tip_txt = f"tip=({tip['x']:.0f},{tip['y']:.0f})" if tip else "tip=NONE"
        lines = [
            f"{idx + 1}/{len(stems)}  {stem}_zelux.jpg",
            tip_txt,
            "click tip | a/d [/] x clear w q",
        ]
        for i, text in enumerate(lines):
            y = 24 + i * 22
            cv2.putText(display, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(display, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.imshow(win, display)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("d"), 83):
            idx = min(idx + 1, len(stems) - 1)
        elif key in (ord("a"), 81):
            idx = max(idx - 1, 0)
        elif key == ord("]"):
            idx = min(idx + 10, len(stems) - 1)
        elif key == ord("["):
            idx = max(idx - 10, 0)
        elif key == ord("x"):
            clear_tip(labels, stem)
            dirty = True
        elif key == ord("w"):
            save_labels(episode_dir, labels)
            dirty = False
            print(f"Wrote {episode_dir / 'labels.json'}")
        elif key in (ord("q"), 27):
            if dirty:
                save_labels(episode_dir, labels)
                print(f"Auto-saved {episode_dir / 'labels.json'}")
            break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
