"""Save a single signed-Δ image (curr − prev) for two Zelux frames.

Red = brighter in curr, blue = darker. Writes into epoxy_app/debug by default.

::

    python -m high_level_jobs.epoxy_app.vision.signed_delta --episode 13 --frames 101 102
    python -m high_level_jobs.epoxy_app.vision.signed_delta --results --episode 12 --frames 298 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_SOFTWARE_WIN = Path(__file__).resolve().parents[3]
if str(_SOFTWARE_WIN) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_WIN))

from high_level_jobs.epoxy_app.config_loader import load_epoxy_config
from high_level_jobs.epoxy_app.paths import DEBUG_DIR
from high_level_jobs.epoxy_app.view_episode import resolve_episode
from high_level_jobs.epoxy_app.vision.dataset import camera_frame_path, normalize_camera_role
from high_level_jobs.epoxy_app.vision.preprocess import bgr_to_gray


def _stem(raw: str | int) -> str:
    text = str(raw).strip()
    if text.isdigit():
        return f"{int(text):06d}"
    return text


def load_gray(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    gray = bgr_to_gray(bgr)
    if gray.ndim == 3:
        gray = gray[:, :, 0]
    return gray


def signed_delta_bgr(
    curr_gray: np.ndarray,
    prev_gray: np.ndarray,
    *,
    vmax: float | None = None,
    pclip: float = 99.5,
) -> tuple[np.ndarray, np.ndarray, float]:
    """RdBu_r signed map: red = curr brighter, blue = curr darker."""
    from matplotlib import cm
    from matplotlib.colors import Normalize

    signed = curr_gray.astype(np.float32) - prev_gray.astype(np.float32)
    if vmax is None:
        vmax = float(np.percentile(np.abs(signed), pclip))
    vmax = max(float(vmax), 1.0)
    rgba = cm.RdBu_r(Normalize(vmin=-vmax, vmax=vmax)(signed))
    rgb = (np.clip(rgba[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), signed, vmax


def default_out_path(
    episode_dir: Path,
    prev_stem: str,
    curr_stem: str,
    *,
    role: str,
    out_dir: Path = DEBUG_DIR,
) -> Path:
    return out_dir / f"{episode_dir.name}_{role}_{prev_stem}_to_{curr_stem}_signed_delta.png"


def save_signed_delta(
    episode_dir: Path,
    prev_stem: str,
    curr_stem: str,
    *,
    role: str = "zelux",
    out: Path | None = None,
    vmax: float | None = None,
    pclip: float = 99.5,
) -> Path:
    role = normalize_camera_role(role)
    prev_path = camera_frame_path(episode_dir, prev_stem, role)
    curr_path = camera_frame_path(episode_dir, curr_stem, role)
    prev = load_gray(prev_path)
    curr = load_gray(curr_path)
    vis, signed, used_vmax = signed_delta_bgr(curr, prev, vmax=vmax, pclip=pclip)
    dest = Path(out) if out is not None else default_out_path(
        episode_dir, prev_stem, curr_stem, role=role
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dest), vis):
        raise OSError(f"Failed to write {dest}")
    print(
        f"wrote {dest}  abs-delta mean={float(np.abs(signed).mean()):.2f}  "
        f"max={float(np.abs(signed).max()):.0f}  scale=+/-{used_vmax:.1f}"
    )
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Save signed (curr−prev) Zelux Δ as one PNG")
    parser.add_argument("--episode", required=True)
    parser.add_argument(
        "--frames",
        nargs=2,
        metavar=("PREV", "CURR"),
        required=True,
        help="Previous then current frame index or stem (signed = curr − prev)",
    )
    parser.add_argument("--results", action="store_true", help="Use dataset/record_result")
    parser.add_argument(
        "--camera",
        default="zelux",
        help="zelux/side (*_zelux.jpg) or zelux_front/front (*_zelux_front.jpg)",
    )
    parser.add_argument("--out", default=None, help="Output PNG (default: epoxy_app/debug/...)")
    parser.add_argument("--scale", type=float, default=None, help="Symmetric |Δ| clip (default: p99.5)")
    parser.add_argument("--epoxy-config", default=None)
    args = parser.parse_args(argv)

    epoxy = load_epoxy_config(args.epoxy_config)
    dataset_dir = epoxy.record_result_dir if args.results else epoxy.dataset_dir
    episode_dir = resolve_episode(
        dataset_dir, args.episode, prefer_prefix="result" if args.results else "episode"
    )
    if episode_dir is None:
        print(f"Episode not found under {dataset_dir}: {args.episode}", file=sys.stderr)
        return 1

    out = Path(args.out) if args.out else None
    save_signed_delta(
        episode_dir,
        _stem(args.frames[0]),
        _stem(args.frames[1]),
        role=args.camera,
        out=out,
        vmax=args.scale,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
