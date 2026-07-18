"""RealSense helpers for insert_sensor (overhead + side [+ optional side2]).

Recording can capture all configured roles. Diffusion may use side2 (optional /
zero-filled when missing). ACT keeps num_cameras=2 (overhead + side only).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Primary (overhead / table-calibrated) keeps legacy filenames for old episodes.
OVERHEAD_ROLE = "overhead"
SIDE_ROLE = "side"
# Extra oblique / opposite-side view (optional for diffusion; unused by ACT).
SIDE2_ROLE = "side2"
# Older dual-cam recordings used "wrist" — still loaded as the side view.
LEGACY_SIDE_ROLE = "wrist"
# Policy / train / deploy slots (order matters). Diffusion may use all three;
# ACT configs keep num_cameras=2 so side2 is ignored.
POLICY_CAMERA_ROLES = (OVERHEAD_ROLE, SIDE_ROLE, SIDE2_ROLE)
# All roles that may appear in recorded episodes.
CAMERA_ROLES = (OVERHEAD_ROLE, SIDE_ROLE, SIDE2_ROLE)
SECOND_CAM_ALIASES = (SIDE_ROLE, LEGACY_SIDE_ROLE)


@dataclass(frozen=True)
class CameraSpec:
    role: str
    serial: str = ""
    exposure_us: float | None = None  # None = freeze AE after warmup

    @property
    def is_primary(self) -> bool:
        return self.role == OVERHEAD_ROLE


def color_filename(stem: str, role: str = OVERHEAD_ROLE) -> str:
    if role == OVERHEAD_ROLE:
        return f"{stem}.color.png"
    return f"{stem}.{role}.color.png"


def depth_filename(stem: str, role: str = OVERHEAD_ROLE) -> str:
    if role == OVERHEAD_ROLE:
        return f"{stem}.depth.npy"
    return f"{stem}.{role}.depth.npy"


def load_rgbd_pair(
    ep_dir: Path,
    stem: str,
    role: str = OVERHEAD_ROLE,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    roles_to_try = (role,)
    if role in SECOND_CAM_ALIASES:
        # Prefer current name, then legacy wrist files.
        roles_to_try = (SIDE_ROLE, LEGACY_SIDE_ROLE) if role == SIDE_ROLE else (role, SIDE_ROLE)

    for r in roles_to_try:
        color_path = ep_dir / color_filename(stem, r)
        depth_path = ep_dir / depth_filename(stem, r)
        if color_path.is_file() and depth_path.is_file():
            color = np.asarray(Image.open(color_path).convert("RGB"))
            depth = np.load(depth_path).astype(np.float32)
            return color, depth
    return None, None


def episode_has_side(ep_dir: Path) -> bool:
    for role in SECOND_CAM_ALIASES:
        if any(ep_dir.glob(f"*.{role}.color.png")):
            return True
    return False


def episode_has_side2(ep_dir: Path) -> bool:
    return any(ep_dir.glob(f"*.{SIDE2_ROLE}.color.png"))


def episode_camera_roles(ep_dir: Path) -> list[str]:
    """Roles present on disk for this episode."""
    roles: list[str] = []
    for p in ep_dir.glob("*.color.png"):
        name = p.name
        if name.endswith(f".{SIDE_ROLE}.color.png"):
            continue
        if name.endswith(f".{SIDE2_ROLE}.color.png"):
            continue
        if name.endswith(f".{LEGACY_SIDE_ROLE}.color.png"):
            continue
        if name.endswith(".color.png"):
            roles.append(OVERHEAD_ROLE)
            break
    if episode_has_side(ep_dir):
        roles.append(SIDE_ROLE)
    if episode_has_side2(ep_dir):
        roles.append(SIDE2_ROLE)
    return roles


# Back-compat alias
episode_has_wrist = episode_has_side


def list_realsense_serials() -> list[str]:
    from high_level_jobs.subtasks.grab_sensor.detection.capture import _list_realsense_devices

    return list(_list_realsense_devices())


def resolve_camera_serials(
    cameras: list[CameraSpec],
    *,
    available: list[str] | None = None,
    require_pinned: bool = False,
) -> list[CameraSpec]:
    """
    Resolve camera serials.

    Prefer explicit pins in config. Auto-assign only when ``require_pinned`` is False
    and a serial is empty (USB order is not stable — prefer pinning).
    """
    serials = list(available) if available is not None else list_realsense_serials()
    used: set[str] = set()
    resolved: list[CameraSpec] = []
    for cam in cameras:
        sn = (cam.serial or "").strip()
        if not sn:
            if require_pinned:
                raise RuntimeError(
                    f"Camera '{cam.role}' has empty serial. Pin it in config "
                    "(USB enumeration order swaps roles). "
                    "Run: python -m high_level_jobs.subtasks.insert_sensor --preview-cameras"
                )
            pick = next((s for s in serials if s not in used), None)
            if pick is None:
                raise RuntimeError(
                    f"Need a RealSense for '{cam.role}', but only found {len(serials)} device(s): {serials}. "
                    "Plug in both cameras (separate USB ports from the robot)."
                )
            print(
                f"[insert_sensor] WARNING: auto-assigned '{cam.role}' -> {pick} "
                "(pin serials in config to avoid swaps)",
                flush=True,
            )
            sn = pick
        elif sn not in serials:
            raise RuntimeError(
                f"Camera '{cam.role}' serial {sn!r} not found. Connected: {serials}"
            )
        if sn in used:
            raise RuntimeError(f"Duplicate camera serial {sn!r} for role '{cam.role}'")
        used.add(sn)
        resolved.append(
            CameraSpec(role=cam.role, serial=sn, exposure_us=cam.exposure_us)
        )
    return resolved


def preview_cameras(
    *,
    out_dir: Path | None = None,
    serials: list[str] | None = None,
) -> Path:
    """
    Capture one RGB frame from each connected RealSense and save labeled previews.

    Files are named ``preview_<serial>.png`` plus a ``preview_index.txt`` mapping.
    Use this to decide which serial is overhead vs side, then pin them in config.
    """
    from high_level_jobs.subtasks.grab_sensor.detection.capture import RealSenseStreamer
    from PIL import Image

    out = (out_dir or Path("camera_previews")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    devices = list(serials) if serials is not None else list_realsense_serials()
    if not devices:
        raise RuntimeError("No RealSense devices found.")

    lines = [
        "Open each preview_*.png and decide roles:",
        "  overhead = top-down view of robot + table",
        "  side     = horizontal view across breadboard",
        "  side2    = additional side / oblique (diffusion optional; ACT unused)",
        "",
        "Then pin serials in default_insert_sensor.yaml:",
        "  cameras.overhead.serial / cameras.side.serial / cameras.side2.serial",
        "",
    ]
    for i, sn in enumerate(devices):
        streamer = None
        try:
            streamer = RealSenseStreamer(serial=sn, lock_exposure=True, warmup=5)
            streamer.start_retries = 1
            streamer.reset_on_start_retry = False
            streamer.start()
            frame = streamer.read()
            path = out / f"preview_{sn}.png"
            Image.fromarray(frame.color_rgb).save(path)
            lines.append(f"[{i}] serial={sn}")
            lines.append(f"    file={path}")
            lines.append("")
            print(f"[insert_sensor] Saved preview for {sn} -> {path}", flush=True)
        finally:
            if streamer is not None:
                try:
                    streamer.stop()
                except Exception:
                    pass

    index = out / "preview_index.txt"
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[insert_sensor] Index: {index}", flush=True)
    return out


def relabel_swapped_episode(
    ep_dir: Path,
    *,
    overhead_serial: str,
    side_serial: str,
    dry_run: bool = False,
) -> int:
    """
    Fix an episode where primary (``.color.png``) and second-cam files were swapped.

    After fix: ``.color.png`` / ``.depth.npy`` = overhead, ``.side.*`` = side.
    Legacy ``.wrist.*`` files are removed after migration.
    """
    ep_dir = ep_dir.resolve()
    frames = sorted(ep_dir.glob("*.frame.json"))
    if not frames:
        raise FileNotFoundError(f"No frames in {ep_dir}")

    n = 0
    for fp in frames:
        stem = fp.name.replace(".frame.json", "")
        primary_c = ep_dir / color_filename(stem, OVERHEAD_ROLE)
        primary_d = ep_dir / depth_filename(stem, OVERHEAD_ROLE)
        # Current second-cam file (misnamed wrist or side)
        second_c = None
        second_d = None
        for role in SECOND_CAM_ALIASES:
            cand_c = ep_dir / color_filename(stem, role)
            cand_d = ep_dir / depth_filename(stem, role)
            if cand_c.is_file() and cand_d.is_file():
                second_c, second_d = cand_c, cand_d
                break
        if not primary_c.is_file() or second_c is None or second_d is None:
            continue

        side_c = ep_dir / color_filename(stem, SIDE_ROLE)
        side_d = ep_dir / depth_filename(stem, SIDE_ROLE)
        tmp_c = ep_dir / f"{stem}._tmp.color.png"
        tmp_d = ep_dir / f"{stem}._tmp.depth.npy"

        if dry_run:
            n += 1
            continue

        # primary (wrong side view) -> temp; second (true overhead) -> primary;
        # temp -> side; drop legacy wrist names.
        primary_c.replace(tmp_c)
        primary_d.replace(tmp_d)
        second_c.replace(primary_c)
        second_d.replace(primary_d)
        tmp_c.replace(side_c)
        tmp_d.replace(side_d)
        # Clean leftover wrist files if second was already side
        for role in SECOND_CAM_ALIASES:
            if role == SIDE_ROLE:
                continue
            leftover_c = ep_dir / color_filename(stem, role)
            leftover_d = ep_dir / depth_filename(stem, role)
            if leftover_c.is_file():
                leftover_c.unlink()
            if leftover_d.is_file():
                leftover_d.unlink()

        data = json.loads(fp.read_text(encoding="utf-8"))
        data["cameras"] = [
            {"role": OVERHEAD_ROLE, "serial": overhead_serial, "t": data.get("t")},
            {"role": SIDE_ROLE, "serial": side_serial, "t": data.get("t")},
        ]
        fp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        n += 1

    meta_path = ep_dir / "meta.json"
    if meta_path.is_file() and not dry_run:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["num_cameras"] = 2
        meta["cameras"] = [
            {"role": OVERHEAD_ROLE, "serial": overhead_serial},
            {"role": SIDE_ROLE, "serial": side_serial},
        ]
        meta["cameras_relabeled"] = True
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return n


def make_streamer(
    serial: str,
    *,
    role: str = "",
    exposure_us: float | None = None,
) -> Any:
    from high_level_jobs.subtasks.grab_sensor.detection.capture import RealSenseStreamer

    streamer = RealSenseStreamer(
        serial=serial,
        lock_exposure=True,
        exposure_us=exposure_us,
        warmup=3,
    )
    streamer.frame_timeout_ms = 4000
    streamer.frame_retries = 2
    streamer.start_retries = 1
    streamer.reset_on_start_retry = False
    streamer.start()
    label = f" ({role})" if role else ""
    locked = getattr(streamer, "color_exposure_locked", None) or {}
    exp = locked.get("exposure_us")
    exp_s = f"{exp:.0f} us" if exp is not None else "AE freeze"
    print(
        f"[insert_sensor] Camera ready{label}: serial={serial or '(default)'} "
        f"exposure={exp_s}",
        flush=True,
    )
    return streamer


def stop_streamer(streamer: Any | None, *, role: str = "") -> None:
    if streamer is None:
        return
    try:
        streamer.stop()
    except Exception:
        pass
    label = f" ({role})" if role else ""
    print(f"[insert_sensor] Camera released{label}.", flush=True)
