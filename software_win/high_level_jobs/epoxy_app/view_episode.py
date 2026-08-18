#!/usr/bin/env python3
"""Episode player for epoxy demos (Zelux front / side / Infinity, one control bar).

Usage::

  python software_win/high_level_jobs/epoxy_app/view_episode.py
  python software_win/high_level_jobs/epoxy_app/view_episode.py --episode episode_000001
  python software_win/high_level_jobs/epoxy_app/view_episode.py --episode 1 --play
  python software_win/high_level_jobs/epoxy_app/view_episode.py --episode 1 --export-zelux
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SOFTWARE_WIN = Path(__file__).resolve().parents[2]
if str(_SOFTWARE_WIN) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_WIN))

from PIL import Image, ImageTk

from high_level_jobs.epoxy_app.config_loader import load_epoxy_config

_ROLES = ("zelux", "zelux_front", "infinity")
_ROLE_ALIASES = {"zelux_front": ("zelux_side",)}
_ROLE_TITLES = {
    "zelux": "ZELUX SIDE",
    "zelux_front": "ZELUX FRONT",
    "infinity": "INFINITY",
}
_PREVIEW_MAX = 720
_SPEEDS = (0.25, 0.5, 1.0, 1.5, 2.0, 4.0)


def _color_filename(stem: str, role: str) -> str:
    return f"{stem}_{role}.jpg"


@dataclass(frozen=True)
class FrameRef:
    index: int
    stem: str
    t: float
    paths: dict[str, Path]
    gripper: float
    joint_pos_deg: list[float]


def list_episodes(dataset_dir: Path) -> list[Path]:
    """List demo (episode_*) and apply-result (result_*) folders."""
    out: list[Path] = []
    for pattern in ("episode_*", "result_*"):
        out.extend(p for p in dataset_dir.glob(pattern) if p.is_dir())
    return sorted(out, key=lambda p: p.name)


def resolve_episode(
    dataset_dir: Path,
    raw: str | Path | None,
    *,
    prefer_prefix: str | None = None,
) -> Path | None:
    if raw is None:
        episodes = list_episodes(dataset_dir)
        return episodes[0] if episodes else None
    path = Path(raw)
    if path.is_dir():
        return path.resolve()
    named = dataset_dir / str(raw)
    if named.is_dir():
        return named.resolve()
    try:
        n = int(str(raw))
    except ValueError:
        return None
    prefixes = ("episode", "result")
    if prefer_prefix == "result":
        prefixes = ("result", "episode")
    elif prefer_prefix == "episode":
        prefixes = ("episode", "result")
    for prefix in prefixes:
        candidate = dataset_dir / f"{prefix}_{n:06d}"
        if candidate.is_dir():
            return candidate.resolve()
    return None


def resolve_episode_in_roots(
    roots: list[Path],
    raw: str | Path | None,
    *,
    prefer_prefix: str | None = None,
) -> Path | None:
    """Find an episode under the first matching dataset root."""
    for root in roots:
        if not root.is_dir():
            continue
        found = resolve_episode(root, raw, prefer_prefix=prefer_prefix)
        if found is not None:
            return found
    return None
def load_frames(episode_dir: Path) -> tuple[list[FrameRef], float]:
    meta_path = episode_dir / "meta.json"
    record_hz = 15.0
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        record_hz = float(meta.get("record_hz", record_hz) or record_hz)

    frames: list[FrameRef] = []
    for json_path in sorted(episode_dir.glob("*.frame.json")):
        stem = json_path.name[: -len(".frame.json")]
        payload: dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))
        paths: dict[str, Path] = {}
        for role in _ROLES:
            img = episode_dir / _color_filename(stem, role)
            if not img.is_file():
                for alias in _ROLE_ALIASES.get(role, ()):
                    alt = episode_dir / _color_filename(stem, alias)
                    if alt.is_file():
                        img = alt
                        break
            if img.is_file():
                paths[role] = img
        if len(paths) < 2:
            continue
        frames.append(
            FrameRef(
                index=len(frames),
                stem=stem,
                t=float(payload.get("t", 0.0)),
                paths=paths,
                gripper=float(payload.get("gripper", 0.0)),
                joint_pos_deg=[float(v) for v in (payload.get("joint_pos_deg") or [])],
            )
        )
    return frames, record_hz


def _fit_image(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    w, h = img.size
    scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
    if scale >= 0.999:
        return img
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR)


def export_role_video(
    episode_dir: Path,
    *,
    role: str = "zelux",
    output: Path | None = None,
    fps: float | None = None,
) -> Path:
    """Write one camera stream from an episode to an MP4 (OpenCV)."""
    import cv2
    import numpy as np

    frames, record_hz = load_frames(episode_dir)
    paths = [f.paths[role] for f in frames if role in f.paths]
    if not paths:
        raise FileNotFoundError(
            f"No {role} frames in {episode_dir} (need paired dual-camera frames)"
        )

    out = (
        output.resolve()
        if output is not None
        else (episode_dir / f"{episode_dir.name}_{role}.mp4").resolve()
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    first = cv2.imread(str(paths[0]))
    if first is None:
        raise RuntimeError(f"Failed to read {paths[0]}")
    height, width = first.shape[:2]
    write_fps = float(fps if fps is not None else record_hz)
    write_fps = max(write_fps, 1.0)

    # mp4v is the most reliable OpenCV default on Windows.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, write_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out}")

    try:
        for i, path in enumerate(paths):
            bgr = first if i == 0 else cv2.imread(str(path))
            if bgr is None:
                raise RuntimeError(f"Failed to read {path}")
            if bgr.shape[0] != height or bgr.shape[1] != width:
                bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
            if bgr.dtype != np.uint8:
                bgr = bgr.astype(np.uint8)
            writer.write(bgr)
    finally:
        writer.release()
    return out


class EpisodeViewer:
    def __init__(
        self,
        dataset_dir: Path,
        episode_dir: Path,
        *,
        autoplay: bool = False,
    ) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.dataset_dir = dataset_dir.resolve()
        self.episode_dir = episode_dir.resolve()
        self.frames: list[FrameRef] = []
        self.record_hz = 15.0
        self.index = 0
        self.playing = False
        self.loop = True
        self.speed = 1.0
        self._photo: dict[str, Any] = {}
        self._play_job: str | None = None
        self._scrubbing = False
        self._was_playing_before_scrub = False

        self.root = tk.Tk()
        self.root.title("Epoxy episode viewer")
        self.root.minsize(960, 560)
        self.root.geometry("1280x780")

        self.episode_var = tk.StringVar()
        self.status_var = tk.StringVar(value="")
        self.frame_var = tk.StringVar(value="0 / 0")
        self.time_var = tk.StringVar(value="0.00 s")
        self.meta_var = tk.StringVar(value="")
        self.play_btn_var = tk.StringVar(value="Play")
        self.speed_var = tk.StringVar(value="1.0x")
        self.loop_var = tk.BooleanVar(value=True)

        self._build_ui()
        self._bind_keys()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._reload_episode_list(select=self.episode_dir.name)
        self.load_episode(self.episode_dir)
        if autoplay and self.frames:
            self.toggle_play()

    def _build_ui(self) -> None:
        tk = self.tk
        ttk = self.ttk

        top = ttk.Frame(self.root, padding=(10, 8, 10, 4))
        top.pack(fill="x")
        ttk.Label(top, text="Episode").pack(side="left")
        self.episode_combo = ttk.Combobox(
            top,
            textvariable=self.episode_var,
            state="readonly",
            width=28,
        )
        self.episode_combo.pack(side="left", padx=(6, 8))
        self.episode_combo.bind("<<ComboboxSelected>>", self._on_episode_chosen)
        ttk.Button(top, text="Reload", command=self._reload_current).pack(side="left")
        ttk.Button(top, text="Save Zelux video", command=self._export_zelux).pack(
            side="left", padx=(8, 0)
        )
        ttk.Label(top, textvariable=self.status_var).pack(side="left", padx=(12, 0))

        views = ttk.Frame(self.root, padding=(10, 4))
        views.pack(fill="both", expand=True)
        views.columnconfigure(0, weight=1)
        views.columnconfigure(1, weight=1)
        views.columnconfigure(2, weight=1)
        views.rowconfigure(1, weight=1)

        self.view_labels: dict[str, Any] = {}
        for col, role in enumerate(_ROLES):
            ttk.Label(
                views, text=_ROLE_TITLES.get(role, role.upper()), font=("Segoe UI", 11, "bold")
            ).grid(row=0, column=col, sticky="w", padx=(4, 4))
            panel = tk.Label(
                views,
                background="#111111",
                foreground="#dddddd",
                text=f"No {role} frame",
                anchor="center",
            )
            panel.grid(row=1, column=col, sticky="nsew", padx=4, pady=4)
            self.view_labels[role] = panel

        bar = ttk.Frame(self.root, padding=(10, 6, 10, 10))
        bar.pack(fill="x")

        controls = ttk.Frame(bar)
        controls.pack(fill="x")
        ttk.Button(controls, text="|<<", width=4, command=lambda: self.seek(0)).pack(
            side="left"
        )
        ttk.Button(controls, text="<", width=3, command=self.step_prev).pack(
            side="left", padx=(4, 0)
        )
        ttk.Button(
            controls, textvariable=self.play_btn_var, width=8, command=self.toggle_play
        ).pack(side="left", padx=6)
        ttk.Button(controls, text=">", width=3, command=self.step_next).pack(side="left")
        ttk.Button(controls, text=">>|", width=4, command=self.seek_end).pack(
            side="left", padx=(4, 0)
        )

        ttk.Label(controls, text="Speed").pack(side="left", padx=(16, 4))
        self.speed_combo = ttk.Combobox(
            controls,
            textvariable=self.speed_var,
            values=[f"{s}x" for s in _SPEEDS],
            state="readonly",
            width=6,
        )
        self.speed_combo.pack(side="left")
        self.speed_combo.bind("<<ComboboxSelected>>", self._on_speed_chosen)

        ttk.Checkbutton(
            controls,
            text="Loop",
            variable=self.loop_var,
            command=self._on_loop_toggle,
        ).pack(side="left", padx=(12, 0))

        ttk.Label(controls, textvariable=self.frame_var, width=14).pack(side="right")
        ttk.Label(controls, textvariable=self.time_var, width=10).pack(
            side="right", padx=(0, 8)
        )

        scrub = ttk.Frame(bar)
        scrub.pack(fill="x", pady=(8, 0))
        self.scrub = ttk.Scale(
            scrub,
            from_=0,
            to=0,
            orient="horizontal",
            command=self._on_scrub,
        )
        self.scrub.pack(fill="x")
        self.scrub.bind("<ButtonPress-1>", self._scrub_press)
        self.scrub.bind("<ButtonRelease-1>", self._scrub_release)

        ttk.Label(bar, textvariable=self.meta_var, foreground="#444444").pack(
            anchor="w", pady=(6, 0)
        )
        ttk.Label(
            bar,
            text=(
                "Space play/pause  |  Left/Right step  |  Home/End  |  "
                "[ ] speed  |  L loop"
            ),
            foreground="#666666",
        ).pack(anchor="w", pady=(2, 0))

    def _bind_keys(self) -> None:
        root = self.root
        root.bind("<space>", lambda _e: self.toggle_play())
        root.bind("<Left>", lambda _e: self.step_prev())
        root.bind("<Right>", lambda _e: self.step_next())
        root.bind("<Home>", lambda _e: self.seek(0))
        root.bind("<End>", lambda _e: self.seek_end())
        root.bind("<Key-l>", lambda _e: self._toggle_loop_key())
        root.bind("<Key-L>", lambda _e: self._toggle_loop_key())
        root.bind("<bracketleft>", lambda _e: self._nudge_speed(-1))
        root.bind("<bracketright>", lambda _e: self._nudge_speed(1))
        root.bind("<Escape>", lambda _e: self._on_close())

    def _reload_episode_list(self, select: str | None = None) -> None:
        names = [p.name for p in list_episodes(self.dataset_dir)]
        self.episode_combo["values"] = names
        if select and select in names:
            self.episode_var.set(select)
        elif names and not self.episode_var.get():
            self.episode_var.set(names[0])

    def _on_episode_chosen(self, _event: object | None = None) -> None:
        name = self.episode_var.get().strip()
        if not name:
            return
        path = self.dataset_dir / name
        if path.is_dir():
            was_playing = self.playing
            self.pause()
            self.load_episode(path)
            if was_playing and self.frames:
                self.toggle_play()

    def _reload_current(self) -> None:
        was_playing = self.playing
        idx = self.index
        self.pause()
        self._reload_episode_list(select=self.episode_dir.name)
        self.load_episode(self.episode_dir)
        if self.frames:
            self.seek(min(idx, len(self.frames) - 1))
        if was_playing and self.frames:
            self.toggle_play()

    def load_episode(self, episode_dir: Path) -> None:
        self.episode_dir = episode_dir.resolve()
        self.frames, self.record_hz = load_frames(self.episode_dir)
        self.episode_var.set(self.episode_dir.name)
        self.index = 0
        n = max(len(self.frames) - 1, 0)
        self.scrub.configure(to=n)
        if not self.frames:
            self.status_var.set(
                f"{self.episode_dir.name}: no complete dual-camera frames"
            )
            self.frame_var.set("0 / 0")
            self.time_var.set("0.00 s")
            self.meta_var.set("")
            for role, label in self.view_labels.items():
                label.configure(image="", text=f"No {role} frame")
                self._photo.pop(role, None)
            return
        self.status_var.set(
            f"{self.episode_dir.name}  ·  {len(self.frames)} frames  ·  "
            f"{self.record_hz:g} Hz"
        )
        self.show_frame(0)

    def show_frame(self, index: int) -> None:
        if not self.frames:
            return
        self.index = max(0, min(int(index), len(self.frames) - 1))
        frame = self.frames[self.index]

        for role in _ROLES:
            panel = self.view_labels[role]
            max_w = max(panel.winfo_width(), 320)
            max_h = max(panel.winfo_height(), 240)
            max_w = min(max_w, _PREVIEW_MAX * 2)
            max_h = min(max_h, _PREVIEW_MAX * 2)
            img = Image.open(frame.paths[role]).convert("RGB")
            fitted = _fit_image(img, max_w, max_h)
            photo = ImageTk.PhotoImage(fitted)
            self._photo[role] = photo
            panel.configure(image=photo, text="")

        total = len(self.frames)
        t0 = self.frames[0].t
        elapsed = max(0.0, frame.t - t0)
        self.frame_var.set(f"{self.index + 1} / {total}")
        self.time_var.set(f"{elapsed:.2f} s")
        joints = ", ".join(f"{v:.1f}" for v in frame.joint_pos_deg[:6]) or "-"
        self.meta_var.set(f"gripper={frame.gripper:+.2f}   joints_deg=[{joints}]")
        if not self._scrubbing:
            self.scrub.set(self.index)

    def seek(self, index: int) -> None:
        self.show_frame(index)

    def seek_end(self) -> None:
        if self.frames:
            self.seek(len(self.frames) - 1)

    def step_prev(self) -> None:
        self.pause()
        self.seek(self.index - 1)

    def step_next(self) -> None:
        self.pause()
        if self.index >= len(self.frames) - 1:
            if self.loop and self.frames:
                self.seek(0)
            return
        self.seek(self.index + 1)

    def toggle_play(self) -> None:
        if self.playing:
            self.pause()
        else:
            self.play()

    def play(self) -> None:
        if not self.frames:
            return
        if self.index >= len(self.frames) - 1:
            self.seek(0)
        self.playing = True
        self.play_btn_var.set("Pause")
        self._schedule_tick()

    def pause(self) -> None:
        self.playing = False
        self.play_btn_var.set("Play")
        if self._play_job is not None:
            try:
                self.root.after_cancel(self._play_job)
            except Exception:
                pass
            self._play_job = None

    def _schedule_tick(self) -> None:
        if not self.playing:
            return
        period_ms = max(8, int(1000.0 / max(self.record_hz * self.speed, 0.1)))
        self._play_job = self.root.after(period_ms, self._tick)

    def _tick(self) -> None:
        self._play_job = None
        if not self.playing or not self.frames:
            return
        if self.index >= len(self.frames) - 1:
            if self.loop:
                self.seek(0)
                self._schedule_tick()
            else:
                self.pause()
            return
        self.seek(self.index + 1)
        self._schedule_tick()

    def _on_scrub(self, value: str) -> None:
        if not self.frames:
            return
        idx = int(float(value) + 0.5)
        if idx != self.index:
            self.show_frame(idx)

    def _scrub_press(self, _event: object) -> None:
        self._scrubbing = True
        self._was_playing_before_scrub = self.playing
        if self.playing:
            self.pause()

    def _scrub_release(self, _event: object) -> None:
        self._scrubbing = False
        self.show_frame(int(float(self.scrub.get()) + 0.5))
        if self._was_playing_before_scrub:
            self.play()

    def _on_speed_chosen(self, _event: object | None = None) -> None:
        raw = self.speed_var.get().rstrip("x")
        try:
            self.speed = float(raw)
        except ValueError:
            self.speed = 1.0
            self.speed_var.set("1.0x")

    def _nudge_speed(self, direction: int) -> None:
        try:
            i = _SPEEDS.index(self.speed)
        except ValueError:
            i = _SPEEDS.index(1.0)
        i = max(0, min(len(_SPEEDS) - 1, i + direction))
        self.speed = _SPEEDS[i]
        self.speed_var.set(f"{self.speed}x")

    def _on_loop_toggle(self) -> None:
        self.loop = bool(self.loop_var.get())

    def _toggle_loop_key(self) -> None:
        self.loop_var.set(not self.loop_var.get())
        self._on_loop_toggle()

    def _export_zelux(self) -> None:
        from tkinter import messagebox

        if not self.frames:
            messagebox.showwarning("Save Zelux video", "No frames loaded.")
            return
        was_playing = self.playing
        self.pause()
        self.status_var.set(f"Writing Zelux video for {self.episode_dir.name}...")
        self.root.update_idletasks()
        try:
            out = export_role_video(
                self.episode_dir, role="zelux", fps=self.record_hz
            )
        except Exception as exc:
            self.status_var.set(f"Export failed: {exc}")
            messagebox.showerror("Save Zelux video", str(exc))
            if was_playing:
                self.play()
            return
        self.status_var.set(f"Saved {out.name}")
        messagebox.showinfo("Save Zelux video", f"Wrote:\n{out}")
        if was_playing:
            self.play()

    def _on_close(self) -> None:
        self.pause()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Play epoxy dual-camera episodes.")
    parser.add_argument(
        "episode",
        nargs="?",
        default=None,
        help="Episode dir, name (episode_000001), or number (1)",
    )
    parser.add_argument(
        "--episode",
        dest="episode_opt",
        default=None,
        help="Same as positional episode argument",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Override dataset episodes directory",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Start playing immediately",
    )
    parser.add_argument(
        "--export-zelux",
        nargs="?",
        const="",
        default=None,
        metavar="OUT.mp4",
        help="Export Zelux stream to MP4 and exit (default: <episode>/<name>_zelux.mp4)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override export FPS (default: episode record_hz)",
    )
    args = parser.parse_args(argv)

    cfg = load_epoxy_config()
    dataset_dir = (args.dataset_dir or cfg.dataset_dir).resolve()
    if not dataset_dir.is_dir():
        print(f"Dataset directory not found: {dataset_dir}", flush=True)
        return 1

    episode = resolve_episode(dataset_dir, args.episode_opt or args.episode)
    if episode is None:
        print(f"No episodes under {dataset_dir}", flush=True)
        return 1

    if args.export_zelux is not None:
        out_arg = Path(args.export_zelux) if args.export_zelux else None
        try:
            out = export_role_video(
                episode, role="zelux", output=out_arg, fps=args.fps
            )
        except Exception as exc:
            print(f"Export failed: {exc}", flush=True)
            return 1
        print(f"Wrote {out}", flush=True)
        return 0

    viewer = EpisodeViewer(dataset_dir, episode, autoplay=args.play)
    viewer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
