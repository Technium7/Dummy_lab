from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
from typing import Any, Optional


def save_convergence_trend(out_dir: Path, history: list[dict[str, Any]]) -> tuple[Path | None, Path | None]:
    """
    Write convergence history and a static trend plot into the run folder.

    Returns (json_path, png_path); either may be None if there is no history or matplotlib is missing.
    """
    if not history:
        return None, None

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "convergence_history.json"
    json_path.write_text(json.dumps({"history": history}, indent=2), encoding="utf-8")

    png_path: Path | None = None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return json_path, None

    xs = list(range(1, len(history) + 1))
    dists = [float(h["distance_m"]) for h in history]
    rolls = [float(h["rpy_deg"][0]) for h in history]
    pitches = [float(h["rpy_deg"][1]) for h in history]
    yaws = [float(h["rpy_deg"][2]) for h in history]
    tspreads = [float(h["translation_spread_m"]) for h in history]
    aspreads = [float(h["angle_spread_deg"]) for h in history]

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    fig.suptitle("Camera calibration convergence (eye-to-hand)")
    ax_dist, ax_rpy, ax_spread = axes
    ax_dist.plot(xs, dists, "o-", color="#1f77b4")
    ax_dist.set_ylabel("distance (m)")
    ax_rpy.plot(xs, rolls, "o-", label="roll", color="#d62728")
    ax_rpy.plot(xs, pitches, "s-", label="pitch", color="#2ca02c")
    ax_rpy.plot(xs, yaws, "^-", label="yaw", color="#9467bd")
    ax_rpy.set_ylabel("rpy (deg)")
    ax_rpy.legend(loc="upper right", fontsize=8)
    ax_spread.plot(xs, tspreads, "o-", label="trans spread (m)", color="#ff7f0e")
    ax_spread.plot(xs, aspreads, "s-", label="angle spread (deg)", color="#17becf")
    ax_spread.set_ylabel("spread")
    ax_spread.set_xlabel("valid sample #")
    ax_spread.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    png_path = out_dir / "convergence_trend.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return json_path, png_path


def _plot_process_main(queue_obj) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError:
        return

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    fig.suptitle("Camera calibration convergence (eye-to-hand)")
    ax_dist, ax_rpy, ax_spread = axes
    ax_dist.set_ylabel("distance (m)")
    ax_rpy.set_ylabel("rpy (deg)")
    ax_spread.set_ylabel("spread")
    ax_spread.set_xlabel("valid sample #")
    (line_dist,) = ax_dist.plot([], [], "o-", color="#1f77b4")
    (line_roll,) = ax_rpy.plot([], [], "o-", label="roll", color="#d62728")
    (line_pitch,) = ax_rpy.plot([], [], "s-", label="pitch", color="#2ca02c")
    (line_yaw,) = ax_rpy.plot([], [], "^-", label="yaw", color="#9467bd")
    (line_tspread,) = ax_spread.plot([], [], "o-", label="trans spread (m)", color="#ff7f0e")
    (line_aspread,) = ax_spread.plot([], [], "s-", label="angle spread (deg)", color="#17becf")
    ax_rpy.legend(loc="upper right", fontsize=8)
    ax_spread.legend(loc="upper right", fontsize=8)

    state: dict[str, Any] = {"history": [], "quit": False}

    def refresh(_frame):
        while True:
            try:
                msg = queue_obj.get_nowait()
            except Exception:
                break
            if not isinstance(msg, dict):
                continue
            if msg.get("cmd") == "quit":
                state["quit"] = True
                plt.close(fig)
                return line_dist, line_roll, line_pitch, line_yaw, line_tspread, line_aspread
            if "history" in msg:
                state["history"] = list(msg["history"])

        hist = state["history"]
        if not hist:
            return line_dist, line_roll, line_pitch, line_yaw, line_tspread, line_aspread

        xs = list(range(1, len(hist) + 1))
        dists = [float(h["distance_m"]) for h in hist]
        rolls = [float(h["rpy_deg"][0]) for h in hist]
        pitches = [float(h["rpy_deg"][1]) for h in hist]
        yaws = [float(h["rpy_deg"][2]) for h in hist]
        tspreads = [float(h["translation_spread_m"]) for h in hist]
        aspreads = [float(h["angle_spread_deg"]) for h in hist]

        line_dist.set_data(xs, dists)
        line_roll.set_data(xs, rolls)
        line_pitch.set_data(xs, pitches)
        line_yaw.set_data(xs, yaws)
        line_tspread.set_data(xs, tspreads)
        line_aspread.set_data(xs, aspreads)

        for ax in axes:
            ax.relim()
            ax.autoscale_view()
        return line_dist, line_roll, line_pitch, line_yaw, line_tspread, line_aspread

    _anim = FuncAnimation(fig, refresh, interval=250, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


class ConvergencePlotWindow:
    def __init__(self) -> None:
        self._proc: Optional[mp.Process] = None
        self._queue = None

    def start(self) -> bool:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            return False
        try:
            self._queue = mp.Queue(maxsize=8)
            self._proc = mp.Process(
                target=_plot_process_main,
                args=(self._queue,),
                name="calibration-convergence-plot",
                daemon=True,
            )
            self._proc.start()
            return self._proc.is_alive()
        except Exception:
            self._queue = None
            self._proc = None
            return False

    def update(self, history: list[dict[str, Any]]) -> None:
        if self._queue is None:
            return
        try:
            self._queue.put_nowait({"history": history})
        except Exception:
            pass

    def close(self) -> None:
        if self._queue is not None:
            try:
                self._queue.put_nowait({"cmd": "quit"})
            except Exception:
                pass
        if self._proc is not None and self._proc.is_alive():
            self._proc.join(timeout=0.5)
            if self._proc.is_alive():
                self._proc.terminate()
        self._queue = None
        self._proc = None
