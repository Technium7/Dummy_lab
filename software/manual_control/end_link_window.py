from __future__ import annotations

import math
import multiprocessing as mp
from typing import Optional, Sequence

try:
    import tkinter as tk
except Exception:
    tk = None  # type: ignore[assignment]


def end_link_gui_process_main(queue_obj) -> None:
    if tk is None:
        return
    root = tk.Tk()
    root.title("End Link Pose")
    root.geometry("520x220")
    frame = tk.Frame(root, padx=12, pady=12)
    frame.pack(fill="both", expand=True)

    title_var = tk.StringVar(value="End link: (waiting)")
    pos_var = tk.StringVar(value="Position (m):  —")
    rpy_deg_var = tk.StringVar(value="Rotation RPY (deg):  —")
    rpy_rad_var = tk.StringVar(value="Rotation RPY (rad):  —")

    tk.Label(frame, textvariable=title_var, font=("Helvetica", 13, "bold"), anchor="w").pack(fill="x")
    tk.Label(frame, textvariable=pos_var, anchor="w", font=("Consolas", 12)).pack(fill="x", pady=(10, 4))
    tk.Label(frame, textvariable=rpy_deg_var, anchor="w", font=("Consolas", 12)).pack(fill="x", pady=(2, 4))
    tk.Label(frame, textvariable=rpy_rad_var, anchor="w", font=("Consolas", 10), fg="#444444").pack(fill="x")

    def poll() -> None:
        while True:
            try:
                msg = queue_obj.get_nowait()
            except Exception:
                break
            if not isinstance(msg, dict):
                continue
            if msg.get("cmd") == "quit":
                try:
                    root.destroy()
                except Exception:
                    pass
                return
            title_var.set(str(msg.get("title", title_var.get())))
            pos_var.set(str(msg.get("position", pos_var.get())))
            rpy_deg_var.set(str(msg.get("rotation_deg", rpy_deg_var.get())))
            rpy_rad_var.set(str(msg.get("rotation_rad", rpy_rad_var.get())))
        root.after(50, poll)

    poll()
    root.mainloop()


class EndLinkPoseWindow:
    """Separate Tk window for end-effector position and orientation."""

    def __init__(self) -> None:
        self._proc: Optional[mp.Process] = None
        self._queue = None

    def start(self) -> bool:
        if tk is None:
            return False
        try:
            self._queue = mp.Queue(maxsize=8)
            self._proc = mp.Process(
                target=end_link_gui_process_main,
                args=(self._queue,),
                name="end-link-pose-gui",
                daemon=True,
            )
            self._proc.start()
            return self._proc.is_alive()
        except Exception:
            self._queue = None
            self._proc = None
            return False

    def update(
        self,
        *,
        link_name: str,
        link_index: int,
        position_m: Sequence[float],
        rpy_rad: Sequence[float],
    ) -> None:
        if self._queue is None:
            return
        x, y, z = (float(position_m[i]) for i in range(3))
        r, p, yaw = (float(rpy_rad[i]) for i in range(3))
        msg = {
            "title": f"End link [{link_index}] {link_name}",
            "position": f"Position (m):  X={x:+.4f}  Y={y:+.4f}  Z={z:+.4f}",
            "rotation_deg": f"Rotation RPY (deg):  R={math.degrees(r):+.2f}  P={math.degrees(p):+.2f}  Y={math.degrees(yaw):+.2f}",
            "rotation_rad": f"Rotation RPY (rad):  ({r:+.4f}, {p:+.4f}, {yaw:+.4f})",
        }
        try:
            self._queue.put_nowait(msg)
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
