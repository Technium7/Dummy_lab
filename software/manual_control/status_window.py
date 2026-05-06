from __future__ import annotations

import multiprocessing as mp
from typing import Optional, Sequence

try:
    import tkinter as tk
except Exception:
    tk = None  # type: ignore[assignment]


def status_gui_process_main(queue_obj) -> None:
    if tk is None:
        return
    root = tk.Tk()
    root.title("End Controller Inputs")
    root.geometry("760x520")
    frame = tk.Frame(root, padx=12, pady=12)
    frame.pack(fill="both", expand=True)
    mapping = (
        "Linear: W/S -> -Y/+Y, A/D -> -X/+X, Q/E -> +Z/-Z\n"
        "Angular: H/K -> +Wy/-Wy, U/J -> +Wx/-Wx, I/Y -> final wrist rotation joint only\n"
        "Action: T -> HOME [0, 0, 90, 0, 0, 0]  |  O -> hand close (!HAND_C)  |  P -> hand open (!HAND_O)"
    )
    tk.Label(frame, text="Keyboard Mapping", font=("Helvetica", 13, "bold"), anchor="w").pack(fill="x")
    tk.Label(frame, text=mapping, justify="left", anchor="w", font=("Helvetica", 11)).pack(fill="x", pady=(4, 10))
    keys_var = tk.StringVar(value="Active keys: (none)")
    twist_var = tk.StringVar(value="vx=0.000 vy=0.000 vz=0.000 m/s | wx=0.000 wy=0.000 wz=0.000 rad/s")
    joints_var = tk.StringVar(value="Joint positions:\n  (waiting for simulator data)")
    tk.Label(frame, textvariable=keys_var, anchor="w", font=("Helvetica", 12)).pack(fill="x", pady=(2, 6))
    tk.Label(frame, textvariable=twist_var, anchor="w", font=("Helvetica", 12)).pack(fill="x")
    tk.Label(frame, text="Current Joint Locations", font=("Helvetica", 12, "bold"), anchor="w").pack(fill="x", pady=(10, 4))
    tk.Label(frame, textvariable=joints_var, justify="left", anchor="nw", font=("Consolas", 10)).pack(fill="both", expand=True)

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
            keys_var.set(f"Active keys: {msg.get('active_keys', '(none)')}")
            twist_var.set(str(msg.get("twist", twist_var.get())))
            joints_var.set(str(msg.get("joint_locations", joints_var.get())))
        root.after(50, poll)

    poll()
    root.mainloop()


class ControlStatusWindow:
    def __init__(self) -> None:
        self._proc: Optional[mp.Process] = None
        self._queue = None

    def start(self) -> bool:
        if tk is None:
            return False
        try:
            self._queue = mp.Queue(maxsize=8)
            self._proc = mp.Process(target=status_gui_process_main, args=(self._queue,), name="control-status-gui", daemon=True)
            self._proc.start()
            return self._proc.is_alive()
        except Exception:
            self._queue = None
            self._proc = None
            return False

    def update(
        self,
        active_keys: Sequence[str],
        twist: Sequence[float],
        joint_names: Sequence[str],
        transformed_joint_positions_deg: Sequence[float],
    ) -> None:
        if self._queue is None:
            return
        key_text = " ".join(active_keys) if active_keys else "(none)"
        twist_text = (
            f"vx={twist[0]:+.3f} vy={twist[1]:+.3f} vz={twist[2]:+.3f} m/s"
            f" | wx={twist[3]:+.3f} wy={twist[4]:+.3f} wz={twist[5]:+.3f} rad/s"
        )
        joint_lines = [f"{name}: {pos:+.3f} deg" for name, pos in zip(joint_names, transformed_joint_positions_deg)]
        msg = {"active_keys": key_text, "twist": twist_text, "joint_locations": "Joint positions (transformed):\n" + "\n".join(joint_lines)}
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

