"""Subprocess Tk window: target vs current end-link pose during playback."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

from .scene import pose_error_rad


def run_compare_status_window() -> None:
    try:
        import tkinter as tk
    except Exception as exc:
        raise SystemExit(f"Tkinter not available: {exc}") from exc

    import queue
    import threading

    root = tk.Tk()
    root.title("End link: target vs current")
    root.geometry("520x320")
    root.attributes("-topmost", True)

    target_pos_var = tk.StringVar(value="Target pos (m): -")
    target_rpy_var = tk.StringVar(value="Target rpy (rad): -")
    current_pos_var = tk.StringVar(value="Current pos (m): -")
    current_rpy_var = tk.StringVar(value="Current rpy (rad): -")
    err_var = tk.StringVar(value="Error: -")
    phase_var = tk.StringVar(value="Phase: -")
    note_var = tk.StringVar(value="")

    frame = tk.Frame(root, padx=12, pady=10)
    frame.pack(fill="both", expand=True)
    tk.Label(frame, textvariable=phase_var, font=("Helvetica", 12, "bold")).pack(anchor="w")
    tk.Label(frame, text="Target", font=("Helvetica", 11, "bold"), foreground="#004488").pack(
        anchor="w", pady=(8, 0)
    )
    tk.Label(frame, textvariable=target_pos_var, font=("Consolas", 11)).pack(anchor="w")
    tk.Label(frame, textvariable=target_rpy_var, font=("Consolas", 11)).pack(anchor="w")
    tk.Label(frame, text="Current", font=("Helvetica", 11, "bold"), foreground="#228822").pack(
        anchor="w", pady=(8, 0)
    )
    tk.Label(frame, textvariable=current_pos_var, font=("Consolas", 11)).pack(anchor="w")
    tk.Label(frame, textvariable=current_rpy_var, font=("Consolas", 11)).pack(anchor="w")
    tk.Label(frame, textvariable=err_var, font=("Consolas", 12, "bold"), foreground="#aa4400").pack(
        anchor="w", pady=(10, 0)
    )
    tk.Label(frame, textvariable=note_var, font=("Helvetica", 10), foreground="#880000").pack(
        anchor="w", pady=(4, 0)
    )

    message_queue: queue.Queue[Optional[str]] = queue.Queue()

    def read_stdin_thread() -> None:
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    message_queue.put(None)
                    break
                message_queue.put(line.strip())
        except Exception:
            message_queue.put(None)

    threading.Thread(target=read_stdin_thread, daemon=True).start()

    def apply_payload(payload: Dict[str, object]) -> None:
        phase_var.set(str(payload.get("phase", "")))
        note_var.set(str(payload.get("note", "")))
        tp = payload.get("target_pos", [0, 0, 0])
        tr = payload.get("target_rpy", [0, 0, 0])
        cp = payload.get("current_pos", [0, 0, 0])
        cr = payload.get("current_rpy", [0, 0, 0])
        target_pos_var.set(f"Target pos (m): ({tp[0]:+.4f}, {tp[1]:+.4f}, {tp[2]:+.4f})")
        target_rpy_var.set(f"Target rpy (rad): ({tr[0]:+.4f}, {tr[1]:+.4f}, {tr[2]:+.4f})")
        current_pos_var.set(f"Current pos (m): ({cp[0]:+.4f}, {cp[1]:+.4f}, {cp[2]:+.4f})")
        current_rpy_var.set(f"Current rpy (rad): ({cr[0]:+.4f}, {cr[1]:+.4f}, {cr[2]:+.4f})")
        pe = float(payload.get("pos_err_mm", 0.0))
        oe = float(payload.get("ori_err_deg", 0.0))
        err_var.set(f"Error: position {pe:.2f} mm  |  orientation {oe:.2f} deg")

    def poll() -> None:
        try:
            while True:
                try:
                    msg = message_queue.get_nowait()
                except queue.Empty:
                    break
                if msg is None:
                    root.destroy()
                    return
                try:
                    payload = json.loads(msg)
                    if isinstance(payload, dict):
                        apply_payload(payload)
                except Exception:
                    pass
        except Exception:
            pass
        root.after(50, poll)

    root.after(50, poll)
    root.mainloop()


def start_compare_window(script_path: Path) -> Optional[subprocess.Popen[str]]:
    cmd = [sys.executable, str(script_path), "--compare-window"]
    try:
        return subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
    except Exception:
        return None


def send_compare_update(
    proc: Optional[subprocess.Popen[str]],
    phase: str,
    target_pos: Sequence[float],
    target_rpy: Sequence[float],
    current_pos: Sequence[float],
    current_rpy: Sequence[float],
    note: str = "",
) -> None:
    if proc is None or proc.stdin is None:
        return
    pos_err_m, ori_err_rad = pose_error_rad(current_pos, current_rpy, target_pos, target_rpy)
    payload = {
        "phase": phase,
        "note": note,
        "target_pos": list(target_pos),
        "target_rpy": list(target_rpy),
        "current_pos": list(current_pos),
        "current_rpy": list(current_rpy),
        "pos_err_mm": pos_err_m * 1000.0,
        "ori_err_deg": math.degrees(ori_err_rad),
    }
    try:
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()
    except Exception:
        pass


def stop_compare_window(proc: Optional[subprocess.Popen[str]]) -> None:
    if proc is None:
        return
    try:
        if proc.stdin:
            proc.stdin.close()
    except Exception:
        pass
    try:
        proc.terminate()
    except Exception:
        pass
