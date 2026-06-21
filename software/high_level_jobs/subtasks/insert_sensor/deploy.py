"""Deploy trained insert_sensor policy on robot (RGB-D + base keypoint loop)."""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from high_level_jobs.subtasks.grab_sensor.detection.capture import RealSenseStreamer

from .base_keypoint import detect_base_keypoint
from .config_loader import InsertSensorConfig, default_config_path, load_grasp_config, load_insert_sensor_config
from .dataset import build_state_vector, normalize
from .model import InsertionPolicy, load_policy, preprocess_rgbd_batch


class PolicyRunner:
    """Rolling-window policy inference at deploy_hz."""

    def __init__(
        self,
        cfg: InsertSensorConfig,
        *,
        device: str | None = None,
        model_path: Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        path = model_path or cfg.model_path
        self.model, self.policy_cfg, self.norm_stats = load_policy(path, device=str(self.device))
        self.grasp_cfg = load_grasp_config(cfg)
        self.history_len = self.policy_cfg.history_len
        self._rgbd_buf: deque[np.ndarray] = deque(maxlen=self.history_len)
        self._state_buf: deque[np.ndarray] = deque(maxlen=self.history_len)
        self._kp_buf: deque[np.ndarray] = deque(maxlen=self.history_len)
        self._gripper = 0.0

    def _pad_history(self, buf: deque[np.ndarray], sample: np.ndarray) -> np.ndarray:
        buf.append(sample)
        items = list(buf)
        while len(items) < self.history_len:
            items.insert(0, items[0])
        return np.stack(items[-self.history_len :])

    def observe(
        self,
        color_rgb: np.ndarray,
        depth_m: np.ndarray,
        *,
        joint_pos_deg: list[float],
        joint_vel_deg_s: list[float] | None = None,
        intrinsics: object | None = None,
    ) -> np.ndarray:
        kp_obs = detect_base_keypoint(color_rgb, depth_m, self.grasp_cfg, intrinsics=intrinsics)
        frame = {
            "joint_pos_deg": joint_pos_deg,
            "joint_vel_deg_s": joint_vel_deg_s or [0.0] * 6,
            "gripper": self._gripper,
        }
        st = build_state_vector(frame)
        kp = kp_obs.vector()
        if self.norm_stats:
            st = normalize(st, self.norm_stats["state_mean"], self.norm_stats["state_std"])
            kp = normalize(kp, self.norm_stats["keypoint_mean"], self.norm_stats["keypoint_std"])

        rgbd = preprocess_rgbd_batch(
            color_rgb[np.newaxis, ...],
            depth_m[np.newaxis, ...],
            out_w=self.policy_cfg.img_width,
            out_h=self.policy_cfg.img_height,
            depth_min_m=self.policy_cfg.depth_min_m,
            depth_max_m=self.policy_cfg.depth_max_m,
        )[0]

        rgbd_seq = self._pad_history(self._rgbd_buf, rgbd)
        st_seq = self._pad_history(self._state_buf, st)
        kp_seq = self._pad_history(self._kp_buf, kp)

        rgbd_t = torch.from_numpy(rgbd_seq).unsqueeze(0).to(self.device)
        st_t = torch.from_numpy(st_seq).unsqueeze(0).to(self.device)
        kp_t = torch.from_numpy(kp_seq).unsqueeze(0).to(self.device)

        action_norm = self.model.predict_action(rgbd_t, kp_t, st_t)[0, -1].cpu().numpy()
        if self.norm_stats:
            mean = np.asarray(self.norm_stats["action_mean"], dtype=np.float32)
            std = np.asarray(self.norm_stats["action_std"], dtype=np.float32)
            action = action_norm * std + mean
        else:
            action = action_norm

        action = action * float(self.cfg.action_scale)
        if action.shape[0] >= 7 and abs(action[6]) > self.cfg.gripper_close_threshold:
            self._gripper = 1.0 if action[6] > 0 else -1.0
        return action

    def reset(self) -> None:
        self._rgbd_buf.clear()
        self._state_buf.clear()
        self._kp_buf.clear()
        self._gripper = 0.0


def run_deploy_loop(
    cfg: InsertSensorConfig,
    *,
    port: str,
    baudrate: int,
    sim_only: bool = False,
    model_path: Path | None = None,
) -> None:
    """Closed-loop policy at deploy_hz (sim-only prints actions; real robot TODO: wire stream)."""
    runner = PolicyRunner(cfg, model_path=model_path)
    streamer = RealSenseStreamer(serial=cfg.record_camera_serial, lock_exposure=True)
    streamer.start()
    period = 1.0 / max(cfg.deploy_hz, 1.0)

    joint_pos_deg = [0.0, 0.0, 90.0, 0.0, 0.0, 0.0]
    if not sim_only:
        from robot_ascii_interface.robot_ascii_interface import RobotAsciiInterface

        robot = RobotAsciiInterface(port=port, baudrate=baudrate)
        if not robot.connect():
            raise RuntimeError(f"Failed to connect to {port}")
        hw = robot.get_joint_pos()
        if hw is not None and len(hw) >= 6:
            joint_pos_deg = [float(v) for v in hw[:6]]

    print(f"[insert_sensor] Deploy loop @ {cfg.deploy_hz} Hz  (Ctrl+C to stop)", flush=True)
    try:
        while True:
            t0 = time.perf_counter()
            frame = streamer.read()
            action = runner.observe(
                frame.color_rgb,
                frame.depth_m,
                joint_pos_deg=joint_pos_deg,
                intrinsics=frame.color_intrinsics,
            )
            joint_vel = action[:6]
            print(
                f"  action vel(rad/s)={[round(float(v), 4) for v in joint_vel]}  grip={runner._gripper:.0f}",
                flush=True,
            )
            if not sim_only:
                # Integrate one step for display; wire to velocity streaming as needed.
                dt = period
                for i in range(6):
                    joint_pos_deg[i] += float(joint_vel[i]) * 180.0 / np.pi * dt
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, period - elapsed))
    except KeyboardInterrupt:
        print("\n[insert_sensor] Deploy stopped.", flush=True)
    finally:
        streamer.stop()


def main(argv: list[str] | None = None) -> int:
    import argparse
    import os

    from robot_config import default_robot_port

    p = argparse.ArgumentParser(description="Deploy insert_sensor policy.")
    p.add_argument("--config", type=Path, default=default_config_path())
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--port", default=default_robot_port())
    p.add_argument("--baudrate", type=int, default=int(os.environ.get("ROBOT_BAUDRATE", "115200")))
    p.add_argument("--sim-only", action="store_true")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    cfg = load_insert_sensor_config(args.config)
    run_deploy_loop(
        cfg,
        port=args.port,
        baudrate=args.baudrate,
        sim_only=args.sim_only,
        model_path=args.model,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
