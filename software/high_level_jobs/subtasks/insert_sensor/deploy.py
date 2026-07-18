"""Deploy trained insert_sensor policy on robot (dual RGB-D + state loop)."""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from manual_control.mapping import (
    MIN_SPEED,
    clamp,
    clamp_firmware_joints_deg,
    deg_speed_to_ascii_speed,
    max_ascii_speed,
    sim_rad_to_transformed_deg,
    sim_vel_rad_to_transformed_deg_s,
    transformed_deg_to_sim_rad,
)
from manual_control.robot_session import connect_robot_for_teleop, disconnect_robot
from trajectory_run.run_planned_trajectory import ensure_robot_armed

from .cameras import (
    OVERHEAD_ROLE,
    SIDE2_ROLE,
    SIDE_ROLE,
    make_streamer,
    resolve_camera_serials,
    stop_streamer,
)
from .config_loader import InsertSensorConfig, default_config_path, load_insert_sensor_config
from .dataset import build_state_vector, normalize
from .policies import load_policy_for_arch

_STREAM_VEL_SCALE = 0.92
# Cap catch-up so a stalled camera frame does not jump too far (seconds).
_MAX_INTEGRATE_DT_S = 0.25
_HOLD_SPEED = 8.0


class PolicyRunner:
    """Rolling-window policy inference at deploy_hz (ACT or diffusion)."""

    def __init__(
        self,
        cfg: InsertSensorConfig,
        *,
        device: str | None = None,
        model_path: Path | None = None,
        arch: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        path = model_path or cfg.model_path
        resolved_arch = arch or getattr(cfg, "arch", None)
        self.model, self.policy_cfg, self.norm_stats, self.arch = load_policy_for_arch(
            path,
            arch=resolved_arch,  # type: ignore[arg-type]
            device=str(self.device),
        )
        if self.arch == "diffusion":
            from .policies.diffusion import preprocess_rgbd_batch
        else:
            from .model import preprocess_rgbd_batch
        self._preprocess_rgbd_batch = preprocess_rgbd_batch
        self.history_len = self.policy_cfg.history_len
        self.action_horizon = int(getattr(self.policy_cfg, "action_horizon", 8))
        self.num_cameras = int(self.policy_cfg.num_cameras)
        self._use_table_xyz = bool(getattr(self.policy_cfg, "use_table_xyz", False))
        self._table_xyz_ctx = None
        if self._use_table_xyz:
            from .table_xyz import TableXyzContext

            self._table_xyz_ctx = TableXyzContext.from_config(
                table_calib_path=getattr(self.policy_cfg, "table_calib_path", "") or None,
                xy_min_m=float(getattr(self.policy_cfg, "table_xyz_xy_min_m", -0.40)),
                xy_max_m=float(getattr(self.policy_cfg, "table_xyz_xy_max_m", 0.40)),
                z_min_m=float(getattr(self.policy_cfg, "table_xyz_z_min_m", -0.05)),
                z_max_m=float(getattr(self.policy_cfg, "table_xyz_z_max_m", 0.35)),
            )
            print(
                f"[insert_sensor] Table XYZ from {self._table_xyz_ctx.calib_path}",
                flush=True,
            )
        self._rgbd_buf: deque[np.ndarray] = deque(maxlen=self.history_len)
        self._state_buf: deque[np.ndarray] = deque(maxlen=self.history_len)
        self._action_chunk: np.ndarray | None = None
        self._action_idx = 0
        self._replan_every = max(1, self.action_horizon // 2)
        self._gripper = 0.0
        print(f"[insert_sensor] Loaded {self.arch} policy from {path}", flush=True)

    def _pad_history(self, buf: deque[np.ndarray], sample: np.ndarray) -> np.ndarray:
        buf.append(sample)
        items = list(buf)
        while len(items) < self.history_len:
            items.insert(0, items[0])
        return np.stack(items[-self.history_len :])

    def _preprocess_cam(
        self,
        color_rgb: np.ndarray,
        depth_m: np.ndarray,
        *,
        is_overhead: bool = False,
    ) -> np.ndarray:
        kwargs: dict = {
            "out_w": self.policy_cfg.img_width,
            "out_h": self.policy_cfg.img_height,
            "depth_min_m": self.policy_cfg.depth_min_m,
            "depth_max_m": self.policy_cfg.depth_max_m,
            "use_depth": bool(getattr(self.policy_cfg, "use_depth", True)),
        }
        if self._use_table_xyz:
            kwargs["use_table_xyz"] = True
            kwargs["table_xyz_ctx"] = self._table_xyz_ctx
            kwargs["is_overhead"] = is_overhead
        return self._preprocess_rgbd_batch(
            color_rgb[np.newaxis, ...],
            depth_m[np.newaxis, ...],
            **kwargs,
        )[0]

    def observe(
        self,
        color_rgb: np.ndarray,
        depth_m: np.ndarray,
        *,
        joint_pos_deg: list[float],
        joint_vel_deg_s: list[float] | None = None,
        wrist_color_rgb: np.ndarray | None = None,
        wrist_depth_m: np.ndarray | None = None,
        side_color_rgb: np.ndarray | None = None,
        side_depth_m: np.ndarray | None = None,
        side2_color_rgb: np.ndarray | None = None,
        side2_depth_m: np.ndarray | None = None,
        overhead_intrinsics: object | None = None,
    ) -> np.ndarray:
        # side_* is preferred; wrist_* kept as alias
        if side_color_rgb is None:
            side_color_rgb = wrist_color_rgb
        if side_depth_m is None:
            side_depth_m = wrist_depth_m
        if (
            self._use_table_xyz
            and self._table_xyz_ctx is not None
            and overhead_intrinsics is not None
        ):
            self._table_xyz_ctx = self._table_xyz_ctx.with_intrinsics(overhead_intrinsics)
        frame = {
            "joint_pos_deg": joint_pos_deg,
            "joint_vel_deg_s": joint_vel_deg_s or [0.0] * 6,
            "gripper": self._gripper,
        }
        st = build_state_vector(frame)
        if self.norm_stats:
            st = normalize(st, self.norm_stats["state_mean"], self.norm_stats["state_std"])

        overhead = self._preprocess_cam(color_rgb, depth_m, is_overhead=True)
        cams = [overhead]

        if self.num_cameras >= 2:
            if side_color_rgb is None or side_depth_m is None:
                cams.append(np.zeros_like(overhead))
            else:
                cams.append(self._preprocess_cam(side_color_rgb, side_depth_m))

        if self.num_cameras >= 3:
            if side2_color_rgb is None or side2_depth_m is None:
                cams.append(np.zeros_like(overhead))
            else:
                cams.append(self._preprocess_cam(side2_color_rgb, side2_depth_m))

        rgbd = np.stack(cams, axis=0)  # (N, C, H, W)
        rgbd_seq = self._pad_history(self._rgbd_buf, rgbd)  # (T, N, C, H, W)
        st_seq = self._pad_history(self._state_buf, st)

        need_sample = (
            self._action_chunk is None
            or self._action_idx >= self._replan_every
            or self._action_idx >= int(self._action_chunk.shape[0])
        )
        if need_sample:
            rgbd_t = torch.from_numpy(rgbd_seq).unsqueeze(0).to(self.device)
            st_t = torch.from_numpy(st_seq).unsqueeze(0).to(self.device)
            chunk = self.model.predict_action(rgbd_t, st_t)[0].cpu().numpy()
            self._action_chunk = chunk
            self._action_idx = 0

        action_norm = self._action_chunk[self._action_idx]
        self._action_idx += 1

        if self.norm_stats:
            mean = np.asarray(self.norm_stats["action_mean"], dtype=np.float32)
            std = np.asarray(self.norm_stats["action_std"], dtype=np.float32)
            action = action_norm * std + mean
        else:
            action = action_norm

        action = action * float(self.cfg.action_scale)
        return action

    def reset(self) -> None:
        self._rgbd_buf.clear()
        self._state_buf.clear()
        self._action_chunk = None
        self._action_idx = 0
        self._gripper = 0.0


def _apply_gripper(robot, runner: PolicyRunner, grip_cmd: float, *, last_sent: float) -> float:
    """Edge-trigger hand open/close from policy gripper channel. Returns new last_sent."""
    thr = float(runner.cfg.gripper_close_threshold)
    if abs(grip_cmd) < thr:
        return last_sent
    want = 1.0 if grip_cmd > 0 else -1.0
    if want == last_sent:
        runner._gripper = want
        return last_sent
    try:
        robot.hand_enable()
        if want > 0:
            print(f"[insert_sensor] HAND_C -> {robot.hand_close()}", flush=True)
        else:
            print(f"[insert_sensor] HAND_O -> {robot.hand_open()}", flush=True)
        # Hand cmds leave stream mode; restore interruptible & streaming.
        robot.ensure_cmdmode_stream()
    except Exception as exc:
        print(f"[insert_sensor] gripper command failed: {exc}", flush=True)
    runner._gripper = want
    return want


def _policy_vel_to_joint_target(
    stream_pose_deg: list[float],
    action: np.ndarray,
    *,
    dt_s: float,
    speed_scale: float = 1.0,
) -> tuple[list[float], list[float], float]:
    """
    Map policy joint velocities → next firmware joint **position** target.

    Matches ``manual_control``:

    1. Policy / teleop ``commanded_velocity`` is **sim** rad/s.
    2. PyBullet integrates that vel in sim, then ``sim_rad_to_transformed_deg`` → ``&`` target
       (see ``teleop_tick`` ``arm_deg`` / ``robot_target_deg``).
    3. The robot stream thread also applies ``sim_vel_rad_to_transformed_deg_s`` when integrating;
       both paths are the analytic derivative of (2). We use (2) so absolute targets match demos.
    """
    scale = max(0.25, float(speed_scale))
    dt = float(np.clip(dt_s, 1e-3, _MAX_INTEGRATE_DT_S))

    # Firmware pose → sim rad (same helper as dataset state encoding).
    q_sim = [
        transformed_deg_to_sim_rad(i, float(stream_pose_deg[i])) for i in range(6)
    ]
    # Integrate sim vel like PyBullet VELOCITY_CONTROL + stepSimulation.
    q_sim_next = [
        q_sim[i] + float(action[i]) * scale * _STREAM_VEL_SCALE * dt for i in range(6)
    ]
    # Absolute firmware target (same as teleop ``arm_deg`` when keys are held).
    target = clamp_firmware_joints_deg(
        [sim_rad_to_transformed_deg(i, q_sim_next[i]) for i in range(6)]
    )

    # ``&`` speed field: same transformed joint rates as teleop_tick.
    vel_deg_s = [
        sim_vel_rad_to_transformed_deg_s(i, float(action[i]) * scale) for i in range(6)
    ]
    min_ascii = MIN_SPEED * 50.0 / 36.0
    speed = float(clamp(deg_speed_to_ascii_speed(vel_deg_s), min_ascii, max_ascii_speed()))
    return target, vel_deg_s, speed


def _print_deploy_banner(*, sim_only: bool, model_path: Path, countdown_s: float) -> None:
    print(
        "\n=== insert_sensor deploy ===\n"
        "Prep (do this FIRST in a separate terminal, then exit teleop):\n"
        "  1. python manual_control/manual_control_app.py\n"
        "  2. T = home\n"
        "  3. Grasp sensor, P = close gripper (same as demo start)\n"
        "  4. ESC / close window to free the serial port\n"
        "  5. Re-run this deploy command\n"
        f"\nModel: {model_path}\n"
        f"Mode: {'sim-only (print actions, no motors)' if sim_only else 'REAL ROBOT STREAM'}\n"
        f"Motion starts in {countdown_s:.0f}s — Ctrl+C anytime to stop.\n",
        flush=True,
    )


def run_deploy_loop(
    cfg: InsertSensorConfig,
    *,
    port: str,
    baudrate: int,
    sim_only: bool = False,
    model_path: Path | None = None,
    countdown_s: float = 5.0,
    speed_scale: float | None = None,
    arch: str | None = None,
) -> None:
    """
    Closed-loop policy at deploy_hz.

    Real robot: connects like teleop, streams ``&`` waypoints from integrated policy
    velocities (same path as manual_control). Prep with teleop home+grasp, exit teleop,
    then run deploy — only one process may hold the serial port.
    """
    path = (model_path or cfg.model_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Policy not found: {path}\n"
            "Train first: python -m high_level_jobs.subtasks.insert_sensor --train "
            "(or --arch diffusion --train)"
        )

    motion_scale = float(speed_scale if speed_scale is not None else cfg.deploy_speed_scale)
    motion_scale = max(0.25, motion_scale)

    _print_deploy_banner(sim_only=sim_only, model_path=path, countdown_s=countdown_s)
    print(f"[insert_sensor] deploy_speed_scale={motion_scale:.2f}  action_scale={cfg.action_scale:.2f}", flush=True)
    runner = PolicyRunner(cfg, model_path=path, arch=arch or getattr(cfg, "arch", None))
    policy_cams = list(cfg.active_cameras())
    required = [c for c in policy_cams if c.role != SIDE2_ROLE]
    optional_side2 = next((c for c in policy_cams if c.role == SIDE2_ROLE), None)
    resolved = resolve_camera_serials(
        required,
        require_pinned=cfg.require_camera_serials,
    )
    streamers: dict[str, object] = {}
    for cam in resolved:
        streamers[cam.role] = make_streamer(
            cam.serial,
            role=cam.role,
            exposure_us=cam.exposure_us,
        )
    if optional_side2 is not None and (optional_side2.serial or "").strip():
        try:
            side2_resolved = resolve_camera_serials(
                [optional_side2],
                require_pinned=cfg.require_camera_serials,
            )
            streamers[SIDE2_ROLE] = make_streamer(
                side2_resolved[0].serial,
                role=SIDE2_ROLE,
                exposure_us=side2_resolved[0].exposure_us,
            )
        except Exception as exc:
            print(
                f"[insert_sensor] side2 optional — skipping ({exc})",
                flush=True,
            )
    period = 1.0 / max(cfg.deploy_hz, 1.0)

    robot = None
    stream_pose_deg: list[float] = [0.0, 0.0, 90.0, 0.0, 0.0, 0.0]
    last_grip_sent = 0.0
    last_stream_send_s = 0.0

    if not sim_only:
        robot = connect_robot_for_teleop(
            port=port,
            baudrate=baudrate,
            timeout=1.0,
            sim_only=False,
        )
        if robot is None:
            raise RuntimeError(f"Failed to open robot on {port}")
        hw = robot.get_joint_pos()
        if hw is not None and len(hw) >= 6:
            stream_pose_deg = clamp_firmware_joints_deg([float(v) for v in hw[:6]])
            print(f"[insert_sensor] Seed pose from GETJPOS: {[round(v, 2) for v in stream_pose_deg]}", flush=True)
        else:
            print("[insert_sensor] WARNING: GETJPOS failed — using default home pose", flush=True)
        # Assume gripper already closed from teleop prep; policy state starts closed.
        runner._gripper = 1.0
        last_grip_sent = 1.0
        # Re-arm after camera init delay — firmware ignores & when drives still disabled.
        ensure_robot_armed(robot, cmd_mode=2, label="[insert_sensor] ")
        robot.hold_stream_pose(stream_pose_deg, _HOLD_SPEED)
        time.sleep(0.15)

    for remaining in range(int(max(countdown_s, 0.0)), 0, -1):
        print(f"[insert_sensor] starting in {remaining}...", flush=True)
        time.sleep(1.0)

    if not sim_only and robot is not None:
        ensure_robot_armed(robot, cmd_mode=2, label="[insert_sensor] ")
        robot.hold_stream_pose(stream_pose_deg, _HOLD_SPEED)

    print(
        f"[insert_sensor] Deploy loop @ {cfg.deploy_hz} Hz  "
        f"({len(streamers)} camera(s); sending & position targets; Ctrl+C to stop)",
        flush=True,
    )
    tick = 0
    try:
        while True:
            t0 = time.perf_counter()
            frames = {role: streamer.read() for role, streamer in streamers.items()}
            oh = frames[OVERHEAD_ROLE]
            side = frames.get(SIDE_ROLE)
            side2 = frames.get(SIDE2_ROLE)
            oh_intr = getattr(oh, "color_intrinsics", None) or getattr(
                streamers.get(OVERHEAD_ROLE), "color_intrinsics", None
            )

            # Fixed control dt like teleop's robot thread (not wall-clock catch-up).
            dt_s = 1.0 / max(cfg.deploy_hz, 1.0)

            action = runner.observe(
                oh.color_rgb,
                oh.depth_m,
                joint_pos_deg=list(stream_pose_deg),
                joint_vel_deg_s=[0.0] * 6,
                side_color_rgb=None if side is None else side.color_rgb,
                side_depth_m=None if side is None else side.depth_m,
                side2_color_rgb=None if side2 is None else side2.color_rgb,
                side2_depth_m=None if side2 is None else side2.depth_m,
                overhead_intrinsics=oh_intr,
            )
            grip_cmd = float(action[6]) if action.shape[0] >= 7 else 0.0
            target_deg, vel_deg_s, speed = _policy_vel_to_joint_target(
                stream_pose_deg, action, dt_s=dt_s, speed_scale=motion_scale
            )
            delta = [target_deg[j] - stream_pose_deg[j] for j in range(6)]

            if tick % 5 == 0:
                print(
                    f"  [& pos] {[round(v, 2) for v in target_deg]}  "
                    f"Δ={[round(v, 2) for v in delta]}  "
                    f"v_deg/s={[round(v, 1) for v in vel_deg_s]}  "
                    f"speed={speed:.1f}  grip={grip_cmd:+.2f}",
                    flush=True,
                )
            tick += 1

            if not sim_only and robot is not None:
                ok = robot.send_trajectory_point_async(
                    target_deg[0],
                    target_deg[1],
                    target_deg[2],
                    target_deg[3],
                    target_deg[4],
                    target_deg[5],
                    speed,
                )
                if ok:
                    stream_pose_deg = list(target_deg)
                    last_stream_send_s = t0
                else:
                    print("[insert_sensor] stream write failed", flush=True)
                last_grip_sent = _apply_gripper(
                    robot, runner, grip_cmd, last_sent=last_grip_sent
                )
            else:
                stream_pose_deg = list(target_deg)
                last_stream_send_s = t0
                if abs(grip_cmd) >= runner.cfg.gripper_close_threshold:
                    runner._gripper = 1.0 if grip_cmd > 0 else -1.0

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, period - elapsed))
    except KeyboardInterrupt:
        print("\n[insert_sensor] Deploy stopped.", flush=True)
        if not sim_only and robot is not None:
            try:
                robot.hold_stream_pose(stream_pose_deg, _HOLD_SPEED)
            except Exception:
                pass
    finally:
        for role, streamer in streamers.items():
            stop_streamer(streamer, role=role)
        if robot is not None:
            disconnect_robot(robot, label="[insert_sensor] ")


def main(argv: list[str] | None = None) -> int:
    import argparse
    import os

    from robot_config import default_robot_port

    p = argparse.ArgumentParser(description="Deploy insert_sensor policy.")
    p.add_argument("--arch", choices=("act", "diffusion"), default=None)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--port", default=default_robot_port())
    p.add_argument("--baudrate", type=int, default=int(os.environ.get("ROBOT_BAUDRATE", "115200")))
    p.add_argument("--sim-only", action="store_true")
    p.add_argument("--countdown", type=float, default=5.0, help="Seconds before motion starts")
    p.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Deploy motion speed scale (default: config deploy_speed_scale, usually 2)",
    )
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    cfg_path = args.config or default_config_path(arch=args.arch or "act")
    cfg = load_insert_sensor_config(cfg_path, arch=args.arch)
    run_deploy_loop(
        cfg,
        port=args.port,
        baudrate=args.baudrate,
        sim_only=args.sim_only,
        model_path=args.model,
        countdown_s=args.countdown,
        speed_scale=args.speed,
        arch=cfg.arch,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
