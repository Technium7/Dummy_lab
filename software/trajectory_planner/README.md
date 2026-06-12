# Trajectory planner

Plan joint trajectories from **HOME** to a target end-link pose in the packaging-station PyBullet scene.

## Pipeline

1. **Goal resolution** (`goals.py`) — teleop joints, or IK + refinement from `target_end_link` (`--goal-pose`).
2. **Nominal iLQR** (`ilqr_core.py` + `lqr_jax.py`) — JAX joint-space tracking to `goal_q`.
3. **SCP collision** (`scp_planner.py`, optional) — linearized env clearance constraints.
4. **Playback** (`playback.py` + `plan_and_demo.py`) — kinematic joint motion in PyBullet GUI.

## Layout

| Module | Role |
|--------|------|
| `plan_and_demo.py` | CLI: plan, cache, playback |
| `targets.py` | `plan_targets.json` + `last_plan.npz` cache |
| `goals.py` | Teleop / IK goal joint resolution |
| `planning.py` | Run iLQR/SCP and verify terminal FK |
| `config.py` | Shared defaults (horizon, SCP, margins) |
| `lqr_planner.py` | Public planning API |
| `ilqr_core.py` | iLQR iteration helpers |
| `lqr_jax.py` | JIT Riccati backward / forward / cost |
| `scp_planner.py` | Nominal iLQR + SCP refinement |
| `collision.py` | PyBullet queries + SCP constraint batches |
| `collision_report.py` | `--show-collision` terminal report |
| `trajectory_utils.py` | Resample, pin endpoints, dedupe |
| `scene.py` | PyBullet scene, FK, IK, pose error |
| `compare_window.py` | Live target vs current pose window |
| `playback.py` | GUI trajectory playback |

## Goal modes

| Mode | Flag | How goal is chosen |
|------|------|-------------------|
| Joint angles (default) | — | `goal_joints_deg` from teleop, or IK fallback |
| Pose → IK joints | `--goal-pose` | IK/refine `target_end_link`, plan tracks those joints |

**Target pose in `plan_targets.json`** uses the same convention as `manual_control` End Link GUI:
- `position_robot_m` — robot base frame (m)
- `orientation_abc_deg` — semantic A/B/C (deg); HOME → `A=0, B=0, C=180`
- `orientation_c_from_xy: true` — sets `C = atan2(Y, X)` from `position_robot_m`

Legacy `position_m` + `orientation_rpy_rad` (world frame) still load if ABC fields are absent.

## Collision model

| API | Meaning |
|-----|---------|
| `is_state_valid` | No env penetration (> 0.1 mm) on arm links |
| `signed_obstacle_distance` | Env `+clearance` or `−penetration` (m) |
| `evaluate_collision_metrics` | Full metrics for `--show-collision` |

Base link (`-1`) is excluded from env checks. Self-collision is reported but not used for planning.

## Run

```bash
pip install pybullet numpy scipy xacro jax
python software/trajectory_planner/plan_and_demo.py
```

Collision planning is **off** by default. Enable with `--collision-penalty` (SCP).

```bash
python software/trajectory_planner/plan_and_demo.py --collision-penalty
python software/trajectory_planner/plan_and_demo.py --goal-pose
python software/trajectory_planner/plan_and_demo.py --replay --show-collision
```

Edit targets in [`plan_targets.json`](plan_targets.json). Delete `last_plan.npz` after changing targets or planner flags.

`last_plan.npz` stores the planned trajectory, `goal_q`, targets, and plan flags. Replay loads only the cache: no second IK, no teleop joint override, no rewriting trajectory endpoints. Playback follows the saved waypoints and ends on `trajectory[-1]`.

## Tuning

| Flag | Effect |
|------|--------|
| `--planner-collision-stride N` | PyBullet every Nth SCP waypoint (default 12) |
| `--scp-weight W` | Constraint strength (default 60) |
| `--clearance-margin M` | Required clearance in m (default 3 mm) |
| `--collision-exact` | Per-link clearance (slow) |

Manual teleop: `env_robot_control_sim.py`.
