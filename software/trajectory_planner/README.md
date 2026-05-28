# Trajectory planner

Plan joint trajectories from **HOME** to a target end-link pose in the packaging-station PyBullet scene.

## Pipeline

1. **Collision queries** — binary validity from penetration; optimization uses signed clearance/penetration distance (`collision.py`).
2. **Discrete iLQR (JAX)** — JIT Riccati backward pass + vectorized line search; PyBullet collision penalty/gradient per waypoint in NumPy.
3. **Playback** — kinematic joint interpolation in PyBullet GUI.

## Collision model

| API | Type | Meaning |
|-----|------|---------|
| `is_state_valid` | **0/1** | `True` if worst penetration ≤ 1e-4 m |
| `max_penetration_depth` | continuous | Penetration depth (m), 0 if separated |
| `min_clearance_distance` | continuous | Minimum `getClosestPoints` separation (m) |
| `signed_obstacle_distance` | continuous | `+clearance` or `-penetration` |
| `collision_penalty` | continuous | Quadratic on penetration; barrier inside clearance margin |

## Edit targets

Update [`plan_targets.json`](plan_targets.json) (include `goal_joints_deg` from `env_robot_control_sim` for best results).

## Run

```bash
pip install pybullet numpy scipy xacro jax
python software/trajectory_planner/plan_and_demo.py
```

Collision penalty is **off** by default (fast joint tracking only). Re-enable with `--collision-penalty`.

```bash
python software/trajectory_planner/plan_and_demo.py --plan-only
rm software/trajectory_planner/last_plan.npz   # after changing targets
python software/trajectory_planner/plan_and_demo.py --replay
```

## Scripts

| File | Role |
|------|------|
| `plan_and_demo.py` | Main entry: plan + play + compare GUI |
| `lqr_planner.py` | iLQR orchestration (JAX + PyBullet) |
| `lqr_jax.py` | JIT backward / forward / cost |
| `collision.py` | Validity + signed distance + penalty |
| `trajectory_utils.py` | Resample / pin endpoints |
| `scene.py` | Load robot + environment |
| `env_robot_control_sim.py` | Manual IK teleop in scene |
