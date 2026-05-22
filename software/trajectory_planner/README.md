# Trajectory planner

Plan joint trajectories from **HOME** to a target end-link pose in the packaging-station PyBullet scene.

## Pipeline

1. **PyBullet collision checking** — robot vs static OBJ environment (+ self-collision).
2. **RRTConnect** — bidirectional RRT-Connect (same logic as OMPL `RRTConnect`).
3. **CHOMP** — gradient smoothing (smoothness + obstacle + path prior).
4. **Playback** — kinematic joint interpolation (no motor fighting), capped joint step per frame, 1 s settle at HOME before moving.
5. **Goal accuracy** — coordinate-descent IK refinement targets under **0.1 mm** end-link position error.

> The official `ompl` Python wheel is not available on Windows; planners are implemented in Python with the same algorithm structure.

## Edit targets

Update [`plan_targets.json`](plan_targets.json):

```json
{
  "start_joints_deg": [0, 0, 90, 0, 0, 0],
  "target_end_link": {
    "position_m": [0.0010, 0.3062, 0.1715],
    "orientation_rpy_rad": [0.4912, 0.0265, -3.1190]
  }
}
```

Default target values match the reference screenshot in `test_end_position.png`.

## Run

From repo root:

```bash
pip install pybullet numpy scipy xacro
python software/trajectory_planner/plan_and_demo.py
```

Plan without GUI (faster check):

```bash
python software/trajectory_planner/plan_and_demo.py --plan-only
```

**Deterministic planning:** same path every time with default `--seed 42`. Cached to `last_plan.npz`.

**Identical playback:** replay the cached plan without replanning:

```bash
python software/trajectory_planner/plan_and_demo.py --replay
```

## Scripts

| File | Role |
|------|------|
| `plan_and_demo.py` | Main entry: plan + play + compare GUI |
| `plan_targets.json` | Editable start joints and target end pose |
| `scene.py` | Load robot (Z flipped 180°) + environment |
| `collision.py` | PyBullet validity / obstacle cost |
| `rrt_connect.py` | RRTConnect planner |
| `chomp.py` | CHOMP-style smoother |
| `env_robot_preview.py` | Static preview (sliders) |
| `env_robot_control_sim.py` | Manual IK teleop in scene |
