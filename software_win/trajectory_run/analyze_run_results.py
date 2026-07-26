"""Analyze saved robot run logs: joint and Cartesian (FK) position errors."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

_SOFTWARE_DIR = Path(__file__).resolve().parent.parent
if str(_SOFTWARE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOFTWARE_DIR))

from trajectory_run.fk_error import cartesian_error_vs_target, fk_scene, position_error_vs_reference_q  # noqa: E402
from trajectory_run.run_results import default_run_results_path, load_run_results  # noqa: E402


def _fmt_stats(values: List[float]) -> str:
    if not values:
        return "n/a"
    return f"min={min(values):.3f}  max={max(values):.3f}  mean={float(np.mean(values)):.3f}"


def analyze_run(data: dict, *, export_csv: Path | None = None) -> None:
    meta = data.get("meta", {})
    n = int(meta.get("n_cycles", len(data["cycles"])))
    has_target = "target_pos" in data and "target_rpy" in data
    target_pos = data["target_pos"] if has_target else None
    target_rpy = data["target_rpy"] if has_target else None

    print(f"Run log: {data['path']}")
    print(f"Timestamp (UTC): {meta.get('timestamp_utc', '?')}")
    print(f"Plan: {meta.get('plan_path', '?')}")
    print(f"Cycles: {n}")
    print(f"Planned start (deg): {np.round(data['start_deg'], 4)}")
    print(f"Planned goal  (deg): {np.round(data['goal_deg'], 4)}")
    if has_target:
        print(f"Planned target (m): pos={np.round(target_pos, 4)} rpy={np.round(target_rpy, 4)}")
    print()

    rows: list[dict] = []

    with fk_scene() as scene:
        from trajectory_run.fk_error import ee_pose_from_sim_q

        goal_ref_pos, _ = ee_pose_from_sim_q(scene, data["goal_q"])
        home_ref_pos, _ = ee_pose_from_sim_q(scene, data["start_q"])

        print(
            f"{'Cy':>3}  {'Goal joint':>10}  {'Goal EE mm':>10}  "
            f"{'Goal cart mm':>12}  {'Home joint':>10}  {'Home EE mm':>10}"
        )
        print("-" * 62)

        goal_ee: List[float] = []
        home_ee: List[float] = []
        goal_cart: List[float] = []
        goal_joint: List[float] = []
        home_joint: List[float] = []

        for i in range(n):
            cycle = int(data["cycles"][i])
            at_goal = data["at_target_deg"][i]
            at_home = data["at_home_deg"][i]

            g_joint = float(data["goal_max_err_deg"][i])
            h_joint = float(data["home_max_err_deg"][i])
            goal_joint.append(g_joint)
            home_joint.append(h_joint)

            g_ee = float("nan")
            h_ee = float("nan")
            g_cart = float("nan")

            if not np.any(np.isnan(at_goal)):
                err = position_error_vs_reference_q(scene, at_goal, data["goal_q"])
                g_ee = err.pos_mm
                goal_ee.append(g_ee)
                if has_target:
                    g_cart, g_ori = cartesian_error_vs_target(
                        scene, at_goal, target_pos, target_rpy
                    )
                    goal_cart.append(g_cart)
                else:
                    g_ori = err.ori_deg
            else:
                g_ori = float("nan")

            if not np.any(np.isnan(at_home)):
                err_h = position_error_vs_reference_q(scene, at_home, data["start_q"])
                h_ee = err_h.pos_mm
                home_ee.append(h_ee)
            else:
                err_h = None

            cart_str = f"{g_cart:10.2f}" if has_target and not np.isnan(g_cart) else "         n/a"
            print(
                f"{cycle:3d}  {g_joint:10.4f}  {g_ee:10.2f}  {cart_str:>12}  "
                f"{h_joint:10.4f}  {h_ee:10.2f}"
            )

            rows.append(
                {
                    "cycle": cycle,
                    "goal_max_joint_err_deg": g_joint,
                    "goal_ee_err_mm": g_ee,
                    "goal_cart_err_mm": g_cart if has_target else "",
                    "goal_ori_err_deg": g_ori if not np.isnan(g_ori) else "",
                    "home_max_joint_err_deg": h_joint,
                    "home_ee_err_mm": h_ee,
                    "at_target_J1": at_goal[0],
                    "at_target_J2": at_goal[1],
                    "at_target_J3": at_goal[2],
                    "at_target_J4": at_goal[3],
                    "at_target_J5": at_goal[4],
                    "at_target_J6": at_goal[5],
                    "at_home_J1": at_home[0],
                    "at_home_J2": at_home[1],
                    "at_home_J3": at_home[2],
                    "at_home_J4": at_home[3],
                    "at_home_J5": at_home[4],
                    "at_home_J6": at_home[6 - 1],
                }
            )

        print()
        print(f"Goal joint err (deg):  {_fmt_stats(goal_joint)}")
        print(f"Goal EE err (mm):      {_fmt_stats(goal_ee)}")
        if goal_cart:
            print(f"Goal cart err (mm):    {_fmt_stats(goal_cart)}")
        print(f"Home joint err (deg):  {_fmt_stats(home_joint)}")
        print(f"Home EE err (mm):      {_fmt_stats(home_ee)}")
        print()
        print(f"Reference goal EE (m): {np.round(goal_ref_pos, 4)}")
        print(f"Reference home EE (m): {np.round(home_ref_pos, 4)}")

    if export_csv is not None:
        export_csv = Path(export_csv)
        export_csv.parent.mkdir(parents=True, exist_ok=True)
        with export_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nExported: {export_csv.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze saved trajectory_run logs (joint + FK Cartesian errors)."
    )
    parser.add_argument(
        "--run",
        type=Path,
        default=None,
        help=f"Run log npz (default: {default_run_results_path()})",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV export path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        data = load_run_results(args.run)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"No run log at {exc}. Run: python software/trajectory_run/run_planned_trajectory.py"
        ) from exc
    analyze_run(data, export_csv=args.csv)


if __name__ == "__main__":
    main()
