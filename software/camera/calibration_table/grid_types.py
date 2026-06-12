from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DetectedHole:
    center_px: tuple[float, float]
    radius_px: float
    circularity: float
    nn_score: float = 0.0
    grid_col: int = 0
    grid_row: int = 0
    grid_residual_px: float = 0.0
    table_xy_m: tuple[float, float] = (0.0, 0.0)
    is_origin: bool = False
    source: str = ""
    status: str = "detected"  # detected | recovered | locked | rejected | confirmed


@dataclass
class GridSlot:
    grid_col: int
    grid_row: int
    expected_px: tuple[float, float]
    status: str  # detected | recovered | missing | rejected | confirmed
    hole: DetectedHole | None = None
