from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

INCH_TO_M = 0.0254
DEFAULT_HOLE_MODEL = Path(__file__).resolve().parent / "models" / "hole_patch.pt"


@dataclass(frozen=True)
class TableGridConfig:
    """Optical table hole grid geometry and detection tuning."""

    pitch_m: float = INCH_TO_M
    hole_diameter_m: float = 0.00635  # 1/4 inch typical
    min_circularity: float = 0.65
    min_area_px: float = 20.0
    max_area_px: float = 8000.0
    min_holes_for_pose: int = 6
    row_cluster_frac: float = 0.35
    min_pitch_px: float = 8.0
    expected_radius_px: float = 10.0
    max_radius_px: float | None = None
    edge_margin_px: float = 15.0
    hough_param2: int = 18
    nn_confidence: float = 0.38
    ring_confidence: float = 0.35
    use_nn_filter: bool = False
    hole_model_path: Path | None = None
    grid_snap_max_frac: float = 0.50
    min_neighbors: int = 0
    grid_margin_cells: int = 1
    local_search: bool = True

    @property
    def pitch_in(self) -> float:
        return self.pitch_m / INCH_TO_M

    def resolved_model_path(self) -> Path:
        return self.hole_model_path or DEFAULT_HOLE_MODEL

    def hole_radius_limits(self, pitch_px: float | None = None) -> tuple[float, float]:
        """Loose bounds only. Hole radius varies with distance — not used as a hard filter."""
        min_r = 1.0
        if self.max_radius_px is not None:
            return min_r, float(self.max_radius_px)
        max_r = max(64.0, self.expected_radius_px * 8.0)
        if pitch_px is not None and np.isfinite(pitch_px) and pitch_px > 0:
            max_r = max(max_r, pitch_px * 0.55)
        return min_r, max_r

    def resolved_max_area_px(self) -> float:
        """Reject only very large blobs (screw heads, table edge), not perspective-sized holes."""
        _, max_r = self.hole_radius_limits()
        return math.pi * (max_r * 1.1) ** 2

    def to_dict(self) -> dict:
        return {
            "pitch_m": self.pitch_m,
            "pitch_in": self.pitch_in,
            "hole_diameter_m": self.hole_diameter_m,
            "min_holes_for_pose": self.min_holes_for_pose,
            "nn_confidence": self.nn_confidence,
            "use_nn_filter": self.use_nn_filter,
            "grid_snap_max_frac": self.grid_snap_max_frac,
        }

    @classmethod
    def from_cli(
        cls,
        *,
        pitch_in: float = 1.0,
        hole_diameter_in: float = 0.25,
        min_holes: int = 6,
        expected_radius_px: float = 10.0,
        max_radius_px: float | None = None,
        edge_margin_px: float = 15.0,
        min_circularity: float = 0.65,
        nn_confidence: float = 0.38,
        use_nn_filter: bool = False,
        hole_model_path: Path | None = None,
        no_nn: bool = False,
    ) -> TableGridConfig:
        exp = float(expected_radius_px)
        return cls(
            pitch_m=float(pitch_in) * INCH_TO_M,
            hole_diameter_m=float(hole_diameter_in) * INCH_TO_M,
            min_holes_for_pose=int(min_holes),
            expected_radius_px=exp,
            max_radius_px=max_radius_px,
            edge_margin_px=float(edge_margin_px),
            min_circularity=float(min_circularity),
            max_area_px=math.pi * 64.0**2,
            min_area_px=4.0,
            nn_confidence=float(nn_confidence),
            use_nn_filter=bool(use_nn_filter and not no_nn),
            hole_model_path=hole_model_path,
        )
