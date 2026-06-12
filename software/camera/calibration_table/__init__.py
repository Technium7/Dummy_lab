"""Optical table hole-grid camera calibration (primary workflow).

See ``calibration_table/README.md``. For moving-checkerboard hand-eye, use ``camera.calibration``.
"""

from .grid_config import TableGridConfig
from .grid_detect import GridObservation, detect_table_grid
from .table_calib import TableCalibrationResult, build_calibration_result, save_calibration_result

__all__ = [
    "TableGridConfig",
    "GridObservation",
    "TableCalibrationResult",
    "build_calibration_result",
    "detect_table_grid",
    "save_calibration_result",
]
