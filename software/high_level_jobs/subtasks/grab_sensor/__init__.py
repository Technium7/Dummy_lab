"""Pose-based sensor grasp: YOLO + grasp net, or depth segment → iLQR → execute."""

__all__ = ["GRAB_SENSOR_SUBTASKS"]


def __getattr__(name: str):
    if name == "GRAB_SENSOR_SUBTASKS":
        from .actions import GRAB_SENSOR_SUBTASKS

        return GRAB_SENSOR_SUBTASKS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
