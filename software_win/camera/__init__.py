"""Windows camera backends for epoxy / high-level jobs (Zelux + Infinity + RealSense)."""

from camera.base import CameraBackend, CameraInfo
from camera.infinity import InfinityCamera, list_infinity_devices
from camera.realsense import RealSenseCamera, list_realsense_devices
from camera.zelux import ZeluxCamera, list_zelux_devices

__all__ = [
    "CameraBackend",
    "CameraInfo",
    "InfinityCamera",
    "RealSenseCamera",
    "ZeluxCamera",
    "list_infinity_devices",
    "list_realsense_devices",
    "list_zelux_devices",
]
