"""Vision: capture, YOLO + grasp CNN inference, labeling, and training."""

__all__ = [
    "AlignedFrame",
    "CaptureResult",
    "RealSenseStreamer",
    "capture_realsense_frame",
    "detect_table_object",
    "detect_with_yolo_grasp",
    "load_capture_from_files",
    "save_debug_capture",
    "save_grasp_overlay",
    "segment_table_object",
    "validate_color_image_file",
]


def __getattr__(name: str):
    if name in ("AlignedFrame", "CaptureResult", "RealSenseStreamer"):
        from .capture import AlignedFrame, CaptureResult, RealSenseStreamer

        return {"AlignedFrame": AlignedFrame, "CaptureResult": CaptureResult, "RealSenseStreamer": RealSenseStreamer}[name]
    if name == "capture_realsense_frame":
        from .capture import capture_realsense_frame

        return capture_realsense_frame
    if name == "load_capture_from_files":
        from .capture import load_capture_from_files

        return load_capture_from_files
    if name == "save_debug_capture":
        from .capture import save_debug_capture

        return save_debug_capture
    if name == "validate_color_image_file":
        from .capture import validate_color_image_file

        return validate_color_image_file
    if name == "detect_table_object":
        from .pose import detect_table_object

        return detect_table_object
    if name == "segment_table_object":
        from .segment import segment_table_object

        return segment_table_object
    if name == "save_grasp_overlay":
        from .visualize import save_grasp_overlay

        return save_grasp_overlay
    if name == "detect_with_yolo_grasp":
        from .yolo_grasp import detect_with_yolo_grasp

        return detect_with_yolo_grasp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
