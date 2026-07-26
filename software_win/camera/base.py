"""Shared camera backend protocol for software_win viewers and future epoxy_app."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class CameraInfo:
    """Human-readable device identity used by list / live-view UIs."""

    backend: str
    name: str
    index: int = 0
    serial: str = ""
    model: str = ""

    def label(self) -> str:
        parts = [self.backend]
        if self.model:
            parts.append(self.model)
        if self.serial:
            parts.append(f"sn={self.serial}")
        elif self.name:
            parts.append(self.name)
        else:
            parts.append(f"#{self.index}")
        return " | ".join(parts)


class CameraBackend(ABC):
    """Minimal grab interface: open → grab BGR uint8 frames → close."""

    @abstractmethod
    def open(self) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    @abstractmethod
    def grab_bgr(self) -> Optional[np.ndarray]:
        """Return latest frame as HxWx3 BGR uint8, or None if not ready."""

    @property
    @abstractmethod
    def info(self) -> CameraInfo:
        ...

    def __enter__(self) -> "CameraBackend":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
