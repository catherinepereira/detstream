from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import numpy as np
from ..registry import Registry


@dataclass
class Detection:
    present: bool
    confidence: float
    box: tuple[float, float, float, float] | None = None


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> Detection: ...


detectors: Registry[Detector] = Registry("detstream.detectors")

# Import built-ins so they self-register
# A deployment that did not install the yolo extra can still use a plugin detector without importing ultralytics
try:
    from . import yolo_world  # noqa: F401
except ImportError:
    pass

# roboflow imports inference_sdk lazily (inside the detector), so the module itself is
# safe to import without the roboflow extra installed
from . import roboflow  # noqa: F401
