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
    # Which class matched, for detectors that look for more than one (yoloe). Single-class
    # detectors leave it None, and sinks that don't care ignore it
    label: str | None = None


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> Detection: ...


detectors: Registry[Detector] = Registry("detstream.detectors")

# Import built-ins so they self-register
# A deployment that did not install the yolo extra can still use a plugin detector without importing ultralytics
try:
    from . import yolo_world  # noqa: F401
except ImportError:
    pass

# yoloe shares the yolo extra (ultralytics ships it), and imports ultralytics lazily in its
# factory, so importing the module to self-register is safe without the extra installed
try:
    from . import yoloe  # noqa: F401
except ImportError:
    pass

# roboflow imports inference_sdk lazily (inside the detector), so the module itself is
# safe to import without the roboflow extra installed
from . import roboflow  # noqa: F401

# rf_detr likewise imports rfdetr lazily (inside the detector), so importing the module
# to self-register is safe without the rf-detr extra installed
from . import rf_detr  # noqa: F401
