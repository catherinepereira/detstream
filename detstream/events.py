from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class SightingStarted:
    feed_id: str
    confidence: float
    # The class that matched, when the detector tracks more than one (yoloe). None for
    # single-class detectors
    label: str | None = None


@dataclass
class SightingEnded:
    feed_id: str
    peak_confidence: float
    # The raw peak-confidence frame captured over the sighting's life. A sink that wants a
    # box draws it from peak_box, so this frame stays clean for sinks that want the pixels
    peak_frame: np.ndarray | None = None
    peak_box: tuple[float, float, float, float] | None = None
    # The class matched at peak confidence, mirroring SightingStarted.label
    label: str | None = None
