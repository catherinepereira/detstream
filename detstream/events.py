from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class SightingStarted:
    feed_id: str
    confidence: float


@dataclass
class SightingEnded:
    feed_id: str
    peak_confidence: float
    # The peak-confidence frame, annotated, captured over the sighting's life. The
    # thumbnail is uploaded from this so it matches the stored peak confidence
    peak_frame: np.ndarray | None = None
