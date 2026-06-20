from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .events import SightingEnded, SightingStarted


# State tracking and cooldown over a stream of per-frame detections per feed
# Fires SightingStarted when a sighting begins, tracks peak confidence for the
# duration of the sighting, and doesn't fire again until the cooldown has elapsed
@dataclass
class SightingTracker:
    feed_id: str
    enter_frames: int
    exit_frames: int
    cooldown_s: float

    present: bool = False
    _over: int = 0
    _under: int = 0
    _peak: float = 0.0
    _peak_frame: np.ndarray | None = None
    _peak_box: tuple[float, float, float, float] | None = None
    _peak_label: str | None = None
    _last_alert_at: float | None = field(default=None)

    def update(
        self,
        detected: bool,
        confidence: float,
        frame: np.ndarray,
        now: float,
        box: tuple[float, float, float, float] | None = None,
        label: str | None = None,
    ) -> SightingStarted | SightingEnded | None:
        if detected:
            self._over += 1
            self._under = 0
        else:
            self._under += 1
            self._over = 0

        if self.present:
            # Keep the frame, box, and label from the highest-confidence moment for the thumbnail
            if confidence > self._peak:
                self._peak = confidence
                self._peak_frame = frame
                self._peak_box = box
                self._peak_label = label
            if self._under >= self.exit_frames:
                self.present = False
                ended = SightingEnded(
                    self.feed_id, self._peak, self._peak_frame, self._peak_box, self._peak_label
                )
                self._peak = 0.0
                self._peak_frame = None
                self._peak_box = None
                self._peak_label = None
                return ended
            return None

        if self._over >= self.enter_frames and not self._in_cooldown(now):
            self.present = True
            self._last_alert_at = now
            self._peak = confidence
            self._peak_frame = frame
            self._peak_box = box
            self._peak_label = label
            return SightingStarted(self.feed_id, confidence, label)

        return None

    def _in_cooldown(self, now: float) -> bool:
        if self._last_alert_at is None:
            return False
        return (now - self._last_alert_at) < self.cooldown_s
