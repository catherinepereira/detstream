from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .events import SightingEnded, SightingStarted


# Per-class sighting state: the enter/exit counters, peak frame, and cooldown for one label.
# The tracker holds one of these per class so two animals on screen at once each get their
# own sighting and their own cooldown
@dataclass
class _ClassState:
    present: bool = False
    over: int = 0
    under: int = 0
    peak: float = 0.0
    peak_frame: np.ndarray | None = None
    peak_box: tuple[float, float, float, float] | None = None
    peak_label: str | None = None
    last_alert_at: float | None = None


# State tracking and cooldown over a stream of per-frame detections per feed. Each class is
# tracked independently: a fresh sighting of one class can start while another class is still
# on screen or cooling down. A single-class detector reports label None, which is just one
# class to this tracker
@dataclass
class SightingTracker:
    feed_id: str
    enter_frames: int
    exit_frames: int
    cooldown_s: float

    _classes: dict[str | None, _ClassState] = field(default_factory=dict)

    @property
    def present(self) -> bool:
        return any(st.present for st in self._classes.values())

    def update(
        self,
        detected: bool,
        confidence: float,
        frame: np.ndarray,
        now: float,
        box: tuple[float, float, float, float] | None = None,
        label: str | None = None,
    ) -> SightingStarted | SightingEnded | None:
        # A frame detects at most one class. That class sees a hit; every other class with an
        # open or pending sighting sees a miss this frame, so an animal leaving frame still
        # times out. The detected class is advanced last so its event is the one returned
        event: SightingEnded | None = None
        for other, st in list(self._classes.items()):
            if not (detected and other == label):
                ended = self._advance(st, False, 0.0, frame, now, None, other)
                event = event or ended
                if not st.present and st.over == 0 and st.last_alert_at is None:
                    del self._classes[other]

        if detected:
            st = self._classes.setdefault(label, _ClassState())
            return self._advance(st, True, confidence, frame, now, box, label) or event

        return event

    def _advance(
        self,
        st: _ClassState,
        detected: bool,
        confidence: float,
        frame: np.ndarray,
        now: float,
        box: tuple[float, float, float, float] | None,
        label: str | None,
    ) -> SightingStarted | SightingEnded | None:
        if detected:
            st.over += 1
            st.under = 0
        else:
            st.under += 1
            st.over = 0

        if st.present:
            # Keep the frame, box, and label from the highest-confidence moment for the thumbnail
            if confidence > st.peak:
                st.peak = confidence
                st.peak_frame = frame
                st.peak_box = box
                st.peak_label = label
            if st.under >= self.exit_frames:
                st.present = False
                ended = SightingEnded(
                    self.feed_id, st.peak, st.peak_frame, st.peak_box, st.peak_label
                )
                st.peak = 0.0
                st.peak_frame = None
                st.peak_box = None
                st.peak_label = None
                return ended
            return None

        if st.over >= self.enter_frames and not self._in_cooldown(st, now):
            st.present = True
            st.last_alert_at = now
            st.peak = confidence
            st.peak_frame = frame
            st.peak_box = box
            st.peak_label = label
            return SightingStarted(self.feed_id, confidence, label)

        return None

    def _in_cooldown(self, st: _ClassState, now: float) -> bool:
        if st.last_alert_at is None:
            return False
        return (now - st.last_alert_at) < self.cooldown_s
