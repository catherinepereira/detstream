from __future__ import annotations
import cv2
import numpy as np
from .detectors import Detection

BOX_COLOR = (208, 147, 0)  # BGR (a mid blue)
BOX_THICKNESS = 3


# Draw the detection box and a confidence label onto a copy of the frame. Returns the
# frame unchanged when there is no box, so it is safe to call on every sampled frame
def annotate(frame: np.ndarray, detection: Detection, label: str = "") -> np.ndarray:
    if detection.box is None:
        return frame
    out = frame.copy()
    x1, y1, x2, y2 = (int(v) for v in detection.box)
    cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

    text = f"{label} {detection.confidence:.0%}".strip()
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    ty = max(y1, th + 6)
    cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 6, ty), BOX_COLOR, -1)
    cv2.putText(
        out, text, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    )
    return out
