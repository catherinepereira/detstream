from __future__ import annotations
import numpy as np
from . import Detection, detectors


# Open-vocabulary detector prompted with a class name
class YoloWorldDetector:
    def __init__(self, prompt: str, confidence_threshold: float, weights: str = ""):
        from ultralytics import YOLO

        self.prompt = prompt
        self.threshold = confidence_threshold
        self.model = YOLO(weights or "yolov8s-world.pt")
        self.model.set_classes([prompt])

    def detect(self, frame: np.ndarray) -> Detection:
        results = self.model.predict(frame, conf=0.01, verbose=False)
        best = 0.0
        box = None
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            top = int(r.boxes.conf.argmax())
            conf = float(r.boxes.conf[top])
            if conf > best:
                best = conf
                box = tuple(r.boxes.xyxy[top].cpu().tolist())
        return Detection(present=best >= self.threshold, confidence=best, box=box)


@detectors.register("yolo-world")
def _build(config: dict) -> YoloWorldDetector:
    return YoloWorldDetector(
        prompt=config.get("prompt", "object"),
        confidence_threshold=config.get("confidence_threshold", 0.4),
        weights=config.get("weights", ""),
    )
