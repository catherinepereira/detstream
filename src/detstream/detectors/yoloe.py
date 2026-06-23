from __future__ import annotations
import numpy as np
from . import Detection, detectors


# Open-vocabulary detector prompted with several class names at once. Unlike yolo-world
# (one class), this reports which of the prompts matched via Detection.label, so a feed
# can watch for many kinds of wildlife and a sink can say which was spotted.
class YoloeDetector:
    def __init__(
        self,
        prompts: list[str],
        confidence_threshold: float,
        weights: str = "",
    ):
        from ultralytics import YOLOE

        if not prompts:
            raise ValueError("yoloe detector needs at least one prompt")
        self.prompts = list(prompts)
        self.threshold = confidence_threshold
        self.model = YOLOE(weights or "yoloe-11s-seg.pt")
        # Text prompts are turned into class embeddings once, at build time, so detect()
        # is a plain forward pass with no per-frame prompt encoding
        self.model.set_classes(self.prompts, self.model.get_text_pe(self.prompts))

    def detect(self, frame: np.ndarray) -> Detection:
        results = self.model.predict(frame, conf=0.01, verbose=False)
        best = 0.0
        box = None
        label = None
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            top = int(r.boxes.conf.argmax())
            conf = float(r.boxes.conf[top])
            if conf > best:
                best = conf
                box = tuple(r.boxes.xyxy[top].cpu().tolist())
                label = self._label_for(r.boxes.cls[top])
        return Detection(present=best >= self.threshold, confidence=best, box=box, label=label)

    # Map a class index back to its prompt. The index arrives as a tensor element, so it
    # goes through float() (which the tensor defines) before int(). Out-of-range is treated
    # as no label rather than an error, so an unexpected index degrades to an unlabelled
    # sighting instead of crashing the feed
    def _label_for(self, cls_index) -> str | None:
        i = int(float(cls_index))
        return self.prompts[i] if 0 <= i < len(self.prompts) else None


@detectors.register("yoloe")
def _build(config: dict) -> YoloeDetector:
    prompts = config.get("prompts")
    if prompts is None:
        # Accept a single `prompt` too, so a one-class feed reads the same as yolo-world
        prompt = config.get("prompt")
        prompts = [prompt] if prompt else []
    return YoloeDetector(
        prompts=prompts,
        confidence_threshold=config.get("confidence_threshold", 0.4),
        weights=config.get("weights", ""),
    )
