from __future__ import annotations
from collections.abc import AsyncIterator
import cv2
import numpy as np
from . import capture_frames, sources


# A direct stream URL (RTSP, HLS, or HTTP) read straight by OpenCV, no resolve step.
# Covers IP cameras and raw stream endpoints
class StreamSource:
    def __init__(self, url: str, interval_s: float):
        self.url = url
        self.interval_s = interval_s

    def frames(self) -> AsyncIterator[tuple[float, np.ndarray]]:
        return capture_frames(
            lambda: cv2.VideoCapture(self.url), self.interval_s, label=f"stream {self.url}"
        )


@sources.register("stream")
def _build(config: dict) -> StreamSource:
    return StreamSource(url=config["url"], interval_s=config.get("interval_s", 2.0))
