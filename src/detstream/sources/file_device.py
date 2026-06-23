from __future__ import annotations
from collections.abc import AsyncIterator
import cv2
import numpy as np
from . import capture_frames, sources


# A local video file or a webcam/capture device (integer index)
# By default a file stops at EOF, set loop: true to replay it (handy for
# demos). A device index always reconnects, like a live stream
class FileDeviceSource:
    def __init__(self, path: str | int, interval_s: float, loop: bool):
        self.path = path
        self.interval_s = interval_s
        self.loop = loop

    def frames(self) -> AsyncIterator[tuple[float, np.ndarray]]:
        is_device = isinstance(self.path, int)
        reconnect = self.loop or is_device
        return capture_frames(
            lambda: cv2.VideoCapture(self.path),
            self.interval_s,
            label=f"file {self.path}",
            reconnect=reconnect,
        )


@sources.register("file")
def _build(config: dict) -> FileDeviceSource:
    path = config["path"]
    # An all-digit path is a capture device index, not a filename
    if isinstance(path, str) and path.isdigit():
        path = int(path)
    return FileDeviceSource(
        path=path,
        interval_s=config.get("interval_s", 2.0),
        loop=config.get("loop", False),
    )
