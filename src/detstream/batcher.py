from __future__ import annotations
import asyncio
import numpy as np
from .detectors import Detection, Detector


# One detector shared by every feed with the same detector config. Frames that arrive while
# a batch is running wait and go together in the next batch, so feeds share one copy of the
# weights and the model never runs concurrently with itself. Each feed awaits its result
# before sending another frame, so a batch has at most one frame per feed
class DetectorBatcher:
    def __init__(self, detector: Detector):
        self.detector = detector
        self._pending: list[tuple[np.ndarray, asyncio.Future]] = []
        self._worker: asyncio.Task | None = None

    async def detect(self, frame: np.ndarray) -> Detection:
        future = asyncio.get_running_loop().create_future()
        self._pending.append((frame, future))
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._drain())
        return await future

    async def _drain(self) -> None:
        while self._pending:
            batch, self._pending = self._pending, []
            frames = [frame for frame, _ in batch]
            try:
                results = await asyncio.to_thread(self._run, frames)
            except Exception as e:
                for _, future in batch:
                    if not future.done():
                        future.set_exception(e)
                continue
            for (_, future), result in zip(batch, results):
                # A feed cancelled while waiting has already given up on its result
                if not future.done():
                    future.set_result(result)

    def _run(self, frames: list[np.ndarray]) -> list[Detection]:
        if len(frames) > 1 and hasattr(self.detector, "detect_batch"):
            return self.detector.detect_batch(frames)
        return [self.detector.detect(frame) for frame in frames]
