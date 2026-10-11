import asyncio
import time
import numpy as np
import pytest
from detstream.batcher import DetectorBatcher
from detstream.detectors import Detection


class _OneAtATime:
    # Records the frames each call saw. Each frame is a 0-d array holding its feed number
    def __init__(self, delay_s=0.05):
        self.calls = []
        self.delay_s = delay_s

    def detect(self, frame):
        self.calls.append([int(frame)])
        time.sleep(self.delay_s)
        return Detection(True, float(frame))


class _Batched(_OneAtATime):
    def detect_batch(self, frames):
        self.calls.append([int(f) for f in frames])
        time.sleep(self.delay_s)
        return [Detection(True, float(f)) for f in frames]


class _Broken(_Batched):
    def detect_batch(self, frames):
        raise RuntimeError("model fell over")


def test_concurrent_frames_run_as_one_batch():
    det = _Batched()

    async def _run():
        batcher = DetectorBatcher(det)
        return await asyncio.gather(*(batcher.detect(np.array(i)) for i in range(4)))

    results = asyncio.run(_run())

    assert det.calls == [[0, 1, 2, 3]]
    # Each feed gets its own frame's result back
    assert [r.confidence for r in results] == [0.0, 1.0, 2.0, 3.0]


def test_frames_arriving_mid_batch_wait_for_the_next():
    det = _Batched()

    async def _run():
        batcher = DetectorBatcher(det)
        first = asyncio.create_task(batcher.detect(np.array(0)))
        await asyncio.sleep(0.01)
        rest = await asyncio.gather(batcher.detect(np.array(1)), batcher.detect(np.array(2)))
        return [await first, *rest]

    results = asyncio.run(_run())

    assert det.calls == [[0], [1, 2]]
    assert [r.confidence for r in results] == [0.0, 1.0, 2.0]


def test_detector_without_batch_support_runs_frames_in_turn():
    det = _OneAtATime(delay_s=0.0)

    async def _run():
        batcher = DetectorBatcher(det)
        return await asyncio.gather(*(batcher.detect(np.array(i)) for i in range(3)))

    results = asyncio.run(_run())

    assert det.calls == [[0], [1], [2]]
    assert [r.confidence for r in results] == [0.0, 1.0, 2.0]


def test_detector_error_reaches_every_waiting_feed():
    async def _run():
        batcher = DetectorBatcher(_Broken())
        return await asyncio.gather(
            *(batcher.detect(np.array(i)) for i in range(3)), return_exceptions=True
        )

    results = asyncio.run(_run())

    assert all(isinstance(r, RuntimeError) for r in results)


def test_batcher_recovers_after_an_error():
    det = _Batched(delay_s=0.0)

    async def _run():
        batcher = DetectorBatcher(det)
        det.detect_batch, ok = (lambda frames: 1 / 0), det.detect_batch
        with pytest.raises(ZeroDivisionError):
            await asyncio.gather(batcher.detect(np.array(0)), batcher.detect(np.array(1)))
        det.detect_batch = ok
        return await batcher.detect(np.array(5))

    assert asyncio.run(_run()).confidence == 5.0
