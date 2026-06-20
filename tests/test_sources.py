import asyncio
import cv2
import numpy as np
import pytest
from detstream.sources import capture_frames, sources, STREAM_DOWN_AFTER_S


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (64, 48))
    for i in range(10):
        frame = np.full((48, 64, 3), i * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return str(path)


def collect(agen, limit):
    async def _run():
        out = []
        async for ts, frame in agen:
            out.append((ts, frame))
            if len(out) >= limit:
                break
        return out

    return asyncio.run(_run())


def test_file_source_yields_frames(clip):
    source = sources.create("file", {"path": clip, "interval_s": 0.0})
    frames = collect(source.frames(), limit=5)
    assert len(frames) == 5
    assert all(f.shape == (48, 64, 3) for _ts, f in frames)


def test_file_source_stops_at_eof_without_loop(clip):
    source = sources.create("file", {"path": clip, "interval_s": 0.0, "loop": False})
    frames = collect(source.frames(), limit=1000)
    assert len(frames) == 10


def test_failed_open_without_reconnect_yields_nothing():
    def bad_open():
        return cv2.VideoCapture("/nonexistent/path.mp4")

    async def _run():
        out = []
        async for item in capture_frames(bad_open, 0.0, label="bad", reconnect=False):
            out.append(item)
        return out

    assert asyncio.run(_run()) == []


def test_digit_path_is_treated_as_device_index():
    source = sources.create("file", {"path": "0", "interval_s": 0.0})
    assert source.path == 0


class _FakeCapture:
    # Returns frames from a list, one per read(), repeating the last one forever. A list of
    # identical frames simulates a frozen stream, distinct frames simulate a live one
    def __init__(self, frames):
        self._frames = frames
        self._i = 0

    def read(self):
        frame = self._frames[min(self._i, len(self._frames) - 1)]
        self._i += 1
        return True, frame

    def release(self):
        pass


@pytest.fixture
def fast_clock(monkeypatch):
    # Virtual monotonic clock that advances 1s per read, so the 15s freeze threshold is
    # reached in a few iterations without real waiting
    import detstream.sources as src

    t = {"now": 0.0}

    def fake_monotonic():
        return t["now"]

    async def fake_sleep(_s):
        t["now"] += 1.0

    monkeypatch.setattr(src.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(src.asyncio, "sleep", fake_sleep)
    return t


def test_frozen_stream_reconnects(fast_clock):
    from detstream.sources import capture_frames

    frozen = np.full((48, 64, 3), 7, dtype=np.uint8)
    opens = {"n": 0}

    def open_capture():
        opens["n"] += 1
        # First capture is frozen, the second (after reconnect) raises to end the loop
        if opens["n"] == 1:
            return _FakeCapture([frozen])
        raise RuntimeError("stop")

    async def _run():
        out = []
        try:
            async for ts, frame in capture_frames(open_capture, 0.0, label="frozen"):
                out.append(frame)
        except RuntimeError:
            pass
        return out

    yielded = asyncio.run(_run())
    # It reconnected (opened a second capture) rather than emitting the frozen frame forever
    assert opens["n"] == 2
    # The frozen frames yielded before the threshold are bounded, not infinite
    assert len(yielded) <= int(STREAM_DOWN_AFTER_S) + 1


def test_advancing_stream_is_not_flagged_as_frozen(fast_clock):
    from detstream.sources import capture_frames

    # 30 distinct frames, well past the freeze threshold, so a live stream must never reconnect
    distinct = [np.full((48, 64, 3), i, dtype=np.uint8) for i in range(30)]
    opens = {"n": 0}

    def open_capture():
        opens["n"] += 1
        return _FakeCapture(distinct)

    frames = collect(capture_frames(open_capture, 0.0, label="live"), limit=25)

    assert len(frames) == 25
    # Never reconnected: the single capture served every frame
    assert opens["n"] == 1
