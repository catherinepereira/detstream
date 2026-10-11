import asyncio
import time
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


class _Stop(BaseException):
    # Not an Exception, so capture_frames' open-retry loop lets it through and the test ends
    pass


def test_frozen_stream_reconnects(fast_clock):
    from detstream.sources import capture_frames

    frozen = np.full((48, 64, 3), 7, dtype=np.uint8)
    opens = {"n": 0}

    def open_capture():
        opens["n"] += 1
        # First capture is frozen, the second (after reconnect) ends the loop
        if opens["n"] == 1:
            return _FakeCapture([frozen])
        raise _Stop

    async def _run():
        out = []
        try:
            async for ts, frame in capture_frames(open_capture, 0.0, label="frozen"):
                out.append(frame)
        except _Stop:
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


class _LiveCapture:
    # A decoder fed by a live stream: a new frame every few ms, each holding its sequence
    # number so a test can see which frames were skipped. After `frames` reads it fails, or
    # with frozen=True it keeps returning the first frame
    def __init__(self, frames=None, frozen=False):
        self.n = 0
        self.frames = frames
        self.frozen = frozen
        self.released = False

    def read(self):
        time.sleep(0.002)
        self.n += 1
        if self.frames is not None and self.n > self.frames:
            return False, None
        value = 1 if self.frozen else self.n
        return True, np.full((4, 4, 3), value, dtype=np.int32)

    def get(self, prop):
        # No timestamps, so the reader never paces this capture
        return 0.0

    def release(self):
        self.released = True


def _run_until_stop(agen):
    async def _run():
        out = []
        try:
            async for _ts, frame in agen:
                out.append(frame)
        except _Stop:
            pass
        return out

    return asyncio.run(_run())


def test_live_source_yields_newest_frame():
    cap = _LiveCapture()
    frames = collect(capture_frames(lambda: cap, 0.05, label="live", live=True), limit=4)

    seqs = [int(f[0, 0, 0]) for _ts, f in frames]
    # Several frames arrive per interval. Reading in order would yield 1, 2, 3, 4
    assert all(b - a > 1 for a, b in zip(seqs, seqs[1:]))
    assert cap.released


def test_live_source_reconnects_when_frames_stop(monkeypatch):
    import detstream.sources as src

    monkeypatch.setattr(src, "STREAM_DOWN_AFTER_S", 0.2)
    caps = []

    def open_capture():
        if caps:
            raise _Stop
        caps.append(_LiveCapture(frames=3))
        return caps[0]

    _run_until_stop(capture_frames(open_capture, 0.0, label="live", live=True))

    assert len(caps) == 1 and caps[0].released


def test_live_source_reconnects_when_frozen(monkeypatch):
    import detstream.sources as src

    monkeypatch.setattr(src, "STREAM_DOWN_AFTER_S", 0.2)
    caps = []

    def open_capture():
        if caps:
            raise _Stop
        caps.append(_LiveCapture(frozen=True))
        return caps[0]

    frames = _run_until_stop(capture_frames(open_capture, 0.0, label="live", live=True))

    assert caps[0].released
    # A frozen feed stops yielding once the threshold passes instead of repeating forever
    assert len(frames) < 1000


class _TimestampedCapture:
    # Decodes instantly, like a buffered HLS segment, with a timestamp step_s apart per frame.
    # jump_at shifts every later timestamp forward by a minute, like skipped segments
    def __init__(self, step_s, jump_at=None):
        self.n = 0
        self.step_s = step_s
        self.jump_at = jump_at

    def read(self):
        self.n += 1
        return True, np.full((4, 4, 3), self.n, dtype=np.int32)

    def get(self, prop):
        jump = 60.0 if self.jump_at is not None and self.n >= self.jump_at else 0.0
        return (self.n * self.step_s + jump) * 1000

    def release(self):
        pass


def _seq_after(cap, wait_s):
    from detstream.sources import _LatestFrame

    reader = _LatestFrame(cap)
    time.sleep(wait_s)
    _frame, seq, _last_ok = reader.latest()
    reader.close()
    return seq


def test_live_reader_paces_buffered_frames_to_real_time():
    # 20fps timestamps over 0.3s is about 6 frames. Unpaced, the reader would decode thousands
    assert 3 <= _seq_after(_TimestampedCapture(step_s=0.05), 0.3) <= 12


def test_live_reader_restarts_its_clock_after_a_timestamp_jump():
    # Without the rebase, the minute-long jump at frame 3 would put the reader to sleep
    assert _seq_after(_TimestampedCapture(step_s=0.05, jump_at=3), 0.5) >= 6
