import asyncio
import cv2
import numpy as np
import pytest
from detstream.sources import capture_frames, sources


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
