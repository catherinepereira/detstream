import asyncio
import sqlite3
import cv2
import numpy as np
from detstream.events import SightingEnded, SightingStarted
from detstream.sinks import sinks


def make_sink(tmp_path, **cfg):
    config = {"dir": str(tmp_path), "fps": 10, "pre_s": 0.5, "post_s": 0.5, "width": 0}
    config.update(cfg)
    return sinks.create("clips", config)


def frame(val):
    f = np.zeros((16, 16, 3), dtype=np.uint8)
    f[:] = val
    return f


async def feed(sink, feed_id, n, start_val=0):
    for i in range(n):
        await sink.on_frame(feed_id, float(i), frame(start_val + i))


def index_rows(tmp_path):
    db = sqlite3.connect(tmp_path / "index.db")
    try:
        return db.execute("select * from sightings").fetchall()
    finally:
        db.close()


def test_clip_holds_pre_and_post_frames(tmp_path):
    # fps 10, pre 0.5s, post 0.5s -> 5 pre frames, 5 post frames
    sink = make_sink(tmp_path)

    async def run():
        await feed(sink, "forest", 5)  # fill the pre-roll
        await sink.on_sighting_start(SightingStarted("forest", 0.8, "deer"), "Forest")
        await feed(sink, "forest", 5, start_val=100)  # post-roll
        await sink.on_sighting_end(SightingEnded("forest", 0.9, frame(50), None, "deer"))

    asyncio.run(run())

    clips = list((tmp_path / "forest").glob("*.mp4"))
    assert len(clips) == 1
    cap = cv2.VideoCapture(str(clips[0]))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    # 5 pre seeded into the post-roll + 5 post = 10 frames
    assert count == 10


def test_short_pre_roll_records_what_exists(tmp_path):
    # A sighting right after startup has fewer than pre_frames buffered; record anyway
    sink = make_sink(tmp_path)

    async def run():
        await feed(sink, "forest", 2)  # only 2 frames before the trigger
        await sink.on_sighting_start(SightingStarted("forest", 0.8), "Forest")
        await feed(sink, "forest", 5, start_val=100)
        await sink.on_sighting_end(SightingEnded("forest", 0.9, frame(50)))

    asyncio.run(run())

    cap = cv2.VideoCapture(str(next((tmp_path / "forest").glob("*.mp4"))))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert count == 7  # 2 pre + 5 post


def test_writes_peak_thumbnail_and_index_row(tmp_path):
    sink = make_sink(tmp_path)

    async def run():
        await feed(sink, "forest", 5)
        await sink.on_sighting_start(SightingStarted("forest", 0.8, "fox"), "Forest")
        await feed(sink, "forest", 3, start_val=100)
        await sink.on_sighting_end(SightingEnded("forest", 0.93, frame(77), None, "fox"))

    asyncio.run(run())

    thumbs = list((tmp_path / "forest").glob("*.jpg"))
    assert len(thumbs) == 1
    rows = index_rows(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    # id, feed_id, started_at, ended_at, peak_confidence, label, clip, thumb
    assert row[1] == "forest"
    assert row[4] == 0.93
    assert row[5] == "fox"
    assert row[6].endswith(".mp4")
    assert row[7].endswith(".jpg")


def test_no_thumbnail_when_peak_frame_missing(tmp_path):
    sink = make_sink(tmp_path)

    async def run():
        await feed(sink, "forest", 5)
        await sink.on_sighting_start(SightingStarted("forest", 0.8), "Forest")
        await feed(sink, "forest", 2, start_val=100)
        await sink.on_sighting_end(SightingEnded("forest", 0.9, None))

    asyncio.run(run())

    assert not list((tmp_path / "forest").glob("*.jpg"))
    rows = index_rows(tmp_path)
    assert rows[0][7] is None  # thumb is null


def test_end_without_start_records_nothing(tmp_path):
    sink = make_sink(tmp_path)
    asyncio.run(sink.on_sighting_end(SightingEnded("forest", 0.9, frame(1))))
    assert not list(tmp_path.rglob("*.mp4"))
    assert index_rows(tmp_path) == []


def test_config_defaults(tmp_path):
    sink = make_sink(tmp_path, fps=30, pre_s=5, post_s=5, width=1280)
    assert sink.fps == 30
    assert sink._pre_frames == 150
    assert sink._post_frames == 150
    assert sink.width == 1280


def test_width_cap_downscales_buffered_frames(tmp_path):
    sink = make_sink(tmp_path, width=8)
    wide = np.zeros((16, 32, 3), dtype=np.uint8)
    fitted = sink._fit(wide)
    assert fitted.shape == (4, 8, 3)


def test_no_index_row_when_clip_write_fails(tmp_path, monkeypatch):
    # If the codec cannot open a writer, the sink must not record a row pointing at a clip
    # that was never written, or the server would serve a 404
    sink = make_sink(tmp_path)

    class DeadWriter:
        def isOpened(self):
            return False

        def write(self, f):
            pass

        def release(self):
            pass

    monkeypatch.setattr("detstream.sinks.clips.cv2.VideoWriter", lambda *a, **k: DeadWriter())

    async def run():
        await feed(sink, "forest", 5)
        await sink.on_sighting_start(SightingStarted("forest", 0.8), "Forest")
        await feed(sink, "forest", 3, start_val=100)
        await sink.on_sighting_end(SightingEnded("forest", 0.9, frame(1)))

    asyncio.run(run())

    assert index_rows(tmp_path) == []


def test_clips_registered():
    assert "clips" in sinks._factories
