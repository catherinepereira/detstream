from __future__ import annotations
import asyncio
import logging
import sqlite3
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
from ..events import SightingEnded, SightingStarted
from . import sinks

log = logging.getLogger(__name__)


# Records a short MP4 around each sighting plus the peak JPEG, and indexes both in a SQLite
# file a local server can read. Unlike the event-only sinks this one implements on_frame: the
# runner tees every captured frame here so a pre-roll buffer holds the seconds before the
# trigger. Detection runs on a subsample, so on_frame is the only way the sink sees video-rate
# frames
class ClipSink:
    def __init__(self, config: dict):
        self.dir = Path(config.get("dir", "clips"))
        self.fps = float(config.get("fps", 30))
        self.pre_s = float(config.get("pre_s", 5))
        self.post_s = float(config.get("post_s", 5))
        self.codec = config.get("codec", "mp4v")
        self.jpg_quality = int(config.get("jpg_quality", 90))
        # Cap stored width so a 5s pre-roll at 30fps does not pin a gigabyte of 1080p frames
        # in memory per feed. 0 keeps frames at source resolution
        self.width = int(config.get("width", 1280))

        self._pre_frames = max(1, round(self.pre_s * self.fps))
        self._post_frames = max(1, round(self.post_s * self.fps))
        # Per feed: a rolling pre-roll deque, plus the post-roll frames being collected while a
        # sighting is open. recording[feed_id] is None when no sighting is in progress
        self._buffers: dict[str, deque] = {}
        self._recording: dict[str, dict] = {}
        self._started_at: dict[str, str] = {}

        self.dir.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.dir / "index.db", check_same_thread=False)
        # WAL so a separate reader process (the mh-nature-cam server) can read committed rows
        # while this sink keeps writing, without blocking each other
        self._db.execute("pragma journal_mode=wal")
        self._db.execute(
            """
            create table if not exists sightings (
                id text primary key,
                feed_id text not null,
                started_at text not null,
                ended_at text,
                peak_confidence real,
                label text,
                clip text,
                thumb text
            )
            """
        )
        self._db.execute(
            "create index if not exists sightings_started_at on sightings (started_at desc)"
        )
        self._db.commit()

    async def on_frame(self, feed_id: str, ts: float, frame) -> None:
        frame = self._fit(frame)
        buf = self._buffers.get(feed_id)
        if buf is None:
            buf = self._buffers[feed_id] = deque(maxlen=self._pre_frames)
        buf.append(frame)
        rec = self._recording.get(feed_id)
        if rec is not None and rec["post_left"] > 0:
            rec["frames"].append(frame)
            rec["post_left"] -= 1

    async def on_sighting_start(self, event: SightingStarted, feed_name: str) -> None:
        # Seed the clip with the pre-roll already buffered, then collect post_frames more as
        # on_frame fires until the post-roll is full or the sighting ends, whichever comes first
        buf = self._buffers.get(event.feed_id)
        self._recording[event.feed_id] = {
            "frames": list(buf) if buf else [],
            "post_left": self._post_frames,
        }
        self._started_at[event.feed_id] = datetime.now(timezone.utc).isoformat()

    async def on_sighting_end(self, event: SightingEnded) -> None:
        rec = self._recording.pop(event.feed_id, None)
        started_at = self._started_at.pop(event.feed_id, None)
        frames = rec["frames"] if rec else None
        if not frames:
            return
        sighting_id = uuid.uuid4().hex
        out_dir = self.dir / event.feed_id
        clip_path = out_dir / f"{sighting_id}.mp4"
        thumb_path = out_dir / f"{sighting_id}.jpg" if event.peak_frame is not None else None
        # The index must only name files that landed on disk, or the server serves a 404. So
        # _write reports what it actually wrote and _index records only those
        wrote_clip, wrote_thumb = await asyncio.to_thread(
            self._write, out_dir, frames, clip_path, thumb_path, event.peak_frame
        )
        if not wrote_clip:
            return
        self._index(
            sighting_id, event, started_at, clip_path, thumb_path if wrote_thumb else None
        )

    def _write(self, out_dir, frames, clip_path, thumb_path, peak_frame) -> tuple[bool, bool]:
        out_dir.mkdir(parents=True, exist_ok=True)
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(str(clip_path), fourcc, self.fps, (w, h))
        if not writer.isOpened():
            log.warning("clip sink could not open writer for %s", clip_path)
            return False, False
        try:
            for f in frames:
                # A stream that changes resolution mid-clip yields odd-sized frames, which
                # VideoWriter.write silently drops. Resize them to the opening size so the
                # clip keeps every frame
                if f.shape[:2] != (h, w):
                    f = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
                writer.write(f)
        finally:
            writer.release()
        wrote_thumb = False
        if thumb_path is not None:
            wrote_thumb = cv2.imwrite(
                str(thumb_path), self._fit(peak_frame), [cv2.IMWRITE_JPEG_QUALITY, self.jpg_quality]
            )
            if not wrote_thumb:
                log.warning("clip sink failed to write thumbnail %s", thumb_path)
        return True, wrote_thumb

    def _index(self, sighting_id, event: SightingEnded, started_at, clip_path, thumb_path) -> None:
        ended_at = datetime.now(timezone.utc).isoformat()
        self._db.execute(
            "insert into sightings"
            " (id, feed_id, started_at, ended_at, peak_confidence, label, clip, thumb)"
            " values (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                sighting_id,
                event.feed_id,
                started_at or ended_at,
                ended_at,
                event.peak_confidence,
                event.label,
                clip_path.name,
                thumb_path.name if thumb_path else None,
            ),
        )
        self._db.commit()

    # Downscale to the width cap, keeping aspect. Frames are copied so the deque does not
    # pin OpenCV's reused capture buffer
    def _fit(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if self.width and w > self.width:
            new_h = round(h * self.width / w)
            return cv2.resize(frame, (self.width, new_h), interpolation=cv2.INTER_AREA)
        return frame.copy()


@sinks.register("clips")
def _build(config: dict) -> ClipSink:
    return ClipSink(config)
