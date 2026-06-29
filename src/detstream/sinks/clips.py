from __future__ import annotations
import asyncio
import logging
import shutil
import sqlite3
import subprocess
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
from ..events import SightingEnded, SightingStarted
from . import sinks

log = logging.getLogger(__name__)

_FFMPEG = shutil.which("ffmpeg")


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

        # fps <= 0 opens an invalid VideoWriter that drops clips or writes a corrupt file, and a
        # negative window is meaningless. Fail at construction rather than silently per sighting
        if self.fps <= 0:
            raise ValueError(f"clips sink fps must be > 0, got {self.fps}")
        if self.pre_s < 0 or self.post_s < 0:
            raise ValueError("clips sink pre_s and post_s must be >= 0")

        self._pre_frames = max(1, round(self.pre_s * self.fps))
        self._post_frames = max(1, round(self.post_s * self.fps))
        # Pre-roll deque is per feed. Recording state is per (feed_id, label) so two classes on
        # one feed each get their own clip
        self._buffers: dict[str, deque] = {}
        self._recording: dict[tuple[str, str | None], dict] = {}
        self._started_at: dict[tuple[str, str | None], str] = {}

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
        # Extend every class currently recording on this feed. Two animals on screen at once
        # each have their own open recording but share the incoming frames
        for (rec_feed, _label), rec in self._recording.items():
            if rec_feed == feed_id and rec["post_left"] > 0:
                rec["frames"].append(frame)
                rec["post_left"] -= 1

    async def on_sighting_start(self, event: SightingStarted, feed_name: str) -> None:
        # Key recording by (feed_id, label) so concurrent sightings of different classes on one
        # feed each get their own clip. Seed with the pre-roll, then collect post_frames more as
        # on_frame fires until the post-roll is full or the sighting ends, whichever comes first
        key = (event.feed_id, event.label)
        buf = self._buffers.get(event.feed_id)
        self._recording[key] = {
            "frames": list(buf) if buf else [],
            "post_left": self._post_frames,
        }
        self._started_at[key] = datetime.now(timezone.utc).isoformat()

    async def on_sighting_end(self, event: SightingEnded) -> None:
        key = (event.feed_id, event.label)
        rec = self._recording.pop(key, None)
        started_at = self._started_at.pop(key, None)
        frames = rec["frames"] if rec else None
        if not frames:
            return
        sighting_id = uuid.uuid4().hex
        out_dir = (self.dir / event.feed_id).resolve()
        # feed_id comes from config, but a stray "../" or absolute path would write outside the
        # clips dir and let a reader serve files from anywhere. Keep every clip under self.dir
        if not out_dir.is_relative_to(self.dir.resolve()):
            log.warning("clip sink: feed_id %r escapes the clips dir, skipping", event.feed_id)
            return
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
        # resize any mid-clip resolution change back to the opening size
        sized = (f if f.shape[:2] == (h, w) else cv2.resize(f, (w, h)) for f in frames)
        wrote_clip = (
            self._write_ffmpeg(sized, clip_path, w, h)
            if _FFMPEG
            else self._write_cv2(sized, clip_path, w, h)
        )
        if not wrote_clip:
            return False, False
        wrote_thumb = False
        if thumb_path is not None:
            wrote_thumb = cv2.imwrite(
                str(thumb_path), self._fit(peak_frame), [cv2.IMWRITE_JPEG_QUALITY, self.jpg_quality]
            )
            if not wrote_thumb:
                log.warning("clip sink failed to write thumbnail %s", thumb_path)
        return True, wrote_thumb

    def _write_ffmpeg(self, frames, clip_path, w, h) -> bool:
        # H.264 + yuv420p + faststart is what browsers play. cv2's mp4v is MPEG-4 Part 2, which
        # Chrome will not decode, and it leaves the moov atom at the end so playback can't stream
        cmd = [
            _FFMPEG, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(self.fps),
            "-i", "-",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(clip_path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for f in frames:
                proc.stdin.write(np.ascontiguousarray(f).tobytes())
        except BrokenPipeError:
            pass
        _, err = proc.communicate()
        if proc.returncode != 0:
            log.warning("clip sink ffmpeg failed for %s: %s", clip_path, err.decode(errors="replace"))
            return False
        return True

    def _write_cv2(self, frames, clip_path, w, h) -> bool:
        # fallback when ffmpeg is absent. Uses the configured fourcc, may not play in a browser
        writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*self.codec), self.fps, (w, h))
        if not writer.isOpened():
            log.warning("clip sink could not open writer for %s", clip_path)
            return False
        try:
            for f in frames:
                writer.write(f)
        finally:
            writer.release()
        return True

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

    # Downscale to the width cap, keeping aspect. cap.read() hands out a fresh array per frame
    # and the tee awaits this before the next read, so the frame is already private to the
    # deque, no copy needed. resize already returns a new array
    def _fit(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if self.width and w > self.width:
            new_h = round(h * self.width / w)
            return cv2.resize(frame, (self.width, new_h), interpolation=cv2.INTER_AREA)
        return frame


@sinks.register("clips")
def _build(config: dict) -> ClipSink:
    return ClipSink(config)
