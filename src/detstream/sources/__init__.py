from __future__ import annotations
import asyncio
import contextlib
import logging
import threading
import time
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Callable, Protocol
import cv2
import numpy as np
from ..registry import Registry

log = logging.getLogger(__name__)


class Source(Protocol):
    def frames(self) -> AsyncIterator[tuple[float, np.ndarray]]: ...


# How long VideoCapture reads can keep failing, or keep returning the same frozen frame,
# before the stream is treated as down rather than emitting a stale frame
STREAM_DOWN_AFTER_S = 15.0
RECONNECT_BACKOFF_S = (2.0, 5.0, 15.0, 30.0, 60.0)


# A cheap signature for spotting a frozen stream. A live feed jitters every frame from
# sensor noise and compression, so two reads sharing a signature means the decoder is
# handing back the same buffered frame. A coarse stride keeps this fast on 1080p frames
def _frame_signature(frame: np.ndarray) -> int:
    return hash(frame[::16, ::16].tobytes())


async def capture_frames(
    open_capture: Callable[[], "cv2.VideoCapture"],
    interval_s: float,
    *,
    label: str,
    reconnect: bool = True,
    live: bool = False,
) -> AsyncIterator[tuple[float, np.ndarray]]:
    """Read frames from an OpenCV VideoCapture with reconnect + frozen-frame handling.

    open_capture is called to (re)open the capture. Sources that need to re-resolve a
    URL (YouTube) pass a closure that re-resolves. Sources reading a finite file pass
    reconnect=False so EOF ends the stream instead of looping forever.

    live=True is for sources that produce frames in real time (streams, devices). Those
    are drained on a thread and each yield is the newest frame. A file is read in order
    instead, one frame per interval.
    """
    attempt = 0
    while True:
        try:
            cap = await asyncio.to_thread(open_capture)
        except Exception as e:
            if not reconnect:
                raise
            backoff = RECONNECT_BACKOFF_S[min(attempt, len(RECONNECT_BACKOFF_S) - 1)]
            log.warning("%s: open failed: %s; retrying in %ss", label, e, backoff)
            attempt += 1
            await asyncio.sleep(backoff)
            continue

        attempt = 0
        if live:
            frames = _live_frames(cap, interval_s, label)
        else:
            frames = _sequential_frames(cap, interval_s, label, reconnect)
        async with contextlib.aclosing(frames):
            async for item in frames:
                yield item
        if not reconnect:
            return


# Reads frames in order, one per interval. Right for a file, wrong for a live stream: a
# 30fps stream sampled this way plays at 1/30 speed and falls further behind every second
async def _sequential_frames(
    cap, interval_s: float, label: str, reconnect: bool
) -> AsyncGenerator[tuple[float, np.ndarray], None]:
    last_ok = time.monotonic()
    last_change = last_ok
    last_sig: int | None = None
    try:
        while True:
            ok, frame = await asyncio.to_thread(cap.read)
            now = time.monotonic()
            if ok and frame is not None:
                last_ok = now
                # A feed that keeps returning the same frame while reads still succeed is
                # treated as down, which reconnects
                sig = _frame_signature(frame)
                if sig != last_sig:
                    last_sig = sig
                    last_change = now
                elif reconnect and now - last_change > STREAM_DOWN_AFTER_S:
                    log.warning("%s: stream frozen; reconnecting", label)
                    return
                yield time.time(), frame
                await asyncio.sleep(interval_s)
                continue
            if not reconnect:
                return
            if now - last_ok > STREAM_DOWN_AFTER_S:
                log.warning("%s: stream down; reconnecting", label)
                return
            await asyncio.sleep(1.0)
    finally:
        await asyncio.to_thread(cap.release)


# How far a frame's timestamp can run ahead of the playback clock before it is treated as a
# jump in the stream (skipped segments) and the clock restarts from it
PACE_REBASE_S = 1.0


# Drains a capture on its own thread and keeps only the newest frame. The thread is the only
# one that touches the capture, and it releases it on exit.
# HLS arrives a segment at a time and decodes faster than real time, so reads are paced by
# frame timestamp. Unpaced, the newest frame would only change once per segment
class _LatestFrame:
    def __init__(self, cap):
        self._cap = cap
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._last_ok = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        # (wall clock, stream position) the playback clock counts from
        base: tuple[float, float] | None = None
        try:
            while not self._stop.is_set():
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    self._stop.wait(0.5)
                    continue
                now = time.monotonic()
                with self._lock:
                    self._frame = frame
                    self._seq += 1
                    self._last_ok = now
                # A source without timestamps (some devices) reports 0 and is never paced
                pos = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                if base is None:
                    base = (now, pos)
                    continue
                ahead = (pos - base[1]) - (now - base[0])
                if ahead > PACE_REBASE_S:
                    base = (now, pos)
                elif ahead > 0:
                    self._stop.wait(ahead)
        finally:
            self._cap.release()

    def latest(self) -> tuple[np.ndarray | None, int, float]:
        with self._lock:
            return self._frame, self._seq, self._last_ok

    # Waits for the thread, which can take up to one read timeout if a read is blocked
    def close(self) -> None:
        self._stop.set()
        self._thread.join()


async def _live_frames(
    cap, interval_s: float, label: str
) -> AsyncGenerator[tuple[float, np.ndarray], None]:
    reader = _LatestFrame(cap)
    last_seq = 0
    last_sig: int | None = None
    last_change = time.monotonic()
    try:
        while True:
            frame, seq, last_ok = reader.latest()
            now = time.monotonic()
            if seq == last_seq or frame is None:
                if now - last_ok > STREAM_DOWN_AFTER_S:
                    log.warning("%s: stream down; reconnecting", label)
                    return
                await asyncio.sleep(0.1)
                continue
            last_seq = seq
            sig = _frame_signature(frame)
            if sig != last_sig:
                last_sig = sig
                last_change = now
            elif now - last_change > STREAM_DOWN_AFTER_S:
                log.warning("%s: stream frozen; reconnecting", label)
                return
            yield time.time(), frame
            await asyncio.sleep(interval_s)
    finally:
        await asyncio.to_thread(reader.close)


sources: Registry[Source] = Registry("detstream.sources")

from . import file_device, stream, youtube  # noqa: E402,F401
