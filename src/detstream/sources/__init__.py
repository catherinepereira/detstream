from __future__ import annotations
import asyncio
import logging
import time
from collections.abc import AsyncIterator
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
) -> AsyncIterator[tuple[float, np.ndarray]]:
    """Read frames from an OpenCV VideoCapture with reconnect + frozen-frame handling.

    open_capture is called to (re)open the capture. Sources that need to re-resolve a
    URL (YouTube) pass a closure that re-resolves. Sources reading a finite file pass
    reconnect=False so EOF ends the stream instead of looping forever.
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
        last_ok = time.monotonic()
        last_change = last_ok
        last_sig: int | None = None
        try:
            while True:
                ok, frame = await asyncio.to_thread(cap.read)
                now = time.monotonic()
                if ok and frame is not None:
                    last_ok = now
                    # A stream can fall behind the live edge or freeze while reads still
                    # succeed, so a feed that keeps returning the same frame is treated as
                    # down too, which reconnects and re-seeks to the live edge
                    sig = _frame_signature(frame)
                    if sig != last_sig:
                        last_sig = sig
                        last_change = now
                    elif reconnect and now - last_change > STREAM_DOWN_AFTER_S:
                        log.warning("%s: stream frozen; reconnecting", label)
                        break
                    yield time.time(), frame
                    await asyncio.sleep(interval_s)
                    continue
                if not reconnect:
                    return
                if now - last_ok > STREAM_DOWN_AFTER_S:
                    log.warning("%s: stream down; reconnecting", label)
                    break
                await asyncio.sleep(1.0)
        finally:
            await asyncio.to_thread(cap.release)


sources: Registry[Source] = Registry("detstream.sources")

from . import file_device, stream, youtube  # noqa: E402,F401
