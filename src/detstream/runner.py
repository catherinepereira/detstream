from __future__ import annotations
import asyncio
import logging
import os
from .config import AppConfig, FeedConfig
from .detectors import Detector, detectors
from .events import SightingStarted
from .sinks import Sink, sinks as sink_registry
from .sources import sources as source_registry
from .state import SightingTracker

log = logging.getLogger(__name__)


def build_sinks(feed: FeedConfig, app: AppConfig) -> list[Sink]:
    built: list[Sink] = []
    for name in feed.sinks:
        config = app.sinks.get(name, {})
        built.append(sink_registry.create(name, config))
    return built


# Restart backoff for a feed whose watch loop crashed
FEED_RESTART_BACKOFF_S = (5.0, 15.0, 30.0, 60.0)


async def _dispatch(sinks: list[Sink], call) -> None:
    # One sink failing (a network blip on the supabase POST) must not stop the others or
    # kill the feed, so each call is awaited independently and its error logged
    results = await asyncio.gather(*(call(s) for s in sinks), return_exceptions=True)
    for sink, result in zip(sinks, results):
        if isinstance(result, Exception):
            log.warning("sink %s failed: %s", type(sink).__name__, result)


# Frame-sink failures are counted per sink type so a sink that throws on every frame logs on a
# backoff instead of spamming at the tee rate
_tee_failures: dict[str, int] = {}


# Tee one captured frame to every frame sink. Runs on the per-frame hot path, so a sink
# raising here is logged rather than allowed to kill the feed. on_frame is expected to be
# cheap (append to a buffer), a sink that blocks here throttles capture for the whole feed
async def _tee(frame_sinks: list[Sink], feed_id: str, ts: float, frame) -> None:
    results = await asyncio.gather(
        *(s.on_frame(feed_id, ts, frame) for s in frame_sinks), return_exceptions=True
    )
    for sink, result in zip(frame_sinks, results):
        if isinstance(result, Exception):
            name = type(sink).__name__
            n = _tee_failures.get(name, 0) + 1
            _tee_failures[name] = n
            # log the first failure, then every 100th, so a persistently broken sink at 30fps
            # does not fill the log
            if n == 1 or n % 100 == 0:
                log.warning("frame sink %s failed (%d times): %s", name, n, result)


async def watch_feed(feed: FeedConfig, detector: Detector, sinks: list[Sink]) -> None:
    tracker = SightingTracker(
        feed_id=feed.id,
        enter_frames=feed.debounce.enter_frames,
        exit_frames=feed.debounce.exit_frames,
        cooldown_s=feed.debounce.cooldown_s,
    )
    loop = asyncio.get_running_loop()

    # When a clip sink wants every frame, read at the faster tee rate and run detection only
    # every detect_every-th frame so the detector keeps its sample_interval_s cadence. With
    # no frame sink, read at the detection cadence as before
    frame_sinks = [s for s in sinks if hasattr(s, "on_frame")]
    sample_s = feed.debounce.sample_interval_s
    if frame_sinks and feed.debounce.tee_fps > 0:
        interval_s = 1.0 / feed.debounce.tee_fps
        detect_every = max(1, round(sample_s * feed.debounce.tee_fps))
    else:
        if frame_sinks:
            log.warning(
                "feed %s has a frame sink but tee_fps is 0, clips get only the detection "
                "subsample and will be choppy. Set debounce.tee_fps to the clip fps",
                feed.id,
            )
        interval_s = sample_s
        detect_every = 1

    source = source_registry.create(
        feed.source.type,
        {**feed.source.options(), "interval_s": interval_s},
    )
    i = 0
    async for ts, frame in source.frames():
        if frame_sinks:
            await _tee(frame_sinks, feed.id, ts, frame)
        i += 1
        if i % detect_every != 0:
            continue
        det = await asyncio.to_thread(detector.detect, frame)
        # The tracker keeps the raw frame plus the box, so a sink can annotate at use time
        # while the dataset sink gets clean pixels
        event = tracker.update(det.present, det.confidence, frame, loop.time(), det.box, det.label)
        if event is None:
            continue
        if isinstance(event, SightingStarted):
            await _dispatch(sinks, lambda s: s.on_sighting_start(event, feed.name))
        else:
            await _dispatch(sinks, lambda s: s.on_sighting_end(event))


async def supervise_feed(feed: FeedConfig, detector: Detector, sinks: list[Sink]) -> None:
    attempt = 0
    while True:
        try:
            await watch_feed(feed, detector, sinks)
            return
        except asyncio.CancelledError:
            raise
        except Exception as e:
            backoff = FEED_RESTART_BACKOFF_S[min(attempt, len(FEED_RESTART_BACKOFF_S) - 1)]
            log.exception("feed %s crashed: %s; restarting in %ss", feed.id, e, backoff)
            attempt += 1
            await asyncio.sleep(backoff)


CLEANUP_INTERVAL_S = float(os.environ.get("DETSTREAM_CLEANUP_INTERVAL_S", str(60 * 60)))


# Run each sink's cleanup() on an interval (hourly by default)
async def cleanup_loop(sinks: list[Sink]) -> None:
    targets = [s for s in sinks if hasattr(s, "cleanup")]
    if not targets:
        return
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_S)
        for sink in targets:
            try:
                await asyncio.to_thread(sink.cleanup)
            except Exception as e:
                log.warning("cleanup failed for %s: %s", type(sink).__name__, e)


async def run(app: AppConfig) -> None:
    if not app.feeds:
        log.error("no feeds configured")
        return

    tasks = []
    all_sinks: list[Sink] = []
    for feed in app.feeds:
        detector = detectors.create(feed.detector.type, feed.detector.options())
        sinks = build_sinks(feed, app)
        all_sinks.extend(sinks)
        log.info("watching %s with %d sink(s)", feed.id, len(sinks))
        tasks.append(asyncio.create_task(supervise_feed(feed, detector, sinks)))

    # Dedupe sinks of the same type so cleanup runs once
    unique_sinks = list({type(s): s for s in all_sinks}.values())
    tasks.append(asyncio.create_task(cleanup_loop(unique_sinks)))

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        for t in tasks:
            t.cancel()
        raise
