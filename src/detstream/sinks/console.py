from __future__ import annotations
import logging
from ..events import SightingEnded, SightingStarted
from . import sinks

log = logging.getLogger(__name__)


class ConsoleSink:
    async def on_sighting_start(self, event: SightingStarted, feed_name: str) -> None:
        log.info("sighting on %s (%.2f)", feed_name, event.confidence)

    async def on_sighting_end(self, event: SightingEnded) -> None:
        log.info("ended on %s (peak %.2f)", event.feed_id, event.peak_confidence)


@sinks.register("console")
def _build(config: dict) -> ConsoleSink:
    return ConsoleSink()
