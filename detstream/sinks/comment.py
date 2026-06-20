from __future__ import annotations
import os
from datetime import datetime
import httpx
from ..events import SightingEnded, SightingStarted
from . import sinks

DEFAULT_TEMPLATE = "{label} spotted at {time}"
DEFAULT_TIME_FORMAT = "%H:%M"


# Posts a short comment line on each new sighting, e.g. "deer spotted at 14:32", to a chat
# webhook (Discord/Slack and most chat services accept a {"content": ...} JSON body). The
# species comes from the detection label, so a multi-class feed says which animal it saw.
# Honors the per-feed cooldown upstream in SightingTracker.
class CommentSink:
    def __init__(self, config: dict):
        self.webhook_url = os.environ.get("DETSTREAM_COMMENT_WEBHOOK_URL", "")
        if not self.webhook_url:
            raise ValueError("comment sink needs DETSTREAM_COMMENT_WEBHOOK_URL")
        self.template = config.get("template", DEFAULT_TEMPLATE)
        self.time_format = config.get("time_format", DEFAULT_TIME_FORMAT)
        self.min_confidence = float(config.get("min_confidence", 0.0))

    async def on_sighting_start(self, event: SightingStarted, feed_name: str) -> None:
        if event.confidence < self.min_confidence:
            return
        # Single-class detectors leave label None; fall back to the feed name so the line
        # still reads as a sentence ("Forest Cam spotted at 14:32")
        label = event.label or feed_name
        text = self.template.format(
            label=label,
            time=datetime.now().strftime(self.time_format),
            feed_name=feed_name,
            confidence=event.confidence,
        )
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(self.webhook_url, json={"content": text})

    async def on_sighting_end(self, event: SightingEnded) -> None:
        return


@sinks.register("comment")
def _build(config: dict) -> CommentSink:
    return CommentSink(config)
