from __future__ import annotations
import os
import httpx
from ..events import SightingEnded, SightingStarted
from . import sinks

DEFAULT_COLOR = 0x6EC9A9


# Posts a rich embed on each new sighting with feed name and peak confidence. 
# Honors the per-feed cooldown upstream in SightingTracker
# A watch live link is per-feed, so it is passed via config.watch_urls
class DiscordSink:
    def __init__(self, config: dict):
        self.webhook_url = os.environ.get("DETSTREAM_DISCORD_WEBHOOK_URL", "")
        if not self.webhook_url:
            raise ValueError("discord sink needs DETSTREAM_DISCORD_WEBHOOK_URL")
        self.color = config.get("color", DEFAULT_COLOR)
        self.watch_urls: dict[str, str] = config.get("watch_urls", {})

    async def on_sighting_start(self, event: SightingStarted, feed_name: str) -> None:
        embed = {
            "title": f"spotted on {feed_name}",
            "color": self.color,
            "fields": [{"name": "confidence", "value": f"{event.confidence:.0%}", "inline": True}],
        }
        watch_url = self.watch_urls.get(event.feed_id)
        if watch_url:
            embed["url"] = watch_url
            embed["description"] = f"[watch live]({watch_url})"
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(self.webhook_url, json={"embeds": [embed]})

    async def on_sighting_end(self, event: SightingEnded) -> None:
        return


@sinks.register("discord")
def _build(config: dict) -> DiscordSink:
    return DiscordSink(config)
