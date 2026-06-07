from __future__ import annotations
from typing import Protocol
from ..events import SightingEnded, SightingStarted
from ..registry import Registry


class Sink(Protocol):
    async def on_sighting_start(self, event: SightingStarted, feed_name: str) -> None: ...
    async def on_sighting_end(self, event: SightingEnded) -> None: ...


sinks: Registry[Sink] = Registry("detstream.sinks")

# console has no optional deps. The others self-register but their factories import their
# client libraries lazily, so a deployment only pays for the sinks it enables
from . import console  # noqa: E402,F401

try:
    from . import supabase  # noqa: F401
except ImportError:
    pass
try:
    from . import discord  # noqa: F401
except ImportError:
    pass
