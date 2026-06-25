from __future__ import annotations
from typing import Protocol
from ..events import SightingEnded, SightingStarted
from ..registry import Registry


class Sink(Protocol):
    async def on_sighting_start(self, event: SightingStarted, feed_name: str) -> None: ...
    async def on_sighting_end(self, event: SightingEnded) -> None: ...


# A sink that wants the full frame stream (not just sighting events) implements on_frame.
# The runner tees every captured frame to these so a sink can keep a rolling buffer for
# clip recording. Detection still runs on a subsample, so on_frame is the only place a sink
# sees frames between the peak of one sighting and the next
class FrameSink(Protocol):
    async def on_frame(self, feed_id: str, ts: float, frame) -> None: ...


sinks: Registry[Sink] = Registry("detstream.sinks")

# console, dataset, and comment have no optional deps (comment posts with httpx, a core
# dep). The others self-register but their factories import their client libraries lazily,
# so a deployment only pays for the sinks it enables
from . import console  # noqa: E402,F401
from . import dataset  # noqa: E402,F401
from . import comment  # noqa: E402,F401
from . import clips  # noqa: E402,F401

try:
    from . import supabase  # noqa: F401
except ImportError:
    pass
try:
    from . import discord  # noqa: F401
except ImportError:
    pass
