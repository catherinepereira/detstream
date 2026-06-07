import sys
import types
import numpy as np
import pytest
from detstream.events import SightingEnded, SightingStarted


@pytest.fixture
def fake_supabase(monkeypatch):
    created = {}

    def create_client(url, key):
        created["url"] = url
        created["key"] = key
        return object()

    module = types.ModuleType("supabase")
    module.create_client = create_client
    monkeypatch.setitem(sys.modules, "supabase", module)
    return created


def test_supabase_reads_secrets_from_env_tuning_from_config(fake_supabase, monkeypatch):
    monkeypatch.setenv("DETSTREAM_SUPABASE_URL", "https://proj.supabase.co")
    monkeypatch.setenv("DETSTREAM_SUPABASE_KEY", "secret")
    from detstream.sinks.supabase import SupabaseSink

    sink = SupabaseSink({"bucket": "thumbs", "detector_label": "yolo-ft", "thumbnail_width": 640})

    assert fake_supabase["url"] == "https://proj.supabase.co"
    assert sink.bucket == "thumbs"
    assert sink.detector_label == "yolo-ft"
    assert sink.thumbnail_width == 640


def test_supabase_config_defaults(fake_supabase, monkeypatch):
    monkeypatch.setenv("DETSTREAM_SUPABASE_URL", "https://proj.supabase.co")
    monkeypatch.setenv("DETSTREAM_SUPABASE_KEY", "secret")
    from detstream.sinks.supabase import SupabaseSink

    sink = SupabaseSink({})

    assert sink.bucket == "thumbnails"
    assert sink.detector_label == ""
    assert sink.thumbnail_width == 960
    assert sink.retention_hours == 3


def test_supabase_missing_credentials_raises(fake_supabase, monkeypatch):
    monkeypatch.delenv("DETSTREAM_SUPABASE_URL", raising=False)
    monkeypatch.delenv("DETSTREAM_SUPABASE_KEY", raising=False)
    from detstream.sinks.supabase import SupabaseSink

    with pytest.raises(ValueError):
        SupabaseSink({})


def test_supabase_downscale_only_when_wider_than_target(fake_supabase, monkeypatch):
    monkeypatch.setenv("DETSTREAM_SUPABASE_URL", "https://proj.supabase.co")
    monkeypatch.setenv("DETSTREAM_SUPABASE_KEY", "secret")
    from detstream.sinks.supabase import SupabaseSink

    sink = SupabaseSink({"thumbnail_width": 100})
    wide = np.zeros((50, 200, 3), dtype=np.uint8)
    narrow = np.zeros((50, 80, 3), dtype=np.uint8)

    assert sink._downscale(wide).shape == (25, 100, 3)
    assert sink._downscale(narrow) is narrow


def test_discord_reads_webhook_from_env(monkeypatch):
    monkeypatch.setenv("DETSTREAM_DISCORD_WEBHOOK_URL", "https://discord/wh")
    from detstream.sinks.discord import DiscordSink

    sink = DiscordSink({"color": 123, "watch_urls": {"f1": "https://watch/f1"}})

    assert sink.webhook_url == "https://discord/wh"
    assert sink.color == 123
    assert sink.watch_urls == {"f1": "https://watch/f1"}


def test_discord_missing_webhook_raises(monkeypatch):
    monkeypatch.delenv("DETSTREAM_DISCORD_WEBHOOK_URL", raising=False)
    from detstream.sinks.discord import DiscordSink

    with pytest.raises(ValueError):
        DiscordSink({})


def test_discord_posts_embed_with_watch_link(monkeypatch):
    monkeypatch.setenv("DETSTREAM_DISCORD_WEBHOOK_URL", "https://discord/wh")
    from detstream.sinks import discord as discord_mod

    posted = {}

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json):
            posted["url"] = url
            posted["json"] = json

    monkeypatch.setattr(discord_mod.httpx, "AsyncClient", FakeClient)
    sink = discord_mod.DiscordSink({"watch_urls": {"f1": "https://watch/f1"}})

    import asyncio

    asyncio.run(sink.on_sighting_start(SightingStarted("f1", 0.91), "Feed One"))

    assert posted["url"] == "https://discord/wh"
    embed = posted["json"]["embeds"][0]
    assert embed["url"] == "https://watch/f1"
    assert "Feed One" in embed["title"]


def test_console_sink_builds_and_handles_events():
    from detstream.sinks import sinks

    sink = sinks.create("console", {})
    import asyncio

    asyncio.run(sink.on_sighting_start(SightingStarted("f1", 0.9), "Feed One"))
    asyncio.run(sink.on_sighting_end(SightingEnded("f1", 0.9, None)))
