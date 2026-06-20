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


def test_dataset_sink_writes_clean_frame(tmp_path):
    import asyncio
    import cv2
    from detstream.sinks import sinks

    sink = sinks.create("dataset", {"dir": str(tmp_path)})
    # A frame with distinct channels, plus a box that must NOT be drawn on the saved image
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[:] = (10, 20, 30)
    asyncio.run(sink.on_sighting_end(SightingEnded("monterey", 0.82, frame, (1, 1, 5, 5))))

    files = list((tmp_path / "monterey").glob("*.jpg"))
    assert len(files) == 1
    saved = cv2.imread(str(files[0]))
    # The whole image is the flat fill, proving no box/label was drawn over it
    assert np.array_equal(np.unique(saved.reshape(-1, 3), axis=0), np.array([[10, 20, 30]]))


def test_dataset_sink_skips_when_no_frame(tmp_path):
    import asyncio
    from detstream.sinks import sinks

    sink = sinks.create("dataset", {"dir": str(tmp_path)})
    asyncio.run(sink.on_sighting_end(SightingEnded("f1", 0.5, None)))

    assert not list(tmp_path.rglob("*.jpg"))


def test_supabase_annotates_thumbnail_but_keeps_source_raw(fake_supabase, monkeypatch):
    monkeypatch.setenv("DETSTREAM_SUPABASE_URL", "https://proj.supabase.co")
    monkeypatch.setenv("DETSTREAM_SUPABASE_KEY", "secret")
    from detstream.sinks.supabase import SupabaseSink

    sink = SupabaseSink({})
    captured = {}
    # Intercept the encode step to see the frame after annotation, without hitting storage
    monkeypatch.setattr(sink, "_downscale", lambda f: f)

    def fake_imencode(ext, frame, params):
        captured["frame"] = frame.copy()
        return False, None

    monkeypatch.setattr("detstream.sinks.supabase.cv2.imencode", fake_imencode)

    raw = np.zeros((40, 40, 3), dtype=np.uint8)
    url = sink._upload_thumbnail("f1", raw, 0.82, (5, 5, 30, 30))

    # Encode failed (stubbed), so no URL, but the frame handed to encode had the box drawn:
    # the source array is untouched and the annotated copy differs from it
    assert url is None
    assert not np.array_equal(captured["frame"], raw)
    assert np.array_equal(raw, np.zeros((40, 40, 3), dtype=np.uint8))


def test_supabase_thumbnail_unboxed_when_no_box(fake_supabase, monkeypatch):
    monkeypatch.setenv("DETSTREAM_SUPABASE_URL", "https://proj.supabase.co")
    monkeypatch.setenv("DETSTREAM_SUPABASE_KEY", "secret")
    from detstream.sinks.supabase import SupabaseSink

    sink = SupabaseSink({})
    captured = {}
    monkeypatch.setattr(sink, "_downscale", lambda f: f)

    def fake_imencode(ext, frame, params):
        captured["frame"] = frame.copy()
        return False, None

    monkeypatch.setattr("detstream.sinks.supabase.cv2.imencode", fake_imencode)

    raw = np.zeros((40, 40, 3), dtype=np.uint8)
    # A sighting can be present with no box; the thumbnail is then just unannotated, not an error
    sink._upload_thumbnail("f1", raw, 0.82, None)

    assert np.array_equal(captured["frame"], raw)


class _FakeCommentClient:
    posted = {}

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json):
        type(self).posted = {"url": url, "json": json}


def test_comment_reads_webhook_from_env(monkeypatch):
    monkeypatch.setenv("DETSTREAM_COMMENT_WEBHOOK_URL", "https://chat/wh")
    from detstream.sinks.comment import CommentSink

    sink = CommentSink({"min_confidence": 0.5})

    assert sink.webhook_url == "https://chat/wh"
    assert sink.min_confidence == 0.5


def test_comment_missing_webhook_raises(monkeypatch):
    monkeypatch.delenv("DETSTREAM_COMMENT_WEBHOOK_URL", raising=False)
    from detstream.sinks.comment import CommentSink

    with pytest.raises(ValueError):
        CommentSink({})


def test_comment_posts_species_line(monkeypatch):
    monkeypatch.setenv("DETSTREAM_COMMENT_WEBHOOK_URL", "https://chat/wh")
    from detstream.sinks import comment as comment_mod

    monkeypatch.setattr(comment_mod.httpx, "AsyncClient", _FakeCommentClient)
    # Pin the clock so the formatted time is deterministic
    fixed = comment_mod.datetime(2026, 6, 19, 14, 32)

    class _FixedDatetime:
        @classmethod
        def now(cls):
            return fixed

    monkeypatch.setattr(comment_mod, "datetime", _FixedDatetime)
    sink = comment_mod.CommentSink({})

    import asyncio

    asyncio.run(sink.on_sighting_start(SightingStarted("f1", 0.91, "deer"), "Forest Cam"))

    assert _FakeCommentClient.posted["url"] == "https://chat/wh"
    assert _FakeCommentClient.posted["json"] == {"content": "deer spotted at 14:32"}


def test_comment_falls_back_to_feed_name_when_unlabelled(monkeypatch):
    monkeypatch.setenv("DETSTREAM_COMMENT_WEBHOOK_URL", "https://chat/wh")
    from detstream.sinks import comment as comment_mod

    monkeypatch.setattr(comment_mod.httpx, "AsyncClient", _FakeCommentClient)
    sink = comment_mod.CommentSink({"template": "{label}"})

    import asyncio

    # A single-class detector leaves label None; the line still reads with the feed name
    asyncio.run(sink.on_sighting_start(SightingStarted("f1", 0.91, None), "Forest Cam"))

    assert _FakeCommentClient.posted["json"]["content"] == "Forest Cam"


def test_comment_skips_below_min_confidence(monkeypatch):
    monkeypatch.setenv("DETSTREAM_COMMENT_WEBHOOK_URL", "https://chat/wh")
    from detstream.sinks import comment as comment_mod

    _FakeCommentClient.posted = {}
    monkeypatch.setattr(comment_mod.httpx, "AsyncClient", _FakeCommentClient)
    sink = comment_mod.CommentSink({"min_confidence": 0.8})

    import asyncio

    asyncio.run(sink.on_sighting_start(SightingStarted("f1", 0.5, "deer"), "Forest Cam"))

    # Below the floor, nothing was posted
    assert _FakeCommentClient.posted == {}


def test_comment_registered():
    from detstream.sinks import sinks

    assert "comment" in sinks._factories
