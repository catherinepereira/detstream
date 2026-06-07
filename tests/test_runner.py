import asyncio
import pytest
from detstream import runner


class RecordingSink:
    def __init__(self, fail=False):
        self.fail = fail
        self.starts = 0

    async def on_sighting_start(self, event, feed_name):
        self.starts += 1
        if self.fail:
            raise RuntimeError("sink down")

    async def on_sighting_end(self, event):
        pass


def test_dispatch_isolates_a_failing_sink():
    bad = RecordingSink(fail=True)
    good = RecordingSink()

    # The bad sink raising must not stop the good one or propagate
    asyncio.run(
        runner._dispatch([bad, good], lambda s: s.on_sighting_start(object(), "feed"))
    )

    assert bad.starts == 1
    assert good.starts == 1


def test_supervise_feed_returns_when_source_finishes(monkeypatch):
    # A finite source ends cleanly, so the supervisor returns rather than restarting
    calls = 0

    async def fake_watch(feed, detector, sinks):
        nonlocal calls
        calls += 1

    feed = type("F", (), {"id": "f1"})()
    monkeypatch.setattr(runner, "watch_feed", fake_watch)
    asyncio.run(runner.supervise_feed(feed, None, []))
    assert calls == 1


def test_supervise_feed_restarts_after_a_crash(monkeypatch):
    feed = type("F", (), {"id": "f1"})()
    attempts = 0

    async def flaky_watch(feed, detector, sinks):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("boom")
        # third attempt succeeds and returns

    # No waiting between restarts
    async def no_sleep(_):
        return None

    monkeypatch.setattr(runner, "watch_feed", flaky_watch)
    monkeypatch.setattr(runner.asyncio, "sleep", no_sleep)
    asyncio.run(runner.supervise_feed(feed, None, []))
    assert attempts == 3


def test_supervise_feed_propagates_cancellation(monkeypatch):
    feed = type("F", (), {"id": "f1"})()

    async def cancelling_watch(feed, detector, sinks):
        raise asyncio.CancelledError()

    monkeypatch.setattr(runner, "watch_feed", cancelling_watch)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(runner.supervise_feed(feed, None, []))
