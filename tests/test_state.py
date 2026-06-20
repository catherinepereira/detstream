import numpy as np
from detstream.events import SightingEnded, SightingStarted
from detstream.state import SightingTracker

FRAME = np.zeros((4, 4, 3), dtype=np.uint8)


def make_tracker(**kwargs):
    defaults = dict(
        feed_id="feed",
        enter_frames=3,
        exit_frames=5,
        cooldown_s=120.0,
    )
    defaults.update(kwargs)
    return SightingTracker(**defaults)


def hit(s, conf, now, frame=FRAME, box=None):
    return s.update(True, conf, frame, now, box)


def miss(s, conf, now, frame=FRAME, box=None):
    return s.update(False, conf, frame, now, box)


def test_fires_once_on_enter_after_n_frames():
    s = make_tracker()
    assert hit(s, 0.9, now=0) is None
    assert hit(s, 0.9, now=1) is None
    event = hit(s, 0.9, now=2)
    assert isinstance(event, SightingStarted)
    assert event.confidence == 0.9
    assert hit(s, 0.9, now=3) is None


def test_misses_do_not_count():
    s = make_tracker()
    for t in range(10):
        assert miss(s, 0.39, now=t) is None
    assert not s.present


def test_one_dip_resets_enter_counter():
    s = make_tracker()
    hit(s, 0.9, now=0)
    hit(s, 0.9, now=1)
    miss(s, 0.1, now=2)
    assert hit(s, 0.9, now=3) is None
    assert hit(s, 0.9, now=4) is None
    assert isinstance(hit(s, 0.9, now=5), SightingStarted)


def test_exit_after_m_sub_threshold_frames():
    s = make_tracker()
    for t in range(3):
        hit(s, 0.9, now=t)
    assert s.present
    for t in range(3, 7):
        assert miss(s, 0.1, now=t) is None
    event = miss(s, 0.1, now=7)
    assert isinstance(event, SightingEnded)
    assert not s.present


def test_peak_confidence_tracked_over_sighting():
    s = make_tracker()
    hit(s, 0.5, now=0)
    hit(s, 0.5, now=1)
    hit(s, 0.5, now=2)
    hit(s, 0.85, now=3)
    for t in range(4, 9):
        ev = miss(s, 0.1, now=t)
    assert isinstance(ev, SightingEnded)
    assert ev.peak_confidence == 0.85


def test_cooldown_blocks_second_alert():
    s = make_tracker(exit_frames=1, cooldown_s=100.0)
    for t in range(3):
        hit(s, 0.9, now=t)
    miss(s, 0.1, now=3)
    assert not s.present
    hit(s, 0.9, now=10)
    hit(s, 0.9, now=11)
    assert hit(s, 0.9, now=12) is None
    miss(s, 0.1, now=13)
    hit(s, 0.9, now=200)
    hit(s, 0.9, now=201)
    assert isinstance(hit(s, 0.9, now=202), SightingStarted)


def test_feed_id_carried_on_events():
    s = make_tracker(feed_id="eagles")
    for t in range(3):
        ev = hit(s, 0.9, now=t)
    assert isinstance(ev, SightingStarted)
    assert ev.feed_id == "eagles"


def test_stays_absent_when_never_detected():
    s = make_tracker()
    for t in range(20):
        assert miss(s, 0.0, now=t) is None
    assert not s.present


def test_peak_frame_is_the_highest_confidence_frame():
    s = make_tracker(enter_frames=1, exit_frames=1)
    enter = np.full((4, 4, 3), 1, dtype=np.uint8)
    peak = np.full((4, 4, 3), 9, dtype=np.uint8)
    hit(s, 0.5, now=0, frame=enter)
    hit(s, 0.95, now=1, frame=peak)
    hit(s, 0.8, now=2, frame=np.full((4, 4, 3), 5, dtype=np.uint8))
    ended = miss(s, 0.1, now=3)
    assert isinstance(ended, SightingEnded)
    assert ended.peak_confidence == 0.95
    assert np.array_equal(ended.peak_frame, peak)


def test_peak_box_travels_with_peak_frame():
    s = make_tracker(enter_frames=1, exit_frames=1)
    hit(s, 0.5, now=0, box=(0, 0, 1, 1))
    hit(s, 0.95, now=1, box=(10, 10, 20, 20))
    hit(s, 0.8, now=2, box=(5, 5, 6, 6))
    ended = miss(s, 0.1, now=3)
    assert isinstance(ended, SightingEnded)
    # The box from the peak-confidence frame, not the last one
    assert ended.peak_box == (10, 10, 20, 20)


def test_started_label_is_the_entry_class():
    s = make_tracker(enter_frames=2, exit_frames=1)
    assert s.update(True, 0.9, FRAME, 0, None, "deer") is None
    started = s.update(True, 0.9, FRAME, 1, None, "deer")
    assert isinstance(started, SightingStarted)
    assert started.label == "deer"


def test_ended_label_is_the_peak_class():
    s = make_tracker(enter_frames=1, exit_frames=1)
    # Entry class is "deer", but the peak-confidence frame is a "fox"; the ended event
    # reports the peak's class, mirroring peak_frame/peak_box
    s.update(True, 0.5, FRAME, 0, None, "deer")
    s.update(True, 0.95, FRAME, 1, None, "fox")
    s.update(True, 0.8, FRAME, 2, None, "raccoon")
    ended = s.update(False, 0.1, FRAME, 3, None, None)
    assert isinstance(ended, SightingEnded)
    assert ended.label == "fox"


def test_label_resets_between_sightings():
    s = make_tracker(enter_frames=1, exit_frames=1, cooldown_s=0.0)
    s.update(True, 0.9, FRAME, 0, None, "deer")
    ended = s.update(False, 0.1, FRAME, 1, None, None)
    assert isinstance(ended, SightingEnded)
    # A second sighting with no label must not inherit "deer" from the first
    s.update(True, 0.9, FRAME, 2, None, None)
    ended2 = s.update(False, 0.1, FRAME, 3, None, None)
    assert isinstance(ended2, SightingEnded)
    assert ended2.label is None
