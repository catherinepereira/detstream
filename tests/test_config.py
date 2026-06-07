from detstream.config import AppConfig, load_config


def test_minimal_feed_defaults():
    app = AppConfig(
        feeds=[
            {
                "id": "f1",
                "name": "Feed One",
                "source": {"type": "youtube", "url": "http://x"},
                "detector": {"type": "yolo-world", "prompt": "otter"},
            }
        ]
    )
    feed = app.feeds[0]
    assert feed.sinks == ["console"]
    assert feed.debounce.enter_frames == 3
    assert feed.detector.options()["prompt"] == "otter"
    assert app.sinks == {}


def test_commented_out_sinks_block_is_empty(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "feeds:\n"
        "  - id: f1\n"
        "    name: Feed One\n"
        "    source: { type: youtube, url: 'http://x' }\n"
        "    detector: { type: yolo-world, prompt: otter }\n"
        "    sinks: [console]\n"
        "sinks:\n"
        "  # supabase:\n"
        "  #   url: x\n"
    )
    app = load_config(cfg)
    assert app.sinks == {}
    assert app.feeds[0].sinks == ["console"]


def test_sink_settings_block_is_passed_through(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "feeds:\n"
        "  - id: f1\n"
        "    name: Feed One\n"
        "    source: { type: youtube, url: 'http://x' }\n"
        "    detector: { type: yolo-world, prompt: otter }\n"
        "    sinks: [console, supabase]\n"
        "sinks:\n"
        "  supabase: { bucket: thumbs, detector_label: yolo-world, retention_hours: 3 }\n"
    )
    app = load_config(cfg)
    assert app.feeds[0].sinks == ["console", "supabase"]
    assert app.sinks["supabase"] == {
        "bucket": "thumbs",
        "detector_label": "yolo-world",
        "retention_hours": 3,
    }


def test_env_refs_expand_from_environment(tmp_path, monkeypatch):
    # ${VAR} in the config resolves from the environment, an unset var expands to empty
    monkeypatch.setenv("MY_STREAM_URL", "rtsp://cam/1")
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "feeds:\n"
        "  - id: f1\n"
        "    name: Feed One\n"
        "    source: { type: stream, url: '${MY_STREAM_URL}' }\n"
        "    detector: { type: yolo-world, prompt: otter }\n"
    )
    app = load_config(cfg)
    assert app.feeds[0].source.options()["url"] == "rtsp://cam/1"


def test_dotenv_beside_config_is_loaded(tmp_path, monkeypatch):
    # load_config reads a .env next to the config, already-set vars are not overridden
    monkeypatch.delenv("FROM_DOTENV", raising=False)
    (tmp_path / ".env").write_text("FROM_DOTENV=hello\n")
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "feeds:\n"
        "  - id: f1\n"
        "    name: Feed One\n"
        "    source: { type: youtube, url: 'http://x' }\n"
        "    detector: { type: yolo-world, prompt: otter }\n"
    )
    load_config(cfg)
    import os

    assert os.environ["FROM_DOTENV"] == "hello"


def test_dotenv_parses_comments_quotes_and_export(tmp_path, monkeypatch):
    import os

    for k in ("DE_PLAIN", "DE_INLINE", "DE_QUOTED_HASH", "DE_EXPORTED"):
        monkeypatch.delenv(k, raising=False)
    (tmp_path / ".env").write_text(
        "DE_PLAIN=value\n"
        "DE_INLINE=value  # trailing comment\n"
        "DE_QUOTED_HASH='a#b c'\n"
        "export DE_EXPORTED=ex\n"
        "# whole-line comment\n"
    )
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "feeds:\n"
        "  - id: f1\n"
        "    name: Feed One\n"
        "    source: { type: youtube, url: 'http://x' }\n"
        "    detector: { type: yolo-world, prompt: otter }\n"
    )
    load_config(cfg)

    assert os.environ["DE_PLAIN"] == "value"
    assert os.environ["DE_INLINE"] == "value"  # inline comment stripped
    assert os.environ["DE_QUOTED_HASH"] == "a#b c"  # # inside quotes survives
    assert os.environ["DE_EXPORTED"] == "ex"  # export prefix stripped
