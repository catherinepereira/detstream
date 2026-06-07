import pytest
from detstream.registry import Registry


def test_register_and_create():
    reg: Registry[str] = Registry("test.group")

    @reg.register("upper")
    def _build(config: dict) -> str:
        return config["text"].upper()

    assert reg.create("upper", {"text": "hi"}) == "HI"


def test_unknown_name_raises_with_known_listed():
    reg: Registry[str] = Registry("test.group")

    @reg.register("a")
    def _a(config: dict) -> str:
        return "a"

    with pytest.raises(ValueError) as exc:
        reg.create("missing", {})
    assert "missing" in str(exc.value)
    assert "a" in str(exc.value)


def test_duplicate_registration_raises():
    reg: Registry[str] = Registry("test.group")

    @reg.register("dup")
    def _one(config: dict) -> str:
        return "1"

    with pytest.raises(ValueError):

        @reg.register("dup")
        def _two(config: dict) -> str:
            return "2"


def test_builtins_self_register():
    from detstream.sinks import sinks
    from detstream.sources import sources

    sinks.create("console", {})
    assert "console" in sinks._factories
    assert "supabase" in sinks._factories
    assert "discord" in sinks._factories
    assert "youtube" in sources._factories
    assert "stream" in sources._factories
    assert "file" in sources._factories
