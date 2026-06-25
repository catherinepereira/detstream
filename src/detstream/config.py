from __future__ import annotations
import os
import re
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, field_validator

_ENV_REF = re.compile(r"\$\{(\w+)\}")


class ComponentConfig(BaseModel):
    type: str
    # Everything else is the component's own config sub-block, passed through to its
    # factory. Sources/detectors vary by type, so the schema is intentionally open
    model_config = {"extra": "allow"}

    def options(self) -> dict:
        return self.model_dump(exclude={"type"})


class DebounceConfig(BaseModel):
    sample_interval_s: float = 2.0
    enter_frames: int = 3
    exit_frames: int = 5
    cooldown_s: float = 120.0
    # Frames per second the runner pulls from the source when a clip-recording sink is
    # attached. Detection still runs every sample_interval_s; the extra frames only feed
    # on_frame so a clip looks like video. 0 keeps the old behavior of reading at the
    # detection cadence
    tee_fps: float = 0.0


class FeedConfig(BaseModel):
    id: str
    name: str
    source: ComponentConfig
    detector: ComponentConfig
    debounce: DebounceConfig = Field(default_factory=DebounceConfig)
    sinks: list[str] = Field(default_factory=lambda: ["console"])


class AppConfig(BaseModel):
    feeds: list[FeedConfig]
    # Shared sink settings keyed by sink name, e.g. {"supabase": {...}, "discord": {...}}.
    # A feed enables a sink by listing its name, the settings come from here
    sinks: dict[str, dict] = Field(default_factory=dict)

    # A `sinks:` block with everything commented out parses as None
    @field_validator("sinks", mode="before")
    @classmethod
    def _none_sinks_to_empty(cls, v):
        return v or {}


def _expand_env(value):
    if isinstance(value, str):
        return _ENV_REF.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _load_dotenv(path: Path) -> None:
    env_file = path.parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if value and value[0] in "'\"":
            # Quoted value: everything up to the matching quote is literal
            quote = value[0]
            end = value.find(quote, 1)
            value = value[1:end] if end != -1 else value[1:]
        else:
            # Unquoted: an inline ` # comment` is not part of the value
            value = value.split(" #", 1)[0].strip()
        os.environ.setdefault(key, value)


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    _load_dotenv(path)
    data = _expand_env(yaml.safe_load(path.read_text()))
    return AppConfig(**data)
