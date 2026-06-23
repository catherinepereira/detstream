from __future__ import annotations
import logging
from importlib.metadata import entry_points
from typing import Callable, Generic, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")

# A factory takes the component's config sub-block (a plain dict) and returns an
# instance. Built-ins register a factory under a name, config selects by that name
Factory = Callable[[dict], T]


class Registry(Generic[T]):
    def __init__(self, entry_point_group: str):
        self._factories: dict[str, Factory[T]] = {}
        self._group = entry_point_group
        self._loaded_entry_points = False

    def register(self, name: str) -> Callable[[Factory[T]], Factory[T]]:
        def decorator(factory: Factory[T]) -> Factory[T]:
            if name in self._factories:
                raise ValueError(f"{name!r} is already registered in {self._group}")
            self._factories[name] = factory
            return factory

        return decorator

    def create(self, name: str, config: dict) -> T:
        if name not in self._factories:
            self._load_entry_points()
        try:
            factory = self._factories[name]
        except KeyError:
            known = ", ".join(sorted(self._factories)) or "(none)"
            raise ValueError(f"unknown {self._group} {name!r}; known: {known}") from None
        return factory(config)

    # Discover plugins declared by installed packages under this group. A plugin's
    # entry point points at a callable that registers its factories on import, so
    # loading the module is enough. The value is called if it is callable
    def _load_entry_points(self) -> None:
        if self._loaded_entry_points:
            return
        self._loaded_entry_points = True
        for ep in entry_points(group=self._group):
            try:
                obj = ep.load()
                if callable(obj):
                    obj()
            except Exception as e:
                log.warning("failed to load plugin %r in %s: %s", ep.name, self._group, e)
