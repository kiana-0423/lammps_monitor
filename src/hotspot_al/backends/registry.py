"""Backend registration and entry-point discovery."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from importlib.metadata import EntryPoint, entry_points
from typing import Any

from hotspot_al.backends.base import Backend, BackendRole

BackendFactory = Callable[[Mapping[str, Any]], Backend]
ENTRY_POINT_GROUP = "hotspot_al.backends"


class BackendRegistry:
    """Registry of backend factories keyed by role and engine name."""

    def __init__(self) -> None:
        self._factories: dict[tuple[BackendRole, str], BackendFactory] = {}
        self._entry_points_loaded = False

    def register(
        self,
        role: BackendRole | str,
        name: str,
        factory: BackendFactory,
        *,
        replace: bool = False,
    ) -> None:
        key = (BackendRole(role), _normalize_name(name))
        if key in self._factories and not replace:
            raise ValueError(f"Backend already registered for {key[0].value}:{key[1]}.")
        self._factories[key] = factory

    def unregister(self, role: BackendRole | str, name: str) -> None:
        self._factories.pop((BackendRole(role), _normalize_name(name)), None)

    def create(self, role: BackendRole | str, name: str, config: Mapping[str, Any]) -> Backend:
        normalized_role = BackendRole(role)
        normalized_name = _normalize_name(name)
        self.load_entry_points()
        factory = self._factories.get((normalized_role, normalized_name))
        if factory is None:
            available = ", ".join(self.names(normalized_role)) or "none"
            raise KeyError(
                f"Unknown {normalized_role.value} backend {normalized_name!r}; "
                f"registered engines: {available}."
            )
        backend = factory(config)
        if backend.role != normalized_role:
            raise TypeError(
                f"Backend {normalized_name!r} declared role {backend.role.value!r}, "
                f"expected {normalized_role.value!r}."
            )
        return backend

    def names(self, role: BackendRole | str | None = None) -> list[str]:
        selected = None if role is None else BackendRole(role)
        return sorted(name for (item_role, name) in self._factories if selected is None or item_role == selected)

    def load_entry_points(self, points: Iterable[EntryPoint] | None = None) -> None:
        """Load ``role:name`` factories from ``hotspot_al.backends`` entry points."""

        if points is None:
            if self._entry_points_loaded:
                return
            discovered = entry_points()
            points = discovered.select(group=ENTRY_POINT_GROUP)
            self._entry_points_loaded = True
        for point in points:
            try:
                raw_role, raw_name = point.name.split(":", 1)
                loaded = point.load()
                factory = _coerce_factory(loaded)
                self.register(BackendRole(raw_role), raw_name, factory)
            except (AttributeError, ImportError, TypeError, ValueError) as exc:
                raise RuntimeError(f"Failed to load PHAL backend entry point {point.name!r}.") from exc


def _coerce_factory(value: Any) -> BackendFactory:
    if isinstance(value, type) and issubclass(value, Backend):
        return value.from_config
    if callable(value):
        return value
    raise TypeError("A backend entry point must expose a Backend subclass or factory.")


def _normalize_name(name: str) -> str:
    normalized = str(name).strip().lower().replace("-", "_")
    if not normalized:
        raise ValueError("Backend name must not be empty.")
    return normalized


DEFAULT_REGISTRY = BackendRegistry()


__all__ = ["BackendFactory", "BackendRegistry", "DEFAULT_REGISTRY", "ENTRY_POINT_GROUP"]
