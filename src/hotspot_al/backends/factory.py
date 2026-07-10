"""Configuration-driven backend construction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hotspot_al.backends.base import Backend, BackendRole, MLIPBackend
from hotspot_al.backends.registry import DEFAULT_REGISTRY, BackendRegistry


def backend_engine(config: Mapping[str, Any], role: BackendRole | str) -> str:
    """Resolve an engine name from the nested schema or legacy flat schema."""

    normalized_role = BackendRole(role)
    backend_cfg = config.get("backend", {})
    if not isinstance(backend_cfg, Mapping):
        raise TypeError("backend configuration must be a mapping.")
    value = backend_cfg.get(normalized_role.value)
    if isinstance(value, Mapping):
        engine = value.get("engine")
    else:
        engine = value
    if engine is None:
        legacy_key = {BackendRole.MD: "md_engine", BackendRole.DFT: "dft_engine"}.get(normalized_role)
        engine = backend_cfg.get(legacy_key) if legacy_key is not None else None
    if engine is None and normalized_role == BackendRole.SCHEDULER:
        engine = config.get("cp2k", {}).get("submit_mode", "local") if isinstance(config.get("cp2k"), Mapping) else "local"
    if not isinstance(engine, str) or not engine.strip():
        raise ValueError(f"Missing backend.{normalized_role.value}.engine configuration.")
    return engine.strip().lower().replace("-", "_")


def create_backend(
    config: Mapping[str, Any],
    role: BackendRole | str,
    *,
    registry: BackendRegistry | None = None,
) -> Backend:
    """Create the configured backend for one role."""

    selected_registry = registry or DEFAULT_REGISTRY
    _register_builtins(selected_registry)
    normalized_role = BackendRole(role)
    return selected_registry.create(normalized_role, backend_engine(config, normalized_role), config)


def create_typed_backend(
    config: Mapping[str, Any],
    role: BackendRole | str,
    backend_type: type[Any],
    *,
    registry: BackendRegistry | None = None,
) -> Any:
    backend = create_backend(config, role, registry=registry)
    if not isinstance(backend, backend_type):
        raise TypeError(f"Configured {BackendRole(role).value} backend does not implement {backend_type.__name__}.")
    return backend


def create_mlip_backend(config: Mapping[str, Any], *, legacy_runner: Any | None = None) -> MLIPBackend:
    """Create an MLIP backend, adapting the pre-platform runner API if supplied."""

    if legacy_runner is not None:
        from hotspot_al.backends.allegro import RealAllegroBackend

        return RealAllegroBackend(runner=legacy_runner, config=config)
    return create_typed_backend(config, BackendRole.MLIP, MLIPBackend)


def _register_builtins(registry: BackendRegistry) -> None:
    from hotspot_al.backends.allegro import RealAllegroBackend
    from hotspot_al.backends.cp2k import CP2KBackend
    from hotspot_al.backends.lammps import LAMMPSBackend
    from hotspot_al.backends.schedulers import LocalSchedulerBackend, PBSSchedulerBackend, SlurmSchedulerBackend

    builtins = (
        (BackendRole.MLIP, "allegro", RealAllegroBackend.from_config),
        (BackendRole.MD, "lammps", LAMMPSBackend.from_config),
        (BackendRole.DFT, "cp2k", CP2KBackend.from_config),
        (BackendRole.SCHEDULER, "local", LocalSchedulerBackend.from_config),
        (BackendRole.SCHEDULER, "slurm", SlurmSchedulerBackend.from_config),
        (BackendRole.SCHEDULER, "pbs", PBSSchedulerBackend.from_config),
    )
    existing = set((role, name) for role in BackendRole for name in registry.names(role))
    for role, name, factory in builtins:
        if (role, name) not in existing:
            registry.register(role, name, factory)


__all__ = ["backend_engine", "create_backend", "create_mlip_backend", "create_typed_backend"]
