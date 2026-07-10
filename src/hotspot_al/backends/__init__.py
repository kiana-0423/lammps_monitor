"""Public backend contracts, factories, and built-in adapters."""

from hotspot_al.backends.allegro import RealAllegroBackend
from hotspot_al.backends.base import (
    Backend,
    BackendJob,
    BackendRole,
    DFTBackend,
    ExecutionRequest,
    ForceBackend,
    JobState,
    MDBackend,
    MLIPBackend,
    RuntimeStatus,
    SchedulerBackend,
)
from hotspot_al.backends.cp2k import CP2KBackend
from hotspot_al.backends.factory import backend_engine, create_backend, create_mlip_backend, create_typed_backend
from hotspot_al.backends.lammps import LAMMPSBackend
from hotspot_al.backends.registry import BackendRegistry, DEFAULT_REGISTRY
from hotspot_al.backends.schedulers import LocalSchedulerBackend, PBSSchedulerBackend, SlurmSchedulerBackend

__all__ = [
    "Backend",
    "BackendJob",
    "BackendRegistry",
    "BackendRole",
    "CP2KBackend",
    "DEFAULT_REGISTRY",
    "DFTBackend",
    "ExecutionRequest",
    "ForceBackend",
    "JobState",
    "LAMMPSBackend",
    "LocalSchedulerBackend",
    "MDBackend",
    "MLIPBackend",
    "PBSSchedulerBackend",
    "RealAllegroBackend",
    "RuntimeStatus",
    "SchedulerBackend",
    "SlurmSchedulerBackend",
    "backend_engine",
    "create_backend",
    "create_mlip_backend",
    "create_typed_backend",
]
