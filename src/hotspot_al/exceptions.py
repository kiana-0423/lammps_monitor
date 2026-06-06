"""Project-specific exception types."""

from __future__ import annotations


class HotspotALError(Exception):
    """Base class for hotspot active-learning errors."""


class ConfigError(HotspotALError):
    """Configuration loading or validation failed."""


class BackendError(HotspotALError):
    """External backend setup or execution failed."""


class InferenceError(BackendError):
    """MLIP inference failed."""


class LAMMPSRuntimeError(BackendError):
    """LAMMPS execution failed."""


class CP2KRuntimeError(BackendError):
    """CP2K execution failed."""


class DataError(HotspotALError):
    """Input or generated data is malformed."""
