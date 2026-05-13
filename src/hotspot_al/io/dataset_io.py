"""Common dataset I/O helpers for masked local training data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hotspot_al.models import ExtractedRegion
from hotspot_al.training.dataset_writer import write_dataset_entry


def write_common_dataset(
    region: ExtractedRegion,
    *,
    forces: np.ndarray,
    mask: np.ndarray,
    output_dir: str | Path,
    prefix: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write one training sample in the common extxyz + npz representation."""

    return write_dataset_entry(
        region,
        forces=forces,
        mask=mask,
        output_dir=output_dir,
        prefix=prefix,
        extra_metadata=metadata,
    )
