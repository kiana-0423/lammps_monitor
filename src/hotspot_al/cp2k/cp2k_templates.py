"""Template helpers for CP2K cluster and slab jobs."""

from __future__ import annotations


def global_block(run_type: str, project_name: str = "hotspot_region") -> str:
    """Return a minimal CP2K GLOBAL block."""

    return "\n".join(
        [
            "&GLOBAL",
            f"  PROJECT {project_name}",
            f"  RUN_TYPE {run_type}",
            "  PRINT_LEVEL MEDIUM",
            "&END GLOBAL",
        ]
    )


def constraint_block(fixed_atom_indices: list[int] | None) -> str:
    """Return a FIXED_ATOMS block when needed."""

    if not fixed_atom_indices:
        return ""
    indices = " ".join(str(index + 1) for index in sorted(fixed_atom_indices))
    return "\n".join(
        [
            "  &CONSTRAINT",
            "    &FIXED_ATOMS",
            "      COMPONENTS_TO_FIX XYZ",
            f"      LIST {indices}",
            "    &END FIXED_ATOMS",
            "  &END CONSTRAINT",
        ]
    )
