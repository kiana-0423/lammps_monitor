"""Embedding hooks for future QM/MM or electrostatic boundary models."""

from __future__ import annotations

from typing import Any

from hotspot_al.models import ExtractedRegion


def build_embedding(region: ExtractedRegion, config: dict[str, Any]) -> dict[str, Any]:
    """Return embedding metadata for a region."""

    embedding_cfg = config.get("embedding", config)
    if not embedding_cfg.get("enabled", False):
        result = {"enabled": False, "status": "disabled"}
        region.metadata["embedding"] = result
        return result

    method = str(embedding_cfg.get("method", "point_charge"))
    if method not in {"point_charge", "point_charges"}:
        result = {"enabled": True, "status": "unsupported", "method": method}
        region.metadata["embedding"] = result
        return result

    charges = {
        str(element): float(charge)
        for element, charge in embedding_cfg.get("charges", embedding_cfg.get("point_charges", {})).items()
    }
    default_charge = embedding_cfg.get("default_charge")
    point_charges: list[dict[str, Any]] = []
    for index, atom in enumerate(region.atoms):
        if atom.symbol in charges:
            charge = charges[atom.symbol]
        elif default_charge is not None:
            charge = float(default_charge)
        else:
            continue
        point_charges.append(
            {
                "index": index,
                "element": atom.symbol,
                "charge": charge,
                "position": region.atoms.positions[index].astype(float).tolist(),
            }
        )

    result = {
        "enabled": True,
        "status": "ok",
        "method": "point_charge",
        "n_point_charges": len(point_charges),
        "point_charges": point_charges,
    }
    region.metadata["embedding"] = result
    return result
