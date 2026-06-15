"""Tests for extraction embedding metadata."""

from __future__ import annotations

from ase import Atoms

from hotspot_al.extraction.embedding import build_embedding
from hotspot_al.models import ExtractedRegion


def _region() -> ExtractedRegion:
    return ExtractedRegion(
        atoms=Atoms("CH", positions=[[0.0, 0.0, 0.0], [1.09, 0.0, 0.0]]),
        original_indices=[0, 1],
        core_indices=[0],
        inner_buffer_indices=[1],
        outer_buffer_indices=[],
        boundary_indices=[],
        h_cap_indices=[],
        hotspot_indices=[0],
    )


def test_build_embedding_disabled() -> None:
    region = _region()

    result = build_embedding(region, {"embedding": {"enabled": False}})

    assert result == {"enabled": False, "status": "disabled"}
    assert region.metadata["embedding"] == result


def test_build_embedding_point_charges_from_config() -> None:
    region = _region()

    result = build_embedding(region, {"embedding": {"enabled": True, "charges": {"C": 0.2, "H": -0.2}}})

    assert result["status"] == "ok"
    assert result["method"] == "point_charge"
    assert result["n_point_charges"] == 2
    assert [item["charge"] for item in result["point_charges"]] == [0.2, -0.2]
    assert region.metadata["embedding"] == result
