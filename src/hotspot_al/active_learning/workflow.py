"""Minimal PHAL/HAL workflow orchestration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from hotspot_al.active_learning.candidate_pool import CandidatePool
from hotspot_al.extraction.block import extract_block_regions
from hotspot_al.extraction.cluster_extractor import extract_cluster_region
from hotspot_al.extraction.graph_extractor import extract_graph_region
from hotspot_al.extraction.slab_extractor import extract_slab_patch
from hotspot_al.hotspot.hotspot_detector import detect_hotspots
from hotspot_al.models import ExtractedRegion, FrameData, Hotspot, OODFrameResult

ExtractionStrategy = Callable[[FrameData, OODFrameResult, list[Hotspot], dict[str, Any]], list[ExtractedRegion]]


def extract_regions_for_result(frame: FrameData, result: OODFrameResult, *, config: dict[str, Any]) -> list:
    """Turn a scored frame into extracted hotspot regions."""

    hotspot_cfg = config.get("hotspot", {})
    hotspots = detect_hotspots(
        frame.atoms,
        result.atom_scores,
        threshold=float(hotspot_cfg.get("threshold", config.get("ood_score", {}).get("label_threshold", 6.0))),
        merge_radius=float(hotspot_cfg.get("merge_radius", 8.0)),
        step=frame.step,
        trigger_reasons=result.trigger_reason,
        event_id=result.metadata.get("event_id"),
        backend=result.metadata.get("backend"),
    )
    mode = str(config.get("extraction", {}).get("mode", "cluster"))
    strategy = EXTRACTION_STRATEGIES.get(mode)
    if strategy is None:
        msg = f"Unsupported extraction mode: {mode}"
        raise ValueError(msg)
    return strategy(frame, result, hotspots, config)


def _extract_block_strategy(
    frame: FrameData,
    result: OODFrameResult,
    _hotspots: list[Hotspot],
    config: dict[str, Any],
) -> list[ExtractedRegion]:
    block_regions = extract_block_regions(frame.atoms, result.hotspot_indices, config=config, step=frame.step)
    for region_index, region in enumerate(block_regions):
        region.metadata = {
            **region.metadata,
            "event_id": result.metadata.get("event_id"),
            "hotspot_id": f"{result.metadata.get('event_id') or frame.step}_blk{region_index:03d}",
            "backend": result.metadata.get("backend"),
            "original_frame_id": frame.step,
        }
    return block_regions


def _extract_hotspot_strategy(
    frame: FrameData,
    _result: OODFrameResult,
    hotspots: list[Hotspot],
    config: dict[str, Any],
    *,
    mode: str,
) -> list[ExtractedRegion]:
    regions: list[ExtractedRegion] = []
    for hotspot_index, hotspot in enumerate(hotspots):
        if mode == "graph":
            region = extract_graph_region(frame.atoms, hotspot.core_atom_indices, config=config)
        elif mode == "slab":
            slab_cfg = config.get("extraction", {}).get("slab", {})
            region = extract_slab_patch(
                frame.atoms,
                hotspot.core_atom_indices,
                config=config,
                xy_lengths=(float(slab_cfg.get("lx", 16.0)), float(slab_cfg.get("ly", 16.0))),
                z_thickness=float(slab_cfg.get("z_margin", 8.0)) * 2.0,
            )
        else:
            region = extract_cluster_region(frame.atoms, hotspot.core_atom_indices, config=config)
        region.metadata = {
            **region.metadata,
            "event_id": hotspot.event_id,
            "hotspot_id": f"{hotspot.event_id or frame.step}_hs{hotspot_index:03d}",
            "backend": hotspot.backend,
            "original_frame_id": frame.step,
        }
        regions.append(region)
    return regions


def _extract_cluster_strategy(
    frame: FrameData,
    result: OODFrameResult,
    hotspots: list[Hotspot],
    config: dict[str, Any],
) -> list[ExtractedRegion]:
    return _extract_hotspot_strategy(frame, result, hotspots, config, mode="cluster")


def _extract_graph_strategy(
    frame: FrameData,
    result: OODFrameResult,
    hotspots: list[Hotspot],
    config: dict[str, Any],
) -> list[ExtractedRegion]:
    return _extract_hotspot_strategy(frame, result, hotspots, config, mode="graph")


def _extract_slab_strategy(
    frame: FrameData,
    result: OODFrameResult,
    hotspots: list[Hotspot],
    config: dict[str, Any],
) -> list[ExtractedRegion]:
    return _extract_hotspot_strategy(frame, result, hotspots, config, mode="slab")


EXTRACTION_STRATEGIES: dict[str, ExtractionStrategy] = {
    "block": _extract_block_strategy,
    "cluster": _extract_cluster_strategy,
    "graph": _extract_graph_strategy,
    "slab": _extract_slab_strategy,
}


def build_candidate_pool(frames: list[tuple[FrameData, OODFrameResult]], *, config: dict[str, Any]) -> CandidatePool:
    """Build a candidate pool from scored frames."""

    pool_cfg = config.get("candidate_pool", {})
    pool = CandidatePool(
        diversity_threshold=float(pool_cfg.get("diversity_threshold", 0.1)),
        max_candidates=int(pool_cfg.get("max_candidates_per_round", 50)),
        deduplicate=bool(pool_cfg.get("deduplicate", True)),
        fingerprint_mode=str(pool_cfg.get("fingerprint", "pair_distance_histogram")),
    )
    for frame, result in frames:
        for region in extract_regions_for_result(frame, result, config=config):
            pool.add(region, score=result.max_score, metadata={"step": frame.step})
    return pool
