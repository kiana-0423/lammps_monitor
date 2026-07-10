"""Platform workflow facade composed exclusively from domain services."""

from hotspot_al.active_learning.workflow import build_candidate_pool, extract_regions_for_result

__all__ = ["build_candidate_pool", "extract_regions_for_result"]
