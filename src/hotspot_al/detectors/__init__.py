"""Physics-aware detector facade independent of simulation runtimes."""

from hotspot_al.hotspot.hotspot_detector import detect_hotspots
from hotspot_al.monitor.ood_score import OODScorer

__all__ = ["OODScorer", "detect_hotspots"]
