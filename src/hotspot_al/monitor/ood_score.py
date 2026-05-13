"""Atom-wise OOD score aggregation and trigger logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hotspot_al.models import OODFrameResult


_REASON_MAP = {
    "force": "force_large",
    "delta_force": "force_burst",
    "rmin": "close_contact",
    "delta_q": "coordination_change",
    "lj_residual": "lj_residual",
    "committee": "committee_deviation",
    "displacement": "trajectory_jump",
}


@dataclass(slots=True)
class RunningMetricStats:
    """Scalar running statistics for one metric across many frames."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, values: np.ndarray) -> None:
        flattened = np.asarray(values, dtype=float).ravel()
        for value in flattened:
            self.count += 1
            delta = value - self.mean
            self.mean += delta / self.count
            delta2 = value - self.mean
            self.m2 += delta * delta2

    @property
    def std(self) -> float:
        if self.count < 2:
            return 0.0
        return float(np.sqrt(self.m2 / (self.count - 1)))


@dataclass(slots=True)
class OODScorer:
    """Aggregate atom-wise metrics into a triggerable OOD score."""

    config: dict[str, Any]
    stats: dict[str, RunningMetricStats] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("force", "delta_force", "rmin", "delta_q", "lj_residual", "committee", "displacement"):
            self.stats.setdefault(name, RunningMetricStats())

    def _z_score(self, name: str, values: np.ndarray) -> np.ndarray:
        section = self.config.get("ood_score", {}).get("running_stats", {})
        warmup_frames = int(section.get("warmup_frames", 5))
        min_std = float(section.get("min_std", 1.0e-8))
        stats = self.stats[name]
        if stats.count < warmup_frames:
            return np.zeros_like(values, dtype=float)
        std = max(stats.std, min_std)
        return np.abs((np.asarray(values, dtype=float) - stats.mean) / std)

    def _threshold_score(self, values: np.ndarray, threshold: float | None, *, lower_is_worse: bool = False) -> np.ndarray:
        if threshold is None:
            return np.zeros_like(values, dtype=float)
        values = np.asarray(values, dtype=float)
        if lower_is_worse:
            return np.clip((threshold - values) / max(abs(threshold), 1.0e-8), 0.0, None)
        return np.clip((values - threshold) / max(abs(threshold), 1.0e-8), 0.0, None)

    def score_frame(
        self,
        metrics: dict[str, np.ndarray],
        *,
        stage: str = "full",
        update_stats: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> OODFrameResult:
        """Score one frame from precomputed atom-wise metrics."""

        monitor_cfg = self.config.get("monitor", {})
        weights = self.config.get("ood_score", {}).get("weights", {})
        score_cfg = self.config.get("ood_score", {})
        screen_threshold = float(score_cfg.get("screen_threshold", 4.0))
        physics_threshold = float(score_cfg.get("physics_threshold", 5.0))
        label_threshold = float(score_cfg.get("label_threshold", 6.0))
        min_trigger_atoms = int(self.config.get("ood_score", {}).get("min_trigger_atoms", 1))

        metric_scores: dict[str, np.ndarray] = {}

        force_values = np.asarray(metrics.get("force", np.zeros(0)), dtype=float)
        delta_force_values = np.asarray(metrics.get("delta_force", np.zeros_like(force_values)), dtype=float)
        rmin_values = np.asarray(metrics.get("rmin", np.zeros_like(force_values)), dtype=float)
        delta_q_values = np.asarray(metrics.get("delta_q", np.zeros_like(force_values)), dtype=float)
        lj_values = np.asarray(metrics.get("lj_residual", np.zeros_like(force_values)), dtype=float)
        committee_values = np.asarray(metrics.get("committee", np.zeros_like(force_values)), dtype=float)
        displacement_values = np.asarray(metrics.get("displacement", np.zeros_like(force_values)), dtype=float)

        metric_scores["force"] = self._z_score("force", force_values)
        metric_scores["delta_force"] = self._z_score("delta_force", delta_force_values)
        metric_scores["rmin"] = self._threshold_score(
            rmin_values,
            monitor_cfg.get("rmin_threshold"),
            lower_is_worse=True,
        )
        metric_scores["delta_q"] = np.maximum(
            self._z_score("delta_q", delta_q_values),
            self._threshold_score(delta_q_values, monitor_cfg.get("delta_q_threshold")),
        )
        metric_scores["displacement"] = np.maximum(
            self._z_score("displacement", displacement_values),
            self._threshold_score(displacement_values, monitor_cfg.get("displacement_z_threshold")),
        )
        metric_scores["lj_residual"] = np.maximum(lj_values, self._z_score("lj_residual", lj_values))
        metric_scores["committee"] = np.maximum(committee_values, self._z_score("committee", committee_values))

        if stage == "light":
            active_metrics = ("force", "delta_force", "rmin", "delta_q", "displacement")
            trigger_threshold = screen_threshold
        elif stage == "physics":
            active_metrics = ("force", "delta_force", "rmin", "delta_q", "displacement", "lj_residual")
            trigger_threshold = physics_threshold
        else:
            active_metrics = ("force", "delta_force", "rmin", "delta_q", "displacement", "lj_residual", "committee")
            trigger_threshold = label_threshold

        atom_scores = np.zeros_like(force_values, dtype=float)
        for name in active_metrics:
            values = metric_scores[name]
            atom_scores += float(weights.get(name, 0.0)) * values

        max_score = float(np.max(atom_scores)) if len(atom_scores) else 0.0
        triggered = bool(np.sum(atom_scores >= trigger_threshold) >= min_trigger_atoms)
        hotspot_indices = np.where(atom_scores >= label_threshold)[0].tolist()
        if triggered and not hotspot_indices and len(atom_scores):
            hotspot_indices = np.where(atom_scores == np.max(atom_scores))[0].tolist()

        active_metric_scores = {name: metric_scores[name] for name in active_metrics}
        reasons = self._infer_reasons(active_metric_scores, hotspot_indices)
        result = OODFrameResult(
            atom_scores=atom_scores,
            metric_scores=active_metric_scores,
            max_score=max_score,
            hotspot_indices=hotspot_indices,
            trigger_reason=reasons,
            triggered=triggered,
            stage=stage,
            metadata={
                "screen_threshold": screen_threshold,
                "physics_threshold": physics_threshold,
                "label_threshold": label_threshold,
                **(metadata or {}),
            },
        )

        if update_stats:
            for name, values in metrics.items():
                if name in self.stats and len(values):
                    self.stats[name].update(np.asarray(values, dtype=float))
        return result

    def score_light(self, metrics: dict[str, np.ndarray], *, update_stats: bool = True, metadata: dict[str, Any] | None = None) -> OODFrameResult:
        """Stage 1 light monitoring."""

        return self.score_frame(metrics, stage="light", update_stats=update_stats, metadata=metadata)

    def score_physics(self, metrics: dict[str, np.ndarray], *, update_stats: bool = True, metadata: dict[str, Any] | None = None) -> OODFrameResult:
        """Stage 2 physics-aware monitoring."""

        return self.score_frame(metrics, stage="physics", update_stats=update_stats, metadata=metadata)

    def score_full(self, metrics: dict[str, np.ndarray], *, update_stats: bool = True, metadata: dict[str, Any] | None = None) -> OODFrameResult:
        """Stage 3 committee-confirmed monitoring."""

        return self.score_frame(metrics, stage="full", update_stats=update_stats, metadata=metadata)

    def _infer_reasons(self, metric_scores: dict[str, np.ndarray], hotspot_indices: list[int]) -> list[str]:
        if not metric_scores:
            return []
        if hotspot_indices:
            candidate = hotspot_indices
        else:
            sample = next(iter(metric_scores.values()), np.zeros(0, dtype=float))
            candidate = np.arange(len(sample)).tolist()
        ranked: list[tuple[str, float]] = []
        for name, values in metric_scores.items():
            score = float(np.mean(values[candidate])) if len(candidate) else 0.0
            if score > 0.0:
                ranked.append((name, score))
        ranked.sort(key=lambda item: item[1], reverse=True)
        reasons: list[str] = []
        for name, _score in ranked:
            label = _REASON_MAP.get(name, name)
            if label not in reasons:
                reasons.append(label)
        return reasons[:3]
