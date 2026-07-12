"""Atom-wise OOD score aggregation and trigger logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hotspot_al.models import OODFrameResult
from hotspot_al.monitor.lj_residual_fast import compute_lj_residuals_fast
from hotspot_al.monitor.neighbor_utils import MonitorNeighbors

_REASON_MAP = {
    "force": "force_large",
    "delta_force": "force_burst",
    "rmin": "close_contact",
    "delta_q": "coordination_change",
    "lj_residual": "lj_residual",
    "committee": "committee_deviation",
    "displacement": "trajectory_jump",
    "mlip_force_deviation": "model_drift",
}


@dataclass(slots=True)
class RunningMetricStats:
    """Scalar running statistics for one metric across many frames."""

    count: int = 0
    frame_count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, values: np.ndarray) -> None:
        """Update running statistics with a batch of values."""

        batch = np.asarray(values, dtype=float).ravel()
        n_new = len(batch)
        if n_new == 0:
            return
        self.frame_count += 1
        batch_mean = float(batch.mean())
        batch_m2 = float(np.sum((batch - batch_mean) ** 2))
        if self.count == 0:
            self.mean = batch_mean
            self.m2 = batch_m2
            self.count = n_new
            return
        total_count = self.count + n_new
        delta = batch_mean - self.mean
        self.mean += delta * n_new / total_count
        self.m2 += batch_m2 + delta**2 * self.count * n_new / total_count
        self.count = total_count

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
    neighbors: MonitorNeighbors | None = None

    def __post_init__(self) -> None:
        for name in (
            "force",
            "delta_force",
            "rmin",
            "delta_q",
            "lj_residual",
            "committee",
            "displacement",
            "mlip_force_deviation",
        ):
            self.stats.setdefault(name, RunningMetricStats())

    def _z_score(self, name: str, values: np.ndarray) -> np.ndarray:
        section = self.config.get("ood_score", {}).get("running_stats", {})
        if not bool(section.get("enabled", True)):
            return np.abs(np.asarray(values, dtype=float))
        warmup_frames = int(section.get("warmup_frames", 5))
        min_std = float(section.get("min_std", 1.0e-8))
        stats = self.stats[name]
        if stats.frame_count < warmup_frames:
            return np.abs(np.asarray(values, dtype=float))
        std = max(stats.std, min_std)
        return np.abs((np.asarray(values, dtype=float) - stats.mean) / std)

    def _z_metric_score(self, name: str, values: np.ndarray, z_threshold: float | None) -> np.ndarray:
        z_scores = self._z_score(name, values)
        if z_threshold is None:
            return z_scores
        return np.where(z_scores >= float(z_threshold), z_scores, 0.0)

    def _threshold_score(self, values: np.ndarray, threshold: float | None, *, lower_is_worse: bool = False) -> np.ndarray:
        if threshold is None:
            return np.zeros_like(values, dtype=float)
        values = np.asarray(values, dtype=float)
        if lower_is_worse:
            return np.clip((threshold - values) / max(abs(threshold), 1.0e-8), 0.0, None)
        return np.clip((values - threshold) / max(abs(threshold), 1.0e-8), 0.0, None)

    def _get_neighbors(self, atoms: Any) -> MonitorNeighbors:
        monitor_cfg = self.config.get("monitor", {})
        lj_cutoff = float(monitor_cfg.get("lj_cutoff", 6.0))
        coordination_cutoff = float(monitor_cfg.get("coordination_cutoff", lj_cutoff))
        if self.neighbors is None or len(atoms) != self.neighbors.n_atoms:
            self.neighbors = MonitorNeighbors(atoms, lj_cutoff=lj_cutoff, coordination_cutoff=coordination_cutoff)
        else:
            self.neighbors.rebuild(atoms)
        return self.neighbors

    def _light_metric_scores(
        self,
        metrics: dict[str, np.ndarray],
        force_values: np.ndarray,
    ) -> dict[str, np.ndarray]:
        monitor_cfg = self.config.get("monitor", {})
        delta_force_values = np.asarray(metrics.get("delta_force", np.zeros_like(force_values)), dtype=float)
        rmin_values = np.asarray(metrics.get("rmin", np.zeros_like(force_values)), dtype=float)
        delta_q_values = np.asarray(metrics.get("delta_q", np.zeros_like(force_values)), dtype=float)
        displacement_values = np.asarray(metrics.get("displacement", np.zeros_like(force_values)), dtype=float)

        return {
            "force": self._z_metric_score("force", force_values, monitor_cfg.get("force_z_threshold")),
            "delta_force": self._z_metric_score(
                "delta_force",
                delta_force_values,
                monitor_cfg.get("delta_force_z_threshold"),
            ),
            "rmin": self._threshold_score(
                rmin_values,
                monitor_cfg.get("rmin_threshold"),
                lower_is_worse=True,
            ),
            "delta_q": np.maximum(
                self._z_score("delta_q", delta_q_values),
                self._threshold_score(delta_q_values, monitor_cfg.get("delta_q_threshold")),
            ),
            "displacement": self._z_metric_score(
                "displacement",
                displacement_values,
                monitor_cfg.get("displacement_z_threshold"),
            ),
        }

    # -- stage-specific metric computation ------------------------------------

    _LIGHT_METRICS: tuple[str, ...] = ("force", "delta_force", "rmin", "delta_q", "displacement")
    _PHYSICS_METRICS: tuple[str, ...] = (
        "force",
        "delta_force",
        "rmin",
        "delta_q",
        "displacement",
        "lj_residual",
        "mlip_force_deviation",
    )
    _FULL_METRICS: tuple[str, ...] = (
        "force",
        "delta_force",
        "rmin",
        "delta_q",
        "displacement",
        "lj_residual",
        "committee",
        "mlip_force_deviation",
    )

    def _active_metrics_for_stage(self, stage: str) -> tuple[str, ...]:
        if stage == "light":
            return self._LIGHT_METRICS
        if stage == "physics":
            return self._PHYSICS_METRICS
        return self._FULL_METRICS

    def _trigger_threshold_for_stage(self, stage: str) -> float:
        score_cfg = self.config.get("ood_score", {})
        if stage == "light":
            return float(score_cfg.get("screen_threshold", 4.0))
        if stage == "physics":
            return float(score_cfg.get("physics_threshold", 5.0))
        return float(score_cfg.get("label_threshold", 6.0))

    def _compute_lj_residuals(
        self,
        metrics: dict[str, np.ndarray],
        metric_scores: dict[str, np.ndarray],
        force_values: np.ndarray,
        atoms: Any,
        forces: np.ndarray,
    ) -> tuple[np.ndarray, int, dict[str, np.ndarray]]:
        """Compute LJ residuals lazily for suspicious atoms."""

        weights = self.config.get("ood_score", {}).get("weights", {})
        lj_lazy_threshold = float(self.config.get("ood_score", {}).get("lj_lazy_threshold", 3.0))
        lj_values = np.asarray(metrics.get("lj_residual", np.zeros_like(force_values)), dtype=float)
        lj_fit_count = 0
        metrics_for_stats = metrics
        if "lj_residual" not in metrics:
            light_total = np.zeros_like(force_values, dtype=float)
            for name in self._LIGHT_METRICS:
                light_total += float(weights.get(name, 0.0)) * metric_scores[name]
            suspicious_mask = light_total >= lj_lazy_threshold
            if not np.any(suspicious_mask) and len(force_values):
                suspicious_mask[int(np.argmax(light_total))] = bool(np.max(light_total) > 0.0)
            nl = self._get_neighbors(atoms)
            monitor_cfg = self.config.get("monitor", {})
            lj_values, _fits = compute_lj_residuals_fast(
                atoms,
                forces,
                nl,
                cutoff=float(monitor_cfg.get("lj_cutoff", 6.0)),
                suspicious_mask=suspicious_mask,
            )
            lj_fit_count = int(np.sum(suspicious_mask))
            metrics_for_stats = {**metrics, "lj_residual": lj_values}
        return lj_values, lj_fit_count, metrics_for_stats

    def _compute_all_metric_scores(
        self,
        metrics: dict[str, np.ndarray],
        stage: str,
        atoms: Any | None,
        forces: np.ndarray | None,
    ) -> tuple[dict[str, np.ndarray], int, dict[str, np.ndarray]]:
        """Compute scored metrics for the given stage."""

        force_values = np.asarray(metrics.get("force", np.zeros(0)), dtype=float)
        committee_values = np.asarray(metrics.get("committee", np.zeros_like(force_values)), dtype=float)
        mlip_deviation_values = np.asarray(metrics.get("mlip_force_deviation", np.zeros_like(force_values)), dtype=float)

        metric_scores = self._light_metric_scores(metrics, force_values)
        lj_fit_count = 0
        metrics_for_stats = metrics
        if stage in {"physics", "full"} and atoms is not None and forces is not None:
            lj_values, lj_fit_count, metrics_for_stats = self._compute_lj_residuals(
                metrics,
                metric_scores,
                force_values,
                atoms,
                forces,
            )
        else:
            lj_values = np.asarray(metrics.get("lj_residual", np.zeros_like(force_values)), dtype=float)

        metric_scores["lj_residual"] = np.maximum(lj_values, self._z_score("lj_residual", lj_values))
        metric_scores["committee"] = np.maximum(committee_values, self._z_score("committee", committee_values))
        metric_scores["mlip_force_deviation"] = self._z_score("mlip_force_deviation", mlip_deviation_values)
        return metric_scores, lj_fit_count, metrics_for_stats

    def _aggregate_and_trigger(
        self,
        metric_scores: dict[str, np.ndarray],
        stage: str,
        lj_fit_count: int,
        metadata: dict[str, Any] | None,
    ) -> OODFrameResult:
        """Aggregate weighted scores and determine trigger status."""

        weights = self.config.get("ood_score", {}).get("weights", {})
        score_cfg = self.config.get("ood_score", {})
        min_trigger_atoms = int(score_cfg.get("min_trigger_atoms", 1))
        label_threshold = float(score_cfg.get("label_threshold", 6.0))
        active_metrics = self._active_metrics_for_stage(stage)
        trigger_threshold = self._trigger_threshold_for_stage(stage)

        sample = next(iter(metric_scores.values()), np.zeros(0, dtype=float))
        atom_scores = np.zeros_like(sample, dtype=float)
        for name in active_metrics:
            atom_scores += float(weights.get(name, 0.0)) * metric_scores[name]

        max_score = float(np.max(atom_scores)) if len(atom_scores) else 0.0
        triggered = bool(np.sum(atom_scores >= trigger_threshold) >= min_trigger_atoms)
        hotspot_indices = np.where(atom_scores >= label_threshold)[0].tolist()
        if triggered and not hotspot_indices and len(atom_scores):
            hotspot_indices = np.where(atom_scores == np.max(atom_scores))[0].tolist()

        active_metric_scores = {name: metric_scores[name] for name in active_metrics}
        return OODFrameResult(
            atom_scores=atom_scores,
            metric_scores=active_metric_scores,
            max_score=max_score,
            hotspot_indices=hotspot_indices,
            trigger_reason=self._infer_reasons(active_metric_scores, hotspot_indices),
            triggered=triggered,
            stage=stage,
            metadata={
                "screen_threshold": float(score_cfg.get("screen_threshold", 4.0)),
                "physics_threshold": float(score_cfg.get("physics_threshold", 5.0)),
                "label_threshold": label_threshold,
                "lj_lazy_threshold": float(score_cfg.get("lj_lazy_threshold", 3.0)),
                "lj_fit_count": lj_fit_count,
                **(metadata or {}),
            },
        )

    # -- public entry points ---------------------------------------------------

    def score_frame(
        self,
        metrics: dict[str, np.ndarray],
        *,
        stage: str = "full",
        update_stats: bool = True,
        metadata: dict[str, Any] | None = None,
        atoms: Any | None = None,
        forces: np.ndarray | None = None,
    ) -> OODFrameResult:
        """Score one frame from precomputed atom-wise metrics."""

        metric_scores, lj_fit_count, metrics_for_stats = self._compute_all_metric_scores(metrics, stage, atoms, forces)
        result = self._aggregate_and_trigger(metric_scores, stage, lj_fit_count, metadata)
        if update_stats:
            for name, values in metrics_for_stats.items():
                if name in self.stats and len(values):
                    self.stats[name].update(np.asarray(values, dtype=float))
        return result

    def score_light(self, metrics: dict[str, np.ndarray], *, update_stats: bool = True, metadata: dict[str, Any] | None = None) -> OODFrameResult:
        """Stage 1 light monitoring."""

        return self.score_frame(metrics, stage="light", update_stats=update_stats, metadata=metadata)

    def score_physics(
        self,
        metrics: dict[str, np.ndarray],
        *,
        update_stats: bool = True,
        metadata: dict[str, Any] | None = None,
        atoms: Any | None = None,
        forces: np.ndarray | None = None,
    ) -> OODFrameResult:
        """Stage 2 physics-aware monitoring."""

        return self.score_frame(
            metrics,
            stage="physics",
            update_stats=update_stats,
            metadata=metadata,
            atoms=atoms,
            forces=forces,
        )

    def score_full(
        self,
        metrics: dict[str, np.ndarray],
        *,
        update_stats: bool = True,
        metadata: dict[str, Any] | None = None,
        atoms: Any | None = None,
        forces: np.ndarray | None = None,
    ) -> OODFrameResult:
        """Stage 3 committee-confirmed monitoring."""

        return self.score_frame(
            metrics,
            stage="full",
            update_stats=update_stats,
            metadata=metadata,
            atoms=atoms,
            forces=forces,
        )

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
        return reasons
