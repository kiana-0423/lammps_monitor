"""Simple scheduling helpers for active-learning rounds."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RoundSummary:
    """Bookkeeping summary for one active-learning round."""

    round_id: int
    n_events: int
    n_candidates: int
    n_selected: int


def summarize_round(round_id: int, *, n_events: int, n_candidates: int, n_selected: int) -> RoundSummary:
    """Build a small round summary object."""

    return RoundSummary(round_id=round_id, n_events=n_events, n_candidates=n_candidates, n_selected=n_selected)
