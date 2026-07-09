"""Compatibility wrappers for mask-weighted force losses.

.. deprecated::
   Import directly from :mod:`hotspot_al.training.loss_mask` instead.
   This module will be removed in a future version.
"""

from __future__ import annotations

from hotspot_al.training.loss_mask import masked_force_mse

__all__ = ["masked_force_mse"]
