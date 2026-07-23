"""Shared LR-schedule construction + warmup-aware early-stop accounting.

The three v50 geometry trainers (coarse ``direct_geometry_train``, residual
``geometry_detailer_train``, and Spec 117 ``lattice_train``) all pair an optional
``OneCycleLR`` schedule with a patience-based early stopper. ``OneCycleLR``
*deliberately* holds the learning rate low during its warmup phase (the first
``pct_start`` fraction of total steps); a residual detailer with a zero-init head
cannot improve validation MAE until the LR rises. When ``patience`` is shorter
than the warmup length, the early stopper fires *during* warmup -- before the
schedule ever reaches its peak LR -- and the run "goes stale very early" without
ever getting a chance to learn.

This module makes that contract explicit and prevents the failure mode: the stale
counter is suppressed until the warmup phase completes. ``pct_start`` is exposed
so a small dataset (e.g. 43 steps/epoch, 100 epochs) can use a shorter warmup
than OneCycleLR's 0.3 default, which spends 30% of a short run just warming up.

Kept dependency-free at import time (``torch`` is imported lazily inside
``make_onecycle_scheduler``) so the warmup-arithmetic helpers can be unit-tested
without a torch/CUDA install.
"""

from __future__ import annotations

import math

PCT_START_DEFAULT = 0.3  # matches torch.optim.lr_scheduler.OneCycleLR's default


def warmup_epochs_for(pct_start: float, epochs: int, steps_per_epoch: int) -> int:
    """Number of whole epochs the OneCycleLR warmup phase spans.

    OneCycleLR warms up over the first ``pct_start`` fraction of ``total_steps``
    (``epochs * steps_per_epoch``); with a constant ``steps_per_epoch`` that is
    ``pct_start * epochs`` epochs. Rounded up so the stale counter is never
    unsuppressed while any warmup step is still in flight.
    """
    if epochs <= 0 or steps_per_epoch <= 0:
        return 0
    warmup_steps = pct_start * epochs * steps_per_epoch
    return max(1, math.ceil(warmup_steps / steps_per_epoch)) if warmup_steps > 0 else 0


def warmup_complete(epoch: int, warmup_epochs: int) -> bool:
    """True once ``epoch`` is past the schedule's warmup phase (1-based epochs)."""
    return epoch > warmup_epochs


def make_onecycle_scheduler(opt, *, max_lr: float, epochs: int, steps_per_epoch: int,
                            pct_start: float = PCT_START_DEFAULT):
    """Build a OneCycleLR and report how many epochs its warmup occupies.

    Returns ``(scheduler, warmup_epochs)``. ``warmup_epochs`` is 0 for a degenerate
    (non-positive) configuration so callers can pass it straight to
    :func:`warmup_complete` without a separate constant-schedule branch.
    """
    import torch  # lazy: keeps this module importable without torch installed

    if not 0.0 < pct_start < 1.0:
        raise ValueError(f"pct_start must be in (0, 1); got {pct_start}")
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=max_lr, epochs=epochs, steps_per_epoch=steps_per_epoch,
        pct_start=pct_start,
    )
    return scheduler, warmup_epochs_for(pct_start, epochs, steps_per_epoch)


__all__ = [
    "PCT_START_DEFAULT",
    "warmup_epochs_for",
    "warmup_complete",
    "make_onecycle_scheduler",
]
