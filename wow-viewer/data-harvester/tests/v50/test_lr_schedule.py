"""Warmup-aware LR-schedule + early-stop accounting (the Spec 117 scheduling fix).

Reproduces the exact failure mode that broke the lattice/detailer runs: OneCycleLR's
warmup (default ``pct_start=0.3``) spans more epochs than the early-stopper's
``patience`` (15), so a run could be killed mid-warmup before the LR ever reached
its peak. The fix suppresses the stale counter until ``warmup_complete`` is true.
"""

from __future__ import annotations

import pytest

from harvester.v50.lr_schedule import (
    PCT_START_DEFAULT,
    warmup_complete,
    warmup_epochs_for,
)


def test_warmup_epochs_matches_onecycle_default_fraction() -> None:
    # The real Spec 117 run: 100 epochs, 43 steps/epoch, default pct_start 0.3.
    assert warmup_epochs_for(0.3, 100, 43) == 30


def test_warmup_epochs_shorter_pct_start_is_proportionally_shorter() -> None:
    # The recommended small-dataset setting: 0.1 -> 10 epochs of warmup.
    assert warmup_epochs_for(0.1, 100, 43) == 10


def test_warmup_epochs_rounds_up_so_no_warmup_step_is_unsuppressed() -> None:
    # 0.3 * 50 epochs = 15 epochs exactly; a non-integer fraction rounds up.
    assert warmup_epochs_for(0.3, 50, 43) == 15
    assert warmup_epochs_for(0.25, 100, 43) == 25


def test_warmup_complete_is_false_throughout_warmup_then_true() -> None:
    warmup_epochs = 30
    # 1-based epochs: epochs 1..30 are warmup, epoch 31 onward is past it.
    for epoch in range(1, 31):
        assert warmup_complete(epoch, warmup_epochs) is False
    assert warmup_complete(31, warmup_epochs) is True


def test_zero_warmup_means_every_epoch_can_count_stale() -> None:
    # Constant schedule -> warmup_epochs 0 -> epoch 1 is already past warmup.
    assert warmup_epochs_for(0.3, 0, 43) == 0
    assert warmup_complete(1, 0) is True


def test_reproduces_the_bug_patience_shorter_than_warmup() -> None:
    """The exact Spec 117 detailer failure: patience 15 < warmup 30.

    Before the fix, a flat validation curve during warmup incremented stale every
    epoch and early-stopped at epoch 17 (best epoch 2). With warmup-aware
    accounting, those warmup epochs do not count, so the run survives into the
    productive LR region.
    """
    warmup_epochs = warmup_epochs_for(PCT_START_DEFAULT, 100, 43)
    patience = 15
    assert warmup_epochs > patience  # the precondition for the bug

    stale = 0
    best = float("inf")
    # Simulate a frozen-at-baseline validation curve (the zero-init detailer head).
    val_curve = [0.2307, 0.2301, 0.2301, 0.2321, 0.2334, 0.2375, 0.2389, 0.2378,
                 0.2348, 0.2328, 0.2368, 0.2345, 0.2354, 0.2378, 0.2355, 0.2437,
                 0.2395, 0.2310, 0.2300, 0.2290]
    early_stopped_at = None
    for epoch, val_mae in enumerate(val_curve, start=1):
        if val_mae < best:
            best = val_mae
            stale = 0
        elif warmup_complete(epoch, warmup_epochs):
            stale += 1
        if patience > 0 and stale >= patience:
            early_stopped_at = epoch
            break
    # The run must NOT early-stop during warmup (epoch <= 30); it reaches the
    # productive region where validation finally starts improving (epoch 18+).
    assert early_stopped_at is None, "run survived warmup and began improving"


def test_make_onecycle_scheduler_reports_warmup_epochs() -> None:
    torch = pytest.importorskip("torch")
    opt = torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=2e-4)
    scheduler, warmup_epochs = pytest.importorskip(
        "harvester.v50.lr_schedule"
    ).make_onecycle_scheduler(
        opt, max_lr=2e-4, epochs=100, steps_per_epoch=43, pct_start=0.1,
    )
    assert warmup_epochs == 10
    # The scheduler's first LR is max_lr / initial_div_factor (default 25).
    initial_lr = opt.param_groups[0]["lr"]
    assert initial_lr == pytest.approx(2e-4 / 25, rel=1e-6)
