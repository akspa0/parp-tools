from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import train_v23_height  # noqa: E402

pytestmark = pytest.mark.v23


def test_apply_memory_profile_12gb_caps_runtime_defaults() -> None:
    args = train_v23_height.build_arg_parser().parse_args(
        [
            "--target-vram-gb",
            "12",
            "--batch-size",
            "4",
            "--grad-accum-steps",
            "1",
            "--gpct-weight",
            "0.1",
            "--gpct-K",
            "4",
        ]
    )

    runtime = train_v23_height._apply_memory_profile(args)

    assert runtime.effective_memory_profile == "12gb"
    assert runtime.batch_size == 1
    assert runtime.grad_accum_steps >= 4
    assert runtime.gpct_K == 2
    assert runtime.amp_dtype == "fp16"


def test_apply_memory_profile_24gb_keeps_batch_and_promotes_bf16() -> None:
    args = train_v23_height.build_arg_parser().parse_args(
        [
            "--target-vram-gb",
            "24",
            "--batch-size",
            "2",
            "--grad-accum-steps",
            "1",
            "--gpct-weight",
            "0.1",
            "--gpct-K",
            "4",
        ]
    )

    runtime = train_v23_height._apply_memory_profile(args)

    assert runtime.effective_memory_profile == "24gb"
    assert runtime.batch_size == 2
    assert runtime.grad_accum_steps >= 2
    assert runtime.gpct_K == 4
    assert runtime.amp_dtype == "bf16"
