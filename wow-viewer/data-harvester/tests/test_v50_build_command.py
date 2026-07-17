"""scripts/v50_build_dataset.py's CLI contract.

Spec 109 Phase 1 shipped this script as a fail-closed placeholder that refused every
invocation with a fixed "legacy Spec 108 mixed builder cannot create a clean-room V50 store"
message. Phase 5 (US3) replaced that placeholder with five real subcommands (migrate-v18,
build, verify, finalize, curriculum); the old refusal message no longer exists by design --
see specs/109-v50-clean-room-audit/tasks.md T034. This test now covers the current contract:
a subcommand is mandatory, and an unrecognized one is rejected the same way.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_SCRIPT = Path(__file__).parents[1] / "scripts" / "v50_build_dataset.py"


def test_v50_build_dataset_requires_an_explicit_subcommand() -> None:
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        capture_output=True, text=True,
    )
    assert completed.returncode == 2
    assert "the following arguments are required: command" in completed.stderr


def test_v50_build_dataset_rejects_an_unrecognized_subcommand() -> None:
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), "legacy-mixed-build"],
        capture_output=True, text=True,
    )
    assert completed.returncode == 2
    assert "invalid choice: 'legacy-mixed-build'" in completed.stderr
