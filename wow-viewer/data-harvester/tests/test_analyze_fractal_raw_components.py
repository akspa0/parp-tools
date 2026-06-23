from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from analyze_fractal_raw_components import resolve_target_maps  # noqa: E402


def test_resolve_target_maps_expands_all() -> None:
    assert resolve_target_maps(["all"], ["Azeroth", "Kalimdor"]) == ["Azeroth", "Kalimdor"]


def test_resolve_target_maps_keeps_explicit_names() -> None:
    assert resolve_target_maps(["Azeroth"], ["Azeroth", "Kalimdor"]) == ["Azeroth"]
