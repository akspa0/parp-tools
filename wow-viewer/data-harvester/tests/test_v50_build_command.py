from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_v50_build_command():
    path = Path(__file__).parents[1] / "scripts" / "v50_build_dataset.py"
    spec = importlib.util.spec_from_file_location("v50_build_dataset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v50_command_refuses_legacy_mixed_store_construction(capsys) -> None:
    command = _load_v50_build_command()

    assert command.main([]) == 2

    assert "legacy Spec 108 mixed builder cannot create a clean-room V50 store" in capsys.readouterr().out
