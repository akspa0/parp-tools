from pathlib import Path
import importlib.util

import numpy as np


def _module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "audit_v25_minimap_contract.py"
    spec = importlib.util.spec_from_file_location("audit_v25_minimap_contract", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_split_holds_out_entire_era_and_map():
    module = _module()
    assert module.assign_split("0_5_3_3368", "Azeroth") == "test_era"
    assert module.assign_split("3_3_5_12340", "Northrend") == "validation_map"
    assert module.assign_split("3_3_5_12340", "Azeroth") == "train"


def test_rgb_baseline_fits_from_rgb_and_train_targets_only():
    module = _module()
    rgb = np.asarray([0.0, 0.5, 1.0])
    target = 100.0 * rgb + 20.0
    slope, intercept = module.fit_rgb_flat_baseline(rgb, target)
    assert slope == pytest.approx(100.0)
    assert intercept == pytest.approx(20.0)


import pytest
