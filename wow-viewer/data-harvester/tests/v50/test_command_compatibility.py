"""Spec 109 Phase 6 (T038): command-ownership convergence.

Every v50_*.py entry point must import its `main` from a canonical harvester.v50 owner, not
from a historical spec103-named script. The historical scripts must still exist (some tests
and the RunPod packager import specific symbols from them by that exact name) but must be thin
re-export shims, not a second copy of the real implementation that could silently drift.

Cross-release rejection proves the moved command owners share the *same* harvester.v50.contracts
gate object rather than a locally reimplemented copy.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"

_CANONICAL_ENTRIES = {
    "v50_train_wdl_prior.py": "harvester.v50.wdl_prior_train",
    "v50_generate_wdl_priors.py": "harvester.v50.wdl_prior_infer",
    "v50_review_wdl_prior.py": "harvester.v50.wdl_prior_evaluate",
    "v50_visualize_wdl_prior.py": "harvester.v50.wdl_prior_visualize",
    "v50_train_terrain.py": "harvester.v50.terrain_refiner_train",
    "v50_infer_terrain.py": "harvester.v50.terrain_refiner_infer",
}

_LEGACY_SHIMS = {
    "train_spec103_wdl_prior.py": "harvester.v50.wdl_prior_train",
    "infer_spec103_wdl_prior.py": "harvester.v50.wdl_prior_infer",
    "evaluate_spec103_wdl_prior.py": "harvester.v50.wdl_prior_evaluate",
    "visualize_spec103_wdl_prior.py": "harvester.v50.wdl_prior_visualize",
    "train_spec103_v7.py": "harvester.v50.terrain_refiner_train",
    "infer_spec103_v7.py": "harvester.v50.terrain_refiner_infer",
}


class TestCanonicalCommandOwnership:
    @pytest.mark.parametrize("script_name,owner_module", sorted(_CANONICAL_ENTRIES.items()))
    def test_v50_entry_point_imports_main_only_from_its_canonical_v50_owner(self, script_name, owner_module):
        source = (_SCRIPTS / script_name).read_text(encoding="utf-8")
        assert f"from {owner_module} import main" in source
        for legacy_name in _LEGACY_SHIMS:
            legacy_module = legacy_name.removesuffix(".py")
            assert f"from {legacy_module} import" not in source, (
                f"{script_name} still delegates to historical module {legacy_module!r}; "
                "it must import main from its harvester.v50 owner directly"
            )

    @pytest.mark.parametrize("legacy_name,owner_module", sorted(_LEGACY_SHIMS.items()))
    def test_historical_shim_re_exports_the_v50_owner_and_defines_no_second_main(self, legacy_name, owner_module):
        source = (_SCRIPTS / legacy_name).read_text(encoding="utf-8")
        assert f"from {owner_module} import" in source
        assert "def main(" not in source, (
            f"{legacy_name} must not define its own main(); it must re-export the v50 owner's "
            "so there is exactly one implementation, not two that can drift apart"
        )


class TestCrossReleaseRejection:
    def test_wdl_prior_infer_load_model_rejects_a_checkpoint_from_a_different_release(self, tmp_path):
        from harvester.spec103.wdl_prior_model import INPUT_CONTRACT, TARGET_CONTRACT, WdlPriorNet
        from harvester.v50.wdl_prior_infer import load_model

        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save({
            "model_family": "v50",
            "release": "v50.2",
            "input_contract": INPUT_CONTRACT,
            "target_contract": TARGET_CONTRACT,
            "model": WdlPriorNet().state_dict(),
        }, checkpoint_path)
        with pytest.raises(ValueError):
            load_model(checkpoint_path, torch.device("cpu"), release="v50.1")

    def test_moved_command_owners_share_the_one_contracts_release_gate_not_a_local_copy(self):
        import harvester.v50.terrain_refiner_infer as terrain_refiner_infer
        import harvester.v50.terrain_refiner_train as terrain_refiner_train
        import harvester.v50.wdl_prior_infer as wdl_prior_infer
        import harvester.v50.wdl_prior_train as wdl_prior_train
        from harvester.v50.contracts import require_metadata_release as canonical_metadata_gate
        from harvester.v50.contracts import require_store_release as canonical_store_gate

        assert wdl_prior_train.require_store_release is canonical_store_gate
        assert wdl_prior_infer.require_metadata_release is canonical_metadata_gate
        assert wdl_prior_infer.require_store_release is canonical_store_gate
        assert terrain_refiner_train.require_store_release is canonical_store_gate
        assert terrain_refiner_train.require_metadata_release is canonical_metadata_gate
        assert terrain_refiner_infer.require_metadata_release is canonical_metadata_gate
        assert terrain_refiner_infer.require_store_release is canonical_store_gate
