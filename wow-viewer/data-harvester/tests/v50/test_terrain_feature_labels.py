"""Spec 115: terrain-feature label derivation and classifier contract tests.

The rule-ordering tests use REAL texture paths taken from the 0.5.3.3368 Kalimdor+Azeroth corpus,
not invented ones -- they pin the two traps that a naive matcher gets wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from harvester.v50.terrain_feature_labels import (
    CLASS_COUNT,
    DOMINANT_ALPHA_THRESHOLD,
    FAMILY_NAMES,
    ROAD,
    STRUCTURE,
    TERRAIN,
    UNKNOWN,
    TerrainFeatureLabelError,
    classify_texture_name,
    derive_row_labels,
    resolve_dominant_layer,
    rule_set_sha256,
    summarize_labels,
    texture_leaf,
)


class TestTextureClassification:
    @pytest.mark.parametrize(
        "path,expected",
        [
            # Plain cases.
            (r"Tileset\Darkshore\DarkshoreGrass.blp", TERRAIN),
            (r"Tileset\Darkshore\DarkshoreRoad.blp", ROAD),
            (r"Tileset\Duskwood\DuskwoodCobblestone.blp", ROAD),
            (r"Tileset\AraithiHighlands\ArathiHighlandsBrickFloor.blp", STRUCTURE),
            # TRAP 1: the zone DIRECTORY says "Swamp" but the texture is a road. Full-path matching
            # against a water rule would mislabel it; leaf-only matching plus road-first ordering
            # gets it right.
            (r"Tileset\Swamp of Sorrows\SwampSorrowsStoneRoad07.blp", ROAD),
            # TRAP 2: "Brick" is a structure token and "Road" a road token, in one leaf. Road must
            # win, because the surface is a road built of brick.
            (r"Tileset\Loch Modan\LochModanBrickRoadBase.blp", ROAD),
            # "stone" only reaches TERRAIN after every authored-surface rule has had its chance.
            (r"Tileset\RedRidge\RedridgeStoneHighlight.blp", TERRAIN),
            # Genuine placeholders/void stay unknown rather than being forced into a real class.
            (r"Tileset\Generic\Black.blp", UNKNOWN),
            (r"Tileset\Generic\Checkers.blp", UNKNOWN),
        ],
    )
    def test_real_corpus_paths_classify_correctly(self, path: str, expected: int) -> None:
        assert classify_texture_name(path) == expected

    def test_empty_name_is_unknown_not_an_error(self) -> None:
        assert classify_texture_name("") == UNKNOWN

    def test_matching_is_case_insensitive(self) -> None:
        assert classify_texture_name(r"Tileset\X\WETLANDSROAD01.blp") == ROAD

    def test_forward_slashes_are_normalized(self) -> None:
        assert texture_leaf("Tileset/Durotar/DurotarRoad.blp") == "DurotarRoad"

    def test_rule_set_hash_is_stable_and_hex(self) -> None:
        digest = rule_set_sha256()
        assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
        assert digest == rule_set_sha256()


class TestDominantLayer:
    def test_absent_alpha_is_base_layer_everywhere(self) -> None:
        dominant = resolve_dominant_layer(None, None)
        assert dominant.shape == (256, 256)
        assert not dominant.any()

    def test_highest_qualifying_layer_wins(self) -> None:
        alpha = np.zeros((256, 256, 4), dtype=np.float32)
        alpha[:, :, 1] = 0.9
        alpha[:, :, 3] = 0.9
        dominant = resolve_dominant_layer(alpha, None)
        assert (dominant == 3).all()

    def test_alpha_below_threshold_does_not_take_over(self) -> None:
        alpha = np.zeros((256, 256, 4), dtype=np.float32)
        alpha[:, :, 2] = DOMINANT_ALPHA_THRESHOLD - 0.01
        assert not resolve_dominant_layer(alpha, None).any()

    def test_layer_not_declared_by_chunk_is_ignored(self) -> None:
        alpha = np.zeros((256, 256, 4), dtype=np.float32)
        alpha[:, :, 2] = 1.0
        layer_mask = np.zeros((16, 16, 4), dtype=np.float32)  # chunk declares no overlay layers
        assert not resolve_dominant_layer(alpha, layer_mask).any()

    def test_malformed_alpha_shape_is_refused(self) -> None:
        with pytest.raises(TerrainFeatureLabelError):
            resolve_dominant_layer(np.zeros((8, 8, 4), dtype=np.float32), None)


class TestRowLabelDerivation:
    @staticmethod
    def _texture_ids(layer0: int = 0, layer1: int = 1) -> np.ndarray:
        ids = np.zeros((16, 16, 4), dtype=np.int32)
        ids[:, :, 0] = layer0
        ids[:, :, 1] = layer1
        return ids

    def test_base_layer_labels_whole_tile(self) -> None:
        labels, valid = derive_row_labels(
            texture_ids=self._texture_ids(),
            texture_names=[r"Tileset\X\XGrass.blp", r"Tileset\X\XRoad.blp"],
            alpha_256=None,
            layer_mask=None,
        )
        assert valid.all()
        assert (labels == TERRAIN).all()

    def test_overlay_layer_relabels_only_where_alpha_is_high(self) -> None:
        alpha = np.zeros((256, 256, 4), dtype=np.float32)
        alpha[:128, :, 1] = 1.0  # road overlay covers the top half only
        labels, valid = derive_row_labels(
            texture_ids=self._texture_ids(),
            texture_names=[r"Tileset\X\XGrass.blp", r"Tileset\X\XRoad.blp"],
            alpha_256=alpha,
            layer_mask=None,
        )
        assert valid.all()
        assert (labels[:128] == ROAD).all()
        assert (labels[128:] == TERRAIN).all()

    def test_missing_mtex_table_marks_everything_invalid(self) -> None:
        labels, valid = derive_row_labels(
            texture_ids=self._texture_ids(), texture_names=[], alpha_256=None, layer_mask=None
        )
        assert not valid.any()
        assert (labels == UNKNOWN).all()

    def test_out_of_range_local_index_is_invalid_not_guessed(self) -> None:
        ids = np.full((16, 16, 4), 7, dtype=np.int32)  # index 7 with a 1-entry table
        labels, valid = derive_row_labels(
            texture_ids=ids,
            texture_names=[r"Tileset\X\XGrass.blp"],
            alpha_256=None,
            layer_mask=None,
        )
        assert not valid.any()
        assert (labels == UNKNOWN).all()

    def test_malformed_texture_ids_shape_is_refused(self) -> None:
        with pytest.raises(TerrainFeatureLabelError):
            derive_row_labels(
                texture_ids=np.zeros((4, 4, 4), dtype=np.int32),
                texture_names=[r"Tileset\X\XGrass.blp"],
                alpha_256=None,
                layer_mask=None,
            )

    def test_summary_reconciles_to_the_full_tile(self) -> None:
        labels, valid = derive_row_labels(
            texture_ids=self._texture_ids(),
            texture_names=[r"Tileset\X\XGrass.blp", r"Tileset\X\XRoad.blp"],
            alpha_256=None,
            layer_mask=None,
        )
        summary = summarize_labels(labels, valid)
        assert sum(summary[name] for name in FAMILY_NAMES) + summary["invalid"] == 256 * 256


class TestClassifierContract:
    def test_model_emits_one_logit_per_family_at_input_resolution(self) -> None:
        torch = pytest.importorskip("torch")
        from harvester.v50.terrain_feature_model import build_terrain_feature_model

        model, identity = build_terrain_feature_model(base=4)
        out = model(torch.zeros(2, 3, 256, 256))
        assert out.shape == (2, CLASS_COUNT, 256, 256)
        assert identity["architecture"]["parameter_count"] > 0
        # Ground truth never reaches inference: the model takes exactly one RGB argument.
        assert identity["pretrained_source"] is None

    def test_model_refuses_non_rgb_input(self) -> None:
        torch = pytest.importorskip("torch")
        from harvester.v50.terrain_feature_model import (
            TerrainFeatureModelError,
            build_terrain_feature_model,
        )

        model, _ = build_terrain_feature_model(base=4)
        with pytest.raises(TerrainFeatureModelError):
            model(torch.zeros(1, 4, 256, 256))

    def test_taxonomy_revision_is_in_the_architecture_hash(self) -> None:
        """Same weights under a different taxonomy must not present the same identity."""
        pytest.importorskip("torch")
        from harvester.v50 import terrain_feature_model as tfm

        model, identity = tfm.build_terrain_feature_model(base=4)
        original = tfm.TAXONOMY_REVISION
        try:
            tfm.TAXONOMY_REVISION = "different-revision"
            shifted = tfm.terrain_feature_identity(model, base=4)
        finally:
            tfm.TAXONOMY_REVISION = original
        assert shifted["config_sha256"] != identity["architecture"]["config_sha256"]


class TestClassWeighting:
    def test_rare_class_outweighs_common_class(self) -> None:
        from harvester.v50.terrain_feature_train import compute_class_weights

        # Real measured distribution shape: terrain dominant, road ~0.26%.
        weights = compute_class_weights(
            {"unknown": 92484, "terrain": 169382676, "road": 471556,
             "water": 1557054, "structure": 7343974}
        )
        assert len(weights) == CLASS_COUNT
        assert weights[ROAD] > weights[TERRAIN]

    def test_weights_are_capped(self) -> None:
        from harvester.v50.terrain_feature_train import compute_class_weights

        weights = compute_class_weights(
            {"unknown": 1, "terrain": 10**9, "road": 1, "water": 1, "structure": 1},
            max_weight=25.0,
        )
        assert max(weights) <= 25.0

    def test_absent_class_gets_neutral_weight_not_infinity(self) -> None:
        from harvester.v50.terrain_feature_train import compute_class_weights

        weights = compute_class_weights(
            {"unknown": 0, "terrain": 100, "road": 0, "water": 0, "structure": 0}
        )
        assert all(np.isfinite(w) for w in weights)
        assert weights[ROAD] == 1.0

    def test_unknown_is_masked_out_by_default(self) -> None:
        """Unknown is an absence-of-information marker, not a terrain class: weight 0 unless opted in."""
        from harvester.v50.terrain_feature_train import compute_class_weights

        counts = {"unknown": 92484, "terrain": 169382676, "road": 471556,
                  "water": 1557054, "structure": 7343974}
        default = compute_class_weights(counts)
        assert default[UNKNOWN] == 0.0
        assert default[ROAD] > 0.0

        supervised = compute_class_weights(counts, supervise_unknown=True)
        assert supervised[UNKNOWN] > 0.0

    def test_empty_label_counts_are_refused(self) -> None:
        from harvester.v50.height_relative_train import TrainerContractError
        from harvester.v50.terrain_feature_train import compute_class_weights

        with pytest.raises(TrainerContractError):
            compute_class_weights(dict.fromkeys(FAMILY_NAMES, 0))


class TestConfusionMetrics:
    def test_degenerate_majority_prediction_scores_zero_road_iou(self) -> None:
        """The 'predict terrain everywhere' solution must be visibly worthless on the road metric
        even though its pixel accuracy is high -- this is the trap the gate exists to catch."""
        from harvester.v50.terrain_feature_train import confusion_metrics

        confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
        confusion[TERRAIN, TERRAIN] = 9900
        confusion[ROAD, TERRAIN] = 100  # every road pixel predicted as terrain
        metrics = confusion_metrics(confusion)
        assert metrics["per_class"]["road"]["iou"] == 0.0
        assert metrics["per_class"]["road"]["recall"] == 0.0
        assert metrics["pixel_accuracy"] == pytest.approx(0.99)

    def test_perfect_prediction_scores_unit_iou(self) -> None:
        from harvester.v50.terrain_feature_train import confusion_metrics

        confusion = np.diag(np.array([10, 100, 20, 30, 40], dtype=np.int64))
        metrics = confusion_metrics(confusion)
        assert metrics["macro_iou"] == pytest.approx(1.0)
        assert metrics["per_class"]["road"]["iou"] == pytest.approx(1.0)
