"""Spec 119 quality-lens tests (T027): determinism, near-duplicates, mislabels, coverage."""

from __future__ import annotations

import numpy as np
import torch

from harvester.spec119 import quality_lens
from harvester.spec119.classifier_model import ObjectClassifier


def _frozen_classifier(base: int = 8) -> ObjectClassifier:
    torch.manual_seed(3)
    model = ObjectClassifier(base=base, num_classes=4)
    model.eval()
    return model


def test_embedding_determinism_byte_identical(library_store) -> None:
    import zarr

    from harvester.spec119.library_data import captured_rows, load_asset_rows

    group = zarr.open_group(str(library_store), mode="r")
    rows = captured_rows(load_asset_rows(library_store))
    first = quality_lens.compute_embeddings(_frozen_classifier(), group, rows)
    second = quality_lens.compute_embeddings(_frozen_classifier(), group, rows)
    assert first.dtype == np.float32
    assert first.shape == (len(rows), 8 * 8)
    assert first.tobytes() == second.tobytes()  # FR-009: frozen checkpoint -> identical


def test_near_duplicate_pair_detection() -> None:
    ids = ["a", "b", "c"]
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1e-7, 0.0],  # near-identical to 'a'
            [0.0, 1.0, 0.0],   # orthogonal
        ],
        dtype=np.float32,
    )
    pairs = quality_lens.find_near_duplicates(embeddings, ids, threshold=0.95, top_k=10)
    assert len(pairs) == 1
    assert {pairs[0]["library_id_a"], pairs[0]["library_id_b"]} == {"a", "b"}
    assert pairs[0]["cosine_similarity"] >= 0.95
    assert quality_lens.find_near_duplicates(embeddings[:1], ids[:1]) == []


def test_mislabel_report_sorted_by_wrong_class_confidence() -> None:
    rows = [
        {"library_id": "x", "normalized_asset_path": "p/x"},
        {"library_id": "y", "normalized_asset_path": "p/y"},
    ]
    class_index = {"empty": 0, "m2": 1, "mdx": 2, "wmo": 3}
    predictions = [(3, 0.60, None), (2, 0.95, None)]  # both disagree with label 1 (m2)
    labeled = [1, 1]
    report = quality_lens.find_mislabels(rows, predictions, labeled, class_index)
    assert [entry["library_id"] for entry in report] == ["y", "x"]  # 0.95 before 0.60
    assert report[0]["labeled_class"] == "m2"
    assert report[0]["predicted_class"] == "mdx"
    # Agreement produces no report rows.
    assert quality_lens.find_mislabels(rows, [(1, 0.5, None), (1, 0.5, None)], labeled, class_index) == []


def test_low_coverage_flag_list() -> None:
    rows = [
        {"library_id": "x", "normalized_asset_path": "p/x"},
        {"library_id": "y", "normalized_asset_path": "p/y"},
    ]
    flags = quality_lens.flag_low_coverage(rows, [0.005, 0.5], 0.01)
    assert len(flags) == 1
    assert flags[0]["library_id"] == "x"
    assert flags[0]["coverage"] == 0.005
