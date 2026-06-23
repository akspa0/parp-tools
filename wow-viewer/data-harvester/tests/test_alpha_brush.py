from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from harvester.alpha_brush import (  # noqa: E402
    BrushComponent,
    build_catalog_entries,
    build_cluster_catalog,
    cluster_components,
    compute_dinov2_embeddings,
    extract_components,
    render_component_patch,
    save_catalog,
    save_clusters,
    save_components,
)


def _alpha_pack() -> np.ndarray:
    alpha = np.zeros((64, 64, 4), dtype=np.float32)
    alpha[10:22, 11:23, 1] = 0.8
    alpha[34:42, 36:48, 1] = 0.9
    alpha[0:8, 0:8, 1] = 0.7
    return alpha


def test_extract_components_filters_edges_and_preserves_metadata() -> None:
    components = extract_components(
        _alpha_pack(),
        layer_idx=1,
        threshold=0.05,
        min_area=16,
        reject_edge=True,
        build="0_5_3_3368",
        map_name="Azeroth",
        tile_id=7,
        tile_x=24,
        tile_y=53,
    )

    assert len(components) == 2
    assert [component.area for component in components] == [144, 96]
    assert components[0].map_name == "Azeroth"
    assert components[0].bbox_xywh == (11, 10, 12, 12)
    assert components[0].alpha_patch is not None
    assert components[0].mask_patch is not None
    assert not components[0].touches_edge


def test_render_component_patch_centers_alpha_crop() -> None:
    component = extract_components(_alpha_pack(), layer_idx=1, reject_edge=True)[0]

    patch = render_component_patch(component, target_size=64, padding=4)

    assert patch.shape == (64, 64)
    assert patch.dtype == np.float32
    assert patch.max() > 0.7
    assert patch[:4, :].max() == 0.0
    assert patch[:, :4].max() == 0.0


class _FakeProcessor:
    def __call__(self, images, return_tensors: str):
        assert return_tensors == "pt"
        tensors = []
        for image in images:
            arr = np.asarray(image, dtype=np.float32) / 255.0
            tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
        return {"pixel_values": torch.stack(tensors, dim=0)}


class _FakeDinov2(torch.nn.Module):
    def forward(self, pixel_values: torch.Tensor):
        mean = pixel_values.mean(dim=(1, 2, 3))
        std = pixel_values.std(dim=(1, 2, 3))
        cls = torch.stack([mean, 1.0 - mean, std], dim=1)
        patch = torch.stack([std, mean, 1.0 - std], dim=1)
        hidden = torch.stack([cls, patch, patch * 0.5 + 0.25], dim=1)
        return SimpleNamespace(last_hidden_state=hidden)


def test_compute_dinov2_embeddings_uses_dinov2_shaped_outputs() -> None:
    patches = np.zeros((2, 16, 16), dtype=np.float32)
    patches[1, 4:12, 4:12] = 1.0

    embeddings = compute_dinov2_embeddings(
        patches,
        _FakeDinov2(),
        _FakeProcessor(),
        batch_size=1,
        token_strategy="mean",
    )

    assert embeddings.shape == (2, 3)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)
    assert not np.allclose(embeddings[0], embeddings[1])


def test_cluster_components_groups_similar_synthetic_embeddings() -> None:
    components = [
        BrushComponent("a", "b", "m", 0, 0, 0, 1, (0, 0, 8, 8), 64, 0.05, False, embedding=np.array([1.0, 0.0])),
        BrushComponent("b", "b", "m", 1, 0, 1, 1, (0, 0, 8, 8), 64, 0.05, False, embedding=np.array([0.99, 0.01])),
        BrushComponent("c", "b", "m", 2, 0, 2, 2, (0, 0, 8, 8), 64, 0.05, False, embedding=np.array([-1.0, 0.0])),
        BrushComponent("d", "b", "m", 3, 0, 3, 2, (0, 0, 8, 8), 64, 0.05, False, embedding=np.array([-0.99, 0.01])),
    ]

    clustered = cluster_components(components, algorithm="kmeans", fallback_k=2)

    assert clustered[0].cluster_id == clustered[1].cluster_id
    assert clustered[2].cluster_id == clustered[3].cluster_id
    assert clustered[0].cluster_id != clustered[2].cluster_id


def test_catalog_and_jsonl_serialization(tmp_path) -> None:
    components = [
        BrushComponent("a", "b", "m", 0, 0, 0, 1, (0, 0, 8, 8), 64, 0.05, False, embedding=np.array([1.0, 0.0]), cluster_id=0),
        BrushComponent("b", "b", "m", 1, 0, 1, 1, (2, 2, 8, 8), 64, 0.05, False, embedding=np.array([0.9, 0.1]), cluster_id=0),
    ]

    clusters = build_cluster_catalog(components)
    entries = build_catalog_entries(components)
    save_components(tmp_path / "components.jsonl", components)
    save_clusters(tmp_path / "clusters.jsonl", clusters)
    save_catalog(tmp_path / "catalog.jsonl", entries)

    assert len(clusters) == 1
    assert clusters[0].member_count == 2
    assert clusters[0].dominant_layer == 1
    assert len(entries) == 2
    assert (tmp_path / "components.jsonl").read_text(encoding="utf-8").count("\n") == 2
    assert (tmp_path / "clusters.jsonl").read_text(encoding="utf-8").count("\n") == 1
    assert (tmp_path / "catalog.jsonl").read_text(encoding="utf-8").count("\n") == 2
