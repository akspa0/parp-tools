"""Alpha-brush extraction and catalog helpers for spec 074.

The library operates on V18 `alpha_256` arrays and keeps all client-file parsing
outside this module.  Inputs are already-harvested NumPy/Zarr arrays; outputs are
plain dataclasses that can be serialized to JSONL for later bulk analysis.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import find_objects, label
from scipy.ndimage import sum as nd_sum


@dataclass(slots=True)
class BrushComponent:
    component_id: str
    build: str
    map_name: str
    tile_id: int
    tile_x: int
    tile_y: int
    layer_idx: int
    bbox_xywh: tuple[int, int, int, int]
    area: int
    threshold: float
    touches_edge: bool
    alpha_patch: np.ndarray | None = None
    mask_patch: np.ndarray | None = None
    embedding: np.ndarray | None = None
    cluster_id: int | None = None


@dataclass(slots=True)
class BrushCluster:
    cluster_id: int
    member_count: int
    centroid_embedding: list[float]
    representative_component_ids: list[str]
    dominant_layer: int | None
    dominant_map: str | None


@dataclass(slots=True)
class BrushCatalogEntry:
    component_id: str
    cluster_id: int
    build: str
    map_name: str
    tile_id: int
    tile_x: int
    tile_y: int
    layer_idx: int
    bbox_xywh: tuple[int, int, int, int]
    area: int
    threshold: float
    touches_edge: bool


def extract_components(
    alpha_pack: np.ndarray,
    layer_idx: int,
    threshold: float = 0.05,
    min_area: int = 16,
    reject_edge: bool = True,
    *,
    build: str = "",
    map_name: str = "",
    tile_id: int = -1,
    tile_x: int = -1,
    tile_y: int = -1,
    connectivity: int = 8,
) -> list[BrushComponent]:
    """Extract connected alpha-mask components from one layer of a 256x256x4 pack."""
    alpha = np.asarray(alpha_pack, dtype=np.float32)
    if alpha.ndim != 3:
        raise ValueError(f"alpha_pack must be HxWxL, got shape {alpha.shape}")
    if layer_idx < 0 or layer_idx >= alpha.shape[2]:
        raise ValueError(f"layer_idx {layer_idx} outside alpha layer count {alpha.shape[2]}")

    layer = alpha[:, :, layer_idx]
    structure = np.ones((3, 3), dtype=np.uint8) if int(connectivity) == 8 else None
    labeled, count = label(layer > float(threshold), structure=structure)
    if count == 0:
        return []

    areas = nd_sum(labeled > 0, labeled, range(1, count + 1))
    objects = find_objects(labeled)
    height, width = layer.shape
    components: list[BrushComponent] = []
    for label_idx, raw_area in enumerate(areas, start=1):
        area = int(raw_area)
        if area < int(min_area):
            continue
        bounds = objects[label_idx - 1]
        if bounds is None:
            continue
        slice_y, slice_x = bounds
        x0, x1 = int(slice_x.start), int(slice_x.stop)
        y0, y1 = int(slice_y.start), int(slice_y.stop)
        touches_edge = x0 == 0 or y0 == 0 or x1 == width or y1 == height
        if reject_edge and touches_edge:
            continue

        mask_patch = labeled[y0:y1, x0:x1] == label_idx
        alpha_patch = np.where(mask_patch, layer[y0:y1, x0:x1], 0.0).astype(np.float32, copy=False)
        component_id = _component_id(
            build=build,
            map_name=map_name,
            tile_id=tile_id,
            layer_idx=layer_idx,
            threshold=threshold,
            bbox_xywh=(x0, y0, x1 - x0, y1 - y0),
        )
        components.append(
            BrushComponent(
                component_id=component_id,
                build=build,
                map_name=map_name,
                tile_id=int(tile_id),
                tile_x=int(tile_x),
                tile_y=int(tile_y),
                layer_idx=int(layer_idx),
                bbox_xywh=(x0, y0, x1 - x0, y1 - y0),
                area=area,
                threshold=float(threshold),
                touches_edge=bool(touches_edge),
                alpha_patch=alpha_patch,
                mask_patch=mask_patch.astype(np.uint8, copy=False),
            )
        )
    return components


def render_component_patch(
    component: BrushComponent,
    target_size: int = 224,
    padding: int = 16,
    fill: float = 0.0,
) -> np.ndarray:
    """Render a component's alpha crop as a centered square float32 grayscale patch."""
    if component.alpha_patch is None:
        raise ValueError(f"Component {component.component_id} has no alpha_patch")
    crop = np.clip(np.asarray(component.alpha_patch, dtype=np.float32), 0.0, 1.0)
    padded = np.pad(
        crop,
        ((int(padding), int(padding)), (int(padding), int(padding))),
        mode="constant",
        constant_values=float(fill),
    )
    src_h, src_w = padded.shape
    scale = min(float(target_size) / max(1, src_w), float(target_size) / max(1, src_h))
    dst_w = max(1, int(round(src_w * scale)))
    dst_h = max(1, int(round(src_h * scale)))

    image = Image.fromarray((padded * 255.0).astype(np.uint8), mode="L")
    image = image.resize((dst_w, dst_h), resample=Image.Resampling.BILINEAR)
    canvas = Image.new("L", (int(target_size), int(target_size)), color=int(np.clip(fill, 0.0, 1.0) * 255.0))
    canvas.paste(image, ((int(target_size) - dst_w) // 2, (int(target_size) - dst_h) // 2))
    return (np.asarray(canvas, dtype=np.float32) / 255.0).astype(np.float32, copy=False)


def load_dinov2_model(model_name: str = "facebook/dinov2-small", device: str = "cuda") -> tuple[Any, Any]:
    """Load a Hugging Face DINOv2 image processor and model."""
    import torch
    from transformers import AutoImageProcessor, Dinov2Model

    actual_device = device
    if actual_device == "cuda" and not torch.cuda.is_available():
        actual_device = "cpu"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = Dinov2Model.from_pretrained(model_name)
    model.to(torch.device(actual_device))
    model.eval()
    return model, processor


def compute_dinov2_embeddings(
    patches: np.ndarray,
    model: Any,
    processor: Any,
    batch_size: int = 64,
    *,
    token_strategy: str = "mean",
) -> np.ndarray:
    """Compute L2-normalized DINOv2 embeddings for grayscale component patches."""
    import torch

    patch_array = np.asarray(patches, dtype=np.float32)
    if patch_array.ndim != 3:
        raise ValueError(f"patches must be NxHxW, got shape {patch_array.shape}")
    if patch_array.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)

    try:
        device = next(model.parameters()).device
    except (AttributeError, StopIteration):
        device = torch.device("cpu")

    chunks: list[np.ndarray] = []
    rgb = np.repeat((np.clip(patch_array, 0.0, 1.0) * 255.0).astype(np.uint8)[:, :, :, None], 3, axis=3)
    with torch.inference_mode():
        for start in range(0, rgb.shape[0], int(batch_size)):
            images = [Image.fromarray(arr, mode="RGB") for arr in rgb[start : start + int(batch_size)]]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state.detach().float()
            if token_strategy == "cls":
                embedding = hidden[:, 0]
            elif token_strategy == "mean":
                embedding = hidden[:, 1:].mean(dim=1)
            else:
                raise ValueError(f"Unknown token_strategy: {token_strategy}")
            chunks.append(_l2_normalize(embedding.cpu().numpy()))
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def cluster_components(
    components: list[BrushComponent],
    algorithm: str = "hdbscan",
    min_cluster_size: int = 10,
    fallback_k: int = 100,
    random_state: int = 74,
) -> list[BrushComponent]:
    """Assign cluster IDs from component embeddings, returning updated components."""
    if not components:
        return []
    embeddings = _component_embeddings(components)
    labels: np.ndarray | None = None

    if algorithm == "hdbscan":
        labels = _try_hdbscan(embeddings, min_cluster_size)
        if labels is not None:
            noise_fraction = float(np.mean(labels < 0)) if labels.size else 1.0
            if noise_fraction > 0.5 or len(set(labels.tolist()) - {-1}) == 0:
                labels = None
    if labels is None:
        k = max(1, min(int(fallback_k), len(components)))
        labels = _kmeans_labels(embeddings, k, random_state=random_state)

    remapped = _remap_cluster_labels(labels, embeddings)
    return [replace(component, cluster_id=int(cluster_id)) for component, cluster_id in zip(components, remapped, strict=True)]


def build_cluster_catalog(components: list[BrushComponent]) -> list[BrushCluster]:
    """Build per-cluster summary rows from clustered components."""
    by_cluster: dict[int, list[BrushComponent]] = {}
    for component in components:
        if component.cluster_id is None or component.cluster_id < 0:
            continue
        by_cluster.setdefault(int(component.cluster_id), []).append(component)

    clusters: list[BrushCluster] = []
    for cluster_id, members in sorted(by_cluster.items()):
        embeddings = _component_embeddings(members)
        centroid = _l2_normalize(embeddings.mean(axis=0, keepdims=True))[0]
        distances = 1.0 - np.dot(embeddings, centroid)
        representatives = [members[idx].component_id for idx in np.argsort(distances)[: min(16, len(members))]]
        clusters.append(
            BrushCluster(
                cluster_id=int(cluster_id),
                member_count=len(members),
                centroid_embedding=[float(value) for value in centroid.tolist()],
                representative_component_ids=representatives,
                dominant_layer=_dominant_value([member.layer_idx for member in members]),
                dominant_map=_dominant_value([member.map_name for member in members]),
            )
        )
    return clusters


def build_catalog_entries(components: list[BrushComponent]) -> list[BrushCatalogEntry]:
    """Build component-to-cluster rows for `catalog.jsonl`."""
    entries: list[BrushCatalogEntry] = []
    for component in components:
        if component.cluster_id is None:
            continue
        entries.append(
            BrushCatalogEntry(
                component_id=component.component_id,
                cluster_id=int(component.cluster_id),
                build=component.build,
                map_name=component.map_name,
                tile_id=component.tile_id,
                tile_x=component.tile_x,
                tile_y=component.tile_y,
                layer_idx=component.layer_idx,
                bbox_xywh=component.bbox_xywh,
                area=component.area,
                threshold=component.threshold,
                touches_edge=component.touches_edge,
            )
        )
    return entries


def save_components(path: str | Path, components: Iterable[BrushComponent]) -> None:
    _write_jsonl(path, (_component_to_row(component) for component in components))


def save_clusters(path: str | Path, clusters: Iterable[BrushCluster]) -> None:
    _write_jsonl(path, (_json_ready(asdict(cluster)) for cluster in clusters))


def save_catalog(path: str | Path, entries: Iterable[BrushCatalogEntry]) -> None:
    _write_jsonl(path, (_json_ready(asdict(entry)) for entry in entries))


def _component_id(
    *,
    build: str,
    map_name: str,
    tile_id: int,
    layer_idx: int,
    threshold: float,
    bbox_xywh: tuple[int, int, int, int],
) -> str:
    payload = f"{build}|{map_name}|{tile_id}|{layer_idx}|{threshold:.4f}|{bbox_xywh}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"brush_{digest}"


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    denom = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(denom, 1e-8)


def _component_embeddings(components: list[BrushComponent]) -> np.ndarray:
    missing = [component.component_id for component in components if component.embedding is None]
    if missing:
        raise ValueError(f"Components missing embeddings: {missing[:5]}")
    embeddings = np.stack([np.asarray(component.embedding, dtype=np.float32) for component in components], axis=0)
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got shape {embeddings.shape}")
    return _l2_normalize(embeddings)


def _try_hdbscan(embeddings: np.ndarray, min_cluster_size: int) -> np.ndarray | None:
    try:
        import hdbscan  # type: ignore[import-untyped]
    except ImportError:
        return None
    clusterer = hdbscan.HDBSCAN(min_cluster_size=max(2, int(min_cluster_size)), metric="euclidean")
    return np.asarray(clusterer.fit_predict(embeddings), dtype=np.int32)


def _kmeans_labels(embeddings: np.ndarray, k: int, random_state: int) -> np.ndarray:
    try:
        from sklearn.cluster import KMeans

        return np.asarray(KMeans(n_clusters=int(k), n_init=10, random_state=int(random_state)).fit_predict(embeddings), dtype=np.int32)
    except ImportError:
        return _fallback_kmeans_labels(embeddings, k)


def _fallback_kmeans_labels(embeddings: np.ndarray, k: int, iterations: int = 30) -> np.ndarray:
    k = max(1, min(int(k), embeddings.shape[0]))
    centroids = embeddings[np.linspace(0, embeddings.shape[0] - 1, k, dtype=np.int32)].copy()
    labels = np.zeros((embeddings.shape[0],), dtype=np.int32)
    for _ in range(int(iterations)):
        distances = 1.0 - np.dot(embeddings, centroids.T)
        next_labels = np.argmin(distances, axis=1).astype(np.int32)
        if np.array_equal(labels, next_labels):
            break
        labels = next_labels
        for cluster_id in range(k):
            members = embeddings[labels == cluster_id]
            if members.size:
                centroids[cluster_id] = _l2_normalize(members.mean(axis=0, keepdims=True))[0]
    return labels


def _remap_cluster_labels(labels: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    unique = sorted(int(label_value) for label_value in np.unique(labels) if int(label_value) >= 0)
    summaries: list[tuple[int, int, str]] = []
    for label_value in unique:
        members = embeddings[labels == label_value]
        centroid = _l2_normalize(members.mean(axis=0, keepdims=True))[0]
        digest = hashlib.sha256(np.round(centroid, 6).tobytes()).hexdigest()
        summaries.append((label_value, -int(members.shape[0]), digest))
    mapping = {old: new for new, (old, _, _) in enumerate(sorted(summaries, key=lambda item: (item[1], item[2])))}
    return np.asarray([mapping.get(int(label_value), -1) for label_value in labels], dtype=np.int32)


def _dominant_value(values: list[Any]) -> Any | None:
    if not values:
        return None
    counts: dict[Any, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))[0][0]


def _component_to_row(component: BrushComponent) -> dict[str, Any]:
    row = asdict(component)
    row.pop("alpha_patch", None)
    row.pop("mask_patch", None)
    row["embedding"] = None if component.embedding is None else [float(value) for value in component.embedding.tolist()]
    return _json_ready(row)


def _write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_ready(row), sort_keys=True) + "\n")


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value
