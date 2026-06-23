"""Phase 0 research for spec 074: alpha-mask brush components.

Reads V18 Zarr `alpha_256` arrays, extracts connected components per alpha
layer, renders component patches, optionally embeds them with DINOv2, and saves
a PCA projection for visual inspection.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import zarr
from PIL import Image
from scipy.ndimage import find_objects, label
from scipy.ndimage import sum as nd_sum

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATASET_DIR = _PROJECT_ROOT / "output" / "datasets" / "v18"
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "output" / "analysis" / "alpha-brush-library" / "research"


@dataclass(frozen=True)
class AlphaComponent:
    component_id: str
    build: str
    map_name: str
    tile_id: int
    tile_x: int
    tile_y: int
    layer_idx: int
    threshold: float
    bbox_xywh: tuple[int, int, int, int]
    area: int
    touches_edge: bool
    alpha_crop: np.ndarray
    mask_crop: np.ndarray


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research MCAL alpha connected components and DINOv2 embeddings."
    )
    parser.add_argument("--dataset-dir", type=Path, default=_DEFAULT_DATASET_DIR)
    parser.add_argument("--build", default="0_5_3_3368")
    parser.add_argument("--map", dest="map_name", default=None)
    parser.add_argument("--tile-limit", type=int, default=12)
    parser.add_argument("--thresholds", default="0.03,0.05,0.10")
    parser.add_argument("--min-area", type=int, default=16)
    parser.add_argument("--reject-edge", action="store_true")
    parser.add_argument("--max-components", type=int, default=192)
    parser.add_argument("--examples-per-layer", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default="facebook/dinov2-small")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--skip-dinov2", action="store_true")
    parser.add_argument("--seed", type=int, default=74)
    return parser.parse_args()


def _parse_thresholds(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one threshold is required")
    return values


def _row_get(table: Any | None, row_idx: int, column: str, default: Any) -> Any:
    if table is None or column not in table.column_names:
        return default
    value = table.column(column)[row_idx].as_py()
    return default if value is None else value


def _open_build(dataset_dir: Path, build: str) -> tuple[zarr.storage.LocalStore, zarr.Group, Any | None]:
    zarr_path = dataset_dir / f"{build}.zarr"
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr store not found: {zarr_path}")

    store = zarr.storage.LocalStore(str(zarr_path), read_only=True)
    root = zarr.open_group(store=store, mode="r")
    if "alpha_256" not in root:
        store.close()
        raise KeyError(f"alpha_256 array not found in {zarr_path}")

    index_path = zarr_path / "index.parquet"
    table = pq.read_table(str(index_path)) if index_path.exists() else None
    return store, root, table


def _select_tile_ids(
    root: zarr.Group,
    table: Any | None,
    map_name: str | None,
    tile_limit: int,
) -> list[int]:
    alpha_count = int(root["alpha_256"].shape[0])
    candidates: list[int] = []

    if table is None:
        candidates = list(range(alpha_count))
    else:
        for row_idx in range(min(table.num_rows, alpha_count)):
            if map_name is not None and str(_row_get(table, row_idx, "map", "")) != map_name:
                continue
            if "has_alpha_256" in table.column_names and not bool(_row_get(table, row_idx, "has_alpha_256", False)):
                continue
            candidates.append(row_idx)

    if not candidates:
        selector = f" map={map_name!r}" if map_name else ""
        raise RuntimeError(f"No alpha-bearing tiles found for{selector}")
    return candidates[: max(1, int(tile_limit))]


def _extract_components(
    alpha_pack: np.ndarray,
    *,
    build: str,
    map_name: str,
    tile_id: int,
    tile_x: int,
    tile_y: int,
    layer_idx: int,
    threshold: float,
    min_area: int,
    reject_edge: bool,
) -> list[AlphaComponent]:
    layer = np.asarray(alpha_pack[:, :, layer_idx], dtype=np.float32)
    binary = layer > float(threshold)
    labeled, count = label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return []

    areas = nd_sum(binary, labeled, range(1, count + 1))
    slices = find_objects(labeled)
    components: list[AlphaComponent] = []
    height, width = layer.shape

    for label_idx, area_value in enumerate(areas, start=1):
        area = int(area_value)
        if area < int(min_area):
            continue
        bounds = slices[label_idx - 1]
        if bounds is None:
            continue
        slice_y, slice_x = bounds
        x0, x1 = int(slice_x.start), int(slice_x.stop)
        y0, y1 = int(slice_y.start), int(slice_y.stop)
        touches_edge = x0 == 0 or y0 == 0 or x1 == width or y1 == height
        if reject_edge and touches_edge:
            continue

        mask_crop = labeled[y0:y1, x0:x1] == label_idx
        alpha_crop = np.where(mask_crop, layer[y0:y1, x0:x1], 0.0).astype(np.float32, copy=False)
        component_id = f"{build}:{map_name}:{tile_id}:l{layer_idx}:t{threshold:.2f}:c{label_idx}"
        components.append(
            AlphaComponent(
                component_id=component_id,
                build=build,
                map_name=map_name,
                tile_id=tile_id,
                tile_x=tile_x,
                tile_y=tile_y,
                layer_idx=layer_idx,
                threshold=float(threshold),
                bbox_xywh=(x0, y0, x1 - x0, y1 - y0),
                area=area,
                touches_edge=touches_edge,
                alpha_crop=alpha_crop,
                mask_crop=mask_crop.astype(np.uint8, copy=False),
            )
        )
    return components


def _render_component_patch(component: AlphaComponent, target_size: int = 224, padding: int = 16) -> np.ndarray:
    crop = np.clip(component.alpha_crop, 0.0, 1.0)
    padded = np.pad(crop, ((padding, padding), (padding, padding)), mode="constant", constant_values=0.0)
    src_h, src_w = padded.shape
    scale = min(float(target_size) / max(1, src_w), float(target_size) / max(1, src_h))
    dst_w = max(1, int(round(src_w * scale)))
    dst_h = max(1, int(round(src_h * scale)))

    img = Image.fromarray((padded * 255.0).astype(np.uint8), mode="L")
    img = img.resize((dst_w, dst_h), resample=Image.Resampling.BILINEAR)
    canvas = Image.new("L", (target_size, target_size), color=0)
    canvas.paste(img, ((target_size - dst_w) // 2, (target_size - dst_h) // 2))
    return np.asarray(canvas, dtype=np.float32) / 255.0


def _save_patch_examples(
    components: list[AlphaComponent],
    output_dir: Path,
    examples_per_layer: int,
) -> None:
    patch_dir = output_dir / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    by_layer: dict[int, list[AlphaComponent]] = {0: [], 1: [], 2: [], 3: []}
    for component in sorted(components, key=lambda c: c.area, reverse=True):
        bucket = by_layer.setdefault(component.layer_idx, [])
        if len(bucket) < int(examples_per_layer):
            bucket.append(component)

    for layer_idx, layer_components in sorted(by_layer.items()):
        for idx, component in enumerate(layer_components):
            patch = _render_component_patch(component)
            path = patch_dir / f"layer{layer_idx}_example{idx:02d}_area{component.area}.png"
            Image.fromarray((patch * 255.0).astype(np.uint8), mode="L").save(path)


def _load_dinov2(model_name: str, device: torch.device):
    from transformers import AutoImageProcessor, Dinov2Model

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = Dinov2Model.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return processor, model


def _embed_patches(
    patches: np.ndarray,
    *,
    model_name: str,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    processor, model = _load_dinov2(model_name, device)
    cls_chunks: list[np.ndarray] = []
    mean_chunks: list[np.ndarray] = []

    rgb_u8 = np.repeat((np.clip(patches, 0.0, 1.0) * 255.0).astype(np.uint8)[:, :, :, None], 3, axis=3)
    with torch.inference_mode():
        for start in range(0, rgb_u8.shape[0], int(batch_size)):
            batch = [Image.fromarray(arr, mode="RGB") for arr in rgb_u8[start : start + int(batch_size)]]
            inputs = processor(images=batch, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            output = model(**inputs)
            hidden = output.last_hidden_state.detach().float()
            cls = hidden[:, 0]
            mean = hidden[:, 1:].mean(dim=1)
            cls_chunks.append(_l2_normalize(cls.cpu().numpy()))
            mean_chunks.append(_l2_normalize(mean.cpu().numpy()))

    return np.concatenate(cls_chunks, axis=0), np.concatenate(mean_chunks, axis=0)


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(denom, 1e-8)


def _pca2(values: np.ndarray) -> np.ndarray:
    centered = values.astype(np.float64, copy=False) - values.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return (centered @ vt[:2].T).astype(np.float32)


def _save_projection(
    points: np.ndarray,
    components: list[AlphaComponent],
    output_dir: Path,
    name: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
    layers = np.asarray([component.layer_idx for component in components], dtype=np.int32)
    scatter = ax.scatter(points[:, 0], points[:, 1], c=layers, s=14, cmap="tab10", alpha=0.82)
    ax.set_title(f"DINOv2 {name} PCA projection colored by alpha layer")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, linewidth=0.35, alpha=0.35)
    legend = ax.legend(*scatter.legend_elements(), title="Layer", loc="best")
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(output_dir / f"projection_{name}.png")
    if name == "mean":
        fig.savefig(output_dir / "projection.png")
    plt.close(fig)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _component_summary_row(component: AlphaComponent) -> dict[str, Any]:
    return {
        "component_id": component.component_id,
        "build": component.build,
        "map": component.map_name,
        "tile_id": component.tile_id,
        "tile_x": component.tile_x,
        "tile_y": component.tile_y,
        "layer_idx": component.layer_idx,
        "threshold": component.threshold,
        "bbox_xywh": list(component.bbox_xywh),
        "area": component.area,
        "touches_edge": component.touches_edge,
    }


def _cluster_preview(points: np.ndarray, components: list[AlphaComponent], k: int = 8) -> list[dict[str, Any]]:
    if len(components) == 0:
        return []
    k = max(1, min(int(k), len(components)))
    order = np.argsort(points[:, 0])
    buckets = np.array_split(order, k)
    rows: list[dict[str, Any]] = []
    for idx, bucket in enumerate(buckets):
        if len(bucket) == 0:
            continue
        layer_counts: dict[str, int] = {}
        areas: list[int] = []
        for component_idx in bucket.tolist():
            component = components[component_idx]
            layer_counts[str(component.layer_idx)] = layer_counts.get(str(component.layer_idx), 0) + 1
            areas.append(component.area)
        rows.append(
            {
                "bucket": idx,
                "count": int(len(bucket)),
                "layer_counts": layer_counts,
                "area_mean": float(np.mean(areas)),
                "area_min": int(min(areas)),
                "area_max": int(max(areas)),
            }
        )
    return rows


def main() -> None:
    args = _parse_args()
    thresholds = _parse_thresholds(args.thresholds)
    rng = np.random.default_rng(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

    store, root, table = _open_build(args.dataset_dir, args.build)
    try:
        tile_ids = _select_tile_ids(root, table, args.map_name, args.tile_limit)
        alpha = root["alpha_256"]

        summary: dict[str, Any] = {
            "build": args.build,
            "dataset_dir": str(args.dataset_dir),
            "map_filter": args.map_name,
            "tile_ids": tile_ids,
            "thresholds": thresholds,
            "min_area": int(args.min_area),
            "reject_edge": bool(args.reject_edge),
            "component_counts": {},
            "outputs": {},
        }
        all_default_components: list[AlphaComponent] = []

        for threshold in thresholds:
            threshold_components: list[AlphaComponent] = []
            layer_counts = {str(layer_idx): 0 for layer_idx in range(4)}
            for tile_id in tile_ids:
                alpha_pack = np.asarray(alpha[tile_id], dtype=np.float32)
                map_name = str(_row_get(table, tile_id, "map", "unknown"))
                tile_x = int(_row_get(table, tile_id, "tile_x", -1) or -1)
                tile_y = int(_row_get(table, tile_id, "tile_y", -1) or -1)
                for layer_idx in range(min(4, alpha_pack.shape[-1])):
                    components = _extract_components(
                        alpha_pack,
                        build=args.build,
                        map_name=map_name,
                        tile_id=tile_id,
                        tile_x=tile_x,
                        tile_y=tile_y,
                        layer_idx=layer_idx,
                        threshold=threshold,
                        min_area=args.min_area,
                        reject_edge=args.reject_edge,
                    )
                    layer_counts[str(layer_idx)] += len(components)
                    threshold_components.extend(components)

            summary["component_counts"][f"{threshold:.2f}"] = {
                "total": len(threshold_components),
                "by_layer": layer_counts,
            }
            if math.isclose(threshold, 0.05, abs_tol=1e-6):
                all_default_components = threshold_components

        if not all_default_components:
            default_threshold = thresholds[0]
            all_default_components = []
            for tile_id in tile_ids:
                alpha_pack = np.asarray(alpha[tile_id], dtype=np.float32)
                map_name = str(_row_get(table, tile_id, "map", "unknown"))
                tile_x = int(_row_get(table, tile_id, "tile_x", -1) or -1)
                tile_y = int(_row_get(table, tile_id, "tile_y", -1) or -1)
                for layer_idx in range(min(4, alpha_pack.shape[-1])):
                    all_default_components.extend(
                        _extract_components(
                            alpha_pack,
                            build=args.build,
                            map_name=map_name,
                            tile_id=tile_id,
                            tile_x=tile_x,
                            tile_y=tile_y,
                            layer_idx=layer_idx,
                            threshold=default_threshold,
                            min_area=args.min_area,
                            reject_edge=args.reject_edge,
                        )
                    )

        all_default_components.sort(key=lambda component: component.area, reverse=True)
        selected_components = all_default_components[: int(args.max_components)]
        if len(selected_components) > int(args.max_components):
            selected_components = list(rng.choice(selected_components, size=int(args.max_components), replace=False))
        _save_patch_examples(selected_components, args.output_dir, args.examples_per_layer)

        component_rows = [_component_summary_row(component) for component in selected_components]
        component_jsonl = args.output_dir / "components_sample.jsonl"
        with component_jsonl.open("w", encoding="utf-8") as handle:
            for row in component_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        summary["outputs"]["components_sample"] = str(component_jsonl)

        patches = np.stack([_render_component_patch(component) for component in selected_components], axis=0) if selected_components else np.zeros((0, 224, 224), dtype=np.float32)
        if patches.shape[0] > 0:
            patch_path = args.output_dir / "patches_sample.npy"
            np.save(patch_path, patches)
            summary["outputs"]["patches_sample"] = str(patch_path)

        if not args.skip_dinov2 and patches.shape[0] > 1:
            cls_embeddings, mean_embeddings = _embed_patches(
                patches,
                model_name=args.model_name,
                device=device,
                batch_size=args.batch_size,
            )
            embeddings_path = args.output_dir / "embeddings_sample.npz"
            np.savez_compressed(
                embeddings_path,
                cls=cls_embeddings,
                mean=mean_embeddings,
                component_id=np.asarray([component.component_id for component in selected_components]),
            )
            summary["outputs"]["embeddings_sample"] = str(embeddings_path)

            cls_points = _pca2(cls_embeddings)
            mean_points = _pca2(mean_embeddings)
            _save_projection(cls_points, selected_components, args.output_dir, "cls")
            _save_projection(mean_points, selected_components, args.output_dir, "mean")
            summary["outputs"]["projection_cls"] = str(args.output_dir / "projection_cls.png")
            summary["outputs"]["projection_mean"] = str(args.output_dir / "projection_mean.png")
            summary["outputs"]["projection"] = str(args.output_dir / "projection.png")
            summary["embedding_preview"] = {
                "cls_pc1_buckets": _cluster_preview(cls_points, selected_components),
                "mean_pc1_buckets": _cluster_preview(mean_points, selected_components),
            }
        elif args.skip_dinov2:
            summary["embedding_preview"] = "skipped_by_flag"
        else:
            summary["embedding_preview"] = "not_enough_components"

        summary_path = args.output_dir / "summary.json"
        _write_json(summary_path, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        store.close()


if __name__ == "__main__":
    main()
