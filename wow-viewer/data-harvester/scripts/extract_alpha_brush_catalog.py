"""Extract and cluster alpha-brush components from V18 Zarr stores."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import zarr

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from harvester.alpha_brush import (  # noqa: E402
    BrushComponent,
    build_catalog_entries,
    build_cluster_catalog,
    cluster_components,
    compute_dinov2_embeddings,
    extract_components,
    load_dinov2_model,
    render_component_patch,
    save_catalog,
    save_clusters,
    save_components,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATASET_DIR = _PROJECT_ROOT / "output" / "datasets" / "v18"
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "output" / "analysis" / "alpha-brush-library"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract the spec 074 alpha-brush catalog.")
    parser.add_argument("--dataset-dir", type=Path, default=_DEFAULT_DATASET_DIR)
    parser.add_argument("--builds", nargs="*", default=None)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--alpha-threshold", type=float, default=0.05)
    parser.add_argument("--min-area", type=int, default=16)
    parser.add_argument("--reject-edge", action="store_true", default=False)
    parser.add_argument("--cluster-algo", default="hdbscan", choices=["hdbscan", "kmeans"])
    parser.add_argument("--min-cluster-size", type=int, default=10)
    parser.add_argument("--fallback-k", type=int, default=100)
    parser.add_argument("--model-name", default="facebook/dinov2-small")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=74)
    parser.add_argument("--token-strategy", default="mean", choices=["mean", "cls"])
    parser.add_argument("--tile-limit", type=int, default=None, help="Optional smoke-run tile cap per build.")
    return parser.parse_args()


def _discover_builds(dataset_dir: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return sorted(requested)
    builds = sorted(path.name.removesuffix(".zarr") for path in dataset_dir.glob("*.zarr") if path.is_dir())
    if not builds:
        raise FileNotFoundError(f"No *.zarr builds found under {dataset_dir}")
    return builds


def _row_get(table, row_idx: int, column: str, default):
    if table is None or column not in table.column_names:
        return default
    value = table.column(column)[row_idx].as_py()
    return default if value is None else value


def _tile_rows(zarr_path: Path, root: zarr.Group, tile_limit: int | None) -> list[dict[str, object]]:
    alpha_count = int(root["alpha_256"].shape[0])
    index_path = zarr_path / "index.parquet"
    table = pq.read_table(str(index_path)) if index_path.exists() else None
    rows: list[dict[str, object]] = []
    max_rows = min(alpha_count, table.num_rows if table is not None else alpha_count)
    for tile_id in range(max_rows):
        if table is not None and "has_alpha_256" in table.column_names and not bool(_row_get(table, tile_id, "has_alpha_256", False)):
            continue
        rows.append(
            {
                "tile_id": tile_id,
                "map": str(_row_get(table, tile_id, "map", "unknown")),
                "tile_x": int(_row_get(table, tile_id, "tile_x", -1) or -1),
                "tile_y": int(_row_get(table, tile_id, "tile_y", -1) or -1),
            }
        )
        if tile_limit is not None and len(rows) >= int(tile_limit):
            break
    return rows


def _extract_build_components(
    dataset_dir: Path,
    build: str,
    threshold: float,
    min_area: int,
    reject_edge: bool,
    tile_limit: int | None,
) -> list[BrushComponent]:
    zarr_path = dataset_dir / f"{build}.zarr"
    if not zarr_path.exists():
        raise FileNotFoundError(f"Missing build store: {zarr_path}")
    store = zarr.storage.LocalStore(str(zarr_path), read_only=True)
    root = zarr.open_group(store=store, mode="r")
    try:
        if "alpha_256" not in root:
            raise KeyError(f"alpha_256 missing in {zarr_path}")
        alpha = root["alpha_256"]
        rows = _tile_rows(zarr_path, root, tile_limit)
        components: list[BrushComponent] = []
        for idx, row in enumerate(rows, start=1):
            tile_id = int(row["tile_id"])
            alpha_pack = np.asarray(alpha[tile_id], dtype=np.float32)
            for layer_idx in range(min(4, alpha_pack.shape[-1])):
                components.extend(
                    extract_components(
                        alpha_pack,
                        layer_idx=layer_idx,
                        threshold=threshold,
                        min_area=min_area,
                        reject_edge=reject_edge,
                        build=build,
                        map_name=str(row["map"]),
                        tile_id=tile_id,
                        tile_x=int(row["tile_x"]),
                        tile_y=int(row["tile_y"]),
                    )
                )
            if idx == 1 or idx % 100 == 0 or idx == len(rows):
                print(f"[{build}] tiles {idx}/{len(rows)} components={len(components)}", flush=True)
        return components
    finally:
        store.close()


def _attach_embeddings(
    components: list[BrushComponent],
    model,
    processor,
    batch_size: int,
    token_strategy: str,
) -> list[BrushComponent]:
    if not components:
        return []
    embedded: list[BrushComponent] = []
    for start in range(0, len(components), int(batch_size)):
        batch_components = components[start : start + int(batch_size)]
        patches = np.stack([render_component_patch(component) for component in batch_components], axis=0)
        embeddings = compute_dinov2_embeddings(
            patches,
            model,
            processor,
            batch_size=batch_size,
            token_strategy=token_strategy,
        )
        embedded.extend(
            replace(component, embedding=embedding)
            for component, embedding in zip(batch_components, embeddings, strict=True)
        )
        if len(embedded) == len(batch_components) or len(embedded) % max(1, int(batch_size) * 10) == 0:
            print(f"Embedded {len(embedded)}/{len(components)} components", flush=True)
    return embedded


def main() -> None:
    args = _parse_args()
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    builds = _discover_builds(args.dataset_dir, args.builds)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    print(f"Builds: {', '.join(builds)}", flush=True)
    print(f"Loading DINOv2 model {args.model_name} on {device}", flush=True)
    model, processor = load_dinov2_model(args.model_name, device=device)

    components: list[BrushComponent] = []
    for build in builds:
        components.extend(
            _extract_build_components(
                args.dataset_dir,
                build,
                threshold=float(args.alpha_threshold),
                min_area=int(args.min_area),
                reject_edge=bool(args.reject_edge),
                tile_limit=args.tile_limit,
            )
        )

    print(f"Embedding {len(components)} components", flush=True)
    components = _attach_embeddings(components, model, processor, int(args.batch_size), args.token_strategy)

    print(f"Clustering {len(components)} components", flush=True)
    clustered = cluster_components(
        components,
        algorithm=args.cluster_algo,
        min_cluster_size=int(args.min_cluster_size),
        fallback_k=int(args.fallback_k),
        random_state=int(args.seed),
    )
    clusters = build_cluster_catalog(clustered)
    entries = build_catalog_entries(clustered)

    save_components(args.output_dir / "components.jsonl", clustered)
    save_clusters(args.output_dir / "clusters.jsonl", clusters)
    save_catalog(args.output_dir / "catalog.jsonl", entries)

    non_singleton = sum(1 for cluster in clusters if cluster.member_count > 1)
    print(f"Wrote {len(clustered)} components, {len(clusters)} clusters, {non_singleton} non-singleton clusters", flush=True)
    print(f"Output: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
