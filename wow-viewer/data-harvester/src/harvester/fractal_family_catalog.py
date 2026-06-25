"""Cross-map brush-family catalog builder for Spec 076 near-duplicate clusters.

A "family" is either a near-duplicate cluster of raw alpha components or a
rectangle-page region. The catalog materializes a representative crop per family
and preserves cross-map/build/layer provenance so a human can review candidate
brush families before any model target is selected.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr


@dataclass(frozen=True, slots=True)
class BrushFamily:
    family_id: str
    family_type: str
    representative_region_id: str
    build: str
    map_name: str
    layer_idx: int
    layer_slot: int
    bbox_xywh: tuple[int, int, int, int]
    area: int
    member_count: int
    build_count: int
    map_count: int
    layer_count: int
    builds: list[str]
    maps: list[str]
    layer_indices: list[int]
    mcly_texture_ids: list[int]
    mcly_active_layers: list[int]
    tensor_index: int


def load_near_clusters(analysis_root: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    near_dir = Path(analysis_root) / "dedupe" / "near"
    if not near_dir.exists():
        raise FileNotFoundError(f"Near-dedupe output not found: {near_dir}")
    patterns = pq.read_table(near_dir / "near_patterns.parquet").to_pylist()
    members = pq.read_table(near_dir / "near_pattern_members.parquet").to_pylist()
    return patterns, members


def filter_families(
    patterns: list[dict[str, Any]],
    *,
    min_members: int = 2,
    min_builds: int = 1,
    min_maps: int = 1,
    max_families: int | None = None,
) -> list[dict[str, Any]]:
    """Select near-duplicate clusters that meet review thresholds."""
    selected = [
        row
        for row in patterns
        if int(row.get("member_count", 0)) >= int(min_members)
        and int(row.get("build_count", 0)) >= int(min_builds)
        and int(row.get("map_count", 0)) >= int(min_maps)
    ]
    selected.sort(key=lambda row: (-int(row.get("member_count", 0)), -int(row.get("area", 0)), str(row.get("cluster_id", ""))))
    if max_families is not None:
        selected = selected[: max(0, int(max_families))]
    return selected


def group_members_by_cluster(members: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for member in sorted(members, key=lambda item: (str(item.get("build", "")), str(item.get("map_name", "")), int(item.get("area", 0)), str(item.get("region_id", "")))):
        cluster_id = str(member.get("cluster_id", ""))
        if not cluster_id:
            continue
        out.setdefault(cluster_id, []).append(member)
    return out


class CanvasCache:
    """Lazy cache of per-build/map canvas groups."""

    def __init__(self, target_index: dict[tuple[str, str], Path]) -> None:
        self.target_index = target_index
        self._roots: dict[tuple[str, str], zarr.Group] = {}

    def canvas(self, build: str, map_name: str) -> zarr.Group:
        key = (str(build), str(map_name))
        root = self._roots.get(key)
        if root is None:
            canvas_dir = self.target_index.get(key)
            if canvas_dir is None:
                raise FileNotFoundError(f"No canvas target for build={build} map={map_name}")
            root = zarr.open_group(str(canvas_dir / "canvas.zarr"), mode="r")
            self._roots[key] = root
        return root

    def close(self) -> None:
        self._roots.clear()


def discover_canvas_dirs(analysis_root: str | Path) -> dict[tuple[str, str], Path]:
    """Discover <build>_<map>_tile<*>/canvas directories under an analysis root."""
    out: dict[tuple[str, str], Path] = {}
    for target_dir in sorted(Path(analysis_root).glob("*_tile*")):
        if not target_dir.is_dir():
            continue
        parts = target_dir.name.split("_")
        tile_marker = -1
        for idx, part in enumerate(parts):
            if part.startswith("tile"):
                tile_marker = idx
                break
        if tile_marker <= 1:
            continue
        build = "_".join(parts[: tile_marker - 1])
        map_name = parts[tile_marker - 1]
        canvas_dir = target_dir / "canvas"
        if canvas_dir.exists():
            out[(build, map_name)] = canvas_dir
    return out


def extract_alpha_crop(
    canvas: zarr.Group,
    bbox_xywh: tuple[int, int, int, int],
    layer_slot: int,
) -> np.ndarray:
    x, y, w, h = bbox_xywh
    alpha = canvas["alpha_256"][y : y + h, x : x + w, layer_slot].astype(np.float32)
    if alpha.size == 0:
        return np.zeros((1, 1), dtype=np.float32)
    return alpha


def pad_crop(crop: np.ndarray, target_size: int) -> np.ndarray:
    """Center-pad or clip a crop to a square target size."""
    h, w = crop.shape[:2]
    out = np.zeros((target_size, target_size), dtype=np.float32)
    if h > target_size or w > target_size:
        scale = target_size / max(h, w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        from PIL import Image

        resized = Image.fromarray((np.clip(crop, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L").resize((new_w, new_h), Image.Resampling.BILINEAR)
        scaled = np.asarray(resized, dtype=np.float32) / 255.0
        off_y = (target_size - new_h) // 2
        off_x = (target_size - new_w) // 2
        out[off_y : off_y + new_h, off_x : off_x + new_w] = scaled
        return out
    off_y = (target_size - h) // 2
    off_x = (target_size - w) // 2
    out[off_y : off_y + h, off_x : off_x + w] = crop
    return out


def build_families(
    patterns: list[dict[str, Any]],
    members_by_cluster: dict[str, list[dict[str, Any]]],
    cache: CanvasCache,
    *,
    crop_size: int = 128,
) -> tuple[list[BrushFamily], np.ndarray]:
    """Materialize representative crops and family metadata."""
    crop_size = int(crop_size)
    if crop_size <= 0:
        raise ValueError("crop_size must be positive")

    families: list[BrushFamily] = []
    crops: list[np.ndarray] = []

    for tensor_index, cluster in enumerate(patterns):
        cluster_id = str(cluster.get("cluster_id", ""))
        members = members_by_cluster.get(cluster_id, [])
        if not members:
            continue
        rep = members[0]
        build = str(rep.get("build", ""))
        map_name = str(rep.get("map_name", ""))
        layer_slot = int(rep.get("layer_slot", 0))
        layer_idx = int(rep.get("layer_idx", layer_slot))
        bbox_xywh = tuple(int(v) for v in rep.get("bbox_xywh", [0, 0, 1, 1]))

        try:
            canvas = cache.canvas(build, map_name)
            crop = extract_alpha_crop(canvas, bbox_xywh, layer_slot)
            crop = pad_crop(crop, crop_size)
        except Exception:
            crop = np.zeros((crop_size, crop_size), dtype=np.float32)

        families.append(
            BrushFamily(
                family_id=cluster_id,
                family_type="near_cluster",
                representative_region_id=str(rep.get("region_id", "")),
                build=build,
                map_name=map_name,
                layer_idx=layer_idx,
                layer_slot=layer_slot,
                bbox_xywh=bbox_xywh,
                area=int(rep.get("area", 0)),
                member_count=int(cluster.get("member_count", 0)),
                build_count=int(cluster.get("build_count", 0)),
                map_count=int(cluster.get("map_count", 0)),
                layer_count=int(cluster.get("layer_count", 0)),
                builds=sorted(str(b) for b in cluster.get("builds", [])),
                maps=sorted(str(m) for m in cluster.get("maps", [])),
                layer_indices=sorted(int(layer) for layer in cluster.get("layer_indices", [])),
                mcly_texture_ids=sorted(int(t) for t in cluster.get("mcly_texture_ids", [])),
                mcly_active_layers=sorted(int(layer) for layer in cluster.get("mcly_active_layers", [])),
                tensor_index=tensor_index,
            )
        )
        crops.append(crop)

    tensor = np.stack(crops, axis=0) if crops else np.zeros((0, crop_size, crop_size), dtype=np.float32)
    return families, tensor


def write_family_outputs(
    output_dir: str | Path,
    families: list[BrushFamily],
    tensor: np.ndarray,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = [_json_ready(asdict(family)) for family in families]
    _write_table(out / "families.parquet", rows)
    _write_jsonl(out / "families.jsonl", rows)

    if tensor.shape[0] > 0:
        group = zarr.open_group(str(out / "families.zarr"), mode="w")
        group.create_array("alpha_crop", data=tensor)
        encoded_ids = [f.family_id.encode("utf-8") for f in families]
        max_len = max((len(item) for item in encoded_ids), default=1)
        family_id_bytes = np.zeros((len(encoded_ids), max_len), dtype=np.uint8)
        family_id_lengths = np.zeros((len(encoded_ids),), dtype=np.int16)
        for idx, encoded in enumerate(encoded_ids):
            family_id_bytes[idx, : len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
            family_id_lengths[idx] = len(encoded)
        group.create_array("family_id_utf8", data=family_id_bytes)
        group.create_array("family_id_lengths", data=family_id_lengths)

    summary = {
        "family_count": int(len(families)),
        "tensor_shape": list(tensor.shape),
        "outputs": {
            "families_parquet": str(out / "families.parquet"),
            "families_jsonl": str(out / "families.jsonl"),
            "families_zarr": str(out / "families.zarr"),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def render_family_contact_sheet(
    families: list[BrushFamily],
    tensor: np.ndarray,
    output_path: str | Path,
    *,
    families_per_page: int = 20,
) -> list[Path]:
    from PIL import Image, ImageDraw, ImageFont

    out = Path(output_path).parent
    out.mkdir(parents=True, exist_ok=True)
    pages: list[Path] = []
    cell_size = 128
    padding = 10
    label_width = 220
    legend_h = 70
    width = label_width + (cell_size + padding) * 1 + padding
    row_h = cell_size + padding

    font = ImageFont.load_default()
    for page_idx, start in enumerate(range(0, len(families), families_per_page), start=1):
        page_families = families[start : start + families_per_page]
        height = legend_h + len(page_families) * row_h + padding
        image = Image.new("RGB", (width, height), color=(10, 10, 12))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, width - 1, legend_h - 1), fill=(20, 20, 24), outline=(58, 58, 64))
        draw.text((10, 10), "Spec 076 Brush Family Catalog", fill=(245, 245, 245), font=font)
        draw.text((10, 30), "Each row is one near-duplicate family representative crop.", fill=(190, 190, 195), font=font)

        for row_idx, family in enumerate(page_families):
            y = legend_h + padding + row_idx * row_h
            x = padding
            label = "\n".join(
                [
                    family.family_id,
                    f"members {family.member_count} builds {family.build_count}",
                    f"maps {family.map_count} layers {family.layer_indices}",
                    f"box {family.bbox_xywh[2]}x{family.bbox_xywh[3]} area {family.area}",
                ]
            )
            draw.text((x, y + 8), label, fill=(230, 230, 230), font=font)

            crop = tensor[start + row_idx]
            gray = Image.fromarray((np.clip(crop, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
            cell_x = label_width
            image.paste(Image.merge("RGB", (gray, gray, gray)), (cell_x, y))
            draw.rectangle((cell_x, y, cell_x + cell_size - 1, y + cell_size - 1), outline=(255, 255, 255), width=1)

        page_path = out / f"family_catalog_page_{page_idx:03d}.png"
        image.save(page_path)
        pages.append(page_path)

    return pages


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    table = pa.Table.from_pylist(rows) if rows else pa.Table.from_pylist([])
    pq.write_table(table, path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


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
