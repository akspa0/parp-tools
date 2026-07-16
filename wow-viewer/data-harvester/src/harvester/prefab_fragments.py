"""Bounded local terrain-art fragment discovery.

This deliberately does *not* segment connected full-map alpha zones.  It samples
chunk-aligned windows from individual ADT pages, records a small local signature,
and only calls repeated signatures a fragment family.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from PIL import Image, ImageDraw, ImageFont

from harvester.fractal_canvas import CanvasTileRecord


@dataclass(frozen=True, slots=True)
class PrefabFragment:
    fragment_id: str
    family_id: str
    build: str
    map_name: str
    tile_id: int
    tile_x: int
    tile_y: int
    layer_slot: int
    support_px: int
    local_x: int
    local_y: int
    alpha_coverage: float
    alpha_edge_density: float
    height_range: float
    score: float


def extract_prefab_fragments(
    root: zarr.Group,
    records: Iterable[CanvasTileRecord],
    *,
    supports: tuple[int, ...] = (32, 64, 128),
    stride: int = 16,
    alpha_threshold: float = 0.05,
    min_alpha_coverage: float = 0.08,
    min_height_range: float = 4.0,
    max_candidates_per_tile: int = 48,
) -> list[PrefabFragment]:
    """Extract bounded candidates from tile-local alpha/height evidence.

    Supports and stride must align to 16 alpha pixels (one terrain chunk). Each
    candidate's bbox is therefore always bounded by ``support_px`` and cannot
    become a whole connected zone.
    """
    _validate_window_config(supports, stride)
    if "alpha_256" not in root:
        raise KeyError("prefab fragment discovery requires alpha_256")

    fragments: list[PrefabFragment] = []
    for record in records:
        alpha = root["alpha_256"][record.tile_id].astype(np.float32)
        if alpha.ndim != 3:
            continue
        height = root["height_257"][record.tile_id].astype(np.float32) if "height_257" in root else None
        height_256 = height[:256, :256] if height is not None else None
        candidates: list[PrefabFragment] = []
        for layer_slot in range(alpha.shape[2]):
            layer = alpha[:, :, layer_slot]
            for support in supports:
                for local_y in range(0, 257 - support, stride):
                    for local_x in range(0, 257 - support, stride):
                        crop = layer[local_y : local_y + support, local_x : local_x + support]
                        coverage = float(np.mean(crop > alpha_threshold))
                        if coverage < min_alpha_coverage:
                            continue
                        edge_density = _edge_density(crop, alpha_threshold)
                        relief = 0.0
                        if height_256 is not None:
                            height_crop = height_256[local_y : local_y + support, local_x : local_x + support]
                            relief = float(np.max(height_crop) - np.min(height_crop))
                        # Flat solid texture pages remain useful texture evidence. Sparse,
                        # featureless windows do not.
                        if edge_density < 0.015 and relief < min_height_range:
                            continue
                        family_id = _canonical_family_id(crop, support, alpha_threshold)
                        score = coverage * (0.5 + edge_density) + min(relief / 128.0, 1.0) * 0.25
                        fragment_id = _fragment_id(record, layer_slot, support, local_x, local_y, family_id)
                        candidates.append(
                            PrefabFragment(
                                fragment_id=fragment_id,
                                family_id=family_id,
                                build=record.build,
                                map_name=record.map_name,
                                tile_id=int(record.tile_id),
                                tile_x=int(record.tile_x),
                                tile_y=int(record.tile_y),
                                layer_slot=int(layer_slot),
                                support_px=int(support),
                                local_x=int(local_x),
                                local_y=int(local_y),
                                alpha_coverage=round(coverage, 6),
                                alpha_edge_density=round(edge_density, 6),
                                height_range=round(relief, 6),
                                score=round(score, 6),
                            )
                        )
        fragments.extend(_select_non_overlapping(candidates, max_candidates_per_tile))
    return fragments


def build_fragment_families(fragments: Iterable[PrefabFragment]) -> list[dict[str, object]]:
    grouped: dict[str, list[PrefabFragment]] = {}
    for fragment in fragments:
        grouped.setdefault(fragment.family_id, []).append(fragment)
    families: list[dict[str, object]] = []
    for family_id, members in grouped.items():
        ordered = sorted(members, key=lambda item: (item.map_name, item.tile_y, item.tile_x, item.fragment_id))
        first = ordered[0]
        families.append(
            {
                "family_id": family_id,
                "member_count": len(ordered),
                "map_count": len({item.map_name for item in ordered}),
                "tile_count": len({item.tile_id for item in ordered}),
                "support_px": first.support_px,
                "example_fragment_id": first.fragment_id,
                "example_tile": [first.tile_x, first.tile_y],
                "example_local_xy": [first.local_x, first.local_y],
                "mean_height_range": round(float(np.mean([item.height_range for item in ordered])), 6),
                "members": [item.fragment_id for item in ordered[:128]],
            }
        )
    return sorted(families, key=lambda row: (-int(row["member_count"]), str(row["family_id"])))


def write_fragment_outputs(
    output_dir: Path,
    root: zarr.Group,
    fragments: list[PrefabFragment],
    *,
    alpha_threshold: float,
    max_families: int = 120,
    members_per_family: int = 8,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    families = build_fragment_families(fragments)
    member_rows = [asdict(fragment) for fragment in fragments]
    pq.write_table(pa.Table.from_pylist(member_rows), output_dir / "fragment_members.parquet")
    pq.write_table(pa.Table.from_pylist(families), output_dir / "fragment_families.parquet")
    (output_dir / "fragment_members.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in member_rows), encoding="utf-8")
    (output_dir / "fragment_families.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in families), encoding="utf-8")
    pages = _write_contact_sheets(output_dir / "contact_sheets", root, fragments, families, alpha_threshold, max_families, members_per_family)
    repeated = [family for family in families if int(family["member_count"]) > 1]
    summary = {
        "fragment_count": len(fragments),
        "family_count": len(families),
        "repeated_family_count": len(repeated),
        "max_candidate_support_px": max((fragment.support_px for fragment in fragments), default=0),
        "outputs": {"members": str(output_dir / "fragment_members.parquet"), "families": str(output_dir / "fragment_families.parquet"), "contact_sheets": [str(page) for page in pages]},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _validate_window_config(supports: tuple[int, ...], stride: int) -> None:
    if not supports or any(support < 16 or support > 256 or support % 16 for support in supports):
        raise ValueError("supports must be non-empty 16-pixel multiples in [16, 256]")
    if stride < 1 or stride % 16:
        raise ValueError("stride must be a positive 16-pixel multiple")


def _edge_density(crop: np.ndarray, threshold: float) -> float:
    binary = crop > threshold
    horizontal = np.count_nonzero(binary[:, 1:] != binary[:, :-1])
    vertical = np.count_nonzero(binary[1:, :] != binary[:-1, :])
    return float(horizontal + vertical) / float(max(1, binary.size * 2))


def _canonical_family_id(crop: np.ndarray, support: int, threshold: float) -> str:
    binary = crop > threshold
    active = np.argwhere(binary)
    if active.size:
        y0, x0 = active.min(axis=0)
        y1, x1 = active.max(axis=0) + 1
        binary = binary[y0:y1, x0:x1]
    normalized = _normalized_thumbnail(binary, size=16)
    variants = [np.rot90(normalized, k) for k in range(4)]
    variants.extend(np.fliplr(item) for item in variants[:])
    digest = min(hashlib.sha256(np.packbits(item.reshape(-1).astype(np.uint8)).tobytes()).hexdigest() for item in variants)
    return f"frag_{int(support)}_" + digest[:20]


def _normalized_thumbnail(binary: np.ndarray, *, size: int) -> np.ndarray:
    if binary.size == 0:
        return np.zeros((size, size), dtype=bool)
    image = Image.fromarray(binary.astype(np.uint8) * 255, mode="L")
    image.thumbnail((size, size), Image.Resampling.NEAREST)
    out = np.zeros((size, size), dtype=bool)
    arr = np.asarray(image, dtype=np.uint8) > 0
    y = (size - arr.shape[0]) // 2
    x = (size - arr.shape[1]) // 2
    out[y : y + arr.shape[0], x : x + arr.shape[1]] = arr
    return out


def _fragment_id(record: CanvasTileRecord, layer_slot: int, support: int, x: int, y: int, family_id: str) -> str:
    payload = f"{record.build}|{record.map_name}|{record.tile_id}|{layer_slot}|{support}|{x}|{y}|{family_id}"
    return "pf_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _select_non_overlapping(candidates: list[PrefabFragment], limit: int) -> list[PrefabFragment]:
    selected: list[PrefabFragment] = []
    for candidate in sorted(candidates, key=lambda item: (-item.score, item.support_px, item.fragment_id)):
        if len(selected) >= max(1, int(limit)):
            break
        if any(_iou(candidate, existing) >= 0.5 and candidate.layer_slot == existing.layer_slot for existing in selected):
            continue
        selected.append(candidate)
    return selected


def _iou(left: PrefabFragment, right: PrefabFragment) -> float:
    lx1, ly1 = left.local_x + left.support_px, left.local_y + left.support_px
    rx1, ry1 = right.local_x + right.support_px, right.local_y + right.support_px
    overlap_w = max(0, min(lx1, rx1) - max(left.local_x, right.local_x))
    overlap_h = max(0, min(ly1, ry1) - max(left.local_y, right.local_y))
    intersection = overlap_w * overlap_h
    union = (left.support_px ** 2) + (right.support_px ** 2) - intersection
    return float(intersection) / float(max(1, union))


def _write_contact_sheets(output_dir: Path, root: zarr.Group, fragments: list[PrefabFragment], families: list[dict[str, object]], threshold: float, max_families: int, members_per_family: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_family: dict[str, list[PrefabFragment]] = {}
    for fragment in fragments:
        by_family.setdefault(fragment.family_id, []).append(fragment)
    selected = [family for family in families if int(family["member_count"]) > 1][:max_families]
    pages: list[Path] = []
    for page_index, start in enumerate(range(0, len(selected), 12), start=1):
        page_families = selected[start : start + 12]
        image = Image.new("RGB", (4 * 256, 3 * 222), (14, 14, 18))
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        for index, family in enumerate(page_families):
            x = (index % 4) * 256
            y = (index // 4) * 222
            members = sorted(by_family[str(family["family_id"])], key=lambda item: (-item.score, item.fragment_id))[:members_per_family]
            thumbs = [_fragment_thumbnail(root, member, threshold) for member in members]
            for thumb_index, thumb in enumerate(thumbs):
                px = x + 8 + (thumb_index % 4) * 56
                py = y + 8 + (thumb_index // 4) * 56
                image.paste(thumb, (px, py))
            draw.rectangle((x, y, x + 255, y + 221), outline=(80, 220, 120), width=2)
            draw.text((x + 8, y + 126), f"{str(family['family_id'])[:18]}", fill=(230, 230, 230), font=font)
            draw.text((x + 8, y + 142), f"members={family['member_count']} support={family['support_px']}", fill=(255, 205, 120), font=font)
            draw.text((x + 8, y + 158), "bounded local windows; not zone components", fill=(170, 170, 175), font=font)
        page = output_dir / f"fragment_families_{page_index:03d}.png"
        image.save(page)
        pages.append(page)
    return pages


def _fragment_thumbnail(root: zarr.Group, fragment: PrefabFragment, threshold: float) -> Image.Image:
    alpha = root["alpha_256"][fragment.tile_id, fragment.local_y : fragment.local_y + fragment.support_px, fragment.local_x : fragment.local_x + fragment.support_px, fragment.layer_slot].astype(np.float32)
    image = Image.fromarray(((alpha > threshold).astype(np.uint8) * 255), mode="L").convert("RGB")
    image.thumbnail((52, 52), Image.Resampling.NEAREST)
    tile = Image.new("RGB", (52, 52), (0, 0, 0))
    tile.paste(image, ((52 - image.width) // 2, (52 - image.height) // 2))
    return tile
