"""Recover repeated, irregular terrain motifs from real chunk-cell payloads.

The authored unit is a graph of 16x16 alpha cells, not a connected zone and not
a rectangle.  Each node carries its real four-layer alpha plus offset-normalized
17x17 height payload.  Families are exact graph/payload matches under the eight
mirror/rotation transforms and may cross ADT borders.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from PIL import Image, ImageDraw, ImageFont

from harvester.fractal_canvas import CanvasTileRecord

_CHUNK_ALPHA = 16
_TRANSFORMS = tuple((rotation, mirror) for rotation in range(4) for mirror in (False, True))


@dataclass(frozen=True, slots=True)
class ChunkCell:
    global_x: int
    global_y: int
    build: str
    map_name: str
    tile_id: int
    tile_x: int
    tile_y: int
    chunk_x: int
    chunk_y: int
    alpha_mean: float
    alpha_variation: float
    height_relief: float
    signatures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ChunkMotif:
    motif_id: str
    family_id: str
    build: str
    map_name: str
    anchor_global_x: int
    anchor_global_y: int
    cell_count: int
    bbox_chunk_xywh: tuple[int, int, int, int]
    cell_offsets: tuple[tuple[int, int], ...]
    source_tiles: tuple[tuple[int, int], ...]
    crosses_tile_border: bool
    canonical_transform: str
    alpha_mean: float
    alpha_variation: float
    height_relief: float


def build_chunk_cells(
    root: zarr.Group,
    records: list[CanvasTileRecord],
    *,
    min_alpha_variation: float = 0.025,
    min_height_relief: float = 2.0,
) -> dict[tuple[int, int], ChunkCell]:
    """Read real cells. Uniform bucket-fill cells are not motif graph nodes."""
    if "alpha_256" not in root or "height_257" not in root:
        raise KeyError("chunk motif extraction requires alpha_256 and height_257")
    cells: dict[tuple[int, int], ChunkCell] = {}
    for record in records:
        alpha = root["alpha_256"][record.tile_id].astype(np.float32)
        height = root["height_257"][record.tile_id].astype(np.float32)
        for chunk_y in range(16):
            for chunk_x in range(16):
                y0, x0 = chunk_y * _CHUNK_ALPHA, chunk_x * _CHUNK_ALPHA
                alpha_patch = alpha[y0 : y0 + _CHUNK_ALPHA, x0 : x0 + _CHUNK_ALPHA, :]
                height_patch = height[y0 : y0 + 17, x0 : x0 + 17]
                alpha_mean = float(alpha_patch.mean())
                alpha_variation = float(alpha_patch.std())
                height_relief = float(height_patch.max() - height_patch.min())
                if alpha_variation < min_alpha_variation and height_relief < min_height_relief:
                    continue
                global_x = int(record.tile_x) * 16 + chunk_x
                global_y = int(record.tile_y) * 16 + chunk_y
                cells[(global_x, global_y)] = ChunkCell(
                    global_x=global_x,
                    global_y=global_y,
                    build=record.build,
                    map_name=record.map_name,
                    tile_id=int(record.tile_id),
                    tile_x=int(record.tile_x),
                    tile_y=int(record.tile_y),
                    chunk_x=chunk_x,
                    chunk_y=chunk_y,
                    alpha_mean=round(alpha_mean, 6),
                    alpha_variation=round(alpha_variation, 6),
                    height_relief=round(height_relief, 6),
                    signatures=tuple(_cell_signature(alpha_patch, height_patch, transform) for transform in _TRANSFORMS),
                )
    return cells


def extract_chunk_motifs(
    cells: dict[tuple[int, int], ChunkCell],
    *,
    max_hops: int = 3,
    max_cells: int = 32,
    min_cells: int = 2,
) -> list[ChunkMotif]:
    """Grow local connected cell graphs and retain irregular topology only."""
    motifs: list[ChunkMotif] = []
    seen_placements: set[tuple[tuple[int, int], ...]] = set()
    for anchor in sorted(cells):
        nodes = _grow_graph(cells, anchor, max_hops=max_hops, max_cells=max_cells)
        if len(nodes) < min_cells:
            continue
        placement = tuple(sorted(nodes))
        if placement in seen_placements:
            continue
        seen_placements.add(placement)
        offsets = _normalize_offsets(nodes)
        # A fully filled rectangular graph is bucket/page evidence, not an irregular motif.
        if len(offsets) == _bbox_area(offsets):
            continue
        family_id, canonical_transform = _family_id(cells, nodes)
        member_cells = [cells[node] for node in nodes]
        source_tiles = tuple(sorted({(cell.tile_x, cell.tile_y) for cell in member_cells}))
        bbox = _bbox(offsets)
        motif_id = _motif_id(member_cells[0].build, member_cells[0].map_name, anchor, family_id)
        motifs.append(
            ChunkMotif(
                motif_id=motif_id,
                family_id=family_id,
                build=member_cells[0].build,
                map_name=member_cells[0].map_name,
                anchor_global_x=anchor[0],
                anchor_global_y=anchor[1],
                cell_count=len(member_cells),
                bbox_chunk_xywh=bbox,
                cell_offsets=offsets,
                source_tiles=source_tiles,
                crosses_tile_border=len(source_tiles) > 1,
                canonical_transform=canonical_transform,
                alpha_mean=round(float(np.mean([cell.alpha_mean for cell in member_cells])), 6),
                alpha_variation=round(float(np.mean([cell.alpha_variation for cell in member_cells])), 6),
                height_relief=round(float(np.mean([cell.height_relief for cell in member_cells])), 6),
            )
        )
    return motifs


def write_motif_outputs(output_dir: Path, motifs: list[ChunkMotif], *, min_occurrences: int = 2) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[ChunkMotif]] = {}
    for motif in motifs:
        grouped.setdefault(motif.family_id, []).append(motif)
    kept = {family_id: members for family_id, members in grouped.items() if len(members) >= min_occurrences}
    member_rows = [asdict(motif) for members in kept.values() for motif in members]
    family_rows = []
    for family_id, members in kept.items():
        ordered = sorted(members, key=lambda item: (item.map_name, item.anchor_global_y, item.anchor_global_x))
        first = ordered[0]
        family_rows.append({
            "family_id": family_id,
            "member_count": len(ordered),
            "map_count": len({item.map_name for item in ordered}),
            "cross_tile_member_count": sum(1 for item in ordered if item.crosses_tile_border),
            "cell_count": first.cell_count,
            "cell_offsets": list(first.cell_offsets),
            "example_motif_id": first.motif_id,
            "example_anchor_global_xy": [first.anchor_global_x, first.anchor_global_y],
            "members": [item.motif_id for item in ordered[:128]],
        })
    family_rows.sort(key=lambda row: (-int(row["member_count"]), -int(row["cell_count"]), str(row["family_id"])))
    pq.write_table(pa.Table.from_pylist(member_rows), output_dir / "motif_members.parquet")
    pq.write_table(pa.Table.from_pylist(family_rows), output_dir / "motif_families.parquet")
    _write_jsonl(output_dir / "motif_members.jsonl", member_rows)
    _write_jsonl(output_dir / "motif_families.jsonl", family_rows)
    pages = _write_contact_sheets(output_dir / "contact_sheets", kept, family_rows)
    summary = {
        "candidate_count": len(motifs),
        "repeated_family_count": len(family_rows),
        "repeated_member_count": len(member_rows),
        "outputs": {"families": str(output_dir / "motif_families.parquet"), "members": str(output_dir / "motif_members.parquet"), "contact_sheets": [str(page) for page in pages]},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _grow_graph(cells: dict[tuple[int, int], ChunkCell], anchor: tuple[int, int], *, max_hops: int, max_cells: int) -> set[tuple[int, int]]:
    found = {anchor}
    queue = deque([(anchor, 0)])
    while queue and len(found) < max_cells:
        node, hops = queue.popleft()
        if hops >= max_hops:
            continue
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = (node[0] + dx, node[1] + dy)
            if neighbor not in cells or neighbor in found:
                continue
            found.add(neighbor)
            queue.append((neighbor, hops + 1))
            if len(found) >= max_cells:
                break
    return found


def _family_id(cells: dict[tuple[int, int], ChunkCell], nodes: set[tuple[int, int]]) -> tuple[str, str]:
    candidates: list[tuple[bytes, str]] = []
    for transform_index, (rotation, mirror) in enumerate(_TRANSFORMS):
        transformed = []
        for x, y in nodes:
            tx, ty = _transform_xy(x, y, rotation, mirror)
            transformed.append((tx, ty, cells[(x, y)].signatures[transform_index]))
        min_x = min(item[0] for item in transformed)
        min_y = min(item[1] for item in transformed)
        payload = "|".join(f"{x - min_x},{y - min_y}:{signature}" for x, y, signature in sorted(transformed))
        candidates.append((payload.encode("ascii"), f"rotate_{rotation * 90}_mirror_{str(mirror).lower()}"))
    payload, transform = min(candidates, key=lambda item: item[0])
    return "mot_" + hashlib.sha256(payload).hexdigest()[:20], transform


def _cell_signature(alpha: np.ndarray, height: np.ndarray, transform: tuple[int, bool]) -> str:
    rotation, mirror = transform
    alpha_t = np.rot90(alpha, rotation, axes=(0, 1))
    height_t = np.rot90(height, rotation)
    if mirror:
        alpha_t = np.fliplr(alpha_t)
        height_t = np.fliplr(height_t)
    alpha_u8 = np.clip(np.rint(alpha_t * 255.0), 0, 255).astype(np.uint8)
    relative = height_t - float(height_t.mean())
    height_i16 = np.clip(np.rint(relative * 4.0), -32768, 32767).astype(np.int16)
    return hashlib.sha256(alpha_u8.tobytes() + height_i16.tobytes()).hexdigest()[:20]


def _transform_xy(x: int, y: int, rotation: int, mirror: bool) -> tuple[int, int]:
    for _ in range(rotation % 4):
        x, y = -y, x
    return (-x, y) if mirror else (x, y)


def _normalize_offsets(nodes: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_x = min(x for x, _ in nodes)
    min_y = min(y for _, y in nodes)
    return tuple(sorted((x - min_x, y - min_y) for x, y in nodes))


def _bbox(offsets: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    max_x = max(x for x, _ in offsets)
    max_y = max(y for _, y in offsets)
    return 0, 0, max_x + 1, max_y + 1


def _bbox_area(offsets: tuple[tuple[int, int], ...]) -> int:
    _x, _y, width, height = _bbox(offsets)
    return width * height


def _motif_id(build: str, map_name: str, anchor: tuple[int, int], family_id: str) -> str:
    return "mi_" + hashlib.sha256(f"{build}|{map_name}|{anchor}|{family_id}".encode("ascii")).hexdigest()[:16]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_contact_sheets(output_dir: Path, grouped: dict[str, list[ChunkMotif]], families: list[dict[str, object]]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pages: list[Path] = []
    font = ImageFont.load_default()
    for page_index, start in enumerate(range(0, len(families), 12), start=1):
        page = Image.new("RGB", (1024, 666), (14, 14, 18))
        draw = ImageDraw.Draw(page)
        for index, family in enumerate(families[start : start + 12]):
            x0, y0 = (index % 4) * 256, (index // 4) * 222
            members = grouped[str(family["family_id"])]
            for member_index, member in enumerate(members[:8]):
                motif = _render_motif(member)
                page.paste(motif, (x0 + 8 + (member_index % 4) * 56, y0 + 8 + (member_index // 4) * 56))
            draw.rectangle((x0, y0, x0 + 255, y0 + 221), outline=(90, 220, 120), width=2)
            draw.text((x0 + 8, y0 + 126), str(family["family_id"])[:18], fill=(235, 235, 235), font=font)
            draw.text((x0 + 8, y0 + 142), f"members={family['member_count']} cells={family['cell_count']}", fill=(255, 205, 120), font=font)
            draw.text((x0 + 8, y0 + 158), "green cells = irregular motif mask", fill=(175, 175, 180), font=font)
        path = output_dir / f"chunk_motif_families_{page_index:03d}.png"
        page.save(path)
        pages.append(path)
    return pages


def _render_motif(motif: ChunkMotif) -> Image.Image:
    _x, _y, width, height = motif.bbox_chunk_xywh
    scale = max(1, min(48 // max(1, width), 48 // max(1, height)))
    image = Image.new("RGB", (52, 52), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    for x, y in motif.cell_offsets:
        draw.rectangle((2 + x * scale, 2 + y * scale, 1 + (x + 1) * scale, 1 + (y + 1) * scale), fill=(80, 225, 130))
    return image
