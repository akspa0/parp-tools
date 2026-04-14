#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageOps

PATCHES_PER_TILE = 256
DEFAULT_THUMB_SIZE = 224
DEFAULT_OCCURRENCES_PER_PREFAB = 1
DEFAULT_MAX_PREFABS = 400
SIZE_BUCKETS: Sequence[Tuple[str, int, int]] = (
    ("Tiny (1-16)", 1, 16),
    ("Small (17-64)", 17, 64),
    ("Medium (65-256)", 65, 256),
    ("Large (257-1024)", 257, 1024),
    ("Huge (1025+)", 1025, 1_000_000_000),
)


@dataclass
class PrefabEntry:
    summary: Dict[str, Any]
    prefab_path: Path
    prefab_payload: Dict[str, Any]
    representative_patch_count: int
    representative_patch_area: int
    representative_dimensions: Tuple[int, int]
    size_bucket: str


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_texture_path(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/").lower()
    return text or "none"


def short_texture_label(texture_path: str, max_length: int = 24) -> str:
    normalized = normalize_texture_path(texture_path)
    if normalized == "none":
        return "none"
    label = Path(normalized).stem or normalized
    if len(label) <= max_length:
        return label
    return f"{label[: max_length - 3]}..."


def safe_slug(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "-" for char in value]
    text = "".join(chars)
    while "--" in text:
        text = text.replace("--", "-")
    return text.strip("-") or "item"


def classify_size_bucket(patch_count: int) -> str:
    for label, minimum, maximum in SIZE_BUCKETS:
        if minimum <= patch_count <= maximum:
            return label
    return SIZE_BUCKETS[-1][0]


def patch_bounds_to_pixels(
    occurrence: Dict[str, Any],
    image_width: int,
    image_height: int,
    padding_patches: float = 3.0,
) -> Tuple[int, int, int, int]:
    min_patch_x = float(occurrence.get("patch_min_x", 0) or 0.0) - padding_patches
    min_patch_y = float(occurrence.get("patch_min_y", 0) or 0.0) - padding_patches
    max_patch_x = float(occurrence.get("patch_max_x", 0) or 0.0) + 1.0 + padding_patches
    max_patch_y = float(occurrence.get("patch_max_y", 0) or 0.0) + 1.0 + padding_patches

    min_patch_x = max(0.0, min_patch_x)
    min_patch_y = max(0.0, min_patch_y)
    max_patch_x = min(float(PATCHES_PER_TILE), max_patch_x)
    max_patch_y = min(float(PATCHES_PER_TILE), max_patch_y)

    left = int(math.floor((min_patch_x / float(PATCHES_PER_TILE)) * float(image_width)))
    top = int(math.floor((min_patch_y / float(PATCHES_PER_TILE)) * float(image_height)))
    right = int(math.ceil((max_patch_x / float(PATCHES_PER_TILE)) * float(image_width)))
    bottom = int(math.ceil((max_patch_y / float(PATCHES_PER_TILE)) * float(image_height)))

    right = max(left + 1, min(right, image_width))
    bottom = max(top + 1, min(bottom, image_height))
    left = max(0, min(left, image_width - 1))
    top = max(0, min(top, image_height - 1))
    return left, top, right, bottom


def build_placeholder_thumbnail(message: str, thumb_size: int) -> Image.Image:
    image = Image.new("RGB", (thumb_size, thumb_size), color=(34, 37, 41))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    wrapped = []
    words = message.split()
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= 22:
            current = candidate
            continue
        wrapped.append(current)
        current = word
    if current:
        wrapped.append(current)
    total_height = len(wrapped) * 12
    y_pos = max(8, (thumb_size - total_height) // 2)
    for line in wrapped:
        draw.text((8, y_pos), line, fill=(230, 230, 230), font=font)
        y_pos += 12
    return image


def load_group_payload(dataset_root: Path, group_file: str, cache: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    cached = cache.get(group_file)
    if cached is not None:
        return cached
    group_path = dataset_root / "brush_imprints" / group_file
    payload = load_json(group_path)
    cache[group_file] = payload
    return payload


def load_tile_payload(dataset_root: Path, tile_name: str, cache: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    cached = cache.get(tile_name)
    if cached is not None:
        return cached
    tile_path = dataset_root / "dataset" / f"{tile_name}.json"
    payload = load_json(tile_path)
    cache[tile_name] = payload
    return payload


def get_chunk_layers_by_idx(tile_payload: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    terrain = tile_payload.get("terrain_data", {})
    return {
        int(entry.get("idx", -1)): entry
        for entry in terrain.get("chunk_layers", [])
        if int(entry.get("idx", -1)) >= 0
    }


def decode_alpha_bits_image(alpha_bits: str) -> Optional[Image.Image]:
    encoded = str(alpha_bits or "").strip()
    if not encoded:
        return None
    try:
        payload = base64.b64decode(encoded, validate=False)
    except Exception:
        return None

    if len(payload) == 4096:
        return Image.frombytes("L", (64, 64), payload)
    if len(payload) == 2048:
        expanded = bytearray(4096)
        for index, value in enumerate(payload):
            expanded[index * 2] = (value & 0x0F) * 17
            expanded[index * 2 + 1] = ((value >> 4) & 0x0F) * 17
        return Image.frombytes("L", (64, 64), bytes(expanded))
    return None


def load_chunk_alpha_patch_image(
    dataset_root: Path,
    layer: Dict[str, Any],
    cache_key: Tuple[str, int, str],
    alpha_cache: Dict[Tuple[str, int, str], Optional[Image.Image]],
) -> Optional[Image.Image]:
    if cache_key in alpha_cache:
        return alpha_cache[cache_key]

    alpha_image: Optional[Image.Image] = None
    alpha_path = str(layer.get("alpha_path") or "").strip()
    if alpha_path:
        candidate_path = dataset_root / Path(alpha_path)
        if candidate_path.exists():
            with Image.open(candidate_path) as loaded:
                alpha_image = loaded.convert("L")

    if alpha_image is None:
        alpha_bits = layer.get("alpha_bits")
        if alpha_bits is not None:
            alpha_image = decode_alpha_bits_image(str(alpha_bits))

    if alpha_image is not None and alpha_image.size != (16, 16):
        alpha_image = alpha_image.resize((16, 16), Image.Resampling.BOX)

    alpha_cache[cache_key] = alpha_image
    return alpha_image


def fit_on_canvas(image: Image.Image, panel_size: int, background: str) -> Image.Image:
    contained = ImageOps.contain(image, (panel_size, panel_size), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (panel_size, panel_size), color=background)
    offset_x = (panel_size - contained.width) // 2
    offset_y = (panel_size - contained.height) // 2
    canvas.paste(contained, (offset_x, offset_y))
    return canvas


def add_panel_label(image: Image.Image, title: str, subtitle: str) -> Image.Image:
    panel = image.convert("RGBA")
    overlay = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    bar_height = 34
    draw.rectangle((0, panel.height - bar_height, panel.width, panel.height), fill=(0, 0, 0, 176))
    draw.text((8, panel.height - bar_height + 4), title[:20], fill=(255, 255, 255, 255), font=font)
    draw.text((8, panel.height - 14), subtitle[:28], fill=(214, 214, 214, 255), font=font)
    return Image.alpha_composite(panel, overlay).convert("RGB")


def load_heightmap_panel(dataset_root: Path, occurrence: Dict[str, Any], panel_size: int) -> Image.Image:
    heightmap_rel = str(occurrence.get("heightmap_global_path") or "").strip()
    if not heightmap_rel:
        return add_panel_label(build_placeholder_thumbnail("missing heightmap", panel_size), "Heightmap", "missing")

    heightmap_path = dataset_root / Path(heightmap_rel)
    if not heightmap_path.exists():
        return add_panel_label(build_placeholder_thumbnail("missing heightmap", panel_size), "Heightmap", "missing")

    with Image.open(heightmap_path) as heightmap_image:
        grayscale = ImageOps.autocontrast(heightmap_image.convert("L"))
        crop_box = patch_bounds_to_pixels(occurrence, grayscale.width, grayscale.height)
        crop = grayscale.crop(crop_box)
        canvas = fit_on_canvas(ImageOps.colorize(crop, black="#0b1020", white="#f2d388"), panel_size, "#0d1117")
    return add_panel_label(canvas, "Heightmap", "global")


def select_representative_layers(occurrence: Dict[str, Any], tile_payload: Dict[str, Any], limit: int = 3) -> List[Dict[str, Any]]:
    chunk_layers_by_idx = get_chunk_layers_by_idx(tile_payload)
    candidates: Dict[str, Dict[str, Any]] = {}
    for chunk_index in occurrence.get("chunk_indices", []):
        chunk = chunk_layers_by_idx.get(int(chunk_index), {})
        for layer_index, layer in enumerate((chunk.get("layers") or [])[1:], start=1):
            texture_path = normalize_texture_path(layer.get("texture_path"))
            if texture_path == "none":
                continue
            candidate = candidates.setdefault(
                texture_path,
                {
                    "texture_path": texture_path,
                    "chunk_count": 0,
                    "min_layer_index": layer_index,
                },
            )
            candidate["chunk_count"] += 1
            candidate["min_layer_index"] = min(int(candidate["min_layer_index"]), layer_index)

    ordered = sorted(
        candidates.values(),
        key=lambda item: (-int(item["chunk_count"]), int(item["min_layer_index"]), str(item["texture_path"])),
    )
    return ordered[:limit]


def build_layer_mask_panel(
    dataset_root: Path,
    occurrence: Dict[str, Any],
    tile_payload: Dict[str, Any],
    texture_path: str,
    panel_size: int,
    alpha_cache: Dict[Tuple[str, int, str], Optional[Image.Image]],
) -> Tuple[Image.Image, str]:
    patch_min_x = int(occurrence.get("patch_min_x", 0) or 0)
    patch_min_y = int(occurrence.get("patch_min_y", 0) or 0)
    patch_width = int(occurrence.get("patch_width", 0) or 0)
    patch_height = int(occurrence.get("patch_height", 0) or 0)
    if patch_width <= 0 or patch_height <= 0:
        placeholder = add_panel_label(build_placeholder_thumbnail("missing layer", panel_size), "Layer", "missing")
        return placeholder, "missing"

    mask = Image.new("L", (patch_width, patch_height), color=0)
    pixels = mask.load()
    chunk_layers_by_idx = get_chunk_layers_by_idx(tile_payload)
    used_true_alpha = False

    for local_y in range(patch_height):
        abs_y = patch_min_y + local_y
        chunk_y = abs_y // 16
        patch_local_y = abs_y % 16
        for local_x in range(patch_width):
            abs_x = patch_min_x + local_x
            chunk_x = abs_x // 16
            patch_local_x = abs_x % 16
            chunk_index = chunk_y * 16 + chunk_x
            chunk = chunk_layers_by_idx.get(chunk_index, {})
            matched_layer = None
            for layer in (chunk.get("layers") or [])[1:]:
                if normalize_texture_path(layer.get("texture_path")) == texture_path:
                    matched_layer = layer
                    break
            if matched_layer is None:
                continue

            alpha_key = (str(occurrence.get("tile_name") or ""), chunk_index, texture_path)
            alpha_image = load_chunk_alpha_patch_image(dataset_root, matched_layer, alpha_key, alpha_cache)
            if alpha_image is not None:
                value = int(alpha_image.getpixel((patch_local_x, patch_local_y)))
                used_true_alpha = True
            else:
                value = 255
            pixels[local_x, local_y] = max(int(pixels[local_x, local_y]), value)

    if mask.getbbox() is None:
        placeholder = add_panel_label(build_placeholder_thumbnail("missing layer", panel_size), "Layer", "missing")
        return placeholder, "missing"

    contained = ImageOps.contain(mask, (panel_size, panel_size), Image.Resampling.NEAREST)
    canvas = Image.new("L", (panel_size, panel_size), color=0)
    offset_x = (panel_size - contained.width) // 2
    offset_y = (panel_size - contained.height) // 2
    canvas.paste(contained, (offset_x, offset_y))
    if used_true_alpha:
        rgb = ImageOps.colorize(canvas, black="#0d1117", white="#7ee787")
        mode = "alpha"
    else:
        rgb = ImageOps.colorize(canvas, black="#0d1117", white="#58a6ff")
        mode = "layout"
    subtitle = short_texture_label(texture_path)
    panel = add_panel_label(rgb, "Layer", subtitle)
    return panel, mode


def compose_occurrence_review_image(panels: Sequence[Image.Image], panel_size: int) -> Image.Image:
    gap = 8
    canvas_width = panel_size * 2 + gap * 3
    canvas_height = panel_size * 2 + gap * 3
    canvas = Image.new("RGB", (canvas_width, canvas_height), color="#f6f8fa")
    positions = [
        (gap, gap),
        (panel_size + gap * 2, gap),
        (gap, panel_size + gap * 2),
        (panel_size + gap * 2, panel_size + gap * 2),
    ]
    for image, (pos_x, pos_y) in zip(panels, positions):
        canvas.paste(image, (pos_x, pos_y))
    return canvas


def build_occurrence_review_image(
    dataset_root: Path,
    occurrence: Dict[str, Any],
    panel_size: int,
    group_cache: Dict[str, Dict[str, Any]],
    tile_cache: Dict[str, Dict[str, Any]],
    alpha_cache: Dict[Tuple[str, int, str], Optional[Image.Image]],
) -> Tuple[Image.Image, Dict[str, Any]]:
    group_file = str(occurrence.get("group_file") or "").strip()
    tile_name = str(occurrence.get("tile_name") or "").strip()
    if group_file:
        load_group_payload(dataset_root, group_file, group_cache)
    tile_payload = load_tile_payload(dataset_root, tile_name, tile_cache)

    panels: List[Image.Image] = [load_heightmap_panel(dataset_root, occurrence, panel_size)]
    layer_descriptions: List[str] = []
    layer_modes: List[str] = []

    selected_layers = select_representative_layers(occurrence, tile_payload, limit=3)
    for layer_index in range(3):
        if layer_index < len(selected_layers):
            selected = selected_layers[layer_index]
            panel, mode = build_layer_mask_panel(
                dataset_root=dataset_root,
                occurrence=occurrence,
                tile_payload=tile_payload,
                texture_path=str(selected["texture_path"]),
                panel_size=panel_size,
                alpha_cache=alpha_cache,
            )
            panels.append(panel)
            layer_descriptions.append(f"L{layer_index + 1}: {selected['texture_path']}")
            layer_modes.append(mode)
        else:
            placeholder = add_panel_label(build_placeholder_thumbnail("no layer", panel_size), "Layer", "none")
            panels.append(placeholder)
            layer_descriptions.append(f"L{layer_index + 1}: none")
            layer_modes.append("missing")

    alpha_source = "true-alpha" if any(mode == "alpha" for mode in layer_modes) else "layout-proxy"
    return compose_occurrence_review_image(panels, panel_size), {
        "layer_descriptions": layer_descriptions,
        "alpha_source": alpha_source,
    }


def write_contact_sheet_image(
    bucket_name: str,
    cards: Sequence[Tuple[str, Path]],
    output_path: Path,
    thumb_size: int,
) -> Optional[Path]:
    if not cards:
        return None

    font = ImageFont.load_default()
    columns = max(1, min(4, math.ceil(math.sqrt(len(cards)))))
    rows = math.ceil(len(cards) / float(columns))
    with Image.open(cards[0][1]) as sample_image:
        image_width, image_height = sample_image.size
    card_width = image_width + 24
    card_height = image_height + 48
    sheet_width = columns * card_width + 24
    sheet_height = rows * card_height + 64
    sheet = Image.new("RGB", (sheet_width, sheet_height), color=(248, 249, 250))
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 16), bucket_name, fill=(33, 37, 41), font=font)

    for index, (label, thumb_path) in enumerate(cards):
        row = index // columns
        column = index % columns
        origin_x = 12 + column * card_width
        origin_y = 40 + row * card_height
        with Image.open(thumb_path) as thumb_image:
            thumb = thumb_image.convert("RGB")
            sheet.paste(thumb, (origin_x, origin_y))
        draw.text((origin_x, origin_y + image_height + 8), label[:28], fill=(33, 37, 41), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path


def select_prefabs(
    report_dir: Path,
    library_payload: Dict[str, Any],
    max_prefabs: Optional[int],
) -> List[PrefabEntry]:
    entries: List[PrefabEntry] = []
    for summary in library_payload.get("prefabs", []):
        prefab_rel = str(summary.get("file") or "").strip()
        if not prefab_rel:
            continue
        prefab_path = report_dir / prefab_rel
        if not prefab_path.exists():
            continue
        prefab_payload = load_json(prefab_path)
        occurrences = prefab_payload.get("occurrences", [])
        if not occurrences:
            continue
        representative = occurrences[0]
        patch_count = int(representative.get("patch_count", 0) or 0)
        patch_width = int(representative.get("patch_width", 0) or 0)
        patch_height = int(representative.get("patch_height", 0) or 0)
        entries.append(
            PrefabEntry(
                summary=summary,
                prefab_path=prefab_path,
                prefab_payload=prefab_payload,
                representative_patch_count=patch_count,
                representative_patch_area=patch_width * patch_height,
                representative_dimensions=(patch_width, patch_height),
                size_bucket=classify_size_bucket(patch_count),
            )
        )

    entries.sort(
        key=lambda entry: (
            list(bucket for bucket, _, _ in SIZE_BUCKETS).index(entry.size_bucket),
            -entry.summary.get("count", 0),
            -entry.representative_patch_count,
            entry.summary.get("prefab_id", ""),
        )
    )
    if max_prefabs is not None and max_prefabs > 0:
        return entries[:max_prefabs]
    return entries


def render_report(
    prefab_library_dir: Path,
    output_dir: Path,
    occurrences_per_prefab: int,
    thumb_size: int,
    max_prefabs: Optional[int],
) -> Dict[str, Any]:
    library_payload = load_json(prefab_library_dir / "prefab_library.json")
    dataset_root = Path(str(library_payload.get("dataset_root") or "")).resolve()
    prefabs = select_prefabs(prefab_library_dir, library_payload, max_prefabs)

    thumbnails_dir = output_dir / "thumbnails"
    sheets_dir = output_dir / "contact_sheets"
    thumbnails_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)
    group_cache: Dict[str, Dict[str, Any]] = {}
    tile_cache: Dict[str, Dict[str, Any]] = {}
    alpha_cache: Dict[Tuple[str, int, str], Optional[Image.Image]] = {}

    bucket_cards: Dict[str, List[Tuple[str, Path]]] = {label: [] for label, _, _ in SIZE_BUCKETS}
    bucket_counts: Dict[str, int] = {label: 0 for label, _, _ in SIZE_BUCKETS}
    bucket_occurrences: Dict[str, int] = {label: 0 for label, _, _ in SIZE_BUCKETS}
    card_sections: List[str] = []

    for entry in prefabs:
        summary = entry.summary
        prefab_payload = entry.prefab_payload
        prefab_id = str(summary.get("prefab_id") or entry.prefab_path.stem)
        occurrences = list(prefab_payload.get("occurrences", []))[:occurrences_per_prefab]
        bucket_counts[entry.size_bucket] += 1
        bucket_occurrences[entry.size_bucket] += int(summary.get("count", 0) or 0)

        thumb_tags: List[str] = []
        representative_layer_summary = "none"
        representative_alpha_source = "layout-proxy"
        for occurrence_index, occurrence in enumerate(occurrences, start=1):
            thumb_name = f"{safe_slug(prefab_id)}-{occurrence_index:02d}.png"
            thumb_path = thumbnails_dir / thumb_name
            thumbnail, preview_meta = build_occurrence_review_image(
                dataset_root=dataset_root,
                occurrence=occurrence,
                panel_size=thumb_size,
                group_cache=group_cache,
                tile_cache=tile_cache,
                alpha_cache=alpha_cache,
            )
            thumbnail.save(thumb_path)
            if occurrence_index == 1:
                representative_layer_summary = " | ".join(preview_meta.get("layer_descriptions", []))
                representative_alpha_source = str(preview_meta.get("alpha_source") or "layout-proxy")
            thumb_tags.append(
                "".join(
                    [
                        '<figure class="thumb">',
                        f'<img src="thumbnails/{html.escape(thumb_name)}" alt="{html.escape(prefab_id)} occurrence {occurrence_index}" />',
                        f'<figcaption>{html.escape(str(occurrence.get("tile_name") or ""))} :: {html.escape(str(occurrence.get("group_id") or ""))}<br/>',
                        f'Layers: {html.escape(" | ".join(preview_meta.get("layer_descriptions", [])))}<br/>',
                        f'Alpha source: {html.escape(str(preview_meta.get("alpha_source") or "layout-proxy"))}</figcaption>',
                        "</figure>",
                    ]
                )
            )
            if occurrence_index == 1:
                bucket_cards[entry.size_bucket].append((prefab_id, thumb_path))

        object_models = list(summary.get("object_models", []))
        object_summary = ", ".join(object_models[:6]) if object_models else "none"
        if len(object_models) > 6:
            object_summary += f", +{len(object_models) - 6} more"

        tiles = list(summary.get("tiles", []))
        tile_summary = ", ".join(tiles[:6])
        if len(tiles) > 6:
            tile_summary += f", +{len(tiles) - 6} more"

        card_sections.append(
            "".join(
                [
                    f'<article class="card" data-bucket="{html.escape(entry.size_bucket)}">',
                    f'<h3>{html.escape(prefab_id)}</h3>',
                    '<div class="meta-grid">',
                    f'<div><strong>Bucket</strong><span>{html.escape(entry.size_bucket)}</span></div>',
                    f'<div><strong>Occurrences</strong><span>{int(summary.get("count", 0) or 0)}</span></div>',
                    f'<div><strong>Patch Count</strong><span>{entry.representative_patch_count}</span></div>',
                    f'<div><strong>Bounds</strong><span>{entry.representative_dimensions[0]} x {entry.representative_dimensions[1]}</span></div>',
                    f'<div><strong>Mean Fractal</strong><span>{summary.get("mean_fractal_detail_score", 0.0)}</span></div>',
                    f'<div><strong>Objects</strong><span>{len(object_models)}</span></div>',
                    f'<div><strong>Alpha Source</strong><span>{html.escape(representative_alpha_source)}</span></div>',
                    "</div>",
                    '<div class="thumb-grid">',
                    "".join(thumb_tags),
                    "</div>",
                    f'<p><strong>Layers:</strong> {html.escape(representative_layer_summary)}</p>',
                    f'<p><strong>Tiles:</strong> {html.escape(tile_summary or "none")}</p>',
                    f'<p><strong>Object Models:</strong> {html.escape(object_summary)}</p>',
                    f'<p><strong>Representative Group:</strong> {html.escape(str(summary.get("representative_group_id") or ""))}</p>',
                    "</article>",
                ]
            )
        )

    section_markup: List[str] = []
    for label, _, _ in SIZE_BUCKETS:
        bucket_cards_for_label = bucket_cards[label]
        sheet_name = f"{safe_slug(label)}.png"
        sheet_path = write_contact_sheet_image(label, bucket_cards_for_label, sheets_dir / sheet_name, thumb_size)
        sheet_markup = ""
        if sheet_path is not None:
            sheet_markup = f'<img class="sheet" src="contact_sheets/{html.escape(sheet_name)}" alt="{html.escape(label)} contact sheet" />'
        section_markup.append(
            "".join(
                [
                    f'<section id="{html.escape(safe_slug(label))}" class="bucket-section">',
                    f'<h2>{html.escape(label)}</h2>',
                    f'<p>{bucket_counts[label]} prefabs, {bucket_occurrences[label]} occurrences</p>',
                    sheet_markup,
                    '<div class="cards">',
                    "".join(card for card in card_sections if f'data-bucket="{html.escape(label)}"' in card),
                    "</div>",
                    "</section>",
                ]
            )
        )

    summary_lines = []
    for label, _, _ in SIZE_BUCKETS:
        summary_lines.append(
            f'<li><a href="#{html.escape(safe_slug(label))}">{html.escape(label)}</a>: {bucket_counts[label]} prefabs / {bucket_occurrences[label]} occurrences</li>'
        )

    html_payload = "".join(
        [
            '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8" />',
            '<meta name="viewport" content="width=device-width, initial-scale=1" />',
            '<title>Prefab Review Report</title>',
            '<style>',
            'body{font-family:Segoe UI,Arial,sans-serif;background:#f5f3ef;color:#1f2328;margin:0;padding:24px;}',
            'main{max-width:1600px;margin:0 auto;}',
            'h1,h2,h3{margin:0 0 12px 0;}',
            '.summary{background:#fff;border:1px solid #d0d7de;border-radius:16px;padding:20px;margin-bottom:24px;}',
            '.summary ul{margin:12px 0 0 18px;padding:0;}',
            '.bucket-section{margin-bottom:40px;}',
            '.sheet{display:block;max-width:100%;border:1px solid #d0d7de;border-radius:12px;background:#fff;margin:12px 0 20px 0;}',
            '.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px;}',
            '.card{background:#fff;border:1px solid #d0d7de;border-radius:16px;padding:16px;box-shadow:0 4px 20px rgba(31,35,40,0.06);}',
            '.meta-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px 12px;margin-bottom:14px;}',
            '.meta-grid div{background:#f6f8fa;border-radius:10px;padding:8px 10px;display:flex;flex-direction:column;gap:4px;}',
            '.meta-grid strong{font-size:12px;text-transform:uppercase;color:#57606a;}',
            '.thumb-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin-bottom:12px;}',
            '.thumb{margin:0;}',
            '.thumb img{width:100%;height:auto;border-radius:10px;display:block;background:#0d1117;}',
            '.thumb figcaption{font-size:12px;line-height:1.35;color:#57606a;margin-top:6px;word-break:break-word;}',
            'p{margin:8px 0 0 0;line-height:1.45;}',
            '@media (max-width:900px){body{padding:12px;}.meta-grid{grid-template-columns:repeat(2,minmax(0,1fr));}}',
            '</style></head><body><main>',
            '<section class="summary">',
            '<h1>Prefab Review Report</h1>',
            f'<p><strong>Source:</strong> {html.escape(str(prefab_library_dir))}</p>',
            f'<p><strong>Dataset Root:</strong> {html.escape(str(dataset_root))}</p>',
            f'<p><strong>Prefabs Shown:</strong> {len(prefabs)} / {int(library_payload.get("unique_prefabs", 0) or 0)}. ',
            f'<strong>Grouping:</strong> {html.escape(str(library_payload.get("grouping_strategy") or "unknown"))} at threshold {html.escape(str(library_payload.get("similarity_threshold") or ""))}.</p>',
            '<p>Each card now shows a representative heightmap crop and the first three terrain layer masks with their tileset labels. When Azeroth has no stored per-layer alpha payload, the layer panels fall back to chunk-layout coverage for that texture instead of inventing MCAL detail.</p>',
            '<ul>',
            "".join(summary_lines),
            '</ul></section>',
            "".join(section_markup),
            '</main></body></html>',
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "index.html", "w", encoding="utf-8") as handle:
        handle.write(html_payload)

    manifest = {
        "source_prefab_library_dir": str(prefab_library_dir),
        "dataset_root": str(dataset_root),
        "prefabs_rendered": len(prefabs),
        "unique_prefabs_available": int(library_payload.get("unique_prefabs", 0) or 0),
        "occurrences_per_prefab": occurrences_per_prefab,
        "thumb_size": thumb_size,
        "layer_source_mode": "true-alpha-when-present-otherwise-layout-proxy",
        "bucket_counts": bucket_counts,
        "bucket_occurrences": bucket_occurrences,
        "index_html": str(output_dir / "index.html"),
    }
    with open(output_dir / "report_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a visual prefab review report from a prefab library output directory.")
    parser.add_argument("prefab_library_dir", help="Directory that contains prefab_library.json and prefabs/.")
    parser.add_argument("--output-dir", help="Where to write the review report. Defaults to <prefab_library_dir>/review_report.")
    parser.add_argument("--occurrences-per-prefab", type=int, default=DEFAULT_OCCURRENCES_PER_PREFAB, help="Number of occurrence crops to show for each prefab card.")
    parser.add_argument("--thumb-size", type=int, default=DEFAULT_THUMB_SIZE, help="Thumbnail edge size in pixels.")
    parser.add_argument("--max-prefabs", type=int, default=DEFAULT_MAX_PREFABS, help="Maximum number of prefabs to render. Use 0 for all.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefab_library_dir = Path(args.prefab_library_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else prefab_library_dir / "review_report"
    max_prefabs = None if int(args.max_prefabs) <= 0 else int(args.max_prefabs)
    manifest = render_report(
        prefab_library_dir=prefab_library_dir,
        output_dir=output_dir,
        occurrences_per_prefab=max(1, int(args.occurrences_per_prefab)),
        thumb_size=max(64, int(args.thumb_size)),
        max_prefabs=max_prefabs,
    )
    print("Prefab review report complete")
    print(f"  prefab_library_dir: {prefab_library_dir}")
    print(f"  output_dir: {output_dir}")
    print(f"  prefabs_rendered: {manifest['prefabs_rendered']}")
    print(f"  index_html: {manifest['index_html']}")


if __name__ == "__main__":
    main()