"""Spec 115: derive terrain-feature-family labels from real per-tile MTEX texture names.

Why this exists: the promoted geometry chain maps minimap RGB straight to height, and real
out-of-distribution testing showed it treating a road's distinct color as a slope. Fixing that needs
a model that can recognise "this is an authored flat surface, not geometry" from appearance alone --
which needs training labels. This module produces them, with no manual annotation.

Label source, and why it is not the obvious one: the v50 curriculum store carries
``mcly_tileset_ids`` (a GLOBAL tileset index), but the global index-to-name list is not persisted
anywhere -- it lives only in the transient build-time enrichment stream
(``v22_zarr_io.py`` builds it from ``sorted(self._tilesets, key=casefold)``). The plausible
substitute, ``asset_inventory.parquet``'s ``texture_rgb`` rows sorted the same way, was tested
against the real client and FALSIFIED: it maps curriculum row 50 (Kalimdor tile 24,40) to
``Aerie Peaks``/``Alterac`` textures when that tile's true MTEX table is four ``Darkshore`` textures.

So this module uses ``mcly_texture_ids`` -- the per-tile LOCAL index into that tile's own MTEX table
-- joined against a texture-name dump produced by ``WowViewer.Tool.Harvest dump-texture-names``.
The local index needs no global registry, and the dump is verified to reproduce the real client
table exactly.

Nothing here ever reaches a model's inference input. These labels train the classifier; the
classifier's *generated* output is the only thing downstream geometry may consume (spec FR-001/007).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from harvester.v50.model_stage_contract import sha256_json

# Bump whenever families, ordinals, rules, rule order, or the threshold change. Hashed into every
# derived label set's provenance so a label store can never be silently re-derived under new rules.
TAXONOMY_REVISION = "v115.1"

TILE_PIXELS = 256
CHUNKS_PER_AXIS = 16
PIXELS_PER_CHUNK = TILE_PIXELS // CHUNKS_PER_AXIS  # 16
MAX_LAYERS = 4

# A pixel's overlay layer counts as dominant only above this alpha. Mirrors how
# TerrainMinimapCompositor.BlendLayers actually composites (later layers Lerp over earlier ones),
# so a label describes the texture a viewer actually sees at that pixel.
DOMINANT_ALPHA_THRESHOLD = 0.5

# Ordinal IS the channel index of every predicted feature map. Contract-stable; see data-model.md.
UNKNOWN = 0
TERRAIN = 1
ROAD = 2
WATER = 3
STRUCTURE = 4

FAMILY_NAMES: tuple[str, ...] = ("unknown", "terrain", "road", "water", "structure")
CLASS_COUNT = len(FAMILY_NAMES)

# Ordered rules; FIRST match on the texture's LEAF filename wins. Both properties are load-bearing:
#
# - Leaf-only, because zone directories poison full-path matching: the real path
#   ``Tileset\Swamp of Sorrows\SwampSorrowsStoneRoad07.blp`` would match a "swamp" -> water rule on
#   its directory even though the texture is a road.
# - Order, because real names combine tokens: ``LochModanBrickRoadBase`` is a road, and
#   ``ArathiHighlandsBrickFloor`` is a building floor. "road" therefore precedes "floor", which
#   precedes bare "brick".
#
# Derived from a survey of all 323 distinct texture paths in the real 0.5.3.3368 Kalimdor+Azeroth
# corpus, not from guesswork.
TEXTURE_FAMILY_RULES: tuple[tuple[str, int], ...] = (
    # --- road / authored flat travel surfaces (the confound this feature isolates) ---
    ("road", ROAD),
    ("cobblestone", ROAD),
    ("cobble", ROAD),
    ("trail", ROAD),
    ("pave", ROAD),
    # --- structure: floors/foundations read as buildings, not travel surfaces ---
    ("floor", STRUCTURE),
    ("foundation", STRUCTURE),
    ("wall", STRUCTURE),
    ("brick", STRUCTURE),
    ("pillar", STRUCTURE),
    ("tiles", STRUCTURE),
    ("tile", STRUCTURE),
    # --- water ---
    ("water", WATER),
    ("river", WATER),
    ("lake", WATER),
    ("ocean", WATER),
    ("seafloor", WATER),
    # --- natural terrain ---
    ("grass", TERRAIN),
    ("dirt", TERRAIN),
    ("rock", TERRAIN),
    ("sand", TERRAIN),
    ("snow", TERRAIN),
    ("mud", TERRAIN),
    ("muck", TERRAIN),
    ("gravel", TERRAIN),
    ("moss", TERRAIN),
    ("leaf", TERRAIN),
    ("leaves", TERRAIN),
    ("needles", TERRAIN),
    ("brush", TERRAIN),
    ("bush", TERRAIN),
    ("crop", TERRAIN),
    ("lava", TERRAIN),
    ("ash", TERRAIN),
    ("ice", TERRAIN),
    # Extended from a coverage pass over the real corpus: these were the highest-frequency
    # unmatched leaves, and every one is natural ground cover rather than an authored surface.
    ("fern", TERRAIN),
    ("root", TERRAIN),
    ("ground", TERRAIN),
    ("shore", TERRAIN),
    ("plant", TERRAIN),
    ("flower", TERRAIN),
    ("wood", TERRAIN),
    ("straw", TERRAIN),
    ("weed", TERRAIN),
    ("rubble", TERRAIN),
    ("shale", TERRAIN),
    ("charcoal", TERRAIN),
    ("barnacle", TERRAIN),
    ("creep", TERRAIN),
    ("blight", TERRAIN),
    ("crack", TERRAIN),
    ("earth", TERRAIN),
    ("web", TERRAIN),
    ("footprint", TERRAIN),
    ("corrupt", TERRAIN),
    # Safe only because every authored-surface rule above already fired: "StoneRoad" is caught by
    # "road", "CobbleStone" by "cobble", "StoneWall" by "wall". Bare stone reaching here is ground.
    ("stone", TERRAIN),
)


class TerrainFeatureLabelError(ValueError):
    """Raised when label derivation cannot produce an honest result."""


def rule_set_sha256() -> str:
    """Content hash of the taxonomy + rules + threshold, for label-set provenance."""
    return sha256_json(
        {
            "taxonomy_revision": TAXONOMY_REVISION,
            "families": list(FAMILY_NAMES),
            "rules": [[token, family] for token, family in TEXTURE_FAMILY_RULES],
            "dominant_alpha_threshold": DOMINANT_ALPHA_THRESHOLD,
        }
    )


def texture_leaf(texture_path: str) -> str:
    """Filename without directories or extension; matching is leaf-only by design (see rules)."""
    normalized = str(texture_path).replace("/", "\\")
    leaf = normalized.rsplit("\\", 1)[-1]
    return leaf.rsplit(".", 1)[0]


def classify_texture_name(texture_path: str) -> int:
    """Map one texture path to a family ordinal; no rule match yields UNKNOWN, never a guess."""
    if not texture_path:
        return UNKNOWN
    leaf = texture_leaf(texture_path).lower()
    for token, family in TEXTURE_FAMILY_RULES:
        if token in leaf:
            return family
    return UNKNOWN


def family_lookup_for_tile(texture_names: Iterable[str]) -> np.ndarray:
    """Per-tile array mapping local MTEX index -> family ordinal."""
    return np.asarray([classify_texture_name(name) for name in texture_names], dtype=np.uint8)


def load_texture_name_dump(paths: Iterable[Path]) -> dict[tuple[str, int, int], list[str]]:
    """Load one or more ``dump-texture-names`` JSON files into a (map, tile_x, tile_y) lookup.

    The name list's ORDER is the contract: position equals the value stored in ``mcly_texture_ids``.
    """
    lookup: dict[tuple[str, int, int], list[str]] = {}
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        fmt = payload.get("Format")
        if fmt != "terrain-texture-name-dump-v1":
            raise TerrainFeatureLabelError(
                f"{path}: unexpected dump format {fmt!r}; expected 'terrain-texture-name-dump-v1'"
            )
        map_name = payload.get("Map")
        if not map_name:
            raise TerrainFeatureLabelError(f"{path}: dump has no Map name")
        for tile in payload.get("Tiles", []):
            key = (str(map_name), int(tile["TileX"]), int(tile["TileY"]))
            lookup[key] = [str(name) for name in tile.get("TextureNames", [])]
    if not lookup:
        raise TerrainFeatureLabelError("texture-name dump(s) contained no tiles")
    return lookup


def resolve_dominant_layer(
    alpha_256: np.ndarray | None,
    layer_mask: np.ndarray | None,
    *,
    threshold: float = DOMINANT_ALPHA_THRESHOLD,
) -> np.ndarray:
    """Per-pixel dominant MCAL layer index, (256, 256) uint8.

    Layer 0 is the opaque base; layers 1..3 composite over it. The highest-index layer whose alpha
    clears ``threshold`` (and whose chunk actually declares that layer) wins, matching the
    compositor's paint order. Absent alpha means a base-only tile, which is layer 0 everywhere --
    a real state, not missing data.
    """
    dominant = np.zeros((TILE_PIXELS, TILE_PIXELS), dtype=np.uint8)
    if alpha_256 is None:
        return dominant

    alpha = np.asarray(alpha_256, dtype=np.float32)
    if alpha.shape[:2] != (TILE_PIXELS, TILE_PIXELS) or alpha.shape[2] < MAX_LAYERS:
        raise TerrainFeatureLabelError(
            f"alpha_256 must be ({TILE_PIXELS}, {TILE_PIXELS}, >={MAX_LAYERS}), got {alpha.shape}"
        )

    declared = None
    if layer_mask is not None:
        mask = np.asarray(layer_mask, dtype=np.float32)
        if mask.shape[:2] == (CHUNKS_PER_AXIS, CHUNKS_PER_AXIS) and mask.shape[2] >= MAX_LAYERS:
            declared = np.repeat(
                np.repeat(mask > 0, PIXELS_PER_CHUNK, axis=0), PIXELS_PER_CHUNK, axis=1
            )

    # Ascending order with overwrite ⇒ the highest qualifying layer wins.
    for layer in range(1, MAX_LAYERS):
        selected = alpha[:, :, layer] > threshold
        if declared is not None:
            selected = selected & declared[:, :, layer]
        dominant = np.where(selected, np.uint8(layer), dominant)
    return dominant


def derive_row_labels(
    *,
    texture_ids: np.ndarray,
    texture_names: list[str],
    alpha_256: np.ndarray | None,
    layer_mask: np.ndarray | None,
    threshold: float = DOMINANT_ALPHA_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """One curriculum row -> ``(labels (256,256) uint8, valid (256,256) bool)``.

    ``valid`` is False wherever the source data cannot justify a label (no MTEX table, or a local
    index outside it). Those pixels stay UNKNOWN and are excluded from loss/metrics; they are never
    relabelled into a real class.
    """
    ids = np.asarray(texture_ids)
    if ids.shape[:2] != (CHUNKS_PER_AXIS, CHUNKS_PER_AXIS) or ids.shape[2] < MAX_LAYERS:
        raise TerrainFeatureLabelError(
            f"mcly_texture_ids must be ({CHUNKS_PER_AXIS}, {CHUNKS_PER_AXIS}, >={MAX_LAYERS}), "
            f"got {ids.shape}"
        )

    labels = np.zeros((TILE_PIXELS, TILE_PIXELS), dtype=np.uint8)
    if not texture_names:
        return labels, np.zeros((TILE_PIXELS, TILE_PIXELS), dtype=bool)

    dominant = resolve_dominant_layer(alpha_256, layer_mask, threshold=threshold)

    chunk_y = np.repeat(np.arange(TILE_PIXELS) // PIXELS_PER_CHUNK, TILE_PIXELS).reshape(
        TILE_PIXELS, TILE_PIXELS
    )
    chunk_x = chunk_y.T
    local_index = ids[chunk_y, chunk_x, dominant]

    lookup = family_lookup_for_tile(texture_names)
    valid = (local_index >= 0) & (local_index < len(lookup))
    labels = np.where(valid, lookup[np.clip(local_index, 0, len(lookup) - 1)], np.uint8(UNKNOWN))
    return labels.astype(np.uint8), valid


def summarize_labels(labels: np.ndarray, valid: np.ndarray) -> dict[str, int]:
    """Per-family pixel counts plus the invalid count, for coverage reporting."""
    counts = {
        FAMILY_NAMES[family]: int(np.count_nonzero((labels == family) & valid))
        for family in range(CLASS_COUNT)
    }
    counts["invalid"] = int(np.count_nonzero(~valid))
    return counts
