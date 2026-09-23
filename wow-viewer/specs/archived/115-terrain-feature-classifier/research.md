# Phase 0 Research: Terrain Feature Classification for Geometry Deconfounding

All findings below were established by direct inspection of the real v50.1 store and the real
0.5.3.3368 client on 2026-07-20, not inferred from documentation.

## Decision 1: Label supervision source — per-tile MTEX names via local `mcly_texture_ids`

**Decision**: Derive classifier labels from **`mcly_texture_ids`** (the per-tile local MTEX index,
already in the curriculum store) joined against a **side-car per-tile texture-name dump** produced by
a new, additive C# command. Do **not** use `mcly_tileset_ids`.

**Rationale**: this was the load-bearing unknown and it resolved against expectation.

- The curriculum store `curriculum-0_5_3_3368-dual_v1.zarr` (2990 rows) carries both
  `mcly_texture_ids` and `mcly_tileset_ids`, each `(rows, 16, 16, 4) int32`.
- `mcly_tileset_ids` **is** a global index (row 702 = Kalimdor 32,47 holds `{5, 7, 52, 53, 57, 66,
  102}` against local `mcly_texture_ids` `{0..6}`; store-wide max 193, 195 distinct values).
- **But the global ID→name list is not persisted anywhere in the v50 store.** Its only producer is
  `v22_zarr_io.py:345`, `tileset_path_to_idx = {key(p): i for i, p in enumerate(sorted(self._tilesets,
  key=casefold))}`, where `self._tilesets` comes from the *build-time enrichment stream*. The v50
  store has no `tilesets` group (`list(group.group_keys()) == []`) and no tileset parquet.
- The obvious substitute — `asset_inventory.parquet`'s 182 `texture_rgb` rows, sorted casefold — was
  tested and **falsified**. For curriculum row 50 (Kalimdor tile 24,40) it resolves IDs 0–3 to
  `Aerie Peaks`/`Alterac` (Eastern Kingdoms) textures. Harvesting that exact tile from the real
  client shows its true MTEX table is
  `Darkshore\DarkshoreSandRocks|DarkshoreSand|DarkshoreRockLighter|DarkshoreGrass` — Kalimdor
  textures, as expected. The inventory list is a different, larger set than the enrichment tileset
  list, so their orderings do not agree.

Using the **local** `mcly_texture_ids` sidesteps the lost global list entirely: the local index is
just a position in that tile's own MTEX table, and the harvester already exposes that table by name.

**Alternatives considered**:

- *Rebuild the v50 store to persist the tileset name list*: correct long-term, but requires changing
  the builder and re-running a multi-hour client-backed rebuild of stores that were just validated
  this session. Rejected as disproportionate.
- *Reconstruct the global list from enrichment-stream artifacts*: the stream is transient; no
  on-disk artifact of it survives next to the v50 stores. Rejected as not reliably recoverable.
- *Re-harvest full NPZ shards for all 2990 rows to read names*: works (the NPZ metadata does contain
  `mcly_texture_names`), but writes gigabytes of redundant arrays to read one small string list per
  tile. Rejected in favour of a names-only dump.

## Decision 2: Texture-name dump is a new, additive C# command

**Decision**: add `dump-texture-names` to `WowViewer.Tool.Harvest`, emitting one JSON record per
occupied tile: map, tile_x, tile_y, and the ordered MTEX name table. No existing command changes.

**Rationale**: the harvester already decodes and exposes this exact table — the probe above read
`mcly_texture_names` / `mcly_texture_name_table` straight out of a harvested tile's metadata. A
names-only command is small, fast, produces a few hundred KB instead of gigabytes, and touches no
existing code path. It joins to the curriculum index on `(map, tile_x, tile_y)`, which
`index.parquet` carries on every row.

**Alternatives considered**: reuse `harvest-map-mpq` with a discard step (wasteful, as above);
extend `extract-tilesets` (that command decodes BLP *pixels*, a different and much heavier job).

## Decision 3: Labels are per-pixel at 256×256, not per-chunk at 16×16

**Decision**: derive a `(256, 256)` per-pixel label map per row, by resolving each pixel's dominant
texture layer, not a coarse `(16, 16)` per-chunk label.

**Rationale**: the curriculum store carries `alpha_256` as `(rows, 256, 256, 4) float32` — real
per-pixel blend weights per layer, already aligned to the `minimap_rgb` `(256, 256, 3)` input. Roads
are sub-chunk features: a 16×16 chunk grid is 16 px per cell, far coarser than a road's width, so a
per-chunk label would smear roads across whole cells and teach the classifier a blurrier target than
the data actually supports. Per-pixel labels cost nothing extra — the alpha weights and the chunk's
texture IDs are both already present.

**Dominant-layer rule** (this is the spec's required "blended-layer resolution policy", made
explicit): layer 0 is the opaque base; layers 1–3 composite over it with weight `alpha_256[..., k]`.
A pixel's dominant layer is the highest-index layer whose alpha exceeds `DOMINANT_ALPHA_THRESHOLD`
(default 0.5), else layer 0. This matches how `TerrainMinimapCompositor.BlendLayers` actually
composites (later layers `Lerp` over earlier ones), so the label describes the texture a viewer
actually sees at that pixel.

## Decision 4: Family taxonomy is name-substring based, versioned, unknown-preserving

**Decision**: a versioned lookup maps a texture *path* to one canonical family by case-insensitive
substring match against an ordered rule list; no match ⇒ `unknown`.

**Rationale**: real 0.5.3 tileset paths are strongly self-describing and consistently named
(`Tileset\Darkshore\DarkshoreGrass.blp`, `...\DarkshoreSand.blp`). Road/path textures in this era
carry explicit tokens (`road`, `path`, `trail`, `cobble`, `brick`, `pave`). A small ordered
substring table is auditable, requires no annotation, and is trivially extendable as coverage gaps
appear — matching the spec's assumption that a hand-curated starting list is acceptable.

Ordering matters and is part of the contract: the first matching rule wins, so specific tokens
(`roadside`) must precede generic ones (`road`). The rule list, its revision string, and the
threshold are hashed into the label-pipeline provenance so a label set can never be silently
re-derived under different rules.

## Decision 5: Classifier consumes RGB only; geometry consumes the *generated* map

**Decision**: classifier input is exactly `minimap_rgb` `(3, 256, 256)`. The retrained geometry model
input becomes `(3 + C, 256, 256)` where `C` is the class-probability channel count, sourced only from
a classifier checkpoint's generated output.

**Rationale**: directly enforces the spec's governing principle and FR-001/FR-007. Feeding
ground-truth labels into geometry training would produce a model that cannot run at deployment on
`ek.jpg`-style input at all — the precise failure this feature exists to prevent. Training the
geometry model on *generated* (imperfect) classifier output also matches Spec 114 FR-006's existing
rule that downstream models must see the upstream model's real output distribution, errors included.

## Decision 6: Reuse the existing OOD proof set; do not collect new data

**Decision**: the out-of-distribution gate reuses the `ek.jpg` tiles already produced this session
(`data-harvester/output/ek_tiles_256/`, 700 tiles from `split-minimap-image`) and the existing
`v50_infer_geometry_detailer.py` composition path.

**Rationale**: this is the exact artifact that exposed the roads-as-hills failure, so it is the
correct regression witness. It is already on disk, already 256×256 contract-conformant, and needs no
new collection. Its lack of any ground truth is the point — it proves the chain runs on an arbitrary
image.

## Resolved unknowns

| Unknown | Resolution |
|---|---|
| Does the curriculum store carry texture-family ground truth? | Yes — `mcly_texture_ids`, `mcly_tileset_ids`, `mcly_layer_mask`, `alpha_256`, all populated (2665/2990 rows have non-trivial tileset IDs). |
| Can tileset IDs be resolved to names from the store alone? | **No.** Global list is not persisted; the `asset_inventory` substitute was tested and falsified. Use local `mcly_texture_ids` + a names dump instead (Decision 1/2). |
| Per-chunk or per-pixel labels? | Per-pixel 256×256 (Decision 3) — `alpha_256` supports it and roads are sub-chunk. |
| Is a new contract schema needed for the run record? | No. `model_stage_contract.py` already accepts `stage: "terrain_features"` in `STAGES`; the existing `v50-model-stage-run-v1` document validates unchanged. |
| Does the geometry trainer need architectural change for extra channels? | Yes, but bounded — `direct_geometry_model.py` builds a fixed 3-channel stem; adding an `in_channels` parameter defaulted to 3 keeps every existing RGB-only variant and checkpoint bit-identical. |
