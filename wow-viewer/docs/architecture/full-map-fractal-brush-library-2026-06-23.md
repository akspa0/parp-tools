# Full-Map Fractal Brush Library

**Status**: Phases 1-3 implemented for bounded compact-map proof and full-map strip processing. Macro paste/scar grouping now supersedes raw-stroke connected components as the active review target. Phase 4 texture/BLP evidence join remains open.

## Decision

The primary terrain-art decomposition path is now full-map, fractal-aware, mesh-coupled, and provenance-preserving. Tile-local connected alpha components are no longer considered authoritative brush labels.

Treat each map's terrain mesh, alpha layers, MCLY texture/layer assignments, and candidate source BLP/effect/decal textures as stacked ZBrush-like documents/layers that together form one cohesive 3D digital painting.

## Why

The 074 contact sheets showed that many extracted candidates are wrong for training:

- Tiny one-off hand-painted details and roads were cataloged as brushes even when they do not repeat.
- Large connected regions sometimes combine many unrelated brush placements into a single chonker.
- Useful 3D brush/paste structures can be fragments of larger fractals.
- Alpha masks can span ADT tile boundaries and can represent multiple virtual canvases inside one ADT tile.
- Terrain reconstruction needs height, normals, MCLY texture context, tileset variant evidence, and possible original BLP brush/decal/effect sources, not just minimap RGB or binary alpha masks.
- Some real brush/decal/effect source shapes may already exist as small transparent BLP assets in client data, especially under FX, environment, weather, particle, decal, `textures\BloodSplats`, or similar categories.

## Superseded Work

- `074-alpha-brush-library`: retained as evidence/candidate extraction only.
- `075-scar-mask-segmentation`: retained as coarse diagnostic baseline only; do not use as the primary training path.
- V18 paste mining outputs: retained as prior art and comparison inputs, but not final brush truth until joined to full-map fractal segmentation and texture/spatial provenance.

## Required Signals

Existing V18 Zarr stores provide the initial substrate:

| Signal | Purpose |
|--------|---------|
| `alpha_256` | Artist alpha/pressure masks per layer. |
| `height_257` | 3D spatial relationship and relief signature. |
| `normal_xyz` / `normal_mask` | Surface direction and valid terrain mask. |
| `mcly_texture_ids` | Texture identity per 16x16 chunk cell/layer. |
| `mcly_layer_mask` | Layer activity per 16x16 chunk cell/layer. |
| `minimap_rgb` | Visual input context after terrain/object/texture contributors are understood. |
| object/liquid/shadow masks | Non-terrain contamination controls. |
| decoded BLP texture/effect candidates | Possible original brush/decal/effect source assets for alpha/fractal motifs. |

## Primitive Definition

A valid terrain-art primitive is not an alpha mask alone. It is a coupled reusable unit that may include:

- source BLP/decal/effect-like brush evidence
- alpha mask imprint
- MCLY texture/layer assignment
- height/normal sculpt response
- repeated placement/provenance across the map
- curation status separating reusable motifs from chonkers and one-off hand-painted details

## Phasing

1. Assemble full-map signal canvases with provenance. Implemented for bounded dense canvases and tile-chunked full-map strips.
2. Segment alpha/fractal regions in full-map coordinates and link 074 components as evidence. Implemented for bounded canvas outputs and full-map strip views.
3. Build a trainable Zarr/Parquet library of accepted candidates and rejected review rows. Implemented with fixed-size sample tensors, stable IDs, splits, rejected metadata, and a smoke loader.
4. Join tileset texture/variant evidence and likely BLP brush/effect source candidates.
5. Only then define the first model target.

## Current Proof

- Phase 1 strict-gate Azeroth smoke: `0_5_3_3368`, 16-tile row window, alpha `(256,4096,4)`, height `(257,4097)`, MCLY `(16,256,4)`.
- Phase 1 full-map strip smoke: `0_5_3_3368` Azeroth (622 tiles), processed in horizontal strips of 8 ADT tiles with 1-tile overlap; full chunked canvas written to `wow-viewer/output/analysis/full-map-fractal-brush-library/full_map_smoke/0_5_3_3368_Azeroth_tilefull/canvas.zarr`.
- Phase 2 full-map strip smoke on one map (Azeroth 0.5.3): 12,906 raw components; exact dedupe produced 12,163 unique patterns and 566 duplicates. Near-duplicate clustering (translation/mirror/rotation-invariant normalized thumbnails, size 16, radius 0) reduced this to 11,976 clusters with 668 duplicate clusters and a max cluster size of 40. This raw-stroke path is diagnostic evidence, not the final paste/scar target.
- Macro paste/scar grouping mode (`--macro-pastes`) groups nearby alpha strokes by streamed max-pooling, coarse-grid dilation, full-resolution bbox reprojection, and original-alpha area filtering. Bounded real-data smoke on `0_5_3_3368` Azeroth tile-limit 16 produced 7 `macro_paste` regions. Small full-map visual proof on `0_7_0_3694` `PVPZone02` produced 4 `macro_paste` regions under `wow-viewer/output/analysis/full-map-fractal-brush-library/macro_visual_composite_pvpzone02_close8_area1024` with `macro_paste_overview.png`, `macro_paste_contact_sheet_001.png`, and `composite_signal_overview.png`.
- Macro visual sweep output exists at `wow-viewer/output/analysis/full-map-fractal-brush-library/macro_sweep_pvpzone02_r8_16_32_area1024_4096/index.html`; close radius 8/16/32 and min-area 1024/4096 still produced only 3-4 regions on `PVPZone02`, proving alpha-only macro grouping is broad/layer-wide on this map.
- Blocky paste/scar grouping mode (`--blocky-pastes`) now emits dense middle-scale child regions inside broad macro parent canvases. Visual proof on `0_7_0_3694` `PVPZone02` with `--block-size 16 --block-min-coverage 0.45 --block-close-radius 0 --block-max-footprint 160` produced 10 `blocky_paste` regions under `wow-viewer/output/analysis/full-map-fractal-brush-library/blocky_visual_pvpzone02_b16_cov045_close0_max160`, removing the large parent remnants while keeping the internal blocky chunks.
- Rectangle-page detection (solid axis-aligned alpha pages with extent >= 0.85) found 72 additional rectangle_page regions on full Azeroth 0.5.3, bringing total regions to 12,978 and near-duplicate clusters to 11,976 with 688 duplicate clusters and a max cluster size of 76.
- Canonical validation runs use `--maps all` to analyze every map present in each selected build index; single-map runs are smoke/proof shortcuts only.
- Phase 2 strict-gate smoke: 961 regions, with 11 accepted candidates, 24 fractal members, 1 composite chonker, 2 one-off details, and 923 too-small rows. The too-small rows are preserved for review but excluded from default atomic samples.
- Phase 3 strict-gate smoke: 35 trainable default atomic samples, 926 rejected/review rows, split counts `train=26`, `val=8`, `test=1`; loader read 32 samples with no rejected labels.
- Phase 3 output: `wow-viewer/output/datasets/fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile16_compact/`.

## Curation Correction

- `composite_chonker` does not mean invalid. It means the region is likely a composite canvas made of smaller placements and should be preserved for composite-specific harvesting.
- Tiny connected components inside or near composite regions should not automatically become accepted atomic samples.
- Default atomic samples currently require at least an `8x8` alpha-pixel footprint, the smallest authoring block size for the data we care about.
- Exact alpha-shape dedupe is too strict; most repeated motifs are near-duplicates (translated, mirrored, or slightly varied), not pixel-identical bitmaps.
- Near-duplicate clustering over raw components is now diagnostic only unless fed macro regions; raw brush dots/strokes are not the primary training target.
- Rectangle-page detection finds solid axis-aligned rectangular alpha regions that are likely authored paste/boundary areas, separately from fractal connected components.
- Macro paste/scar grouping now means parent canvas context. The current candidate-unit review lane is `blocky_paste`: dense 16x16-ish child chunks inside those broader detections.
- Composite hard-region overview is now the required visual cross-check: it overlays the same macro boxes on height-gradient, normal-gradient, alpha-transition, MCLY-transition, object-mask, and liquid-mask signals derived from the V18 Zarr source.

## Training Rule

No new model training should proceed from raw 074 connected components, 075 binary scar masks, alpha-only labels, or minimap-only labels. Future models must consume the curated full-map primitive library and obey the one-signal/one-output rule.
