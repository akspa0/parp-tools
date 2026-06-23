# Full-Map Fractal Brush Library

**Status**: Phases 1-3 implemented for bounded compact-map proof and full-map strip processing; Phase 4 texture/BLP evidence join remains open. Near-duplicate clustering is the next unproven gap.

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
- Phase 2 full-map strip smoke: 12,906 raw components on full Azeroth 0.5.3; exact dedupe produced 12,163 unique patterns and 566 duplicates. Near-duplicate clustering (translation/mirror/rotation-invariant normalized thumbnails, size 16, radius 0) reduced this to 11,976 clusters with 668 duplicate clusters and a max cluster size of 40. Exact matching is too brittle; near-duplicate clustering is required before brush family definition.
- Phase 2 strict-gate smoke: 961 regions, with 11 accepted candidates, 24 fractal members, 1 composite chonker, 2 one-off details, and 923 too-small rows. The too-small rows are preserved for review but excluded from default atomic samples.
- Phase 3 strict-gate smoke: 35 trainable default atomic samples, 926 rejected/review rows, split counts `train=26`, `val=8`, `test=1`; loader read 32 samples with no rejected labels.
- Phase 3 output: `wow-viewer/output/datasets/fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile16_compact/`.

## Curation Correction

- `composite_chonker` does not mean invalid. It means the region is likely a composite canvas made of smaller placements and should be preserved for composite-specific harvesting.
- Tiny connected components inside or near composite regions should not automatically become accepted atomic samples.
- Default atomic samples currently require at least an `8x8` alpha-pixel footprint, the smallest authoring block size for the data we care about.
- Exact alpha-shape dedupe is too strict; most repeated motifs are near-duplicates (translated, mirrored, or slightly varied), not pixel-identical bitmaps.
- Near-duplicate clustering uses translation/mirror/rotation-invariant normalized binary thumbnails to group raw components into candidate brush families.

## Training Rule

No new model training should proceed from raw 074 connected components, 075 binary scar masks, alpha-only labels, or minimap-only labels. Future models must consume the curated full-map primitive library and obey the one-signal/one-output rule.
