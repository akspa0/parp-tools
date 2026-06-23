# Full-Map Fractal Brush Library

**Status**: Draft replacement architecture for 074/075 terrain brush work.

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

1. Assemble full-map signal canvases with provenance.
2. Segment alpha/fractal regions in full-map coordinates and link 074 components as evidence.
3. Build a trainable Zarr/Parquet library of accepted candidates and rejected review rows.
4. Join tileset texture/variant evidence and likely BLP brush/effect source candidates.
5. Only then define the first model target.

## Training Rule

No new model training should proceed from raw 074 connected components, 075 binary scar masks, alpha-only labels, or minimap-only labels. Future models must consume the curated full-map primitive library and obey the one-signal/one-output rule.
