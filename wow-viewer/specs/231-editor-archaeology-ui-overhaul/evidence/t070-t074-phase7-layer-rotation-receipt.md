# Receipt — Phase 7: whole-layer rotation/mirror wired end-to-end (T070–T073)

**Date**: 2026-09-07 · **Spec**: [231](../spec.md) Phase 7 · Operator directive: "extend the
tile/map rotation to the layers"

## What was wired

| # | Consumer | Change |
|---|---|---|
| T070 | [ViewerApp_PhaseLayers.cs](../../../src/viewer/WoWViewer/ViewerApp_PhaseLayers.cs) | Layers panel gains: **Layer rotation** combo (None / 90° CW / 180° / 90° CCW), **Mirror left-right** + **Mirror up-down** checkboxes, rotation-origin display with **Center origin on layer** (donor-footprint centroid) and **Origin at 0,0** |
| T071 | [AlphaTerrainAdapter.cs](../../../src/viewer/WoWViewer/Terrain/AlphaTerrainAdapter.cs) | Donor-tile resolution now goes through `PhaseComposition.ResolveTileSource` (per-tile mappings + whole-layer offset + rotation/mirror inverse lookup); transformed chunks re-homed onto the target tile via `AlphaChunkTransform.TransformChunksForTarget` (content rotation + slot remap + WorldPosition recompute for the target tile); placement poses rotated via `ForwardTransformWorldPoint`/`ForwardTransformYawDegrees` |
| T071 | [StandardTerrainAdapter.cs](../../../src/viewer/WoWViewer/Terrain/StandardTerrainAdapter.cs) | Same resolution + transform wiring for the split-ADT path (local chunk/placement types, in-place list re-fill) |
| T072 | [PhaseComposition.cs](../../../src/core/WowViewer.Core/Maps/PhaseComposition.cs) | Core additions: `ForwardTransformTile` (inverse of `InverseTransformTile`), `ForwardTransformWorldPoint` (continuous world map using the adapters' chunk-corner convention), `ForwardTransformYawDegrees`, `ForwardTransformPlacementPoses` (returns transformed copies — placements are init-only) |
| T071/T072 | [TileContentTransform.cs](../../../src/core/WowViewer.Core/Maps/TileContentTransform.cs) | `TransformTileChunksForTarget` (transform + re-home onto the target tile with recomputed world corners) and a public raw-array surface (`GetVertexIndexMap`, `TransformHeightsRaw`, `TransformNormalsRaw`, `GetNormalTransform`, `TransformHoleMaskRaw`, `TransformSquareGridRaw`, `TransformMccvRaw`) so the Alpha adapter's distinct chunk type uses the identical math without drift |
| T073 | [WorldScene.cs](../../../src/viewer/WoWViewer/Terrain/WorldScene.cs) | `GetLayerFootprints` now composes donor tiles through rotation/mirror + offset — minimap footprints, click hit-tests, and cartography all see composed coordinates |
| T073 | [ViewerApp_PhaseLayers.cs](../../../src/viewer/WoWViewer/ViewerApp_PhaseLayers.cs) | The "overlapping the base map" status check composes the footprint through the transform |
| T073 | [MinimapHelpers.cs](../../../src/viewer/WoWViewer/MinimapHelpers.cs) | Minimap tile textures resolve through `ResolveTileSource` (donor tile via the policy) and the tile image is rotated/flipped with per-kind corner-UV permutations matching the terrain content transform |
| T073 | [ViewerApp_MinimapAndStatus.cs](../../../src/viewer/WoWViewer/ViewerApp_MinimapAndStatus.cs) | Footprint hit-test uses composed coordinates; minimap offset-drag is disabled for rotated/mirrored layers with a status hint (a drag delta in composed space does not map to a simple offset delta) |
| — | [TerrainManager.cs](../../../src/viewer/WoWViewer/Terrain/TerrainManager.cs), WorldScene | New `LayerTileExists`/`LayerHasTile` exposure feeding the minimap resolution predicate |

## How the transform works

Quarter turns are **exact-grid**: `ResolveTileSource` inverse-maps the target tile to the donor
tile, `TileContentTransform` (core) / `AlphaChunkTransform` (Alpha adapter) rotates the donor
tile's heights, normals, hole mask, alpha maps, shadow map, MCCV, liquid, and chunk slots, and
placements rotate about the layer's rotation origin (default donor-footprint centroid, one click
in the panel) with yaw adjustment. Mirrors compose after rotation, matching
`ComposeTileTransforms`' application order.

## Verification

- `dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug` — **0 Errors** (exit 0).
- `dotnet test ... --filter 'FullyQualifiedName~Maps'` — **157/157 passed** (includes the
  PhaseTileSource, TileContentTransform, and MapFootprint convention tests pinning the
  rotation/mirror math).
- **T074 visual gate (operator)**: load a base map, add a phase layer, set 90° rotation, and
  verify terrain + textures + liquid + objects rotate together on the real map.

## Fix round — T074 gate feedback (2026-09-07, same day)

The operator's first visual pass failed: rotated layers rendered nothing, offsets acted
inverted, and layers could land outside the grid. Root causes found and fixed:

1. **Nothing rendered (root cause)**: the adapters' `TileExists` streaming-admission still used
   the raw offset math (`tileX − TileOffsetX`), so rotated/mirrored target tiles were never
   admitted for streaming. Both adapters now admit through `ResolveTileSource` — the same
   resolution the composition uses, so streaming and rendering can no longer disagree.
2. **Grid confinement (operator rule)**: `ResolveTileSource` now returns empty for any target
   tile outside the 64×64 grid; footprint composition (WorldScene + panel overlap check) drops
   out-of-grid targets. A layer can never extend past the map grid.
3. **Origin fling**: a first rotation about origin (0,0) sent donor tiles to negative coordinates.
   The rotation combo now auto-centers the origin on the donor footprint the first time a layer
   is rotated, and resets it to (0,0) when rotation returns to none.
4. **Minimap double-offset**: `RenderPhaseFootprints` added the layer offset on top of the already
   composed footprints — removed.

Verification: build **0 Errors**; Maps tests **157/157 passed**. Interactive re-check of the
T074 visual gate returns to the operator.

## Known limits (recorded)

1. Free-angle (non-90°) rotations are core-supported (`ResolveRotationApproximation.FreeRotate`)
   but contribute no chunk-content transform (research R2) — the panel therefore offers quarter
   turns only; free angles remain a research item.
2. Placement mirroring flips yaw only (matching the core's own `TransformPlacementRotation`
   convention); roll/pitch chirality is not remirrored.
3. Minimap tile images rotate via corner-UV permutation (axis-aligned quads, pixel-exact for the
   4 exact-grid kinds).
