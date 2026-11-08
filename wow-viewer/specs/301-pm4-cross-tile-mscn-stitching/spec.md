# Spec 301: PM4 Cross-Tile MSCN Compound Stitching & Multi-Tile Selection

## 1. Overview & Problem Statement

In Spec 300, PM4 collision hulls fragmented across CK24 elevation slices were assembled into compound whole objects *within* individual map tiles (`AssembleTileCompoundObjects(tileX, tileY, rawHulls)`). However, large game structures (such as shipyards, castles, large bridges, fortresses, and massive dungeons) straddle map tile boundaries. When the map compiler partitioned terrain and PM4 navmesh into 533.33-yard ADT/PM4 tiles, these structures were sliced at tile borders (e.g. between `development_13_51.pm4` and `development_13_50.pm4`).

In the current viewer, selecting a structure that crosses a tile boundary selects only the slice on the clicked tile (e.g., hovering or clicking the shipyard corner on tile `13_51` selects only 7 hulls, 8.8x26.2x5.1 yd, leaving the remaining 90% of the building on tile `13_50` unselected).

Ground-truth PM4 analysis reveals that the PM4 `MSCN` (scene-graph connector anchors) chunk explicitly encodes 3D connector points along tile boundaries that coincide between adjacent tiles with millimeter precision (measured $0.0022\,\text{yd} \approx 2.2\,\text{mm}$ between `development_13_51` and `development_13_50` along the $X = 27200.0$ seam).

This specification implements two-stage cross-tile compound stitching using MSCN connector anchor points and boundary spatial adjacency to merge multi-tile structural pieces into unified cross-tile compound whole objects across all loaded tiles.

## 2. Requirements & Acceptance Criteria

### Functional Requirements

- **FR-001: Core Cross-Tile Compound Clustering Engine**
  - Implement `ClusterCrossTileCompounds` in `Pm4SpatialClustering.cs` (`WowViewer.Core.PM4.Services`).
  - Given compound object descriptors across adjacent tiles, cluster them using Union-Find based on:
    1. Coincident MSCN connector points along tile boundaries (world distance $\le 0.5\,\text{yd}$ or matching `Pm4ConnectorKey`).
    2. Tile boundary AABB adjacency ($\le 2.5\,\text{yd}$ boundary distance) with seam MSCN proximity.
  - Return connected component clusters spanning multiple tiles.

- **FR-002: Multi-Tile Compound Object Model**
  - Update `Pm4CompoundObject.cs` to support multi-tile structures:
    - Expose `IReadOnlySet<(int TileX, int TileY)> SpanningTiles` and boolean `IsCrossTile`.
    - Compute aggregate 3D bounds (`WorldBoundsMin`, `WorldBoundsMax`), centroid, dimensions, and totals (triangles, lines, surfaces, hulls) across all constituent hulls on all spanning tiles.
    - Format `DisplayName` to identify spanning tiles, e.g. `[13_50 + 13_51] Cross-Tile Whole Obj #X (<N> hulls across 2 tiles, <X>x<Y>x<Z> yd)`.

- **FR-003: Cross-Tile Stitching Service & Scene Lifecycle Integration**
  - Add `StitchCrossTileCompoundObjects` in `Pm4CompoundObjectService.cs`.
  - In `Pm4OverlayScene.cs`, after loading tile compounds (during cache merge or direct file load), run the cross-tile stitching pass using loaded MSCN points (`EnsurePm4MscnData()`).
  - Update `_pm4HullToCompoundLookup` so every constituent hull on all spanning tiles references the unified cross-tile compound object.

- **FR-004: Unified Single-Click Viewport Selection & Multi-Tile Bounding Box**
  - Clicking any constituent hull on any spanning tile selects the entire cross-tile compound object.
  - `Pm4OverlayMeshRenderer` highlights all constituent hulls across all spanning tiles in the selection color (purple wireframe + solid tint).
  - `WorldScene` draws a single unified wireframe bounding box enclosing the multi-tile structure's world extents across the tile border seam.

- **FR-005: Cockpit UI, Tooltips & Multi-Tile GLB Export**
  - Hover tooltips and sidebar cockpit identify cross-tile whole structures with multi-tile tags, true dimensions, and the complete constituent hull hierarchy grouped by tile.
  - "Export Selected PM4 to GLB" exports all constituent hulls across all spanning tiles into the exported model.

### Success Criteria

- **SC-001**: Clicking a multi-tile structure (e.g. `development_13_51.pm4` and `development_13_50.pm4`) selects all hulls on all spanning tiles simultaneously.
- **SC-002**: Selection bounding box seamlessly wraps both sides of the tile boundary without truncation.
- **SC-003**: 100% pass on focused PM4 spatial clustering unit tests including cross-tile MSCN stitching.
- **SC-004**: Clean solution build with 0 errors and `Verify-UiReference.ps1` passes with 0 errors and 0 warnings.

