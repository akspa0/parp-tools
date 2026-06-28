# Research: Minimap Deconstruction Engine

## Goal

Ground the new spec in existing `wow-viewer` ownership so implementation reuses the current harvester and capture surfaces instead of inventing a parallel pipeline.

## Existing Reusable Surfaces

### 1. Placement metadata already exists in the tensor-pack contract

- `src/core/WowViewer.Core/Maps/TerrainTileTensorPack.cs`
  - `PlacementMddfData`: `nameId, uniqueId, posX, posY, posZ, rotX, rotY, rotZ, scale`
  - `PlacementModfData`: `nameId, uniqueId, posX, posY, posZ, rotX, rotY, rotZ, bounds, flags`
  - `PlacementMddfNames` / `PlacementModfNames`: `nameId -> asset path`
  - `ObjectInstanceMask257`, `ObjectPreciseMask257`, `ObjectFilteredMask257`, `ObjectRoofMask256`

### 2. Placement names come from real ADT string tables

- `src/core/WowViewer.Core.IO/Maps/AdtPlacementReader.cs`
  - Reads `MMDX/MMID` and `MWMO/MWID`
  - Resolves original asset paths via `ResolveNameViaXid()`
  - Produces `AdtPlacementCatalog`, `AdtModelPlacement`, `AdtWorldModelPlacement`

This is the right provenance surface for the object library. It already carries the original per-placement asset path strings.

### 3. Object-mask generation already exists, but not yet as a per-object reusable library

- `src/core/WowViewer.Core.IO/Maps/AdtTensorPackBuilder.cs`
  - `BuildObjectMasks()` reads placements once and emits object masks and placement arrays
  - `BuildObjectRoofMasks()` emits roof-oriented mask signals
  - `TryPaintWmoFootprint()` already rasterizes WMO geometry into tile masks

The missing step is not "new mask logic." The missing step is one-object-at-a-time capture and object-library persistence.

### 4. Object footprint extraction logic already exists elsewhere in the repo

- `src/viewer/WoWViewer/Terrain/Vlm/VlmDatasetExporter.cs`
  - `GetModelFootprintPolygons()`
  - `TryReadM2RuntimeFootprintPolygons()`
  - `TryReadWmoFootprintPolygons()`

This is strong prior art for any shared footprint/library extraction that needs to move into a canonical core owner later.

### 5. Capture tooling already exists and already carries pose metadata lanes

- `tools/validation-capture/`
- `docs/architecture/spec025-t002-object-capture-audit-2026-05-26.md`

The capture lane already supports object-focused policy control and pose metadata carry-through. This is the correct seam for one-object-at-a-time capture orchestration.

### 6. Python dataset consumers already prefer precise or filtered object signals

- `data-harvester/src/harvester/v16_1_dataset.py`
  - prefers `object_precise_mask`
  - uses filtered/object-aware weighting paths

This supports the new teacher-prior plan: preserve the current better object signals and build the prior on top of them.

## Gaps

### Gap A - No reusable per-object library store

Today the repo stores tile-centric object signals. It does not yet store a canonical asset-centric object library with:

- original asset path
- normalized path
- asset type
- image artifact
- precise mask artifact
- capture metadata
- review state
- visibility / usefulness classification

### Gap B - No teacher-prior dataset for terrain-only minimap inputs

Today the height lanes read raw minimap plus object-aware loss gating. They do not yet read a generated processed minimap prior that already suppresses known object pixels.

### Gap C - No ADT-free object explanation stage

Development-map inference needs a minimap-only object explanation path. Current harvested placement arrays solve training supervision, not runtime ADT-free operation.

### Gap D - Existing older spec direction is too large and too model-heavy

Spec 025 assumed a broader object-roof identification lane with transformer-heavy phrasing and a "complete" status that no longer matches the current direction. The new execution plan must be smaller, simpler, and explicit about one-signal models.

## Execution Consequences

1. Build the asset library first.
2. Use that library to generate teacher priors from ADT-backed tiles.
3. Reboot terrain as height-only on processed priors.
4. Add minimap-only object explanation only after the teacher path is proven.
5. Keep normals separate until height proof is validated.
