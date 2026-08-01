# Feature Specification: Precise M2 Masks in Tensor Packs

**Feature Branch**: `001-precise-m2-masks`

**Created**: 2026-06-30

**Status**: Complete (implementation) — Validation pending (see tasks.md T001-T004)

**Input**: User description: "Fix M2 mask rasterization from rectangle/centroid dots to triangle-level footprints in tensor pack building, validated through existing harvest tool `extract-unified` pipeline."

## User Scenarios & Testing

### User Story 1 - Precise M2 Masks in `extract-unified` Output (Priority: P1)

As a dataset consumer, I want M2 doodad masks in tensor packs to show actual model geometry (triangles) instead of rectangles or 2-pixel dots, so that training data actually reflects real object shapes.

**Why this priority**: Without this, all prior V18 stores and teacher-prior outputs are contaminated with wrong masks. Training cannot proceed correctly.

**Independent Test**: Run `extract-unified` on a single ADT tile with known M2 doodads, inspect the `object_mask` array — shapes should match model silhouettes, not bounding boxes.

**Acceptance Scenarios**:

1. **Given** a staged 3.3.5 client with azeroth_32_32.adt,
   **When** `extract-unified --map azeroth_32_32 --build <build> --staging <path>` completes,
   **Then** the resulting `object_mask` contains triangular fills for M2 doodads, not just centroid dots or bounding rectangles.

2. **Given** a tile with a known doodad (e.g., `world\doodads\foo.m2`),
   **When** the `.skin` file for that model is available in the MPQ,
   **Then** the extracted triangle vertices match the model's actual shape within tile pixel resolution.

---

### User Story 2 - Fallback Behavior for Unloadable M2s (Priority: P2)

As a dataset builder, I want the mask builder to gracefully degrade when M2 geometry cannot be loaded, so the pipeline doesn't crash on corrupt/missing `.skin` files.

**Why this priority**: Ensures pipeline robustness across all 50k+ ADT tiles.

**Independent Test**: Build a tile with a doodad whose `.skin` file is missing — masks should fall back to bounds rectangle or centroid circle without crashing.

**Acceptance Scenarios**:

1. **Given** a tile containing an M2 doodad with no companion `.skin` file in the MPQ,
   **When** `BuildObjectMasks` processes that placement,
   **Then** it falls back to bounds-rectangle fill, then centroid circle, without throwing.

2. **Given** an M2 file whose geometry data is corrupt or unreadable,
   **When** `TryLoadDoodadModelMetadata` encounters the parse failure,
   **Then** it catches the exception, logs the path, caches a null entry, and continues.

---

### User Story 3 - WMO Masks Remain Unchanged (Priority: P3)

As an existing consumer, I want WMO masks to remain exactly as they were — already at triangle precision via `TryPaintWmoFootprint`. No regression.

**Why this priority**: WMO mask quality is already acceptable; changes must not break it.

**Independent Test**: Run `extract-unified` on a tile with MODF placements and verify WMO mask triangle coverage matches prior behavior.

**Acceptance Scenarios**:

1. **Given** a tile with MODF placements that previously produced correct WMO masks,
   **When** the same tile is processed with the new code,
   **Then** the WMO `object_mask` array is pixel-identical to the prior run.

## Requirements

### Functional Requirements

- **FR-001**: `DoodadModelMetadata` MUST carry `TriangleVertices` (flat `IReadOnlyList<Vector3>`, 3 consecutive entries per triangle in model space).
- **FR-002**: `TryLoadDoodadModelMetadata` MUST attempt M2 triangle extraction via `M2GeometryReader` + companion `.skin` file (`M2SkinReader`) when the model path ends with `.m2`.
- **FR-003**: Companion `.skin` path MUST be resolved via `M2ModelIdentity.FromPath(modelPath).BuildSkinPath(0)`.
- **FR-004**: M2 triangle extraction MUST use skin `VertexLookup` to map skin triangle indices to geometry vertex positions.
- **FR-005**: The MDDF placement loop in `BuildObjectMasks` MUST attempt triangle rasterization via `PaintClippedTriangle` when `DoodadModelMetadata.TriangleVertices` is non-null.
- **FR-006**: Triangle rasterization MUST use `TryResolveProjectionMode` to determine projection (XZ/XZ-Y-only) per triangle, matching existing WMO behavior.
- **FR-007**: Triangle rasterization MUST fall back to bounds rectangle (`TryProjectBoundsToTilePixels`) when triangle vertices list is null/empty.
- **FR-008**: Bounds rectangle fallback MUST fall back to centroid circle (`PaintCircle`) when bounds cannot be determined.
- **FR-009**: WMO mask painting (`TryPaintWmoFootprint`) MUST remain entirely unchanged.
- **FR-010**: The existing harvest tool (`WowViewer.Tool.Harvest`) MUST be the sole validation target — no separate mask-validate tool.

### Key Entities

- **DoodadModelMetadata**: Per-model cache entry. Carries model-space bounds (Min, Max) and optionally `TriangleVertices` (flat list from M2 geometry + skin).
- **AdtTensorPackBuilder**: The C# class in `WowViewer.Core.IO.Maps` that builds tensor-sized NPZ/Zarr stores. Contains `BuildObjectMasks` method.
- **M2GeometryReader**: Reads M2 geometry vertex data (positions, normals, etc.) from raw bytes.
- **M2SkinReader**: Reads companion `.skin` file with vertex lookup tables and triangle index arrays.

## Success Criteria

### Measurable Outcomes

- **SC-001**: M2 doodad masks in `extract-unified` output show triangle-fill footprints for at least 90% of placement entries on azeroth_32_32 (MDDF=764, MODF=7).
- **SC-002**: Zero pipeline crashes when processing the full tileset of any staged client when geometry/skin loading fails.
- **SC-003**: WMO mask output is byte-identical to prior implementation (no regression).

## Assumptions

- M2 vertex positions from `M2GeometryReader` are in model space and do not require bone transforms for doodad-level models (doodads use single-bone or bone-less skeletons; positions are pre-skinned).
- Companion `.skin` files are named `<modelname>00.skin` for LOD 0 and exist in the same MPQ as the parent `.m2`.
- The `PaintClippedTriangle`, `ClipTriangleToTile`, `TryResolveProjectionMode`, and triangle rasterization helpers already exist in `AdtTensorPackBuilder.cs` and are shared between WMO and M2 code paths.
- Validation occurs via the existing harvest tool's `extract-unified` command, which loads MPQ archives via `NativeMpqService` and produces NPZ/Zarr output in the correct directory structure.
- All relevant `using` directives for `M2GeometryReader` and `M2SkinReader` are already present in `AdtTensorPackBuilder.cs` (they are in `WowViewer.Core.IO.M2`, already referenced by the project).
