# Implementation Plan: Precise M2 Masks in Tensor Packs

**Branch**: `001-precise-m2-masks` | **Date**: 2026-06-30 | **Spec**: `specs/001-precise-m2-masks/spec.md`

**Input**: Feature specification from `/specs/001-precise-m2-masks/spec.md`

**Key Discovery**: All FR-001 through FR-010 are **already implemented** in `AdtTensorPackBuilder.cs` (lines 1736-2441). The code already has `DoodadModelMetadata` with `TriangleVertices`, the M2+skin triangle extraction in `TryLoadDoodadModelMetadata`, triangle rasterization in the MDDF loop, and all three fallback levels. This plan covers validation and task decomposition only.

## Summary

Fix M2 doodad masks in tensor packs from rectangle/centroid-dot fallbacks to actual triangle-level geometry. Implementation is complete — `TryLoadDoodadModelMetadata` reads M2 geometry + companion `.skin` files, maps skin triangle indices through `VertexLookup` to geometry vertices, and `BuildObjectMasks` rasterizes those triangles onto all six mask arrays (`mask`, `preciseMask`, `instanceMask`, `mddfMask`, `filteredMask`). Falls back to bounds rectangle, then centroid circle, on failure.

## Technical Context

**Language/Version**: C# / .NET 10

**Primary Dependencies**: `WowViewer.Core.IO.M2` (`M2GeometryReader`, `M2SkinReader`), `WowViewer.Core.M2` (`M2ModelIdentity`)

**Storage**: Tensor packs (NPZ/Zarr) — no new storage

**Testing**: `dotnet test` on `WowViewer.Core.Tests` / `WowViewer.Core.IO.Tests`

**Target Platform**: wow-viewer harvest pipeline (CLI tool)

## Constitution Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Repo Independence | PASS | All changes in `wow-viewer/src/core/WowViewer.Core.IO/` |
| II. Library-First | PASS | Uses existing `M2GeometryReader` and `M2SkinReader` in shared I/O |
| III. Real-Data Validation | REQUIRED | Phase 1 validates on staged 3.3.5 client tile azeroth_32_32 |
| IV. Residual Model Chain | N/A | Not a training change |
| V. Streaming-First Dataset | N/A | No pipeline change |
| VI. No H:\CLIENTS | PASS | All validation uses `output/tmp/wowarchive-clients/` |

## Project Structure

### Existing (no new files needed — implementation is complete)

```text
wow-viewer/src/core/WowViewer.Core.IO/Maps/
├── AdtTensorPackBuilder.cs        # DoodadModelMetadata, TryLoadDoodadModelMetadata, BuildObjectMasks — all done
├── AdtPm4MaskBuilder.cs           # Unchanged
└── AlphaTensorPackBuilder.cs      # Unchanged

wow-viewer/src/core/WowViewer.Core.IO/M2/
├── M2GeometryReader.cs            # Existing reader
└── M2SkinReader.cs                # Existing reader

wow-viewer/src/core/WowViewer.Core/M2/
├── M2ModelIdentity.cs             # Existing identity / .skin path resolver
├── M2GeometryDocument.cs          # Existing model types
└── M2SkinDocument.cs              # Existing skin types
```

## Implementation Phases

The implementation is already complete. This plan covers validation.

### Phase 1: Real-Data Validation (P1)

**Goal**: Verify that `extract-unified` on a real tile produces triangle-filled M2 masks, not dots/rectangles.

**Approach**:
1. Run `extract-unified` on azeroth_32_32 from a staged 3.3.5 client (MDDF=764, MODF=7 as per SC-001).
2. Inspect `object_precise_mask` array — verify triangular fill shapes for M2 doodads.
3. Confirm at least 90% of MDDF entries produce triangle footprints (SC-001).
4. Confirm zero crashes (SC-002).
5. Run on a tile without `.skin` companion — confirm graceful fallback to bounds rectangle, then centroid circle, without throwing (User Story 2).

### Phase 2: WMO Regression Check (P2)

**Goal**: Verify WMO mask output is byte-identical to prior implementation.

**Approach**:
1. Run `extract-unified` on a tile with MODF placements previously known to produce correct WMO masks.
2. Compare `object_mask` (precise WMO part) output — must be pixel-identical to a prior known-good run.
3. If no prior artifact exists, verify WMO triangles look correct by inspection (compare triangle counts, coverage pattern).

### Phase 3: Edge Cases & Caching (P2)

**Goal**: Confirm `doodadModelCache` handles reloads, null entries, and concurrent patterns.

**Approach**:
1. Verify that `doodadModelCache[modelPath] = null` is set on failure (already at line 2428).
2. Verify that `TriangleVertices` is null when skin load fails (already at line 2416-2419 — `triangleVertices` defaults to null on catch).
3. Verify that repeated requests for the same model path hit the cache (already at line 2346).

## Complexity Tracking

No constitution violations. Implementation is complete — this plan covers validation only.