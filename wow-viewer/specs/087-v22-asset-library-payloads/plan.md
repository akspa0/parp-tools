# Implementation Plan: V22 Asset Library Payloads

**Branch**: `087-v22-asset-library-payloads` | **Date**: 2026-06-30 | **Spec**: `specs/087-v22-asset-library-payloads/spec.md`

## Summary

Three additions to close the gap between the V22 tile stream and a fully-populated Zarr store:

1. Capture the full `M2GeometryDocument` and `WmoRenderDocument` payloads into `TerrainTileTensorPack` from the existing mask-rasterization path, then serialize them in the V22 stream.
2. Add per-tile model payload arrays (`m2_model_{hash}_*`, `wmo_model_{hash}_*`) to the V22 stream profile.
3. Make the Python Zarr writer accumulate model and tileset payloads into build-wide groups and remap placement IDs.

## Technical Context

**Language/Version**: C# .NET 10 + Python 3.11+

**Primary Dependencies**:
- `WowViewer.Core.M2.M2GeometryDocument` (already exists)
- `WowViewer.Core.Wmo.WmoRenderDocument` (already exists)
- `WowViewer.Core.IO.Maps.AdtTensorPackBuilder` (existing builder — model data already loaded here)
- `WowViewer.Core.IO.Maps.RawArraySerializer` (V22 stream profile)
- `wow-viewer/data-harvester/src/harvester/v22_zarr_io.py` (Python Zarr writer/reader)

**Storage**: Existing RawArraySerializer binary stream + Zarr v3.

**Testing**: xUnit tests for C# serialization (per-tile model payload arrays in V22 stream), pytest for Python accumulation.

**Target Platform**: Staged `0_5_3_3368`, `3_3_5_12340`, `4_0_0_11927` clients.

## Constitution Check

| Principle | Status | Notes |
|-----------|--------|-------|
| Repo Independence | PASS | All work inside `wow-viewer/` |
| Library-First | PASS | Model cache added to `TerrainTileTensorPack` in `WowViewer.Core.Maps` |
| Real-Data Validation | REQUIRED | Bounded proof on single tile with known M2 + WMO placements |
| No `H:\CLIENTS` | PASS | Only staged clients |
| One Phase at a Time | PASS | Three phases, sequential dependencies |
| Streaming-First | PASS | Model data goes through the existing binary V22 stream |

## Project Structure

```text
wow-viewer/src/core/WowViewer.Core/Maps/
|-- TerrainTileTensorPack.cs       # + ModelCache dict, TilesetCache dict, PerTileModelPayloads

wow-viewer/src/core/WowViewer.Core.IO/Maps/
|-- AdtTensorPackBuilder.cs        # populate ModelCache from existing TryLoadDoodadModelMetadata path
|-- RawArraySerializer.cs          # V22 profile: serialize model payloads as named arrays

wow-viewer/src/core/WowViewer.Core.IO/Wmo/
|-- WmoRenderDocumentReader.cs     # (no changes needed — already works)

wow-viewer/src/core/WowViewer.Core.IO/M2/
|-- M2GeometryReader.cs            # (no changes needed — already works)

wow-viewer/data-harvester/src/harvester/
|-- v22_zarr_io.py                 # V22ZarrWriter: accumulate models/tilesets, remap IDs

wow-viewer/tests/WowViewer.Core.Tests/
|-- RawArraySerializerTests.cs     # + model payload array assertions
```

## Implementation Phases

### Phase 1: Model Cache In TerrainTileTensorPack

Goal: Thread the full M2/WMO documents from the mask rasterizer through to the pack.

Work:
- Add `PerTileModelPayloads` dictionary to `TerrainTileTensorPack` (path → serializable payload record).
- In `AdtTensorPackBuilder.BuildObjectMasks`, when `TryLoadDoodadModelMetadata` succeeds, also capture the full `M2GeometryDocument` or `WmoRenderDocument` into a build-level dictionary threaded through the builder.
- After masks are built, transfer the build-level dict entries relevant to this tile into `pack.PerTileModelPayloads`.
- Define a serializable intermediate record type for model payloads that flattens `M2GeometryDocument` and `WmoRenderDocument` into a common shape suitable for the stream.

Validation:
- Unit test: synthetic pack with 1 M2 placement has non-null `PerTileModelPayloads` with correct path key.
- Unit test: model with missing `.skin` produces `load_error=1` entry.

### Phase 2: V22 Stream Serialization For Model Payloads

Goal: Emit model payload arrays for each unique model on each tile.

Work:
- In `RawArraySerializer.WriteV22Arrays`, iterate `pack.PerTileModelPayloads`.
- For each model payload, compute a stable 8-char hash from the canonical path.
- Write arrays named `m2_model_{hash}_vertices`, `m2_model_{hash}_triangles`, etc. (or `wmo_model_{hash}_*`).
- Update the metadata JSON to include `tile_model_paths` and `tile_model_kinds`.
- Null-safe: if `PerTileModelPayloads` is null/empty, skip model arrays.

Validation:
- `RawArraySerializerTests`: tile with 1 M2 payload in the pack produces `m2_model_*` arrays in the stream.
- Empty pack (no models) skips model arrays without crashing.

### Phase 3: Python Accumulation And ID Remap

Goal: The Python writer consumes model/tileset payloads and writes build-wide Zarr groups.

Work:
- In `V22ZarrWriter.add_tile`, extract `m2_model_{hash}_*` and `wmo_model_{hash}_*` arrays from the record, accumulate by canonical path.
- Extract `tileset_texture_rgb_*` arrays and MTEX paths, accumulate by path into build-wide tileset table.
- Write `models/model_paths`, `models/model_kind`, `models/load_error` in `finalize()`.
- Write `tilesets/tileset_paths`, `tilesets/load_error`, `tilesets/texture_shape`, per-tileset texture RGB.
- Compute `mcly_tileset_ids`: for each tile, map per-chunk `mcly_texture_ids` → build-wide tileset index.
- Remap `mddf_model_ids` / `modf_model_ids` from per-tile nameId to build-wide path index.

Validation:
- `test_v22_zarr_io`: model payloads in tile records get accumulated into `models/model_paths`.
- Two tiles sharing same texture produce one tileset entry.
- `mcly_tileset_ids` round-trips correctly against source texture ids.

## Validation And Diagnostics Matrix

### Phase 1
- M2 geometry fields all present in pack entry
- WMO geometry fields all present in pack entry
- Empty pack does not crash

### Phase 2
- Model array naming follows `{kind}_model_{hash}_{field}` convention
- Metadata JSON includes `tile_model_paths` and `tile_model_kinds`
- Zero-model tile produces clean stream

### Phase 3
- Build-wide model/tileset groups populated correctly
- `mcly_tileset_ids` index bounds valid
- `mddf_model_ids` / `modf_model_ids` index into `models/model_paths`
- Synthetic round-trip test passes
