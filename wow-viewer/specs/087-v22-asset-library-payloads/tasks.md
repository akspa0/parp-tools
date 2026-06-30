# Tasks: V22 Asset Library Payloads

**Input**: `specs/087-v22-asset-library-payloads/spec.md`, `specs/087-v22-asset-library-payloads/plan.md`

---

## Phase 1: Model Cache In TerrainTileTensorPack

**Purpose**: Thread the full M2/WMO documents from mask rasterization into the pack so the serializer can emit them.

- [ ] T001 **[P]** Add `PerTileModelPayloads` field to `TerrainTileTensorPack` as `IReadOnlyDictionary<string, V22ModelPayload>?`. Define `V22ModelPayload` record with: `Kind` (m2/wmo), `LoadError`, and a flattened `byte[]` blob of serialized arrays. Place it in a new file `wow-viewer/src/core/WowViewer.Core/Maps/V22ModelPayload.cs`.
- [ ] T002 Define `V22ModelPayloadSerializer` in `WowViewer.Core.IO.Maps` that takes an `M2GeometryDocument` and returns a `V22ModelPayload` (flattens vertices, normals, texcoords, skin triangles, render flags, blend modes, texture paths, bounds to byte arrays). Same for `WmoRenderDocument`.
- [ ] T003 **[P]** In `AdtTensorPackBuilder.BuildObjectMasks`, after `TryLoadDoodadModelMetadata` succeeds, capture the full `M2GeometryDocument`/`WmoRenderDocument` into a tile-local dictionary. After masks are built, convert each entry via `V22ModelPayloadSerializer` into `V22ModelPayload` and assign to `pack.PerTileModelPayloads`.
- [ ] T004 In `AdtTensorPackBuilder.Build(Basic)` and `BuildFromBytes`, thread the model capture path into the `TerrainTileTensorPack` constructor call so model payloads are populated for both single-tile and archive-backed paths.
- [ ] T005 Create `tests/WowViewer.Core.Tests/V22ModelPayloadTests.cs` — test that a synthetic `M2GeometryDocument` round-trips through the serializer to a `V22ModelPayload` with correct array sizes and load_error=0. Test a corrupt model produces load_error=1.

**Checkpoint**: Phase 1 complete — `TerrainTileTensorPack` carries model payloads for every M2/WMO on the tile.

---

## Phase 2: V22 Stream Serialization For Model Payloads

**Purpose**: Emit model payload arrays in the V22 stream profile.

- [ ] T06 **[P]** In `RawArraySerializer.WriteV22Arrays`, iterate `pack.PerTileModelPayloads`. For each entry, compute a stable 8-char path hash (`Path.GetHashCode().ToString("x8")` — deterministic, not cryptographic). Write flattened arrays as `m2_model_{hash}_vertices`, `m2_model_{hash}_triangles`, etc. Emit `load_error` as a scalar int array.
- [ ] T07 In `RawArraySerializer.BuildMetadataJson`, append `tile_model_paths` (ordered list of canonical paths matching the hash-based array naming order) and `tile_model_kinds` (matching `"m2"`/`"wmo"`/`"unknown"` labels).
- [ ] T08 **[P]** Update `RawArraySerializerTests.Serialize_V22_WritesFinalDatasetKeysAndDerivedArrays` — add a `V22ModelPayload` to the synthetic pack, assert `m2_model_*_vertices` key exists in the stream output.

**Checkpoint**: Phase 2 complete — V22 stream contains model payload arrays when models are present, and skips them cleanly when absent.

---

## Phase 3: Python Accumulation And ID Remap

**Purpose**: Consume model/tileset payloads in the Python writer and build Zarr groups.

- [ ] T09 In `V22ZarrWriter.add_tile`, scan the tile record for `m2_model_{hash}_*` and `wmo_model_{hash}_*` arrays. Use `tile_model_paths` and `tile_model_kinds` from metadata to resolve hashes to paths. Accumulate into `self._models` dict keyed by canonical path.
- [ ] T10 **[P]** In `V22ZarrWriter.add_tile`, scan for `tileset_texture_rgb_*` arrays. Use `mtex_texture_paths` from metadata to resolve indices to paths. Accumulate into `self._tilesets` dict keyed by canonical path.
- [ ] T11 In `V22ZarrWriter.finalize`, write the `models/` and `tilesets/` Zarr groups from accumulated dicts. Emit `mcly_tileset_ids` per tile: map each `mcly_texture_ids` value through the build-wide path table → index in `tileset_paths`. Emit remapped `mddf_model_ids`/`modf_model_ids` using the build-wide `model_paths` index.
- [ ] T12 **[P]** Update `test_v22_zarr_io.py` — add model payloads and tileset payloads to synthetic records, assert `models/model_paths` and `tilesets/tileset_paths` have correct counts after finalize.
- [ ] T13 In `build_v22_dataset.py._parse_tile_blob`, after parsing arrays and metadata, extract `m2_model_{hash}_*` / `wmo_model_{hash}_*` arrays from the per_tile dict and attach them to the `V22TileRecord`. Use `tile_model_paths` / `tile_model_kinds` from metadata to resolve the path mapping.

**Checkpoint**: Phase 3 complete — synthetic Zarr store has populated model/tileset groups and correctly remapped IDs.

---

## Validation

- [ ] V01 Run bounded proof: `WowViewer.Tool.Harvest harvest-stream --stream-profile v22` on one staged-client tile with known M2/WMO placements. Pipe to `build_v22_dataset.py build`. Open the Zarr store and verify `models/` group is populated.
- [ ] V02 Run bounded proof on a tile with no placements. Verify `models/` group is empty, no errors.
- [ ] V03 Run bounded proof on `4_0_0_11927` development-map tile (Cata-only assets). Verify WMO model payloads appear correctly.