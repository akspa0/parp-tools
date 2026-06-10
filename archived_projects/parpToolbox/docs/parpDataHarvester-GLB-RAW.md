# parpDataHarvester — GLB-RAW Hierarchical Export (Plan & Spec)

## Purpose
Export PM4 data to a single-file GLB that preserves:
- Full MSLK container hierarchy as glTF nodes (no loss of structure).
- Geometry subdivisions as per objects (default) or per surfaces.
- All critical PM4 linkages serialized into node/mesh `extras`.

This enables Blender import for digital archaeology without redistributing raw PM4 files.

## Minimal CLI
```
parpDataHarvester export-glb-raw --in <pm4_tile_dir|region_dir> --out <glb_out_dir> [--mode objects|surfaces] [--per-region] [--flip-x]
```
- Default mode: `--mode objects` (group by MSUR.IndexCount).
- Output:
  - Per-tile mode: one GLB per tile under `<out>/tiles/>`.
  - Per-region mode (`--per-region`): single GLB under `<out>/region/`.

## Data Sources (existing code)
- `src/PM4Rebuilder/PM4MapLoader.cs` — loads tiles, MSLK links, MSPV/MSVT/MSVI, MPRL, MSUR.
- `src/WoWToolbox/WoWToolbox.Core.v2/Services/PM4/MSURSurfaceExtractionService.cs` — surface geometry.
- `src/WoWToolbox/WoWToolbox.Core/Navigation/PM4/MslkObjectMeshExporter.cs` — object assembly patterns.
- Reference writer: `src/PM4FacesTool/GltfWriter.cs` — GLB packing flow.

## GLB-RAW Structure
- Nodes (MSLK):
  - One glTF node per MSLK entry; parent-child via MSLK parent links.
  - `MspiFirstIndex == -1` → grouping-only nodes.
  - Node `extras`: `{ id, type, parentIndex, mspiFirstIndex, tileId, tileX, tileY }`

- Meshes:
  - Default objects mode: group surfaces by `MSUR.IndexCount` (critical discovery).
  - Surfaces mode: one primitive per MSUR.
  - Primitive `extras`: `{ surfaceId, indexCount, vertexPoolOffsets, indexPoolOffsets, mprlRef? }`

- Transforms:
  - No transform is applied by default; positions are passed through as provided upstream.
  - Optional: `--flip-x` to invert X (apply `-vertex.X`) when visualization parity requires it.
  - MPRL placement cross-refs captured in extras; transforms can remain identity for v1.

- Attributes & Materials:
  - Positions required; normals optional (omit in v1 if unavailable).
  - Materials deferred; geometry-first. Future hooks via `extras`.

## Implementation Plan
1. Project scaffold: `src/parpDataHarvester/` (net9.0), add to `parpToolbox.sln`.
2. CLI: `export-glb-raw` with args above; load tiles via `PM4MapLoader`.
3. Hierarchy: `MslkNodeGraphBuilder` builds node tree + node `extras`.
4. Geometry: `RawGeometryAssembler`
   - objects mode (default): group by `MSUR.IndexCount`
   - surfaces mode: per-MSUR primitive
   - pull MSPV/MSVT/MSVI; optional X inversion with `--flip-x`
5. Writer: `GltfRawWriter`
   - multi-node, multi-mesh, buffers/bufferViews/accessors
   - pack JSON+BIN to GLB (adapting `GltfWriter.cs` logic)
6. Output: `<out>/tiles/<tileId>.glb`
7. Validation: open in Blender; check node/mesh counts and `extras` presence.

## Risks & Mitigations
- Parentage specifics: verify MSLK parent map from loader on a few tiles.
- Normals: skip in v1 if cost/availability is high.
- Buffer layout: reuse proven packing patterns, add internal asserts.

## Deliverables
- `Program.cs` (CLI)
- `MslkNodeGraphBuilder.cs`
- `RawGeometryAssembler.cs`
- `GltfRawWriter.cs`
- `README.md` with usage and Blender notes

## Validation
- Run on real PM4 tile directory.
- Sanity checks:
  - Node count ≈ MSLK entries per tile
  - Mesh/primitive counts match mode (objects vs surfaces)
  - Metadata present in `extras`

## Defaults & Preferences
- Minimal knobs; geometry preserved; no PM4 redistribution.
- Defaults compatible with anchor-preserving pipelines (no surface filtering).
