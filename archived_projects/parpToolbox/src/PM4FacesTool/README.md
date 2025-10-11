# PM4FacesTool

Export PM4 tile geometry to OBJ with optional glTF 2.0 outputs and detailed export indexes.

This tool supports both single-tile export and batch export across a directory, produces per-object and per-tile assets, and writes JSON/CSV indexes for downstream processing and validation.

Assembly strategy: tiles and the merged render mesh are built object-first (grouping by `MSUR.IndexCount`) to avoid cross-object vertex deduplication and preserve complete faces/surfaces.

## Build & Run

- Requires: .NET 9 SDK
- Build solution:
  ```bash
  dotnet build parpToolbox.sln -c Release
  ```
- Run (single file):
  ```bash
  dotnet run --project src/PM4FacesTool -c Release -- --input "C:\Data\Zone_10_12.pm4"
  ```
- Run (batch directory):
  ```bash
  dotnet run --project src/PM4FacesTool -c Release -- --input "C:\Data\RegionDir" --batch
  ```

## CLI Options

- `--input|-i <tile.pm4|dir>` Input PM4 file (single) or directory (with `--batch`).
- `--out <dir>` Output root. Defaults to `project_output/pm4faces_YYYYMMDD_HHmmss`.
- `--batch` Process all tiles in input directory that share the same prefix.
- `--group-by <composite-instance|type-instance|type-attr-instance|surface|groupkey|composite|render-mesh|surfaces-all>` Object grouping strategy. `render-mesh`/`surfaces-all` emits a single merged mesh using object-first assembly.
- `--render-mesh-merged` Emit a merged `render_mesh.obj` alongside any group-by mode.
- `--snap-to-plane` Project vertices to each surface's MSUR plane (optional; off by default).
- `--height-scale <float>` Multiply MSUR Height by this factor during snapping (e.g., 0.02777778 for 1/36).
- `--no-mscn-remap` Disable MSCN vertex remap during region load (often improves face completeness).
- `--legacy-parity` Apply legacy X-flip/parity for objects (tiles always force X-flip).
- `--project-local` Recenters geometry about the mean before export.
- `--ck-use-mslk` Use MSLK linkage hints when building composite instances.
- `--ck-allow-unlinked-ratio <0..1>` Threshold for allowing unlinked surfaces in an instance.
- `--ck-min-tris <int>` Minimum triangles per component to include (use `0` to preserve anchors).
- `--ck-merge-components` Keep per-object DSU components under each CK24 (no monolithic merged OBJ).
- `--gltf` Also export glTF 2.0 JSON (`.gltf`) with external binary (`.bin`).
- `--glb` Also export binary glTF (`.glb`).

Notes:
- Single-file input loads only that tile. Use `--batch` to expand to a region.
- Upcoming options (not yet implemented): `--group-by surfacekey`, `--xflip-objects` (default ON) to unify with tile flipping.

## Output Layout

Within the session directory (e.g., `project_output/pm4faces_YYYYMMDD_HHmmss/<TileName>/`):

- `objects/` Per-object exports according to `--group-by`.
  - `*.obj` Always written
  - `*.gltf` + `*.bin` When `--gltf` is set
  - `*.glb` When `--glb` is set
- `tiles/` Per-tile exports assembled object-first (always flip X for consistent orientation)
  - Same optional glTF/GLB outputs as above
  - `render_mesh.obj` when `--group-by render-mesh|surfaces-all` is used, or when `--render-mesh-merged` is provided
- `surface_coverage.csv` Per-surface export coverage and face counts
- `tile_coverage.csv` Per-tile export coverage and face counts
- `ck_instances.csv` Component instance summary per CK24 (composite-instance mode)
- `instance_members.csv` Surface membership for each component instance
- `objects_index.json` JSON index for all object exports
- `tiles_index.json` JSON index for all tile exports

## JSON Index Schemas

### objects_index.json
Each entry:
```json
{
  "Id": "ck24:000123:inst:00005",   // stable identifier for the exported object
  "Name": "CK_000123_inst_00005",   // human-readable
  "Group": "composite-instance",    // "surface" | "group" | "composite-instance"
  "ObjPath": "objects/CK_000123/CK_000123_inst_00005.obj",
  "GltfPath": "objects/CK_000123/CK_000123_inst_00005.gltf",  // optional
  "GlbPath": "objects/CK_000123/CK_000123_inst_00005.glb",    // optional
  "FacesWritten": 12345,
  "FacesSkipped": 0,
  "SurfaceIndices": [10, 11, 24],
  "IndexFirst": 1024,                 // first MSVI index included
  "IndexCount": 8192,                 // total indices across members
  "FlipX": false                      // objects flip only with --legacy-parity (currently)
}
```
Paths are relative to the session directory.

### tiles_index.json
Each entry:
```json
{
  "TileId": 2469,                     // linear tile id (Y*64 + X)
  "Name": "Tile_37_49",
  "ObjPath": "tiles/Tile_37_49.obj",
  "GltfPath": "tiles/Tile_37_49.gltf",   // optional
  "GlbPath": "tiles/Tile_37_49.glb",     // optional
  "StartIndex": 30528,                // first MSVI index for this tile
  "IndexCount": 65536,                // index count for this tile
  "FacesWritten": 21845,
  "FacesSkipped": 0,
  "FlipX": true                       // tiles always force X-flip
}
```

## Canonical References

- PM4 Format: ../../docs/formats/PM4.md
- PM4 Chunk Reference: ../../docs/formats/PM4-Chunk-Reference.md
- PM4 Field Reference (Complete): ../../docs/formats/PM4-Field-Reference-Complete.md
- PM4 Assembly Relationships: ../../docs/formats/PM4_Assembly_Relationships.md

## Coordinate System & Flipping

- **Tiles**: Always force X-flip with winding swap to preserve normals.
- **Objects**: Currently flip only when `--legacy-parity` is provided.
- **glTF/GLB**: Mirrors OBJ behavior (same flips and winding), and supports `--project-local` recentering.

## Grouping Modes

- `composite-instance` (default):
  - Groups by CK24, then retains DSU components per CK bucket.
  - Writes instance membership CSVs.
- `groupkey`: Groups by `MSUR.GroupKey`.
- `composite`: Groups by full `MSUR.CompositeKey`.
- `surface`: One file per `MSUR` surface.
- `render-mesh`: Emits a single merged mesh using object-first assembly.
- `surfaces-all`: Emits a single merged mesh using object-first assembly.

Planned: `surfacekey` mode using `MSUR.SurfaceKey` as identity.

## Examples

- Single tile, OBJ only:
  ```bash
  dotnet run --project src/PM4FacesTool -- -i "C:\Data\Zone_10_12.pm4"
  ```
- Single tile with glTF + GLB and composite-instance grouping:
  ```bash
  dotnet run --project src/PM4FacesTool -- -i "C:\Data\Zone_10_12.pm4" --group-by composite-instance --gltf --glb
  ```
- Batch region, group by composite key:
  ```bash
  dotnet run --project src/PM4FacesTool -- -i "C:\Data\RegionDir" --batch --group-by composite --gltf
  ```
- Real data, recommended quick check (anchors preserved, MSCN remap off, merged mesh alongside tiles/objects):
  ```bash
  dotnet run --project src/PM4FacesTool -c Release -- --input ".\test_data\original_development\development_00_00.pm4" --batch --no-mscn-remap --ck-min-tris 0 --render-mesh-merged
  ```

## Validation Checklist

- **Object count** roughly matches expectations for the tile (e.g., ~15 for known reference).
- **Normals** correct (no inverted faces) and geometry is **solid** (no missing triangles from index issues).
- **Indexes** (`objects_index.json`, `tiles_index.json`) reference all emitted assets.
- **Coverage CSVs** totals align with expected face counts.

## Current Limitations & Next Steps

- Investigating face/surface orientation inconsistencies and potential double-flip scenarios.
- Considering a new default `--xflip-objects` to unify object flipping with tiles.
- Adding `--group-by surfacekey` for alternative identity logic.
- Long-term: richer object identity via `MSLK.ParentIndex_0x04` and combined geometry systems.

See also: `memory-bank/restart_guide.md` for active context and next steps.
