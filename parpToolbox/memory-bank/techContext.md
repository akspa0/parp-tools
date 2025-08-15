# Technology Context for parpToolbox

## Core Framework
- **Primary Framework:** .NET 9.0
- **Language:** C#

## Key Libraries & Dependencies
- **`wow.tools.local`**: The primary dependency for all low-level World of Warcraft file format parsing and handling.
- **Image Processing:** `SixLabors.ImageSharp` will likely be used for image data manipulation, particularly for texture extraction.
- **Database:** `Microsoft.Data.Sqlite` may be used for local data storage if needed.

*Note: Other dependencies like `BLPSharp`, `CascLib`, and `DBCD` are managed within `wow.tools.local` and will be used transitively.*

## Project Structure & Build
- **Solution File:** The primary solution is `parpToolbox.sln`, located in the project root.
- **Main Project:** `parpToolbox` is a .NET 9.0 console application located in `src/parpToolbox/`. This is the entry point and will contain all high-level orchestration and business logic.
- **Core Dependency:** The `wow.tools.local` project, located in `src/lib/wow.tools.local/`, is included in the solution and referenced by `parpToolbox`.
- **Legacy Reference:** The old `WoWToolbox` project structure remains in the `PM4Tool` directory for historical and reference purposes but is not part of the active build.

## Output Management
- The `ProjectOutput` utility has been implemented. It directs all generated files (e.g., OBJ exports) to a unified, timestamped `project_output` directory, preventing contamination of source data folders.

## PM4NextExporter – Current CLI Flags (2025-08-10)
- `--assembly composite-hierarchy` (default)
- `--ck-split-by-type` (optional split)
- `--export-tiles` (per-tile OBJs)
- `--project-local` (projection-based centroid translation at export time)
- `--export-mscn-obj` (MSCN point OBJ export)
- `--no-remap` (skip cross-tile MSCN remapping)
- `--legacy-obj-parity` (legacy winding/naming parity)

## Scene Model Extensions
- `TileVertexOffsetByTileId`, `TileVertexCountByTileId`, `TileIndexOffsetByTileId`, `TileIndexCountByTileId` for per-tile slicing.
- Aggregated `MscnVertices` and post-build MSCN remapping in `Pm4GlobalTileLoader`.
- `AssembledObject.Meta["tileId"|"tileX"|"tileY"]` for per-object dominant tile grouping.
