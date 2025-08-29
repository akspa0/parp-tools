# ADTPreFabTool.Console — CLI Guide

The Console tool extracts terrain meshes from WoW ADT files, optionally exports GLB/GLTF, aggregates terrain patterns, and can generate a tiles manifest for downstream tools.

## Prerequisites

- .NET SDK 9.0 or later
- WoWFormatLib is included as a project reference under `lib/wow.tools.local/`
- Windows recommended

Build:
```bash
dotnet build src/ADTPreFabTool.Console/ADTPreFabTool.Console.csproj -c Release
```

## Usage

```text
ADTPreFabTool.Console <adt_file_or_folder_path> [output_directory] [--recursive|--no-recursive] [--no-comments]
  [--glb] [--gltf] [--glb-per-file|--no-glb-per-file] [--manifest|--no-manifest]
  [--output-root <path>] [--timestamped|--no-timestamp] [--chunks-manifest|--no-chunks-manifest]
  [--meta|--no-meta] [--similarity-only]
  [--minimap-root <path>] [--trs <path>] [--world-minimap-root <path>] [--data-version <ver>] [--cache-root <path>] [--decode-minimap]
  [--export-minimap-overlays] [--export-minimap-obj] [--yflip|--no-yflip]
```

- `adt_file_or_folder_path`:
  - Single `.adt` file: processes one file
  - Directory: processes all `.adt` files
- `output_directory` (optional): default `output`

### Options

- `--recursive | --no-recursive`  Process subdirectories when input is a folder. Default: `--recursive` in directory mode.
- `--no-comments`                 Omit comments in generated OBJ files.
- `--glb`                         When input is a single file, export GLB next to OBJ.
- `--gltf`                        When input is a single file, export GLTF (JSON) next to OBJ.
- `--glb-per-file | --no-glb-per-file`  When input is a directory, export one GLB per ADT. Default: `--glb-per-file`.
- `--manifest | --no-manifest`    When input is a directory, append an NDJSON manifest entry per ADT. Default: `--manifest`.
- `--output-root <path>`          Override default output root when not supplying `output_directory`.
- `--timestamped | --no-timestamp` Add a timestamped run folder. Default: `--timestamped`.
- `--chunks-manifest | --no-chunks-manifest` Emit chunk-level manifest in run folder. Default: on.
- `--meta | --no-meta`            Write auxiliary metadata. Default: on.

### Minimap inputs and caching

- `--minimap-root <path>`         Root containing `textures/Minimap/` (e.g., extracted CASC tree).
- `--trs <path>`                  Optional path to `md5translate.txt` under `textures/Minimap/`.
- `--world-minimap-root <path>`   Optional root to `World/Minimaps/`. If omitted, tool attempts auto-detection relative to `--minimap-root`.
- `--data-version <ver>`          Version label for cache partitioning, e.g., `0.6.0`.
- `--cache-root <path>`           Cache base directory. Default: under output root.
- `--decode-minimap`              Decode BLP minimaps to PNG into cache.
- `--export-minimap-overlays`     Export `minimap_index.csv` describing all discovered tiles.
- `--export-minimap-obj`          Generate per-tile OBJ/MTL textured with cached minimap PNG, using the real ADT terrain mesh.
- `--yflip | --no-yflip`          Flip V to match minimap orientation (default: `--yflip`); use `--no-yflip` to disable.

Notes:
- `--glb`/`--gltf` apply to single-file mode (`ProcessADTFile`).
- `--glb-per-file`/`--manifest` apply to directory mode (`ProcessADTDirectory`).
- Defaults (directory input): `--recursive --glb-per-file --manifest --yflip` (use `--no-yflip` to disable V flip on minimap texture).

## Outputs

Single-file mode (`terrain_improved.obj` in output directory):
- `terrain_improved.obj`   Improved triangulation mesh for the ADT
- Optional: `<stem>.glb` or `<stem>.gltf` when `--glb` or `--gltf`
- `terrain_patterns.txt`   Pattern analysis for that file

Directory mode (merged outputs + per-file optional):
- `terrain_merged.obj`     Combined OBJ from all ADTs
- `terrain_patterns.txt`   Aggregated pattern report
- Optional per-file GLBs with `--glb-per-file`: `<stem>.glb`
- Optional manifest with `--manifest`: `tiles_manifest.json` (NDJSON)

### Minimap cache structure (when `--decode-minimap`)

Cache root: `<cache_root>/_cache/<data-version>/minimap_png/`

- ADT tiles (TRS / legacy ALT):
  - `<MapName>/<MapName>_<X>_<Y>.png`
  - Alternate differing content for same (MapName,X,Y): `<MapName>/<MapName>_<X>_<Y>__alt.png`

- WMO tiles (preserved hierarchy):
  - `wmo/<relative_path_under_World/Minimaps>/<original_stem>.png`
  - Example: `wmo/transports/passengership/passengership_00_00_00.png`

De-duplication behavior:
- ADT: TRS > ALT precedence for same (MapName,X,Y). Identical content is not decoded twice; differing content produces `__alt`.
- WMO: every file path produces its own output (even if MD5 matches others). We only skip if that exact PNG already exists.

### Minimap-textured Terrain OBJ Export (`--export-minimap-obj`)

Generates per-tile OBJ/MTL textured with cached minimap PNG, using the real ADT terrain mesh.

Outputs in run folder:
- `minimap_obj/<Map>/<Map>_<X>_<Y>.obj|.mtl|.png`

Flags:
- `--yflip` (default): flip V to match minimap orientation; use `--no-yflip` to disable.
- `--tiles X_Y,...` or `--tile-range x1,y1,x2,y2` to limit tiles.

Example:
```bash
dotnet run --project src/ADTPreFabTool.Console/ADTPreFabTool.Console.csproj \
  "test_data/0.6.0/tree/World/Maps/Azeroth" \
  --minimap-root "test_data/0.6.0/tree/textures/Minimap" \
  --data-version 0.6.0 \
  --cache-root "./cachedMinimaps" \
  --decode-minimap \
  --export-minimap-obj \
  --tiles 30_42,31_42
```

## Manifest Details

When `--manifest` is used, each processed ADT appends a line to `tiles_manifest.json` in the output directory:

```json
{"file":"<name>.adt","glb":"<stem>.glb","aabb":[minX,minY,minZ,maxX,maxY,maxZ]}
```

- Coordinates are Z-up: `[minX, minY, minZ, maxX, maxY, maxZ]`
- NDJSON format: one JSON object per line, no commas, no surrounding array
- Intended for downstream tools or custom viewers; this repo does not include a built-in viewer

## Examples

- Process a single file to OBJ + GLB:
```bash
ADTPreFabTool.Console "test_data/Kalidar/Kalidar_30_23.adt" output2a --glb
```

- Process a directory using defaults (recursive + per-file GLBs + manifest):
```bash
ADTPreFabTool.Console "test_data/Kalidar" output3
```

- Process a directory non-recursively and opt out of per-file GLBs and manifest:
```bash
ADTPreFabTool.Console "test_data/Kalidar" output3 --no-recursive --no-glb-per-file --no-manifest
```

- Decode minimaps with TRS and auto-detected WMO root, export CSV index:
```bash
dotnet run --project src/ADTPreFabTool.Console/ADTPreFabTool.Console.csproj \
  "test_data/0.6.0/tree/World/Maps/Azeroth" \
  "./project_output" \
  --minimap-root "test_data/0.6.0/tree/textures/Minimap" \
  --trs "test_data/0.6.0/tree/textures/Minimap/md5translate.txt" \
  --data-version 0.6.0 \
  --cache-root "./cachedMinimaps" \
  --decode-minimap \
  --export-minimap-overlays
```

- Decode minimaps with explicit WMO root:
```bash
dotnet run --project src/ADTPreFabTool.Console/ADTPreFabTool.Console.csproj \
  "test_data/0.6.0/tree/World/Maps/Azeroth" \
  "./project_output" \
  --minimap-root "test_data/0.6.0/tree/textures/Minimap" \
  --trs "test_data/0.6.0/tree/textures/Minimap/md5translate.txt" \
  --world-minimap-root "test_data/0.6.0/tree/World/Minimaps" \
  --data-version 0.6.0 \
  --cache-root "./cachedMinimaps" \
  --decode-minimap \
  --export-minimap-overlays
```

## Viewing Outputs

- Open `.glb` in Windows 3D Viewer, Blender, Babylon Sandbox, or any GLB-capable viewer.
- Import `.obj` in Blender, MeshLab, or similar tools.

## Implementation Notes

- Code paths:
  - `Program.ProcessADTFile(...)` for single file
  - `Program.ProcessADTDirectory(...)` for directory
  - OBJ writer uses wow.export triangulation patterns
  - GLB export uses SharpGLTF Toolkit (`BuildGlbForADT`)
- Coordinate system is Z-up in both OBJ/GLB and the manifest.
