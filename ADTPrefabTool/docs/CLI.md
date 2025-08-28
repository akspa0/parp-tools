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
ADTPreFabTool.Console <adt_file_or_folder_path> [output_directory] [--recursive|--no-recursive] [--no-comments] [--glb] [--gltf] [--glb-per-file|--no-glb-per-file] [--manifest|--no-manifest]
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

Notes:
- `--glb`/`--gltf` apply to single-file mode (`ProcessADTFile`).
- `--glb-per-file`/`--manifest` apply to directory mode (`ProcessADTDirectory`).
- Defaults (directory input): `--recursive --glb-per-file --manifest`.

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
