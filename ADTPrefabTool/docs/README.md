# ADTPreFabTool — Terrain Extraction, Minimap Decoding, and GLB Outputs

This repository contains tools to extract terrain meshes from WoW ADT files, decode minimap tiles (ADT and WMO), and export standard assets (OBJ/GLB) along with a lightweight per-file manifest for downstream tools.

## Prerequisites

- .NET SDK 9.0 or later
- Windows recommended

## Setup

This project depends on external libraries that must be cloned as submodules.

From the repository root:

```bash
# Clone the required repositories into the lib/ directory
git clone https://github.com/Marlamin/wow.tools.local lib/wow.tools.local
git clone https://github.com/Kruithne/wow.export lib/wow.export

# Initialize and update submodules for wow.tools.local
cd lib/wow.tools.local
git submodule init
git submodule update
cd ../..
```

## Features

- Extracts ADT terrain to OBJ and optional GLB/GLTF
- Builds per-file and aggregated manifests for downstream workflows
- Decodes minimaps from multiple sources:
  - TRS-based tiles via `textures/Minimap/md5translate.txt`
  - Legacy ALT layout under `textures/Minimap/<MapName>/`
  - WMO tiles under `World/Minimaps/**` with preserved folder hierarchy

Uses real data files from `test_data/` for examples.

## Data Layout Overview

When exporting per-file GLBs and a manifest, the output directory contains:

- `tiles_manifest.json` — newline-delimited JSON (NDJSON), one JSON object per line
- One `.glb` file per tile

Example directories in this repo:

- `output5-glb-dev/`
- `output5-glb/`

## Manifest Format (NDJSON)

Each line is a minimal JSON object with the following fields:

- `file`: original source file name (e.g., `development_0_0.adt`) — informational
- `glb`: GLB file name (present in the same directory) — optional if you only need AABBs
- `aabb`: 6 floats `[minX, minY, minZ, maxX, maxY, maxZ]` (Z-up coordinates)

Example lines (snipped from `output5-glb-dev/tiles_manifest.json`):

```json
{"file":"development_0_0.adt","glb":"development_0_0.glb","aabb":[16550.000,16565.666,-28.948,17083.332,17067.666,123.012]}
{"file":"development_0_1.adt","glb":"development_0_1.glb","aabb":[16016.666,16565.666,-29.108,16550.000,17067.666,123.012]}
```

Notes:
- Coordinates use a Z-up world (Z is height). XY is the horizontal plane.
- NDJSON: one JSON object per line, no commas, no surrounding array.

## Building

From the repository root:

```bash
# Restore and build the console tool
dotnet build src/ADTPreFabTool.Console/ADTPreFabTool.Console.csproj -c Release
```

## Viewing Outputs

- OBJ: import into Blender, MeshLab, or any OBJ-capable tool.
- GLB: view in Windows 3D Viewer, Blender, Babylon Sandbox, or three.js editors.
- The manifest is provided for downstream processing or custom viewers; this repo no longer includes a bundled viewer.

## Minimap Cache Structure

When `--decode-minimap` is used, the cache is populated at:

```
<cache_root>/_cache/<data-version>/minimap_png/
```

- ADT tiles (TRS / legacy ALT):
  - `<MapName>/<MapName>_<X>_<Y>.png`
  - Alternate differing content: `<MapName>/<MapName>_<X>_<Y>__alt.png`

- WMO tiles (preserved hierarchy):
  - `wmo/<relative_path_under_World/Minimaps>/<original_stem>.png`
  - Example: `wmo/transports/passengership/passengership_00_00_00.png`

CSV `minimap_index.csv` (written to each run folder when `--export-minimap-overlays`) includes columns:

```
mapName,tileX,tileY,source_kind,duplicate_of,wmo_asset,content_md5,blp_path,png_path
```

Notes:
- ADT: de-dup precedence TRS > ALT for identical (MapName,X,Y). Differing content produces `__alt`.
- WMO: every file path becomes its own PNG; we only skip if that exact output already exists.

## Quick Start (real test data)

Prepare minimaps and export overlays:

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

Explicit WMO root:

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

Typical OBJ/GLB extraction remains the same; see `docs/CLI.md` for flags.

## Future Improvements

- Additional export options and metadata.
- Expanded manifest schema as needed by downstream tools.
