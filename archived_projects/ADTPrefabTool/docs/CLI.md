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
  [--export-minimap-overlays] [--export-minimap-obj] [--export-minimap-glb] [--yflip|--no-yflip]
  [--export-merged]
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
- `--export-minimap-obj`          Generate per-tile OBJ/MTL textured with cached minimap PNG, using the real ADT terrain mesh. UVs are slightly inset to minimize tile seams.
- `--export-minimap-glb`          Generate one GLB per tile using the real ADT terrain mesh; each MCNK chunk is a separate node. Uses the tile's minimap PNG as BaseColor with a clamped sampler; UVs are slightly inset to minimize tile seams.
- `--yflip | --no-yflip`          Flip V to match minimap orientation (default: `--yflip`); use `--no-yflip` to disable.
- `--export-merged`              Generate one merged OBJ+MTL and one merged GLB containing all selected tiles.

### Per‑tile GLB Export (`--export-minimap-glb`)
{{ ... }}
- `--tiles X_Y,...` or `--tile-range x1,y1,x2,y2` to limit tiles.

Notes:
**Texturing & seam mitigation**
- BaseColor is sourced from the tile's minimap PNG.
- Texture sampler is set to CLAMP_TO_EDGE (GLB) to avoid sampling outside tile textures.
- UVs are slightly inset by 0.0025 to reduce bilinear bleed along tile borders (applies to OBJ and GLB paths).

Example:
```bash
{{ ... }}
```

### Merged Minimap OBJ Export (`--export-merged`)
+
+Generates a single OBJ+MTL containing all selected tiles. Each tile is a separate group and uses a distinct material referencing its minimap PNG placed next to the merged OBJ.
+
+Seam mitigation:
+- UVs inset by 0.0025 to reduce filtering bleed across tile edges.
+
+Flips:
+- Defaults: `xFlip=true`, `yFlip=true` unless overridden by `--xflip/--no-xflip` and `--yflip/--no-yflip`.
+
+Selection:
+- Use `--tiles X_Y,...` and/or `--tile-range x1,y1,x2,y2`.
+
+Example:
+```bash
+dotnet run --project src/ADTPreFabTool.Console/ADTPreFabTool.Console.csproj \
+  "test_data/0.6.0/tree/World/Maps/Azeroth" \
+  --minimap-root "test_data/0.6.0/tree/textures/Minimap" \
+  --data-version 0.6.0 \
+  --cache-root "./cachedMinimaps" \
+  --decode-minimap \
+  --export-merged \
+  --tile-range 30,42,32,43
+```
+
+### Merged Minimap GLB Export (`--export-merged`)
+
+Generates a single GLB scene containing all selected tiles. Each tile is a parent node (`Map_tx_ty`) with up to 256 child nodes (one per MCNK chunk) using that tile’s minimap texture as BaseColor.
+
+Seam mitigation:
+- Texture sampler uses CLAMP_TO_EDGE.
+- UVs inset by 0.0025 to reduce filtering bleed.
+
+Flips:
+- Defaults: `xFlip=true`, `yFlip=false` for GLB unless overridden.
+
+Selection:
+- Use `--tiles X_Y,...` and/or `--tile-range x1,y1,x2,y2`.
+
+Example:
+```bash
+dotnet run --project src/ADTPreFabTool.Console/ADTPreFabTool.Console.csproj \
+  "test_data/0.6.0/tree/World/Maps/Azeroth" \
+  --minimap-root "test_data/0.6.0/tree/textures/Minimap" \
+  --data-version 0.6.0 \
+  --cache-root "./cachedMinimaps" \
+  --decode-minimap \
+  --export-merged \
+  --tiles 30_42,31_42
+```
{{ ... }}
