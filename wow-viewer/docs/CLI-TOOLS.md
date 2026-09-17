# CLI Tools Comprehensive Reference Guide

**Toolkit**: `wow-viewer/tools/`  
**Execution Pattern**: `dotnet run --project wow-viewer/tools/<tool-dir>/<project>/<project>.csproj -c Debug -- <command> [options]`

> [!IMPORTANT]
> The canonical current CLI entry point is [`../tools/README.md`](../tools/README.md). This file is the expanded historical command reference and can include older workflow notes; prefer the tooling README for current project paths, boundaries, and safety rules.

---

## Table of Contents
1. [Format Inspector (`wowviewer-inspect`)](#1-format-inspector-wowviewer-inspect)
   - [Rosetta Calibration Corpus Generator (`rosetta-generate`)](#rosetta-calibration-corpus-generator-rosetta-generate)
   - [M2 & MDX Model Inspection](#m2--mdx-model-inspection)
   - [PM4 Geometry & Reconciliation Tools](#pm4-geometry--reconciliation-tools)
   - [Terrain & Map Inspection (`map inspect`)](#terrain--map-inspection-map-inspect)
   - [WMO World Model Inspection](#wmo-world-model-inspection)
   - [BLP Texture Inspection](#blp-texture-inspection)
   - [Lighting Inspection (`lit` & `light`)](#lighting-inspection-lit--light)
   - [Audio Catalog Inspection (`audio alpha-area`)](#audio-catalog-inspection-audio-alpha-area)
   - [Archive & Listfile Caching](#archive--listfile-caching)
   - [CASC Installs (`casc`)](#casc-installs-casc)
   - [DAT v26 Terrain (`adt-ahdr`)](#dat-v26-terrain-adt-ahdr)
2. [Format Converter (`wowviewer-converter`)](#2-format-converter-wowviewer-converter)
   - [Alpha WDT → LK Format (`alpha-to-lk`)](#alpha-wdt--lk-format-alpha-to-lk)
   - [LK Format → Alpha WDT (`lk-to-alpha`)](#lk-format--alpha-wdt-lk-to-alpha)
3. [Terrain Tensor Harvester (`wowviewer-harvest`)](#3-terrain-tensor-harvester-wowviewer-harvest)
   - [Harvesting Maps to NPZ / Zarr](#harvesting-maps-to-npz--zarr)
   - [Synthesizing Minimaps (`synthetic-minimap`)](#synthesizing-minimaps-synthetic-minimap)
   - [CASC Clients in Harvest](#casc-clients-in-harvest)
   - [Streaming Pipe to Python (`harvest-stream`)](#streaming-pipe-to-python-harvest-stream)
4. [Python ML Toolchain (`data-harvester`)](#4-python-ml-toolchain-data-harvester)

---

## 1. Format Inspector (`wowviewer-inspect`)

**Project**: `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj`

### Rosetta Calibration Corpus Generator (`rosetta-generate`)
Generates synthetic, fully-labeled ADT and WDT map tiles placing every enumerable model/WMO on a museum pedestal with high-resolution antialiased MCAL/MCLY terrain labels.

```powershell
dotnet run --project wow-viewer/tools/inspect/WowViewer.Tool.Inspect -c Debug -- `
  rosetta-generate `
  --client-root "H:\CLIENTS\WoW-0.5.3.3368-Client" `
  --output "output/rosetta_alpha" `
  --map-name "RosettaAlpha" `
  --pedestal-height 4.0 `
  --pedestal-bevel 12.5 `
  --ground-texture "tileset\ocean\westfallseafloor.blp" `
  --ink-texture "tileset\generic\black.blp"
```

#### Options Reference:
| Option | Default | Description |
|---|---|---|
| `--client-root <dir>` | *(Required)* | Root directory containing client MPQ archives or loose files |
| `--output <dir>` | *(Required)* | Output root directory where `World\Maps\<MapName>\` is created |
| `--map-name <name>` | `Development` | Map identifier for generated WDT/ADTs |
| `--format <lk\|alpha>` | *Auto-detected* | Target era: `alpha` (monolithic WDT) or `lk` (standalone ADT files) |
| `--ground-texture <path>` | *Auto-selected* | Layer 0 base terrain texture (e.g. `tileset\ocean\westfallseafloor.blp`) |
| `--ink-texture <path>` | `tileset\generic\black.blp` | Layer 1 text ink texture for MCAL antialiased labels |
| `--pedestal-height <m>` | `4.0` | Museum plinth elevation in meters above terrain floor |
| `--pedestal-bevel <m>` | `12.5` | Width of the transition bevel ramp in meters |
| `--cell-chunks <1\|2\|4\|8\|16>` | `4` | Width/height of each grid cell in ADT chunks (4 chunks = 133.3m) |
| `--label-band-chunks <n>` | `1` | Number of chunks reserved for the label band along cell bottom |
| `--no-cell-borders` | `false` | Disables vertex border outlines around cells |
| `--no-designkit-grouping` | `false` | Disables grouping assets by folder/kit |
| `--kit-depth <n>` | `0` | Coarsens kit grouping (e.g. `1` groups all `creature\*` into one kit) |
| `--max-tiles-per-map <n>` | `4096` | Max tiles allowed before splitting into `<MapName>00`, `<MapName>01` |
| `--max-assets <n>` | `0` (all) | Limit total assets placed (useful for quick smoke testing) |
| `--existing-map-dir <dir>` | `null` | Reference map directory to reserve occupied tile coordinates |
| `--pm4-dir <dir>` | `null` | Directory containing PM4 guides to copy alongside output tiles |
| `--overwrite` | `false` | Overwrites existing output folders without prompting |

---

### M2 & MDX Model Inspection

```powershell
# Inspect M2 header, bounding box, sequence count, bones, attachments
dotnet run --project wow-viewer/tools/inspect/WowViewer.Tool.Inspect -c Debug -- `
  m2 inspect --input "Creature/Arthas/Arthas.m2"

# Inspect M2 with a specific skin profile
dotnet run --project wow-viewer/tools/inspect/WowViewer.Tool.Inspect -c Debug -- `
  m2 inspect --input "Creature/Arthas/Arthas.m2" --profile-index 0

# Sample an animated pose at sequence 2, time 500ms and export golden JSON
dotnet run --project wow-viewer/tools/inspect/WowViewer.Tool.Inspect -c Debug -- `
  m2 inspect --input "Creature/Murloc/Murloc.m2" `
  --sequence-index 2 --time-ms 500 --golden-output "output/murloc_pose.json"

# Inspect an Alpha-era MDX model
dotnet run --project wow-viewer/tools/inspect/WowViewer.Tool.Inspect -c Debug -- `
  mdx inspect --input "World/Generic/Human/PassiveDoodads/Beds/Bed01.mdx"

# Export full MDX document tree (geometry, materials, bones, collision) to JSON
dotnet run --project wow-viewer/tools/inspect/WowViewer.Tool.Inspect -c Debug -- `
  mdx export-json --input "Creature/Wolf/Wolf.mdx" --output "wolf_report.json" `
  --include-geometry --include-collision --include-hit-test
```

---

### PM4 Geometry & Reconciliation Tools

```powershell
# Inspect PM4 chunk inventory (MSCN, MSPV, MSUR, MSHD, MSLK)
dotnet run --project wow-viewer/tools/inspect/WowViewer.Tool.Inspect -c Debug -- `
  pm4 inspect --input "World/Maps/Azeroth/Azeroth_32_48.pm4"

# Export geometry segments and object clusters from PM4
dotnet run --project wow-viewer/tools/inspect/WowViewer.Tool.Inspect -c Debug -- `
  pm4 export-segments --input "World/Maps/Azeroth/Azeroth_32_48.pm4" --output "segments.json"

# Export asset reference signal corpus from seed ADT placements
dotnet run --project wow-viewer/tools/inspect/WowViewer.Tool.Inspect -c Debug -- `
  pm4 export-asset-signals `
  --archive-root "H:\CLIENTS\World of Warcraft 3.3.5a" `
  --seed-placements "World/Maps/Azeroth/Azeroth_32_48_obj0.adt" `
  --kind all --output "corpus.json"

# Match PM4 segments against reference asset corpus
dotnet run --project wow-viewer/tools/inspect/WowViewer.Tool.Inspect -c Debug -- `
  pm4 match-assets `
  --input "World/Maps/Azeroth/Azeroth_32_48.pm4" `
  --asset-corpus "corpus.json" --output "match_report.json"
```

---

### Terrain & Map Inspection (`map inspect`)

```powershell
# Inspect single ADT tile or WDT file from disk
dotnet run --project wow-viewer/tools/inspect/WowViewer.Tool.Inspect -c Debug -- `
  map inspect --input "World/Maps/Azeroth/Azeroth_32_48.adt"

# Survey and summarize every WDT map discovered inside a client MPQ archive
dotnet run --project wow-viewer/tools/inspect/WowViewer.Tool.Inspect -c Debug -- `
  map inspect --archive-root "H:\CLIENTS\WoW-0.5.3.3368-Client"

# Generate a blank valid ADT tile (for terrain authoring or repair)
dotnet run --project wow-viewer/tools/inspect/WowViewer.Tool.Inspect -c Debug -- `
  map generate-blank --tile-x 32 --tile-y 48 --map-name "TestMap" --format lk `
  --texture "tileset\grass\greensward.blp" --output-dir "output/blank_maps"
```

---

### WMO World Model Inspection

```powershell
# Inspect WMO root header, group count, materials, and bounding boxes
dotnet run --project wow-viewer/tools/inspect/WowViewer.Tool.Inspect -c Debug -- `
  wmo inspect --input "World/wmo/Dungeon/AZ_Subway/Subway.wmo"

# Dump internal light definitions (MOLT chunks)
dotnet run --project wow-viewer/tools/inspect/WowViewer.Tool.Inspect -c Debug -- `
  wmo inspect --input "World/wmo/Dungeon/AZ_Subway/Subway.wmo" --dump-lights
```

---

### BLP Texture Inspection

```powershell
# Inspect image dimensions, mipmap levels, and compression format (DXT1/DXT3/DXT5/Raw)
dotnet run --project wow-viewer/tools/inspect/WowViewer.Tool.Inspect -c Debug -- `
  blp inspect --input "tileset/ocean/westfallseafloor.blp"

# Read directly from an MPQ client archive via virtual path
dotnet run --project wow-viewer/tools/inspect/WowViewer.Tool.Inspect -c Debug -- `
  blp inspect --archive-root "H:\CLIENTS\World of Warcraft 3.3.5a" `
  --virtual-path "tileset/ocean/westfallseafloor.blp"
```

---

### Lighting Inspection (`lit` & `light`)

```powershell
# Inspect Alpha .lit lighting file
dotnet run --project wow-viewer/tools/inspect/WowViewer.Tool.Inspect -c Debug -- `
  lit inspect --input "World/Maps/Kalimdor/Kalimdor.lit"

# Evaluate lighting parameters at specific times of day (0.0 = midnight, 0.5 = noon)
dotnet run --project wow-viewer/tools/inspect/WowViewer.Tool.Inspect -c Debug -- `
  lit profile --input "World/Maps/Kalimdor/Kalimdor.lit" `
  --game-time 0.0,0.25,0.5,0.75 --output "lighting_profile.json"
```

---

### Audio Catalog Inspection (`audio alpha-area`)

```powershell
# Inspect Alpha 0.5.3 AreaMIDIAmbiences catalog and join against AreaTable.dbc
dotnet run --project wow-viewer/tools/inspect/WowViewer.Tool.Inspect -c Debug -- `
  audio alpha-area --archive-root "H:\CLIENTS\WoW-0.5.3.3368-Client" `
  --build "0.5.3.3368" --limit 20
```

---

### Archive & Listfile Caching

```powershell
# Build fast JSON listfile cache from an archive directory
dotnet run --project wow-viewer/tools/inspect/WowViewer.Tool.Inspect -c Debug -- `
  archive build-listfile-cache `
  --archive-root "H:\CLIENTS\World of Warcraft 3.3.5a" `
  --cache-key "3.3.5.12340"
```

### CASC Installs (`casc`)

Added in v0.5.4-alpha. Reads a local CASC install through TACTSharp. `--install` is the folder containing
`.build.info`; `--cache` is a local folder for decoded manifests and CDN downloads. `--listfile` takes
community listfile CSVs (`id;path`) and may be repeated.

```powershell
$exe = 'wow-viewer\tools\inspect\WowViewer.Tool.Inspect\bin\Release\net10.0\WowViewer.Tool.Inspect.exe'
$install = 'D:\Games\World of Warcraft'
$listfile = "$env:LOCALAPPDATA\WoWViewer\community-listfile-withcapitals.csv"

# Products and builds in .build.info
& $exe casc products --install $install

# Read one file by FileDataID or listfile path (--cdn-fill downloads data the install lacks)
& $exe casc read --install $install --product wow_classic_beta --cache output\casc-cache --id 857684 --out output\development.wdt
& $exe casc read --install $install --product wow_classic_beta --cache output\casc-cache `
  --path 'World\Maps\Azeroth\Azeroth_31_40.adt' --listfile $listfile --out output\Azeroth_31_40.adt --cdn-fill

# Which listed paths are readable (one product per line: NotPresent/NotLocal/KeyUnavailable/Failed counts;
# two or more --product values report which product serves each path)
& $exe casc exists --install $install --product wow_classic_beta --product wow_classic_era --cache output\casc-cache `
  --paths-file paths.txt --listfile $listfile [--show-missing]

# Parse a WMO (groups via GFID, textures/doodads via FileDataIDs) and report resolution
& $exe casc wmo --install $install --product wow_classic_beta --cache output\casc-cache --listfile $listfile `
  --path 'WORLD\WMO\AZEROTH\BUILDINGS\CHAPEL\DUSKWOODCHAPEL.WMO'

# Build chunked M2s on the native render path (SFID skin, TXID textures)
& $exe casc m2 --install $install --product wow_classic_beta --cache output\casc-cache --listfile $listfile `
  --path 'WORLD\LORDAERON\SILVERPINE\PASSIVEDOODADS\TREES\SILVERPINETREE03.M2'

# Decode DB2 tables with WoWDBDefs for the product's own build
& $exe casc db2 --install $install --product wow_classic_beta --cache output\casc-cache `
  --defs wow-viewer\libs\wowdev\WoWDBDefs\definitions --listfile $listfile --table Map --table AreaTable

# Survey a map by WDT FileDataID: MAID tiles, MDID/MHID, MCLY/MCAL, placement flags,
# WMO parse outcomes, native M2 build outcomes, asset readability
& $exe casc map-survey --install $install --product wow_classic_beta --cache output\casc-cache --wdt-id 857684 [--cdn-fill]

# Export absolute ADT outer-vertex heights for a tile rectangle (CSV: tileX,tileY,row,col,height)
& $exe casc adt-heights --install $install --product wow_classic_beta --cache output\casc-cache `
  --wdt-id 775971 --x0 24 --x1 38 --y0 14 --y1 30 --out output\azeroth_heights.csv

# Loading benchmark: listfile load, file-list filtering, sequential vs parallel reads (with a
# byte-identity check), WMO parse, WMO texture fetch and decode
& $exe casc bench --install $install --product wow_classic_beta --cache output\casc-cache --wdt-id 857684 `
  --listfile $listfile [--threads 12] [--cdn-fill]
```


Search DB2 tables for integer values (every column, arrays element-wise; also prints the nearest row IDs):

```powershell
& $exe casc db2 --install "I:\wow12\World of Warcraft" --product wow_classic_beta --cache output\cache\casc `
  --defs libs\wowdev\WoWDBDefs\definitions --listfile $env:LOCALAPPDATA\WoWViewer\community-listfile-withcapitals.csv `
  --table Map --table AreaTable --find 2869
```

### DAT v26 Terrain (`adt-ahdr`)

```powershell
# Decode every AHDR-family file in a folder (any names), check ACNK indices, and verify that
# ALOC-adjacent tiles share identical edge heights
& $exe adt-ahdr check --root test_data\v22_adts\unknown

# Decode and re-encode every file with the experimental writer; reports byte-identical files
& $exe adt-ahdr roundtrip --root test_data\v22_adts\unknown

# Resolve ACDO object placements and measure their heights against the terrain (add --list to print each)
& $exe adt-ahdr objects --root test_data\v22_adts\unknown
```

---

## 2. Format Converter (`wowviewer-converter`)

**Project**: `wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj`

### Alpha WDT → LK Format (`convert-alpha-to-lk`)
Converts pre-release Alpha 0.5.3 monolithic WDT maps into standard modern Wrath of the Lich King (LK) ADT files, standalone WDT headers, and WDL terrain horizon files.

```powershell
dotnet run --project wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -c Debug -- `
  convert-alpha-to-lk `
  --input "H:\CLIENTS\WoW-0.5.3.3368-Client\World\Maps\Kalimdor\Kalimdor.wdt" `
  --output "output/converted/Kalimdor_LK"
```

### LK Format → Alpha WDT (`convert-lk-to-alpha`)
Converts standard modern ADT/WDT directories into an Alpha 0.5.3 monolithic WDT container with embedded MCNK terrain, MCVT heightmaps, and MCAL alpha blending.

```powershell
dotnet run --project wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -c Debug -- `
  convert-lk-to-alpha `
  --input "output/rosetta_lk/World/Maps/RosettaLK" `
  --output "output/converted/RosettaAlpha_Monolith.wdt"
```

---

## 3. Terrain Tensor Harvester (`wowviewer-harvest`)

**Project**: `wow-viewer/tools/harvest/WowViewer.Tool.Harvest/WowViewer.Tool.Harvest.csproj`

### Harvesting Maps to NPZ / Zarr

```powershell
# Harvest single map from loose ADT folder
dotnet run --project wow-viewer/tools/harvest/WowViewer.Tool.Harvest -c Debug -- `
  harvest-map --input-dir "C:\Extracted\World\Maps\Azeroth" --output "output/tensors/Azeroth.npz"

# Harvest map directly from MPQ client archives
dotnet run --project wow-viewer/tools/harvest/WowViewer.Tool.Harvest -c Debug -- `
  harvest-map-mpq --client-root "H:\CLIENTS\World of Warcraft 3.3.5a" `
  --map-name "Azeroth" --output "output/tensors/Azeroth.npz"

# Build complete multi-build Zarr dataset
dotnet run --project wow-viewer/tools/harvest/WowViewer.Tool.Harvest -c Debug -- `
  harvest-dataset --client-root "H:\CLIENTS\World of Warcraft 3.3.5a" `
  --builds "3.3.5.12340" --output "output/datasets/warcraft_terrain.zarr"
```

### Synthesizing Minimaps (`synthetic-minimap`)
Composes paired terrain and liquid minimap images directly from raw ADT elevation, vertex colors, and alpha blend layers without requiring shipped minimap graphics.

```powershell
dotnet run --project wow-viewer/tools/harvest/WowViewer.Tool.Harvest -c Debug -- `
  synthetic-minimap `
  --client-root "H:\CLIENTS\World of Warcraft 3.3.5a" `
  --map Azeroth `
  --output-dir "output/synthetic_minimaps/Azeroth"
```

### CASC Clients in Harvest
Added in v0.5.4-alpha. Every `--client-root` command opens a CASC install when the folder (or its
parent) contains `.build.info`, and MPQ archives otherwise. FileDataID-era maps resolve their tile
files and authored minimaps through the WDT `MAID` chunk, and split `_tex0.adt` textures through
`MDID`. Tested with `synthetic-minimap` on `wow_classic_beta` 1.60.1.69876 (`development`).

| Option | Description |
|---|---|
| `--casc-product <a,b>` | Products to read, newest version first (default: all listed) |
| `--casc-listfile <csv>` | Community listfile (`id;path`). Default: `%LOCALAPPDATA%\WoWViewer\community-listfile-withcapitals.csv`, which the viewer downloads |
| `--cdn-fill` | Fetch files the install lists but has no local data for, from Blizzard's CDN for the same build |
| `--casc-cache <dir>` | CDN download cache (default `output/cache/casc`) |

```powershell
dotnet run --project wow-viewer/tools/harvest/WowViewer.Tool.Harvest -c Debug -- `
  synthetic-minimap `
  --client-root "I:\wow12\World of Warcraft" `
  --casc-product wow_classic_beta --cdn-fill `
  --map development `
  --output-dir "output/synthetic_minimaps/development" `
  --per-tile --whole-map --authored-reference
```

Texture slots whose `MDID` entry is 0 have no texture; the compositor substitutes a recorded RGB proxy
and logs a warning for each.

### Streaming Pipe to Python (`harvest-stream`)
Streams binary V22 tile blobs to standard output for direct consumption by Python data pipelines.

```powershell
cmd /c "dotnet run --project wow-viewer/tools/harvest/WowViewer.Tool.Harvest -c Debug -- harvest-stream --stream-profile v22 --client-root ""H:\CLIENTS\World of Warcraft 3.3.5a"" --map Azeroth --limit 10 1> output\tmp\v22_stream.bin 2> output\tmp\v22_stream.log"
```

---

## 4. Python ML Toolchain (`data-harvester`)

Located in `wow-viewer/data-harvester/`, this Python environment manages dataset generation and ML training workflows.

```powershell
cd wow-viewer/data-harvester

# Build V22 Zarr dataset from game client in a single command
uv run python scripts/build_v22_dataset.py harvest-build `
  --client-root "H:\CLIENTS\World of Warcraft 3.3.5a" `
  --map Azeroth --limit 50 --output "../output/datasets/v22/Azeroth_sample.zarr"

# Inspect dataset summary
uv run python scripts/inspect_v22_dataset.py summary `
  --store "../output/datasets/v22/Azeroth_sample.zarr"

# Dump single tile tensor metadata to JSON
uv run python scripts/inspect_v22_dataset.py tile `
  --store "../output/datasets/v22/Azeroth_sample.zarr" --tile-index 0 --output-json "tile_0.json"
```
