# WoWRollback - World of Warcraft Map Analysis & Rollback Toolkit

**Digital archaeology + conversion toolkit** focused on:

## Concise Guide

### 1) Lightweight Static Viewer (WoWDataPlot)
- Generate a static, zero-dependency viewer with overlays in seconds.
```powershell
dotnet run --project WoWDataPlot -- visualize \
  --wdt ..\test_data\0.5.3\tree\World\Maps\Kalidar\Kalidar.wdt \
  --output-dir .\Kalidar_output
```

---

## End-to-End Pipeline (Recommended Read)

For a step-by-step guide from Alpha WDT → UniqueID CSVs → crosswalks → LK ADTs (+WDT) → Viewer, see:

- `docs/pipeline/alpha-to-lk-end-to-end.md`
See: `WoWDataPlot/README.md`.

### 2) Interactive Viewer (Analyze + Serve)
- Analyze loose ADTs and start the built-in server for browsing.
```powershell
dotnet run --project WoWRollback.Cli -- analyze-map-adts \
  --map development \
  --input-dir "..\test_data\development\World\Maps\development" \
  --out analysis_output

dotnet run --project WoWRollback.Cli -- serve-viewer --viewer-dir analysis_output\viewer
```
See: `WoWRollback.Cli/README.md`, `WoWRollback.ViewerModule/README.md`.

### 3) Alpha ↔ LK ADT Conversion
- Alpha → LK ADTs (burial, holes, MCSH, AreaIDs):
```powershell
dotnet run --project WoWRollback.Cli -- alpha-to-lk \
  --input ..\test_data\0.5.3\tree\World\Maps\Azeroth\Azeroth.wdt \
  --max-uniqueid 43000 --fix-holes --disable-mcsh \
  --out wrb_out \
  --lk-out wrb_out\lk_adts\World\Maps\Azeroth \
  --lk-client-path "J:\\wowDev\\modernwow" --default-unmapped 0
```
- LK → Alpha patcher (apply same terrain logic):
```powershell
dotnet run --project WoWRollback.Cli -- lk-to-alpha \
  --lk-adts-dir .\wrb_out\lk_adts\World\Maps\Azeroth \
  --map Azeroth \
  --max-uniqueid 43000 --fix-holes --disable-mcsh \
  --out .\patched_lk_az
```
See: `WoWRollback.Cli/README.md`, `WoWRollback.AdtConverter/README.md`.

For module-specific details, see the Subprojects section below.

---

## 🚀 Quick Start

### Static Visualization Tool (WoWDataPlot - New!)

**Lightweight static HTML viewer for rapid data exploration** - no server needed after generation!

```powershell
cd WoWRollback

# Generate static visualization with minimap overlays
dotnet run --project WoWDataPlot -- visualize \
  --wdt ..\test_data\0.5.3\tree\World\Maps\Kalidar\Kalidar.wdt \
  --output-dir .\Kalidar_output \
  --gap-threshold 50

# Open the generated HTML files in any browser
# Start with: Kalidar_output\Kalidar_legend.html
```

**What you get:**
- ✅ **Per-tile minimap overlays** - Transparent PNG layers showing object placements
- ✅ **Interactive layer toggles** - Show/hide UniqueID ranges on-the-fly
- ✅ **Debug controls** - Flip X/Y, swap axes for coordinate troubleshooting
- ✅ **Continental overview** - Heatmap showing all layers across entire map
- ✅ **Zero server requirement** - Pure static HTML/JS/CSS files
- ✅ **Fast generation** - Kalidar (13 tiles) processes in ~5 seconds

**Key features:**
- Placement dots align 1:1 with minimap pixels (coordinate fix applied)
- Global UniqueID gradient coloring (blue=early, red=late)
- Layer detection with automatic gap-based splitting (configurable threshold)
- Analysis JSON with detailed statistics per tile

See [WoWDataPlot Documentation](#wowdataplot---static-visualization) for full details.

---

### Analyze Loose ADT Files (Dynamic Viewer)

**The fastest way to explore your map data with interactive server:**

```powershell
cd WoWRollback

# Step 1: Analyze ADT files
dotnet run --project WoWRollback.Cli -- analyze-map-adts \
  --map development \
  --input-dir "..\test_data\development\World\Maps\development\" \
  --out "analysis_output"

# Step 2: Start built-in web server (auto-detects viewer location)
dotnet run --project WoWRollback.Cli -- serve-viewer

# Opens browser at http://localhost:8080 automatically!
```

### WoWRollback.AdtConverter (Standalone CLI)

`WoWRollback.AdtConverter` now ships inside `WoWRollback.sln` and builds with the rest of the toolkit. Run it directly for LK ⇄ Alpha terrain workflows:

```powershell
# Optional: build the converter by itself
dotnet build WoWRollback.AdtConverter/WoWRollback.AdtConverter.csproj --no-incremental

# Pack a monolithic Alpha WDT with embedded terrain-only ADTs
dotnet run --project WoWRollback.AdtConverter -- pack-monolithic \
  --lk-dir <path-to-lk-map-dir> \
  --lk-wdt <path-to-lk-wdt> \
  --map <MapName> \
  [--out <output-root>] \
  [--force-area-id <id>] \
  [--main-point-to-data] \
  [--verbose-logging]
```

- **Output** defaults to `project_output/<map>_<timestamp>/` with the packed `<map>.wdt`.
- **Verbose logging** (`--verbose-logging`) writes LK vs Alpha `MCAL` dumps to `debug_mcal/YY_XX/` for troubleshooting mask ordering.
- **More commands**: `dotnet run --project WoWRollback.AdtConverter -- --help` lists `convert-wdt`, `convert-map-terrain`, `inspect-alpha`, `compare-alpha`, `validate-wdt`, `unpack-monolithic`, and `alpha-to-lk-mcse`.

**What you get:**
- ✅ 26K+ M2/WMO placements extracted & overlaid on minimaps
- ✅ MCNK terrain data (AreaIDs, flags, liquids, holes)
- ✅ 3D terrain meshes (GLB) for each tile
- ✅ Spatial clusters showing prefabs & object groups
- ✅ UniqueID analysis with layer detection
- ✅ Interactive viewer with zoom, pan, object details
- ✅ Cross-tile duplicate filtering (clean data!)

---

### Alpha→LK Conversion Pipeline (Original)

### 1. Organize Your Data

Your data should follow the **standard input layout** (see [DBCTool.V2/docs/input-data-prep.md](../DBCTool.V2/docs/input-data-prep.md)):

```
test_data/
├── 0.5.3/
│   └── tree/
│       ├── DBFilesClient/           # Alpha DBCs
│       │   ├── AreaTable.dbc
│       │   └── Map.dbc
│       └── World/Maps/
│           ├── Shadowfang/Shadowfang.wdt
│           ├── Azeroth/Azeroth.wdt
│           └── Kalimdor/Kalimdor.wdt
├── 0.5.5/
│   └── tree/ (same structure)
└── 3.3.5/
    └── tree/
        └── DBFilesClient/           # LK DBCs (target)
            ├── AreaTable.dbc
            └── Map.dbc
```

### 2. Run the Pipeline (One Command!)

```powershell
cd WoWRollback
dotnet run --project WoWRollback.Orchestrator -- \
  --maps Shadowfang \
  --versions 0.5.3 \
  --alpha-root ..\test_data \
  --lk-dbc-dir ..\test_data\3.3.5\tree\DBFilesClient
```

**With viewer server:**
```powershell
dotnet run --project WoWRollback.Orchestrator -- \
  --maps Shadowfang \
  --versions 0.5.3 \
  --alpha-root ..\test_data \
  --lk-dbc-dir ..\test_data\3.3.5\tree\DBFilesClient \
  --serve --port 8080
```

### 3. What Happens

The unified orchestrator runs **4 stages sequentially**:

```
[1/4] DBC Stage
  ✓ Dumps AreaTable CSVs from Alpha and LK DBCs
  ✓ Generates area hierarchy crosswalks (v2 + v3)
  ✓ Creates maps.json metadata

[2/4] ADT Conversion Stage
  ✓ Scans Alpha WDT for tile references
  ✓ Converts Alpha ADTs → LK format
  ✓ Patches AreaIDs using crosswalk mappings
  ✓ Applies asset fuzzy-matching and fallbacks

[3/4] Analysis Stage (Coming Soon)
  ✓ Extracts UniqueID distributions (time-travel CSVs)
  ✓ Generates MCNK terrain metadata CSVs
  ✓ Creates per-tile overlay JSONs
  ✓ Builds overlay manifest for viewer plugins

[4/4] Viewer Stage
  ✓ Copies static viewer assets (HTML/JS/CSS)
  ✓ Extracts minimap tiles (from loose files or MPQ archives)
  ✓ Converts BLP minimap tiles to PNG format
  ✓ Generates index.json and config.json
  ✓ Prepares overlay metadata
```

### 4. Explore Results

**Output Structure:**
```
parp_out/session_20251007_012032/
├── 01_dbcs/                    # DBC dumps
│   └── 0.5.3/raw/
│       ├── AreaTable_0_5_3.csv
│       └── AreaTable_3_3_5.csv
├── 02_crosswalks/              # Area mappings
│   └── 0.5.3/0.5.3/
│       ├── compare/v2/         # V2 crosswalks
│       └── compare/v3/         # V3 hierarchy
├── 03_adts/                    # Converted LK ADTs
│   └── 0.5.3/
│       └── World/Maps/Shadowfang/
│           ├── Shadowfang_25_30.adt
│           └── ...
├── 04_analysis/                # Analysis outputs (future)
│   └── 0.5.3/
│       ├── uniqueids/          # Time-travel CSVs
│       └── terrain/            # MCNK metadata
├── 05_viewer/                  # Web viewer
│   ├── index.html
│   ├── js/
│   ├── styles.css
│   └── overlays/
│       └── metadata.json
├── logs/                       # Per-stage logs
└── manifest.json               # Session metadata
```

**Open the viewer:**
- Without `--serve`: Use any web server: `python -m http.server 8080 --directory parp_out/session_*/05_viewer`
- With `--serve`: Automatically starts at `http://localhost:8080`

---

## 🏗️ Architecture

### Modular Design

WoWRollback follows a **clean modular architecture** with separation of concerns:

```
WoWRollback/
├── WoWRollback.Core/           # Shared utilities
│   ├── IO/FileHelpers.cs       # Directory operations
│   ├── Logging/ConsoleLogger.cs# Structured logging
│   └── Models/                 # Session metadata
│
├── WoWRollback.DbcModule/      # DBC operations (wraps DBCTool.V2)
│   ├── DbcOrchestrator.cs      # Main API
│   └── Models.cs               # Result types
│
├── WoWRollback.AdtModule/      # ADT conversion (wraps AlphaWdtAnalyzer.Core)
│   ├── AdtOrchestrator.cs      # Main API
│   └── Models.cs               # Result types
│
├── WoWRollback.AnalysisModule/ # Analysis & overlays (in progress)
│   ├── AnalysisOrchestrator.cs # Main API
│   ├── UniqueIdAnalyzer.cs     # Time-travel CSVs
│   ├── TerrainCsvGenerator.cs  # MCNK metadata
│   └── OverlayGenerator.cs     # Per-tile JSONs
│
├── WoWRollback.ViewerModule/   # Web viewer server
│   └── ViewerServer.cs         # HttpListener-based server
│
├── WoWRollback.Orchestrator/   # Pipeline coordination
│   ├── Program.cs              # CLI entry point
│   ├── PipelineOrchestrator.cs # Main pipeline
│   ├── DbcStageRunner.cs       # DBC stage
│   ├── AdtStageRunner.cs       # ADT stage
│   ├── AnalysisStageRunner.cs  # Analysis stage (future)
│   └── ViewerStageRunner.cs    # Viewer stage
│
├── WoWRollback.Viewer/         # Static viewer assets
│   └── assets/                 # HTML/JS/CSS
│
├── docs/                       # Documentation
│   ├── planning/               # Implementation plans
│   └── refactor/               # Refactor strategy docs
│
└── memory-bank/                # Project context & history
```

### Benefits

- **No shell execution** - All tools called as library APIs
- **Typed interfaces** - Structured results instead of exit codes  
- **Clean separation** - Each module has a single responsibility
- **Easy testing** - Modules can be tested independently
- **Cross-platform** - Works on Windows, Linux, macOS

### Subprojects

- **WoWRollback.Cli** — Command-line tools for analysis, conversion, serving. [README](./WoWRollback.Cli/README.md)
- **WoWRollback.Orchestrator** — One-command Alpha→LK pipeline runner. [README](./WoWRollback.Orchestrator/README.md)
- **WoWDataPlot** — Static visualization generator. [README](./WoWDataPlot/README.md)
- **WoWRollback.AdtConverter** — LK ⇄ Alpha terrain converter (standalone). [README](./WoWRollback.AdtConverter/README.md)
- **WoWRollback.AdtModule** — ADT/WDT conversion library. [README](./WoWRollback.AdtModule/README.md)
- **WoWRollback.DbcModule** — DBC parsing and crosswalk helpers. [README](./WoWRollback.DbcModule/README.md)
- **WoWRollback.AnalysisModule** — UniqueID analysis and overlays. [README](./WoWRollback.AnalysisModule/README.md)
- **WoWRollback.Core** — Shared utilities and models. [README](./WoWRollback.Core/README.md)
- **WoWRollback.Viewer** — Static web assets for the viewer. [README](./WoWRollback.Viewer/README.md)
- **WoWRollback.ViewerModule** — Kestrel server for the viewer. [README](./WoWRollback.ViewerModule/README.md)
- **WoWRollback.LkToAlphaModule** — LK↔Alpha helpers (liquids, terrain). [README](./WoWRollback.LkToAlphaModule/README.md)
- **WoWRollback.Verifier** — Integrity checks and roundtrip tests. [README](./WoWRollback.Verifier/README.md)

---

## Prerequisites

- **.NET SDK 9.0** (64-bit)
- **Alpha WoW data** - Extracted WDT/ADT/DBC files in standard layout
- **LK 3.3.5 DBCs** - AreaTable.dbc, Map.dbc for crosswalk generation
- Optional: **MPQ Archives** - For minimap extraction from compressed archives
- Optional: **WoWDBDefs** - For DBC schema definitions (auto-resolved)

---

## 📖 CLI Reference

### Analyze Loose ADTs (Primary - New!)

**Analyze ADT files without conversion** - supports pre-Cataclysm through Cataclysm+ formats:

```powershell
dotnet run --project WoWRollback.Cli -- analyze-map-adts \
  --map <name> \
  --input-dir <path> \
  [--out <dir>]
```

**What it does:**
1. **Extracts placements** - Reads MDDF/MODF chunks from `_obj0.adt` files
2. **Extracts terrain** - Reads MCNK chunks (AreaID, flags, liquids, holes)
3. **Extracts meshes** - Generates GLB 3D terrain meshes per tile
4. **Analyzes UniqueIDs** - Detects layers, gaps, ranges per tile
5. **Detects clusters** - Finds spatial object groups (prefabs/brushes)
6. **Generates viewer** - Creates interactive web viewer with overlays

**Output:**
```
analysis_output/
├── development_placements.csv          # All M2/WMO placements
├── development_terrain.csv             # MCNK terrain data
├── development_mesh/                   # 3D terrain meshes (NEW!)
│   ├── tile_30_41.glb
│   ├── tile_30_42.glb
│   └── mesh_manifest.json
├── development_uniqueID_analysis.csv   # UniqueID ranges by tile
├── development_spatial_clusters.json   # Detected object clusters
├── development_patterns.json           # Recurring patterns
├── development_cluster_summary.csv     # Cluster statistics
└── viewer/                             # Self-contained web viewer
    ├── index.html
    ├── js/
    ├── styles.css
    ├── minimap/
    │   └── analysis/development/       # WebP minimap tiles
    ├── overlays/
    │   └── analysis/development/
    │       ├── combined/               # Object overlays (per-tile JSON)
    │       ├── m2/                     # M2-only overlays
    │       ├── wmo/                    # WMO-only overlays
    │       ├── clusters/               # Cluster overlays
    │       ├── terrain_complete/       # MCNK terrain overlays
    │       └── mesh/                   # 3D terrain meshes (NEW!)
    │           ├── tile_30_41.glb
    │           └── mesh_manifest.json
    └── cached_maps/analysis/development/
        └── csv/id_ranges_by_map.csv   # For UniqueID range filtering
```

**Key Features:**
- ✅ **Cross-tile duplicate filtering** - Objects only shown on their primary tile
- ✅ **Cluster visualization** - Default overlay shows ~100 clusters instead of 28K objects
- ✅ **Coordinate system fixes** - Proper ADT placement→world transform (180° flip + axis swap)
- ✅ **WebP minimaps** - 50-70% memory savings vs PNG
- ✅ **UniqueID range loading** - Filter objects by ID ranges in viewer

---

### Serve Viewer (Built-in HTTP Server - New!)

**Self-contained web server** - no Python, Node, or external dependencies needed:

```powershell
# Auto-detect viewer location
dotnet run --project WoWRollback.Cli -- serve-viewer

# Specify directory
dotnet run --project WoWRollback.Cli -- serve-viewer \
  --viewer-dir analysis_output/viewer

# Custom port
dotnet run --project WoWRollback.Cli -- serve-viewer --port 3000

# Don't auto-open browser
dotnet run --project WoWRollback.Cli -- serve-viewer --no-browser
```

**Features:**
- ✅ Built on ASP.NET Core Kestrel (production-grade)
- ✅ Auto-detects common viewer locations
- ✅ Opens browser automatically
- ✅ Proper MIME types (.webp, .json, .geojson)
- ✅ Clean console output (only errors logged)
- ✅ Cross-platform (Windows/Linux/macOS)

**Checked locations:**
1. `analysis_output/viewer`
2. `rollback_outputs/viewer`
3. `viewer`

---

### Alpha ↔ LK ADT Workflows (WoWRollback.Cli)

Use `WoWRollback.Cli` for direct Alpha→LK export and LK→Alpha patching. These commands operate on your files without requiring the full orchestrator.

#### Alpha → LK (alpha-to-lk)

Converts a single Alpha WDT into LK ADTs with burial, optional hole clearing and shadow removal, and AreaID mapping.

```powershell
# Example (Azeroth, Alpha 0.5.3)
dotnet run -c Release --project WoWRollback.Cli -- \
  alpha-to-lk \
  --input ..\test_data\0.5.3\tree\World\Maps\Azeroth\Azeroth.wdt \
  --max-uniqueid 43000 \
  --fix-holes --disable-mcsh \
  --out wrb_out \
  --lk-out wrb_out\lk_adts\World\Maps\Azeroth \
  --lk-client-path "J:\\wowDev\\modernwow" \
  --default-unmapped 0
```

- **Input:** Alpha `.wdt` file (e.g., `Azeroth.wdt`).
- **Burial:** `--max-uniqueid` removes later work; `--bury-depth` optional (default `-5000`).
- **Terrain fixes:** `--fix-holes` clears MCNK hole flags around buried-placement neighborhoods. `--disable-mcsh` zeros baked shadows.
- **LK output:** `--lk-out` root directory for converted ADTs.
  - Also writes a fresh `<map>.wdt` in the same folder.
- **Area IDs:**
  - If `--area-remap-json` provided, it is used verbatim (AlphaAreaId→LKAreaId).
  - Else, LK `AreaTable.dbc` is read from `--lk-client-path` and Alpha IDs passthrough if present; others become `--default-unmapped` (default `0`).
  - You can also supply crosswalk CSVs via `--crosswalk-dir` / `--crosswalk-file` for precise mapping.

Tips:
- `--lk-client-path` should point to a LK (3.3.5) client root. MPQs are detected automatically; no extraction required.
- Prefer `--crosswalk-dir`/`--crosswalk-file` (keeps legacy `--dbctool-patch-*` aliases).

#### LK → Alpha Patcher (lk-to-alpha)

Patches an existing set of LK ADTs by burying placements and optionally clearing holes and removing shadows. Useful for iterating on LK results.

```powershell
dotnet run -c Release --project WoWRollback.Cli -- \
  lk-to-alpha \
  --lk-adts-dir .\wrb_out\lk_adts\World\Maps\Azeroth \
  --map Azeroth \
  --max-uniqueid 43000 \
  --fix-holes --disable-mcsh \
  --out .\patched_lk_az
```

- **Input:** Directory containing LK ADTs (e.g., output from `alpha-to-lk`).
- **Output:** Writes patched copies preserving relative directory structure under `--out`.
- **Same terrain logic:** Neighbor-aware hole clearing and MCSH zeroing are applied in the same way as `alpha-to-lk`.

#### Loose Files vs MPQs

- **Loose files:** Alpha inputs (`.wdt`, `.adt`) are regular files; no special setup.
- **MPQ-backed data:** When LK data is needed (AreaTable), pass `--lk-client-path` to a 3.3.5 client install. MPQs are read directly; no unpack step required.
- **Verification:**
  - `dotnet run --project WoWRollback.Cli -- probe-archive --client-path <lk-root>`
  - `dotnet run --project WoWRollback.Cli -- probe-minimap --client-path <lk-root> --map <MapName>`

---

### Orchestrator Command (Alpha→LK Pipeline)

**Single unified command** that runs the full pipeline:

```powershell
dotnet run --project WoWRollback.Orchestrator -- \
  --maps Shadowfang,Azeroth \
  --versions 0.5.3,0.5.5 \
  --alpha-root ../test_data \
  --lk-dbc-dir ../test_data/3.3.5/tree/DBFilesClient \
  --serve --port 8080
```

**Required Arguments:**
- `--maps` - Comma-separated map names (e.g., `Shadowfang,Azeroth`)
- `--versions` - Comma-separated Alpha version folders (e.g., `0.5.3,0.5.5`)
- `--alpha-root` - Path to Alpha data root
- `--lk-dbc-dir` - Path to LK 3.3.5 DBC directory

**Optional Arguments:**
- `--serve` - Start web server after generation
- `--port` - Web server port (default: 8080)
- `--mpq-path` - Path to MPQ archives for minimap extraction (see [Minimap Sources](#minimap-sources))
- `--verbose` - Enable detailed logging
- `--output-dir` - Custom output directory (default: `parp_out`)
- `--dbd-dir` - Custom WoWDBDefs directory

**Examples:**

```powershell
# Single map, with viewer (loose files)
dotnet run --project WoWRollback.Orchestrator -- \
  --maps Shadowfang \
  --versions 0.5.3 \
  --alpha-root ..\test_data \
  --lk-dbc-dir ..\test_data\3.3.5\tree\DBFilesClient \
  --serve

# Multiple maps, no viewer (loose files)
dotnet run --project WoWRollback.Orchestrator -- \
  --maps Azeroth,Kalimdor \
  --versions 0.5.3,0.5.5 \
  --alpha-root ..\test_data \
  --lk-dbc-dir ..\test_data\3.3.5\tree\DBFilesClient

# Single map with MPQ archives
dotnet run --project WoWRollback.Orchestrator -- \
  --maps Azeroth \
  --versions 0.5.3 \
  --alpha-root ..\test_data \
  --lk-dbc-dir ..\test_data\3.3.5\tree\DBFilesClient \
  --mpq-path E:\WoW_Clients \
  --serve

# Multiple versions with MPQ archives
dotnet run --project WoWRollback.Orchestrator -- \
  --maps Azeroth,Kalimdor \
  --versions 0.5.3,0.5.5,0.6.0 \
  --alpha-root ..\test_data \
  --lk-dbc-dir ..\test_data\3.3.5\tree\DBFilesClient \
  --mpq-path E:\WoW_Clients \
  --serve --port 8080

# Verbose output
dotnet run --project WoWRollback.Orchestrator -- \
  --maps Shadowfang \
  --versions 0.5.3 \
  --alpha-root ..\test_data \
  --lk-dbc-dir ..\test_data\3.3.5\tree\DBFilesClient \
  --verbose
```

---

## 🗺️ Minimap Sources

WoWRollback supports two methods for extracting minimap tiles for the web viewer:

### Method 1: Loose BLP Files (Default)

If `--mpq-path` is **not specified**, the pipeline scans for loose `.blp` minimap files in your Alpha data directories.

**Expected Directory Structure:**
```
test_data/
├── 0.5.3/
│   └── tree/
│       └── World/Textures/Minimap/
│           ├── md5translate.trs         # Translation table
│           ├── Azeroth/
│           │   ├── map00_00.blp
│           │   ├── map00_01.blp
│           │   └── ...
│           └── Kalimdor/
│               ├── map00_00.blp
│               └── ...
└── 0.5.5/
    └── tree/ (same structure)
```

**Command:**
```powershell
dotnet run --project WoWRollback.Orchestrator -- \
  --maps Azeroth,Kalimdor \
  --versions 0.5.3,0.5.5 \
  --alpha-root ..\test_data \
  --lk-dbc-dir ..\test_data\3.3.5\tree\DBFilesClient \
  --serve
```

**Pros:**
- ✅ Fastest extraction (direct file access)
- ✅ No additional tools needed
- ✅ Works with pre-extracted data

**Cons:**
- ❌ Requires pre-extraction from MPQs
- ❌ Takes significant disk space

---

### Method 2: MPQ Archives (Recommended)

If `--mpq-path` **is specified**, the pipeline reads minimap tiles directly from compressed MPQ archives using StormLib.

**Expected Directory Structure:**
```
E:\WoW_Clients\
├── 0.5.3\
│   ├── base.MPQ              # Base data archive
│   ├── patch.MPQ             # Patch archive (optional)
│   └── patch-2.MPQ           # Additional patches (optional)
└── 0.5.5\
    ├── base.MPQ
    └── patch.MPQ
```

**MPQ Archive Contents** (internal paths):
```
base.MPQ:
  └── textures\minimap\
      ├── md5translate.trs     # Translation table (inside MPQ)
      ├── <hash1>.blp          # Hashed minimap tiles
      ├── <hash2>.blp
      └── ...
```

**Command:**
```powershell
dotnet run --project WoWRollback.Orchestrator -- \
  --maps Azeroth,Kalimdor \
  --versions 0.5.3,0.5.5 \
  --alpha-root ..\test_data \
  --lk-dbc-dir ..\test_data\3.3.5\tree\DBFilesClient \
  --mpq-path E:\WoW_Clients \
  --serve
```

**Pros:**
- ✅ No pre-extraction required
- ✅ Saves disk space (keeps files compressed)
- ✅ Handles patched archives automatically
- ✅ Works with original client installations

**Cons:**
- ❌ Slightly slower than loose files (decompression overhead)
- ❌ Requires StormLib wrapper (included)

---

### How It Works

**Loose Files (Default):**
1. Scans `{alpha-root}/{version}/tree/World/Textures/Minimap/`
2. Parses `md5translate.trs` to map tile names to BLP files
3. Opens BLP files directly from filesystem
4. Converts to PNG for web viewer

**MPQ Archives (`--mpq-path`):**
1. Opens all `.MPQ` files in `{mpq-path}/{version}/`
2. Applies patch archives on top of base archives
3. Reads `md5translate.trs` from inside the MPQ
4. Extracts BLP tiles by MD5 hash from MPQ streams
5. Converts to PNG for web viewer

**Path Resolution:**
- `--mpq-path E:\WoW_Clients` + `--versions 0.5.3,0.5.5`
- → Looks for MPQs in `E:\WoW_Clients\0.5.3\` and `E:\WoW_Clients\0.5.5\`

---

## 📊 WoWDataPlot - Static Visualization

**WoWDataPlot** is a lightweight static visualization generator that creates interactive HTML pages with minimap overlays showing object placements. Unlike the dynamic viewer, it generates all files upfront with zero server requirements after generation.

### Usage

```powershell
dotnet run --project WoWDataPlot -- visualize \
  --wdt <path-to-wdt> \
  --output-dir <output-directory> \
  [--gap-threshold <number>] \
  [--tile-size <pixels>] \
  [--map-size <pixels>] \
  [--tile-marker-size <number>] \
  [--map-marker-size <number>]
```

### Arguments

**Required:**
- `--wdt` - Path to Alpha WDT file (e.g., `Kalidar.wdt`)
- `--output-dir` - Directory for generated output

**Optional:**
- `--gap-threshold` - Split layers when UniqueID jumps exceed this value (default: 50)
- `--tile-size` - Per-tile image size in pixels (default: 1024)
- `--map-size` - Overview map size in pixels (default: 2048)
- `--tile-marker-size` - Marker size for per-tile plots (default: 8)
- `--map-marker-size` - Marker size for overview map (default: 5)

### Example

```powershell
# Kalidar with default settings
dotnet run --project WoWDataPlot -- visualize \
  --wdt ..\test_data\0.5.3\tree\World\Maps\Kalidar\Kalidar.wdt \
  --output-dir .\Kalidar_viz

# Azeroth with custom layer detection
dotnet run --project WoWDataPlot -- visualize \
  --wdt ..\test_data\0.5.3\tree\World\Maps\Azeroth\Azeroth.wdt \
  --output-dir .\Azeroth_viz \
  --gap-threshold 100
```

### Output Structure

```
Kalidar_output/
├── Kalidar_legend.html              # ⭐ START HERE - Interactive map + legend
├── Kalidar_overview.png             # Continental heatmap (all layers)
├── Kalidar_analysis.json            # Detailed statistics per tile
├── minimaps/                        # Converted minimap tiles (PNG)
│   ├── Kalidar_33_26.png
│   ├── Kalidar_33_27.png
│   └── ...
└── tiles/                           # Per-tile interactive pages
    ├── tile_33_26.html              # Individual tile viewer
    ├── tile_33_26_heatmap.png       # Tile-level heatmap
    ├── overlays_33_26/              # Transparent layer PNGs
    │   ├── layer_325865_325887.png  # First layer (WMO buildings)
    │   ├── layer_333184_333384.png  # Second layer (M2 props)
    │   └── ...
    └── ...
```

### Key Features

#### 1. Minimap Overlay System
- **Transparent PNG layers** rendered on top of minimap tiles
- **Canvas-based rendering** in browser for smooth toggling
- **1:1 pixel alignment** - placement coordinates match minimap exactly
- **Per-layer isolation** - Each UniqueID range gets its own overlay file

#### 2. Layer Detection
- **Automatic gap analysis** - Splits when UniqueID jumps exceed threshold
- **Configurable threshold** - Adjust sensitivity via `--gap-threshold`
- **Global coloring** - Colors based on position in overall UniqueID distribution
- **Statistics** - Min/max/count per layer saved to JSON

#### 3. Interactive Controls
- **Layer toggles** - Show/hide individual UniqueID ranges
- **All On/Off buttons** - Quick layer management
- **Debug transforms** - Flip X, Flip Y, Swap X↔Y for coordinate troubleshooting
- **Live canvas updates** - Changes apply instantly without reload

#### 4. Continental Overview
- **Heatmap visualization** - 100-bucket gradient showing all placements
- **Global color scale** - Blue (early UniqueIDs) → Red (late UniqueIDs)
- **2048x2048 resolution** - High-detail overview of entire map
- **Legend page** - Clickable overview with full layer list

### Coordinate System

**Critical Fix Applied:** WoWDataPlot includes proper coordinate transformations to align placement data with minimap tiles:

```
WoW World Coords → Tile Pixel Coords:
1. Apply base formula: localX = (32 - worldX/533.33) - floor(...)
2. Convert to pixels: pixelX = localX * imageWidth
3. Flip both axes: pixelX = imageWidth - pixelX
                    pixelY = imageHeight - pixelY
```

This ensures dots appear exactly where objects exist on the minimap terrain.

### Use Cases

**Data Analysis:**
- Quickly identify object placement patterns
- Visualize UniqueID distribution across map
- Detect temporal layers (objects added over time)
- Debug coordinate system issues

**Documentation:**
- Generate static reports for map content
- Share visualizations without server setup
- Archive historical map states
- Compare different map versions

**Development:**
- Verify placement data extraction accuracy
- Debug coordinate transform issues
- Validate minimap tile associations
- Test layer detection thresholds

### Performance

- **Kalidar** (13 tiles, 898 placements): ~5 seconds
- **Small maps** (25 tiles): ~10 seconds
- **Large maps** (140 tiles): ~1-2 minutes

### Limitations

- **Static generation** - No real-time updates (regenerate to refresh)
- **Alpha WDT only** - Designed for Alpha format (0.5.x - 0.6.x)
- **Memory usage** - Large maps with many layers may use significant RAM
- **No 3D visualization** - 2D minimap overlays only

### Comparison with Dynamic Viewer

| Feature | WoWDataPlot (Static) | WoWRollback.Cli (Dynamic) |
|---------|---------------------|---------------------------|
| Server Required | ❌ No (after generation) | ✅ Yes (ASP.NET Core) |
| Real-time Updates | ❌ Regenerate needed | ✅ Live data loading |
| File Size | Small (PNGs + HTML) | Larger (WebP + JSON) |
| Setup Time | Fast (~5s) | Slower (analysis + server) |
| Interactivity | Layer toggles only | Full pan/zoom/filtering |
| 3D Support | ❌ No | ✅ GLB mesh loading |
| Cluster View | ❌ No | ✅ Yes |
| UniqueID Filtering | Layer-based | Range-based |
| Best For | Quick exploration | Deep analysis |

---

### Choosing a Method

**Use Loose Files if:**
- You already have extracted minimap directories
- You need maximum performance
- Your Alpha data is pre-organized

**Use MPQ Archives if:**
- You have original client installations
- You want to save disk space
- You need to handle patched versions
- You're processing many versions

---

## 🔧 Building

### Initial Setup

**1. Initialize Git Submodules** (First time only):

WoWRollback depends on several libraries that use git submodules. You must initialize them before building:

```powershell
cd lib\WoWTools.Minimaps
git submodule init
git submodule update --recursive
```

This will checkout:
- **TACT.NET** - MPQ archive handling
- **Warcraft.NET** - WoW file format library
- **SereniaBLPLib** - BLP texture decoder
- **CascLib** - CASC storage library

**2. Build Solution:**

```powershell
cd ..\..\WoWRollback
dotnet build WoWRollback.sln
```

**Run Tests:**
```powershell
dotnet test
```

### Common Build Issues

**"TACT.Net not found" errors:**
- Run `git submodule update --recursive` from the `lib\WoWTools.Minimaps` directory
- Verify submodules checked out: `git submodule status`

**"StormLibWrapper failed to build":**
- Ensure all submodules are initialized
- Check that `TACT.NET` directory exists and is not empty

---

## 📂 Output Structure

Each run creates a **timestamped session directory**:

```
parp_out/
└── session_20251007_012032/
    ├── manifest.json           # Session metadata
    ├── logs/
    │   ├── dbc_stage.log
    │   ├── adt_stage.log
    │   ├── analysis_stage.log
    │   └── viewer_stage.log
    │
    ├── 01_dbcs/                # DBC Stage outputs
    │   └── {version}/
    │       └── raw/
    │           ├── AreaTable_{version}.csv
    │           └── AreaTable_3_3_5.csv
    │
    ├── 02_crosswalks/          # Area mappings
    │   └── {version}/{alias}/
    │       ├── maps.json
    │       └── compare/
    │           ├── v2/         # V2 crosswalks
    │           │   ├── Area_patch_{alias}_to_335.csv
    │           │   ├── Area_mapping_{alias}_to_335.csv
    │           │   └── Area_unmatched_{alias}.csv
    │           └── v3/         # V3 hierarchy (future)
    │
    ├── 03_adts/                # Converted LK ADTs
    │   └── {version}/
    │       ├── World/Maps/{map}/
    │       │   ├── {map}_{x}_{y}.adt
    │       │   └── ...
    │       └── csv/maps/{map}/
    │           ├── terrain.csv
    │           └── shadow.csv
    │
    ├── 04_analysis/            # Analysis outputs (coming soon)
    │   └── {version}/
    │       ├── uniqueids/      # Time-travel CSVs
    │       │   ├── {map}_uniqueID_analysis.csv
    │       │   └── {map}_layers.json
    │       └── terrain/        # MCNK metadata
    │           └── {map}_mcnk_terrain.csv
    │
    └── 05_viewer/              # Web viewer
        ├── index.html
        ├── js/
        ├── styles.css
        ├── minimap/            # Extracted minimap tiles
        │   ├── {version}/
        │   │   └── {map}/
        │   │       ├── {map}_0_0.png
        │   │       ├── {map}_0_1.png
        │   │       └── ...
        ├── overlays/
        │   ├── {version}/{map}/
        │   │   ├── terrain_complete/
        │   │   ├── objects_combined/
        │   │   └── shadow_map/
        │   └── metadata.json
        └── config/
            └── maps.json
```

---

## 🐛 Troubleshooting

### Pipeline Issues

**"No AreaIDs patched"**
- Check that crosswalk CSVs exist in `02_crosswalks/{version}/{alias}/compare/v2/`
- Verify path structure matches: `02_crosswalks/0.5.3/0.5.3/compare/v2/Area_patch_0_5_3_to_335.csv`
- Enable `--verbose` to see detailed crosswalk resolution logs

**"WDT not found"**
- Verify WDT exists at: `{alpha-root}/{version}/tree/World/Maps/{map}/{map}.wdt`
- Check map name capitalization matches exactly (case-sensitive)
- Ensure WDT is valid Alpha format (not LK)

**"DBC directory not found"**
- LK DBC directory should contain `AreaTable.dbc` and `Map.dbc`
- Path format: `{lk-dbc-dir}/AreaTable.dbc`
- Verify DBCs are LK 3.3.5 format

### Build Issues

**ImageSharp vulnerability warnings**
- Dependency from `Warcraft.NET` library
- Safe to ignore in controlled environments
- Will be resolved when upstream updates

**Project reference errors**
- Ensure you're in the `WoWRollback/` root directory
- Run `dotnet restore` before building
- Check all project references exist

### Minimap Issues

**No minimap tiles generated**
- **Loose Files Mode**: Check that minimap BLPs exist at `{alpha-root}/{version}/tree/World/Textures/Minimap/{map}/`
- **MPQ Mode**: Verify MPQ archives exist at `{mpq-path}/{version}/*.MPQ`
- Check console logs for `[MpqMinimapProvider]` or `[LooseFileMinimapProvider]` messages
- Ensure `md5translate.trs` exists (loose files) or is inside MPQ archives

**MPQ archives not opening**
- Verify MPQ files are readable (not corrupted)
- Check file permissions on MPQ directory
- Ensure StormLib dependencies are present (auto-included)
- Try with a different version to isolate the issue

**Minimap tiles appear black/corrupted**
- BLP format may be incompatible with SereniaBLPLib decoder
- Check BLP file integrity
- Verify tiles display correctly in WoW Model Viewer or similar tools

### Viewer Issues

**Overlays missing in viewer**
- ✅ **Analysis stage not implemented yet** - Coming soon!
- Per-tile overlay JSONs will be generated in Stage 3
- Currently only `metadata.json` is created

**Viewer won't start**
- Check if port 8080 is already in use
- Use `--port 8081` to specify alternative port
- Ensure firewall allows local HTTP server

**404 errors in browser console**
- Normal for missing overlay tiles (sparse coverage)
- Check `05_viewer/overlays/metadata.json` for available overlays
- Verify viewer assets copied correctly
- Minimap tile 404s are normal for unpopulated map regions

### Performance

**Large maps take time**
- Shadowfang: ~30 seconds (25 tiles)
- Azeroth: ~5-10 minutes (128 tiles)
- Kalimdor: ~8-15 minutes (140 tiles)
- Use `--maps Shadowfang` for quick testing

---

## ✨ Features

### Current (v1.0 - Loose ADT Analysis)
`WoWRollback.AdtConverter` now ships inside `WoWRollback.sln` and builds with the rest of the toolkit. Run it directly for LK ⇄ Alpha terrain workflows:

```powershell
dotnet run --project WoWRollback.AdtConverter -- pack-monolithic \
  --lk-dir <path-to-lk-map-dir> \
  --lk-wdt <path-to-lk-wdt> \
  --map <MapName> \
  [--out <output-root>] \
  [--force-area-id <id>] \
  [--main-point-to-data] \
  [--verbose-logging]
```

#### ADT Analysis (NEW!)
- ✅ **Loose ADT analysis** - No conversion needed, reads 0.5.x-4.x+ formats directly
- ✅ **M2/WMO extraction** - Reads MDDF/MODF chunks from `_obj0.adt` files
- ✅ **MCNK terrain extraction** - AreaIDs, flags, textures, liquids, holes
- ✅ **3D mesh extraction** - Generates GLB terrain meshes per tile (NEW!)
- ✅ **Spatial clustering** - Detects prefabs & object brushes (proximity-based)
- ✅ **Pattern recognition** - Finds recurring object compositions
- ✅ **UniqueID analysis** - Ranges, layers, gaps per tile
- ✅ **Cross-tile duplicate filtering** - Removes culling duplicates
- ✅ **Coordinate transform fixes** - Proper ADT placement→world mapping

#### Built-in Web Server (NEW!)
- ✅ **Self-contained HTTP server** - ASP.NET Core Kestrel (no Python!)
- ✅ **Auto-detection** - Finds viewer in common locations
- ✅ **Browser integration** - Auto-opens on startup
- ✅ **Custom MIME types** - WebP, JSON, GeoJSON support
- ✅ **Configurable** - Custom port, optional browser launch

#### Web Viewer Enhancements (NEW!)
- ✅ **Cluster overlays** - Default view shows ~100 clusters vs 28K objects (50-100x faster!)
- ✅ **WebP minimaps** - 50-70% memory savings, lazy loading
- ✅ **UniqueID range filtering** - Load & filter by ID ranges
- ✅ **Popup text wrapping** - Long asset paths no longer escape popups
- ✅ **Coordinate labels fixed** - World X/Y/Z display correctly
- ✅ **CDN failover** - jsdelivr.net instead of unpkg.com (no DNS hangs)
- ✅ **Lazy tile loading** - Start zoomed to top-left, only load visible tiles
- ✅ **All minimap tiles shown** - Even tiles with no placements

#### Data Quality (NEW!)
- ✅ **Cross-tile duplicate detection** - Same UniqueID on multiple tiles filtered
- ✅ **Tile-only filtering** - Objects only shown on tiles where coordinates place them
- ✅ **Coordinate validation** - 180° placement flip + axis swap corrections
- ✅ **Dummy marker filtering** - Internal tile markers removed from overlays

### Previous (v0.5 - Alpha Pipeline)

#### DBC Processing
- ✅ **AreaTable extraction** - Dumps Alpha + LK AreaTable.dbc to CSV
- ✅ **Area hierarchy crosswalks** - V2 zone/subzone matching with confidence scores
- ✅ **Map metadata** - Generates maps.json with continent/instance info

#### ADT Conversion
- ✅ **Alpha → LK format** - Full ADT conversion with chunk patching
- ✅ **AreaID remapping** - Patches MCNK AreaIDs using crosswalk mappings
- ✅ **Asset fuzzy-matching** - Resolves missing textures/models via listfile
- ✅ **Terrain extraction** - MCNK flags, liquids, holes exported to CSV
- ✅ **Shadow map export** - Shadow data exported to CSV

#### Web Viewer
- ✅ **Interactive map viewer** - Leaflet-based tile viewer
- ✅ **Version switching** - Compare multiple Alpha versions
- ✅ **Minimap extraction** - Supports loose BLP files and MPQ archives
- ✅ **MPQ archive support** - Direct extraction from compressed archives
- ✅ **Static file serving** - Built-in HTTP server

### Coming Soon (v1.1+)

#### 3D Viewer (NEW!)
- ⏳ **Three.js/Babylon.js viewer** - 3D terrain visualization
- ⏳ **GLB mesh loading** - On-demand tile loading from mesh manifest
- ⏳ **3D placement markers** - M2/WMO objects in 3D space
- ⏳ **Camera controls** - Orbit, pan, zoom in 3D
- ⏳ **Shared data sources** - Reuses 2D viewer placement data

#### Viewer Enhancements
- ⏳ **Cluster overlay plugin** - Visualize clusters as circles/polygons
- ⏳ **Click cluster → expand** - Toggle from cluster view to individual objects
- ⏳ **Time-travel slider** - Show/hide object layers by UniqueID ranges
- ⏳ **Diff visualization** - Compare versions side-by-side
- ⏳ **AreaID overlay** - Show area boundaries with labels
- ⏳ **Liquids overlay** - Visualize water/lava/slime from MCNK
- ⏳ **Holes overlay** - Show terrain holes (gaps in ground)

#### Analysis Enhancements
- ⏳ **Pattern matching** - Identify identical object groups across map
- ⏳ **Prefab detection** - Find reused building/prop compositions
- ⏳ **Change detection** - Diff between versions for same map
- ⏳ **Asset catalog** - Generate inventory of all M2/WMO paths used

#### Data Export
- ⏳ **GeoJSON export** - Placements as geospatial data
- ⏳ **SQLite export** - Queryable database of all objects
- ⏳ **Filtered ADT export** - Write modified ADTs with selected ranges

### Future Ideas (v2.0+)

- 🔮 **Multi-map comparison** - Analyze multiple maps simultaneously
- 🔮 **ADT grid overlay** - wow.tools-style tile grid with labels
- 🔮 **Heatmap overlays** - Object density, change magnitude
- 🔮 **Alpha backporting** - LK → Alpha format conversion
- 🔮 **Heightmap export** - Generate height data from MCVT
- 🔮 **WDT analysis** - Global map metadata extraction
- 🔮 **OBJ mesh export** - Alternative to GLB for external tools
- 🔮 **Texture baking** - Apply minimap textures to terrain meshes

---

## 📚 Documentation

### Planning Documents
- **`docs/planning/03_Rollback_TimeTravel_Feature.md`** - Phase 0 time-travel design
- **`docs/planning/04_Overlay_Plugin_Architecture.md`** - Viewer plugin system
- **`docs/planning/04_Architecture_Changes.md`** - Before/after architecture comparison
- **`docs/planning/05_AnalysisModule_Implementation.md`** - Analysis stage specification

### Architecture Docs
- **`docs/architecture/overlay-system-architecture.md`** - Complete overlay pipeline
- **`docs/architecture/mcnk-flags-overlay.md`** - MCNK terrain implementation

---

## 🤝 Related Projects

- **[DBCTool.V2](../DBCTool.V2/)** - DBC extraction and area matching engine
- **[AlphaWdtAnalyzer.Core](../AlphaWDTAnalysisTool/AlphaWdtAnalyzer.Core/)** - Alpha WDT/ADT format library
- **[wow.tools](https://wow.tools/)** - WoW file formats and listfiles

---

## 📄 License

See LICENSE file in repository root.
