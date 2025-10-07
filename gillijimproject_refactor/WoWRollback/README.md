# WoWRollback - Unified Alpha Map Conversion Pipeline

**Digital archaeology toolkit** for World of Warcraft Alpha map content - converts Alpha WDTs to Lich King ADTs with AreaID patching, generates comparison data, and produces an interactive web viewer.

---

## 🚀 Quick Start

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

- ✅ **No shell execution** - All tools called as library APIs
- ✅ **Typed interfaces** - Structured results instead of exit codes  
- ✅ **Clean separation** - Each module has a single responsibility
- ✅ **Easy testing** - Modules can be tested independently
- ✅ **Cross-platform** - Works on Windows, Linux, macOS

---

## Prerequisites

- **.NET SDK 9.0** (64-bit)
- **Alpha WoW data** - Extracted WDT/ADT/DBC files in standard layout
- **LK 3.3.5 DBCs** - AreaTable.dbc, Map.dbc for crosswalk generation
- Optional: **WoWDBDefs** - For DBC schema definitions (auto-resolved)

---

## 📖 CLI Reference

### Orchestrator Command (Primary)

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
- `--verbose` - Enable detailed logging
- `--output-dir` - Custom output directory (default: `parp_out`)
- `--dbd-dir` - Custom WoWDBDefs directory

**Examples:**

```powershell
# Single map, with viewer
dotnet run --project WoWRollback.Orchestrator -- \
  --maps Shadowfang \
  --versions 0.5.3 \
  --alpha-root ..\test_data \
  --lk-dbc-dir ..\test_data\3.3.5\tree\DBFilesClient \
  --serve

# Multiple maps, no viewer
dotnet run --project WoWRollback.Orchestrator -- \
  --maps Azeroth,Kalimdor \
  --versions 0.5.3,0.5.5 \
  --alpha-root ..\test_data \
  --lk-dbc-dir ..\test_data\3.3.5\tree\DBFilesClient

# Verbose output
dotnet run --project WoWRollback.Orchestrator -- \
  --maps Shadowfang \
  --versions 0.5.3 \
  --alpha-root ..\test_data \
  --lk-dbc-dir ..\test_data\3.3.5\tree\DBFilesClient \
  --verbose
```

---

## 🔧 Building

```powershell
cd WoWRollback
dotnet build WoWRollback.sln
```

**Run Tests:**
```powershell
dotnet test
```

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

### Performance

**Large maps take time**
- Shadowfang: ~30 seconds (25 tiles)
- Azeroth: ~5-10 minutes (128 tiles)
- Kalimdor: ~8-15 minutes (140 tiles)
- Use `--maps Shadowfang` for quick testing

---

## ✨ Features

### Current (v0.5)

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
- ✅ **Static file serving** - Built-in HTTP server

### Coming Soon (v0.6 - Analysis Stage)

#### UniqueID Analysis (Phase 0: Time-Travel)
- ⏳ **UniqueID distribution CSVs** - Track object ID ranges per tile
- ⏳ **Layer detection** - Identify distinct "work sessions" by ID gaps
- ⏳ **Time-travel filtering** - Timeline slider to show/hide object layers
- ⏳ **JSON layer metadata** - Export detected layers for viewer

#### Per-Tile Overlays (Plugin Architecture)
- ⏳ **Terrain overlays** - MCNK properties, liquids, holes per tile
- ⏳ **Object overlays** - M2/WMO placements with UniqueIDs
- ⏳ **Shadow overlays** - Shadow map visualization
- ⏳ **Overlay manifest** - Plugin system coordination

#### MCNK Metadata
- ⏳ **Terrain CSVs** - Complete MCNK data per tile
- ⏳ **Property analysis** - Flags, layers, holes statistics
- ⏳ **AreaID validation** - Verify patched values

### Future Enhancements (Phase 1+)

- 🔮 **Diff visualization** - Show object additions/removals between versions
- 🔮 **Multi-map comparison** - Side-by-side map views
- 🔮 **ADT grid overlay** - wow.tools-style tile grid with labels
- 🔮 **Heatmap overlays** - Object density, change magnitude
- 🔮 **Export filtered ADTs** - Write modified ADTs with selected ranges
- 🔮 **Alpha backporting** - LK → Alpha format conversion

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
