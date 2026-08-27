# parp-tools

Preservation, conversion, reverse engineering, analysis, and 3D visualization tooling for World of Warcraft client data.

**Current Release Line**: `v0.5.2.2`  
**Primary Project Directory**: `wow-viewer/`  
**Target Runtime**: .NET 10 (`net10.0`) & Python 3.12+ (`uv`)

> [!NOTE]
> `gillijimproject_refactor/` is a legacy read-only reference codebase. All active development, features, tools, tests, and documentation live under [`wow-viewer/`](wow-viewer/).

---

## What is parp-tools?

`parp-tools` is an end-to-end suite for exploring, reconstructing, and analyzing World of Warcraft game assets across the project's active client eras, primarily Alpha 0.5.3 through WotLK 3.3.5a:

- **Interactive 3D World Viewer (`WoWViewer`)**: High-performance multi-platform desktop viewer supporting streaming terrain, interior WMO portal culling, M2/MDX skeletal animations, directional lighting/fog, positional OpenAL audio emitters, camera path authoring, and PM4 placement reconciliation.
- **Format Inspection & Analysis CLI (`wowviewer-inspect`)**: Comprehensive tools to inspect, dump, and audit M2, MDX, BLP, WMO, ADT, WDT, LIT, and PM4 files directly from MPQ archives or disk. The canonical CLI index is [`wow-viewer/tools/README.md`](wow-viewer/tools/README.md).
- **Rosetta Calibration Corpus Generator (`rosetta-generate`)**: Generates synthetic, fully-labeled ADT/WDT maps placing every client model on an elevation-modulated pedestal with high-resolution antialiased MCAL/MCLY terrain labels for exact PM4 geometry calibration.
- **Format Conversion (`wowviewer-converter`)**: Bidirectional format conversion between pre-release Alpha monolithic WDT containers and modern Wrath of the Lich King (LK) ADT/WDT terrain files.
- **Dataset Harvester (`data-harvester` & `wowviewer-harvest`)**: High-throughput extraction of terrain elevation, normals, textures, alpha blending layers, liquids, and synthesized minimap shards into Zarr and NPZ tensor formats for machine learning pipelines.

---

## Project Structure

```
parp-tools/
├── wow-viewer/                     # Active .NET 10 solution and Python ML toolchain
│   ├── src/
│   │   ├── core/                   # Core shared libraries
│   │   │   ├── WowViewer.Core/          # Domain models, tensors, and memory layouts
│   │   │   ├── WowViewer.Core.IO/       # Pure format readers & writers (M2, WMO, ADT, WDT, BLP, MPQ)
│   │   │   ├── WowViewer.Core.PM4/      # PM4 chunk decoding, linkage, & reconciliation
│   │   │   ├── WowViewer.Core.Runtime/  # M2 scene graph, skins, and animation runtime
│   │   │   └── WowViewer.Core.Editor/   # Placement authoring, undo/redo sessions, and change tracking
│   │   └── viewer/
│   │       └── WoWViewer/               # Cross-platform desktop 3D viewer (Silk.NET / OpenGL / ImGui)
│   ├── tools/                      # Standalone CLI tools
│   │   ├── inspect/                     # wowviewer-inspect: multi-format analysis & rosetta generator
│   │   ├── converter/                   # wowviewer-converter: Alpha ↔ LK map format converter
│   │   ├── harvest/                     # wowviewer-harvest: terrain tensor extractor & minimap composer
│   │   ├── capture/                     # Headless validation and golden frame capture
│   │   └── validation-capture/          # Visual screenshot and regression capture
│   ├── data-harvester/             # Python ML dataset builder and training workflows (uv)
│   ├── tests/                      # xUnit test suites (Core, Editor, IO, PM4)
│   ├── docs/                       # Architecture notes, user guides, and reference specs
│   └── specs/                      # Feature specifications and implementation task plans
└── gillijimproject_refactor/       # Read-only legacy reference codebase
```

---

## Documentation Map

- **[WoWViewer project README](wow-viewer/README.md)**: Main active project overview.
- **[Core libraries README](wow-viewer/src/core/README.md)**: Library ownership, boundaries, and validation.
- **[Desktop viewer README](wow-viewer/src/viewer/WoWViewer/README.md)**: App launch, UI surfaces, and viewer proof boundaries.
- **[CLI tooling README](wow-viewer/tools/README.md)**: Single entry point for inspect, convert, harvest, capture, validation, WDL, minimap, and enrichment tools.
- **[Expanded CLI reference](wow-viewer/docs/CLI-TOOLS.md)**: Historical detailed command reference.
- **[Spec status router](wow-viewer/specs/STATUS.md)**: Active SpecKit lanes and next bounded actions.

---

## Quick Start

### 1. Prerequisites
- **.NET 10 SDK** (v10.0.100 or newer)
- **PowerShell 7** (`pwsh`)
- *(Optional for ML)* **Python 3.12+** with [`uv`](https://github.com/astral-sh/uv)

### 2. Building the Solution
```powershell
# Build entire solution in Debug configuration
dotnet build wow-viewer/WowViewer.slnx -c Debug

# Run all unit tests
dotnet test wow-viewer/WowViewer.slnx -c Debug
```

### 3. Launching the 3D Desktop Viewer
```powershell
# Launch viewer UI
dotnet run --project wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug

# Launch directly into a specific client root and world
dotnet run --project wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug -- `
  --game-path "H:\CLIENTS\World of Warcraft 3.3.5a" `
  --world "World\Maps\Azeroth\Azeroth.wdt"
```

### 4. Running CLI Tools

#### Inspect Model / Map / PM4 Assets
```powershell
# Inspect an M2 or MDX model
dotnet run --project wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -c Debug -- `
  m2 inspect --archive-root "H:\CLIENTS\World of Warcraft 3.3.5a" --virtual-path "Creature\Arthas\Arthas.m2"

# Inspect PM4 chunks and connectivity
dotnet run --project wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -c Debug -- `
  pm4 inspect --input "World/Maps/Azeroth/Azeroth_32_48.pm4"

# Survey all WDT maps in a client archive
dotnet run --project wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -c Debug -- `
  map inspect --archive-root "H:\CLIENTS\WoW-0.5.3.3368-Client"
```

#### Generate a Rosetta Calibration Corpus
Generates synthetic, labeled calibration ADT/WDT tiles containing museum display plinths and high-resolution MCAL antialiased text labels for every model:
```powershell
dotnet run --project wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -c Debug -- `
  rosetta-generate `
  --client-root "H:\CLIENTS\WoW-0.5.3.3368-Client" `
  --output "output/rosetta_alpha" `
  --map-name "RosettaAlpha" `
  --pedestal-height 4.0
```

#### Convert Between Alpha and Modern Map Formats
```powershell
# Convert Alpha 0.5.3 monolithic WDT to modern LK ADTs
dotnet run --project wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -c Debug -- `
  convert-alpha-to-lk --input "World\Maps\Kalimdor\Kalimdor.wdt" --output "output\converted\Kalimdor_LK"

# Convert modern LK ADTs to an Alpha monolithic WDT
dotnet run --project wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -c Debug -- `
  convert-lk-to-alpha --input "output\converted\Development_LK" --output "output\converted\Development_Alpha.wdt"
```

---

## Client Era Support Matrix

| Client Era | Version | Current State | Key Features / Notes |
|---|---|---|---|
| **Alpha** | 0.5.3 - 0.5.5 | **Implemented with active proof gates** | Monolithic WDTs, v14 WMO monoliths, MDX/MDL models, 2880-unit world clock, Alpha audio catalog; Rosetta minimap visual proof remains operator-owned |
| **Early Beta** | 0.6.x - 0.10.x | **Partial / research** | Split ADT/WDT transition format, chunked early models, prototype map layouts |
| **Classic** | 1.12.1 | **Implemented surfaces, proof-gated** | Standard ADTs with MCCV/MCLY/MCAL, v17 WMOs, 2004-era M2 structures, AreaTable routing |
| **TBC** | 2.4.3 | **Implemented surfaces, proof-gated** | Embedded skin profiles, expanded WMO materials, multi-layer liquid chunks |
| **WotLK** | 3.3.5a | **Primary reference era** | Reference LK terrain format, separated M2 `.skin` files, PM4 analysis workflows, WDL terrain horizons |

Later client terrain formats are outside the current project scope unless a future spec explicitly reopens that lane.

---

## Detailed Documentation

- **[Desktop Viewer User Guide](wow-viewer/docs/WoWViewer/USERGUIDE.md)**: In-depth manual covering viewer UI, navigation controls, camera-path authoring, audio emitter debugging, and the Spec 176 PM4 Reconciliation panel.
- **[CLI Tooling README](wow-viewer/tools/README.md)**: Canonical single-document entry point for all `wow-viewer/tools/` command-line projects.
- **[Expanded CLI Reference](wow-viewer/docs/CLI-TOOLS.md)**: Longer historical command reference for `wowviewer-inspect`, `wowviewer-converter`, `wowviewer-harvest`, and dataset workflows.
- **[Feature Specifications & Status](wow-viewer/specs/STATUS.md)**: Specification kit tracking active architecture specs, plans, and task lists.
- **[Memory Bank Dashboard](wow-viewer/memory-bank/activeContext.md)**: Current workstream focus, execution lanes, and progress ledger.

---

## License & Safety Notice

This tooling is intended strictly for historical preservation, data analysis, and format interoperability research. Do not distribute copyrighted client data or commercial game assets.
