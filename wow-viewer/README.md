# WoWViewer

A .NET 10 / OpenGL world viewer and analysis toolkit for World of Warcraft game client data, supporting Alpha 0.5.3 through LK 3.3.5.

## Features

- **3D World Rendering** — Terrain, WMOs, M2/MDX models, liquids, and DBC-driven overlays
- **PM4 Overlay** — Per-file cached scene-graph visualization with MSCN/MSPV point clouds, WMO group matching, and surface diagnostics
- **Format Inspection** — Inspect and dump any supported format (ADT, WDT, WMO, M2, MDX, BLP, PM4, DBC) via CLI
- **Format Conversion** — Convert between Alpha and LK terrain formats, WMO versions, M2/MDX
- **Terrain Dataset Harvesting** — Extract NPZ/Zarr tensor shards from staged game clients for ML training
- **M2 Animation Pose Farm** — Extract bone animation keyframes as BVH + normalized pose clips (new)
- **Dataset Pipelines** — V18 terrain training pipeline with multi-build corpus support

## Quick Start

### Prerequisites

- .NET 10 SDK
- A staged game client in `output/tmp/wowarchive-clients/` (see [data-paths](memory-bank/data-paths.md))

### Build

```powershell
dotnet build wow-viewer/WowViewer.slnx -c Debug
```

### Run Viewer

```powershell
dotnet run --project src/viewer/WoWViewer/WoWViewer.csproj -c Debug -- <client-root> <map-name>
```

### Run CLI Tools

```powershell
# Inspect an M2 model
dotnet run --project tools/inspect/WowViewer.Tool.Inspect -c Debug -- m2 inspect --input <path/to/model.m2>

# Inspect a PM4 file
dotnet run --project tools/inspect/WowViewer.Tool.Inspect -c Debug -- pm4 inspect --input <path/to/file.pm4>

# Build a listfile cache (required before batch mode)
dotnet run --project tools/inspect/WowViewer.Tool.Inspect -c Debug -- archive build-listfile-cache --archive-root <staged-client> --cache-key <build>

# Farm animations from an M2 model
dotnet run --project tools/animfarm/WowViewer.Tool.AnimFarm -c Debug -- dump --input <path/to/model.m2> --output <dir>

# Convert Alpha WDT to LK
dotnet run --project tools/converter/WowViewer.Tool.Converter -c Debug -- alpha-to-lk --input <wdt> --output <dir>
```

## Project Structure

```
wow-viewer/
├── src/
│   ├── core/           # Shared libraries
│   │   ├── WowViewer.Core/          # Data models (M2, WMO, maps)
│   │   ├── WowViewer.Core.IO/       # Format readers/writers
│   │   ├── WowViewer.Core.PM4/      # PM4 chunk analysis
│   │   ├── WowViewer.Core.Runtime/  # M2 runtime, rendering pipeline
│   │   └── WowViewer.Core.Anim/     # M2 animation pose extraction
│   └── viewer/
│       └── WoWViewer/               # 3D world viewer app
├── tools/
│   ├── inspect/        # Format inspector
│   ├── converter/      # Format converter
│   ├── harvest/        # Terrain tensor harvest
│   ├── capture/        # Headless validation capture
│   └── animfarm/       # M2 animation pose farm
├── tests/              # xUnit tests
├── data-harvester/     # Python ML pipeline (uv-managed)
├── docs/               # Architecture docs
├── specs/              # Feature specs (Spec Kit)
└── memory-bank/        # Session continuity
```

## Documentation

- **Spec Kit specs**: `specs/` — organized by feature number
- **Architecture**: `docs/architecture/` — chunk semantics, render plans, model specs
- **Memory bank**: `memory-bank/` — active context, progress, data paths, system patterns

## Related

- **Spec 053** (anim farm): `specs/053-m2-animation-pose-farm/`
- **Spec 054** (PM4 cache): `specs/054-pm4-camera-window-cache/`
- **Spec 051** (MSCN/MSPV): `specs/051-pm4-mscn-mspv-visualization/` (in progress)
- **Spec 046** (PM4 asset matching): `specs/046-pm4-asset-matching/`
- **AGENTS.md** (this repo): workspace policies, rules, and build commands
