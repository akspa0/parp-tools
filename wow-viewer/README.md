# WoWViewer

**Part of `parp-tools`** | **Branch**: `v0.5.0-dev` | **Stack**: .NET 10 / OpenGL 3.3 (Silk.NET) / ImGui

A 3D world viewer, format analysis toolkit, and ML dataset pipeline for World of Warcraft game client data. This is the sole active development target in the `parp-tools` monorepo — the legacy `MdxViewer` in `gillijimproject_refactor/` is read-only reference.

**Support range**: Alpha 0.5.3 through LK 3.3.5 (with partial Cataclysm-era PM4 and terrain paths). We effectively support versions of WoW from 0.5.3 through 4.0.0, with varying levels of format support, due to the vast array of changes in each version of every client. We've chosen to do piecemeal for now, on key versions of the game, since no software fully supports the file formats due to the vast amount of micro-changes done to them over time.

## Features

- **3D World Rendering** — Terrain, WMOs, M2/MDX models, liquids, and DBC-driven overlays from staged game clients
- **PM4 Overlay Analysis** — Per-file cached scene-graph with MSCN/MSPV point clouds, WMO group matching, surface diagnostics, and per-object signature extraction
- **Format Inspection** — CLI inspect for ADT, WDT, WMO, M2, MDX, BLP, PM4, DBC — full chunk-level dump
- **Format Conversion** — Alpha ↔ LK terrain round-trip, WMO V14 ↔ V17, M2 ↔ MDX conversion
- **Terrain Dataset Harvesting** — NPZ/Zarr tensor extraction from staged clients for the V16/V18 ML training pipeline
- **M2 Animation Pose Farm** — Bone animation keyframe extraction as BVH motion files + Mixamo-normalized pose clip JSON sidecars
- **Headless Validation Capture** — Automated renderer-truth capture for object-mask ground truth generation

## All Tools at a Glance

| Tool | Purpose | Location |
|------|---------|----------|
| `WoWViewer` | 3D world viewer app | `src/viewer/WoWViewer/` |
| `WowViewer.Tool.Inspect` | CLI format inspector | `tools/inspect/` |
| `WowViewer.Tool.Converter` | CLI format converter | `tools/converter/` |
| `WowViewer.Tool.Harvest` | CLI terrain tensor harvester | `tools/harvest/` |
| `WowViewer.Tool.ValidationCapture` | CLI headless capture | `tools/validation-capture/` |
| `WowViewer.Tool.AnimFarm` | CLI M2 animation farmer | `tools/animfarm/` |
| Data harvester (Python) | V16/V18 dataset build + training | `data-harvester/` |

## Quick Start

### Prerequisites
- `.NET 10 SDK`
- A staged game client in a temporary directory
- OpenGL 3.3 capable GPU

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

See `docs/CLI-TOOLS.md` for full usage and common workflows.

## Project Structure

```
wow-viewer/
├── src/
│   ├── core/              # Shared libraries
│   │   ├── WowViewer.Core/          # Data models (M2, WMO, maps)
│   │   ├── WowViewer.Core.IO/       # Format readers/writers
│   │   ├── WowViewer.Core.PM4/      # PM4 chunk analysis
│   │   ├── WowViewer.Core.Runtime/  # M2 runtime, rendering pipeline
│   │   └── WowViewer.Core.Anim/     # M2 animation pose extraction
│   └── viewer/
│       └── WoWViewer/               # 3D world viewer app
├── tools/                 # CLI tools
├── tests/                 # xUnit tests
├── data-harvester/        # Python ML pipeline (uv-managed)
├── docs/                  # Architecture docs + guides
├── specs/                 # Feature specs (Spec Kit)
└── memory-bank/           # Session continuity
```

## Documentation

| Document | What It Covers |
|----------|----------------|
| [CLI Tools Guide](docs/CLI-TOOLS.md) | Full reference for all CLI tools with workflow recipes |
| [PM4 ADT Restoration](docs/PM4-ADT-RESTORATION.md) | PM4 → ADT placement writing workflow |
| [Plans Overview](docs/PLANS-OVERVIEW.md) | Summary of all 25 remaining active specs |
| [Memory Bank](memory-bank/activeContext.md) | Current focus, spec status, known issues |
| [Data Paths](memory-bank/data-paths.md) | Game client and test data locations, with env-var overrides |
| [Coding Standards](memory-bank/coding_standards.md) | Project layout, C#/Python conventions, tests, commits |
| [Architecture Docs](docs/architecture/) | PM4 semantics, render plans, model specs |
| Feature Specs (`specs/`) | Per-feature spec/plan/tasks (Spec Kit) |

## Key Specs

| Spec | What | Status |
|------|------|--------|
| 046 | PM4 asset matching — surface correlation and fingerprint matching | 217/1604 matched, ADT validation at P@1=1.8% |
| 051 | PM4 MSCN/MSPV visualization and signature extraction | 15/33 done |
| 053 | M2 animation pose farm — BVH + pose clip extraction | 20/105 done (Phase 0-1) |
| 054 | PM4 camera window cache — in-memory + on-disk per-file cache | 17/18 done |
| 071 | Left-right sidebar split — viewer UI overhaul with workbench tabs | **Complete** |
| 076 | Full-map fractal brush library — terrain artist primitive recovery | **Active** — Phases 1-3 done, Phase 4 (BLP/texture inventory) in progress |
| 074 | Alpha brush library — MCAL connected component extraction | **Deprecated** — outputs are evidence rows for 076 |
| 075 | Scar mask segmentation — whole-tile scar-mask model | **Deprecated** — diagnostic baseline only |

## Game Version Support

We support versions of WoW from 0.5.3 through 4.0.0, with varying levels of format support, due to the vast array of changes in each version of every client. We've chosen to do piecemeal for now, on key versions of the game, since no software fully supports the file formats due to the vast amount of micro-changes done to them over time.

| Version | Era | Status |
|---------|-----|--------|
| 0.5.3 | Alpha | Full read/write — terrain, WMO, MDX, BLP, DBC |
| 0.6.0 | Alpha | Full — split ADTs |
| 0.7.0 / 0.8.0 | Pre-Release | MDX (chunked) — partial |
| 1.12.1 | Vanilla | M2 (era-1121 MD20) — full (spec 048) |
| 2.x | TBC | Not yet supported |
| 3.0.1 / 3.3.5 | WotLK | Full — terrain, WMO, M2, PM4 |
| 4.0.0+ | Cataclysm | ADT + PM4 — partial |

## Related

- **Repo root README**: `../README.md` — monorepo overview with legacy project info
- **AGENTS.md**: `../AGENTS.md` — workspace policies, rules, build commands
- **Legacy viewer**: `gillijimproject_refactor/` — read-only reference (MdxViewer)
