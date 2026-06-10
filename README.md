# parp-tools

Preservation, conversion, analysis, and visualization tooling for World of Warcraft game data.

**Active development branch**: `v0.5.0-dev`
**Active project**: `wow-viewer/`

The repository also contains `gillijimproject_refactor/` — a legacy codebase that is **read-only reference**. All new code, features, tools, tests, and fixes go in `wow-viewer/`.

---

## Active Project: `wow-viewer/`

A .NET 10 toolkit for WoW format analysis, terrain reconstruction, PM4-based object matching, and ML dataset generation. Includes a full-featured 3D world viewer (`WoWViewer`), CLI format tools, and a Python ML pipeline.

**Support range**: Alpha 0.5.3 through LK 3.3.5, with partial Cataclysm-era terrain and PM4 support.

### Quick Start

```powershell
# Build everything
dotnet build wow-viewer/WowViewer.slnx -c Debug

# Run the viewer
dotnet run --project wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug -- <client-root> <map-name>

# Run CLI tools
dotnet run --project wow-viewer/tools/inspect/WowViewer.Tool.Inspect -c Debug -- m2 inspect --input <model.m2>
dotnet run --project wow-viewer/tools/animfarm/WowViewer.Tool.AnimFarm -c Debug -- dump --input <model.m2> --output <dir>
```

### Project Structure

```
wow-viewer/
├── src/
│   ├── core/              # Shared libraries
│   │   ├── WowViewer.Core/          # Data models
│   │   ├── WowViewer.Core.IO/       # Format readers/writers
│   │   ├── WowViewer.Core.PM4/      # PM4 chunk analysis
│   │   ├── WowViewer.Core.Runtime/  # M2 rendering pipeline
│   │   └── WowViewer.Core.Anim/     # M2 animation pose extraction
│   └── viewer/
│       └── WoWViewer/               # 3D world viewer app
├── tools/                 # CLI tools
│   ├── inspect/           # Format inspector
│   ├── converter/         # Format converter
│   ├── harvest/           # Terrain tensor harvest (NPZ/Zarr)
│   ├── capture/           # Headless validation capture
│   └── animfarm/          # M2 animation pose farm
├── tests/                 # xUnit tests
├── data-harvester/        # Python ML pipeline
├── docs/                  # Architecture docs + guides
├── specs/                 # Feature specifications
└── memory-bank/           # Session continuity
```

### Key Documentation

- [WoWViewer README](wow-viewer/README.md) — viewer app overview and quick start
- [CLI Tools Guide](wow-viewer/docs/CLI-TOOLS.md) — advanced usage for all CLI tools
- [Plans Overview](wow-viewer/docs/PLANS-OVERVIEW.md) — summary of remaining specs
- [Memory Bank](wow-viewer/memory-bank/activeContext.md) — current focus and status
- [Architecture Docs](wow-viewer/docs/architecture/) — PM4 semantics, render plans, model specs

### What's Supported

- **Terrain**: Alpha 0.5.3 monolithic WDT, 0.6.0 split ADTs, LK 3.3.5 split ADTs
- **WMO**: V14 (Alpha), V17 (LK), with round-trip conversion
- **M2/MDX**: Classic MD20, era-1121 MD20, chunked MDLX; animation extraction and BVH export
- **PM4**: Full decode, per-file caching, MSCN/MSPV visualization, WMO group matching
- **BLP**: Pixel decode and summary inspection
- **DBC/DB2**: Crosswalk generation and lookup
- **Audio**: Alpha-area audio catalog inspection
- **ML Datasets**: V16/V18 terrain tensor extraction and model training pipeline

### Build & Test

```powershell
# Build
dotnet build wow-viewer/WowViewer.slnx -c Debug

# Run all tests
dotnet test wow-viewer/WowViewer.slnx -c Debug

# Run specific test project
dotnet test wow-viewer/tests/WowViewer.Core.Anim.Tests/ -c Debug

# Run specific test category
dotnet test wow-viewer/tests/WowViewer.Core.PM4.Tests/ -c Debug --filter "FullyQualifiedName~Pm4PerFileCache"
```

---

## Legacy: `gillijimproject_refactor/`

**Read-only reference.** This subtree contains the earlier `MdxViewer` and `WoWMapConverter` codebases that served as the foundation for the current `wow-viewer/` project. No new code, features, or bugfixes should be written here.

What `gillijimproject_refactor` is good for:

- **Reference implementation**: How the legacy viewer loaded terrain, WMOs, and MDX models
- **Test data**: `test_data/development/` — development map split ADTs and PM4 files
- **Memory bank (archived)**: Historical context at `gillijimproject_refactor/memory-bank/` — the active memory bank has moved to `wow-viewer/memory-bank/`

What `gillijimproject_refactor` is NOT:

- A development target: all active work is in `wow-viewer/`
- A place for new code: see RULE 1 in `AGENTS.md`

---

## Supported Game Versions

| Version | Era | Support |
|---------|-----|---------|
| 0.5.3 | Alpha | Terrain, WMO, MDX, BLP, DBC — full read/write |
| 0.6.0 | Alpha | Terrain, split ADTs — full |
| 0.7.0 / 0.8.0 | Pre-Release | MDX (chunked) — partial |
| 1.12.1 | Vanilla | M2 (era-1121 MD20) — full (spec 048) |
| 2.x | TBC | Not yet supported |
| 3.0.1 / 3.3.5 | WotLK | Terrain, WMO, M2, PM4 — full |
| 4.0.0+ | Cataclysm | ADT, PM4 partial; terrain reconstruction path exists |

## Branch History

- `v0.5.0-dev` — **Current active branch.** All wow-viewer work.
- `v0.4.9` — Previous branch, freeze point before the wow-viewer split.
- `main` / `v0.4.5` — Older branches, legacy MdxViewer era.

## Disclaimer

This project is not an official Blizzard Entertainment product and is not affiliated with or endorsed by Blizzard Entertainment or World of Warcraft.
