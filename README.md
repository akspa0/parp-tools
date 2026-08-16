# parp-tools

Preservation, conversion, analysis, and visualization tooling for World of Warcraft game data.

**Current viewer release line**: `v0.5.2.1`
**Active project**: `wow-viewer/`

The repository also contains `gillijimproject_refactor/` — a legacy codebase that is **read-only reference**. All new code, features, tools, tests, and fixes go in `wow-viewer/`.

---

## Active Project: `wow-viewer/`

A .NET 10 toolkit for WoW format analysis, terrain reconstruction, PM4-based object matching, and ML dataset generation. Includes a full-featured 3D world viewer (`WoWViewer`), CLI format tools, and a Python ML pipeline.

**Support range**: Alpha 0.5.3 through Cataclysm-era 4.0.x, with client-era differences called out in the support table below. Rendering and FPS claims remain build/map-specific and require real-client validation.

### Download

Prebuilt, self-contained viewer binaries for **Windows x64, Linux x64, macOS arm64, and macOS x64** are attached to every tagged release on the [releases page](https://github.com/akspa0/parp-tools/releases). No .NET install is required.

Native file dialogs are Windows-only; on Linux and macOS load content with `--game-path`, `--build`, and `--world`. See the [v0.5.2.1 release notes](wow-viewer/docs/releases/v0.5.2.1.md) for what changed — it is an out-of-band patch for the frame-pacing jank that shipped in v0.5.2.

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
- [User Guide](wow-viewer/docs/WoWViewer/USERGUIDE.md) — controls, UI layout, and common workflows
- [Release Notes — v0.5.2.1](wow-viewer/docs/releases/v0.5.2.1.md) — frame-pacing patch (current)
- [Release Notes — v0.5.2](wow-viewer/docs/releases/v0.5.2.md) — what changed since v0.5.1
- [CLI Tools Guide](wow-viewer/docs/CLI-TOOLS.md) — advanced usage for all CLI tools
- [Spec Status](wow-viewer/specs/STATUS.md) — the current-spec router
- [Plans Overview](wow-viewer/docs/PLANS-OVERVIEW.md) — summary of remaining specs
- [Memory Bank](wow-viewer/memory-bank/activeContext.md) — current focus and status
- [Architecture Docs](wow-viewer/docs/architecture/) — PM4 semantics, render plans, model specs

### What's Implemented

- **Terrain**: Alpha monolithic WDT, early split ADTs, LK/Cataclysm ADTs, WDL/WL* and minimap routes, bounded camera-centered streaming
- **WMO**: V14/V17 parsing, rendering, portal-aware visibility, placement inspection, doodad batching, and round-trip conversion paths
- **M2/MDX**: Classic/era-specific M2 profiles, embedded early-M2 route, MDLX/MDX routes, material/light parity work, animation extraction, and BVH export; cross-era visual proof is incomplete
- **PM4**: Full decode with solved coordinate frames, wall-mesh geometry, per-doodad identity, per-file caching, MSCN/MSPV visualization, WMO group matching
- **BLP**: Pixel decode, summary inspection, pure-C# DXT1 codec
- **DBC/DB2**: Crosswalk generation and lookup, dual-era AreaTable identity routing
- **Lighting**: LIT profile decode with DBC fallback, native Alpha 0.5.3 time-of-day clock
- **Audio**: OpenAL runtime for positional MCSE/MCNK emitters, `SoundEntries` preview, Alpha-area audio catalog inspection
- **Camera**: Path authoring, cross-era `.m2` camera import, native M2 export, capture automation
- **ML Datasets**: v50/v60 terrain tensor extraction into Zarr, synthesized minimaps, and model training pipeline

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

## Supported Game Versions and proof boundary

| Version | Era | Support |
|---------|-----|---------|
| 0.5.3 | Alpha | Terrain/WMO/MDX/BLP/DBC routes; MDX visual coverage is not fully release-proven |
| 0.6.0–0.10.x | Alpha/pre-release | Early terrain and chunked-model routes with partial client-specific coverage |
| 1.12.1 | Vanilla | Era-specific M2 route; broad in-viewer visual proof remains incomplete |
| 2.x | TBC | Profiled embedded-native M2 route implemented; material/visibility proof pending |
| 3.0.x | WotLK transition | Profiled early-M2 route implemented; visual coverage provisional |
| 3.3.5 | WotLK | Strongest current late-client terrain/WMO/M2/PM4 path |
| 4.0.0+ | Cataclysm | ADT/WMO/PM4 and terrain reconstruction paths; client-specific rendering/performance partial |

## Branches and releases

- `main` — **Current trunk.** Brought up to the v0.5.2 release line on 2026-08-15; it had previously
  lagged behind the working branches. New work branches from here.
- Releases are cut by pushing a `v*` tag. That triggers
  [`wowviewer-release.yml`](.github/workflows/wowviewer-release.yml), which builds all four platforms
  and publishes a GitHub Release using `wow-viewer/docs/releases/<tag>.md` as the notes — add that
  file before tagging.
- `main-pre-v0.5.2` — Snapshot of the previous `main` tip, kept for history.
- `v0.4.9`, `v0.4.5` — Older release branches from the legacy MdxViewer era.
- Numbered branches (e.g. `151-portal-game-mode-surface`) are per-spec working branches.

## Disclaimer

This project is not an official Blizzard Entertainment product and is not affiliated with or endorsed by Blizzard Entertainment or World of Warcraft.
