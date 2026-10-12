# WoWViewer Toolkit (`wow-viewer`)

The primary development and runtime workspace for `parp-tools`. This project provides a library-first .NET 10 architecture for parsing, inspecting, rendering, and reconciling World of Warcraft client data. **Current development version: v0.6.1-dev**, following the first stable v0.6.0 release. Coverage includes clients **0.5.3 – 0.11**; working support across **1.x – 4.x**; **5.0.1 tested**; and modern CASC tested on **WoW: Forever** (`wow_classic_beta` 1.60.1, build 703xx+, based on 11.2.7). The 12.x M2 path is Warcraft.NET-backed; support remains format/build-specific.

> **Bring your own data.** parp-tools ships no game data. Point it at your own client install, MPQ set, CASC install or extracted map folder.

---

## What's Included

### 1. 3D World Viewer (`src/viewer/WoWViewer`)
An interactive desktop viewer built with Silk.NET, OpenGL, and ImGui:
- **Modern CASC Clients**: File → Open CASC Install (local, or local + CDN fill), then pick the game version. Tested with WoW: Forever (`wow_classic_beta` 1.60.1.69876); works with every build since support landed. Covers FileDataID-era maps (`MAID`, `MDID`, FileDataID placements), chunked `MD21` M2s, `GFID`/`MODI` WMOs with modern per-LOD groups and materials, and DB2 tables. Terrain blends up to 8 tileset layers per chunk with 64-bit high-resolution holes.
- **Archaeology: phase maps & map layering**: the Archaeology tab lists the DBC/DB2-defined phase maps for the loaded base map when the client has them; otherwise any map can be loaded on top of any other map.
- **Development-map restoration (partial)**: PM4 collision meshes are matched to museum object placements (persistent association ledger), minimap-only terrain is baked to vertex colour, and restored tiles are written as monolithic LK 3.3.5 ADTs. Objects on tiles adjacent to PM4-covered tiles are still missing; see the [restoration guide](docs/WoWViewer/development-map-restoration.md).
- **DAT v26 Terrain (alpha)**: File → Open DAT v26 Terrain Folder renders the raw terrain project files that shipped in that build — heights (inches), normals, vertex colours, layers and `ACDO` objects — and File → Export Nearby Tiles as DAT v26 (experimental) writes them back ([format write-up](docs/architecture/adt-v26-format.md)).
- **Terrain Streaming**: Bounded camera-centered tile streaming with directional lookahead and WDL low-detail horizon fallback.
- **WMO Portals & Interiors**: Hierarchical culling with bounded portal traversal and group-level admission.
- **M2/MDX Skeletal Animation**: Multi-track sequence evaluation, bone poses, particle systems, and ribbons.
- **Dynamic Lighting & Skies**: Native Alpha 0.5.3 2,880-unit world clock, Light DBC fallback, LIT profiles, and fog.
- **Positional Audio**: OpenAL runtime with resident MCSE / MCNK positional emitters and `SoundEntries` preview.
- **Camera Path Authoring**: Full timeline keyframing, roll control, JSON project save/load, and native `.m2` camera export.
- **PM4 Reconciliation Workbench**: Match real PM4 geometry segments to ADT placements with guarded preview, hash verification, and undoable apply operations.
- **Cartography Composition**: Multi-layer map composition with donor tiles, placement locks, rotation/mirror, Z transform, WDL edge-snap, and project save/load.
- **Map Save & New Map Creator**: A New Map creator service exists; **saving composed/merged maps to Alpha 0.5.3 WDT / LK v18 ADT is not implemented yet**.

### 2. Shared Core Libraries (`src/core`)
- **`WowViewer.Core`**: Data contracts, vertex/index buffers, terrain tensor layouts, and coordinate transformations.
- **`WowViewer.Core.IO`**: Binary readers and writers for ADT, WDT, WDL, WMO (v14/v17), supported-era M2/MDX layouts, BLP, LIT, MPQ archives, CASC installs (via TACTSharp), and DAT v26 terrain.
- **`WowViewer.Core.PM4`**: Deep PM4 chunk decoding (MSCN, MSPV, MSUR, MSHD, MSLK), geometry segment construction, and signature extractors.
- **`WowViewer.Core.Runtime`**: M2 animation evaluators, bone matrices, and skin profile resolvers.
- **`WowViewer.Core.Editor`**: Transactional placement manipulation, history stacks, ID allocation with high-water marks, and provenance metadata tracking.

### 3. CLI Tools (`tools/`)
- **`wowviewer-inspect` (`tools/inspect`)**: Comprehensive format inspector, PM4 segment exporter, audio analyzer, and Rosetta Calibration Corpus generator (`rosetta-generate`).
- **`wowviewer-converter` (`tools/converter`)**: Bidirectional converter between Alpha monolithic WDT maps and modern LK ADT/WDT directories.
- **`wowviewer-harvest` (`tools/harvest`)**: High-throughput extraction of terrain tensors (elevation, normals, alpha masks) into Zarr/NPZ archives and synthetic minimap images.
- **`wowviewer-capture` (`tools/capture`) & `validation-capture`**: Automated headless camera renders and regression visual captures.

## Pre-Alpha Restoration Project (PARP)

PARP — the Pre-Alpha Restoration Project — works to recover World of Warcraft's pre-2003 game world from surviving files and historical evidence. Current development-map recovery combines museum-era base terrain and placements with PM4 collision data and minimap-derived colors; the restoration is partial, with known placement and terrain gaps. Longer-term reconstruction aims to use early-game screenshots and even minimaps printed in period magazines to recover lost geography, cross-checking those sources against surviving client and map data. See the [development-map restoration guide](docs/WoWViewer/development-map-restoration.md) for the current tooling workflow.

---

## Documentation Map

- **[Core libraries](src/core/README.md)**: Ownership boundaries for shared models, I/O, runtime, PM4, editor, renderer, and curation code.
- **[Desktop viewer](src/viewer/WoWViewer/README.md)**: App launch, UI surfaces, and viewer-specific proof gates.
- **[CLI tooling](tools/README.md)**: Canonical single-document index for all command-line projects.
- **[Desktop user guide](docs/WoWViewer/USERGUIDE.md)**: End-user controls and workflows.
- **[Development map restoration guide](docs/WoWViewer/development-map-restoration.md)**: Inputs, safe restore commands, validation, and viewer overlay setup.

---

## Building and Running

### Build Solution
```powershell
# Build entire solution in Debug
dotnet build wow-viewer/WowViewer.slnx -c Debug

# Run all unit tests
dotnet test wow-viewer/WowViewer.slnx -c Debug
```

### Run 3D Desktop Viewer
```powershell
# Default launch
dotnet run --project wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug

# Direct launch into a specific client map
dotnet run --project wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug -- `
  --game-path "H:\CLIENTS\WoW-0.5.3.3368-Client" `
  --world "World\Maps\Kalimdor\Kalimdor.wdt"
```

### Run CLI Format Inspector
```powershell
# Inspect any M2, MDX, WMO, BLP, ADT, or PM4 file
dotnet run --project wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -c Debug -- `
  m2 inspect --input "Creature/Murloc/Murloc.m2"
```

### Run Rosetta Calibration Corpus Generator
```powershell
# Generates calibration terrain with 4m pedestals and sharp 1024x1024 MCAL text labels
dotnet run --project wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -c Debug -- `
  rosetta-generate `
  --client-root "H:\CLIENTS\WoW-0.5.3.3368-Client" `
  --output "output/rosetta_053" `
  --map-name "RosettaAlpha" `
  --pedestal-height 4.0 `
  --pedestal-bevel 12.5
```

---

## Desktop Viewer UI Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  WoWViewer v0.6.1-dev                                           [_][□][X]    │
├──────────────┬───────────────────────────────────────────────┬──────────────┤
│  NAVIGATOR   │                                               │  WORKBENCH   │
│  (Left Bar)  │               3D VIEWPORT                     │  (Right Bar) │
│              │                                               │              │
│ • Client     │  [WASD + Mouse Fly Camera]                    │ Destinations:│
│   Sources    │                                               │ • Quick      │
│ • File Tree  │  [Bounded Terrain Stream Ring: Radius 2-3]    │ • Inspect    │
│ • World Maps │                                               │ • Scene      │
│ • Phase Maps │  [Dynamic Lighting, Sky & Fog]                │ • Utilities  │
│              │                                               │ • Experiment │
├──────────────┴───────────────────────────────────────────────┴──────────────┤
│ [Scene Toggles]   [Subzone / Area Readout]   [Audio: ON/MUTED]   [FPS / ms] │
└─────────────────────────────────────────────────────────────────────────────┘
```

See the **[Desktop Viewer User Guide](docs/WoWViewer/USERGUIDE.md)** for a full manual detailing every keybinding, workbench destination, and workflow.

---

## CLI Tools Overview

| Command | Tool Project | Purpose |
|---|---|---|
| `m2 inspect` | `WowViewer.Tool.Inspect` | Inspect M2/MDX bones, sequences, attachments, and bounding boxes |
| `pm4 inspect` | `WowViewer.Tool.Inspect` | Dump and audit PM4 MSCN/MSPV/MSUR geometry and linkages |
| `rosetta-generate` | `WowViewer.Tool.Inspect` | Generate synthetic museum calibration maps with MCAL text and pedestals |
| `rosetta-datastore-info` | `WowViewer.Tool.Inspect` | Inspect registered builds, maps, and deduplication statistics in Zarr datastore |
| `rosetta-datastore-diff` | `WowViewer.Tool.Inspect` | Compute cross-build asset additions, removals, format migrations, and geometry changes |
| `map inspect` | `WowViewer.Tool.Inspect` | Analyze ADT/WDT terrain chunks, layers, and bounding boxes |
| `casc products\|read\|exists\|wmo\|m2\|db2\|map-survey\|adt-heights\|bench` | `WowViewer.Tool.Inspect` | Open a local CASC install: list products, read by path or FileDataID, validate WMO/M2/DB2 decoding, survey a map, benchmark loading |
| `adt-ahdr check` | `WowViewer.Tool.Inspect` | Decode a folder of DAT v26 terrain files and verify tile seams |
| `convert-alpha-to-lk` | `WowViewer.Tool.Converter` | Convert 0.5.3 monolithic WDT maps into modern LK ADT/WDT files |
| `convert-lk-to-alpha` | `WowViewer.Tool.Converter` | Convert modern LK ADT maps into 0.5.3 monolithic WDT containers |
| `harvest-map` | `WowViewer.Tool.Harvest` | Extract terrain height, normals, and texture layers into tensor stores |
| `synthetic-minimap`| `WowViewer.Tool.Harvest` | Compose high-fidelity minimap images directly from raw terrain data |

See the **[CLI tooling README](tools/README.md)** for the canonical tool index, current command paths, safety boundaries, and examples. The older **[expanded CLI reference](docs/CLI-TOOLS.md)** remains available for deeper historical command notes.

---

## Client Era Support Matrix

Alpha 0.5.3–0.5.5 uses a distinct monolithic WDT map format; later clients use split map layouts.

| Client Era | Version | Current State | Key Implemented Features / Notes |
|---|---|---|---|
| **Alpha** | 0.5.3 - 0.5.5 | **Implemented with active proof gates** | Distinct monolithic WDT map format, v14 WMO monoliths, MDX/MDL models, 2880-unit world clock, Alpha audio catalog; Rosetta minimap visual proof remains operator-owned |
| **Early Beta** | 0.6.x - 0.11.x | **Supported** | Split ADT/WDT transition formats, early models, prototype map layouts |
| **Classic** | 1.x (1.12.1 tested) | **Working** | Standard ADTs with MCCV/MCLY/MCAL, v17 WMOs, M2 structures, AreaTable routing |
| **TBC** | 2.x (2.4.3 tested) | **Working** | Embedded skin profiles, expanded WMO materials, multi-layer liquid chunks |
| **WotLK** | 3.x (3.3.5a reference) | **Working / primary reference** | Reference LK terrain format, separated M2 `.skin` files, PM4 analysis workflows, WDL terrain horizons |
| **Cataclysm** | 4.x | **Working** | Later M2/WMO layouts and split terrain formats |
| **Later legacy** | Through 5.0.1 | **5.0.1 tested; overall partial** | Format-family coverage varies across later legacy builds |
| **Modern CASC** | WoW: Forever 1.60.1 (build 703xx+) | **Tested** | `wow_classic_beta` is 11.2.7-based; `MAID`/`MDID` terrain, FileDataID placements, `MD21` M2, `GFID`/`MODI` WMO, DB2, and DAT v26 terrain project files; 12.x M2 parsing uses Warcraft.NET |

**Support summary (v0.6.1-dev):** 0.5.3–0.11 complete; client coverage works across 1.x–4.x; later 0.12–5.x is partial, with 5.0.1 tested; modern CASC tested on WoW: Forever 1.60.1 (11.2.7-based, build 703xx+).

The M2 reader path for 12.x data uses Warcraft.NET; per-format and per-build validation remains distinct from the tested 1.60.1 and 703xx+ builds.

---

## Safety and Architecture Rules

1. **Library-First Architecture**: Format readers/writers (`WowViewer.Core.IO`), domain algorithms, and edit policies live strictly in `src/core/` and contain no UI or viewer dependencies.
2. **Configuration vs Code**: Client paths are runtime configuration (`--game-path` or UI selections). Machine-local absolute paths are never hardcoded in source.
3. **Non-Destructive Storage**: Placement modifications and conversions write to designated output paths (`output/projects/...`) with source-hash verification and JSON provenance sidecars.
4. **Base Tooling Preservation**: New generators, probes, captures, and experiments layer above proven readers, writers, terrain loading, camera, and renderer paths unless a separately reviewed compatibility fix is approved.

---

## Community & Acknowledgments

Special thanks to **Kruithne** for [wow.export](https://github.com/Kruithne/wow.export), whose tooling, format research, and community contributions have been a tremendous reference and inspiration for modern World of Warcraft asset extraction and analysis.
We also gratefully acknowledge Marlamin, schlumpf, Dovah, Pirate the Explorer, fean, implave, IS4, Mjollna, Adspartan (Noggit), Skarn (Noggit-Red), and the entire WoW Exploration & Pre-Alpha Restoration communities.
