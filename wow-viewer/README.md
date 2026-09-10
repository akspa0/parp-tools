# WoWViewer Toolkit (`wow-viewer`)

The primary development and runtime workspace for `parp-tools`. This project provides a library-first .NET 10 architecture for parsing, inspecting, rendering, and reconciling World of Warcraft client data across the active project eras, primarily Alpha 0.5.3 through WotLK 3.3.5a.

---

## What's Included

### 1. 3D World Viewer (`src/viewer/WoWViewer`)
An interactive desktop viewer built with Silk.NET, OpenGL, and ImGui:
- **Terrain Streaming**: Bounded camera-centered tile streaming with directional lookahead and WDL low-detail horizon fallback.
- **WMO Portals & Interiors**: Hierarchical culling with bounded portal traversal and group-level admission.
- **M2/MDX Skeletal Animation**: Multi-track sequence evaluation, bone poses, particle systems, and ribbons.
- **Dynamic Lighting & Skies**: Native Alpha 0.5.3 2,880-unit world clock, Light DBC fallback, LIT profiles, and fog.
- **Positional Audio**: OpenAL runtime with resident MCSE / MCNK positional emitters and `SoundEntries` preview.
- **Camera Path Authoring**: Full timeline keyframing, roll control, JSON project save/load, and native `.m2` camera export.
- **PM4 Reconciliation Workbench**: Spec 176 reconciliation panel for matching real PM4 geometry segments to ADT placements with guarded preview, hash verification, and undoable apply operations.
- **Cartography Composition (Spec 232)**: multi-layer map composition — donor tiles, placement locks, rotation/mirror, Z transform, WDL edge-snap — with project save/load.
- **Map Save & New Map Creator (Spec 234, spec authored 2026-09-09)**: planned save of merged/composed maps to Alpha 0.5.3 WDT and LK v18 ADT from both Archaeology and the Editor's Data I/O page, plus a New Map creator in the Editor tab. Multi-map support is explicitly deferred to a future spec. See [specs/234-map-save-new-map/spec.md](specs/234-map-save-new-map/spec.md).

### 2. Shared Core Libraries (`src/core`)
- **`WowViewer.Core`**: Data contracts, vertex/index buffers, terrain tensor layouts, and coordinate transformations.
- **`WowViewer.Core.IO`**: Binary readers and writers for ADT, WDT, WDL, WMO (v14/v17), supported-era M2/MDX layouts, BLP, LIT, and MPQ archives.
- **`WowViewer.Core.PM4`**: Deep PM4 chunk decoding (MSCN, MSPV, MSUR, MSHD, MSLK), geometry segment construction, and signature extractors.
- **`WowViewer.Core.Runtime`**: M2 animation evaluators, bone matrices, and skin profile resolvers.
- **`WowViewer.Core.Editor`**: Transactional placement manipulation, history stacks, ID allocation with high-water marks, and provenance metadata tracking.

### 3. CLI Tools (`tools/`)
- **`wowviewer-inspect` (`tools/inspect`)**: Comprehensive format inspector, PM4 segment exporter, audio analyzer, and Rosetta Calibration Corpus generator (`rosetta-generate`).
- **`wowviewer-converter` (`tools/converter`)**: Bidirectional converter between Alpha monolithic WDT maps and modern LK ADT/WDT directories.
- **`wowviewer-harvest` (`tools/harvest`)**: High-throughput extraction of terrain tensors (elevation, normals, alpha masks) into Zarr/NPZ archives and synthetic minimap images.
- **`wowviewer-capture` (`tools/capture`) & `validation-capture`**: Automated headless camera renders and regression visual captures.

---

## Documentation Map

- **[Core libraries](src/core/README.md)**: Ownership boundaries for shared models, I/O, runtime, PM4, editor, renderer, and curation code.
- **[Desktop viewer](src/viewer/WoWViewer/README.md)**: App launch, UI surfaces, and viewer-specific proof gates.
- **[CLI tooling](tools/README.md)**: Canonical single-document index for all command-line projects.
- **[Desktop user guide](docs/WoWViewer/USERGUIDE.md)**: End-user controls and workflows.
- **[Spec status router](specs/STATUS.md)**: Current SpecKit lanes, implementation state, and next bounded actions.

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
│  WoWViewer v0.5.2.2                                            [_][□][X]    │
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
| `convert-alpha-to-lk` | `WowViewer.Tool.Converter` | Convert 0.5.3 monolithic WDT maps into modern LK ADT/WDT files |
| `convert-lk-to-alpha` | `WowViewer.Tool.Converter` | Convert modern LK ADT maps into 0.5.3 monolithic WDT containers |
| `harvest-map` | `WowViewer.Tool.Harvest` | Extract terrain height, normals, and texture layers into tensor stores |
| `synthetic-minimap`| `WowViewer.Tool.Harvest` | Compose high-fidelity minimap images directly from raw terrain data |

See the **[CLI tooling README](tools/README.md)** for the canonical tool index, current command paths, safety boundaries, and examples. The older **[expanded CLI reference](docs/CLI-TOOLS.md)** remains available for deeper historical command notes.

---

## Client Era Support Matrix

| Client Era | Version | Current State | Key Implemented Features / Notes |
|---|---|---|---|
| **Alpha** | 0.5.3 - 0.5.5 | **Implemented with active proof gates** | Monolithic WDTs, v14 WMO monoliths, MDX/MDL models, 2880-unit world clock, Alpha audio catalog; Rosetta minimap visual proof remains operator-owned |
| **Early Beta** | 0.6.x - 0.10.x | **Partial / research** | Split ADT/WDT transition format, chunked early models, prototype map layouts |
| **Classic** | 1.12.1 | **Implemented surfaces, proof-gated** | Standard ADTs with MCCV/MCLY/MCAL, v17 WMOs, 2004-era M2 structures, AreaTable routing |
| **TBC** | 2.4.3 | **Implemented surfaces, proof-gated** | Embedded skin profiles, expanded WMO materials, multi-layer liquid chunks |
| **WotLK** | 3.3.5a | **Primary reference era** | Reference LK terrain format, separated M2 `.skin` files, PM4 analysis workflows, WDL terrain horizons |

Later client terrain formats are outside the current project scope unless a future spec explicitly reopens that lane.

---

## Safety and Architecture Rules

1. **Library-First Architecture**: Format readers/writers (`WowViewer.Core.IO`), domain algorithms, and edit policies live strictly in `src/core/` and contain no UI or viewer dependencies.
2. **Configuration vs Code**: Client paths are runtime configuration (`--game-path` or UI selections). Machine-local absolute paths are never hardcoded in source.
3. **Non-Destructive Storage**: Placement modifications and conversions write to designated output paths (`output/projects/...`) with source-hash verification and JSON provenance sidecars.
4. **Base Tooling Preservation**: New generators, probes, captures, and experiments layer above proven readers, writers, terrain loading, camera, and renderer paths unless an explicit spec reopens a base compatibility bug.
