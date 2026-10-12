# Tech Context — wow-viewer

Last verified against `WowViewer.slnx`, `Directory.Build.props`, `.gitmodules` and source: 2026-09-23.

## Stack

- **Runtime**: .NET 10 (`net10.0`), `LangVersion` = `preview` (`Directory.Build.props`)
- **Build**: `dotnet build wow-viewer/WowViewer.slnx -c Debug`
- **Graphics**: OpenGL 3.3 via Silk.NET; Vulkan/WebGL are future
- **UI**: ImGui via ImGui.NET (viewer app only)
- **Testing**: xUnit — `WowViewer.Core.Tests`, `WowViewer.Core.PM4.Tests`, `WowViewer.Core.Anim.Tests`,
  `WowViewer.Core.Curation.Tests`, `WowViewer.Core.Editor.Tests`
- **Python**: uv-managed environment at `wow-viewer/data-harvester/` (PyTorch, Zarr, TensorStore)
- **Storage**: Zarr v3 for dataset stores (Python owns datastore IO; C# emits blobs), Parquet for
  index/metadata, plain files for cache and exports

## Key Projects

| Project | Purpose | Location |
|---------|---------|----------|
| **WowViewer** | Primary 3D world viewer app (the only UI project) | `src/viewer/WoWViewer/` |
| `WowViewer.App.Defunct` | Quarantined legacy MdxViewer-era bootstrap code | `src/viewer/WowViewer.App.Defunct/` |
| `WowViewer.Core` | Shared data models | `src/core/WowViewer.Core/` |
| `WowViewer.Core.IO` | Format readers/writers (M2/MDX, WDT/ADT incl. DAT/AHDR, WMO, BLP, DBC/DB2, PM4), MPQ + CASC | `src/core/WowViewer.Core.IO/` |
| `WowViewer.Core.Runtime` | M2 skin/profile runtime, animation sampling, world/scene-graph contracts | `src/core/WowViewer.Core.Runtime/` |
| `WowViewer.Core.Renderer` | OpenGL rendering library used by the headless tools (Capture/Harvest/ValidationCapture), not by the viewer app | `src/core/WowViewer.Core.Renderer/` |
| `WowViewer.Core.PM4` | PM4 chunk models, research analyzers, matching | `src/core/WowViewer.Core.PM4/` |
| `WowViewer.Core.Editor` | Editor plugin host, session, bridge, operations, integrity, staging | `src/core/WowViewer.Core.Editor/` |
| `WowViewer.Core.Curation` | Dataset curation/bucketing (Parquet) | `src/core/WowViewer.Core.Curation/` |
| `WowViewer.Core.Anim` | M2 animation loader layer | `src/core/WowViewer.Core.Anim/` |
| `WowViewer.Tools.Shared` | Shared CLI support | `src/tools-shared/WowViewer.Tools.Shared/` |
| `WowViewer.Tool.Inspect` | CLI inspector (m2, mdx, blp, map, pm4, wmo, archive, casc, adt-ahdr, …) | `tools/inspect/` |
| `WowViewer.Tool.Converter` | CLI format converter (ADT/WDT round-trip) | `tools/converter/` |
| `WowViewer.Tool.Harvest` | CLI terrain tensor harvester | `tools/harvest/` |
| `WowViewer.Tool.Capture` / `ValidationCapture` | Headless capture | `tools/capture/`, `tools/validation-capture/` |
| `WowViewer.Tool.MaskValidate` · `WdlRead` · `WmoMinimap` | Focused CLIs | `tools/mask-validate/`, `tools/wdl-read/`, `tools/wmo-minimap/` |
| `WowViewer.Tool.V22Enrich` | On disk, **not** in `WowViewer.slnx` | `tools/enrich/` |
| ~~`WowViewer.Tool.AnimFarm`~~ | **Does not exist.** Spec 053 stopped after its loader layer; `tools/animfarm/` is an empty solution folder. | — |
| Data harvester (Python) | Dataset building, model training/inference | `data-harvester/` |

## Format Readers

| Format | Reader | Status |
|--------|--------|--------|
| M2 classic MD20 (3.x) | `M2ModelReader` (`Core.IO/M2`) | Complete (3.3.5) |
| M2 MD20 1.x (`0x100`–`0x107`) | `M2Era100ModelReader` via `M2ModelReaderDispatcher` | Implemented 2026-09-11 (Spec 235, archived); operator visual gate owed |
| M2 era 1121 | `M2Era1121ModelReader` | Implemented |
| M2 chunked (MDLX / MD21) | `M2ChunkedModelReader`, `WarcraftNetM2Adapter` | Implemented; modern MD21 via CASC |
| M2 external anim | `M2AnimationReader` | Complete |
| MDX | `MdxFile.Load` (`Core.IO/Mdx`) | Complete |
| WDT/ADT (Alpha + LK + MoP split + FDID) | `AlphaWdtReader`, `AdtSummaryReader`, `MopAdtChunkParser`; adapters in `systemPatterns.md` | Complete (MoP native split **writer** disabled) |
| DAT v22/v23/v26 (AHDR family) | `AdtAhdrReader` / `AdtAhdrWriter` | Read + render + v26 byte-identical rewrite; v22 `AMAP` codec unidentified |
| WMO (V14/V17 + modern) | `WmoSummaryReader` + ~15 detail readers (`WmoGroupMeshDetailReader`, `WmoMaterialDetailReader`, `WmoPortalDetailReader`, `WmoDoodadDetailReader`, …) aggregated by `WmoRenderDocumentReader` | Complete |
| BLP | `BlpPixelDecoder`, `BlpSummaryReader` | Complete |
| DBC/DB2 | `DbcReader`, `DbcTableLoader`, `ArchiveReaderDbcProvider`, table readers in `Core.IO/Dbc/`; `DbClientFileReader`; DB2 by id via DBCD + WoWDBDefs | Complete |
| PM4 | `Pm4DocumentReader` (via Pm4Research) | Complete (field semantics partly open — Epic 253) |
| MPQ | `MpqArchiveCatalog`, `AlphaArchiveReader`; `NativeMpqService` (alternative path) | Complete |
| CASC | `CascStorage`, `CascArchiveCatalog` (`Core.IO/Casc`) + `CascDataSource` | Local install (+CDN fill); CDN-only streaming absent |
| `.phys` (5.0.1) | `PhysReader` | Implemented (Spec 214, archived) |

## External Dependencies

- **Silk.NET**: OpenGL + windowing + input
- **ImGui.NET**: Immediate-mode GUI
- **SereniaBLPLib**: BLP texture decoding
- **SixLabors.ImageSharp**: Image processing
- **DBCD** (`libs/wowdev/DBCD`) + **WoWDBDefs** (`libs/wowdev/WoWDBDefs`, submodule): DBC/DB2 parsing
- **TACTSharp** (`libs/wowdev/TACTSharp`, submodule): CASC/TACT access
- **wow-listfile** (`libs/wowdev/wow-listfile`), **WoWTools.Minimaps** (`libs/Marlamin/WoWTools.Minimaps`): submodules
- **Warcraft.NET**: WMO/M2 format helpers
- **GillijimProject** (`libs/WoW-Tools/GillijimProject`): vendored source, **not** a registered submodule
- `libs/alpha-core`: vendored server-side reference tree (own `.git`, not a registered submodule)
- **PyTorch**, **Zarr-Python**, **TensorStore**: Python training and dataset stores
