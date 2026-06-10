# Tech Context — wow-viewer

## Stack
- **Runtime**: .NET 10, C# 13
- **Build**: `dotnet build wow-viewer/WowViewer.slnx -c Debug`
- **Graphics**: OpenGL 3.3 via Silk.NET (WoWViewer); Vulkan/WebGL are future
- **UI**: ImGui via ImGui.NET
- **Testing**: xUnit (`tests/WowViewer.Core.Tests`, `tests/WowViewer.Core.PM4.Tests`, `tests/WowViewer.Core.Anim.Tests`)
- **Python**: uv-managed environment at `wow-viewer/data-harvester/` (PyTorch, Zarr, Blosc)
- **Storage**: Zarr v3 for dataset stores, Parquet for index/metadata, plain files for cache and exports

## Key Projects

| Project | Purpose | Location |
|---------|---------|----------|
| **WowViewer** | Primary 3D world viewer app | `src/viewer/WoWViewer/` |
| `WowViewer.Core` | Shared M2/MDX/MC/format data models | `src/core/WowViewer.Core/` |
| `WowViewer.Core.IO` | Format readers/writers (M2, WDT, ADT, WMO, BLP, DBC, PM4) | `src/core/WowViewer.Core.IO/` |
| `WowViewer.Core.Runtime` | M2 skin/profile runtime, M2 render pipeline | `src/core/WowViewer.Core.Runtime/` |
| `WowViewer.Core.PM4` | PM4 chunk models, research analyzers, matching | `src/core/WowViewer.Core.PM4/` |
| `WowViewer.Core.Anim` | M2 animation pose extraction library | `src/core/WowViewer.Core.Anim/` |
| `WowViewer.Tool.Inspect` | CLI format inspector (m2, mdx, blp, map, pm4, wmo, archive) | `tools/inspect/` |
| `WowViewer.Tool.Converter` | CLI format converter (ADT/WDT round-trip) | `tools/converter/` |
| `WowViewer.Tool.Harvest` | CLI terrain tensor harvester | `tools/harvest/` |
| `WowViewer.Tool.Capture` | CLI headless validation capture | `tools/capture/` |
| `WowViewer.Tool.AnimFarm` | CLI M2 animation pose farm (new, partial) | `tools/animfarm/` |
| Data harvester (Python) | V16/V18 dataset building, model training/inference | `data-harvester/` |

## Format Readers

| Format | Reader | Status |
|--------|--------|--------|
| M2 (classic MD20) | `M2ModelReader` in Core.IO/M2 | Complete (3.3.5) |
| M2 (chunked MDLX) | `M2ChunkedModelReader` in Core.IO/M2Chunked | In progress (spec 048) |
| M2 (era 1121) | `M2Era1121ModelReader` in Core.IO/M2Era1121 | Complete (spec 048) |
| M2 external anim | `M2AnimationReader` | Complete |
| MDX | `MdxFile.Load` in Core.IO/Mdx | Complete |
| WDT (Alpha + LK) | `AlphaWdtReader`, `AdtSummaryReader` | Complete |
| ADT (Alpha + LK) | `AlphaTerrainAdapter`, `StandardTerrainAdapter` | Complete |
| WMO (V14/V17) | `WmoDetailReader`, `WmoSummaryReader` | Complete |
| BLP | `BlpPixelDecoder`, `BlpSummaryReader` | Complete |
| DBC/DB2 | `DbcLookup`, `DbClientFileReader` | Complete |
| PM4 | `Pm4DocumentReader` (via Pm4Research) | Complete (spec 051) |
| MPQ | `MpqArchiveCatalog`, `AlphaArchiveReader` | Complete |

## External Dependencies
- **Silk.NET**: OpenGL + windowing + input
- **ImGui.NET**: Immediate-mode GUI
- **SereniaBLPLib**: BLP texture decoding
- **SixLabors.ImageSharp**: Image processing
- **DBCD**: DBC parsing
- **Warcraft.NET**: WMO/M2 format helpers (libs/ submodule)
- **GillijimProject**: Alpha WDT/ADT helpers (libs/ submodule)
- **PyTorch**: Python model training
- **Zarr-Python**: Python dataset stores
