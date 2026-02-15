# Tech Context

## Stack
- **Runtime**: .NET 10, C#
- **Build**: `dotnet build` / `dotnet run`
- **Graphics**: OpenGL 3.3 via Silk.NET
- **UI**: ImGui via ImGui.NET
- **MPQ**: Native C# MPQ service in active viewer path (StormLib remains present in broader repo tooling)

## Key Projects

| Project | Purpose | Status |
|---------|---------|--------|
| **`src/MdxViewer`** | **Primary** — 3D world viewer for 0.5.3, 0.6.0, 3.3.5 | Active development |
| `src/MDX-L_Tool` | Alpha 0.5.3 Model Archaeology (MDX to M2/OBJ/MDL) | Stable |
| `src/WoWMapConverter` | Map conversion, VLM, ADT format library | Stable |
| `WoWRollback.Core` | Shared format library | Stable |
| `WoWRollback.Cli` | Main CLI entry point | Stable |
| `WoWRollback.PM4Module` | ADT merger, PM4 tools, MCCV painting | Stable |
| `DBCTool.V2` | DBC/crosswalk generation | Stable |
| `BlpResizer` | Texture conversion | Stable |

## Critical Files — MdxViewer

### Rendering
- `src/MdxViewer/Rendering/ModelRenderer.cs` — MDX GPU rendering (two-pass, blend modes)
- `src/MdxViewer/Rendering/WmoRenderer.cs` — WMO GPU rendering (4-pass, liquid, doodads)
- `src/MdxViewer/Terrain/LiquidRenderer.cs` — MCLQ/MLIQ liquid mesh rendering

### Terrain & World
- `src/MdxViewer/Terrain/WorldScene.cs` — Placement transforms, instance management, culling
- `src/MdxViewer/Terrain/TerrainManager.cs` — AOI streaming, persistent cache, MPQ throttling
- `src/MdxViewer/Terrain/AlphaTerrainAdapter.cs` — Alpha 0.5.3 monolithic WDT terrain
- `src/MdxViewer/Terrain/StandardTerrainAdapter.cs` — 0.6.0 / 3.3.5 split ADT terrain + WMO-only maps
- `src/MdxViewer/Terrain/WorldAssetManager.cs` — MDX/WMO asset loading + caching

### Data & Services
- `src/MdxViewer/DataSources/MpqDataSource.cs` — MPQ reading + FindInFileSet
- `src/MdxViewer/Services/LightService.cs` — DBC Light/LightData zone-based lighting
- `src/MdxViewer/Services/AreaTableService.cs` — AreaID → name with MapID filtering
- `src/MdxViewer/Services/ReplaceableTextureResolver.cs` — DBC-based texture resolution

### Format Parsers
- `src/WoWMapConverter/WoWMapConverter.Core/Formats/LichKing/Mcnk.cs` — MCNK chunk parser (0.6.0/3.3.5)
- `src/WoWMapConverter/WoWMapConverter.Core/Converters/WmoV14ToV17Converter.cs` — WMO parser

## External Dependencies
- **Silk.NET**: OpenGL + windowing + input
- **ImGui.NET**: Immediate-mode GUI
- **SereniaBLPLib**: BLP texture decoding
- **SixLabors.ImageSharp**: Image processing
- **StormLib**: MPQ reading (native DLL)
- **DBCD**: DBC parsing via `lib/wow.tools.local`
- **wow-mdx-viewer (reference implementation)**: `lib/wow-mdx-viewer` cloned as behavioral source-of-truth for MDX version-compat parser parity (GEOS/PRE2/RIBB/SEQS routing).

## Test Data Locations
- `test_data/0.5.3/` — Alpha 0.5.3 Reference Assets (MDX, BLP, DBC)
- `test_data/development/` — Development map split ADTs + PM4 files
- `PM4ADTs/clean/` — Merged ADTs (352 tiles)
- `DBCTool/out/0.5.3/` — CSV exports for CreatureDisplayInfo & Extra
