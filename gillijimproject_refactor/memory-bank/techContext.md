# Tech Context

## Stack
- **Runtime**: .NET 9, C#
- **Build**: `dotnet build` / `dotnet run`
- **Output**: Console CLI tools

## Key Projects

| Project | Purpose |
|---------|---------|
| `MDX-L_Tool` | **Alpha 0.5.3** Model Archaeology (MDX to M2/OBJ/MDL) |
| `WoWRollback.Core` | Shared format library |
| `WoWRollback.Cli` | Main CLI entry point |
| `WoWRollback.PM4Module` | ADT merger, PM4 tools, MCCV painting |
| `AlphaWdtInspector` | Standalone diagnostics |
| `BlpResizer` | Texture conversion |
| `DBCTool.V2` | DBC/crosswalk generation |

## Critical Files

### MdxViewer (3D World Viewer)
- `src/MdxViewer/Rendering/ModelRenderer.cs` — MDX GPU rendering (two-pass, blend modes)
- `src/MdxViewer/Rendering/WmoRenderer.cs` — WMO GPU rendering (4-pass, liquid, doodads)
- `src/MdxViewer/Terrain/WorldScene.cs` — Placement transforms, instance management
- `src/MdxViewer/Terrain/WorldAssetManager.cs` — MDX/WMO asset loading + caching
- `src/WoWMapConverter/WoWMapConverter.Core/Converters/WmoV14ToV17Converter.cs` — WMO parser
- `src/MdxViewer/DataSources/MpqDataSource.cs` — MPQ reading + FindInFileSet

### MDX Archaeology
- `src/MDX-L_Tool/Formats/Mdx/MdxFile.cs` — Chunk scanner & padding handler
- `src/MDX-L_Tool/Services/DbcService.cs` — DBC variation and skin lookup
- `src/MDX-L_Tool/Services/TextureService.cs` — Integrated resolution pipeline
- `src/MDX-L_Tool/Formats/Obj/ObjWriter.cs` — Multi-body OBJ export logic

### Reference Implementations (in lib/)
- `lib/noggit-red/src/noggit/rendering/ModelRender.cpp` — Noggit M2 blend modes, render passes
- `lib/wow-mdx-viewer/src/renderer/model/modelRenderer.ts` — Barncastle MDX FilterMode/blend
- `lib/wow-mdx-viewer/src/renderer/model/particles.ts` — Barncastle particle system

### ADT Merge/Generation
- `WoWRollback.PM4Module/AdtPatcher.cs` — ✅ Single source of truth for merging

## External Dependencies
- **SereniaBLPLib**: BLP texture decoding
- **SixLabors.ImageSharp**: Image processing
- **StormLib**: MPQ reading
- **DBCD**: DBC parsing via `lib/wow.tools.local`

## Test Data Locations
- `test_data/0.5.3/` — Alpha 0.5.3 Reference Assets (MDX, BLP, DBC)
- `DBCTool/out/0.5.3/` — CSV exports for CreatureDisplayInfo & Extra
- `test_data/development/` — Development map split ADTs
- `PM4ADTs/clean/` — Merged ADTs (352 tiles)
