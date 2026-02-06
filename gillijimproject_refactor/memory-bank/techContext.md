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

### MDX Archaeology
- `src/MDX-L_Tool/Formats/Mdx/MdxFile.cs` — Chunk scanner & padding handler
- `src/MDX-L_Tool/Services/DbcService.cs` — DBC variation and skin lookup
- `src/MDX-L_Tool/Services/TextureService.cs` — Integrated resolution pipeline
- `src/MDX-L_Tool/Formats/Obj/ObjWriter.cs` — Multi-body OBJ export logic

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
