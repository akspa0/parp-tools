# Tech Context

## Stack
- **Runtime**: .NET 9, C#
- **Build**: `dotnet build` / `dotnet run`
- **Output**: Console CLI tools

## Key Projects

| Project | Purpose |
|---------|---------|
| `WoWRollback.Core` | Shared format library |
| `WoWRollback.Cli` | Main CLI entry point |
| `WoWRollback.PM4Module` | ADT merger, PM4 tools, MCCV painting |
| `AlphaWdtInspector` | Standalone diagnostics |
| `BlpResizer` | Texture conversion |
| `DBCTool.V2` | DBC/crosswalk generation |

## Critical Files

### ADT Merge/Generation
- `WoWRollback.PM4Module/AdtPatcher.cs` — ✅ Single source of truth for merging
- `WoWRollback.PM4Module/MccvPainter.cs` — ✅ Minimap→MCCV conversion
- `WoWRollback.PM4Module/WdlToAdtProgram.cs` — WDL→ADT CLI with `--minimap` support
- `WoWRollback.PM4Module/WdlToAdtTest.cs` — ADT generation logic

### Format Specs
- `memory-bank/specs/Alpha-0.5.3-Format.md` — Definitive Alpha spec
- `memory-bank/coding_standards.md` — FourCC handling rules

## Reference Libraries (USE THESE!)

These libraries provide battle-tested ADT parsing/writing:

| Library | Path | Key Classes |
|---------|------|-------------|
| **Warcraft.NET** | `lib/Warcraft.NET/` | `Files.ADT.Terrain.Wotlk.Terrain`, `MCNK`, all chunk types |
| **MapUpconverter** | `lib/MapUpconverter/` | `ADT/Tex0.cs`, `ADT/Root.cs`, `ADT/Obj0.cs` |
| **WoWFormatLib** | `lib/wow.tools.local/WoWFormatLib/` | Additional format utilities |

### How MapUpconverter Works
- Parses WotLK monolithic ADT using `Warcraft.NET.Files.ADT.Terrain.Wotlk.Terrain`
- Extracts texture data → creates `_tex0.adt` (Legion/BfA format)
- Extracts object data → creates `_obj0.adt`
- **We need the REVERSE**: parse split files → combine → write monolithic

## External Dependencies
- **StormLib**: MPQ reading
- **WoWFormatLib**: CASC support
- **DBCD**: DBC parsing via `lib/wow.tools.local`
- **SixLabors.ImageSharp**: Image loading for MCCV painting

## Test Data Locations
- `test_data/development/` — Development map split ADTs (root + _obj0 + _tex0)
- `test_data/WoWMuseum/335-dev/` — Reference monolithic 3.3.5 ADTs
- `test_data/development/World/Textures/Minimap/` — Minimap PNGs for MCCV
- `PM4ADTs/clean/` — Merged ADTs (352 tiles)
- `PM4ADTs/wdl_generated/` — Gap-fill terrain (1144 tiles)
- `test_output/mccv_test/` — WDL→ADT with MCCV painting (1496 tiles)
