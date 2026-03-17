# Tech Context

## Stack
- .NET 10, C#, OpenGL 3.3 (Silk.NET), ImGui.NET, SereniaBLPLib, SixLabors.ImageSharp, DBCD
- Native C# MPQ service (no StormLib in active viewer path)
- `lib/wow-mdx-viewer` = reference impl for MDX parser parity

## Active Projects
- **`src/MdxViewer`** — Primary 3D viewer (0.5.3, 0.6.0, 3.3.5)
- `src/WoWMapConverter` — Format library + VLM + converters
- Other tools (MDX-L_Tool, WoWRollback, DBCTool, BlpResizer) are stable

## High-Risk Files (Terrain Alpha)
- `src/MdxViewer/Terrain/TerrainRenderer.cs`
- `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`
- `src/MdxViewer/Terrain/TerrainTileMeshBuilder.cs`
- `src/MdxViewer/Terrain/TerrainChunkData.cs`
- `src/MdxViewer/Export/TerrainImageIo.cs`
- `src/MdxViewer/ViewerApp.cs`
- `src/WoWMapConverter/WoWMapConverter.Core/Formats/LichKing/Mcal.cs`

## Test Data
- `test_data/0.5.3/` — Alpha reference assets
- `test_data/development/` — Split ADTs + PM4
- `test_data/WoWMuseum/335-dev/` — Archival 3.3.5 samples (parser ref only, not 3.x signoff)
