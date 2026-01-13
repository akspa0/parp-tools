# WoWRollback Active Context

## Current Focus: VLM Terrain Data Export (Dec 2025)

### Session Summary (2025-12-26)

**Goal**: Export VLM dataset with terrain + placements for AI training.

**Status**: PARTIAL SUCCESS - Placements work, terrain still null

**What Works**:
- VLM Export exports minimap PNGs ✓
- Placements from MPQ (M2/WMO with UniqueID, position, rotation, scale) ✓
- JSON metadata files created with full placement data ✓
- CSV fallback for cached placements ✓

**What's Still Broken**:
- Terrain data is always `null` in JSON
- MpqAdtExtractor finds MPQ but terrain extraction returns null

**New Code Added (PM4Module)**:
- `MpqAdtExtractor.ExtractPlacements()` - MDDF/MODF parsing ✓
- `MpqAdtExtractor.ExtractTerrain()` - MTEX/MCNK/MCVT parsing (not returning data)
- `AlphaWdtExtractor.cs` - Alpha 0.5.3 WDT format support (NEW)
- `AdtDataService.cs` - Unified extraction API (NEW)
- Data models: `M2Placement`, `WmoPlacement`, `TileTerrainData`, `ChunkTerrainData`

**VLM Output Now Includes**:
```json
{
  "tile_id": "Azeroth_32_48",
  "placements": [
    { "type": "M2", "uniqueId": 12345, "posX": ..., "rotX": ..., "scale": ... },
    { "type": "WMO", "uniqueId": 67890, "doodadSet": 0, ... }
  ],
  "terrain": null  // <-- Still broken
}
```

**Next Steps to Fix Terrain**:
1. Port `src/WoWMapConverter/WoWMapConverter.Core/Services/AdtMeshExporter.cs` to `WoWRollback.MinimapModule/Services/TerrainMeshExporter.cs`.
2. ???
3. Generate OBJ/MTL files for each tile alongside the metadata JSON.
4. Update `VlmTrainingSample` to include `mesh_path`.

## [RATIFIED PLAN] VLM Terrain Data Export (2026-01-13)
**Objective**: "Insane Idea" - Correlate terrain meshes with minimap data for ML training.
**Strategy**:
- **Tooling**: `WoWRollback.MinimapModule` (VlmDatasetExporter).
- **Core Logic**: Port `WoWMapConverter.Core.Services.AdtMeshExporter` (proven logic) to a new `TerrainMeshExporter` service in `WoWRollback`.
- **Parsing**: Use `Warcraft.NET` (already a dependency) to read ADT `MCVT` (Height), `MCNR` (Normal - optional), and `Holes`.
- **Output**:
    - `images/{map}_{x}_{y}.png` (Minimap)
    - `metadata/{map}_{x}_{y}.json` (Metadata + mesh_path)
    - `meshes/{map}_{x}_{y}.obj` (New Terrain Mesh)
    - `meshes/{map}_{x}_{y}.mtl` (Material)

**Implementation Steps (2026-01-13 - COMPLETED)**:
1.  **Create Service**: `WoWRollback.MinimapModule/Services/TerrainMeshExporter.cs` created, porting logic from `ADTPreFabTool`.
2.  **Integrate**: Modified `VlmDatasetExporter.TryExtractAdtMetadata` to:
    - Parse ADT using `Warcraft.NET` (using `LowResHoles`).
    - Extract `Heights`, `Positions`, `Holes`.
    - Call `TerrainMeshExporter.ExportToObj`.
    - Generates `meshes/{map}_{x}_{y}.obj`.
    - Updates `VlmTrainingSample` with `mesh_path`.
3.  **Next**: Run export on a test map and inspect OBJ output.

---

## Key Extraction Code Paths

| Component | File | Status |
|-----------|------|--------|
| MPQ Reading | `PM4Module/MpqAdtExtractor.cs` | ✓ Working |
| Placements | `MpqAdtExtractor.ExtractPlacements()` | ✓ Working |
| Terrain | `MpqAdtExtractor.ExtractTerrain()` | ✗ Returns null |
| Alpha WDT | `PM4Module/AlphaWdtExtractor.cs` | NEW (untested) |
| Unified API | `PM4Module/AdtDataService.cs` | NEW (untested) |

---

## Previous Focus: PM4 → ADT Pipeline (Dec 2025)

## CK24 Structure
```
CK24 = [Type:8bit][ObjectID:16bit]
- 0x00XXXX = Nav mesh (SKIP)
- 0x40XXXX = Has pathfinding data
- 0x42XXXX / 0x43XXXX = WMO-type objects
```

## Do NOT
- Match CK24=0x000000 to WMOs (it's nav mesh)
- Read PM4 tiles independently for cross-tile objects
