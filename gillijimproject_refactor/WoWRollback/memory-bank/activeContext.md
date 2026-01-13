# WoWRollback Active Context

## Current Focus: VLM Export Migration to WoWMapConverter (Jan 2026)

### Session Summary (2026-01-13)

**Decision**: Migrate VLM export from `WoWRollback.MinimapModule` to `WoWMapConverter.Core`.

**Reason**: Custom `AdtParser` in WoWRollback fails for Alpha 0.5.3 MCNK sub-chunks (heights/layers null). `WoWMapConverter.Core` has tested Alpha parsing in `Formats/Alpha/`.

**Status**: PLAN APPROVED - Ready for implementation in next session.

---

## [NEXT SESSION] VLM Migration Plan

### Files to Create in `WoWMapConverter.Core`

1. **`VLM/VlmDataModels.cs`** - Port from `WoWRollback.MinimapModule.Models.VlmTrainingSample`
2. **`VLM/VlmDatasetExporter.cs`** - Main export logic using Core services
3. **`VLM/AlphaMapGenerator.cs`** - Port from WoWRollback

### Files to Modify

1. **`Services/AdtMeshExporter.cs`** - Add `GenerateObjStrings()` for in-memory OBJ/MTL
2. **`WoWMapConverter.Cli/Program.cs`** - Add `vlm-export` command

### Key Integration Points

| WoWRollback                  | → WoWMapConverter.Core           |
|------------------------------|----------------------------------|
| `PrioritizedArchiveSource`   | `AlphaMpqReader`                 |
| `AdtParser.Parse()`          | `Formats/Alpha/AdtAlpha.Parse()` |
| `TerrainMeshExporter`        | `Services/AdtMeshExporter`       |
| `ListfileService`            | `Services/ListfileService`       |

### Existing WoWMapConverter Services (Already Implemented)
- `AdtMeshExporter.cs` - OBJ/MTL terrain export
- `BlpService.cs` - BLP → PNG texture extraction
- `ListfileService.cs` - FileDataID resolution
- `MinimapService.cs` - Minimap tile extraction
- `AlphaMpqReader.cs` - Alpha archive access

### CLI Command
```bash
vlm-export --client <path> --map <name> --out <dir> [--listfile <csv>] [--limit N]
```

---

## What Was Attempted (This Session)

- Added `NameId` fallback field to `ObjectPlacement` ✓
- Fixed MMID/MWID offset-based name resolution ✓
- Fixed Alpha MCNK header size (100 vs 128 bytes) ✗ (still fails)
- Root cause: Need to use WoWMapConverter's proven Alpha parsing
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

## [REFINED PLAN] VLM JSON Dataset for Unsloth (2026-01-13)
**Objective**: Generate a VLM training dataset where each entry contains all necessary data for Unsloth training (image + rich JSON metadata).
**Changes**:
- **Coordinate Fix**: Corrected OBJ export to Y-up mapping (was previously swizzled).
- **Embedded Data**:
    - `obj_content`: Raw OBJ string data.
    - `mtl_content`: Raw MTL string data.
    - `alpha_maps`: Base64 encoded MCAL data.
    - `shadow_map`: Base64 encoded MCSH data.
    - `textures`: List of texture filenames.
    - `layers`: Detailed texture layer info (flags, IDs).
- **Format**: Single JSON file per tile containing `image` path and nested `terrain_data` object with above fields.
