# WoWRollback Active Context

## Current Focus: VLM Terrain Data Export (Dec 2025)

### Session Summary (2025-12-25)

**Goal**: Export VLM dataset with terrain ground truth (heightmaps, textures, alpha maps) for AI training.

**Status**: BLOCKED - GUI changes not taking effect

**What Works**:
- VLM Export button exports minimap PNGs
- Placements from CSV cache files work
- JSON metadata files created

**What's Broken**:
- Terrain data is always `null` in JSON
- Debug logging added to `MainWindow.cs` never appears
- Builds succeed but changes don't seem to take effect

**Code Locations**:
- `WoWRollback.Gui/MainWindow.cs` - `ExportVlmBtn_Click` (~line 776) - inline terrain extraction NOT WORKING
- `WoWRollback.AnalysisModule/AdtMpqTerrainExtractor.cs` - Full implementation (works in CLI)
- CLI "Prepare Layers" uses `--placements-only`, skips terrain extraction

**Root Cause Theories**:
1. GUI not actually rebuilding (DLL caching issue)
2. Need to use CLI's AdtMpqTerrainExtractor instead of inline GUI code
3. "Prepare Layers" should run terrain extraction first, GUI reads from cache

**Next Steps**:
1. Verify MainWindow.cs changes are in compiled DLL
2. Option A: Remove `--placements-only` from GUI's CLI call, let cache populate terrain JSON
3. Option B: GUI VLM export reads cached terrain JSON (already implemented, but cache empty)

---

## Previous Focus: PM4 → ADT Pipeline (Dec 2025)

## Key Files

| File | Purpose |
|------|---------|
| `Pm4Decoder.cs` | Decodes single PM4 tile chunks |
| `Pm4ObjectBuilder.cs` | Groups surfaces by CK24, splits by MSVI gaps |
| `Pm4ModfReconstructor.cs` | Matches PM4 objects to WMOs |
| `MuseumAdtPatcher.cs` | Injects MODF/MWMO into ADTs |
| `Pm4Reader/Program.cs` | Original standalone reader (reference) |

## CK24 Structure

```
CK24 = [Type:8bit][ObjectID:16bit]
- 0x00XXXX = Nav mesh (SKIP)
- 0x40XXXX = Has pathfinding data
- 0x42XXXX / 0x43XXXX = WMO-type objects
- 0xC0XXXX+ = Various object types
```

## Do NOT
- Match CK24=0x000000 to WMOs (it's nav mesh)
- Read PM4 tiles independently for cross-tile objects
- Use coordinate swaps in `Pm4Decoder` (original format is X,Y,Z)
