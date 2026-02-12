# Rollback Feature - Comprehensive Implementation Plan

**Goal**: Allow users to selectively preserve/remove objects from Alpha maps by UniqueID range, with per-tile granularity, and patch the actual ADT/WDT files to produce a "rolled back" version.

---

## 📋 Overview

The Rollback feature is the core vision of WoWRollback: enabling surgical removal of objects from Alpha maps to create custom "rollback" versions that preserve only selected content. This involves:

1. **Per-tile CSV generation** - Generate UniqueID ranges for each tile individually
2. **Tile selection UI** - Reactivate and enhance `tile.html` for per-tile range selection
3. **Selection persistence** - Save user selections to a rollback configuration
4. **ADT patching** - Replace unwanted object model paths with invisible model
5. **WDT patching** - Null out object entries in MMDX/MMID/MODF/MWID/MWMO chunks

---

## 🎯 Phase 1: Per-Tile CSV Generation ⭐ **IN PROGRESS**

### Current State
- ✅ Map-wide CSV generation exists (`id_ranges_by_map.csv`)
- ✅ 10K clustering algorithm works well
- ❌ No per-tile CSV generation

### Requirements

**Two modes of operation:**

1. **Simple Mode (Global)** - ✅ **ALREADY IMPLEMENTED**
   - Uses map-wide ranges: `id_ranges_by_map.csv`
   - Single selection applies to entire map
   - Quick and easy for users

2. **Advanced Mode (Per-Tile)** - 🔨 **NEED TO IMPLEMENT**
   - Uses per-tile ranges: `id_ranges_by_tile.csv`
   - Granular control per 64x64 tile grid
   - For power users who want surgical precision

### Per-Tile CSV Format
Generate `id_ranges_by_tile.csv` for each tile:
```csv
TileRow,TileCol,MinUniqueID,MaxUniqueID,Count,ModelType
39,37,4531,5788,371,M2
39,37,7694,14693,1245,M2
39,38,14694,24693,892,WMO
...
```

**Additional column:** `ModelType` (M2 or WMO) to determine which invisible model to use

### Implementation Steps

1. **Update `WoWRollback.Core/Services/Analysis/UniqueIdRangeCsvWriter.cs`**
   ```csharp
   public void WritePerTileRangesCsv(
       List<Placement> placements, 
       string outputPath)
   {
       // Group by tile (row, col)
       var byTile = placements
           .GroupBy(p => (p.TileRow, p.TileCol))
           .OrderBy(g => g.Key.TileRow)
           .ThenBy(g => g.Key.TileCol);
       
       foreach (var tile in byTile)
       {
           // Cluster within each tile (10K ranges)
           var ranges = ClusterIntoRanges(tile.ToList(), 10000);
           
           // Write to CSV with tile coordinates
           foreach (var range in ranges)
           {
               csv.WriteLine($"{tile.Key.TileRow},{tile.Key.TileCol}," +
                           $"{range.Min},{range.Max},{range.Count}," +
                           $"{range.ModelType}");
           }
       }
   }
   ```

2. **Integrate into Pipeline**
   - Call from `rebuild-and-regenerate.ps1` during cache generation
   - Output to `cached_maps/analysis/{version}/{map}/csv/id_ranges_by_tile.csv`
   - Log generation with checkmarks

3. **Copy to Viewer**
   - Copy per-tile CSVs to viewer output directory
   - Enable tile.html to load them

### Success Criteria
- [ ] Per-tile CSVs generated for all maps
- [ ] CSV format validated and parseable
- [ ] Total object count matches map-wide CSV
- [ ] Tile coordinates (0-63) valid range
- [ ] ModelType (M2/WMO) correctly identified

---

## 🎯 Phase 2: Tile Selection UI (tile.html) ⭐ **NEXT PRIORITY**

### Current State
- ✅ `tile.html` exists but is disabled
- ✅ Basic tile grid visualization infrastructure
- ❌ No per-tile range selection UI
- ❌ No integration with map viewer
- ❌ No selection persistence

### Requirements

**Two-Mode System:**

1. **Simple Mode** (index.html)
   - ✅ Already implemented
   - Uses map-wide ranges from `id_ranges_by_map.csv`
   - Quick checkbox selection
   - One config for entire map

2. **Advanced Mode** (tile.html) - 🔨 **NEED TO IMPLEMENT**
   - Per-tile granular control
   - 64x64 tile grid interface
   - Click tile → load its specific ranges
   - Save selections per-tile, per-map, per-version

### UI Components

#### 1. Tile Grid View
- **64x64 grid** representing ADT tiles (0-63 each axis)
- **Color coding**:
  - 🟢 Green: All ranges selected (keep all objects)
  - 🟡 Yellow: Partial selection (some ranges unchecked)
  - 🔴 Red: No ranges selected (remove all objects)
  - ⚫ Gray: Empty tile (no objects)
- **Hover tooltip**: Show tile coordinates + object count
- **Click tile**: Open range selection modal

#### 2. Per-Tile Range Selection Modal
- Load `id_ranges_by_tile.csv` filtered by tile coordinates
- Checkbox list (like current Sedimentary Layers)
- Show: UniqueID range, object count, model type (M2/WMO)
- **Buttons**:
  - Select All / Deselect All
  - Apply (save and update grid colors)
  - Cancel (discard changes)

#### 3. Selection Persistence
Store selections in browser localStorage:
```json
{
  "map": "Azeroth",
  "version": "0.5.3.3368",
  "mode": "advanced",  // or "simple"
  "tiles": {
    "39_37": {
      "selectedRanges": [
        {"min": 4531, "max": 5788, "type": "M2"},
        {"min": 7694, "max": 14693, "type": "M2"}
      ]
    },
    "39_38": { ... }
  }
}
```

#### 4. Export Configuration Page
- **New page**: `export.html`
- Display current selections (simple or advanced mode)
- Show summary statistics:
  - Total objects to keep
  - Total objects to remove
  - Affected tiles count
- **Export buttons**:
  - Download JSON config
  - Generate patched WDT (future)
  - Generate patched ADTs (future)

### Implementation Steps
1. **Reactivate tile.html**
   - Enable in viewer navigation
   - Fix any broken links/scripts

2. **Build Tile Grid Component**
   - Canvas or SVG-based 64x64 grid
   - Fetch per-tile CSV on load
   - Calculate density heatmap

3. **Build Range Selection Modal**
   - Reuse checkbox code from sedimentary-layers-csv.js
   - Add per-tile context (show tile coordinates)
   - Save selections to localStorage or session state

4. **Integrate with Map Viewer**
   - Add "Open Tile Selector" button to index.html
   - Pass map/version state to tile.html
   - Apply tile selections to marker visibility

### Success Criteria
- [ ] Tile grid displays all 64x64 tiles
- [ ] Click tile opens range selection
- [ ] Selections persist during session
- [ ] Map updates in real-time when selections change

---

## 🎯 Phase 3: Selection Persistence & Export

### Requirements
Save user selections to a rollback configuration file for later patching:

**Format: `rollback_config.json`**
```json
{
  "map": "Azeroth",
  "version": "0.5.3.3368",
  "tiles": [
    {
      "row": 39,
      "col": 37,
      "selectedRanges": [
        {"min": 4531, "max": 5788},
        {"min": 7694, "max": 14693}
      ],
      "mode": "keep"
    }
  ],
  "globalMode": "keep",
  "timestamp": "2025-10-05T12:00:00Z"
}
```

### Implementation Steps
1. **Export Configuration**
   - Add "Export Rollback Config" button
   - Generate JSON from current selections
   - Download as file

2. **Import Configuration**
   - Add "Import Rollback Config" button
   - Parse JSON and restore selections
   - Validate format and map/version compatibility

3. **Configuration Validation**
   - Check map name matches
   - Verify tile coordinates in valid range (0-63)
   - Confirm UniqueID ranges exist in CSV

### Success Criteria
- [ ] Export produces valid JSON
- [ ] Import restores selections correctly
- [ ] Validation catches invalid configs

---

## 🎯 Phase 4: Model Path Replacement Strategy

### The Challenge
Alpha ADTs store model paths in MDDF/MODF chunks. We can't simply delete entries because:
- It would change chunk offsets
- Other data structures reference by index
- File integrity would break

### Solution: Use Existing Debug Models
Replace unwanted model paths with existing invisible debug models from WoW data.

### Available Debug Models
**These exist across all WoW versions (Alpha 0.5.3 → 3.3.5 → Current 12.0 beta):**

1. **M2 Models (Doodads):**
   - **`SPELLS\Invisible.m2`** (19 chars) - ✅ **PREFERRED**
     - Used as invisible server-side script anchor
     - Confirmed invisible in-game
     - Shortest path option

2. **WMO Models (World Objects):**
   - **`world\wmo\dungeon\test\test.wmo`** (30 chars) - ✅ **PREFERRED**
     - Three nested square boxes for coordinate system testing
     - Small footprint, minimal geometry
     - Perfect for invisible/testing purposes
     - **User confirmed as best option**

### Path Length Strategy
**Key Decision**: `SPELLS\Invisible.m2` = 19 characters

**Challenge**: Alpha model paths vary in length (e.g., 25-50+ chars)

**Solutions to Investigate:**

#### Option A: Pad to Original Length ✅ **RECOMMENDED**
```
Original: "World\Azeroth\Elwynn\Building.m2" (33 chars)
Replace:  "SPELLS\Invisible.m2\0\0\0\0\0\0\0\0\0\0\0\0\0\0" (33 chars, null-padded)
```
- Preserves exact file size
- No offset changes
- **Risk**: WoW client might reject padded paths

#### Option B: Path-in-Path Embedding
```
Original: "World\Azeroth\Elwynn\Building.m2" (33 chars)
Replace:  "SPELLS\Invisible.m2__padding__" (33 chars, underscore padding)
```
- Maintains file size
- More "natural" looking
- **Risk**: Client might parse full string including padding

#### Option C: Truncate Longer Paths (RISKY)
```
Original: "World\Azeroth\Elwynn\Building.m2" (33 chars)
Replace:  "SPELLS\Invisible.m2" (19 chars + null terminator)
         Shift remaining data left, update all offsets
```
- ❌ **NOT RECOMMENDED** - breaks offsets, extremely complex

### Model Path Length Analysis (Still Needed)
1. Scan all Alpha model paths
2. Generate distribution histogram (10-20, 20-30, 30-40, etc.)
3. Identify edge cases (very long paths > 100 chars)
4. Test padding strategies on sample ADTs

### Path Replacement Logic
For each object to remove:
1. Get original model path and length (N chars)
2. Build replacement: `SPELLS\Invisible.m2` + null padding to N chars
3. Replace path in-place in ADT MMDX/MWMO chunk
4. Preserve null terminator at end

### Investigation Steps
1. **Verify Debug Models Exist**
   - Confirm `SPELLS\Invisible.m2` exists in Alpha 0.5.3 data
   - Confirm `SPELLS\ErrorCube.m2` exists as fallback
   - Extract and examine M2 structure (optional)

2. **Analyze Model Path Lengths**
   ```csharp
   // WoWRollback.Core/Services/Analysis/ModelPathAnalyzer.cs
   Dictionary<int, List<string>> GroupPathsByLength(List<Placement> placements)
   ```
   - Generate histogram of path lengths
   - Identify shortest path (min padding needed)
   - Identify longest path (max padding needed)

3. **Test Null-Padding Strategy**
   - Manually hex-edit one ADT
   - Replace a model path with `SPELLS\Invisible.m2` + null bytes
   - Test in WoW Alpha client:
     - Does ADT load?
     - Is object invisible?
     - Any crashes or errors?

4. **Test Alternative Padding Strategies** (if null-padding fails)
   - Try space-padding: `SPELLS\Invisible.m2     ` (spaces to length)
   - Try underscore-padding: `SPELLS\Invisible.m2_______`
   - Document which strategy client accepts

### Success Criteria
- [ ] Debug models confirmed to exist in Alpha data
- [ ] Model path length distribution mapped (min, max, avg)
- [ ] Padding strategy tested and validated in-game
- [ ] Manual hex-edit replacement verified (ADT loads, object invisible)
- [ ] No file corruption or offset issues

---

## 🎯 Phase 5A: AlphaWDT Patching Implementation 🦀 **← PRIORITY**

### Why AlphaWDT First?
1. **More Authentic** - Uses original Alpha client (0.5.3/0.5.5) for testing
2. **Single File** - WDT is one file vs 4096 ADT tiles per map
3. **User Requested** - Community already asking for this feature
4. **Faster Iteration** - Easier to test and validate
5. **Foundation for ADT** - Same techniques apply to ADT patching later

### AlphaWDT Structure Overview
Alpha WDT files contain:
- **MVER** - Version (4 bytes)
- **MPHD** - Header/flags
- **MAIN** - 64x64 tile existence flags
- **MWMO** - WMO model name strings (null-separated)
- **MWID** - WMO name offsets (index into MWMO)
- **MODF** - WMO placement data (position, rotation, scale, bounding box)
- **MMDX** - M2 model name strings (null-separated)
- **MMID** - M2 name offsets (index into MMDX)
- **MDDF** - M2 placement data (position, rotation, scale, UniqueID)

### Requirements
Patch AlphaWDT files to replace unwanted object model paths with invisible models.

**Target**: MMDX (M2 names) and MWMO (WMO names) chunks

### Implementation Steps

1. **Build AlphaWDT Parser**
   ```csharp
   // WoWRollback.Core/Services/Parsing/AlphaWdtParser.cs
   public class AlphaWdtParser
   {
       public AlphaWdtData Parse(string wdtPath)
       {
           // Read WDT file
           // Parse MVER, MPHD, MAIN chunks
           // Parse MMDX/MMID (M2 data)
           // Parse MWMO/MWID (WMO data)
           // Parse MDDF/MODF (placement data)
           // Return structured data
       }
       
       public Dictionary<string, List<int>> GetModelNameOffsets()
       {
           // Map model names to their byte offsets in MMDX/MWMO chunks
           // Return: { "World\\Azeroth\\Building.m2": [offset1, offset2], ... }
       }
   }
   
   public class AlphaWdtData
   {
       public byte[] MverChunk { get; set; }
       public byte[] MphdChunk { get; set; }
       public byte[] MainChunk { get; set; }
       
       // M2 data
       public byte[] MmdxChunk { get; set; }  // Model name strings
       public byte[] MmidChunk { get; set; }  // Name offsets
       public byte[] MddfChunk { get; set; }  // Placement data
       
       // WMO data
       public byte[] MwmoChunk { get; set; }  // Model name strings
       public byte[] MwidChunk { get; set; }  // Name offsets
       public byte[] ModfChunk { get; set; }  // Placement data
       
       public List<M2Placement> M2Placements { get; set; }
       public List<WmoPlacement> WmoPlacements { get; set; }
   }
   ```

2. **Build AlphaWDT Patcher**
   ```csharp
   // WoWRollback.Core/Services/Patching/AlphaWdtPatcher.cs
   public class AlphaWdtPatcher
   {
       private const string INVISIBLE_MODEL = "SPELLS\\Invisible.m2";
       
       public void PatchWdt(string wdtPath, RollbackConfig config, string outputPath)
       {
           // 1. Parse WDT
           var wdtData = parser.Parse(wdtPath);
           
           // 2. Identify objects to remove (not in config.selectedRanges)
           var objectsToRemove = GetObjectsToRemove(wdtData, config);
           
           // 3. For each object to remove:
           //    - Get model path from MMDX/MWMO
           //    - Calculate original path length
           //    - Replace with SPELLS\Invisible.m2 + null padding
           
           // 4. Write patched WDT
           WriteWdt(outputPath, wdtData);
       }
       
       private void ReplaceModelPath(byte[] chunkData, int offset, string originalPath)
       {
           int originalLength = originalPath.Length;
           
           // Build replacement: "SPELLS\Invisible.m2" + null padding
           string replacement = INVISIBLE_MODEL.PadRight(originalLength, '\0');
           byte[] replacementBytes = Encoding.ASCII.GetBytes(replacement);
           
           // Replace in-place
           Array.Copy(replacementBytes, 0, chunkData, offset, originalLength);
       }
   }
   ```

3. **Add UniqueID Filtering Logic**
   - Parse MDDF chunk to get UniqueIDs for each M2 placement
   - Match UniqueIDs against config.selectedRanges
   - If UniqueID NOT in selected ranges → replace model path

4. **Validation**
   - Verify chunk sizes unchanged
   - Confirm file size identical (critical!)
   - Parse patched WDT to verify structure
   - Check no corruption in MAIN chunk (tile flags)

5. **Batch Processing**
   - Patch entire map (single WDT file)
   - Progress reporting
   - Backup original before patching

### Success Criteria
- [ ] Single WDT parses successfully
- [ ] Model paths identified correctly
- [ ] Patching preserves file size
- [ ] Patched WDT loads in Alpha 0.5.3 client
- [ ] Removed objects are invisible in-game
- [ ] No crashes or corruption

---

## 🎯 Phase 5B: LK ADT Patching Implementation (LOWER PRIORITY)

### Requirements
Patch converted LK ADT files to replace unwanted object model paths with invisible models.

**Note**: This is lower priority since AlphaWDT patching is more authentic and addresses immediate user needs.

### Implementation Steps
1. **Build ADT Patcher Service**
   ```csharp
   // WoWRollback.Core/Services/Patching/AdtPatcher.cs
   public class AdtPatcher
   {
       public void PatchAdt(string adtPath, RollbackConfig config, InvisibleModelLibrary library)
       {
           // 1. Parse ADT
           // 2. For each MDDF/MODF entry:
           //    - Check if UniqueID in config
           //    - If NOT selected, replace model path
           // 3. Write patched ADT
       }
   }
   ```

2. **Model Path Replacement Logic**
   - Locate MMDX (M2 names) or MWMO (WMO names) chunk
   - Find offset of model name string
   - Replace with same-length invisible model name
   - Preserve null terminators and padding

3. **Validation**
   - Verify chunk sizes unchanged
   - Confirm file size identical
   - Check CRC/checksums if applicable

4. **Batch Processing**
   - Patch all tiles in map
   - Progress reporting
   - Rollback on error (keep backups)

### Success Criteria
- [ ] Single ADT patches successfully
- [ ] Patched ADT loads in client
- [ ] Removed objects are invisible
- [ ] No crashes or corruption

---

## 🎯 Phase 6: WDT Patching (Optional)

### Requirements
Optionally patch the WDT root file to null out object references globally.

### Implementation Steps
1. **WDT Object Chunk Analysis**
   - MMDX/MMID - M2 model names and IDs
   - MWMO/MWID - WMO model names and IDs
   - MODF - WMO placement data

2. **Null Out Strategy**
   - Option A: Replace model names with invisible models (same as ADT)
   - Option B: Zero out placement entries (risky, may break offsets)
   - **Recommend Option A** for consistency

3. **Test Implications**
   - Does WDT patching override ADT data?
   - Is WDT patching necessary if ADTs are patched?
   - **Decision**: Start with ADT-only patching, WDT is optional

### Success Criteria
- [ ] WDT structure documented
- [ ] Null-out strategy tested
- [ ] Optional WDT patching implemented

---

## 🎯 Phase 7: CLI Integration

### Requirements
Add CLI commands for rollback patching:

```powershell
# Apply rollback configuration to ADTs
dotnet run --project WoWRollback.Cli -- rollback apply \
  --config rollback_config.json \
  --input-dir test_data/0.5.3.3368/tree/World/Maps/Azeroth \
  --output-dir rollback_outputs/patched/Azeroth \
  --invisible-models invisible_models/

# Validate rollback configuration
dotnet run --project WoWRollback.Cli -- rollback validate \
  --config rollback_config.json
```

### Implementation Steps
1. **Add Rollback Commands**
   - `rollback apply` - Apply config and patch ADTs
   - `rollback validate` - Validate config format
   - `rollback preview` - Dry-run showing what would be removed

2. **Progress Reporting**
   - Show tiles processed
   - Report objects removed per tile
   - Display total statistics

3. **Error Handling**
   - Backup original files before patching
   - Rollback on error
   - Log all operations

### Success Criteria
- [ ] Commands implemented and documented
- [ ] Validation catches all errors
- [ ] Progress reporting clear and accurate

---

## 📊 Implementation Order (UPDATED PRIORITY)

### **🎯 Next "Crabbing" Feature: AlphaWDT Patching**

Per user request, we're prioritizing **AlphaWDT patching** first (most authentic, client uses original Alpha data), then LK ADT conversion later.

### Sprint 1: Foundation (Week 1)
1. ✅ Per-tile CSV generation
2. ✅ Model path length analysis
3. ✅ Verify `SPELLS\Invisible.m2` exists in Alpha data
4. ✅ Manual hex-edit test (single WDT entry)

### Sprint 2: AlphaWDT Patching 🦀 (Week 2) **← PRIORITY**
5. Build AlphaWDT parser (read MMDX/MMID/MODF chunks)
6. Build AlphaWDT patcher (replace model paths in-place)
7. Test patched WDT in Alpha 0.5.3 client
8. CLI command: `rollback patch-alpha-wdt`

### Sprint 3: UI for Selection (Week 3)
9. Reactivate tile.html
10. Build tile grid component
11. Build range selection modal
12. Configuration export/import

### Sprint 4: LK ADT Patching (Week 4) **← LATER**
13. Build LK ADT patcher (optional, lower priority)
14. Test in 3.3.5 client
15. CLI command: `rollback patch-lk-adt`

### Sprint 5: Polish (Week 5)
16. Batch processing for full maps
17. Documentation
18. End-to-end testing

---

## 🚧 Risks & Mitigations

### Risk 1: Model Path Length Mismatch
**Problem**: Can't find invisible model with exact length  
**Mitigation**: Create invisible models for all common lengths (10, 15, 20, 25, 30, 35, 40, 50)

### Risk 2: ADT Corruption
**Problem**: Patching breaks ADT structure  
**Mitigation**: Always backup originals, validate before/after, extensive testing

### Risk 3: Client Doesn't Load Patched Files
**Problem**: WoW client rejects modified ADTs  
**Mitigation**: Research client validation, use same file size, preserve checksums

### Risk 4: UI Complexity
**Problem**: Per-tile selection is overwhelming for users  
**Mitigation**: Add "Apply to All Tiles" option, smart defaults, presets

---

## 📝 Open Questions

1. **Q**: Can we use `SPELLS\Invisible.m2` for all model types (M2 and WMO)?  
   **A**: Need to verify - might need `SPELLS\Invisible.wmo` equivalent for WMO objects

2. **Q**: Will null-padding work, or do we need space/underscore padding?  
   **A**: **Must test in hex editor** - client string parsing is unknown

3. **Q**: Should we patch WDT in addition to ADTs?  
   **A**: Start ADT-only, measure impact, add WDT if needed

4. **Q**: How to handle objects spanning multiple tiles?  
   **A**: Use tile where object anchor point is located

5. **Q**: Should "keep" or "remove" be the default mode?  
   **A**: "Keep" is safer - explicitly mark what to preserve

6. **Q**: Do we need to handle MODF (WMO placement) differently than MDDF (M2)?  
   **A**: Both use model paths, same replacement strategy applies

7. **Q**: What if `SPELLS\Invisible.m2` doesn't exist in very early Alpha builds?  
   **A**: Fall back to `SPELLS\ErrorCube.m2` or research alternative debug models

---

## 🎉 Success Metrics

### MVP (Minimum Viable Product)
- [ ] Per-tile CSV generation working
- [ ] Tile.html displays grid and allows range selection
- [ ] Configuration export/import functional
- [ ] Single ADT patching works and loads in-game
- [ ] Removed objects are invisible

### Full Feature
- [ ] All 64x64 tiles patchable
- [ ] Batch processing for entire map
- [ ] CLI commands documented
- [ ] End-to-end workflow tested
- [ ] README updated with rollback instructions

---

## 📚 References

- **Alpha ADT Format**: `docs/architecture/adt-format.md` (TODO: create if not exists)
- **MDDF/MODF Chunks**: Warcraft.NET documentation
- **Invisible Model Creation**: Research M2/WMO minimal structure

---

**Next Step**: Start with Phase 1 (Per-Tile CSV Generation) to establish the data foundation.
