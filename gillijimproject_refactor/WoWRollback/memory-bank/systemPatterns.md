# System Patterns - WoWRollback.RollbackTool Architecture

## Three-Tool Separation of Concerns

```
┌─────────────────────────────┐
│ AlphaWDTAnalysisTool        │  Analysis Phase
│ (EXISTS - Reuse!)           │
├─────────────────────────────┤
│ • Scan WDT/ADT files        │
│ • Extract UniqueID ranges   │
│ • Output CSVs/JSON          │
│ • Already has clustering    │
└─────────────────────────────┘
           ↓ CSVs/JSON
┌─────────────────────────────┐
│ WoWRollback.RollbackTool    │  Modification Phase
│ (NEW - Build This!)         │
├─────────────────────────────┤
│ • Read analysis data        │
│ • Modify WDT in-place       │
│ • Bury objects by Z coord   │
│ • Fix terrain holes (MCNK)  │
│ • Disable shadows (MCSH)    │
│ • Generate MD5 checksums    │
└─────────────────────────────┘
           ↓ Modified WDT
┌─────────────────────────────┐
│ WoWDataPlot                 │  Visualization Phase
│ (REFOCUS - Viz Only!)       │
├─────────────────────────────┤
│ • Pre-generate overlays     │
│ • Lightweight HTML viewer   │
│ • No modification logic     │
│ • Pure presentation layer   │
└─────────────────────────────┘
```

## Core Rollback Architecture

### In-Memory Modification Pattern
```
1. Load entire WDT into byte array (wdtBytes)
2. Parse ADT offsets from MAIN chunk
3. For each ADT:
   a. Create AdtAlpha instance (offset into wdtBytes)
   b. Get Mddf/Modf chunk references
   c. Modify Z coordinate in chunk.Data
   d. Copy modified data back to wdtBytes
4. Write modified wdtBytes to output file
5. Generate MD5 checksum
```

### Chunk Access Pattern
```csharp
// AdtAlpha provides parsed chunk access
var adt = new AdtAlpha(wdtPath, offsetInFile, adtNum);
var mddf = adt.GetMddf();           // Returns Mddf chunk
var modf = adt.GetModf();           // Returns Modf chunk

// Modify in-place
for (int i = 0; i < mddf.Data.Length; i += 36) {
    uint uid = BitConverter.ToUInt32(mddf.Data, i + 4);
    if (uid > threshold) {
        // Bury it: modify Z at offset +12
        byte[] newZ = BitConverter.GetBytes(-5000.0f);
        Array.Copy(newZ, 0, mddf.Data, i + 12, 4);
    }
}

// Write back to file
int fileOffset = adt.GetMddfDataOffset();
Array.Copy(mddf.Data, 0, wdtBytes, fileOffset, mddf.Data.Length);
```

## Spatial MCNK Mapping Pattern

### Coordinate Space Transformations
```
World Coords → ADT Tile → MCNK Chunk

Given placement at (worldX, worldY, worldZ):
1. tileX = floor(worldX / 533.33)
2. tileY = floor(worldY / 533.33)
3. localX = worldX - (tileX * 533.33)
4. localY = worldY - (tileY * 533.33)
5. mcnkX = floor(localX / 33.33)
6. mcnkY = floor(localY / 33.33)
7. mcnkIndex = (mcnkY * 16) + mcnkX
```

### MCNK Hole Flag Management
```
For each buried WMO:
1. Calculate which MCNK(s) it overlaps
2. Locate MCNK header via MHDR offsets
3. Clear Holes field at offset +0x40
4. Write modified header back to wdtBytes
```

## Pre-Generation Pattern (Not On-the-Fly)

### Why Pre-Generate?
- **Performance**: No computation during viewing
- **Reliability**: Generate once, view forever
- **Simplicity**: Pure HTML+JS, no server needed
- **Portability**: Works on any platform with a browser

### Overlay Generation Strategy
```
For each percentile threshold (10%, 25%, 50%, 75%, 90%):
1. Load minimap BLPs
2. Plot placements (green=kept, red=buried)
3. Save as PNG: overlays/{map}_uid_{min}-{max}.png
4. Add entry to overlay-index.json
```

## Command Structure

### Analyze Command
```bash
WoWRollback analyze --input Azeroth.wdt --output analysis/azeroth.json
```
Output: UniqueID statistics, suggested thresholds

### Generate Overlays Command
```bash
WoWRollback generate-overlays --analysis azeroth.json --output overlays/azeroth/
```
Output: PNG overlays + manifest JSON

### Rollback Command
```bash
WoWRollback rollback --input Azeroth.wdt --output rollback/Azeroth.wdt \
  --max-uniqueid 10000 --clear-holes --disable-shadows
```
Output: Modified WDT + MD5 checksum

## Error Handling Philosophy
- **Fail Early on Invalid Input**: Bad WDT path = immediate error
- **Continue on Chunk Errors**: Skip malformed chunks, process rest
- **Validate Before Writing**: Sanity check modifications before output
- **Preserve Originals**: Never modify input files in-place

## Data Integrity Patterns

### MD5 Checksum Generation
```csharp
using var md5 = System.Security.Cryptography.MD5.Create();
var hash = md5.ComputeHash(wdtBytes);
var hashString = BitConverter.ToString(hash).Replace("-", "").ToLowerInvariant();
File.WriteAllText($"{mapName}.md5", hashString);
```
**Purpose**: WoW client validates minimap files via .md5 files

### File Offset Tracking
```csharp
// AdtAlpha stores its offset for writeback calculations
private readonly int _adtFileOffset;

public int GetMddfDataOffset() {
    int mhdrStartOffset = GetAdtFileOffset() + ChunkLettersAndSize;
    int mddfChunkOffset = mhdrStartOffset + _mhdr.GetOffset(0x0C);
    return mddfChunkOffset + ChunkLettersAndSize; // Skip header
}
```

### Byte Array Modification Pattern
```csharp
// Modify chunk data in-memory
byte[] newValue = BitConverter.GetBytes(valueToWrite);
Array.Copy(newValue, 0, chunk.Data, offsetInChunk, sizeof(float));

// Copy back to file bytes
Array.Copy(chunk.Data, 0, wdtBytes, fileOffset, chunk.Data.Length);
```

## Testing Strategy

### Validation Checklist
- [x] Small maps (< 100 tiles) - Tested on development maps
- [x] Large maps (900+ tiles) - **TESTED: Kalimdor 951 tiles!**
- [x] Alpha 0.5.3 compatibility - **PROVEN WORKING**
- [ ] Later Alpha versions (0.5.5, 0.6.0)
- [ ] Beta versions
- [ ] 1.x retail versions
- [ ] WotLK 3.x versions

### Success Criteria Per Test
1. Tool completes without crashing
2. Output WDT is valid (correct file size)
3. MD5 checksum generated
4. Modified placements have Z = -5000.0
5. Unmodified placements unchanged
6. File loadable in WoW client (manual test)
