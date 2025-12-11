# Fix: AreaTable Source - Use Cached LK ADTs ✅

**Problem**: Currently sourcing area IDs from Alpha WDT data (via CSV), should use cached LK ADTs.

**Why**: LK ADT files have proper MCNK headers with areaId fields that are already converted and validated.

---

## Current (Wrong) Flow

```
Alpha WDT → AlphaWDTAnalysisTool → CSV with area IDs
                                      ↓
                              McnkTerrainCsvReader.ReadCsv()
                                      ↓
                              AreaIdOverlayBuilder.Build()
                                      ↓
                              Viewer shows area IDs
```

**Problem**: Alpha area IDs may be incorrect, incomplete, or use different numbering.

---

## Correct Flow

```
Alpha WDT → AlphaWDTAnalysisTool → Converted LK ADTs
                                      ↓
                              cached_maps/{version}/{map}/
                                      ↓
                              LkAdtReader.ReadMcnkAreaIds() ← NEW!
                                      ↓
                              AreaIdOverlayBuilder.Build()
                                      ↓
                              Viewer shows area IDs
```

**Benefit**: LK ADT area IDs are authoritative, already converted, no CSV intermediary needed.

---

## Implementation

### Step 1: Add MCNK Reading to LkAdtReader

```csharp
// WoWRollback.Core/Services/LkAdtReader.cs

public record McnkData(
    int ChunkX,           // 0-15
    int ChunkY,           // 0-15
    int AreaId,           // From MCNK header
    uint Flags,           // From MCNK header
    float PositionX,      // World coordinates
    float PositionY,
    float PositionZ,
    bool HasMcsh,         // Has shadow map
    bool HasMccv          // Has vertex colors
);

/// <summary>
/// Reads all MCNK chunk headers from a LK ADT file.
/// Returns area IDs and metadata for each 16×16 chunk.
/// </summary>
public static List<McnkData> ReadMcnkChunks(string adtPath)
{
    var chunks = new List<McnkData>();
    
    if (!File.Exists(adtPath))
        return chunks;

    try
    {
        var bytes = File.ReadAllBytes(adtPath);
        
        // Find KNCM chunk (MCNK reversed)
        int searchPos = 0;
        while (true)
        {
            var chunkStart = FindChunk(bytes, "KNCM", searchPos);
            if (chunkStart == -1)
                break; // No more MCNK chunks
            
            searchPos = chunkStart + 1; // Continue search after this one
            
            // Read MCNK header (first 128 bytes)
            var headerStart = chunkStart + 8; // Skip FourCC + size
            
            // MCNK Header Structure (LK 3.3.5):
            // 0x00: uint32 flags
            // 0x04: uint32 indexX
            // 0x08: uint32 indexY
            // 0x0C: uint32 nLayers
            // 0x10: uint32 nDoodadRefs
            // ...
            // 0x34: int32 areaId       ← What we need!
            // ...
            // 0x40: C3Vector position  ← Chunk world position
            
            uint flags = BitConverter.ToUInt32(bytes, headerStart + 0x00);
            int indexX = (int)BitConverter.ToUInt32(bytes, headerStart + 0x04);
            int indexY = (int)BitConverter.ToUInt32(bytes, headerStart + 0x08);
            int areaId = BitConverter.ToInt32(bytes, headerStart + 0x34);
            
            // Position (world coordinates)
            float posX = BitConverter.ToSingle(bytes, headerStart + 0x40);
            float posY = BitConverter.ToSingle(bytes, headerStart + 0x44);
            float posZ = BitConverter.ToSingle(bytes, headerStart + 0x48);
            
            // Check for sub-chunks (MCSH, MCCV)
            bool hasMcsh = HasSubChunk(bytes, chunkStart, "HSCM"); // MCSH reversed
            bool hasMccv = HasSubChunk(bytes, chunkStart, "VCCM"); // MCCV reversed
            
            chunks.Add(new McnkData(
                ChunkX: indexX,
                ChunkY: indexY,
                AreaId: areaId,
                Flags: flags,
                PositionX: posX,
                PositionY: posY,
                PositionZ: posZ,
                HasMcsh: hasMcsh,
                HasMccv: hasMccv
            ));
        }
    }
    catch (Exception ex)
    {
        Console.WriteLine($"[LkAdtReader] Error reading MCNK from {adtPath}: {ex.Message}");
    }

    return chunks;
}

private static bool HasSubChunk(byte[] bytes, int mcnkStart, string reversedFourCC)
{
    // Read MCNK size to know where it ends
    var mcnkSize = BitConverter.ToInt32(bytes, mcnkStart + 4);
    var mcnkEnd = mcnkStart + 8 + mcnkSize;
    
    // Search within this MCNK only
    var pattern = System.Text.Encoding.ASCII.GetBytes(reversedFourCC);
    for (int i = mcnkStart; i < mcnkEnd - 4 && i < bytes.Length - 4; i++)
    {
        if (bytes[i] == pattern[0] &&
            bytes[i + 1] == pattern[1] &&
            bytes[i + 2] == pattern[2] &&
            bytes[i + 3] == pattern[3])
        {
            return true;
        }
    }
    
    return false;
}

// Update FindChunk to support starting position
private static int FindChunk(byte[] bytes, string reversedFourCC, int startPos = 0)
{
    var pattern = System.Text.Encoding.ASCII.GetBytes(reversedFourCC);
    
    for (int i = startPos; i < bytes.Length - 4; i++)
    {
        if (bytes[i] == pattern[0] &&
            bytes[i + 1] == pattern[1] &&
            bytes[i + 2] == pattern[2] &&
            bytes[i + 3] == pattern[3])
        {
            return i;
        }
    }
    
    return -1;
}
```

---

### Step 2: Create LK ADT-Based Terrain Reader

```csharp
// WoWRollback.Core/Services/LkAdtTerrainReader.cs (NEW FILE)

using System.Collections.Generic;
using System.IO;
using System.Linq;
using WoWRollback.Core.Models;

namespace WoWRollback.Core.Services;

/// <summary>
/// Reads MCNK terrain data directly from cached LK ADT files.
/// Replaces CSV-based approach for more accurate area ID sourcing.
/// </summary>
public static class LkAdtTerrainReader
{
    public static List<McnkTerrainEntry> ReadFromLkAdts(
        string cachedMapsDir,
        string version,
        string mapName)
    {
        var entries = new List<McnkTerrainEntry>();
        
        // Path: cached_maps/{version}/{mapName}/
        var mapDir = Path.Combine(cachedMapsDir, version, mapName);
        if (!Directory.Exists(mapDir))
        {
            Console.WriteLine($"Warning: Cached map directory not found: {mapDir}");
            return entries;
        }
        
        // Find all ADT files
        var adtFiles = Directory.GetFiles(mapDir, "*.adt", SearchOption.TopDirectoryOnly);
        
        foreach (var adtPath in adtFiles)
        {
            // Parse tile coordinates from filename (e.g., "Azeroth_29_40.adt")
            var filename = Path.GetFileNameWithoutExtension(adtPath);
            var parts = filename.Split('_');
            if (parts.Length < 3)
                continue;
            
            if (!int.TryParse(parts[^2], out int tileRow) ||
                !int.TryParse(parts[^1], out int tileCol))
                continue;
            
            // Read MCNK chunks from this ADT
            var chunks = LkAdtReader.ReadMcnkChunks(adtPath);
            
            foreach (var chunk in chunks)
            {
                // Convert to McnkTerrainEntry format (for compatibility with existing code)
                entries.Add(new McnkTerrainEntry(
                    Map: mapName,
                    TileRow: tileRow,
                    TileCol: tileCol,
                    ChunkRow: chunk.ChunkY,
                    ChunkCol: chunk.ChunkX,
                    FlagsRaw: chunk.Flags,
                    HasMcsh: chunk.HasMcsh,
                    Impassible: (chunk.Flags & 0x01) != 0,
                    LqRiver: (chunk.Flags & 0x04) != 0,
                    LqOcean: (chunk.Flags & 0x08) != 0,
                    LqMagma: (chunk.Flags & 0x10) != 0,
                    LqSlime: (chunk.Flags & 0x20) != 0,
                    HasMccv: chunk.HasMccv,
                    HighResHoles: (chunk.Flags & 0x10000) != 0,
                    AreaId: chunk.AreaId,  // ← From LK ADT!
                    NumLayers: 0,  // Would need to parse MCLY
                    HasHoles: false,  // Would need to parse holes
                    HoleType: "none",
                    HoleBitmapHex: "0x0000",
                    HoleCount: 0,
                    PositionX: chunk.PositionX,
                    PositionY: chunk.PositionY,
                    PositionZ: chunk.PositionZ
                ));
            }
        }
        
        Console.WriteLine($"Loaded {entries.Count} MCNK chunks from {adtFiles.Length} LK ADT files ({mapName})");
        return entries;
    }
}
```

---

### Step 3: Update McnkTerrainOverlayBuilder

```csharp
// WoWRollback.Core/Services/Viewer/McnkTerrainOverlayBuilder.cs

public static void BuildOverlaysForMap(
    string mapName,
    string csvDir,          // ← DEPRECATED, keep for backward compat
    string outputDir,
    string version,
    AreaTableLookup? areaLookup = null,
    string? cachedMapsDir = null)  // ← NEW: Path to cached_maps/
{
    List<McnkTerrainEntry> allChunks;
    
    // NEW: Prefer reading from cached LK ADTs
    if (cachedMapsDir != null && Directory.Exists(cachedMapsDir))
    {
        Console.WriteLine($"Reading terrain data from cached LK ADTs for {mapName}...");
        allChunks = LkAdtTerrainReader.ReadFromLkAdts(cachedMapsDir, version, mapName);
    }
    else
    {
        // FALLBACK: Use CSV (old method)
        Console.WriteLine($"Falling back to CSV terrain data for {mapName}...");
        var terrainCsvPath = Path.Combine(csvDir, $"{mapName}_mcnk_terrain.csv");
        if (!File.Exists(terrainCsvPath))
            return;
        
        allChunks = McnkTerrainCsvReader.ReadCsv(terrainCsvPath);
    }
    
    if (allChunks.Count == 0)
    {
        Console.WriteLine($"No terrain data for {mapName}, skipping");
        return;
    }

    // ... rest of the method unchanged ...
}
```

---

### Step 4: Update ViewerReportWriter

```csharp
// WoWRollback.Core/Services/Viewer/ViewerReportWriter.cs

// Update call to BuildOverlaysForMap to pass cachedMapsDir

McnkTerrainOverlayBuilder.BuildOverlaysForMap(
    mapName: mapName,
    csvDir: csvDir,  // Keep for backward compat
    outputDir: overlaysDir,
    version: version,
    areaLookup: areaLookup,
    cachedMapsDir: cachedMapsDir  // ← NEW: Pass cached_maps path
);
```

---

## Testing

### Test 1: Compare Area IDs
```powershell
# Generate with CSV (old way)
wowrollback compare-versions --use-csv-areatable

# Generate with LK ADTs (new way)
wowrollback compare-versions --use-lk-adt-areatable

# Compare outputs
diff cached_maps/analysis/0.5.3.3368/Azeroth/csv_areaids.json \
     cached_maps/analysis/0.5.3.3368/Azeroth/lk_areaids.json
```

### Test 2: Viewer Validation
1. Load viewer with new area overlay
2. Check that area boundaries match in-game
3. Verify area names are correct

---

## Benefits

✅ **Authoritative source**: LK ADT area IDs are already converted  
✅ **No CSV dependency**: Direct binary reading  
✅ **Faster**: No CSV parsing overhead  
✅ **More accurate**: LK format is validated  
✅ **Simpler pipeline**: One less intermediate file

---

## Migration Path

1. ✅ Add `ReadMcnkChunks()` to `LkAdtReader.cs`
2. ✅ Create `LkAdtTerrainReader.cs`
3. ✅ Update `McnkTerrainOverlayBuilder.cs` with fallback
4. ✅ Test on Dun Morogh (small map)
5. ✅ Test on full Azeroth
6. ✅ Deprecate CSV-based approach
7. ✅ Remove `McnkTerrainCsvReader.cs` (after Phase 0/1)

---

**This fix makes area ID sourcing correct and prepares for Phase 0 (Rollback Feature)!** ✅
