using System;
using System.Collections.Generic;
using System.IO;

namespace AlphaWdtAnalyzer.Core.Terrain;

/// <summary>
/// Extracts terrain data directly from LK (3.3.5) ADT files.
/// This bypasses Alpha WDT parsing entirely, enabling CSV extraction from cached LK ADTs.
/// </summary>
public static class LkAdtTerrainExtractor
{
    /// <summary>
    /// Extract terrain data directly from LK ADT directory (no Alpha WDT needed).
    /// </summary>
    /// <param name="lkAdtDirectory">Directory containing LK ADT files</param>
    /// <param name="mapName">Map name for output</param>
    /// <returns>Terrain entries with LK AreaIDs</returns>
    public static List<McnkTerrainEntry> ExtractFromLkAdts(string lkAdtDirectory, string mapName)
    {
        var results = new List<McnkTerrainEntry>();
        
        if (!Directory.Exists(lkAdtDirectory))
        {
            Console.WriteLine($"[LkTerrainExtractor:error] Directory not found: {lkAdtDirectory}");
            return results;
        }

        var adtFiles = Directory.GetFiles(lkAdtDirectory, "*.adt", SearchOption.TopDirectoryOnly);
        Console.WriteLine($"[LkTerrainExtractor] Extracting terrain from {adtFiles.Length} LK ADT files");

        foreach (var adtPath in adtFiles)
        {
            try
            {
                var (row, col) = ParseTileCoordinates(Path.GetFileNameWithoutExtension(adtPath));
                var chunks = ExtractFromAdtFile(adtPath, mapName, row, col);
                results.AddRange(chunks);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[LkTerrainExtractor:warn] Failed to read {Path.GetFileName(adtPath)}: {ex.Message}");
            }
        }

        Console.WriteLine($"[LkTerrainExtractor] Extracted {results.Count} terrain chunks from {adtFiles.Length} tiles");
        return results;
    }

    private static List<McnkTerrainEntry> ExtractFromAdtFile(string adtPath, string mapName, int tileRow, int tileCol)
    {
        var results = new List<McnkTerrainEntry>();

        try
        {
            // Read entire ADT file
            var adtBytes = File.ReadAllBytes(adtPath);
            
            // Parse with Warcraft.NET
            using var ms = new MemoryStream(adtBytes);
            using var br = new BinaryReader(ms);
            
            // Find MHDR chunk
            ms.Position = 0;
            var mverMagic = br.ReadUInt32();
            if (mverMagic != 0x4D564552 && mverMagic != 0x5245564D) // "MVER" or "REVM" (little/big endian)
            {
                throw new InvalidDataException($"Invalid ADT magic: 0x{mverMagic:X8}");
            }
            
            var mverSize = br.ReadUInt32();
            var version = br.ReadUInt32();
            
            var mhdrMagic = br.ReadUInt32();
            if (mhdrMagic != 0x4D484452 && mhdrMagic != 0x5244484D) // "MHDR" or "RDHM"
            {
                throw new InvalidDataException($"No MHDR found: 0x{mhdrMagic:X8}");
            }
            
            var mhdrSize = br.ReadUInt32();
            var mhdrDataStart = ms.Position;
            
            // MHDR contains offsets to various chunks
            // Offset 0: Usually 0 or MCIN offset
            // Offset 4: MTEX offset (or MCIN if offset 0 is 0)
            var offset0 = br.ReadUInt32();
            var offset1 = br.ReadUInt32();
            
            // Find MCIN chunk (chunk index table)
            long mhdrStart = 12;
            long mcinOffset = offset0 != 0 ? (mhdrStart + 8 + offset0) : (mhdrStart + 8 + offset1);
            
            ms.Position = mcinOffset;
            var mcinMagic = br.ReadUInt32();
            if (mcinMagic != 0x4D43494E && mcinMagic != 0x4E49434D) // "MCIN" or "NICM"
            {
                Console.WriteLine($"[LkTerrainExtractor:warn] No MCIN in {Path.GetFileName(adtPath)} (found 0x{mcinMagic:X8} at {mcinOffset})");
                return results;
            }
            
            var mcinSize = br.ReadUInt32();
            
            // Read MCIN entries (256 entries, each 16 bytes: offset + size + flags + asyncId)
            var mcnkOffsets = new List<uint>();
            for (int i = 0; i < 256; i++)
            {
                var chunkOffset = br.ReadUInt32();
                var chunkSize = br.ReadUInt32();
                var chunkFlags = br.ReadUInt32();
                var chunkAsyncId = br.ReadUInt32();
                
                mcnkOffsets.Add(chunkOffset);
            }
            
            // Read each MCNK chunk using offsets from MCIN
            for (int chunkIdx = 0; chunkIdx < 256; chunkIdx++)
            {
                try
                {
                    var chunkOffset = mcnkOffsets[chunkIdx];
                    if (chunkOffset == 0) continue; // Empty chunk
                    
                    ms.Position = chunkOffset;
                    var mcnkMagic = br.ReadUInt32();
                    
                    if (mcnkMagic != 0x4D434E4B && mcnkMagic != 0x4B4E434D) // "MCNK" or "KNCM"
                    {
                        // Skip this chunk
                        continue;
                    }
                    
                    var chunkSize = br.ReadUInt32();
                    var chunkDataStart = ms.Position;
                    
                    // Read MCNK header (LK WotLK structure)
                    // ChunkDataStart is at +8 from chunk start (after magic+size)
                    var flags = br.ReadUInt32();          // +0x00
                    var indexX = br.ReadUInt32();         // +0x04
                    var indexY = br.ReadUInt32();         // +0x08
                    var nLayers = br.ReadUInt32();        // +0x0C
                    var nDoodadRefs = br.ReadUInt32();    // +0x10
                    
                    // Skip 8 offsets (32 bytes) - +0x14 to +0x34
                    ms.Position += 32;
                    
                    // AreaID at +0x38 (56 decimal) from chunkDataStart
                    var areaId = br.ReadUInt32();
                    
                    var nMapObjRefs = br.ReadUInt32();    // +0x3C (60)
                    var holes = br.ReadUInt16();          // +0x40 (64)
                    var unk = br.ReadUInt16();            // +0x42 (66)
                    
                    // Skip to position fields (from +0x44 to +0x68 = 36 bytes)
                    ms.Position += 36;
                    
                    // Position at +0x68 (104 decimal) from chunkDataStart  
                    var posX = br.ReadSingle();
                    var posY = br.ReadSingle();
                    var posZ = br.ReadSingle();
                    
                    int chunkRow = chunkIdx / 16;
                    int chunkCol = chunkIdx % 16;
                    
                    // Parse flags
                    bool hasMcsh = (flags & 0x1) != 0;
                    bool impassible = (flags & 0x2) != 0;
                    bool lqRiver = (flags & 0x4) != 0;
                    bool lqOcean = (flags & 0x8) != 0;
                    bool lqMagma = (flags & 0x10) != 0;
                    bool lqSlime = (flags & 0x20) != 0;
                    bool hasMccv = (flags & 0x40) != 0;
                    bool hasHighResHoles = (flags & 0x10000) != 0;
                    
                    bool hasHoles = holes != 0;
                    string holeType = hasHighResHoles ? "high_res" : (hasHoles ? "low_res" : "none");
                    string holeBitmapHex = hasHoles ? $"0x{holes:X4}" : "0x0000";
                    int holeCount = hasHoles ? CountBits(holes) : 0;
                    
                    var entry = new McnkTerrainEntry(
                        Map: mapName,
                        TileRow: tileRow,
                        TileCol: tileCol,
                        ChunkRow: chunkRow,
                        ChunkCol: chunkCol,
                        FlagsRaw: flags,
                        HasMcsh: hasMcsh,
                        Impassible: impassible,
                        LqRiver: lqRiver,
                        LqOcean: lqOcean,
                        LqMagma: lqMagma,
                        LqSlime: lqSlime,
                        HasMccv: hasMccv,
                        HighResHoles: hasHighResHoles,
                        AreaId: (int)areaId,
                        NumLayers: (int)nLayers,
                        HasHoles: hasHoles,
                        HoleType: holeType,
                        HoleBitmapHex: holeBitmapHex,
                        HoleCount: holeCount,
                        PositionX: posX,
                        PositionY: posY,
                        PositionZ: posZ
                    );
                    
                    results.Add(entry);
                    
                    // Skip to next MCNK
                    ms.Position = chunkDataStart + chunkSize;
                }
                catch (Exception ex)
                {
                    // Chunk read failed, likely missing/empty tile region
                    // This is normal for sparse maps
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[LkTerrainExtractor:error] Failed to parse {Path.GetFileName(adtPath)}: {ex.Message}");
        }

        return results;
    }

    private static (int row, int col) ParseTileCoordinates(string filename)
    {
        // Parse "MapName_32_45" → (32, 45)
        var parts = filename.Split('_');
        if (parts.Length >= 3)
        {
            if (int.TryParse(parts[^2], out int row) && int.TryParse(parts[^1], out int col))
            {
                return (row, col);
            }
        }
        throw new FormatException($"Cannot parse tile coordinates from: {filename}");
    }

    private static int CountBits(ushort value)
    {
        int count = 0;
        while (value != 0)
        {
            count += value & 1;
            value >>= 1;
        }
        return count;
    }
}
