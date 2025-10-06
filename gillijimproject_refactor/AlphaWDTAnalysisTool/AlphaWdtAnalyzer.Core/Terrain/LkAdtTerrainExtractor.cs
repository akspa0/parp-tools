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
            if (mverMagic != 0x4D564552) // "MVER" (little-endian)
            {
                throw new InvalidDataException($"Invalid ADT magic: 0x{mverMagic:X8}");
            }
            
            var mverSize = br.ReadUInt32();
            var version = br.ReadUInt32();
            
            var mhdrMagic = br.ReadUInt32();
            if (mhdrMagic != 0x4D484452) // "MHDR"
            {
                throw new InvalidDataException($"No MHDR found: 0x{mhdrMagic:X8}");
            }
            
            var mhdrSize = br.ReadUInt32();
            var mcnkOffset = br.ReadUInt32();
            
            if (mcnkOffset == 0)
            {
                Console.WriteLine($"[LkTerrainExtractor:warn] No MCNK in {Path.GetFileName(adtPath)}");
                return results;
            }
            
            // Jump to MCNK chunks
            ms.Position = mcnkOffset;
            
            // Read 16×16 MCNK chunks
            for (int chunkIdx = 0; chunkIdx < 256; chunkIdx++)
            {
                try
                {
                    long chunkStart = ms.Position;
                    var mcnkMagic = br.ReadUInt32();
                    
                    if (mcnkMagic != 0x4D434E4B) // "MCNK"
                    {
                        // Skip this chunk, likely empty/missing
                        continue;
                    }
                    
                    var chunkSize = br.ReadUInt32();
                    var chunkDataStart = ms.Position;
                    
                    // Read MCNK header (128 bytes total in LK)
                    var flags = br.ReadUInt32();
                    var indexX = br.ReadUInt32();
                    var indexY = br.ReadUInt32();
                    var nLayers = br.ReadUInt32();
                    var nDoodadRefs = br.ReadUInt32();
                    
                    // Skip 8 offsets (32 bytes)
                    ms.Position += 32;
                    
                    // AreaID at offset 56 in MCNK header
                    var areaId = br.ReadUInt32();
                    
                    // Skip to position fields (offset 68)
                    ms.Position = chunkDataStart + 68;
                    var nMapObjRefs = br.ReadUInt32();
                    var holes = br.ReadUInt16();
                    var unk = br.ReadUInt16();
                    
                    // Skip to position (offset 80)
                    ms.Position = chunkDataStart + 80;
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
