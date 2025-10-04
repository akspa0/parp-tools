using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace AlphaWdtAnalyzer.Core.Terrain;

/// <summary>
/// Reads AreaID values from converted LK ADT files.
/// LK AreaIDs correctly map to 3.3.5 AreaTable.dbc, unlike Alpha AreaIDs.
/// </summary>
public static class LkAdtAreaReader
{
    public record ChunkAreaInfo(
        string MapName,
        int TileRow,
        int TileCol,
        int ChunkY,
        int ChunkX,
        int AreaId
    );

    /// <summary>
    /// Extracts AreaID from all MCNK chunks in a LK ADT directory.
    /// </summary>
    public static List<ChunkAreaInfo> ExtractAreaIds(string lkAdtDirectory, string mapName)
    {
        var results = new List<ChunkAreaInfo>();
        
        if (!Directory.Exists(lkAdtDirectory))
        {
            Console.WriteLine($"[area] LK ADT directory not found: {lkAdtDirectory}");
            return results;
        }

        var adtFiles = Directory.GetFiles(lkAdtDirectory, "*.adt", SearchOption.TopDirectoryOnly);
        Console.WriteLine($"[area] Reading AreaIDs from {adtFiles.Length} LK ADT files in {Path.GetFileName(lkAdtDirectory)}");

        foreach (var adtPath in adtFiles)
        {
            try
            {
                var (row, col) = ParseTileCoordinates(Path.GetFileNameWithoutExtension(adtPath));
                var chunkAreas = ReadAdtAreaIds(adtPath, mapName, row, col);
                results.AddRange(chunkAreas);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[area:warn] Failed to read {Path.GetFileName(adtPath)}: {ex.Message}");
            }
        }

        Console.WriteLine($"[area] Extracted AreaIDs for {results.Count} chunks from {adtFiles.Length} tiles");
        return results;
    }

    private static List<ChunkAreaInfo> ReadAdtAreaIds(string adtPath, string mapName, int tileRow, int tileCol)
    {
        var results = new List<ChunkAreaInfo>();

        using var fs = File.OpenRead(adtPath);
        using var br = new BinaryReader(fs);
        
        // Find MHDR chunk (always at offset 20)
        fs.Position = 0;
        var mverMagic = br.ReadUInt32(); // MVER
        if (mverMagic != 0x4D564552 && mverMagic != 0x5245564D) // "MVER" or reversed
        {
            throw new InvalidDataException($"Invalid ADT file: {Path.GetFileName(adtPath)}");
        }
        
        var mverSize = br.ReadUInt32();
        var version = br.ReadUInt32();
        
        var mhdrMagic = br.ReadUInt32(); // MHDR
        if (mhdrMagic != 0x4D484452 && mhdrMagic != 0x5244484D) // "MHDR" or reversed
        {
            throw new InvalidDataException($"No MHDR chunk found in {Path.GetFileName(adtPath)}");
        }
        
        var mhdrSize = br.ReadUInt32();
        var mcnkOffset = br.ReadUInt32(); // Offset to MCNK chunks
        
        if (mcnkOffset == 0)
        {
            Console.WriteLine($"[area:warn] No MCNK offset in {Path.GetFileName(adtPath)}");
            return results;
        }
        
        // Jump to MCNK chunks
        fs.Position = mcnkOffset;
        
        // Read 16×16 MCNK chunks
        for (int chunkY = 0; chunkY < 16; chunkY++)
        {
            for (int chunkX = 0; chunkX < 16; chunkX++)
            {
                try
                {
                    var mcnkMagic = br.ReadUInt32();
                    if (mcnkMagic != 0x4D434E4B && mcnkMagic != 0x4B4E434D) // "MCNK" or reversed
                    {
                        Console.WriteLine($"[area:warn] Expected MCNK at chunk ({chunkY},{chunkX}), got 0x{mcnkMagic:X8}");
                        continue;
                    }
                    
                    var chunkSize = br.ReadUInt32();
                    var chunkStart = fs.Position;
                    
                    // Read MCNK header fields
                    var flags = br.ReadUInt32();
                    var indexX = br.ReadUInt32();
                    var indexY = br.ReadUInt32();
                    var nLayers = br.ReadUInt32();
                    var nDoodadRefs = br.ReadUInt32();
                    
                    // Skip offsets (8 × uint32)
                    fs.Position += 32;
                    
                    var areaid = br.ReadUInt32(); // AreaID at offset 56 in MCNK
                    
                    results.Add(new ChunkAreaInfo(
                        mapName,
                        tileRow,
                        tileCol,
                        chunkY,
                        chunkX,
                        (int)areaid
                    ));
                    
                    // Skip to next MCNK chunk
                    fs.Position = chunkStart + chunkSize;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[area:warn] Error reading chunk ({chunkY},{chunkX}): {ex.Message}");
                }
            }
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
}
