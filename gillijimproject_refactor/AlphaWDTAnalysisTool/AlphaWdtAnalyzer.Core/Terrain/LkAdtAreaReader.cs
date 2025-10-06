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
        var offset0 = br.ReadUInt32();
        var offset1 = br.ReadUInt32();
        
        // Find MCIN chunk (chunk index table)
        long mhdrStart = 12;
        long mcinOffset = offset0 != 0 ? (mhdrStart + 8 + offset0) : (mhdrStart + 8 + offset1);
        
        fs.Position = mcinOffset;
        var mcinMagic = br.ReadUInt32();
        if (mcinMagic != 0x4D43494E && mcinMagic != 0x4E49434D) // "MCIN" or "NICM"
        {
            Console.WriteLine($"[area:warn] No MCIN in {Path.GetFileName(adtPath)} (found 0x{mcinMagic:X8})");
            return results;
        }
        
        var mcinSize = br.ReadUInt32();
        
        // Read MCIN entries (256 entries, each 16 bytes)
        var mcnkOffsets = new List<uint>();
        for (int i = 0; i < 256; i++)
        {
            var chunkOffset = br.ReadUInt32();
            br.ReadUInt32(); // size
            br.ReadUInt32(); // flags
            br.ReadUInt32(); // asyncId
            mcnkOffsets.Add(chunkOffset);
        }
        
        // Read AreaID from each MCNK chunk
        for (int chunkIdx = 0; chunkIdx < 256; chunkIdx++)
        {
            int chunkY = chunkIdx / 16;
            int chunkX = chunkIdx % 16;
            
            try
            {
                var chunkOffset = mcnkOffsets[chunkIdx];
                if (chunkOffset == 0) continue; // Empty chunk
                
                fs.Position = chunkOffset;
                var mcnkMagic = br.ReadUInt32();
                if (mcnkMagic != 0x4D434E4B && mcnkMagic != 0x4B4E434D) // "MCNK" or "KNCM"
                {
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
                
                // Skip offsets (8 × uint32 = 32 bytes)
                fs.Position += 32;
                
                var areaid = br.ReadUInt32(); // AreaID at offset 56 from chunkStart
                    
                    results.Add(new ChunkAreaInfo(
                        mapName,
                        tileRow,
                        tileCol,
                        chunkY,
                        chunkX,
                        (int)areaid
                    ));
                    
                // Note: No need to skip to next chunk since we're using MCIN offsets
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[area:warn] Error reading chunk ({chunkY},{chunkX}): {ex.Message}");
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
