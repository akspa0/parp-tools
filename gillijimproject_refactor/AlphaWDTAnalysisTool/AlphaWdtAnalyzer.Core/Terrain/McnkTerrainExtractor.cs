using System;
using System.Collections.Generic;
using System.IO;
using GillijimProject.WowFiles.Alpha;
using GillijimProject.WowFiles.LichKing;

namespace AlphaWdtAnalyzer.Core.Terrain;

/// <summary>
/// Extracts complete MCNK terrain data from Alpha WDT files.
/// Includes: all flags, liquid types, holes, AreaID, chunk positions.
/// </summary>
public sealed class McnkTerrainExtractor
{
    /// <summary>
    /// Extract complete terrain data from all MCNK chunks in the WDT
    /// </summary>
    public static List<McnkTerrainEntry> ExtractTerrain(WdtAlphaScanner wdt)
    {
        var results = new List<McnkTerrainEntry>();

        Console.WriteLine($"[McnkTerrainExtractor] Extracting terrain from {wdt.MapName}, {wdt.AdtNumbers.Count} ADTs");

        foreach (var adtNum in wdt.AdtNumbers)
        {
            var off = (adtNum < wdt.AdtMhdrOffsets.Count) ? wdt.AdtMhdrOffsets[adtNum] : 0;
            if (off <= 0) continue;

            var adt = new AdtAlpha(wdt.WdtPath, off, adtNum);
            var tileX = adt.GetXCoord();
            var tileY = adt.GetYCoord();

            // Get MCIN offsets for all chunks in this tile
            var mcnkOffsets = GetMcnkOffsets(wdt.WdtPath, off);

            using var fs = File.OpenRead(wdt.WdtPath);
            
            for (int chunkIdx = 0; chunkIdx < 256; chunkIdx++)
            {
                var mcnkOff = (chunkIdx < mcnkOffsets.Count) ? mcnkOffsets[chunkIdx] : 0;
                if (mcnkOff <= 0) continue;

                try
                {
                    var mcnk = new McnkAlpha(fs, mcnkOff, headerSize: 0, adtNum);
                    var entry = ExtractChunkData(mcnk, wdt.MapName, tileX, tileY, chunkIdx);
                    results.Add(entry);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Warning: Failed to extract MCNK {chunkIdx} from tile {tileX}_{tileY}: {ex.Message}");
                }
            }
        }

        Console.WriteLine($"[McnkTerrainExtractor] Extracted {results.Count} terrain chunks");
        return results;
    }

    private static McnkTerrainEntry ExtractChunkData(
        McnkAlpha mcnk,
        string mapName,
        int tileX,
        int tileY,
        int chunkIdx)
    {
        // Get header via reflection (the McnkAlpha class doesn't expose it directly)
        var headerField = typeof(McnkAlpha).GetField("_mcnkAlphaHeader",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        var header = (GillijimProject.WowFiles.McnkAlphaHeader)headerField!.GetValue(mcnk)!;

        // Calculate chunk row/col from index
        int chunkRow = chunkIdx / 16;
        int chunkCol = chunkIdx % 16;

        // Parse flags
        uint flags = unchecked((uint)header.Flags);
        bool hasMcsh = (flags & 0x1) != 0;
        bool impassible = (flags & 0x2) != 0;
        bool lqRiver = (flags & 0x4) != 0;
        bool lqOcean = (flags & 0x8) != 0;
        bool lqMagma = (flags & 0x10) != 0;
        bool lqSlime = (flags & 0x20) != 0;
        bool hasMccv = (flags & 0x40) != 0;
        // Note: High-res holes flag doesn't exist in Alpha, only low-res holes

        // AreaID (from Unknown3 field - this is the Alpha AreaID encoding)
        int areaId = header.Unknown3;

        // Holes (16-bit bitmap in Alpha)
        int holes = header.Holes;
        bool hasHoles = holes != 0;
        string holeType = hasHoles ? "low_res" : "none";
        string holeBitmapHex = hasHoles ? $"0x{holes:X4}" : "0x0000";
        int holeCount = hasHoles ? CountBits((ushort)holes) : 0;

        // Calculate chunk world position
        // In Alpha, chunks don't have PosX/Y/Z in header, we need to calculate from tile + chunk indices
        var (posX, posY, posZ) = McnkLk.ComputePositionFromAdt(
            tileY * 64 + tileX, // adtNumber
            header.IndexX,
            header.IndexY);

        return new McnkTerrainEntry(
            Map: mapName,
            TileRow: tileY,
            TileCol: tileX,
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
            HighResHoles: false, // Alpha doesn't have high-res holes
            AreaId: areaId,
            NumLayers: header.NLayers,
            HasHoles: hasHoles,
            HoleType: holeType,
            HoleBitmapHex: holeBitmapHex,
            HoleCount: holeCount,
            PositionX: posX,
            PositionY: posY,
            PositionZ: posZ
        );
    }

    /// <summary>
    /// Get MCNK offsets from MCIN chunk
    /// </summary>
    private static List<int> GetMcnkOffsets(string wdtPath, int mhdrOffset)
    {
        var offsets = new List<int>();
        
        try
        {
            using var fs = File.OpenRead(wdtPath);
            
            // Read MHDR to get MCIN offset
            const int ChunkLettersAndSize = 8;
            const int mcinOffsetInMhdr = 0x0;
            
            fs.Seek(mhdrOffset + ChunkLettersAndSize + mcinOffsetInMhdr, SeekOrigin.Begin);
            byte[] mcinOffsetBytes = new byte[4];
            fs.Read(mcinOffsetBytes, 0, 4);
            int mcinOffset = BitConverter.ToInt32(mcinOffsetBytes, 0);
            
            // Read MCIN chunk (256 entries × 16 bytes each)
            int mcinAbsoluteOffset = mhdrOffset + ChunkLettersAndSize + mcinOffset;
            fs.Seek(mcinAbsoluteOffset + ChunkLettersAndSize, SeekOrigin.Begin);
            
            for (int i = 0; i < 256; i++)
            {
                byte[] entry = new byte[16];
                fs.Read(entry, 0, 16);
                
                int offset = BitConverter.ToInt32(entry, 0);
                offsets.Add(offset);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Warning: Failed to read MCIN offsets: {ex.Message}");
        }
        
        return offsets;
    }

    /// <summary>
    /// Count number of set bits in a 16-bit value (for hole counting)
    /// </summary>
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
