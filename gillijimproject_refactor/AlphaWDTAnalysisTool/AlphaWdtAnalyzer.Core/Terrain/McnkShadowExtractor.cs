using System;
using System.Collections.Generic;
using System.IO;
using GillijimProject.WowFiles.Alpha;

namespace AlphaWdtAnalyzer.Core.Terrain;

/// <summary>
/// Extracts MCSH shadow map data from Alpha WDT files.
/// Shadow maps are 64×64 bit arrays (512 bytes) stored per chunk.
/// </summary>
public sealed class McnkShadowExtractor
{
    /// <summary>
    /// Extract shadow map data from all MCNK chunks in the WDT
    /// </summary>
    public static List<McnkShadowEntry> ExtractShadows(WdtAlphaScanner wdt)
    {
        var results = new List<McnkShadowEntry>();

        Console.WriteLine($"[McnkShadowExtractor] Extracting shadow maps from {wdt.MapName}, {wdt.AdtNumbers.Count} ADTs");

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
                if (mcnkOff <= 0)
                {
                    // Add empty entry for missing chunks
                    int chunkRow = chunkIdx / 16;
                    int chunkCol = chunkIdx % 16;
                    results.Add(new McnkShadowEntry(
                        wdt.MapName, tileY, tileX, chunkRow, chunkCol,
                        HasShadow: false,
                        ShadowSize: 0,
                        ShadowBitmapBase64: string.Empty
                    ));
                    continue;
                }

                try
                {
                    var entry = ExtractShadowData(fs, mcnkOff, wdt.MapName, tileX, tileY, chunkIdx);
                    results.Add(entry);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Warning: Failed to extract shadow from MCNK {chunkIdx} in tile {tileX}_{tileY}: {ex.Message}");
                    
                    // Add empty entry on error
                    int chunkRow = chunkIdx / 16;
                    int chunkCol = chunkIdx % 16;
                    results.Add(new McnkShadowEntry(
                        wdt.MapName, tileY, tileX, chunkRow, chunkCol,
                        HasShadow: false,
                        ShadowSize: 0,
                        ShadowBitmapBase64: string.Empty
                    ));
                }
            }
        }

        int shadowCount = results.Count(e => e.HasShadow);
        Console.WriteLine($"[McnkShadowExtractor] Extracted {shadowCount}/{results.Count} shadow maps");
        return results;
    }

    private static McnkShadowEntry ExtractShadowData(
        FileStream fs,
        int mcnkOffset,
        string mapName,
        int tileX,
        int tileY,
        int chunkIdx)
    {
        int chunkRow = chunkIdx / 16;
        int chunkCol = chunkIdx % 16;

        // Read MCNK header to get flags and MCSH offset/size
        const int ChunkLettersAndSize = 8;
        fs.Seek(mcnkOffset + ChunkLettersAndSize, SeekOrigin.Begin);
        
        byte[] headerBytes = new byte[128]; // Alpha MCNK header is 128 bytes
        fs.Read(headerBytes, 0, 128);

        // Parse header fields
        uint flags = BitConverter.ToUInt32(headerBytes, 0x00);
        int mcshOffset = BitConverter.ToInt32(headerBytes, 0x30);
        int mcshSize = BitConverter.ToInt32(headerBytes, 0x34);

        // Check if MCSH flag is set
        bool hasMcsh = (flags & 0x1) != 0;

        if (!hasMcsh || mcshSize == 0 || mcshOffset == 0)
        {
            return new McnkShadowEntry(
                mapName, tileY, tileX, chunkRow, chunkCol,
                HasShadow: false,
                ShadowSize: 0,
                ShadowBitmapBase64: string.Empty
            );
        }

        // Read MCSH shadow data
        // MCSH is located relative to MCNK header start + 128 bytes (header size) + 8 bytes (chunk header)
        int mcshAbsoluteOffset = mcnkOffset + 128 + ChunkLettersAndSize + mcshOffset;
        fs.Seek(mcshAbsoluteOffset + ChunkLettersAndSize, SeekOrigin.Begin); // Skip "MCSH" + size

        byte[] compressedShadow = new byte[mcshSize];
        int bytesRead = fs.Read(compressedShadow, 0, mcshSize);

        if (bytesRead != mcshSize)
        {
            Console.WriteLine($"Warning: Expected {mcshSize} bytes for shadow map, got {bytesRead}");
        }

        // Decode shadow map (512 bytes compressed → 4096 bytes uncompressed)
        // Store as intensity digit string (4096 chars: '0' or '5')
        string shadowDigits;
        if (mcshSize == 512)
        {
            var uncompressedShadow = McshDecoder.DecodeWithEdgeFix(compressedShadow);
            shadowDigits = McshDecoder.EncodeAsDigits(uncompressedShadow);
        }
        else
        {
            // Fallback for unexpected sizes (though should always be 512)
            Console.WriteLine($"Warning: MCSH size is {mcshSize}, expected 512. Using all-lit fallback.");
            shadowDigits = new string('5', 4096); // All lit
        }

        return new McnkShadowEntry(
            mapName, tileY, tileX, chunkRow, chunkCol,
            HasShadow: true,
            ShadowSize: mcshSize,
            ShadowBitmapBase64: shadowDigits // Actually intensity digits now, not base64
        );
    }

    /// <summary>
    /// Get MCNK offsets from MCIN chunk (same as in McnkTerrainExtractor)
    /// </summary>
    private static List<int> GetMcnkOffsets(string wdtPath, int mhdrOffset)
    {
        var offsets = new List<int>();

        try
        {
            using var fs = File.OpenRead(wdtPath);

            const int ChunkLettersAndSize = 8;
            const int mcinOffsetInMhdr = 0x0;

            fs.Seek(mhdrOffset + ChunkLettersAndSize + mcinOffsetInMhdr, SeekOrigin.Begin);
            byte[] mcinOffsetBytes = new byte[4];
            fs.Read(mcinOffsetBytes, 0, 4);
            int mcinOffset = BitConverter.ToInt32(mcinOffsetBytes, 0);

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
}
