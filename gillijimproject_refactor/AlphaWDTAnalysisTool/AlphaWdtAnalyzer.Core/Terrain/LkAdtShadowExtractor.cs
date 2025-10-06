using System;
using System.Collections.Generic;
using System.IO;

namespace AlphaWdtAnalyzer.Core.Terrain;

/// <summary>
/// Extracts MCSH shadow map data directly from LK (3.3.5) ADT files.
/// Bypasses Alpha WDT parsing, enabling shadow extraction from cached LK ADTs.
/// </summary>
public static class LkAdtShadowExtractor
{
    /// <summary>
    /// Extract shadow data directly from LK ADT directory (no Alpha WDT needed).
    /// </summary>
    public static List<McnkShadowEntry> ExtractFromLkAdts(string lkAdtDirectory, string mapName)
    {
        var results = new List<McnkShadowEntry>();
        
        if (!Directory.Exists(lkAdtDirectory))
        {
            Console.WriteLine($"[LkShadowExtractor:error] Directory not found: {lkAdtDirectory}");
            return results;
        }

        var adtFiles = Directory.GetFiles(lkAdtDirectory, "*.adt", SearchOption.TopDirectoryOnly);
        Console.WriteLine($"[LkShadowExtractor] Extracting shadows from {adtFiles.Length} LK ADT files");

        foreach (var adtPath in adtFiles)
        {
            try
            {
                var (row, col) = ParseTileCoordinates(Path.GetFileNameWithoutExtension(adtPath));
                var shadows = ExtractFromAdtFile(adtPath, mapName, row, col);
                results.AddRange(shadows);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[LkShadowExtractor:warn] Failed to read {Path.GetFileName(adtPath)}: {ex.Message}");
            }
        }

        Console.WriteLine($"[LkShadowExtractor] Extracted {results.Count} shadow entries from {adtFiles.Length} tiles");
        return results;
    }

    private static List<McnkShadowEntry> ExtractFromAdtFile(string adtPath, string mapName, int tileRow, int tileCol)
    {
        var results = new List<McnkShadowEntry>();

        try
        {
            var adtBytes = File.ReadAllBytes(adtPath);
            
            using var ms = new MemoryStream(adtBytes);
            using var br = new BinaryReader(ms);
            
            // Find MHDR chunk
            ms.Position = 0;
            var mverMagic = br.ReadUInt32();
            if (mverMagic != 0x4D564552 && mverMagic != 0x5245564D) // "MVER" or "REVM" (little/big endian)
            {
                throw new InvalidDataException($"Invalid ADT magic: 0x{mverMagic:X8}");
            }
            
            br.ReadUInt32(); // mverSize
            br.ReadUInt32(); // version
            
            var mhdrMagic = br.ReadUInt32();
            if (mhdrMagic != 0x4D484452 && mhdrMagic != 0x5244484D) // "MHDR" or "RDHM"
            {
                throw new InvalidDataException($"No MHDR found: 0x{mhdrMagic:X8}");
            }
            
            br.ReadUInt32(); // mhdrSize
            var offset0 = br.ReadUInt32();
            var offset1 = br.ReadUInt32();
            
            // Find MCIN chunk
            long mhdrStart = 12;
            long mcinOffset = offset0 != 0 ? (mhdrStart + 8 + offset0) : (mhdrStart + 8 + offset1);
            
            ms.Position = mcinOffset;
            var mcinMagic = br.ReadUInt32();
            if (mcinMagic != 0x4D43494E && mcinMagic != 0x4E49434D) // "MCIN" or "NICM"
            {
                Console.WriteLine($"[LkShadowExtractor:warn] No MCIN in {Path.GetFileName(adtPath)}");
                return results;
            }
            
            var mcinSize = br.ReadUInt32();
            
            // Read MCIN entries
            var mcnkOffsets = new List<uint>();
            for (int i = 0; i < 256; i++)
            {
                var chunkOffset = br.ReadUInt32();
                br.ReadUInt32(); // size
                br.ReadUInt32(); // flags
                br.ReadUInt32(); // asyncId
                mcnkOffsets.Add(chunkOffset);
            }
            
            // Read each MCNK chunk
            for (int chunkIdx = 0; chunkIdx < 256; chunkIdx++)
            {
                int chunkRow = chunkIdx / 16;
                int chunkCol = chunkIdx % 16;
                
                try
                {
                    var chunkOffset = mcnkOffsets[chunkIdx];
                    if (chunkOffset == 0)
                    {
                        // Empty chunk - add placeholder
                        results.Add(new McnkShadowEntry(
                            mapName, tileRow, tileCol, chunkRow, chunkCol,
                            HasShadow: false,
                            ShadowSize: 0,
                            ShadowBitmapBase64: string.Empty
                        ));
                        continue;
                    }
                    
                    ms.Position = chunkOffset;
                    var mcnkMagic = br.ReadUInt32();
                    
                    if (mcnkMagic != 0x4D434E4B && mcnkMagic != 0x4B4E434D) // "MCNK" or "KNCM"
                    {
                        // Invalid chunk - add placeholder
                        results.Add(new McnkShadowEntry(
                            mapName, tileRow, tileCol, chunkRow, chunkCol,
                            HasShadow: false,
                            ShadowSize: 0,
                            ShadowBitmapBase64: string.Empty
                        ));
                        continue;
                    }
                    
                    var chunkSize = br.ReadUInt32();
                    var chunkDataStart = ms.Position;
                    
                    // Read MCNK header to get flags and MCSH offset
                    var flags = br.ReadUInt32();
                    bool hasMcsh = (flags & 0x1) != 0;
                    
                    // Skip to shadow offset field (offset 40 in header)
                    ms.Position = chunkDataStart + 36;
                    var mcshOffset = br.ReadUInt32();
                    var mcshSize = br.ReadUInt32();
                    
                    if (!hasMcsh || mcshOffset == 0 || mcshSize == 0)
                    {
                        // No shadow data
                        results.Add(new McnkShadowEntry(
                            mapName, tileRow, tileCol, chunkRow, chunkCol,
                            HasShadow: false,
                            ShadowSize: 0,
                            ShadowBitmapBase64: string.Empty
                        ));
                    }
                    else
                    {
                        // Read MCSH shadow data
                        // MCSH offset is relative to chunk start + 8 (chunk header)
                        long mcshAbsolutePos = chunkDataStart + mcshOffset - 8;
                        ms.Position = mcshAbsolutePos;
                        
                        var mcshMagic = br.ReadUInt32();
                        if (mcshMagic == 0x4D435348) // "MCSH"
                        {
                            var mcshDataSize = br.ReadUInt32();
                            var shadowData = br.ReadBytes((int)mcshDataSize);
                            
                            // Convert to Base64 for CSV storage
                            string shadowBase64 = Convert.ToBase64String(shadowData);
                            
                            results.Add(new McnkShadowEntry(
                                mapName, tileRow, tileCol, chunkRow, chunkCol,
                                HasShadow: true,
                                ShadowSize: shadowData.Length,
                                ShadowBitmapBase64: shadowBase64
                            ));
                        }
                        else
                        {
                            // MCSH magic not found where expected
                            results.Add(new McnkShadowEntry(
                                mapName, tileRow, tileCol, chunkRow, chunkCol,
                                HasShadow: false,
                                ShadowSize: 0,
                                ShadowBitmapBase64: string.Empty
                            ));
                        }
                    }
                    
                    // Skip to next MCNK
                    ms.Position = chunkDataStart + chunkSize;
                }
                catch (Exception ex)
                {
                    // Chunk read failed - add empty entry
                    results.Add(new McnkShadowEntry(
                        mapName, tileRow, tileCol, chunkRow, chunkCol,
                        HasShadow: false,
                        ShadowSize: 0,
                        ShadowBitmapBase64: string.Empty
                    ));
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[LkShadowExtractor:error] Failed to parse {Path.GetFileName(adtPath)}: {ex.Message}");
        }

        return results;
    }

    private static (int row, int col) ParseTileCoordinates(string filename)
    {
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
