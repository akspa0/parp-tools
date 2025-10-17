using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using WoWRollback.LkToAlphaModule.Builders;
using WoWRollback.LkToAlphaModule.Readers;
using WoWRollback.LkToAlphaModule.Services;

namespace WoWRollback.LkToAlphaModule.Validators;

/// <summary>
/// Validates round-trip conversions: Alpha → LK → Alpha.
/// Ensures data integrity through the conversion pipeline.
/// </summary>
public static class RoundTripValidator
{
    public record ValidationResult(
        bool Success, 
        string? ErrorMessage = null,
        int TilesProcessed = 0,
        int TilesMismatched = 0,
        long BytesDifferent = 0,
        List<ChunkDifference>? Differences = null);
    
    public record ChunkDifference(
        int ChunkIndex,
        int IndexX,
        int IndexY,
        long ByteOffset,
        int DifferentBytes,
        string Description);
    
    /// <summary>
    /// Performs a round-trip test: Alpha WDT/ADT → LK ADT → Alpha WDT/ADT.
    /// Handles both monolithic WDTs (0.5.3) and standalone ADT files.
    /// Compares the final output with the original to validate conversion accuracy.
    /// </summary>
    public static ValidationResult ValidateRoundTrip(
        string originalAlphaPath, 
        string outputDir,
        LkToAlphaOptions? options = null)
    {
        try
        {
            if (!File.Exists(originalAlphaPath))
                return new ValidationResult(false, $"Original Alpha file not found: {originalAlphaPath}");
            
            options ??= new LkToAlphaOptions();
            Directory.CreateDirectory(outputDir);
            
            var originalBytes = File.ReadAllBytes(originalAlphaPath);
            var fileName = Path.GetFileName(originalAlphaPath);
            var isWdt = fileName.EndsWith(".wdt", StringComparison.OrdinalIgnoreCase);
            
            Console.WriteLine($"[RoundTrip] Starting validation for: {fileName}");
            Console.WriteLine($"[RoundTrip] File type: {(isWdt ? "Monolithic WDT (0.5.3)" : "Standalone ADT")}");
            Console.WriteLine($"[RoundTrip] Original file size: {originalBytes.Length:N0} bytes");
            
            if (isWdt)
            {
                // Handle monolithic WDT (Alpha 0.5.3 format)
                return ValidateRoundTripMonolithicWdt(originalAlphaPath, outputDir, options, originalBytes, fileName);
            }
            else
            {
                // Handle standalone ADT file
                return ValidateRoundTripStandaloneAdt(originalAlphaPath, outputDir, options, originalBytes, fileName);
            }
        }
        catch (Exception ex)
        {
            return new ValidationResult(false, $"Round-trip validation failed: {ex.Message}\n{ex.StackTrace}");
        }
    }
    
    /// <summary>
    /// Validates round-trip for standalone Alpha ADT files.
    /// </summary>
    private static ValidationResult ValidateRoundTripStandaloneAdt(
        string originalAlphaAdtPath,
        string outputDir,
        LkToAlphaOptions options,
        byte[] originalBytes,
        string fileName)
    {
        // Step 1: Extract Alpha data to LkAdtSource
        Console.WriteLine("[RoundTrip] Step 1: Extracting Alpha ADT data...");
        var lkSource = AlphaDataExtractor.ExtractFromAlphaAdt(originalAlphaAdtPath);
        Console.WriteLine($"[RoundTrip] Extracted {lkSource.Mcnks.Count} MCNK chunks");
        
        // Step 2: Build LK ADT using managed builders
        Console.WriteLine("[RoundTrip] Step 2: Building LK ADT...");
        var lkAdtBytes = LkAdtBuilder.Build(lkSource, options);
        
        // Save intermediate LK ADT for inspection
        var lkAdtPath = Path.Combine(outputDir, fileName.Replace(".adt", "_lk.adt"));
        File.WriteAllBytes(lkAdtPath, lkAdtBytes);
        Console.WriteLine($"[RoundTrip] LK ADT saved: {lkAdtPath} ({lkAdtBytes.Length:N0} bytes)");
        
        // Step 3: Convert LK back to Alpha using AlphaMcnkBuilder
        Console.WriteLine("[RoundTrip] Step 3: Converting LK ADT back to Alpha...");
        var convertedAlphaBytes = ConvertLkToAlpha(lkAdtBytes, options);
        
        // Save converted Alpha ADT for inspection
        var convertedAlphaPath = Path.Combine(outputDir, fileName.Replace(".adt", "_roundtrip.adt"));
        File.WriteAllBytes(convertedAlphaPath, convertedAlphaBytes);
        Console.WriteLine($"[RoundTrip] Converted Alpha ADT saved: {convertedAlphaPath} ({convertedAlphaBytes.Length:N0} bytes)");
        
        // Step 4: Compare original with converted
        Console.WriteLine("[RoundTrip] Step 4: Comparing original with round-trip result...");
        var comparisonResult = CompareAlphaAdtsDetailed(originalBytes, convertedAlphaBytes, fileName);
        
        if (comparisonResult.Success)
        {
            Console.WriteLine("[RoundTrip] ✓ Round-trip validation PASSED - Files are identical!");
        }
        else
        {
            Console.WriteLine($"[RoundTrip] ✗ Round-trip validation FAILED - {comparisonResult.ErrorMessage}");
            if (comparisonResult.Differences != null && comparisonResult.Differences.Count > 0)
            {
                Console.WriteLine($"[RoundTrip] Found {comparisonResult.Differences.Count} chunks with differences:");
                foreach (var diff in comparisonResult.Differences.Take(10))
                {
                    Console.WriteLine($"  - Chunk [{diff.IndexX},{diff.IndexY}] @ offset {diff.ByteOffset}: {diff.DifferentBytes} bytes differ - {diff.Description}");
                }
                if (comparisonResult.Differences.Count > 10)
                {
                    Console.WriteLine($"  ... and {comparisonResult.Differences.Count - 10} more");
                }
            }
        }
        
        return comparisonResult with { TilesProcessed = lkSource.Mcnks.Count };
    }
    
    /// <summary>
    /// Validates round-trip for monolithic Alpha WDT files (0.5.3 format with embedded ADTs).
    /// </summary>
    private static ValidationResult ValidateRoundTripMonolithicWdt(
        string originalWdtPath,
        string outputDir,
        LkToAlphaOptions options,
        byte[] originalBytes,
        string fileName)
    {
        Console.WriteLine("[RoundTrip] Step 1: Reading monolithic WDT structure...");
        
        // Parse the monolithic WDT to understand its structure
        var alphaWdt = Readers.AlphaWdtReader.Read(originalWdtPath);
        Console.WriteLine($"[RoundTrip] Found {alphaWdt.Tiles.Count} tiles in WDT");
        
        if (alphaWdt.Tiles.Count == 0)
        {
            return new ValidationResult(false, "No tiles found in monolithic WDT");
        }
        
        // Step 2: Extract each tile's ADT data and convert to LK
        Console.WriteLine("[RoundTrip] Step 2: Extracting and converting tiles to LK format...");
        var lkAdtDir = Path.Combine(outputDir, "lk_adts");
        Directory.CreateDirectory(lkAdtDir);
        
        var tileResults = new List<(int index, bool success, string? error)>();
        int successCount = 0;
        
        foreach (var tile in alphaWdt.Tiles)
        {
            try
            {
                // Extract tile's ADT data from the monolithic WDT
                var tileAdtBytes = ExtractTileAdtFromWdt(originalBytes, tile);
                
                // Save extracted tile for debugging
                var extractedPath = Path.Combine(outputDir, $"tile_{tile.Index:D4}_extracted.adt");
                File.WriteAllBytes(extractedPath, tileAdtBytes);
                
                // Convert to LK format
                var tempAdtPath = Path.Combine(outputDir, $"temp_tile_{tile.Index:D4}.adt");
                File.WriteAllBytes(tempAdtPath, tileAdtBytes);
                
                var lkSource = AlphaDataExtractor.ExtractFromAlphaAdt(tempAdtPath);
                var lkAdtBytes = LkAdtBuilder.Build(lkSource, options);
                
                // Save LK ADT
                var lkAdtPath = Path.Combine(lkAdtDir, $"tile_{tile.Index:D4}_lk.adt");
                File.WriteAllBytes(lkAdtPath, lkAdtBytes);
                
                // Clean up temp file
                File.Delete(tempAdtPath);
                
                tileResults.Add((tile.Index, true, null));
                successCount++;
                
                if (successCount % 10 == 0)
                {
                    Console.WriteLine($"[RoundTrip] Converted {successCount}/{alphaWdt.Tiles.Count} tiles...");
                }
            }
            catch (Exception ex)
            {
                tileResults.Add((tile.Index, false, ex.Message));
                Console.WriteLine($"[RoundTrip] Warning: Failed to convert tile {tile.Index}: {ex.Message}");
            }
        }
        
        Console.WriteLine($"[RoundTrip] Converted {successCount}/{alphaWdt.Tiles.Count} tiles successfully");
        
        if (successCount == 0)
        {
            return new ValidationResult(
                false,
                $"Failed to convert any tiles. First error: {tileResults.FirstOrDefault(r => !r.success).error}",
                TilesProcessed: alphaWdt.Tiles.Count);
        }
        
        // Step 3: Convert LK ADTs back to Alpha and pack into monolithic WDT
        Console.WriteLine("[RoundTrip] Step 3: Converting LK ADTs back to Alpha format...");
        
        var convertedTiles = new Dictionary<int, byte[]>();
        int convertBackCount = 0;
        
        foreach (var result in tileResults.Where(r => r.success))
        {
            try
            {
                var lkAdtPath = Path.Combine(lkAdtDir, $"tile_{result.index:D4}_lk.adt");
                var lkAdtBytes = File.ReadAllBytes(lkAdtPath);
                
                // Convert back to Alpha
                var alphaAdtBytes = ConvertLkToAlpha(lkAdtBytes, options);
                convertedTiles[result.index] = alphaAdtBytes;
                convertBackCount++;
                
                if (convertBackCount % 10 == 0)
                {
                    Console.WriteLine($"[RoundTrip] Converted back {convertBackCount}/{successCount} tiles...");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[RoundTrip] Warning: Failed to convert tile {result.index} back to Alpha: {ex.Message}");
            }
        }
        
        Console.WriteLine($"[RoundTrip] Converted {convertBackCount}/{successCount} tiles back to Alpha");
        
        // Step 4: Pack converted tiles into monolithic WDT
        Console.WriteLine("[RoundTrip] Step 4: Packing tiles into monolithic WDT...");
        
        // For now, save individual converted tiles and report
        // Full WDT packing would require implementing the packing logic
        var convertedTilesDir = Path.Combine(outputDir, "converted_tiles");
        Directory.CreateDirectory(convertedTilesDir);
        
        foreach (var kvp in convertedTiles)
        {
            var tilePath = Path.Combine(convertedTilesDir, $"tile_{kvp.Key:D4}_roundtrip.adt");
            File.WriteAllBytes(tilePath, kvp.Value);
        }
        
        // Step 5: Compare individual tiles
        Console.WriteLine("[RoundTrip] Step 5: Comparing converted tiles with originals...");
        
        int identicalTiles = 0;
        int differentTiles = 0;
        long totalBytesDifferent = 0;
        
        foreach (var kvp in convertedTiles)
        {
            var originalTilePath = Path.Combine(outputDir, $"tile_{kvp.Key:D4}_extracted.adt");
            if (File.Exists(originalTilePath))
            {
                var originalTileBytes = File.ReadAllBytes(originalTilePath);
                var convertedTileBytes = kvp.Value;
                
                if (originalTileBytes.Length == convertedTileBytes.Length)
                {
                    long diff = 0;
                    for (int i = 0; i < originalTileBytes.Length; i++)
                    {
                        if (originalTileBytes[i] != convertedTileBytes[i])
                            diff++;
                    }
                    
                    if (diff == 0)
                    {
                        identicalTiles++;
                    }
                    else
                    {
                        differentTiles++;
                        totalBytesDifferent += diff;
                    }
                }
                else
                {
                    differentTiles++;
                    totalBytesDifferent += Math.Abs(originalTileBytes.Length - convertedTileBytes.Length);
                }
            }
        }
        
        Console.WriteLine($"[RoundTrip] Tile comparison results:");
        Console.WriteLine($"  Identical tiles: {identicalTiles}/{convertedTiles.Count}");
        Console.WriteLine($"  Different tiles: {differentTiles}/{convertedTiles.Count}");
        Console.WriteLine($"  Total bytes different: {totalBytesDifferent:N0}");
        
        if (identicalTiles == convertedTiles.Count)
        {
            Console.WriteLine("[RoundTrip] ✓ All converted tiles are identical to originals!");
            return new ValidationResult(
                true,
                $"All {identicalTiles} tiles converted successfully with byte-for-byte parity",
                TilesProcessed: alphaWdt.Tiles.Count);
        }
        else
        {
            return new ValidationResult(
                false,
                $"Round-trip completed but {differentTiles} tiles differ. Total bytes different: {totalBytesDifferent:N0}. " +
                $"Note: Full WDT packing not yet implemented - tiles saved individually for inspection.",
                TilesProcessed: alphaWdt.Tiles.Count,
                TilesMismatched: differentTiles,
                BytesDifferent: totalBytesDifferent);
        }
    }
    
    /// <summary>
    /// Extracts a single tile's ADT data from a monolithic WDT.
    /// </summary>
    private static byte[] ExtractTileAdtFromWdt(byte[] wdtBytes, Models.AlphaTile tile)
    {
        // The tile's ADT data starts at MHDR and includes all MCNKs
        // We need to find the end of the last MCNK to know how much data to extract
        
        int mhdrOffset = tile.MhdrOffset;
        if (mhdrOffset <= 0 || mhdrOffset + 8 > wdtBytes.Length)
        {
            throw new InvalidDataException($"Invalid MHDR offset for tile {tile.Index}: {mhdrOffset}");
        }
        
        // Read MHDR to find MCIN
        int mhdrDataStart = mhdrOffset + 8;
        int mcinOffset = BitConverter.ToInt32(wdtBytes, mhdrDataStart + 0); // OffsInfo
        
        if (mcinOffset <= 0)
        {
            throw new InvalidDataException($"Invalid MCIN offset for tile {tile.Index}: {mcinOffset}");
        }
        
        int mcinPos = mhdrDataStart + mcinOffset;
        
        // MCIN has 256 entries, find the last valid MCNK
        int lastMcnkEnd = mcinPos + 8 + 4096; // Default to end of MCIN
        
        for (int i = 0; i < 256; i++)
        {
            int entryPos = mcinPos + 8 + (i * 16);
            if (entryPos + 8 > wdtBytes.Length) break;
            
            int mcnkOffset = BitConverter.ToInt32(wdtBytes, entryPos);
            int mcnkSize = BitConverter.ToInt32(wdtBytes, entryPos + 4);
            
            if (mcnkOffset > 0 && mcnkSize > 0)
            {
                int mcnkEnd = mhdrDataStart + mcnkOffset + 8 + mcnkSize;
                if (mcnkEnd > lastMcnkEnd)
                {
                    lastMcnkEnd = mcnkEnd;
                }
            }
        }
        
        // Extract from MHDR start to end of last MCNK
        int adtSize = lastMcnkEnd - mhdrOffset;
        if (adtSize <= 0 || mhdrOffset + adtSize > wdtBytes.Length)
        {
            throw new InvalidDataException($"Invalid ADT size for tile {tile.Index}: {adtSize}");
        }
        
        var adtBytes = new byte[adtSize];
        Buffer.BlockCopy(wdtBytes, mhdrOffset, adtBytes, 0, adtSize);
        
        return adtBytes;
    }
    
    /// <summary>
    /// Converts a LK ADT back to Alpha format using AlphaMcnkBuilder.
    /// </summary>
    private static byte[] ConvertLkToAlpha(byte[] lkAdtBytes, LkToAlphaOptions options)
    {
        // Read LK MCNK chunks using LkAdtReader
        var reader = new LkAdtReader();
        
        // Find all MCNK chunks in the LK ADT
        var mcnkOffsets = FindMcnkOffsets(lkAdtBytes);
        Console.WriteLine($"[LK→Alpha] Found {mcnkOffsets.Count} MCNK chunks in LK ADT");
        
        using var ms = new MemoryStream();
        
        // Convert each MCNK chunk from LK to Alpha
        int processedCount = 0;
        foreach (var offset in mcnkOffsets)
        {
            var alphaMcnkBytes = AlphaMcnkBuilder.BuildFromLk(lkAdtBytes, offset, options);
            ms.Write(alphaMcnkBytes, 0, alphaMcnkBytes.Length);
            processedCount++;
            
            if (processedCount % 64 == 0)
            {
                Console.WriteLine($"[LK→Alpha] Converted {processedCount}/{mcnkOffsets.Count} chunks...");
            }
        }
        
        Console.WriteLine($"[LK→Alpha] Conversion complete: {processedCount} chunks");
        return ms.ToArray();
    }
    
    /// <summary>
    /// Finds all MCNK chunk offsets in a LK ADT file.
    /// </summary>
    private static List<int> FindMcnkOffsets(byte[] lkAdtBytes)
    {
        var offsets = new List<int>();
        
        // Find MHDR chunk first
        int mhdrOffset = -1;
        for (int i = 0; i + 8 <= lkAdtBytes.Length;)
        {
            string fcc = Encoding.ASCII.GetString(lkAdtBytes, i, 4);
            int size = BitConverter.ToInt32(lkAdtBytes, i + 4);
            if (fcc == "RDHM") // MHDR reversed
            {
                mhdrOffset = i;
                break;
            }
            int next = i + 8 + size + ((size & 1) == 1 ? 1 : 0);
            if (next <= i || next > lkAdtBytes.Length) break;
            i = next;
        }
        
        if (mhdrOffset < 0)
        {
            throw new InvalidDataException("MHDR chunk not found in LK ADT");
        }
        
        // Read MCIN offsets from MHDR
        int mhdrDataStart = mhdrOffset + 8;
        int mcinOffset = BitConverter.ToInt32(lkAdtBytes, mhdrDataStart + 4); // MCIN offset is at +4 in MHDR
        
        if (mcinOffset > 0)
        {
            int mcinPos = mhdrDataStart + mcinOffset;
            if (mcinPos + 8 <= lkAdtBytes.Length)
            {
                string mcinFcc = Encoding.ASCII.GetString(lkAdtBytes, mcinPos, 4);
                if (mcinFcc == "NICM") // MCIN reversed
                {
                    int mcinSize = BitConverter.ToInt32(lkAdtBytes, mcinPos + 4);
                    int mcinDataStart = mcinPos + 8;
                    
                    // MCIN contains 256 entries of 16 bytes each (offset, size, flags, asyncId)
                    for (int i = 0; i < 256 && i * 16 + 4 <= mcinSize; i++)
                    {
                        int mcnkOffset = BitConverter.ToInt32(lkAdtBytes, mcinDataStart + i * 16);
                        if (mcnkOffset > 0 && mcnkOffset < lkAdtBytes.Length)
                        {
                            offsets.Add(mcnkOffset);
                        }
                    }
                }
            }
        }
        
        return offsets;
    }
    
    /// <summary>
    /// Compares two Alpha ADT files byte-by-byte.
    /// </summary>
    public static ValidationResult CompareAlphaAdts(string originalPath, string convertedPath)
    {
        try
        {
            if (!File.Exists(originalPath))
                return new ValidationResult(false, $"Original file not found: {originalPath}");
            if (!File.Exists(convertedPath))
                return new ValidationResult(false, $"Converted file not found: {convertedPath}");
            
            var original = File.ReadAllBytes(originalPath);
            var converted = File.ReadAllBytes(convertedPath);
            
            return CompareAlphaAdtsDetailed(original, converted, Path.GetFileName(originalPath));
        }
        catch (Exception ex)
        {
            return new ValidationResult(false, $"Comparison failed: {ex.Message}");
        }
    }
    
    /// <summary>
    /// Compares two Alpha ADT byte arrays with detailed chunk-level analysis.
    /// </summary>
    private static ValidationResult CompareAlphaAdtsDetailed(byte[] original, byte[] converted, string fileName)
    {
        if (original.Length != converted.Length)
        {
            return new ValidationResult(
                false, 
                $"File size mismatch: {original.Length:N0} vs {converted.Length:N0} bytes",
                BytesDifferent: Math.Abs(original.Length - converted.Length));
        }
        
        // Quick byte-by-byte check first
        long totalDifferentBytes = 0;
        for (int i = 0; i < original.Length; i++)
        {
            if (original[i] != converted[i])
                totalDifferentBytes++;
        }
        
        if (totalDifferentBytes == 0)
        {
            return new ValidationResult(true, "Files are byte-for-byte identical");
        }
        
        // Detailed chunk-level analysis
        var differences = new List<ChunkDifference>();
        int chunkIndex = 0;
        int offset = 0;
        
        while (offset + 8 <= original.Length && offset + 8 <= converted.Length)
        {
            // Read chunk header
            string origFcc = Encoding.ASCII.GetString(original, offset, 4);
            string convFcc = Encoding.ASCII.GetString(converted, offset, 4);
            int origSize = BitConverter.ToInt32(original, offset + 4);
            int convSize = BitConverter.ToInt32(converted, offset + 4);
            
            if (origFcc != convFcc)
            {
                differences.Add(new ChunkDifference(
                    chunkIndex,
                    chunkIndex % 16,
                    chunkIndex / 16,
                    offset,
                    4,
                    $"FourCC mismatch: {origFcc} vs {convFcc}"));
            }
            
            if (origSize != convSize)
            {
                differences.Add(new ChunkDifference(
                    chunkIndex,
                    chunkIndex % 16,
                    chunkIndex / 16,
                    offset + 4,
                    4,
                    $"Size mismatch: {origSize} vs {convSize}"));
            }
            
            // Compare chunk data
            int chunkDataSize = Math.Min(origSize, convSize);
            int chunkDiffBytes = 0;
            
            for (int i = 0; i < chunkDataSize && offset + 8 + i < original.Length && offset + 8 + i < converted.Length; i++)
            {
                if (original[offset + 8 + i] != converted[offset + 8 + i])
                    chunkDiffBytes++;
            }
            
            if (chunkDiffBytes > 0)
            {
                // Parse MCNK header to get IndexX/IndexY
                int indexX = chunkIndex % 16;
                int indexY = chunkIndex / 16;
                
                if (origFcc == "KNCM" && offset + 8 + 12 <= original.Length) // MCNK chunk
                {
                    indexX = BitConverter.ToInt32(original, offset + 8 + 4);
                    indexY = BitConverter.ToInt32(original, offset + 8 + 8);
                }
                
                differences.Add(new ChunkDifference(
                    chunkIndex,
                    indexX,
                    indexY,
                    offset + 8,
                    chunkDiffBytes,
                    $"Chunk data differs: {chunkDiffBytes}/{chunkDataSize} bytes ({100.0 * chunkDiffBytes / chunkDataSize:F1}%)"));
            }
            
            // Move to next chunk
            int nextOffset = offset + 8 + origSize + ((origSize & 1) == 1 ? 1 : 0);
            if (nextOffset <= offset || nextOffset > original.Length) break;
            offset = nextOffset;
            chunkIndex++;
        }
        
        return new ValidationResult(
            false,
            $"Files differ in {totalDifferentBytes:N0} bytes across {differences.Count} chunks",
            TilesMismatched: differences.Count,
            BytesDifferent: totalDifferentBytes,
            Differences: differences);
    }
}
