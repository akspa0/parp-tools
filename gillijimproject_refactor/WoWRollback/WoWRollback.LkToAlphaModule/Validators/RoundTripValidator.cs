using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using WoWRollback.LkToAlphaModule.Builders;
using WoWRollback.LkToAlphaModule.Readers;
using WoWRollback.LkToAlphaModule.Services;
using WoWRollback.LkToAlphaModule.Models;
using GillijimProject.WowFiles.LichKing;

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
        LkToAlphaOptions? options = null,
        bool writeIntermediates = true)
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

            // Handle standalone ADT file
            return ValidateRoundTripStandaloneAdt(originalAlphaPath, outputDir, options, originalBytes, fileName, writeIntermediates);
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
        string fileName,
        bool writeIntermediates)
    {
        // Step 1: Extract Alpha data to LkAdtSource
        Console.WriteLine("[RoundTrip] Step 1: Extracting Alpha ADT data...");
        var lkSource = AlphaDataExtractor.ExtractFromAlphaAdt(originalAlphaAdtPath);
        Console.WriteLine($"[RoundTrip] Extracted {lkSource.Mcnks.Count} MCNK chunks");
        
        // Step 2: Build LK ADT using managed builders
        Console.WriteLine("[RoundTrip] Step 2: Building LK ADT...");
        var lkAdtBytes = LkAdtBuilder.Build(lkSource, options);

        // Validate LK integrity and write intermediate LK ADT
        string lkAdtPath = Path.Combine(outputDir, fileName.Replace(".adt", "_lk.adt"));
        try
        {
            var tempName = Path.GetFileName(lkAdtPath);
            var adt = new AdtLk(lkAdtBytes, tempName);
            bool ok = adt.ValidateIntegrity();
            Console.WriteLine(ok
                ? "[RoundTrip] LK integrity: OK (MHDR/MCIN offsets valid)"
                : "[RoundTrip] LK integrity: WARN (offsets invalid before write)");
            // Always emit intermediate LK ADT for inspection
            adt.ToFile(lkAdtPath);
            Console.WriteLine($"[RoundTrip] LK ADT saved: {lkAdtPath} ({new FileInfo(lkAdtPath).Length:N0} bytes)");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[RoundTrip] LK integrity check/write failed: {ex.Message}");
            // Fallback: write raw bytes if AdtLk path fails
            try
            {
                File.WriteAllBytes(lkAdtPath, lkAdtBytes);
                Console.WriteLine($"[RoundTrip] LK ADT (raw) saved: {lkAdtPath} ({lkAdtBytes.Length:N0} bytes)");
            }
            catch { /* give up on intermediate write */ }
        }
        
        // Step 3: Convert LK back to Alpha using AlphaMcnkBuilder
        Console.WriteLine("[RoundTrip] Step 3: Converting LK ADT back to Alpha...");
        var convertedAlphaBytes = ConvertLkToAlpha(lkAdtBytes, options);
        
        // Save converted Alpha ADT as the final output of the same type as input
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
    /// Validates round-trip for LK root ADT files: LK → Alpha → LK.
    /// </summary>
    public static ValidationResult ValidateRoundTripFromLkAdt(
        string lkAdtPath,
        string outputDir,
        LkToAlphaOptions? options = null)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(lkAdtPath) || !File.Exists(lkAdtPath))
            {
                return new ValidationResult(false, $"LK ADT file not found: {lkAdtPath}");
            }

            options ??= new LkToAlphaOptions();
            Directory.CreateDirectory(outputDir);

            var originalBytes = File.ReadAllBytes(lkAdtPath);
            var fileName = Path.GetFileName(lkAdtPath);

            Console.WriteLine($"[RoundTrip] Starting LK→Alpha→LK validation for: {fileName}");
            Console.WriteLine($"[RoundTrip] Original LK file size: {originalBytes.Length:N0} bytes");

            // Step 1: Convert LK ADT to Alpha using the managed builder
            Console.WriteLine("[RoundTrip] Step 1: Converting LK ADT to Alpha format...");
            var alphaBytes = ConvertLkToAlpha(originalBytes, options);
            var alphaOutPath = Path.Combine(outputDir, fileName.Replace(".adt", "_alpha_from_lk.adt"));
            File.WriteAllBytes(alphaOutPath, alphaBytes);
            Console.WriteLine($"[RoundTrip] Alpha ADT saved: {alphaOutPath} ({alphaBytes.Length:N0} bytes)");

            // Step 2: Extract Alpha data and rebuild LK ADT
            Console.WriteLine("[RoundTrip] Step 2: Extracting Alpha ADT for LK rebuild...");
            var alphaSource = AlphaDataExtractor.ExtractFromAlphaAdt(alphaOutPath);
            Console.WriteLine($"[RoundTrip] Extracted {alphaSource.Mcnks.Count} MCNK chunks from Alpha conversion");

            Console.WriteLine("[RoundTrip] Step 3: Building LK ADT from converted Alpha data...");
            var rebuiltLkBytes = LkAdtBuilder.Build(alphaSource, options);
            var rebuiltLkPath = Path.Combine(outputDir, fileName.Replace(".adt", "_roundtrip_lk.adt"));
            File.WriteAllBytes(rebuiltLkPath, rebuiltLkBytes);
            Console.WriteLine($"[RoundTrip] Rebuilt LK ADT saved: {rebuiltLkPath} ({rebuiltLkBytes.Length:N0} bytes)");

            // Step 4: Compare original LK with rebuilt LK
            Console.WriteLine("[RoundTrip] Step 4: Comparing original LK with round-trip LK result...");
            var comparison = CompareByteArrays(originalBytes, rebuiltLkBytes, fileName);

            if (comparison.Success)
            {
                Console.WriteLine("[RoundTrip] ✓ LK round-trip validation PASSED - Files are identical!");
            }
            else
            {
                Console.WriteLine($"[RoundTrip] ✗ LK round-trip validation FAILED - {comparison.ErrorMessage}");
            }

            return comparison with { TilesProcessed = alphaSource.Mcnks.Count };
        }
        catch (Exception ex)
        {
            return new ValidationResult(false, $"LK round-trip validation failed: {ex.Message}\n{ex.StackTrace}");
        }
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
        
        var extractionDir = Path.Combine(outputDir, "tiles_extracted");
        Directory.CreateDirectory(extractionDir);

        var convertedTilesDir = Path.Combine(outputDir, "tiles_converted_alpha");
        Directory.CreateDirectory(convertedTilesDir);

        var lkIntermediateDir = Path.Combine(outputDir, "tiles_converted_lk");
        Directory.CreateDirectory(lkIntermediateDir);

        var extractionResults = ExtractTiles(alphaWdt, originalBytes, extractionDir, options);

        if (extractionResults.SuccessCount == 0)
        {
            return new ValidationResult(
                false,
                extractionResults.ErrorMessage ?? "Failed to extract tiles from monolithic WDT",
                TilesProcessed: alphaWdt.Tiles.Count);
        }

        Console.WriteLine($"[RoundTrip] Step 3: Alpha → LK → Alpha conversion on {extractionResults.SuccessCount} tiles...");
        var roundTripTiles = ConvertTilesAlphaToLkToAlpha(extractionResults, lkIntermediateDir, convertedTilesDir, options);

        Console.WriteLine("[RoundTrip] Step 4: Packing converted tiles into monolithic WDT...");
        var repackedWdtPath = Path.Combine(outputDir, fileName.Replace(".wdt", "_roundtrip.wdt"));
        PackConvertedTilesIntoWdt(alphaWdt, originalWdtPath, repackedWdtPath, roundTripTiles, options);

        Console.WriteLine("[RoundTrip] Step 5: Comparing repacked WDT with original...");
        var repackedBytes = File.ReadAllBytes(repackedWdtPath);
        var overallComparison = CompareAlphaAdtsDetailed(originalBytes, repackedBytes, fileName);

        // Supplement overall result with per-tile stats
        var tileComparison = CompareTiles(roundTripTiles);

        if (overallComparison.Success && tileComparison.DifferentTiles == 0)
        {
            Console.WriteLine("[RoundTrip] ✓ Monolithic WDT round-trip succeeded with byte parity");
            return overallComparison with
            {
                TilesProcessed = alphaWdt.Tiles.Count,
                TilesMismatched = 0,
                BytesDifferent = 0
            };
        }

        var errorBuilder = new StringBuilder();
        if (!overallComparison.Success)
        {
            errorBuilder.AppendLine(overallComparison.ErrorMessage);
        }
        if (tileComparison.DifferentTiles > 0)
        {
            errorBuilder.AppendLine($"{tileComparison.DifferentTiles} tile(s) differ ({tileComparison.TotalDifferentBytes:N0} bytes)");
        }

        return new ValidationResult(
            false,
            errorBuilder.ToString().TrimEnd(),
            TilesProcessed: alphaWdt.Tiles.Count,
            TilesMismatched: tileComparison.DifferentTiles,
            BytesDifferent: tileComparison.TotalDifferentBytes,
            Differences: overallComparison.Differences
        );
    }
    
    /// <summary>
    /// Extracts a single tile's ADT data from a monolithic WDT.
    /// </summary>
    private static byte[] ExtractTileAdtFromWdt(byte[] wdtBytes, Models.AlphaTile tile)
    {
        int mhdrOffset = tile.MhdrOffset;
        if (mhdrOffset <= 0 || mhdrOffset + 8 > wdtBytes.Length)
        {
            throw new InvalidDataException($"Invalid MHDR offset for tile {tile.Index}: {mhdrOffset}");
        }

        int firstMcnkSize = tile.SizeToFirstMcnk > 0 ? tile.SizeToFirstMcnk : 0;
        int dataEnd = tile.DataEndOffset > mhdrOffset ? tile.DataEndOffset : wdtBytes.Length;

        // Ensure we stay within file bounds
        dataEnd = Math.Min(dataEnd, wdtBytes.Length);

        int safeSize = dataEnd - mhdrOffset;
        if (safeSize <= 0)
        {
            throw new InvalidDataException($"Unable to determine tile length for tile {tile.Index}");
        }

        var adtBytes = new byte[safeSize];
        Buffer.BlockCopy(wdtBytes, mhdrOffset, adtBytes, 0, safeSize);

        if (firstMcnkSize > 0 && firstMcnkSize < safeSize)
        {
            // Trim trailing data beyond first MCNK boundary if DataEndOffset overestimates
            int trimmedSize = firstMcnkSize + ComputeTrailingMcnkLength(adtBytes, firstMcnkSize);
            if (trimmedSize > 0 && trimmedSize <= safeSize)
            {
                Array.Resize(ref adtBytes, trimmedSize);
            }
        }

        return adtBytes;
    }

    private static int ComputeTrailingMcnkLength(byte[] adtBytes, int firstMcnkSize)
    {
        if (adtBytes.Length <= firstMcnkSize)
        {
            return 0;
        }

        int offset = firstMcnkSize;
        int totalLength = 0;
        while (offset + 8 <= adtBytes.Length)
        {
            string fcc = Encoding.ASCII.GetString(adtBytes, offset, 4);
            int size = BitConverter.ToInt32(adtBytes, offset + 4);
            int next = offset + 8 + size + ((size & 1) == 1 ? 1 : 0);
            if (next <= offset || next > adtBytes.Length)
            {
                break;
            }
            totalLength = next - firstMcnkSize;
            offset = next;
        }

        return totalLength;
    }

    private static TileExtractionResult ExtractTiles(AlphaWdt alphaWdt, byte[] wdtBytes, string extractionDir, LkToAlphaOptions options)
    {
        var successes = new List<ExtractedTile>();
        var failures = new List<(int TileIndex, string Error)>();

        foreach (var tile in alphaWdt.Tiles)
        {
            try
            {
                var tileBytes = ExtractTileAdtFromWdt(wdtBytes, tile);
                var extractedPath = Path.Combine(extractionDir, $"tile_{tile.Index:D4}_extracted.adt");
                File.WriteAllBytes(extractedPath, tileBytes);

                successes.Add(new ExtractedTile(tile.Index, tile, extractedPath, tileBytes.Length));
            }
            catch (Exception ex)
            {
                failures.Add((tile.Index, ex.Message));
                Console.WriteLine($"[RoundTrip] Warning: Failed to extract tile {tile.Index}: {ex.Message}");
            }
        }

        var message = failures.Count > 0
            ? $"Extracted {successes.Count} tile(s); {failures.Count} failed (first error: {failures[0].Error})"
            : null;

        return new TileExtractionResult(successes, failures, successes.Count, message);
    }

    private static RoundTripTiles ConvertTilesAlphaToLkToAlpha(
        TileExtractionResult extraction,
        string lkIntermediateDir,
        string convertedTilesDir,
        LkToAlphaOptions options)
    {
        var successes = new List<RoundTripTile>();
        var failures = new List<(int TileIndex, string Error)>();
        int processed = 0;

        foreach (var extracted in extraction.Successes)
        {
            try
            {
                var lkSource = AlphaDataExtractor.ExtractFromAlphaAdt(extracted.ExtractedAlphaPath);
                var lkBytes = LkAdtBuilder.Build(lkSource, options);

                int tileIndex = extracted.Tile.Index;
                var lkPath = Path.Combine(lkIntermediateDir, $"tile_{tileIndex:D4}_lk.adt");
                File.WriteAllBytes(lkPath, lkBytes);

                var alphaBytes = ConvertLkToAlpha(lkBytes, options);
                var convertedPath = Path.Combine(convertedTilesDir, $"tile_{tileIndex:D4}_roundtrip.adt");
                File.WriteAllBytes(convertedPath, alphaBytes);

                successes.Add(new RoundTripTile(extracted.Tile, extracted.ExtractedAlphaPath, lkPath, convertedPath, extracted.OriginalLength, alphaBytes.Length));

                processed++;
                if (processed % 16 == 0)
                {
                    Console.WriteLine($"[RoundTrip] Processed {processed}/{extraction.SuccessCount} tiles...");
                }
            }
            catch (Exception ex)
            {
                failures.Add((extracted.Tile.Index, ex.Message));
                Console.WriteLine($"[RoundTrip] Warning: Failed round-trip for tile {extracted.Tile.Index}: {ex.Message}");
            }
        }

        return new RoundTripTiles(successes, failures);
    }

    private static void PackConvertedTilesIntoWdt(
        AlphaWdt alphaWdt,
        string originalWdtPath,
        string repackedWdtPath,
        RoundTripTiles roundTripTiles,
        LkToAlphaOptions options)
    {
        var originalBytes = File.ReadAllBytes(originalWdtPath);
        var outputBytes = new byte[originalBytes.Length];
        Buffer.BlockCopy(originalBytes, 0, outputBytes, 0, originalBytes.Length);

        foreach (var tile in roundTripTiles.Successes)
        {
            var convertedBytes = File.ReadAllBytes(tile.ConvertedAlphaPath);
            var originalTileBytes = File.ReadAllBytes(tile.ExtractedAlphaPath);

            // Replace only the MCNK region within the tile. Our converted bytes are MCNK-concatenation (no MHDR).
            int prefixLen = Math.Max(0, tile.Tile.SizeToFirstMcnk);
            int destStart = tile.Tile.MhdrOffset + prefixLen;
            int originalMcnkLen = Math.Max(0, originalTileBytes.Length - prefixLen);
            int copyLen = Math.Min(convertedBytes.Length, originalMcnkLen);

            if (destStart < 0 || destStart > outputBytes.Length)
            {
                Console.WriteLine($"[RoundTrip] Skip tile {tile.Tile.Index}: destStart out of bounds ({destStart})");
                continue;
            }

            if (destStart + copyLen > outputBytes.Length)
            {
                copyLen = Math.Max(0, outputBytes.Length - destStart);
            }

            if (copyLen <= 0)
            {
                Console.WriteLine($"[RoundTrip] Skip tile {tile.Tile.Index}: nothing to copy (converted={convertedBytes.Length}, originalMcnk={originalMcnkLen})");
                continue;
            }

            if (convertedBytes.Length != originalMcnkLen)
            {
                Console.WriteLine($"[RoundTrip] Note: tile {tile.Tile.Index} MCNK length differs (orig={originalMcnkLen} conv={convertedBytes.Length}); clamping to {copyLen}");
            }

            Buffer.BlockCopy(convertedBytes, 0, outputBytes, destStart, copyLen);
        }

        File.WriteAllBytes(repackedWdtPath, outputBytes);
    }

    private static TileComparisonResult CompareTiles(RoundTripTiles roundTripTiles)
    {
        int differentTiles = 0;
        long totalDifferentBytes = 0;

        foreach (var tile in roundTripTiles.Successes)
        {
            var originalBytes = File.ReadAllBytes(tile.ExtractedAlphaPath);
            var convertedBytes = File.ReadAllBytes(tile.ConvertedAlphaPath);

            if (originalBytes.Length != convertedBytes.Length)
            {
                differentTiles++;
                totalDifferentBytes += Math.Abs(originalBytes.Length - convertedBytes.Length);
                continue;
            }

            long diff = 0;
            for (int i = 0; i < originalBytes.Length; i++)
            {
                if (originalBytes[i] != convertedBytes[i])
                {
                    diff++;
                }
            }

            if (diff > 0)
            {
                differentTiles++;
                totalDifferentBytes += diff;
            }
        }

        return new TileComparisonResult(differentTiles, totalDifferentBytes);
    }

    private sealed record class TileExtractionResult(
        IReadOnlyList<ExtractedTile> Successes,
        IReadOnlyList<(int TileIndex, string Error)> Failures,
        int SuccessCount,
        string? ErrorMessage);

    private sealed record class ExtractedTile(
        int TileIndex,
        AlphaTile Tile,
        string ExtractedAlphaPath,
        int OriginalLength);

    private sealed record class RoundTripTiles(
        IReadOnlyList<RoundTripTile> Successes,
        IReadOnlyList<(int TileIndex, string Error)> Failures);

    private sealed record class RoundTripTile(
        AlphaTile Tile,
        string ExtractedAlphaPath,
        string LkAdtPath,
        string ConvertedAlphaPath,
        int OriginalLength,
        int ConvertedLength);

    private sealed record class TileComparisonResult(int DifferentTiles, long TotalDifferentBytes);
    
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
        
        Console.WriteLine($"[DEBUG] MHDR at {mhdrOffset}, data start {mhdrDataStart}, MCIN offset (relative) {mcinOffset}");
        
        if (mcinOffset > 0)
        {
            int mcinPos = mhdrDataStart + mcinOffset;
            if (mcinPos + 8 <= lkAdtBytes.Length)
            {
                string mcinFcc = Encoding.ASCII.GetString(lkAdtBytes, mcinPos, 4);
                Console.WriteLine($"[DEBUG] MCIN FourCC: '{mcinFcc}' at position {mcinPos}");
                if (mcinFcc == "NICM") // MCIN reversed
                {
                    int mcinSize = BitConverter.ToInt32(lkAdtBytes, mcinPos + 4);
                    int mcinDataStart = mcinPos + 8;
                    
                    Console.WriteLine($"[DEBUG] MCIN size: {mcinSize}, data start: {mcinDataStart}");
                    
                    // MCIN contains 256 entries of 16 bytes each (offset, size, flags, asyncId)
                    // Offsets are relative to MHDR data start, so convert to absolute
                    int validOffsets = 0;
                    for (int i = 0; i < 256 && i * 16 + 4 <= mcinSize; i++)
                    {
                        int mcnkRelativeOffset = BitConverter.ToInt32(lkAdtBytes, mcinDataStart + i * 16);
                        if (mcnkRelativeOffset > 0)
                        {
                            int mcnkAbsoluteOffset = mhdrDataStart + mcnkRelativeOffset;
                            if (mcnkAbsoluteOffset < lkAdtBytes.Length)
                            {
                                offsets.Add(mcnkAbsoluteOffset);
                                validOffsets++;
                            }
                        }
                    }
                    Console.WriteLine($"[DEBUG] Found {validOffsets} valid MCNK offsets in MCIN");
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

    private static ValidationResult CompareByteArrays(byte[] original, byte[] rebuilt, string fileName)
    {
        if (original.Length != rebuilt.Length)
        {
            return new ValidationResult(false, $"LK round-trip size mismatch: {original.Length:N0} vs {rebuilt.Length:N0}");
        }

        long diff = 0;
        for (int i = 0; i < original.Length; i++)
        {
            if (original[i] != rebuilt[i])
            {
                diff++;
            }
        }

        if (diff == 0)
        {
            return new ValidationResult(true, "LK files are byte-for-byte identical");
        }

        return new ValidationResult(false, $"LK round-trip differs in {diff:N0} bytes");
    }
}
