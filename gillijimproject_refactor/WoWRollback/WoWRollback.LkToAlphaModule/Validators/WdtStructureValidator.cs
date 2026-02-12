using System;
using System.Collections.Generic;
using System.IO;

namespace WoWRollback.LkToAlphaModule.Validators;

/// <summary>
/// Deep structure validation - compares WDT against known-good reference
/// </summary>
public static class WdtStructureValidator
{
    public class ValidationResult
    {
        public bool IsValid { get; set; }
        public List<string> Errors { get; set; } = new();
        public List<string> Warnings { get; set; } = new();
    }

    public static ValidationResult ValidateAgainstReference(string testWdt, string referenceWdt)
    {
        var result = new ValidationResult { IsValid = true };

        using var testFs = File.OpenRead(testWdt);
        using var refFs = File.OpenRead(referenceWdt);

        // Compare MAIN chunk structure
        ValidateMainChunk(testFs, refFs, result);

        // Validate first tile structure in detail
        ValidateFirstTile(testFs, refFs, result);

        result.IsValid = result.Errors.Count == 0;
        return result;
    }

    private static void ValidateMainChunk(FileStream testFs, FileStream refFs, ValidationResult result)
    {
        // Read MAIN from both files
        testFs.Seek(0x9C, SeekOrigin.Begin); // MAIN data start
        refFs.Seek(0x9C, SeekOrigin.Begin);

        var testMain = new byte[65536];
        var refMain = new byte[65536];
        testFs.Read(testMain, 0, 65536);
        refFs.Read(refMain, 0, 65536);

        // Count non-zero tiles
        int testTiles = 0, refTiles = 0;
        for (int i = 0; i < 4096; i++)
        {
            int testOff = BitConverter.ToInt32(testMain, i * 16);
            int refOff = BitConverter.ToInt32(refMain, i * 16);
            if (testOff != 0) testTiles++;
            if (refOff != 0) refTiles++;
        }

        Console.WriteLine($"[MAIN] Test tiles: {testTiles}, Reference tiles: {refTiles}");

        if (testTiles == 0)
        {
            result.Errors.Add("MAIN chunk has zero tiles - data not patched!");
        }
    }

    private static void ValidateFirstTile(FileStream testFs, FileStream refFs, ValidationResult result)
    {
        // Find first non-zero tile in MAIN
        testFs.Seek(0x9C, SeekOrigin.Begin);
        refFs.Seek(0x9C, SeekOrigin.Begin);

        var testMainData = new byte[65536];
        testFs.Read(testMainData, 0, 65536);

        int firstTileIdx = -1;
        int testTileOffset = 0;
        for (int i = 0; i < 4096; i++)
        {
            int offset = BitConverter.ToInt32(testMainData, i * 16);
            if (offset > 0)
            {
                firstTileIdx = i;
                testTileOffset = offset;
                break;
            }
        }

        if (firstTileIdx < 0)
        {
            result.Errors.Add("No tiles found in MAIN");
            return;
        }

        Console.WriteLine($"[VALIDATE] Checking tile {firstTileIdx} at offset 0x{testTileOffset:X}");

        // Read MHDR from test file
        testFs.Seek(testTileOffset, SeekOrigin.Begin);
        var mhdrHeader = new byte[8];
        testFs.Read(mhdrHeader, 0, 8);

        string fourCC = System.Text.Encoding.ASCII.GetString(mhdrHeader, 0, 4);
        if (fourCC != "RDHM") // "MHDR" reversed
        {
            result.Errors.Add($"Tile {firstTileIdx}: Expected MHDR, got '{fourCC}'");
        }

        // Read MHDR data
        var mhdrData = new byte[64];
        testFs.Read(mhdrData, 0, 64);

        int offsInfo = BitConverter.ToInt32(mhdrData, 0);
        int offsTex = BitConverter.ToInt32(mhdrData, 4);
        int offsDoo = BitConverter.ToInt32(mhdrData, 12);
        int offsMob = BitConverter.ToInt32(mhdrData, 20);

        Console.WriteLine($"  MHDR offsets: offsInfo={offsInfo}, offsTex={offsTex}, offsDoo={offsDoo}, offsMob={offsMob}");

        // Validate offsets point to correct chunks
        long mhdrDataStart = testTileOffset + 8;
        ValidateChunkAtOffset(testFs, mhdrDataStart + offsTex, "MTEX", result);
        ValidateChunkAtOffset(testFs, mhdrDataStart + offsDoo, "MDDF", result);
        ValidateChunkAtOffset(testFs, mhdrDataStart + offsMob, "MODF", result);
    }

    private static void ValidateChunkAtOffset(FileStream fs, long offset, string expectedChunk, ValidationResult result)
    {
        fs.Seek(offset, SeekOrigin.Begin);
        var header = new byte[4];
        fs.Read(header, 0, 4);

        string fourCC = System.Text.Encoding.ASCII.GetString(header);
        string reversed = new string(new[] { fourCC[3], fourCC[2], fourCC[1], fourCC[0] });

        if (reversed != expectedChunk)
        {
            result.Errors.Add($"Offset 0x{offset:X}: Expected {expectedChunk}, got {reversed}");
        }
        else
        {
            Console.WriteLine($"  ✓ {expectedChunk} at 0x{offset:X}");
        }
    }
}
