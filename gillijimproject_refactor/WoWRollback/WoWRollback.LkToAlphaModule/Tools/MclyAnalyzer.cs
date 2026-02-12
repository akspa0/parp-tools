using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace WoWRollback.LkToAlphaModule.Tools;

/// <summary>
/// Analyzes MCLY layer tables from Alpha WDT files to diagnose texture layer issues.
/// </summary>
public static class MclyAnalyzer
{
    private const int ChunkHeader = 8;
    private const int McnkHeaderSize = 128;

    public static void AnalyzeTile(string wdtPath, int tileIndex)
    {
        using var fs = File.OpenRead(wdtPath);
        
        // Find MAIN chunk
        long mainOff = FindChunk(fs, "NIAM"); // MAIN reversed
        if (mainOff < 0)
        {
            Console.WriteLine("MAIN chunk not found");
            return;
        }

        // Read tile entry from MAIN
        long tileEntry = mainOff + ChunkHeader + (tileIndex * 16);
        if (tileEntry + 16 > fs.Length)
        {
            Console.WriteLine($"Tile {tileIndex} out of bounds");
            return;
        }

        fs.Seek(tileEntry, SeekOrigin.Begin);
        byte[] entry = new byte[16];
        if (fs.Read(entry, 0, 16) != 16) return;

        int mhdrOff = BitConverter.ToInt32(entry, 0);
        int mhdrSize = BitConverter.ToInt32(entry, 4);

        if (mhdrOff <= 0 || mhdrSize <= 0)
        {
            Console.WriteLine($"Tile {tileIndex} is empty");
            return;
        }

        Console.WriteLine($"\n=== TILE {tileIndex} ===");
        Console.WriteLine($"MHDR offset: 0x{mhdrOff:X}, size: {mhdrSize}");

        // Read MHDR
        fs.Seek(mhdrOff + ChunkHeader, SeekOrigin.Begin);
        byte[] mhdr = new byte[McnkHeaderSize];
        if (fs.Read(mhdr, 0, McnkHeaderSize) != McnkHeaderSize) return;

        int offsInfo = BitConverter.ToInt32(mhdr, 0);
        int offsTex = BitConverter.ToInt32(mhdr, 4);
        int offsDoo = BitConverter.ToInt32(mhdr, 12);
        int offsMob = BitConverter.ToInt32(mhdr, 20);

        // MCIN is at offsInfo
        long mcinOff = mhdrOff + ChunkHeader + offsInfo;
        Console.WriteLine($"MCIN offset: 0x{mcinOff:X}");

        // MTEX is at offsTex
        long mtexOff = mhdrOff + ChunkHeader + offsTex;
        Console.WriteLine($"MTEX offset: 0x{mtexOff:X}");

        // First MCNK
        long firstMcnkOff = mhdrOff + mhdrSize;
        Console.WriteLine($"First MCNK offset: 0x{firstMcnkOff:X}");

        // Read first MCNK header
        fs.Seek(firstMcnkOff + ChunkHeader, SeekOrigin.Begin);
        byte[] mcnkHdr = new byte[McnkHeaderSize];
        if (fs.Read(mcnkHdr, 0, McnkHeaderSize) != McnkHeaderSize) return;

        uint nLayers = BitConverter.ToUInt32(mcnkHdr, 0x10);
        int mclyOffset = BitConverter.ToInt32(mcnkHdr, 0x20);
        int mcalOffset = BitConverter.ToInt32(mcnkHdr, 0x28);
        int mcalSize = BitConverter.ToInt32(mcnkHdr, 0x2C);

        Console.WriteLine($"\nMCNK[0] Header:");
        Console.WriteLine($"  NLayers: {nLayers}");
        Console.WriteLine($"  MclyOffset: 0x{mclyOffset:X}");
        Console.WriteLine($"  McalOffset: 0x{mcalOffset:X}");
        Console.WriteLine($"  McalSize: {mcalSize}");

        // Read MCLY entries
        long mclyAbs = firstMcnkOff + ChunkHeader + mclyOffset;
        Console.WriteLine($"\nMCLY at 0x{mclyAbs:X}:");

        byte[] layer = new byte[16];
        for (int i = 0; i < nLayers && i < 4; i++)
        {
            fs.Seek(mclyAbs + (i * 16), SeekOrigin.Begin);
            if (fs.Read(layer, 0, 16) != 16) break;

            uint textureId = BitConverter.ToUInt32(layer, 0);
            uint props = BitConverter.ToUInt32(layer, 4);
            uint offsAlpha = BitConverter.ToUInt32(layer, 8);
            ushort effectId = BitConverter.ToUInt16(layer, 12);

            Console.WriteLine($"  Layer {i}:");
            Console.WriteLine($"    Hex: {string.Join(" ", layer.Select(b => b.ToString("X2")))}");
            Console.WriteLine($"    TextureId: {textureId} (0x{textureId:X})");
            Console.WriteLine($"    Props: 0x{props:X8}");
            Console.WriteLine($"    OffsAlpha: 0x{offsAlpha:X}");
            Console.WriteLine($"    EffectId: {effectId}");
        }
    }

    private static long FindChunk(FileStream fs, string reversedToken)
    {
        fs.Seek(0, SeekOrigin.Begin);
        byte[] buf = new byte[4];
        long pos = 0;

        while (pos + 8 <= fs.Length)
        {
            fs.Seek(pos, SeekOrigin.Begin);
            if (fs.Read(buf, 0, 4) != 4) break;

            string tok = Encoding.ASCII.GetString(buf);
            if (tok == reversedToken)
                return pos;

            fs.Seek(pos + 4, SeekOrigin.Begin);
            byte[] sizeBuf = new byte[4];
            if (fs.Read(sizeBuf, 0, 4) != 4) break;

            int size = BitConverter.ToInt32(sizeBuf);
            int pad = (size & 1) == 1 ? 1 : 0;
            pos += 8 + size + pad;
        }

        return -1;
    }
}
