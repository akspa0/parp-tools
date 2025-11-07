using System;
using System.IO;
using System.Linq;
using System.Text;

namespace WoWRollback.LkToAlphaModule.Writers;

public static class AlphaWdlWriter
{
    // Minimal WDL (v18-style) with MAOF + per-tile MARE + MAHO
    // - MARE: 545 int16 heights (flat 0)
    // - MAHO: 16 uint16 masks (zeros)
    // Offsets in MAOF point to the start of each tile's MARE chunk (letters)
    public static void WriteMinimalWdl(string outWdlPath, bool[] tilesPresent)
    {
        if (tilesPresent == null || tilesPresent.Length != 64 * 64)
            throw new ArgumentException("tilesPresent must be length 4096");
        Directory.CreateDirectory(Path.GetDirectoryName(outWdlPath) ?? ".");

        // Precompute sizes
        int maofDataSize = 4096 * 4; // UINT32[4096]
        int maofWhole = 8 + maofDataSize; // MAOF header + data, even size
        int mareDataSize = 545 * 2; // int16 * 545
        int mareWhole = 8 + mareDataSize; // even (1098)
        int mahoDataSize = 16 * 2; // uint16 * 16
        int mahoWhole = 8 + mahoDataSize; // 40
        int perTileWhole = mareWhole + mahoWhole; // 1138

        // Calculate offsets
        var offsets = new int[4096];
        int cursor = maofWhole; // first tile block begins after MAOF
        for (int idx = 0; idx < 4096; idx++)
        {
            if (tilesPresent[idx])
            {
                offsets[idx] = cursor; // absolute offset to MARE letters
                cursor += perTileWhole;
            }
            else
            {
                offsets[idx] = 0;
            }
        }

        using var ms = new MemoryStream();
        // MAOF
        var maofData = new byte[maofDataSize];
        for (int i = 0; i < 4096; i++)
        {
            Buffer.BlockCopy(BitConverter.GetBytes(offsets[i]), 0, maofData, i * 4, 4);
        }
        WriteChunk(ms, "MAOF", maofData);

        // Per-tile blocks
        // Prepare flat MARE and empty MAHO once and reuse
        var mare = new byte[mareDataSize]; // all zeros
        var maho = new byte[mahoDataSize]; // all zeros

        for (int idx = 0; idx < 4096; idx++)
        {
            if (!tilesPresent[idx]) continue;
            WriteChunk(ms, "MARE", mare);
            WriteChunk(ms, "MAHO", maho);
        }

        using var fs = File.Create(outWdlPath);
        ms.Position = 0;
        ms.WriteTo(fs);
    }

    private static void WriteChunk(Stream s, string fourcc, byte[] data)
    {
        var letters = Encoding.ASCII.GetBytes(fourcc);
        s.Write(letters, 0, 4);
        s.Write(BitConverter.GetBytes(data?.Length ?? 0), 0, 4);
        if (data != null && data.Length > 0) s.Write(data, 0, data.Length);
        if (((data?.Length ?? 0) & 1) == 1) s.WriteByte(0); // even-size pad
    }
}
