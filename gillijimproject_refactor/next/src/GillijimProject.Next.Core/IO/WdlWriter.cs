using System;
using System.IO;
using System.Text;
using GillijimProject.Next.Core.Domain;

namespace GillijimProject.Next.Core.IO;

/// <summary>
/// Minimal writer for WDL (v18-compatible) files.
/// Writes MVER (optional), MAOF (4096 offsets), then per-tile MARE + MAHO.
/// Offsets in MAOF are absolute positions of each tile's MARE chunk header.
/// </summary>
public static class WdlWriter
{
    public static void Write(Wdl model, string outputPath, bool includeMver = true)
    {
        if (model is null) throw new ArgumentNullException(nameof(model));
        if (string.IsNullOrWhiteSpace(outputPath)) throw new ArgumentException("Output path required", nameof(outputPath));

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputPath)) ?? ".");

        using var fs = new FileStream(outputPath, FileMode.Create, FileAccess.ReadWrite, FileShare.None);
        using var bw = new BinaryWriter(fs, Encoding.ASCII, leaveOpen: true);

        // Optional MVER (18)
        if (includeMver)
        {
            WriteFourCC(bw, "MVER");
            bw.Write((uint)4);
            bw.Write(18);
        }

        // Reserve MAOF payload (4096 * 4 bytes) and remember its payload start
        WriteFourCC(bw, "MAOF");
        uint maofSize = 64u * 64u * 4u;
        bw.Write(maofSize);
        long maofPayloadPos = fs.Position;
        // Write zeros as placeholders
        Span<byte> zeroBlock = stackalloc byte[4096];
        long remaining = maofSize;
        while (remaining > 0)
        {
            int chunk = (int)Math.Min(remaining, zeroBlock.Length);
            bw.Write(zeroBlock[..chunk]);
            remaining -= chunk;
        }

        var offsets = new uint[64, 64];

        // Write per-tile blocks and collect offsets
        for (int y = 0; y < 64; y++)
        {
            for (int x = 0; x < 64; x++)
            {
                var tile = model.Tiles[y, x];
                if (tile is null) { offsets[y, x] = 0; continue; }

                long marePos = fs.Position;
                // Ensure offset fits into uint
                if ((ulong)marePos > uint.MaxValue) throw new IOException("WDL file too large; MAOF offset overflow.");
                offsets[y, x] = (uint)marePos;

                // MARE
                WriteFourCC(bw, "MARE");
                // Expected payload size: 17*17*2 + 16*16*2 = 1090
                const int expectedHeights = (WdlTile.OuterGrid * WdlTile.OuterGrid) + (WdlTile.InnerGrid * WdlTile.InnerGrid);
                const int mareSize = expectedHeights * 2;
                bw.Write((uint)mareSize);
                // Write 17x17 outer
                for (int j = 0; j < WdlTile.OuterGrid; j++)
                    for (int i = 0; i < WdlTile.OuterGrid; i++)
                        bw.Write(tile.Height17[j, i]);
                // Write 16x16 inner
                for (int j = 0; j < WdlTile.InnerGrid; j++)
                    for (int i = 0; i < WdlTile.InnerGrid; i++)
                        bw.Write(tile.Height16[j, i]);
                if (IsOdd(mareSize)) bw.Write((byte)0); // pad if odd

                // MAHO (always include even if all zeros)
                WriteFourCC(bw, "MAHO");
                const int mahoSize = WdlTile.InnerGrid * 2; // 16 * 2 = 32 bytes
                bw.Write((uint)mahoSize);
                var rows = tile.HoleMask16;
                if (rows.Length != WdlTile.InnerGrid)
                {
                    // Write zeros if the tile mask length is unexpected
                    for (int r = 0; r < WdlTile.InnerGrid; r++) bw.Write((ushort)0);
                }
                else
                {
                    for (int r = 0; r < WdlTile.InnerGrid; r++) bw.Write(rows[r]);
                }
                if (IsOdd(mahoSize)) bw.Write((byte)0); // pad if odd (normally even)
            }
        }

        // Backfill MAOF payload with collected offsets (row-major y,x)
        fs.Position = maofPayloadPos;
        for (int i = 0; i < 64 * 64; i++)
        {
            int y = i / 64;
            int x = i % 64;
            bw.Write(offsets[y, x]);
        }
    }

    private static void WriteFourCC(BinaryWriter bw, string tag)
    {
        if (tag.Length != 4) throw new ArgumentException("FourCC must be 4 chars", nameof(tag));
        bw.Write(Encoding.ASCII.GetBytes(tag));
    }

    private static bool IsOdd(int value) => (value & 1) == 1;
}
