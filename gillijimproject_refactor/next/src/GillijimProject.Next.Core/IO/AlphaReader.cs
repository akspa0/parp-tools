using GillijimProject.Next.Core.Domain;

namespace GillijimProject.Next.Core.IO;

/// <summary>
/// Readers for Alpha-era WDT/ADT inputs.
/// </summary>
public static class AlphaReader
{
    /// <summary>
    /// Parses an Alpha WDT file into a minimal model.
    /// </summary>
    public static WdtAlpha ParseWdt(string alphaWdtPath)
    {
        // TODO: Implement real parsing based on existing ported readers.
        return new WdtAlpha(alphaWdtPath);
    }

    /// <summary>
    /// Parses an Alpha ADT file into a minimal model.
    /// </summary>
    public static AdtAlpha ParseAdt(string alphaAdtPath)
    {
        // TODO: Implement real parsing based on existing ported readers.
        return new AdtAlpha(alphaAdtPath);
    }

    /// <summary>
    /// Parses an Alpha WDL file (low-resolution horizon mesh) into a 64x64 grid of tiles.
    /// Uses MAOF to locate MARE tiles and reads their 17x17 and 16x16 int16 height grids.
    /// Reads optional MAHO holes masks (16 x ushort rows) when present.
    /// Unknown chunks are tolerated and skipped.
    /// </summary>
    public static Wdl ParseWdl(string wdlPath)
    {
        if (string.IsNullOrWhiteSpace(wdlPath) || !System.IO.File.Exists(wdlPath))
            throw new System.IO.FileNotFoundException("WDL path not found", wdlPath);

        var tiles = new WdlTile?[64, 64];

        using var fs = System.IO.File.OpenRead(wdlPath);
        using var br = new System.IO.BinaryReader(fs, System.Text.Encoding.ASCII, leaveOpen: true);

        long len = fs.Length;
        bool done = false;
        // MVER is optional; we read and discard the version value when present.
        while (!done && fs.Position + 8 <= len)
        {
            long headerPos = fs.Position;
            string fourcc = ReadFourCC(br);
            uint size = br.ReadUInt32();
            long dataStart = fs.Position;

            // Handle MVER (version=18) when present
            if (Matches(fourcc, "MVER"))
            {
                if (size >= 4 && dataStart + size <= len)
                {
                    _ = br.ReadInt32();
                    // skip any remaining bytes in MVER (if size > 4)
                    fs.Position = dataStart + size;
                }
                else
                {
                    // corrupt MVER size; skip conservatively
                    fs.Position = dataStart + size;
                }
                // pad if odd
                if ((size & 1) == 1) fs.Position++;
                continue;
            }

            if (Matches(fourcc, "MAOF"))
            {
                // MAOF: 64x64 uint32 offsets to MARE tiles (absolute file offsets)
                int totalCells = 64 * 64;
                int availableCells = (int)System.Math.Min((uint)totalCells, size / 4);

                var offsets = new uint[64, 64];
                for (int i = 0; i < availableCells; i++)
                {
                    int y = i / 64;
                    int x = i % 64;
                    offsets[y, x] = br.ReadUInt32();
                }
                // skip any remaining MAOF bytes if present
                long remain = (long)size - (availableCells * 4);
                if (remain > 0) fs.Seek(remain, System.IO.SeekOrigin.Current);

                // Read MARE tiles
                for (int y = 0; y < 64; y++)
                {
                    for (int x = 0; x < 64; x++)
                    {
                        uint off = offsets[y, x];
                        if (off == 0) continue;
                        if (((ulong)off) + 8UL > (ulong)len) continue;

                        long ret = fs.Position;
                        fs.Position = off;
                        string innerFour = ReadFourCC(br);
                        uint innerSize = br.ReadUInt32();
                        if (!Matches(innerFour, "MARE"))
                        {
                            fs.Position = ret;
                            continue;
                        }

                        // Expected payload size: 17*17*2 + 16*16*2 = 1090 (0x442)
                        int expected = (WdlTile.OuterGrid * WdlTile.OuterGrid + WdlTile.InnerGrid * WdlTile.InnerGrid) * 2;
                        // Require enough bytes for expected content, but tolerate larger (skip trailing)
                        if (innerSize < expected) { fs.Position = ret; continue; }
                        if (((ulong)off) + 8UL + (ulong)innerSize > (ulong)len) { fs.Position = ret; continue; }

                        long mareDataStart = fs.Position;
                        long mareDataEnd = mareDataStart + innerSize;

                        var height17 = new short[WdlTile.OuterGrid, WdlTile.OuterGrid];
                        for (int j = 0; j < WdlTile.OuterGrid; j++)
                        {
                            for (int i = 0; i < WdlTile.OuterGrid; i++)
                            {
                                if (fs.Position + 2 > mareDataEnd) { fs.Position = ret; goto NextTile; }
                                height17[j, i] = br.ReadInt16();
                            }
                        }

                        var height16 = new short[WdlTile.InnerGrid, WdlTile.InnerGrid];
                        for (int j = 0; j < WdlTile.InnerGrid; j++)
                        {
                            for (int i = 0; i < WdlTile.InnerGrid; i++)
                            {
                                if (fs.Position + 2 > mareDataEnd) { fs.Position = ret; goto NextTile; }
                                height16[j, i] = br.ReadInt16();
                            }
                        }

                        // Default holes mask (all zeros)
                        var holeMask16 = new ushort[WdlTile.InnerGrid];

                        // Advance to end of MARE payload and apply padding if needed
                        fs.Position = mareDataEnd;
                        if ((innerSize & 1) == 1) fs.Position++;

                        // Try to read a following MAHO chunk (or reversed OHAM)
                        if (fs.Position + 8 <= len)
                        {
                            long mahoHeaderPos = fs.Position;
                            string mahoFour = ReadFourCC(br);
                            uint mahoSize = br.ReadUInt32();
                            if (Matches(mahoFour, "MAHO") && fs.Position + mahoSize <= len)
                            {
                                long mahoDataStart = fs.Position;
                                int rows = (int)System.Math.Min((uint)WdlTile.InnerGrid, mahoSize / 2);
                                for (int r = 0; r < rows; r++)
                                {
                                    holeMask16[r] = br.ReadUInt16();
                                }
                                // Skip any remaining MAHO payload bytes
                                long mahoDataEnd = mahoDataStart + mahoSize;
                                fs.Position = mahoDataEnd;
                                // Pad if MAHO size is odd
                                if ((mahoSize & 1) == 1) fs.Position++;
                            }
                            else
                            {
                                // Not a MAHO chunk; rewind to header position to avoid consuming bytes unnecessarily
                                fs.Position = mahoHeaderPos;
                            }
                        }

                        tiles[y, x] = new WdlTile(height17, height16, holeMask16);

                    NextTile:
                        // Return to where we were scanning MAOF content
                        fs.Position = ret;
                    }
                }

                done = true; // stop after processing MAOF like Noggit
                break;
            }
            else
            {
                // Skip unknown chunks: MWMO, MWID, MODF, MARE (top-level), etc.
                fs.Position = dataStart + size + ((size & 1) == 1 ? 1 : 0);
            }
        }

        // [PORT] We accept files without MVER; if present and not 18, consumer may decide how to handle.
        // TODO: hook a logger here for version mismatch diagnostics if needed.

        return new Wdl(wdlPath, tiles);
    }

    private static string ReadFourCC(System.IO.BinaryReader br)
    {
        var b = br.ReadBytes(4);
        return System.Text.Encoding.ASCII.GetString(b);
    }

    private static bool Matches(string fourCC, string expected)
    {
        if (fourCC.Equals(expected, System.StringComparison.OrdinalIgnoreCase)) return true;
        var rev = new string(new[] { expected[3], expected[2], expected[1], expected[0] });
        return fourCC.Equals(rev, System.StringComparison.OrdinalIgnoreCase);
    }
}
