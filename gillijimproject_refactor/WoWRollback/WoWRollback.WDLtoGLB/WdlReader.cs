using System;
using System.IO;
using System.Text;

namespace WoWRollback.WDLtoGLB;

/// <summary>
/// Alpha-era WDL reader. Parses MAOF→MARE tiles and optional MAHO (holes).
/// Tolerant to reversed FourCC and extra padding.
/// </summary>
public static class WdlReader
{
    public static Wdl Parse(string wdlPath)
    {
        if (string.IsNullOrWhiteSpace(wdlPath) || !File.Exists(wdlPath))
            throw new FileNotFoundException("WDL path not found", wdlPath);

        var tiles = new WdlTile?[64, 64];

        using var fs = File.OpenRead(wdlPath);
        using var br = new BinaryReader(fs, Encoding.ASCII, leaveOpen: true);

        long len = fs.Length;
        bool done = false;
        while (!done && fs.Position + 8 <= len)
        {
            long headerPos = fs.Position;
            string fourcc = ReadFourCC(br);
            uint size = br.ReadUInt32();
            long dataStart = fs.Position;

            if (Matches(fourcc, "MVER"))
            {
                if (size >= 4 && dataStart + size <= len)
                {
                    _ = br.ReadInt32();
                    fs.Position = dataStart + size;
                }
                else
                {
                    fs.Position = dataStart + size;
                }
                if ((size & 1) == 1) fs.Position++;
                continue;
            }

            if (Matches(fourcc, "MAOF"))
            {
                int totalCells = 64 * 64;
                int availableCells = (int)Math.Min((uint)totalCells, size / 4);

                var offsets = new uint[64, 64];
                for (int i = 0; i < availableCells; i++)
                {
                    int y = i / 64;
                    int x = i % 64;
                    offsets[y, x] = br.ReadUInt32();
                }
                long remain = (long)size - (availableCells * 4);
                if (remain > 0) fs.Seek(remain, SeekOrigin.Current);

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

                        int expected = (WdlTile.OuterGrid * WdlTile.OuterGrid + WdlTile.InnerGrid * WdlTile.InnerGrid) * 2;
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

                        var holeMask16 = new ushort[WdlTile.InnerGrid];

                        fs.Position = mareDataEnd;
                        if ((innerSize & 1) == 1) fs.Position++;

                        if (fs.Position + 8 <= len)
                        {
                            long mahoHeaderPos = fs.Position;
                            string mahoFour = ReadFourCC(br);
                            uint mahoSize = br.ReadUInt32();
                            if (Matches(mahoFour, "MAHO") && fs.Position + mahoSize <= len)
                            {
                                long mahoDataStart = fs.Position;
                                int rows = (int)Math.Min((uint)WdlTile.InnerGrid, mahoSize / 2);
                                for (int r = 0; r < rows; r++)
                                {
                                    holeMask16[r] = br.ReadUInt16();
                                }
                                long mahoDataEnd = mahoDataStart + mahoSize;
                                fs.Position = mahoDataEnd;
                                if ((mahoSize & 1) == 1) fs.Position++;
                            }
                            else
                            {
                                fs.Position = mahoHeaderPos;
                            }
                        }

                        tiles[y, x] = new WdlTile(height17, height16, holeMask16);

                    NextTile:
                        fs.Position = ret;
                    }
                }

                done = true;
                break;
            }
            else
            {
                fs.Position = dataStart + size + ((size & 1) == 1 ? 1 : 0);
            }
        }

        return new Wdl(wdlPath, tiles);
    }

    private static string ReadFourCC(BinaryReader br)
    {
        var b = br.ReadBytes(4);
        return Encoding.ASCII.GetString(b);
    }

    private static bool Matches(string fourCC, string expected)
    {
        if (fourCC.Equals(expected, StringComparison.OrdinalIgnoreCase)) return true;
        var rev = new string(new[] { expected[3], expected[2], expected[1], expected[0] });
        return fourCC.Equals(rev, StringComparison.OrdinalIgnoreCase);
    }
}
