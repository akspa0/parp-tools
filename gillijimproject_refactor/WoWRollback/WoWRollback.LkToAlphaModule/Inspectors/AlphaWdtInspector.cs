using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace WoWRollback.LkToAlphaModule.Inspectors;

public static class AlphaWdtInspector
{
    private const int ChunkHeader = 8;

    private static string ReadToken(FileStream fs, long offset)
    {
        fs.Seek(offset, SeekOrigin.Begin);
        Span<byte> hdr = stackalloc byte[4];
        if (fs.Read(hdr) != 4) return "";
        return Encoding.ASCII.GetString(hdr);
    }

    private static int ReadInt32(FileStream fs, long offset)
    {
        fs.Seek(offset, SeekOrigin.Begin);
        Span<byte> buf = stackalloc byte[4];
        if (fs.Read(buf) != 4) return 0;
        return BitConverter.ToInt32(buf);
    }

    private static (string tokenOnDisk, int size) ReadChunkHeader(FileStream fs, long offset)
    {
        fs.Seek(offset, SeekOrigin.Begin);
        Span<byte> hdr = stackalloc byte[8];
        if (fs.Read(hdr) != 8) return ("", 0);
        string tok = Encoding.ASCII.GetString(hdr.Slice(0, 4));
        int size = BitConverter.ToInt32(hdr.Slice(4, 4));
        return (tok, size);
    }

    private static string ForwardFourCC(string onDisk)
    {
        if (string.IsNullOrEmpty(onDisk) || onDisk.Length != 4) return onDisk;
        return new string(new[] { onDisk[3], onDisk[2], onDisk[1], onDisk[0] });
        }

    public static void Inspect(string wdtPath, int sampleTiles)
    {
        using var fs = File.OpenRead(wdtPath);
        Console.WriteLine($"[wdt] file: {wdtPath} size={fs.Length}");

        // Top-level scan
        long off = 0;
        var topOrder = new List<(long off, string onDisk, string fwd, int size)>();
        while (off + ChunkHeader <= fs.Length)
        {
            var (tok, size) = ReadChunkHeader(fs, off);
            if (string.IsNullOrWhiteSpace(tok)) break;
            var fwd = ForwardFourCC(tok);
            topOrder.Add((off, tok, fwd, size));
            int pad = (size & 1) == 1 ? 1 : 0;
            long next = off + ChunkHeader + size + pad;
            if (next <= off) break; // safety
            off = next;
            if (topOrder.Count > 32) break; // enough to see ordering
        }
        Console.WriteLine("[wdt] top-level order:");
        foreach (var t in topOrder)
            Console.WriteLine($"  @0x{t.off:X8} on-disk='{t.onDisk}' fwd='{t.fwd}' size={t.size}");

        // Locate MPHD and MAIN from order
        long mphdDataStart = -1;
        long mainDataStart = -1;
        foreach (var t in topOrder)
        {
            if (t.fwd == "MPHD") mphdDataStart = t.off + ChunkHeader;
            if (t.fwd == "MAIN") mainDataStart = t.off + ChunkHeader;
        }
        if (mphdDataStart < 0 || mainDataStart < 0)
        {
            Console.WriteLine("[wdt] MPHD or MAIN not found, abort");
            return;
        }

        // MDNM/MONM from MPHD
        int offsMdnm = ReadInt32(fs, mphdDataStart + 4);
        int offsMonm = ReadInt32(fs, mphdDataStart + 12);
        Console.WriteLine($"[wdt] MPHD: offsMdnm=0x{offsMdnm:X}, token='{ReadToken(fs, offsMdnm)}' offsMonm=0x{offsMonm:X}, token='{ReadToken(fs, offsMonm)}'");

        // MAIN: 4096 cells * 16
        int nonZero = 0;
        var tiles = new List<(int idx, int mhdrOff, int sizeToFirstMcnk)>();
        for (int i = 0; i < 4096; i++)
        {
            long cell = mainDataStart + (i * 16);
            int mhdrOff = ReadInt32(fs, cell);
            int sizeToFirst = ReadInt32(fs, cell + 4);
            if (mhdrOff != 0)
            {
                nonZero++;
                if (tiles.Count < sampleTiles)
                    tiles.Add((i, mhdrOff, sizeToFirst));
            }
        }
        Console.WriteLine($"[main] non-zero tiles: {nonZero}");

        // For a few tiles, detect MHDR->MCIN base and validate first MCNK
        foreach (var t in tiles)
        {
            int x = t.idx % 64, y = t.idx / 64;
            string mhdrTok = ReadToken(fs, t.mhdrOff);
            int offsInfoDataRel = ReadInt32(fs, t.mhdrOff + 8 + 0);
            long mcinAtDataRel = t.mhdrOff + 8 + offsInfoDataRel;
            long mcinAtStartRel = t.mhdrOff + offsInfoDataRel;
            string tokDataRel = ReadToken(fs, mcinAtDataRel);
            string tokStartRel = ReadToken(fs, mcinAtStartRel);
            long firstMcnkAbs = t.mhdrOff + t.sizeToFirstMcnk;
            string firstMcnkTok = ReadToken(fs, firstMcnkAbs);
            Console.WriteLine($"[tile {y:D2}_{x:D2}] MHDR @0x{t.mhdrOff:X} token='{mhdrTok}' offsInfo(data[+0])={offsInfoDataRel} -> dataRel='{tokDataRel}' startRel='{tokStartRel}' firstMCNK @0x{firstMcnkAbs:X} token='{firstMcnkTok}'");

            // Also inspect first MCIN entry to see if offsets are absolute and sizes plausible
            var (mcinTok, mcinSize) = ReadChunkHeader(fs, mcinAtDataRel);
            if (ForwardFourCC(mcinTok) == "MCIN")
            {
                long mcinDataStart = mcinAtDataRel + ChunkHeader;
                int off0 = ReadInt32(fs, mcinDataStart + 0);
                int size0 = ReadInt32(fs, mcinDataStart + 4);
                string mcnk0Tok = off0 > 0 ? ReadToken(fs, off0) : "";
                Console.WriteLine($"  MCIN: size={mcinSize}, entry0 off=0x{off0:X} size={size0} token='{mcnk0Tok}'");
            }
        }
    }
}
