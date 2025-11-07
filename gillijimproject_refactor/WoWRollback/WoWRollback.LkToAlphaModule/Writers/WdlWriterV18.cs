using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using GillijimProject.WowFiles;
using GillijimProject.WowFiles.LichKing;
using Util = GillijimProject.Utilities.Utilities;
using WoWRollback.Core.Services.Archive;

namespace WoWRollback.LkToAlphaModule.Writers;

public static class WdlWriterV18
{
    private const int GridTiles = 64 * 64; // 4096
    private const int McnkHeaderSize = 0x80; // LK MCNK header size
    private const int ChunkLettersAndSize = 8;

    public static void WriteFromLk(string lkWdtPath, string lkMapDir, string outWdlPath)
    {
        if (string.IsNullOrWhiteSpace(lkWdtPath)) throw new ArgumentException("lkWdtPath required");
        if (string.IsNullOrWhiteSpace(lkMapDir)) throw new ArgumentException("lkMapDir required");
        if (string.IsNullOrWhiteSpace(outWdlPath)) throw new ArgumentException("outWdlPath required");
        Directory.CreateDirectory(Path.GetDirectoryName(outWdlPath) ?? ".");

        string mapName = Path.GetFileNameWithoutExtension(lkWdtPath);
        var rootAdts = Directory.EnumerateFiles(lkMapDir, mapName + "_*.adt", SearchOption.TopDirectoryOnly)
            .Where(p => !p.Contains("_obj", StringComparison.OrdinalIgnoreCase) && !p.Contains("_tex", StringComparison.OrdinalIgnoreCase))
            .OrderBy(p => p, StringComparer.OrdinalIgnoreCase)
            .ToList();

        // Collect per-tile 545 int16 heights (17x17 outer + 16x16 inner)
        var perTileHeights = new Dictionary<int, short[]>(capacity: rootAdts.Count);

        foreach (var rootAdt in rootAdts)
        {
            try
            {
                var file = Path.GetFileNameWithoutExtension(rootAdt);
                var parts = file.Split('_');
                if (parts.Length < 3) continue;
                if (!int.TryParse(parts[^2], out int yy)) continue;
                if (!int.TryParse(parts[^1], out int xx)) continue;
                int tileIndex = xx * 64 + yy;

                var bytes = File.ReadAllBytes(rootAdt);
                int mhdrOff = FindFourCC(bytes, "MHDR");
                if (mhdrOff < 0) continue;
                var lkMhdr = new Mhdr(bytes, mhdrOff);
                int lkMhdrDataStart = mhdrOff + 8;
                int lkMcinOff = lkMhdr.GetOffset(Mhdr.McinOffset);
                if (lkMcinOff == 0) continue;
                var lkMcin = new Mcin(bytes, lkMhdrDataStart + lkMcinOff);
                var mcnkOffsets = lkMcin.GetMcnkOffsets();

                // Aggregators for the tile: full resolution grids
                var outer129 = new float[129, 129];
                var inner128 = new float[128, 128];
                var outerFilled = new bool[129, 129];
                var innerFilled = new bool[128, 128];

                for (int i = 0; i < 256; i++)
                {
                    int off = (i < mcnkOffsets.Count) ? mcnkOffsets[i] : 0;
                    if (off <= 0) continue;

                    // Read LK MCNK header for indices and baseZ
                    var hdrBytes = new byte[McnkHeaderSize];
                    Buffer.BlockCopy(bytes, off + ChunkLettersAndSize, hdrBytes, 0, McnkHeaderSize);
                    var lkHdr = Util.ByteArrayToStruct<McnkHeader>(hdrBytes);
                    int idxX = lkHdr.IndexX;
                    int idxY = lkHdr.IndexY;
                    float baseZ = lkHdr.PosZ;

                    // Locate MCVT payload within this LK MCNK
                    int mcnkSize = BitConverter.ToInt32(bytes, off + 4);
                    int subStart = off + ChunkLettersAndSize + McnkHeaderSize;
                    int subEnd = off + 8 + mcnkSize;
                    if (subEnd > bytes.Length) subEnd = bytes.Length;

                    byte[]? mcvtWhole = null;
                    for (int p = subStart; p + 8 <= subEnd;)
                    {
                        string fcc = Encoding.ASCII.GetString(bytes, p, 4);
                        int sz = BitConverter.ToInt32(bytes, p + 4);
                        if (sz < 0 || p + 8 + sz > subEnd) break;
                        int next = p + 8 + sz + ((sz & 1) == 1 ? 1 : 0);
                        if (fcc == "TVCM") // 'MCVT' on disk
                        {
                            mcvtWhole = new byte[8 + sz + ((sz & 1) == 1 ? 1 : 0)];
                            Buffer.BlockCopy(bytes, p, mcvtWhole, 0, mcvtWhole.Length);
                            break;
                        }
                        if (next <= p) break;
                        p = next;
                    }
                    if (mcvtWhole == null) continue;

                    int mcvtSize = BitConverter.ToInt32(mcvtWhole, 4);
                    if (mcvtSize < 145 * 4) continue; // need 145 floats
                    var lkData = new byte[mcvtSize];
                    Buffer.BlockCopy(mcvtWhole, 8, lkData, 0, mcvtSize);

                    // Convert LK interleaved order to Alpha-like order with baseZ applied
                    // Result: 145 floats (81 outer 9x9, then 64 inner 8x8)
                    var alphaMcvt = ConvertMcvtLkToAlpha(lkData, baseZ);

                    // Scatter into tile grids
                    int rowBase = idxY * 8;
                    int colBase = idxX * 8;
                    // Outer 9x9
                    for (int oy = 0; oy < 9; oy++)
                    {
                        for (int ox = 0; ox < 9; ox++)
                        {
                            int outerIdx = oy * 9 + ox;
                            int rr = rowBase + oy;
                            int cc = colBase + ox;
                            if (rr >= 0 && rr < 129 && cc >= 0 && cc < 129)
                            {
                                outer129[rr, cc] = BitConverter.ToSingle(alphaMcvt, outerIdx * 4);
                                outerFilled[rr, cc] = true;
                            }
                        }
                    }
                    // Inner 8x8
                    int innerBase = 9 * 9 * 4;
                    for (int iy2 = 0; iy2 < 8; iy2++)
                    {
                        for (int ix2 = 0; ix2 < 8; ix2++)
                        {
                            int innerIdx = iy2 * 8 + ix2;
                            int rr = rowBase + iy2;
                            int cc = colBase + ix2;
                            if (rr >= 0 && rr < 128 && cc >= 0 && cc < 128)
                            {
                                inner128[rr, cc] = BitConverter.ToSingle(alphaMcvt, innerBase + innerIdx * 4);
                                innerFilled[rr, cc] = true;
                            }
                        }
                    }
                }

                // Downsample to MARE layout: 17x17 outer + 16x16 inner
                var heights = new short[545];
                int pos = 0;
                for (int y = 0; y < 17; y++)
                {
                    int rr = Math.Min(128, y * 8);
                    for (int x = 0; x < 17; x++)
                    {
                        int cc = Math.Min(128, x * 8);
                        float v = outer129[rr, cc];
                        heights[pos++] = (short)Math.Round(v);
                    }
                }
                for (int y = 0; y < 16; y++)
                {
                    int rr = y * 8;
                    for (int x = 0; x < 16; x++)
                    {
                        int cc = x * 8;
                        float v = inner128[rr, cc];
                        heights[pos++] = (short)Math.Round(v);
                    }
                }
                perTileHeights[tileIndex] = heights;
            }
            catch { /* best-effort per tile */ }
        }

        // Emit WDL: MVER, MAOF, then per-tile MARE + MAHO (zeros)
        using var ms = new MemoryStream();
        // MVER
        ms.Write(new Chunk("MVER", 4, BitConverter.GetBytes(18)).GetWholeChunk());

        // MAOF placeholder
        var maofData = new byte[GridTiles * 4];
        long maofStart = ms.Position; // letters
        long maofDataStart = maofStart + 8;
        ms.Write(new Chunk("MAOF", maofData.Length, maofData).GetWholeChunk());

        // Write tiles in index order; record offsets
        var offsets = new int[GridTiles];
        var mahoData = new byte[16 * 2]; // 32 bytes, zeros
        for (int i = 0; i < GridTiles; i++)
        {
            if (!perTileHeights.TryGetValue(i, out var heights))
            {
                offsets[i] = 0; // unused tile
                continue;
            }
            // MARE data: 545 int16 (outer 17x17 then inner 16x16)
            var mareData = new byte[545 * 2];
            Buffer.BlockCopy(heights, 0, mareData, 0, mareData.Length);
            long marePos = ms.Position;
            ms.Write(new Chunk("MARE", mareData.Length, mareData).GetWholeChunk());
            ms.Write(new Chunk("MAHO", mahoData.Length, mahoData).GetWholeChunk());
            offsets[i] = checked((int)marePos);
        }

        // Patch MAOF data
        long save = ms.Position;
        ms.Position = maofDataStart;
        for (int i = 0; i < GridTiles; i++)
        {
            ms.Write(BitConverter.GetBytes(offsets[i]));
        }
        ms.Position = save;

        var outBytes = ms.ToArray();
        // Simple header dump
        try
        {
            int at = 0; for (int k = 0; k < 5 && at + 8 <= outBytes.Length; k++)
            {
                var ch = new Chunk(outBytes, at);
                Console.WriteLine($"[wdl][hdr] chunk{k} off={at} token={ch.Letters} size={ch.GivenSize}");
                int pad = (ch.GivenSize & 1) == 1 ? 1 : 0; at += 8 + ch.GivenSize + pad;
            }
        }
        catch { }

        File.WriteAllBytes(outWdlPath, outBytes);
        Console.WriteLine($"[ok] WDL written: {outWdlPath}");
    }

    public static void WriteFromArchive(IArchiveSource src, string mapName, string outWdlPath)
    {
        if (src == null) throw new ArgumentNullException(nameof(src));
        if (string.IsNullOrWhiteSpace(mapName)) throw new ArgumentException("mapName required");
        if (string.IsNullOrWhiteSpace(outWdlPath)) throw new ArgumentException("outWdlPath required");
        Directory.CreateDirectory(Path.GetDirectoryName(outWdlPath) ?? ".");

        var perTileHeights = new Dictionary<int, short[]>();
        for (int yy = 0; yy < 64; yy++)
        {
            for (int xx = 0; xx < 64; xx++)
            {
                string rootVPath = $"world/maps/{mapName}/{mapName}_{yy}_{xx}.adt";
                if (!src.FileExists(rootVPath)) continue;
                int tileIndex = xx * 64 + yy;
                try
                {
                    using var s = src.OpenFile(rootVPath);
                    using var msFile = new MemoryStream();
                    s.CopyTo(msFile); var bytes = msFile.ToArray();
                    int mhdrOff = FindFourCC(bytes, "MHDR"); if (mhdrOff < 0) continue;
                    var lkMhdr = new Mhdr(bytes, mhdrOff);
                    int lkMhdrDataStart = mhdrOff + 8;
                    int lkMcinOff = lkMhdr.GetOffset(Mhdr.McinOffset); if (lkMcinOff == 0) continue;
                    var lkMcin = new Mcin(bytes, lkMhdrDataStart + lkMcinOff);
                    var mcnkOffsets = lkMcin.GetMcnkOffsets();

                    var outer129 = new float[129, 129];
                    var inner128 = new float[128, 128];
                    for (int i = 0; i < 256; i++)
                    {
                        int off = (i < mcnkOffsets.Count) ? mcnkOffsets[i] : 0;
                        if (off <= 0) continue;

                        byte[] hdrBytes = new byte[McnkHeaderSize];
                        Buffer.BlockCopy(bytes, off + ChunkLettersAndSize, hdrBytes, 0, McnkHeaderSize);
                        var lkHdr = Util.ByteArrayToStruct<McnkHeader>(hdrBytes);
                        int idxX = lkHdr.IndexX;
                        int idxY = lkHdr.IndexY;
                        float baseZ = lkHdr.PosZ;

                        int mcnkSize = BitConverter.ToInt32(bytes, off + 4);
                        int subStart = off + ChunkLettersAndSize + McnkHeaderSize;
                        int subEnd = off + 8 + mcnkSize; if (subEnd > bytes.Length) subEnd = bytes.Length;

                        byte[]? mcvtWhole = null;
                        for (int p = subStart; p + 8 <= subEnd;)
                        {
                            string fcc = Encoding.ASCII.GetString(bytes, p, 4);
                            int sz = BitConverter.ToInt32(bytes, p + 4);
                            if (sz < 0 || p + 8 + sz > subEnd) break;
                            int next = p + 8 + sz + ((sz & 1) == 1 ? 1 : 0);
                            if (fcc == "TVCM") { mcvtWhole = new byte[8 + sz + ((sz & 1) == 1 ? 1 : 0)]; Buffer.BlockCopy(bytes, p, mcvtWhole, 0, mcvtWhole.Length); break; }
                            if (next <= p) break; p = next;
                        }
                        if (mcvtWhole == null) continue;
                        int mcvtSize = BitConverter.ToInt32(mcvtWhole, 4); if (mcvtSize < 145 * 4) continue;
                        var lkData = new byte[mcvtSize]; Buffer.BlockCopy(mcvtWhole, 8, lkData, 0, mcvtSize);
                        var alphaMcvt = ConvertMcvtLkToAlpha(lkData, baseZ);

                        int rowBase = idxY * 8; int colBase = idxX * 8;
                        for (int oy = 0; oy < 9; oy++)
                        {
                            for (int ox = 0; ox < 9; ox++)
                            {
                                int outerIdx = oy * 9 + ox; int rr = rowBase + oy; int cc = colBase + ox;
                                if (rr >= 0 && rr < 129 && cc >= 0 && cc < 129)
                                    outer129[rr, cc] = BitConverter.ToSingle(alphaMcvt, outerIdx * 4);
                            }
                        }
                        int innerBase = 9 * 9 * 4;
                        for (int iy2 = 0; iy2 < 8; iy2++)
                        {
                            for (int ix2 = 0; ix2 < 8; ix2++)
                            {
                                int innerIdx = iy2 * 8 + ix2; int rr = rowBase + iy2; int cc = colBase + ix2;
                                if (rr >= 0 && rr < 128 && cc >= 0 && cc < 128)
                                    inner128[rr, cc] = BitConverter.ToSingle(alphaMcvt, innerBase + innerIdx * 4);
                            }
                        }
                    }

                    var heights = new short[545]; int pos = 0;
                    for (int y = 0; y < 17; y++) { int rr = Math.Min(128, y * 8); for (int x = 0; x < 17; x++) { int cc = Math.Min(128, x * 8); float v = outer129[rr, cc]; heights[pos++] = (short)Math.Round(v); } }
                    for (int y = 0; y < 16; y++) { int rr = y * 8; for (int x = 0; x < 16; x++) { int cc = x * 8; float v = inner128[rr, cc]; heights[pos++] = (short)Math.Round(v); } }
                    perTileHeights[tileIndex] = heights;
                }
                catch { }
            }
        }

        using var w = new MemoryStream();
        w.Write(new Chunk("MVER", 4, BitConverter.GetBytes(18)).GetWholeChunk());
        var maofData = new byte[GridTiles * 4]; long maofStart = w.Position; long maofDataStart = maofStart + 8; w.Write(new Chunk("MAOF", maofData.Length, maofData).GetWholeChunk());
        var offsets = new int[GridTiles]; var mahoData = new byte[16 * 2];
        for (int i = 0; i < GridTiles; i++)
        {
            if (!perTileHeights.TryGetValue(i, out var heights)) { offsets[i] = 0; continue; }
            var mareData = new byte[545 * 2]; Buffer.BlockCopy(heights, 0, mareData, 0, mareData.Length);
            long marePos = w.Position; w.Write(new Chunk("MARE", mareData.Length, mareData).GetWholeChunk()); w.Write(new Chunk("MAHO", mahoData.Length, mahoData).GetWholeChunk());
            offsets[i] = checked((int)marePos);
        }
        long save = w.Position; w.Position = maofDataStart; for (int i = 0; i < GridTiles; i++) w.Write(BitConverter.GetBytes(offsets[i])); w.Position = save;
        File.WriteAllBytes(outWdlPath, w.ToArray());
    }

    private static byte[] ConvertMcvtLkToAlpha(byte[] mcvtLk, float baseZ)
    {
        // Reorder LK interleaved to Alpha order and add baseZ to get absolute heights
        const int floatSize = 4;
        const int outerRowFloats = 9;
        const int innerRowFloats = 8;
        const int outerRowBytes = outerRowFloats * floatSize; // 36
        const int innerRowBytes = innerRowFloats * floatSize; // 32
        const int outerBlockBytes = outerRowBytes * 9; // 324

        var alphaData = new byte[145 * 4];
        int src = 0;
        for (int i = 0; i < 9; i++)
        {
            // Outer row i: 9 floats
            for (int j = 0; j < outerRowFloats; j++)
            {
                float v = BitConverter.ToSingle(mcvtLk, src + j * floatSize) + baseZ;
                Buffer.BlockCopy(BitConverter.GetBytes(v), 0, alphaData, (i * outerRowFloats + j) * floatSize, floatSize);
            }
            src += outerRowBytes;
            // Inner row i: 8 floats (rows 0..7 only)
            if (i < 8)
            {
                int innerDestBase = outerBlockBytes + (i * innerRowBytes);
                for (int j = 0; j < innerRowFloats; j++)
                {
                    float v = BitConverter.ToSingle(mcvtLk, src + j * floatSize) + baseZ;
                    Buffer.BlockCopy(BitConverter.GetBytes(v), 0, alphaData, innerDestBase + j * floatSize, floatSize);
                }
                src += innerRowBytes;
            }
        }
        return alphaData;
    }

    private static int FindFourCC(byte[] buf, string forwardFourCC)
    {
        if (buf == null || buf.Length < 8) return -1;
        if (string.IsNullOrEmpty(forwardFourCC) || forwardFourCC.Length != 4) return -1;
        string reversed = new string(new[] { forwardFourCC[3], forwardFourCC[2], forwardFourCC[1], forwardFourCC[0] });
        for (int i = 0; i + 8 <= buf.Length;)
        {
            string fcc = Encoding.ASCII.GetString(buf, i, 4);
            int size = BitConverter.ToInt32(buf, i + 4);
            if (size < 0 || size > buf.Length) break;
            int dataStart = i + 8;
            int next = dataStart + size + ((size & 1) == 1 ? 1 : 0);
            if (fcc == reversed) return i;
            if (dataStart + size > buf.Length) break;
            if (next <= i) break;
            i = next;
        }
        return -1;
    }
}
