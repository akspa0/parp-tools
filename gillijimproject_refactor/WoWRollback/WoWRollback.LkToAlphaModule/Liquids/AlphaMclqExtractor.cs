using System;
using System.IO;
using System.Text;

namespace WoWRollback.LkToAlphaModule.Liquids;

/// <summary>
/// Alpha-era MCLQ extractor. Reads raw Alpha ADT files and returns per-MCNK `MclqData` payloads.
/// Ported from the Next pipeline to support round-trip conversion within WoWRollback.
/// </summary>
internal sealed class AlphaMclqExtractor
{
    private readonly string _adtPath;

    public AlphaMclqExtractor(string adtPath)
    {
        if (string.IsNullOrWhiteSpace(adtPath))
            throw new ArgumentException("Alpha ADT path required", nameof(adtPath));
        if (!File.Exists(adtPath))
            throw new FileNotFoundException("Alpha ADT not found", adtPath);
        _adtPath = adtPath;
    }

    public MclqData?[] Extract()
    {
        using var fs = File.OpenRead(_adtPath);
        using var br = new BinaryReader(fs, Encoding.ASCII, leaveOpen: true);

        if (!TryFindChunk(fs, br, "MCIN", out long mcinDataOffset, out uint mcinSize))
        {
            return new MclqData?[256];
        }

        int entryCount = (int)Math.Min(256, mcinSize / 16);
        var mcnkEntries = new (uint offset, uint size)[256];
        fs.Position = mcinDataOffset;
        for (int i = 0; i < entryCount; i++)
        {
            uint off = br.ReadUInt32();
            uint size = br.ReadUInt32();
            _ = br.ReadUInt32();
            _ = br.ReadUInt32();
            mcnkEntries[i] = (off, size);
        }

        var results = new MclqData?[256];

        for (int i = 0; i < entryCount; i++)
        {
            (uint off, uint size) = mcnkEntries[i];
            if (off == 0 || size < 128 + 8) continue;
            if (off + 8u > fs.Length) continue;

            try
            {
                fs.Position = off;
                string fourcc = ReadFourCC(br);
                uint chunkSize = br.ReadUInt32();
                if (!fourcc.Equals("MCNK", StringComparison.OrdinalIgnoreCase))
                {
                    fs.Position = off;
                    if (!LooksLikeMcnkHeader(fs, br))
                        continue;
                    chunkSize = size;
                }

                long chunkDataStart = fs.Position;
                long chunkEnd = checked(chunkDataStart + chunkSize);
                if (chunkEnd > fs.Length)
                    chunkEnd = fs.Length;

                if (chunkDataStart + 128 > fs.Length)
                    continue;
                fs.Position = chunkDataStart;

                var header = ReadMcnkHeader(br);

                if (header.OfsLiquid == 0 || (header.SizeLiquid != 0 && header.SizeLiquid <= 8))
                {
                    continue;
                }

                long baseChunk = off;
                long baseData = chunkDataStart;
                long baseHeaderEnd = chunkDataStart + 128;

                long[] candidates =
                {
                    SafeAdd(baseHeaderEnd, header.OfsLiquid),
                    SafeAdd(baseData, header.OfsLiquid),
                    SafeAdd(baseChunk, header.OfsLiquid),
                };

                MclqData? parsed = null;
                foreach (long cand in candidates)
                {
                    if (cand <= 0) continue;
                    if (cand + 8 > fs.Length) continue;
                    if (cand >= chunkEnd) continue;

                    if (TryParseMclqAt(fs, br, cand, chunkEnd, header.Flags, out parsed))
                    {
                        break;
                    }
                }

                results[i] = parsed;
            }
            catch
            {
                results[i] = null;
            }
        }

        return results;
    }

    private static bool TryFindChunk(FileStream fs, BinaryReader br, string id, out long dataOffset, out uint size)
    {
        dataOffset = 0;
        size = 0;
        fs.Position = 0;
        long len = fs.Length;
        while (fs.Position + 8 <= len)
        {
            long pos = fs.Position;
            string cc = ReadFourCC(br);
            uint sz = br.ReadUInt32();
            long dataPos = fs.Position;
            if (cc.Equals(id, StringComparison.OrdinalIgnoreCase))
            {
                dataOffset = dataPos;
                size = sz;
                return true;
            }

            if (sz > len || dataPos + sz < dataPos)
            {
                fs.Position = pos + 4;
            }
            else
            {
                fs.Position = dataPos + sz;
            }
        }
        return false;
    }

    private static string ReadFourCC(BinaryReader br)
    {
        var b = br.ReadBytes(4);
        return Encoding.ASCII.GetString(b);
    }

    private static bool LooksLikeMcnkHeader(FileStream fs, BinaryReader br)
    {
        long start = fs.Position;
        try
        {
            if (start + 128 > fs.Length) return false;
            _ = br.ReadUInt32();
            _ = br.ReadUInt32();
            _ = br.ReadUInt32();
            float radius = br.ReadSingle();
            if (radius < -1e6f || radius > 1e6f) return false;
            fs.Position = start + 128;
            return true;
        }
        catch
        {
            return false;
        }
        finally
        {
            fs.Position = start;
        }
    }

    private static McnkHeader ReadMcnkHeader(BinaryReader br)
    {
        uint flags = br.ReadUInt32();
        _ = br.ReadUInt32();
        _ = br.ReadUInt32();
        _ = br.ReadSingle();
        _ = br.ReadUInt32();
        _ = br.ReadUInt32();
        _ = br.ReadUInt32();
        _ = br.ReadUInt32();
        _ = br.ReadUInt32();
        _ = br.ReadUInt32();
        _ = br.ReadUInt32();
        _ = br.ReadUInt32();
        _ = br.ReadUInt32();
        _ = br.ReadUInt32();
        _ = br.ReadUInt16();
        _ = br.ReadUInt16();
        for (int i = 0; i < 8; i++) _ = br.ReadUInt16();
        for (int i = 0; i < 8; i++) _ = br.ReadByte();
        _ = br.ReadUInt32();
        _ = br.ReadUInt32();
        uint ofsLiquid = br.ReadUInt32();
        uint sizeLiquid = br.ReadUInt32();
        return new McnkHeader(flags, ofsLiquid, sizeLiquid);
    }

    private static bool TryParseMclqAt(FileStream fs, BinaryReader br, long start, long chunkEnd, uint flags, out MclqData? data)
    {
        data = null;
        if (start + 8 > fs.Length) return false;
        fs.Position = start;

        float heightMin = br.ReadSingle();
        float heightMax = br.ReadSingle();

        const int V = MclqData.VertexGrid * MclqData.VertexGrid;
        var heights = new float[V];
        var depth = new byte[V];

        try
        {
            switch (GuessLayout(flags))
            {
                case VertexLayout.Magma:
                    for (int i = 0; i < V; i++)
                    {
                        if (fs.Position + 8 > chunkEnd) return false;
                        _ = br.ReadUInt16();
                        _ = br.ReadUInt16();
                        heights[i] = br.ReadSingle();
                        depth[i] = 0;
                    }
                    break;
                case VertexLayout.Ocean:
                    for (int i = 0; i < V; i++)
                    {
                        if (fs.Position + 4 > chunkEnd) return false;
                        depth[i] = br.ReadByte();
                        _ = br.ReadByte();
                        _ = br.ReadByte();
                        _ = br.ReadByte();
                        heights[i] = heightMin;
                    }
                    break;
                default:
                    for (int i = 0; i < V; i++)
                    {
                        if (fs.Position + 8 > chunkEnd) return false;
                        depth[i] = br.ReadByte();
                        _ = br.ReadByte();
                        _ = br.ReadByte();
                        _ = br.ReadByte();
                        heights[i] = br.ReadSingle();
                    }
                    break;
            }

            const int T = MclqData.TileGrid;
            var types = new MclqLiquidType[T, T];
            var flagsArr = new MclqTileFlags[T, T];
            if (fs.Position + 64 > chunkEnd) return false;

            for (int ty = 0; ty < T; ty++)
            {
                for (int tx = 0; tx < T; tx++)
                {
                    byte b = br.ReadByte();
                    var type = (MclqLiquidType)(b & 0x0F);
                    if (type != MclqLiquidType.None && type != MclqLiquidType.Ocean && type != MclqLiquidType.Slime && type != MclqLiquidType.River && type != MclqLiquidType.Magma)
                        type = MclqLiquidType.None;
                    var fl = (MclqTileFlags)(b & 0xF0);
                    types[tx, ty] = type;
                    flagsArr[tx, ty] = fl;
                }
            }

            data = new MclqData(heights, depth, types, flagsArr);
            return true;
        }
        catch
        {
            data = null;
            return false;
        }
    }

    private static long SafeAdd(long a, uint b)
    {
        try
        {
            return checked(a + b);
        }
        catch
        {
            return -1;
        }
    }

    private static VertexLayout GuessLayout(uint flags)
    {
        const uint LQ_RIVER = 0x4;
        const uint LQ_OCEAN = 0x8;
        const uint LQ_MAGMA = 0x10;

        if ((flags & LQ_MAGMA) != 0) return VertexLayout.Magma;
        if ((flags & LQ_OCEAN) != 0) return VertexLayout.Ocean;
        return VertexLayout.Water;
    }

    private readonly struct McnkHeader
    {
        public McnkHeader(uint flags, uint ofsLiquid, uint sizeLiquid)
        {
            Flags = flags;
            OfsLiquid = ofsLiquid;
            SizeLiquid = sizeLiquid;
        }

        public readonly uint Flags;
        public readonly uint OfsLiquid;
        public readonly uint SizeLiquid;
    }

    private enum VertexLayout { Water, Ocean, Magma }
}
