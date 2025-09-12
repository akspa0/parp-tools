using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using GillijimProject.Next.Core.Domain;
using GillijimProject.Next.Core.Domain.Liquids;

namespace GillijimProject.Next.Core.IO;

/// <summary>
/// Alpha-era MCLQ extractor.
/// Reads MCIN to locate MCNKs and parses their MCLQ payloads (no internal fourcc/size)
/// into 16x16 per-chunk <see cref="MclqData"/> entries. Robust to offset origin
/// differences between Alpha and ADT v18 by trying multiple base origins.
/// </summary>
public sealed class AlphaMclqExtractor : IAlphaLiquidsExtractor
{
    public MclqData?[] Extract(AdtAlpha adt)
    {
        if (adt is null) throw new ArgumentNullException(nameof(adt));
        if (string.IsNullOrWhiteSpace(adt.Path) || !File.Exists(adt.Path))
            throw new FileNotFoundException("ADT path not found", adt.Path);

        var results = new MclqData?[256];

        using var fs = File.OpenRead(adt.Path);
        using var br = new BinaryReader(fs, Encoding.ASCII, leaveOpen: true);

        // Find MCIN top-level chunk
        if (!TryFindChunk(fs, br, "MCIN", out long mcinDataOffset, out uint mcinSize))
        {
            // No liquids if we cannot even locate MCIN
            return results;
        }

        // MCIN entries (pre-Cata): 256 entries, 16 bytes each
        int entryCount = (int)Math.Min(256, mcinSize / 16);
        var mcnkOffsets = new (uint offset, uint size)[256];
        fs.Position = mcinDataOffset;
        for (int i = 0; i < entryCount; i++)
        {
            uint off = br.ReadUInt32();     // absolute offset to MCNK
            uint size = br.ReadUInt32();    // size of MCNK chunk (data size)
            _ = br.ReadUInt32();            // flags (ignored)
            _ = br.ReadUInt32();            // async id / unused
            mcnkOffsets[i] = (off, size);
        }

        // Iterate MCNKs
        for (int i = 0; i < entryCount; i++)
        {
            uint off = mcnkOffsets[i].offset;
            uint size = mcnkOffsets[i].size;
            if (off == 0 || size < 128 + 8) { results[i] = null; continue; }
            if (off + 8u > fs.Length) { results[i] = null; continue; }

            try
            {
                fs.Position = off;
                string fourcc = ReadFourCC(br);
                uint chunkSize = br.ReadUInt32();
                if (!fourcc.Equals("MCNK", StringComparison.OrdinalIgnoreCase))
                {
                    // Some alpha assets may lack fourcc or differ; try to recover by assuming this is actually MCNK data
                    // If header looks valid at this position, proceed; otherwise skip
                    fs.Position = off; // reset
                    if (!LooksLikeMcnkHeader(fs, br)) { results[i] = null; continue; }
                    // Simulate chunkSize from MCIN when fourcc not present
                    chunkSize = size;
                }

                long chunkDataStart = fs.Position; // after fourcc+size
                long chunkEnd = checked(chunkDataStart + chunkSize);
                if (chunkEnd > fs.Length) chunkEnd = fs.Length;

                // Read 128-byte MCNK header at data start
                if (chunkDataStart + 128 > fs.Length) { results[i] = null; continue; }
                fs.Position = chunkDataStart;

                var header = ReadMcnkHeader(br);

                // If no liquid offset or too small size, skip
                if (header.OfsLiquid == 0 || (header.SizeLiquid != 0 && header.SizeLiquid <= 8))
                {
                    results[i] = null; continue;
                }

                // Compute candidate bases for Alpha/v18 variants
                long baseA = off;                   // beginning of MCNK chunk (includes fourcc+size)
                long baseB = chunkDataStart;        // beginning of MCNK data (after fourcc+size)
                long baseC = chunkDataStart + 128;  // end of MCNK header

                long[] candidates = new[]
                {
                    SafeAdd(baseC, header.OfsLiquid), // Alpha.md says relative to end of header
                    SafeAdd(baseB, header.OfsLiquid), // sometimes relative to data start
                    SafeAdd(baseA, header.OfsLiquid), // ADT_v18.md says relative to chunk begin
                };

                MclqData? parsed = null;
                foreach (var cand in candidates)
                {
                    if (cand <= 0) continue;
                    if (cand + 8 > fs.Length) continue; // need at least CRange and minimal data
                    if (cand >= chunkEnd) continue;

                    if (TryParseMclqAt(fs, br, cand, chunkEnd, header.Flags, out parsed))
                    {
                        break;
                    }
                }

                results[i] = parsed; // may be null if none parsed
            }
            catch
            {
                // Robust extractor: ignore per-chunk failures
                results[i] = null;
            }
        }

        return results;
    }

    private static bool TryFindChunk(FileStream fs, BinaryReader br, string id, out long dataOffset, out uint size)
    {
        dataOffset = 0; size = 0;
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
            // Sanity: avoid infinite loops on corrupt sizes
            if (sz > len || dataPos + sz < dataPos)
            {
                // fallback: step forward 4 bytes and resync
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
            // Peek 128 bytes and perform minimal checks: not beyond file, reasonable values
            if (start + 128 > fs.Length) return false;
            // flags
            uint flags = br.ReadUInt32();
            // indexX/indexY
            _ = br.ReadUInt32();
            _ = br.ReadUInt32();
            // radius
            float radius = br.ReadSingle();
            if (radius < -1e6f || radius > 1e6f) return false; // arbitrary sanity
            // skip to end of header
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
        _ = br.ReadUInt32(); // indexX
        _ = br.ReadUInt32(); // indexY
        _ = br.ReadSingle(); // radius
        _ = br.ReadUInt32(); // nLayers
        _ = br.ReadUInt32(); // nDoodadRefs
        _ = br.ReadUInt32(); // ofsHeight
        _ = br.ReadUInt32(); // ofsNormal
        _ = br.ReadUInt32(); // ofsLayer
        _ = br.ReadUInt32(); // ofsRefs
        _ = br.ReadUInt32(); // ofsAlpha
        _ = br.ReadUInt32(); // sizeAlpha
        _ = br.ReadUInt32(); // ofsShadow
        _ = br.ReadUInt32(); // sizeShadow
        _ = br.ReadUInt32(); // areaid
        _ = br.ReadUInt32(); // nMapObjRefs
        _ = br.ReadUInt16(); // holes
        _ = br.ReadUInt16(); // pad0
        for (int i = 0; i < 8; i++) _ = br.ReadUInt16(); // predTex[8]
        for (int i = 0; i < 8; i++) _ = br.ReadByte();   // noEffectDoodad[8]
        _ = br.ReadUInt32(); // ofsSndEmitters
        _ = br.ReadUInt32(); // nSndEmitters
        uint ofsLiquid = br.ReadUInt32();
        uint sizeLiquid = br.ReadUInt32(); // ADT v18 only; alpha may be padding here
        return new McnkHeader(flags, ofsLiquid, sizeLiquid);
    }

    private static bool TryParseMclqAt(FileStream fs, BinaryReader br, long start, long chunkEnd, uint flags, out MclqData? data)
    {
        data = null;
        if (start + 8 > fs.Length) return false;
        fs.Position = start;

        // height range (CRange)
        float heightMin = br.ReadSingle();
        float heightMax = br.ReadSingle();

        // Choose vertex layout by flags: magma > ocean > water (default)
        var layout = GuessLayout(flags);

        // Read vertices (9x9 = 81)
        const int V = MclqData.VertexGrid * MclqData.VertexGrid; // 81
        var heights = new float[V];
        var depth = new byte[V];

        try
        {
            switch (layout)
            {
                case VertexLayout.Magma:
                    // 81 * (u16 s, u16 t, float height)
                    for (int i = 0; i < V; i++)
                    {
                        if (fs.Position + 8 > chunkEnd) return false;
                        _ = br.ReadUInt16(); // s (UV)
                        _ = br.ReadUInt16(); // t (UV)
                        heights[i] = br.ReadSingle();
                        depth[i] = 0; // undefined
                    }
                    break;
                case VertexLayout.Ocean:
                    // 81 * (byte depth, byte foam, byte wet, byte filler)
                    for (int i = 0; i < V; i++)
                    {
                        if (fs.Position + 4 > chunkEnd) return false;
                        depth[i] = br.ReadByte();
                        _ = br.ReadByte(); // foam
                        _ = br.ReadByte(); // wet
                        _ = br.ReadByte(); // filler
                        heights[i] = heightMin; // infer flat height
                    }
                    break;
                default:
                    // Water: 81 * (byte depth, byte flow0Pct, byte flow1Pct, byte filler, float height)
                    for (int i = 0; i < V; i++)
                    {
                        if (fs.Position + 8 > chunkEnd) return false;
                        depth[i] = br.ReadByte();
                        _ = br.ReadByte(); // flow0
                        _ = br.ReadByte(); // flow1
                        _ = br.ReadByte(); // filler
                        heights[i] = br.ReadSingle();
                    }
                    break;
            }

            // Tiles: 8x8 bytes
            const int T = MclqData.TileGrid; // 8
            var types = new MclqLiquidType[T, T];
            var flagsArr = new MclqTileFlags[T, T];
            if (fs.Position + 64 > chunkEnd) return false;

            for (int ty = 0; ty < T; ty++)
            {
                for (int tx = 0; tx < T; tx++)
                {
                    byte b = br.ReadByte();
                    var type = (MclqLiquidType)(b & 0x0F);
                    // Normalize unsupported low-nibble values to None
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
        // Alpha SMChunkFlags:
        // FLAG_LQ_RIVER = 0x4, FLAG_LQ_OCEAN = 0x8, FLAG_LQ_MAGMA = 0x10
        const uint LQ_RIVER = 0x4;
        const uint LQ_OCEAN = 0x8;
        const uint LQ_MAGMA = 0x10;

        if ((flags & LQ_MAGMA) != 0) return VertexLayout.Magma;
        if ((flags & LQ_OCEAN) != 0) return VertexLayout.Ocean;
        // river or default water
        return VertexLayout.Water;
    }

    private readonly struct McnkHeader
    {
        public readonly uint Flags;
        public readonly uint OfsLiquid;
        public readonly uint SizeLiquid;
        public McnkHeader(uint flags, uint ofsLiquid, uint sizeLiquid)
        {
            Flags = flags; OfsLiquid = ofsLiquid; SizeLiquid = sizeLiquid;
        }
    }

    private enum VertexLayout { Water, Ocean, Magma }
}
