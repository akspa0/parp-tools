using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace GillijimProject.Next.Core.WowFiles.Alpha;

/// <summary>
/// Reader for Alpha-era ADT files.
/// Scans MCIN for MCNK entries, reads MCNK headers, detects MCVT/MCLQ presence, and returns a summary.
/// </summary>
public static class AlphaAdtReader
{
    public static AlphaAdt Read(string adtPath)
    {
        if (string.IsNullOrWhiteSpace(adtPath) || !File.Exists(adtPath))
            throw new FileNotFoundException("ADT path not found", adtPath);

        using var fs = File.OpenRead(adtPath);
        using var br = new BinaryReader(fs, Encoding.ASCII, leaveOpen: true);

        if (!TryFindChunk(fs, br, "MCIN", out long mcinDataOffset, out uint mcinSize))
        {
            return new AlphaAdt(adtPath, 0, Array.Empty<AlphaAdtChunk>());
        }

        int entryCount = (int)Math.Min(256, mcinSize / 16);
        var chunks = new List<AlphaAdtChunk>(entryCount);
        int present = 0;

        fs.Position = mcinDataOffset;
        var entries = new (uint off, uint size)[entryCount];
        for (int i = 0; i < entryCount; i++)
        {
            uint off = br.ReadUInt32();
            uint size = br.ReadUInt32();
            _ = br.ReadUInt32(); // flags
            _ = br.ReadUInt32(); // id
            entries[i] = (off, size);
        }

        for (int idx = 0; idx < entryCount; idx++)
        {
            uint off = entries[idx].off;
            uint size = entries[idx].size;
            if (off == 0 || off + 8u > fs.Length) continue;

            try
            {
                fs.Position = off;
                string fourcc = ReadFourCC(br);
                uint chunkSize = br.ReadUInt32();
                long chunkDataStart = fs.Position;
                if (!fourcc.Equals("MCNK", StringComparison.OrdinalIgnoreCase))
                {
                    // Recover if header is missing by trusting MCIN-provided size
                    if (!LooksLikeMcnkHeader(fs, br)) continue;
                    chunkSize = size;
                    chunkDataStart = off; // headerless case
                }

                long chunkEnd = SafeAdd(chunkDataStart, chunkSize);
                if (chunkEnd <= 0 || chunkEnd > fs.Length) chunkEnd = fs.Length;

                // Read minimal header
                if (chunkDataStart + 128 > fs.Length) continue;
                fs.Position = chunkDataStart;
                var h = ReadMcnkHeader(br);

                bool hasMclq = h.OfsLiquid != 0; // sizeLiquid may be padding in alpha
                bool hasMcvt = h.OfsHeight != 0 && (chunkDataStart + h.OfsHeight < chunkEnd);
                int mcvtFloats = hasMcvt ? 145 : 0; // 81 outer + 64 inner

                chunks.Add(new AlphaAdtChunk(
                    Index: idx,
                    McnkOffset: off,
                    McnkSize: chunkSize,
                    HasMcvt: hasMcvt,
                    McvtFloatCount: mcvtFloats,
                    HasMclq: hasMclq
                ));
                present++;
            }
            catch
            {
                // Skip problematic chunk
            }
        }

        return new AlphaAdt(adtPath, present, chunks);
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
            if (sz > len || dataPos + sz < dataPos)
            {
                fs.Position = pos + 4; // resync
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
            _ = br.ReadUInt32(); // flags
            _ = br.ReadUInt32(); // indexX
            _ = br.ReadUInt32(); // indexY
            float radius = br.ReadSingle();
            if (radius < -1e6f || radius > 1e6f) return false;
            fs.Position = start + 128;
            return true;
        }
        catch { return false; }
        finally { fs.Position = start; }
    }

    private static McnkHeader ReadMcnkHeader(BinaryReader br)
    {
        uint flags = br.ReadUInt32();
        _ = br.ReadUInt32(); // indexX
        _ = br.ReadUInt32(); // indexY
        _ = br.ReadSingle(); // radius
        _ = br.ReadUInt32(); // nLayers
        _ = br.ReadUInt32(); // nDoodadRefs
        uint ofsHeight = br.ReadUInt32();
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
        for (int i = 0; i < 8; i++) _ = br.ReadUInt16(); // predTex
        for (int i = 0; i < 8; i++) _ = br.ReadByte(); // noEffectDoodad
        _ = br.ReadUInt32(); // ofsSndEmitters
        _ = br.ReadUInt32(); // nSndEmitters
        uint ofsLiquid = br.ReadUInt32();
        uint sizeLiquid = br.ReadUInt32();
        return new McnkHeader(flags, ofsHeight, ofsLiquid, sizeLiquid);
    }

    private static long SafeAdd(long a, uint b)
    {
        try { return checked(a + b); } catch { return -1; }
    }

    private readonly struct McnkHeader
    {
        public readonly uint Flags;
        public readonly uint OfsHeight;
        public readonly uint OfsLiquid;
        public readonly uint SizeLiquid;
        public McnkHeader(uint flags, uint ofsHeight, uint ofsLiquid, uint sizeLiquid)
        { Flags = flags; OfsHeight = ofsHeight; OfsLiquid = ofsLiquid; SizeLiquid = sizeLiquid; }
    }
}
