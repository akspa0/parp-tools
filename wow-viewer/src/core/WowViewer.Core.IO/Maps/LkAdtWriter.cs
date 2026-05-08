using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.Chunks;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

public static class LkAdtWriter
{
    private const int MhdrDataSize = 64;
    private const int McinEntryCount = 256;
    private const int McinEntrySize = 16;
    private const int McinDataSize = McinEntryCount * McinEntrySize;
    private const int MddfEntrySize = 36;
    private const int ModfEntrySize = 64;
    private const int McnkHeaderSize = 0x80;
    private const int McvtFloatCount = 145;
    private const int McnrByteCount = 448;
    private const int MclyEntrySize = 16;
    private const int ChunkHeaderSize = 8;
    private const float MapOrigin = 17066.666f;
    private const float ChunkSize = 533.33333f;
    private const float ChunkSubSize = ChunkSize / 16f;

    public static void Write(string path, LkAdtData adt)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(adt);

        string? dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dir))
            Directory.CreateDirectory(dir);

        File.WriteAllBytes(path, Build(adt));
    }

    public static byte[] Build(LkAdtData adt)
    {
        ArgumentNullException.ThrowIfNull(adt);

        byte[] mverData = BuildMver();
        byte[] mtexData = BuildStringTable(adt.TextureNames);
        byte[] mmdxData = BuildStringTable(adt.ModelNames);
        byte[] mmidData = BuildOffsetTable(adt.ModelNames);
        byte[] mwmoData = BuildStringTable(adt.WorldModelNames);
        byte[] mwidData = BuildOffsetTable(adt.WorldModelNames);
        byte[] mddfData = BuildMddf(adt.ModelPlacements);
        byte[] modfData = BuildModf(adt.WorldModelPlacements);

        var mcnkChunks = new List<byte[]>(256);
        for (int i = 0; i < McinEntryCount; i++)
            mcnkChunks.Add(i < adt.Chunks.Count ? BuildMcnk(adt.Chunks[i]) : []);

        byte[] mhdrData = new byte[MhdrDataSize];
        byte[] mcinData = new byte[McinDataSize];

        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms, Encoding.ASCII, leaveOpen: true);

        WriteChunk(bw, "MVER", mverData.Length, w => w.Write(mverData));
        long mhdrPosition = ms.Position;
        WriteChunk(bw, "MHDR", MhdrDataSize, w => w.Write(new byte[MhdrDataSize]));

        long mcinStart = ms.Position + ChunkHeaderSize;
        WriteChunk(bw, "MCIN", McinDataSize, w => w.Write(new byte[McinDataSize]));

        long currentOffset = ms.Position;

        WriteDataChunk(bw, "MTEX", mtexData);
        long mtexOffset = (int)(currentOffset - mhdrPosition - ChunkHeaderSize);
        currentOffset = ms.Position;

        WriteDataChunk(bw, "MMDX", mmdxData);
        long mmdxOffset = (int)(currentOffset - mhdrPosition - ChunkHeaderSize);
        currentOffset = ms.Position;

        WriteDataChunk(bw, "MMID", mmidData);
        long mmidOffset = (int)(currentOffset - mhdrPosition - ChunkHeaderSize);
        currentOffset = ms.Position;

        WriteDataChunk(bw, "MWMO", mwmoData);
        long mwmoOffset = (int)(currentOffset - mhdrPosition - ChunkHeaderSize);
        currentOffset = ms.Position;

        WriteDataChunk(bw, "MWID", mwidData);
        long mwidOffset = (int)(currentOffset - mhdrPosition - ChunkHeaderSize);
        currentOffset = ms.Position;

        WriteDataChunk(bw, "MDDF", mddfData);
        long mddfOffset = (int)(currentOffset - mhdrPosition - ChunkHeaderSize);
        currentOffset = ms.Position;

        WriteDataChunk(bw, "MODF", modfData);
        long modfOffset = (int)(currentOffset - mhdrPosition - ChunkHeaderSize);
        currentOffset = ms.Position;

        int mh2oRelativeOffset = 0;
        byte[]? mh2oData = null;

        int[] mcnkOffsets = new int[McinEntryCount];
        int[] mcnkSizes = new int[McinEntryCount];

        for (int i = 0; i < McinEntryCount; i++)
        {
            mcnkOffsets[i] = (int)ms.Position;
            mcnkSizes[i] = mcnkChunks[i].Length;
            if (mcnkChunks[i].Length > 0)
                bw.Write(mcnkChunks[i]);
            else
                mcnkOffsets[i] = 0;
        }

        byte[] result = ms.ToArray();

        PatchMhdr(result, (int)mhdrPosition, adt.MhdrFlags,
            (int)(mcinStart - mhdrPosition - ChunkHeaderSize),
            (int)(mtexOffset - ChunkHeaderSize),
            (int)(mmdxOffset - ChunkHeaderSize),
            (int)(mmidOffset - ChunkHeaderSize),
            (int)(mwmoOffset - ChunkHeaderSize),
            (int)(mwidOffset - ChunkHeaderSize),
            (int)(mddfOffset - ChunkHeaderSize),
            (int)(modfOffset - ChunkHeaderSize),
            mh2oRelativeOffset);

        PatchMcin(result, (int)mcinStart, mcnkOffsets, mcnkSizes);

        return result;
    }

    private static byte[] BuildMver()
    {
        byte[] data = new byte[4];
        BinaryPrimitives.WriteInt32LittleEndian(data, 18);
        return data;
    }

    private static byte[] BuildStringTable(IReadOnlyList<string> names)
    {
        if (names.Count == 0)
            return [];

        int totalSize = 0;
        foreach (var name in names)
            totalSize += Encoding.UTF8.GetByteCount(name) + 1;

        byte[] data = new byte[totalSize];
        int offset = 0;
        foreach (var name in names)
        {
            int written = Encoding.UTF8.GetBytes(name, data.AsSpan(offset));
            offset += written;
            data[offset++] = 0;
        }

        return data;
    }

    private static byte[] BuildOffsetTable(IReadOnlyList<string> names)
    {
        if (names.Count == 0)
            return [];

        byte[] data = new byte[names.Count * 4];
        int stringOffset = 0;
        for (int i = 0; i < names.Count; i++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(i * 4), (uint)stringOffset);
            stringOffset += Encoding.UTF8.GetByteCount(names[i]) + 1;
        }

        return data;
    }

    private static byte[] BuildMddf(IReadOnlyList<LkMddfEntry> placements)
    {
        if (placements.Count == 0)
            return [];

        byte[] data = new byte[placements.Count * MddfEntrySize];
        for (int i = 0; i < placements.Count; i++)
        {
            var p = placements[i];
            int off = i * MddfEntrySize;
            BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(off), p.NameId);
            BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(off + 4), p.UniqueId);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 8), MapOrigin - p.Position.Y);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 12), p.Position.Z);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 16), MapOrigin - p.Position.X);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 20), p.Rotation.X);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 24), p.Rotation.Z);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 28), p.Rotation.Y);
            BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(off + 32), (ushort)MathF.Round(p.Scale * 1024f));
            BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(off + 34), 0);
        }

        return data;
    }

    private static byte[] BuildModf(IReadOnlyList<LkModfEntry> placements)
    {
        if (placements.Count == 0)
            return [];

        byte[] data = new byte[placements.Count * ModfEntrySize];
        for (int i = 0; i < placements.Count; i++)
        {
            var p = placements[i];
            int off = i * ModfEntrySize;
            BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(off), p.NameId);
            BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(off + 4), p.UniqueId);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 8), MapOrigin - p.Position.Y);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 12), p.Position.Z);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 16), MapOrigin - p.Position.X);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 20), p.Rotation.X);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 24), p.Rotation.Z);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 28), p.Rotation.Y);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 32), MapOrigin - p.BoundsMax.Y);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 36), p.BoundsMax.Z);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 40), MapOrigin - p.BoundsMin.X);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 44), MapOrigin - p.BoundsMin.Y);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 48), p.BoundsMin.Z);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 52), MapOrigin - p.BoundsMax.X);
            BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(off + 56), p.Flags);
            BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(off + 58), p.DoodadSet);
            BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(off + 60), p.NameSet);
            BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(off + 62), (ushort)MathF.Round(p.Scale * 1024f));
        }

        return data;
    }

    private static byte[] BuildMcnk(LkMcnkData chunk)
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms, Encoding.ASCII, leaveOpen: true);

        bw.Write(FourCC.FromString("MCNK").ToFileBytes());
        long sizePosition = ms.Position;
        bw.Write(0);

        byte[] header = new byte[McnkHeaderSize];
        PatchMcnkHeader(header, chunk, 0);
        bw.Write(header);

        long afterHeader = ms.Position;
        int subOffset = (int)(McnkHeaderSize + ChunkHeaderSize);

        int mcvtOffset = subOffset;
        bw.Write(FourCC.FromString("MCVT").ToFileBytes());
        bw.Write(McvtFloatCount * 4);
        for (int i = 0; i < McvtFloatCount; i++)
        {
            float raw = i < chunk.Heights.Length ? chunk.Heights[i] - chunk.BaseHeight : 0f;
            bw.Write(raw);
        }

        int mcnrOffset = (int)(ms.Position - afterHeader + McnkHeaderSize);
        bw.Write(FourCC.FromString("MCNR").ToFileBytes());
        bw.Write(McnrByteCount);
        byte[] normals = chunk.Normals;
        int normalBytes = Math.Min(normals.Length, McnrByteCount);
        bw.Write(normals.AsSpan(0, normalBytes));
        if (McnrByteCount > normalBytes)
            bw.Write(new byte[McnrByteCount - normalBytes]);

        int mclyOffset = 0, mcalOffset = 0, mcalSize = 0;
        if (chunk.Layers.Count > 0)
        {
            mclyOffset = (int)(ms.Position - afterHeader + McnkHeaderSize);

            byte[] mclyPayload = BuildMclyPayload(chunk.Layers);
            bw.Write(FourCC.FromString("MCLY").ToFileBytes());
            bw.Write(mclyPayload.Length);
            bw.Write(mclyPayload);

            if (chunk.AlphaMapData is { Length: > 0 })
            {
                mcalOffset = (int)(ms.Position - afterHeader + McnkHeaderSize);
                mcalSize = chunk.AlphaMapData.Length + ChunkHeaderSize;

                bw.Write(FourCC.FromString("MCAL").ToFileBytes());
                bw.Write(chunk.AlphaMapData.Length);
                bw.Write(chunk.AlphaMapData);
            }
        }

        int mcrfOffset = 0;
        if (chunk.DoodadRefs.Count > 0 || chunk.WorldModelRefs.Count > 0)
        {
            mcrfOffset = (int)(ms.Position - afterHeader + McnkHeaderSize);
            int mcrfSize = 4 + chunk.DoodadRefs.Count * 4 + chunk.WorldModelRefs.Count * 4;
            bw.Write(FourCC.FromString("MCRF").ToFileBytes());
            bw.Write(mcrfSize);
            bw.Write(chunk.DoodadRefs.Count);
            bw.Write(chunk.WorldModelRefs.Count);
            foreach (int r in chunk.DoodadRefs)
                bw.Write(r);
            foreach (int r in chunk.WorldModelRefs)
                bw.Write(r);
        }

        int mcshOffset = 0, mcshSize = 0;
        if (chunk.ShadowMap is { Length: > 0 })
        {
            mcshOffset = (int)(ms.Position - afterHeader + McnkHeaderSize);
            mcshSize = chunk.ShadowMap.Length + ChunkHeaderSize;

            bw.Write(FourCC.FromString("MCSH").ToFileBytes());
            bw.Write(chunk.ShadowMap.Length);
            bw.Write(chunk.ShadowMap);
        }

        int totalSize = (int)ms.Position - 8;
        ms.Position = sizePosition;
        bw.Write(totalSize);

        ms.Position = ChunkHeaderSize;
        PatchMcnkHeader(ms.GetBuffer().AsSpan(ChunkHeaderSize, McnkHeaderSize), chunk, 0);
        BinaryPrimitives.WriteInt32LittleEndian(ms.GetBuffer().AsSpan((int)ChunkHeaderSize + 0x14), mcvtOffset);
        BinaryPrimitives.WriteInt32LittleEndian(ms.GetBuffer().AsSpan((int)ChunkHeaderSize + 0x18), mcnrOffset);
        BinaryPrimitives.WriteInt32LittleEndian(ms.GetBuffer().AsSpan((int)ChunkHeaderSize + 0x1C), mclyOffset);
        BinaryPrimitives.WriteInt32LittleEndian(ms.GetBuffer().AsSpan((int)ChunkHeaderSize + 0x20), mcrfOffset);
        BinaryPrimitives.WriteInt32LittleEndian(ms.GetBuffer().AsSpan((int)ChunkHeaderSize + 0x24), mcalOffset);
        BinaryPrimitives.WriteInt32LittleEndian(ms.GetBuffer().AsSpan((int)ChunkHeaderSize + 0x28), mcalSize);
        BinaryPrimitives.WriteInt32LittleEndian(ms.GetBuffer().AsSpan((int)ChunkHeaderSize + 0x2C), mcshOffset);
        BinaryPrimitives.WriteInt32LittleEndian(ms.GetBuffer().AsSpan((int)ChunkHeaderSize + 0x30), mcshSize);

        ms.Position = ms.Length;
        return ms.ToArray();
    }

    private static void PatchMcnkHeader(Span<byte> header, LkMcnkData chunk, int baseOffset)
    {
        BinaryPrimitives.WriteInt32LittleEndian(header.Slice(0x00), chunk.Flags);
        BinaryPrimitives.WriteInt32LittleEndian(header.Slice(0x04), chunk.IndexX);
        BinaryPrimitives.WriteInt32LittleEndian(header.Slice(0x08), chunk.IndexY);
        BinaryPrimitives.WriteInt32LittleEndian(header.Slice(0x0C), chunk.NLayers);
        BinaryPrimitives.WriteInt32LittleEndian(header.Slice(0x10), chunk.DoodadRefs.Count);
        BinaryPrimitives.WriteSingleLittleEndian(header.Slice(0x68), chunk.PosX);
        BinaryPrimitives.WriteSingleLittleEndian(header.Slice(0x6C), chunk.PosY);
        BinaryPrimitives.WriteSingleLittleEndian(header.Slice(0x70), chunk.BaseHeight);
        BinaryPrimitives.WriteInt32LittleEndian(header.Slice(0x34), chunk.AreaId);
        BinaryPrimitives.WriteInt32LittleEndian(header.Slice(0x38), chunk.WorldModelRefs.Count);
        BinaryPrimitives.WriteInt32LittleEndian(header.Slice(0x3C), chunk.HoleMask);
    }

    private static byte[] BuildMclyPayload(IReadOnlyList<LkMclyEntry> layers)
    {
        byte[] data = new byte[layers.Count * MclyEntrySize];
        for (int i = 0; i < layers.Count; i++)
        {
            var layer = layers[i];
            int off = i * MclyEntrySize;
            BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(off), layer.TextureId);
            BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(off + 4), layer.Flags);
            BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(off + 8), layer.AlphaOffset);
            BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(off + 12), layer.EffectId);
        }
        return data;
    }

    private static void PatchMhdr(byte[] result, int mhdrStart, uint flags,
        int ofsMcin, int ofsMtex, int ofsMmdx, int ofsMmid,
        int ofsMwmo, int ofsMwid, int ofsMddf, int ofsModf,
        int ofsMh2o)
    {
        int dataStart = mhdrStart + ChunkHeaderSize;
        BinaryPrimitives.WriteUInt32LittleEndian(result.AsSpan(dataStart + 0), flags);
        BinaryPrimitives.WriteInt32LittleEndian(result.AsSpan(dataStart + 4), ofsMcin);
        BinaryPrimitives.WriteInt32LittleEndian(result.AsSpan(dataStart + 8), ofsMtex);
        BinaryPrimitives.WriteInt32LittleEndian(result.AsSpan(dataStart + 12), ofsMmdx);
        BinaryPrimitives.WriteInt32LittleEndian(result.AsSpan(dataStart + 16), ofsMmid);
        BinaryPrimitives.WriteInt32LittleEndian(result.AsSpan(dataStart + 20), ofsMwmo);
        BinaryPrimitives.WriteInt32LittleEndian(result.AsSpan(dataStart + 24), ofsMwid);
        BinaryPrimitives.WriteInt32LittleEndian(result.AsSpan(dataStart + 28), ofsMddf);
        BinaryPrimitives.WriteInt32LittleEndian(result.AsSpan(dataStart + 32), ofsModf);
        BinaryPrimitives.WriteInt32LittleEndian(result.AsSpan(dataStart + 36), 0);
        BinaryPrimitives.WriteInt32LittleEndian(result.AsSpan(dataStart + 40), ofsMh2o);
        BinaryPrimitives.WriteInt32LittleEndian(result.AsSpan(dataStart + 44), 0);
        BinaryPrimitives.WriteInt32LittleEndian(result.AsSpan(dataStart + 48), 0);
        for (int i = 48; i < MhdrDataSize; i += 4)
            BinaryPrimitives.WriteInt32LittleEndian(result.AsSpan(dataStart + i), 0);
    }

    private static void PatchMcin(byte[] result, int mcinDataStart, int[] mcnkOffsets, int[] mcnkSizes)
    {
        for (int i = 0; i < McinEntryCount; i++)
        {
            int off = mcinDataStart + i * McinEntrySize;
            BinaryPrimitives.WriteInt32LittleEndian(result.AsSpan(off), mcnkOffsets[i]);
            BinaryPrimitives.WriteInt32LittleEndian(result.AsSpan(off + 4), mcnkSizes[i]);
            BinaryPrimitives.WriteInt32LittleEndian(result.AsSpan(off + 8), 0);
            BinaryPrimitives.WriteInt32LittleEndian(result.AsSpan(off + 12), 0);
        }
    }

    private static void WriteChunk(BinaryWriter bw, string tag, int dataSize, Action<BinaryWriter> writePayload)
    {
        bw.Write(FourCC.FromString(tag).ToFileBytes());
        bw.Write(dataSize);
        writePayload(bw);
    }

    private static void WriteDataChunk(BinaryWriter bw, string tag, byte[] payload)
    {
        bw.Write(FourCC.FromString(tag).ToFileBytes());
        bw.Write(payload.Length);
        bw.Write(payload);
        if ((payload.Length & 1) != 0)
            bw.Write((byte)0);
    }
}