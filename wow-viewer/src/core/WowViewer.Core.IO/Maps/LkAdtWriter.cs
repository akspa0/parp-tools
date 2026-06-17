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

        long mcinStart = ms.Position;
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

        byte[] mh2oData = BuildMh2oPayload(adt.Chunks);
        int mh2oRelativeOffset = 0;

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

        if (mh2oData.Length > 0)
        {
            mh2oRelativeOffset = (int)(ms.Position - mhdrPosition);
            WriteDataChunk(bw, "MH2O", mh2oData);
        }

        byte[] mfboData = BuildMfboPayload(adt.MfboFlightBounds);
        if (mfboData.Length > 0)
        {
            WriteDataChunk(bw, "MFBO", mfboData);
        }

        byte[] result = ms.ToArray();

        PatchMhdr(result, (int)mhdrPosition, adt.MhdrFlags,
            (int)(mcinStart - mhdrPosition - ChunkHeaderSize),
            (int)mtexOffset,
            (int)mmdxOffset,
            (int)mmidOffset,
            (int)mwmoOffset,
            (int)mwidOffset,
            (int)mddfOffset,
            (int)modfOffset,
            mh2oRelativeOffset);

        PatchMcin(result, (int)mcinStart + ChunkHeaderSize, mcnkOffsets, mcnkSizes);

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
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 36), p.BoundsMin.Z);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 40), MapOrigin - p.BoundsMax.X);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 44), MapOrigin - p.BoundsMin.Y);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 48), p.BoundsMax.Z);
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 52), MapOrigin - p.BoundsMin.X);
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
        int mcvtOffset = (int)ms.Position;
        bw.Write(FourCC.FromString("MCVT").ToFileBytes());
        bw.Write(McvtFloatCount * 4);
        for (int i = 0; i < McvtFloatCount; i++)
        {
            float raw = i < chunk.Heights.Length ? chunk.Heights[i] : 0f;
            bw.Write(raw);
        }

        int mcnrOffset = (int)ms.Position;
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
            mclyOffset = (int)ms.Position;

            byte[] mclyPayload = BuildMclyPayload(chunk.Layers);
            bw.Write(FourCC.FromString("MCLY").ToFileBytes());
            bw.Write(mclyPayload.Length);
            bw.Write(mclyPayload);

            if (chunk.AlphaMapData is { Length: > 0 })
            {
                mcalOffset = (int)ms.Position;
                mcalSize = chunk.AlphaMapData.Length + ChunkHeaderSize;

                bw.Write(FourCC.FromString("MCAL").ToFileBytes());
                bw.Write(chunk.AlphaMapData.Length);
                bw.Write(chunk.AlphaMapData);
            }
        }

        int mcrfOffset = 0;
        if (chunk.DoodadRefs.Count > 0 || chunk.WorldModelRefs.Count > 0)
        {
            mcrfOffset = (int)ms.Position;
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
            mcshOffset = (int)ms.Position;
            mcshSize = chunk.ShadowMap.Length + ChunkHeaderSize;

            bw.Write(FourCC.FromString("MCSH").ToFileBytes());
            bw.Write(chunk.ShadowMap.Length);
            bw.Write(chunk.ShadowMap);
        }

        int mccvOffset = 0;
        if (chunk.MccvColors is { Length: >= 580 })
        {
            mccvOffset = (int)ms.Position;
            bw.Write(FourCC.FromString("MCCV").ToFileBytes());
            bw.Write(chunk.MccvColors.Length);
            bw.Write(chunk.MccvColors);
        }

        int mclvOffset = 0;
        if (chunk.MclvLighting is { Length: >= 580 })
        {
            mclvOffset = (int)ms.Position;
            bw.Write(FourCC.FromString("MCLV").ToFileBytes());
            bw.Write(chunk.MclvLighting.Length);
            bw.Write(chunk.MclvLighting);
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

    private static byte[] BuildMh2oPayload(IReadOnlyList<LkMcnkData> chunks)
    {
        const int HeaderCount = 256;
        const int HeaderSize = 12;
        const int LayerInfoSize = 24;
        const int AttributeSize = 16;

        using var ms = new MemoryStream();
        ms.Write(new byte[HeaderCount * HeaderSize]);
        bool hasAnyLayers = false;

        for (int chunkIndex = 0; chunkIndex < HeaderCount; chunkIndex++)
        {
            AdtLiquidChunk? liquidChunk = chunkIndex < chunks.Count ? chunks[chunkIndex].LiquidData : null;
            if (liquidChunk is null || liquidChunk.Layers.Count == 0)
                continue;

            hasAnyLayers = true;
            int infoOffset = (int)ms.Position;
            ms.Write(new byte[liquidChunk.Layers.Count * LayerInfoSize]);

            int attributesOffset = 0;
            if (liquidChunk.FishableMask.HasValue || liquidChunk.DeepMask.HasValue)
            {
                attributesOffset = (int)ms.Position;
                byte[] attributes = new byte[AttributeSize];
                BinaryPrimitives.WriteUInt64LittleEndian(attributes.AsSpan(0, 8), liquidChunk.FishableMask ?? 0);
                BinaryPrimitives.WriteUInt64LittleEndian(attributes.AsSpan(8, 8), liquidChunk.DeepMask ?? 0);
                ms.Write(attributes, 0, attributes.Length);
            }

            for (int layerIndex = 0; layerIndex < liquidChunk.Layers.Count; layerIndex++)
            {
                AdtLiquidLayer layer = liquidChunk.Layers[layerIndex];
                int layerInfoOffset = infoOffset + (layerIndex * LayerInfoSize);

                int existsBitmapOffset = 0;
                if (layer.ExistsBitmap is { Length: > 0 })
                {
                    existsBitmapOffset = (int)ms.Position;
                    ms.Write(layer.ExistsBitmap, 0, layer.ExistsBitmap.Length);
                }

                int vertexCount = Math.Max(0, (layer.Width + 1) * (layer.Height + 1));
                int vertexDataOffset = 0;
                if (vertexCount > 0)
                {
                    vertexDataOffset = (int)ms.Position;
                    WriteLiquidVertexData(ms, layer, vertexCount);
                }

                long resume = ms.Position;
                ms.Position = layerInfoOffset;

                Span<byte> layerInfo = stackalloc byte[LayerInfoSize];
                BinaryPrimitives.WriteUInt16LittleEndian(layerInfo.Slice(0, 2), layer.LiquidTypeId);
                BinaryPrimitives.WriteUInt16LittleEndian(layerInfo.Slice(2, 2), (ushort)layer.VertexFormat);
                BinaryPrimitives.WriteSingleLittleEndian(layerInfo.Slice(4, 4), layer.MinHeight);
                BinaryPrimitives.WriteSingleLittleEndian(layerInfo.Slice(8, 4), layer.MaxHeight);
                layerInfo[12] = (byte)layer.XOffset;
                layerInfo[13] = (byte)layer.YOffset;
                layerInfo[14] = (byte)layer.Width;
                layerInfo[15] = (byte)layer.Height;
                BinaryPrimitives.WriteUInt32LittleEndian(layerInfo.Slice(16, 4), (uint)existsBitmapOffset);
                BinaryPrimitives.WriteUInt32LittleEndian(layerInfo.Slice(20, 4), (uint)vertexDataOffset);
                ms.Write(layerInfo);

                ms.Position = resume;
            }

            long resumeHeader = ms.Position;
            ms.Position = chunkIndex * HeaderSize;

            Span<byte> header = stackalloc byte[HeaderSize];
            BinaryPrimitives.WriteUInt32LittleEndian(header.Slice(0, 4), (uint)infoOffset);
            BinaryPrimitives.WriteUInt32LittleEndian(header.Slice(4, 4), (uint)liquidChunk.Layers.Count);
            BinaryPrimitives.WriteUInt32LittleEndian(header.Slice(8, 4), (uint)attributesOffset);
            ms.Write(header);

            ms.Position = resumeHeader;
        }

        return hasAnyLayers ? ms.ToArray() : [];
    }

    private static void WriteLiquidVertexData(Stream stream, AdtLiquidLayer layer, int vertexCount)
    {
        float[] heights = layer.Heights is { Length: > 0 }
            ? layer.Heights
            : CreateDefaultLiquidHeights(vertexCount, (layer.MinHeight + layer.MaxHeight) * 0.5f);
        byte[] depths = layer.Depths ?? new byte[vertexCount];
        ushort[] uvs = layer.Uvs ?? new ushort[vertexCount * 2];

        switch (layer.VertexFormat)
        {
            case AdtLiquidVertexFormat.HeightDepth:
                WriteSingles(stream, heights, vertexCount);
                stream.Write(depths, 0, Math.Min(depths.Length, vertexCount));
                if (depths.Length < vertexCount)
                    stream.Write(new byte[vertexCount - depths.Length], 0, vertexCount - depths.Length);
                break;

            case AdtLiquidVertexFormat.HeightUv:
                WriteSingles(stream, heights, vertexCount);
                WriteUInt16s(stream, uvs, vertexCount * 2);
                break;

            case AdtLiquidVertexFormat.DepthOnly:
                stream.Write(depths, 0, Math.Min(depths.Length, vertexCount));
                if (depths.Length < vertexCount)
                    stream.Write(new byte[vertexCount - depths.Length], 0, vertexCount - depths.Length);
                break;

            case AdtLiquidVertexFormat.HeightUvDepth:
                WriteSingles(stream, heights, vertexCount);
                WriteUInt16s(stream, uvs, vertexCount * 2);
                stream.Write(depths, 0, Math.Min(depths.Length, vertexCount));
                if (depths.Length < vertexCount)
                    stream.Write(new byte[vertexCount - depths.Length], 0, vertexCount - depths.Length);
                break;
        }
    }

    private static void WriteSingles(Stream stream, float[] values, int count)
    {
        for (int index = 0; index < count; index++)
            WriteSingle(stream, values, index);
    }

    private static void WriteUInt16s(Stream stream, ushort[] values, int count)
    {
        for (int index = 0; index < count; index++)
            WriteUInt16(stream, values, index);
    }

    private static void WriteSingle(Stream stream, float[] values, int index)
    {
        Span<byte> bytes = stackalloc byte[4];
        BinaryPrimitives.WriteSingleLittleEndian(bytes, index < values.Length ? values[index] : 0f);
        stream.Write(bytes);
    }

    private static void WriteUInt16(Stream stream, ushort[] values, int index)
    {
        Span<byte> bytes = stackalloc byte[2];
        BinaryPrimitives.WriteUInt16LittleEndian(bytes, index < values.Length ? values[index] : (ushort)0);
        stream.Write(bytes);
    }

    private static float[] CreateDefaultLiquidHeights(int vertexCount, float height)
    {
        float[] values = new float[vertexCount];
        Array.Fill(values, height);
        return values;
    }

    private static byte[] BuildMfboPayload(int[,,]? flightBounds)
    {
        if (flightBounds is null)
            return [];

        byte[] data = new byte[36];
        for (int plane = 0; plane < 2; plane++)
        {
            for (int row = 0; row < 3; row++)
            {
                for (int col = 0; col < 3; col++)
                {
                    int offset = (((plane * 3) + row) * 3 + col) * 2;
                    short value = (short)Math.Clamp(flightBounds[plane, row, col], short.MinValue, short.MaxValue);
                    BinaryPrimitives.WriteInt16LittleEndian(data.AsSpan(offset), value);
                }
            }
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
