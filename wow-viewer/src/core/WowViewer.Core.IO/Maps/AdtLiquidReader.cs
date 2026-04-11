using System.Buffers.Binary;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

public static class AdtLiquidReader
{
    private const int ChunkCount = 256;
    private const int ChunkHeaderSize = 12;
    private const int LayerSize = 24;

    public static AdtLiquidFile Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, Path.GetFullPath(path));
        return Read(stream, fileSummary);
    }

    public static AdtLiquidFile Read(Stream stream, MapFileSummary fileSummary)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentNullException.ThrowIfNull(fileSummary);

        if (fileSummary.Kind != MapFileKind.Adt)
            throw new InvalidDataException($"ADT liquid reader requires a root ADT file, but found {fileSummary.Kind}.");

        byte[]? payload = MapSummaryReaderCommon.ReadChunkPayload(stream, fileSummary, MapChunkIds.Mh2o);
        if (payload is null)
            return CreateEmpty(fileSummary.SourcePath, fileSummary.Kind);

        return Parse(fileSummary.SourcePath, fileSummary.Kind, payload);
    }

    private static AdtLiquidFile Parse(string sourcePath, MapFileKind kind, byte[] payload)
    {
        if (payload.Length < ChunkCount * ChunkHeaderSize)
            throw new InvalidDataException($"MH2O payload is too small to contain {ChunkCount} chunk headers.");

        List<AdtLiquidChunk> chunks = new(ChunkCount);
        for (int chunkIndex = 0; chunkIndex < ChunkCount; chunkIndex++)
        {
            int headerOffset = chunkIndex * ChunkHeaderSize;
            uint offsetInstances = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(headerOffset, 4));
            uint layerCount = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(headerOffset + 4, 4));
            uint offsetAttributes = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(headerOffset + 8, 4));

            ulong? fishableMask = null;
            ulong? deepMask = null;
            if (offsetAttributes > 0 && offsetAttributes + 16 <= payload.Length)
            {
                int attributeOffset = checked((int)offsetAttributes);
                fishableMask = BinaryPrimitives.ReadUInt64LittleEndian(payload.AsSpan(attributeOffset, 8));
                deepMask = BinaryPrimitives.ReadUInt64LittleEndian(payload.AsSpan(attributeOffset + 8, 8));
            }

            List<AdtLiquidLayer> layers = [];
            if (layerCount > 0 && offsetInstances > 0 && offsetInstances + LayerSize <= payload.Length)
            {
                int layerOffset = checked((int)offsetInstances);
                for (int layerIndex = 0; layerIndex < layerCount && layerOffset + LayerSize <= payload.Length; layerIndex++)
                {
                    layers.Add(ParseLayer(payload, layerOffset));
                    layerOffset += LayerSize;
                }
            }

            chunks.Add(new AdtLiquidChunk(chunkIndex, fishableMask, deepMask, layers));
        }

        return new AdtLiquidFile(sourcePath, kind, chunks);
    }

    private static AdtLiquidLayer ParseLayer(byte[] payload, int offset)
    {
        ushort liquidTypeId = BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(offset, 2));
        AdtLiquidVertexFormat vertexFormat = (AdtLiquidVertexFormat)BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(offset + 2, 2));
        float minHeight = BitConverter.ToSingle(payload, offset + 4);
        float maxHeight = BitConverter.ToSingle(payload, offset + 8);
        int xOffset = payload[offset + 12];
        int yOffset = payload[offset + 13];
        int width = payload[offset + 14];
        int height = payload[offset + 15];
        uint offsetExistsBitmap = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(offset + 16, 4));
        uint offsetVertexData = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(offset + 20, 4));

        int clampedWidth = Math.Clamp(width, 0, 8);
        int clampedHeight = Math.Clamp(height, 0, 8);
        int vertexCount = (clampedWidth + 1) * (clampedHeight + 1);

        byte[]? existsBitmap = null;
        int bitmapSize = ((clampedWidth * clampedHeight) + 7) / 8;
        if (bitmapSize > 0 && offsetExistsBitmap > 0 && offsetExistsBitmap + bitmapSize <= payload.Length)
        {
            existsBitmap = new byte[bitmapSize];
            Buffer.BlockCopy(payload, checked((int)offsetExistsBitmap), existsBitmap, 0, bitmapSize);
        }

        float[]? heights = null;
        byte[]? depths = null;
        ushort[]? uvs = null;

        if (vertexCount > 0 && offsetVertexData > 0 && offsetVertexData < payload.Length)
        {
            int vertexDataOffset = checked((int)offsetVertexData);
            switch (vertexFormat)
            {
                case AdtLiquidVertexFormat.HeightDepth:
                    heights = ReadFloatArray(payload, vertexDataOffset, vertexCount, out vertexDataOffset);
                    depths = ReadByteArray(payload, vertexDataOffset, vertexCount);
                    break;

                case AdtLiquidVertexFormat.HeightUv:
                    heights = ReadFloatArray(payload, vertexDataOffset, vertexCount, out vertexDataOffset);
                    uvs = ReadUInt16Array(payload, vertexDataOffset, vertexCount * 2);
                    break;

                case AdtLiquidVertexFormat.DepthOnly:
                    depths = ReadByteArray(payload, vertexDataOffset, vertexCount);
                    break;

                case AdtLiquidVertexFormat.HeightUvDepth:
                    heights = ReadFloatArray(payload, vertexDataOffset, vertexCount, out vertexDataOffset);
                    uvs = ReadUInt16Array(payload, vertexDataOffset, vertexCount * 2, out vertexDataOffset);
                    depths = ReadByteArray(payload, vertexDataOffset, vertexCount);
                    break;
            }
        }

        return new AdtLiquidLayer(
            liquidTypeId,
            MapLiquidTypeId(liquidTypeId),
            vertexFormat,
            minHeight,
            maxHeight,
            xOffset,
            yOffset,
            clampedWidth,
            clampedHeight,
            existsBitmap,
            heights,
            depths,
            uvs);
    }

    private static float[]? ReadFloatArray(byte[] payload, int offset, int count, out int nextOffset)
    {
        nextOffset = offset;
        if (count <= 0 || offset < 0 || offset + (count * sizeof(float)) > payload.Length)
            return null;

        float[] values = new float[count];
        for (int index = 0; index < count; index++)
            values[index] = BitConverter.ToSingle(payload, offset + (index * sizeof(float)));

        nextOffset = offset + (count * sizeof(float));
        return values;
    }

    private static ushort[]? ReadUInt16Array(byte[] payload, int offset, int count)
    {
        return ReadUInt16Array(payload, offset, count, out _);
    }

    private static ushort[]? ReadUInt16Array(byte[] payload, int offset, int count, out int nextOffset)
    {
        nextOffset = offset;
        if (count <= 0 || offset < 0 || offset + (count * sizeof(ushort)) > payload.Length)
            return null;

        ushort[] values = new ushort[count];
        for (int index = 0; index < count; index++)
            values[index] = BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(offset + (index * sizeof(ushort)), sizeof(ushort)));

        nextOffset = offset + (count * sizeof(ushort));
        return values;
    }

    private static byte[]? ReadByteArray(byte[] payload, int offset, int count)
    {
        if (count <= 0 || offset < 0 || offset + count > payload.Length)
            return null;

        byte[] values = new byte[count];
        Buffer.BlockCopy(payload, offset, values, 0, count);
        return values;
    }

    private static AdtLiquidFile CreateEmpty(string sourcePath, MapFileKind kind)
    {
        List<AdtLiquidChunk> chunks = new(ChunkCount);
        for (int chunkIndex = 0; chunkIndex < ChunkCount; chunkIndex++)
            chunks.Add(new AdtLiquidChunk(chunkIndex, null, null, Array.Empty<AdtLiquidLayer>()));

        return new AdtLiquidFile(sourcePath, kind, chunks);
    }

    private static AdtLiquidBasicType MapLiquidTypeId(ushort liquidTypeId)
    {
        return liquidTypeId switch
        {
            17 => AdtLiquidBasicType.Ocean,
            19 => AdtLiquidBasicType.Magma,
            20 => AdtLiquidBasicType.Slime,
            _ => AdtLiquidBasicType.Water,
        };
    }
}