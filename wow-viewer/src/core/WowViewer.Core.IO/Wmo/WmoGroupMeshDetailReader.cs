using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoGroupMeshDetailReader
{
    private const int Vector3Stride = 12;
    private const int UvStride = 8;
    private const int ColorStride = 4;
    private const int BatchEntrySize = 24;
    private const int IndexStride = 2;

    public static WmoGroupMeshDetail Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoGroupMeshDetail Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        (uint? version, byte[] mogp) = WmoGroupReaderCommon.ReadGroupPayload(stream, sourcePath);
        return ReadMogpPayload(mogp, sourcePath, version);
    }

    internal static WmoGroupMeshDetail ReadMogpPayload(byte[] mogp, string sourcePath, uint? version)
    {
        ArgumentNullException.ThrowIfNull(mogp);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        int headerSizeBytes = WmoGroupReaderCommon.FindHeaderSize(mogp);
        List<Vector3> vertices = ReadVector3Payload(WmoGroupReaderCommon.TryReadFirstSubchunkPayload(mogp, headerSizeBytes, WmoChunkIds.Movt), Vector3Stride, "MOVT");
        List<Vector3> normals = ReadVector3Payload(WmoGroupReaderCommon.TryReadFirstSubchunkPayload(mogp, headerSizeBytes, WmoChunkIds.Monr), Vector3Stride, "MONR");
        (string? indexChunkId, List<ushort> indices) = ReadIndices(mogp, headerSizeBytes);
        (List<Vector2> primaryUvs, List<IReadOnlyList<Vector2>> additionalUvSets) = ReadUvSets(mogp, headerSizeBytes);
        (List<uint> primaryVertexColors, List<IReadOnlyList<uint>> additionalVertexColorSets) = ReadVertexColorSets(mogp, headerSizeBytes);
        List<WmoGroupFaceMaterialDetail> faceMaterials = ReadFaceMaterials(mogp, headerSizeBytes, version);
        List<WmoGroupBatchDetail> batches = ReadBatches(mogp, headerSizeBytes, version);

        return new WmoGroupMeshDetail(
            sourcePath,
            version,
            headerSizeBytes,
            indexChunkId,
            vertices,
            normals,
            indices,
            primaryUvs,
            additionalUvSets,
            primaryVertexColors,
            additionalVertexColorSets,
            faceMaterials,
            batches);
    }

    private static List<Vector3> ReadVector3Payload(byte[]? payload, int stride, string chunkId)
    {
        if (payload is null)
            return [];

        if (payload.Length % stride != 0)
            throw new InvalidDataException($"{chunkId} payload size {payload.Length} is not divisible by {stride}.");

        int count = payload.Length / stride;
        List<Vector3> values = new(count);
        for (int index = 0; index < count; index++)
        {
            int offset = index * stride;
            values.Add(new Vector3(
                BitConverter.ToSingle(payload, offset),
                BitConverter.ToSingle(payload, offset + 4),
                BitConverter.ToSingle(payload, offset + 8)));
        }

        return values;
    }

    private static (string? IndexChunkId, List<ushort> Indices) ReadIndices(byte[] mogp, int headerSizeBytes)
    {
        foreach ((ChunkHeader header, int dataOffset) in WmoGroupReaderCommon.EnumerateSubchunks(mogp, headerSizeBytes))
        {
            if (header.Id != WmoChunkIds.Movi && header.Id != WmoChunkIds.Moin)
                continue;

            int payloadSize = checked((int)header.Size);
            if (payloadSize % IndexStride != 0)
                throw new InvalidDataException($"{header.Id} payload size {payloadSize} is not divisible by {IndexStride}.");

            List<ushort> indices = new(payloadSize / IndexStride);
            ReadOnlySpan<byte> payload = mogp.AsSpan(dataOffset, payloadSize);
            for (int index = 0; index < payloadSize; index += IndexStride)
                indices.Add(BinaryPrimitives.ReadUInt16LittleEndian(payload.Slice(index, IndexStride)));

            return (header.Id.ToString(), indices);
        }

        return (null, []);
    }

    private static (List<Vector2> Primary, List<IReadOnlyList<Vector2>> Additional) ReadUvSets(byte[] mogp, int headerSizeBytes)
    {
        List<Vector2> primary = [];
        List<IReadOnlyList<Vector2>> additional = [];
        bool foundPrimary = false;

        foreach ((ChunkHeader header, int dataOffset) in WmoGroupReaderCommon.EnumerateSubchunks(mogp, headerSizeBytes))
        {
            if (header.Id != WmoChunkIds.Motv)
                continue;

            int payloadSize = checked((int)header.Size);
            if (payloadSize % UvStride != 0)
                throw new InvalidDataException($"MOTV payload size {payloadSize} is not divisible by {UvStride}.");

            List<Vector2> values = new(payloadSize / UvStride);
            for (int offset = 0; offset < payloadSize; offset += UvStride)
            {
                values.Add(new Vector2(
                    BitConverter.ToSingle(mogp, dataOffset + offset),
                    BitConverter.ToSingle(mogp, dataOffset + offset + 4)));
            }

            if (!foundPrimary)
            {
                primary = values;
                foundPrimary = true;
            }
            else
            {
                additional.Add(values);
            }
        }

        return (primary, additional);
    }

    private static (List<uint> Primary, List<IReadOnlyList<uint>> Additional) ReadVertexColorSets(byte[] mogp, int headerSizeBytes)
    {
        List<uint> primary = [];
        List<IReadOnlyList<uint>> additional = [];
        bool foundPrimary = false;

        foreach ((ChunkHeader header, int dataOffset) in WmoGroupReaderCommon.EnumerateSubchunks(mogp, headerSizeBytes))
        {
            if (header.Id != WmoChunkIds.Mocv)
                continue;

            int payloadSize = checked((int)header.Size);
            if (payloadSize % ColorStride != 0)
                throw new InvalidDataException($"MOCV payload size {payloadSize} is not divisible by {ColorStride}.");

            List<uint> values = new(payloadSize / ColorStride);
            for (int offset = 0; offset < payloadSize; offset += ColorStride)
            {
                uint packed = BinaryPrimitives.ReadUInt32LittleEndian(mogp.AsSpan(dataOffset + offset, ColorStride));
                values.Add(packed);
            }

            if (!foundPrimary)
            {
                primary = values;
                foundPrimary = true;
            }
            else
            {
                additional.Add(values);
            }
        }

        return (primary, additional);
    }

    private static List<WmoGroupFaceMaterialDetail> ReadFaceMaterials(byte[] mogp, int headerSizeBytes, uint? version)
    {
        byte[]? payload = WmoGroupReaderCommon.TryReadFirstSubchunkPayload(mogp, headerSizeBytes, WmoChunkIds.Mopy);
        if (payload is null)
            return [];

        int entrySizeBytes = WmoGroupReaderCommon.InferMopyEntrySize(payload.Length, version);
        if (entrySizeBytes <= 0 || payload.Length % entrySizeBytes != 0)
            throw new InvalidDataException($"MOPY payload size {payload.Length} is not compatible with inferred entry size {entrySizeBytes}.");

        int faceCount = payload.Length / entrySizeBytes;
        List<WmoGroupFaceMaterialDetail> details = new(faceCount);
        for (int faceIndex = 0; faceIndex < faceCount; faceIndex++)
        {
            int offset = faceIndex * entrySizeBytes;
            ushort? legacyExtraValue = entrySizeBytes >= 4
                ? BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(offset + 2, 2))
                : null;
            details.Add(new WmoGroupFaceMaterialDetail(faceIndex, payload[offset], payload[offset + 1], legacyExtraValue));
        }

        return details;
    }

    private static List<WmoGroupBatchDetail> ReadBatches(byte[] mogp, int headerSizeBytes, uint? version)
    {
        byte[]? payload = WmoGroupReaderCommon.TryReadFirstSubchunkPayload(mogp, headerSizeBytes, WmoChunkIds.Moba);
        if (payload is null)
            return [];

        if (payload.Length % BatchEntrySize != 0)
            throw new InvalidDataException($"MOBA payload size {payload.Length} is not divisible by {BatchEntrySize}.");

        int entryCount = payload.Length / BatchEntrySize;
        List<WmoGroupBatchDetail> batches = new(entryCount);
        for (int batchIndex = 0; batchIndex < entryCount; batchIndex++)
        {
            int offset = batchIndex * BatchEntrySize;
            byte materialIdRaw = payload[offset + 1];
            bool hasMaterialId = materialIdRaw != byte.MaxValue || version != 16;
            ushort firstIndex = BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(offset + 14, 2));
            ushort indexCount = BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(offset + 16, 2));
            byte flags = payload[offset + 22];
            byte[] rawEntry = payload.AsSpan(offset, BatchEntrySize).ToArray();

            batches.Add(new WmoGroupBatchDetail(
                batchIndex,
                offset,
                materialIdRaw,
                hasMaterialId,
                firstIndex,
                indexCount,
                flags,
                rawEntry));
        }

        return batches;
    }
}