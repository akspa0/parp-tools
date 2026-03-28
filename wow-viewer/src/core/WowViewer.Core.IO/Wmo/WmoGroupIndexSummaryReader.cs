using System.Buffers.Binary;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoGroupIndexSummaryReader
{
    private const int IndexStride = 2;

    public static WmoGroupIndexSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoGroupIndexSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        (uint? version, byte[] mogp) = WmoGroupReaderCommon.ReadGroupPayload(stream, sourcePath);
        return ReadMogpPayload(mogp, sourcePath, version);
    }

    internal static WmoGroupIndexSummary ReadMogpPayload(byte[] mogp, string sourcePath, uint? version)
    {
        ArgumentNullException.ThrowIfNull(mogp);

        int headerSizeBytes = WmoGroupReaderCommon.FindHeaderSize(mogp);

        string? chunkId = null;
        byte[]? indexPayload = null;
        foreach ((var header, int dataOffset) in WmoGroupReaderCommon.EnumerateSubchunks(mogp, headerSizeBytes))
        {
            if (header.Id != WmoChunkIds.Movi && header.Id != WmoChunkIds.Moin)
                continue;

            chunkId = header.Id.ToString();
            indexPayload = mogp.AsSpan(dataOffset, checked((int)header.Size)).ToArray();
            break;
        }

        if (indexPayload is null || chunkId is null)
            throw new InvalidDataException("WMO group index summary requires a MOVI or MOIN subchunk.");

        if (indexPayload.Length % IndexStride != 0)
            throw new InvalidDataException($"{chunkId} payload size {indexPayload.Length} is not divisible by {IndexStride}.");

        int indexCount = indexPayload.Length / IndexStride;
        HashSet<ushort> indices = [];
        int minIndex = 0;
        int maxIndex = 0;
        if (indexCount > 0)
        {
            minIndex = ushort.MaxValue;
            maxIndex = ushort.MinValue;
        }

        for (int index = 0; index < indexCount; index++)
        {
            ushort value = BinaryPrimitives.ReadUInt16LittleEndian(indexPayload.AsSpan(index * IndexStride, IndexStride));
            indices.Add(value);
            minIndex = Math.Min(minIndex, value);
            maxIndex = Math.Max(maxIndex, value);
        }

        int triangleCount = indexCount / 3;
        int degenerateTriangleCount = 0;
        for (int triangleIndex = 0; triangleIndex < triangleCount; triangleIndex++)
        {
            int offset = triangleIndex * 3 * IndexStride;
            ushort a = BinaryPrimitives.ReadUInt16LittleEndian(indexPayload.AsSpan(offset, IndexStride));
            ushort b = BinaryPrimitives.ReadUInt16LittleEndian(indexPayload.AsSpan(offset + IndexStride, IndexStride));
            ushort c = BinaryPrimitives.ReadUInt16LittleEndian(indexPayload.AsSpan(offset + (2 * IndexStride), IndexStride));
            if (a == b || b == c || a == c)
                degenerateTriangleCount++;
        }

        return new WmoGroupIndexSummary(
            sourcePath,
            version,
            chunkId,
            indexPayload.Length,
            indexCount,
            triangleCount,
            distinctIndexCount: indices.Count,
            minIndex,
            maxIndex,
            degenerateTriangleCount);
    }
}
