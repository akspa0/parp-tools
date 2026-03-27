using System.Buffers.Binary;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoPortalVertexRangeSummaryReader
{
    private const int PortalInfoEntrySize = 20;
    private const int PortalVertexStride = 12;

    public static WmoPortalVertexRangeSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoPortalVertexRangeSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        (uint? version, IReadOnlyList<ChunkSpan> chunks) = WmoRootReaderCommon.ReadRootChunks(stream, sourcePath);
        byte[] mopt = WmoRootReaderCommon.ReadRequiredChunkPayload(stream, chunks, WmoChunkIds.Mopt);
        byte[] mopv = WmoRootReaderCommon.ReadRequiredChunkPayload(stream, chunks, WmoChunkIds.Mopv);
        if (mopt.Length % PortalInfoEntrySize != 0)
            throw new InvalidDataException($"MOPT payload size {mopt.Length} is not divisible by {PortalInfoEntrySize}.");
        if (mopv.Length % PortalVertexStride != 0)
            throw new InvalidDataException($"MOPV payload size {mopv.Length} is not divisible by {PortalVertexStride}.");

        int entryCount = mopt.Length / PortalInfoEntrySize;
        int vertexCount = mopv.Length / PortalVertexStride;
        int zeroVertexPortalCount = 0;
        int coveredPortalCount = 0;
        int outOfRangePortalCount = 0;
        int maxVertexEnd = 0;

        for (int i = 0; i < entryCount; i++)
        {
            int offset = i * PortalInfoEntrySize;
            int startVertex = BinaryPrimitives.ReadUInt16LittleEndian(mopt.AsSpan(offset, 2));
            int count = BinaryPrimitives.ReadUInt16LittleEndian(mopt.AsSpan(offset + 2, 2));
            int end = startVertex + count;

            maxVertexEnd = Math.Max(maxVertexEnd, end);
            if (count == 0)
            {
                zeroVertexPortalCount++;
                continue;
            }

            if (end <= vertexCount)
                coveredPortalCount++;
            else
                outOfRangePortalCount++;
        }

        return new WmoPortalVertexRangeSummary(sourcePath, version, entryCount, vertexCount, zeroVertexPortalCount, coveredPortalCount, outOfRangePortalCount, maxVertexEnd);
    }
}
