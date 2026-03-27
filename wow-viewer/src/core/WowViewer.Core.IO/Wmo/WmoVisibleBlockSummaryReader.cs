using System.Buffers.Binary;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoVisibleBlockSummaryReader
{
    private const int EntrySize = 4;

    public static WmoVisibleBlockSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoVisibleBlockSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        (uint? version, IReadOnlyList<ChunkSpan> chunks) = WmoRootReaderCommon.ReadRootChunks(stream, sourcePath);
        byte[] movb = WmoRootReaderCommon.ReadRequiredChunkPayload(stream, chunks, WmoChunkIds.Movb);
        if (movb.Length % EntrySize != 0)
            throw new InvalidDataException($"MOVB payload size {movb.Length} is not divisible by {EntrySize}.");

        int blockCount = movb.Length / EntrySize;
        int totalVertexRefs = 0;
        int minVerticesPerBlock = 0;
        int maxVerticesPerBlock = 0;
        int minFirstVertex = 0;
        int maxFirstVertex = 0;
        int maxVertexEnd = 0;
        if (blockCount > 0)
        {
            minVerticesPerBlock = int.MaxValue;
            minFirstVertex = int.MaxValue;
        }

        for (int i = 0; i < blockCount; i++)
        {
            int offset = i * EntrySize;
            int vertexCount = BinaryPrimitives.ReadUInt16LittleEndian(movb.AsSpan(offset, 2));
            int firstVertex = BinaryPrimitives.ReadUInt16LittleEndian(movb.AsSpan(offset + 2, 2));
            int vertexEnd = firstVertex + vertexCount;

            totalVertexRefs += vertexCount;
            minVerticesPerBlock = Math.Min(minVerticesPerBlock, vertexCount);
            maxVerticesPerBlock = Math.Max(maxVerticesPerBlock, vertexCount);
            minFirstVertex = Math.Min(minFirstVertex, firstVertex);
            maxFirstVertex = Math.Max(maxFirstVertex, firstVertex);
            maxVertexEnd = Math.Max(maxVertexEnd, vertexEnd);
        }

        return new WmoVisibleBlockSummary(
            sourcePath,
            version,
            movb.Length,
            blockCount,
            totalVertexRefs,
            minVerticesPerBlock,
            maxVerticesPerBlock,
            minFirstVertex,
            maxFirstVertex,
            maxVertexEnd);
    }
}
