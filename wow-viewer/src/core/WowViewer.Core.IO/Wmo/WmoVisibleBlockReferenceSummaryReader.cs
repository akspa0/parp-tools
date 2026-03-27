using System.Buffers.Binary;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoVisibleBlockReferenceSummaryReader
{
    private const int MovvVertexStride = 12;
    private const int MovbEntrySize = 4;

    public static WmoVisibleBlockReferenceSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoVisibleBlockReferenceSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        (uint? version, IReadOnlyList<ChunkSpan> chunks) = WmoRootReaderCommon.ReadRootChunks(stream, sourcePath);
        byte[] movv = WmoRootReaderCommon.ReadRequiredChunkPayload(stream, chunks, WmoChunkIds.Movv);
        byte[] movb = WmoRootReaderCommon.ReadRequiredChunkPayload(stream, chunks, WmoChunkIds.Movb);
        if (movv.Length % MovvVertexStride != 0)
            throw new InvalidDataException($"MOVV payload size {movv.Length} is not divisible by {MovvVertexStride}.");
        if (movb.Length % MovbEntrySize != 0)
            throw new InvalidDataException($"MOVB payload size {movb.Length} is not divisible by {MovbEntrySize}.");

        int visibleVertexCount = movv.Length / MovvVertexStride;
        int blockCount = movb.Length / MovbEntrySize;
        int zeroVertexBlocks = 0;
        int coveredBlocks = 0;
        int outOfRangeBlocks = 0;
        int maxVertexEnd = 0;
        for (int i = 0; i < blockCount; i++)
        {
            int offset = i * MovbEntrySize;
            int vertexCount = BinaryPrimitives.ReadUInt16LittleEndian(movb.AsSpan(offset, 2));
            int firstVertex = BinaryPrimitives.ReadUInt16LittleEndian(movb.AsSpan(offset + 2, 2));
            int vertexEnd = firstVertex + vertexCount;
            maxVertexEnd = Math.Max(maxVertexEnd, vertexEnd);

            if (vertexCount == 0)
            {
                zeroVertexBlocks++;
                continue;
            }

            if (vertexEnd <= visibleVertexCount)
                coveredBlocks++;
            else
                outOfRangeBlocks++;
        }

        return new WmoVisibleBlockReferenceSummary(sourcePath, version, blockCount, visibleVertexCount, zeroVertexBlocks, coveredBlocks, outOfRangeBlocks, maxVertexEnd);
    }
}
