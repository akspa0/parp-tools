using System.Numerics;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoVisibleVertexSummaryReader
{
    private const int VertexStride = 12;

    public static WmoVisibleVertexSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoVisibleVertexSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        (uint? version, IReadOnlyList<ChunkSpan> chunks) = WmoRootReaderCommon.ReadRootChunks(stream, sourcePath);
        byte[] movv = WmoRootReaderCommon.ReadRequiredChunkPayload(stream, chunks, WmoChunkIds.Movv);
        if (movv.Length % VertexStride != 0)
            throw new InvalidDataException($"MOVV payload size {movv.Length} is not divisible by {VertexStride}.");

        int vertexCount = movv.Length / VertexStride;
        Vector3 boundsMin = Vector3.Zero;
        Vector3 boundsMax = Vector3.Zero;
        if (vertexCount > 0)
        {
            boundsMin = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
            boundsMax = new Vector3(float.MinValue, float.MinValue, float.MinValue);
            for (int i = 0; i < vertexCount; i++)
            {
                int offset = i * VertexStride;
                Vector3 vertex = new(BitConverter.ToSingle(movv, offset), BitConverter.ToSingle(movv, offset + 4), BitConverter.ToSingle(movv, offset + 8));
                boundsMin = Vector3.Min(boundsMin, vertex);
                boundsMax = Vector3.Max(boundsMax, vertex);
            }
        }

        return new WmoVisibleVertexSummary(sourcePath, version, movv.Length, vertexCount, boundsMin, boundsMax);
    }
}
