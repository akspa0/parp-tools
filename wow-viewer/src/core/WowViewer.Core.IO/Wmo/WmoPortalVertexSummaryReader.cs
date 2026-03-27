using System.Numerics;
using WowViewer.Core.Chunks;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoPortalVertexSummaryReader
{
    private const int VertexStride = 12;

    public static WmoPortalVertexSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoPortalVertexSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        byte[] payload = ReadPortalChunk(stream, sourcePath, WmoChunkIds.Mopv, out uint? version);
        if (payload.Length % VertexStride != 0)
            throw new InvalidDataException($"MOPV payload size {payload.Length} is not divisible by {VertexStride}.");

        int vertexCount = payload.Length / VertexStride;
        Vector3 boundsMin = Vector3.Zero;
        Vector3 boundsMax = Vector3.Zero;
        if (vertexCount > 0)
        {
            boundsMin = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
            boundsMax = new Vector3(float.MinValue, float.MinValue, float.MinValue);
            for (int i = 0; i < vertexCount; i++)
            {
                int offset = i * VertexStride;
                Vector3 vertex = new(BitConverter.ToSingle(payload, offset), BitConverter.ToSingle(payload, offset + 4), BitConverter.ToSingle(payload, offset + 8));
                boundsMin = Vector3.Min(boundsMin, vertex);
                boundsMax = Vector3.Max(boundsMax, vertex);
            }
        }

        return new WmoPortalVertexSummary(sourcePath, version, payload.Length, vertexCount, boundsMin, boundsMax);
    }

    internal static byte[] ReadPortalChunk(Stream stream, string sourcePath, FourCC chunkId, out uint? version)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        var (detectedVersion, chunks) = WmoRootReaderCommon.ReadRootChunks(stream, sourcePath);
        version = detectedVersion;
        return WmoRootReaderCommon.ReadRequiredChunkPayload(stream, chunks, chunkId);
    }
}
