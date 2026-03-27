using System.Numerics;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoGroupVertexSummaryReader
{
    private const int VertexStride = 12;

    public static WmoGroupVertexSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoGroupVertexSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        (uint? version, byte[] mogp) = WmoGroupReaderCommon.ReadGroupPayload(stream, sourcePath);
        int headerSizeBytes = WmoGroupReaderCommon.FindHeaderSize(mogp);

        byte[]? movtPayload = null;
        foreach ((var header, int dataOffset) in WmoGroupReaderCommon.EnumerateSubchunks(mogp, headerSizeBytes))
        {
            if (header.Id != WmoChunkIds.Movt)
                continue;

            movtPayload = mogp.AsSpan(dataOffset, checked((int)header.Size)).ToArray();
            break;
        }

        if (movtPayload is null)
            throw new InvalidDataException("WMO group vertex summary requires a MOVT subchunk.");

        if (movtPayload.Length % VertexStride != 0)
            throw new InvalidDataException($"MOVT payload size {movtPayload.Length} is not divisible by {VertexStride}.");

        int vertexCount = movtPayload.Length / VertexStride;
        Vector3 boundsMin = Vector3.Zero;
        Vector3 boundsMax = Vector3.Zero;
        if (vertexCount > 0)
        {
            boundsMin = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
            boundsMax = new Vector3(float.MinValue, float.MinValue, float.MinValue);
            for (int index = 0; index < vertexCount; index++)
            {
                int offset = index * VertexStride;
                float x = BitConverter.ToSingle(movtPayload, offset);
                float y = BitConverter.ToSingle(movtPayload, offset + 4);
                float z = BitConverter.ToSingle(movtPayload, offset + 8);
                boundsMin = Vector3.Min(boundsMin, new Vector3(x, y, z));
                boundsMax = Vector3.Max(boundsMax, new Vector3(x, y, z));
            }
        }

        return new WmoGroupVertexSummary(
            sourcePath,
            version,
            movtPayload.Length,
            vertexCount,
            boundsMin,
            boundsMax);
    }
}
