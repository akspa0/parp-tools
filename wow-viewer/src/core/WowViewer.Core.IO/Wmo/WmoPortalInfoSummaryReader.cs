using System.Buffers.Binary;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoPortalInfoSummaryReader
{
    private const int EntrySize = 20;

    public static WmoPortalInfoSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoPortalInfoSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        byte[] payload = WmoPortalVertexSummaryReader.ReadPortalChunk(stream, sourcePath, WmoChunkIds.Mopt, out uint? version);
        if (payload.Length % EntrySize != 0)
            throw new InvalidDataException($"MOPT payload size {payload.Length} is not divisible by {EntrySize}.");

        int entryCount = payload.Length / EntrySize;
        int maxStartVertex = 0;
        int maxVertexCount = 0;
        float minPlaneD = 0f;
        float maxPlaneD = 0f;
        if (entryCount > 0)
        {
            minPlaneD = float.MaxValue;
            maxPlaneD = float.MinValue;
        }

        for (int i = 0; i < entryCount; i++)
        {
            int offset = i * EntrySize;
            int startVertex = BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(offset, 2));
            int vertexCount = BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(offset + 2, 2));
            float planeD = BitConverter.ToSingle(payload, offset + 16);
            maxStartVertex = Math.Max(maxStartVertex, startVertex);
            maxVertexCount = Math.Max(maxVertexCount, vertexCount);
            minPlaneD = Math.Min(minPlaneD, planeD);
            maxPlaneD = Math.Max(maxPlaneD, planeD);
        }

        return new WmoPortalInfoSummary(sourcePath, version, payload.Length, entryCount, maxStartVertex, maxVertexCount, minPlaneD, maxPlaneD);
    }
}
