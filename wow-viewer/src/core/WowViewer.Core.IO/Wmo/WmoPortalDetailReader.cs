using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoPortalDetailReader
{
    private const int PortalVertexStride = 12;
    private const int PortalInfoEntrySize = 20;
    private const int PortalRefEntrySize = 8;

    public static IReadOnlyList<WmoPortalVertexDetail> ReadVertices(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return ReadVertices(stream, Path.GetFullPath(path));
    }

    public static IReadOnlyList<WmoPortalVertexDetail> ReadVertices(Stream stream, string sourcePath = "<memory>")
    {
        byte[] payload = WmoPortalVertexSummaryReader.ReadPortalChunk(stream, sourcePath, WmoChunkIds.Mopv, out _);
        if (payload.Length % PortalVertexStride != 0)
            throw new InvalidDataException($"MOPV payload size {payload.Length} is not divisible by {PortalVertexStride}.");

        int entryCount = payload.Length / PortalVertexStride;
        List<WmoPortalVertexDetail> vertices = new(entryCount);
        for (int index = 0; index < entryCount; index++)
        {
            int offset = index * PortalVertexStride;
            vertices.Add(new WmoPortalVertexDetail(
                index,
                new Vector3(
                    BitConverter.ToSingle(payload, offset),
                    BitConverter.ToSingle(payload, offset + 4),
                    BitConverter.ToSingle(payload, offset + 8))));
        }

        return vertices;
    }

    public static IReadOnlyList<WmoPortalDetail> ReadPortals(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return ReadPortals(stream, Path.GetFullPath(path));
    }

    public static IReadOnlyList<WmoPortalDetail> ReadPortals(Stream stream, string sourcePath = "<memory>")
    {
        byte[] payload = WmoPortalVertexSummaryReader.ReadPortalChunk(stream, sourcePath, WmoChunkIds.Mopt, out _);
        if (payload.Length % PortalInfoEntrySize != 0)
            throw new InvalidDataException($"MOPT payload size {payload.Length} is not divisible by {PortalInfoEntrySize}.");

        IReadOnlyList<WmoPortalVertexDetail> allVertices = ReadVertices(stream, sourcePath);
        int entryCount = payload.Length / PortalInfoEntrySize;
        List<WmoPortalDetail> portals = new(entryCount);
        for (int index = 0; index < entryCount; index++)
        {
            int offset = index * PortalInfoEntrySize;
            int startVertex = BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(offset, 2));
            int vertexCount = BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(offset + 2, 2));
            Vector3 normal = new(
                BitConverter.ToSingle(payload, offset + 4),
                BitConverter.ToSingle(payload, offset + 8),
                BitConverter.ToSingle(payload, offset + 12));
            float planeDistance = BitConverter.ToSingle(payload, offset + 16);
            List<Vector3> vertices = new(vertexCount);
            for (int vertexIndex = 0; vertexIndex < vertexCount; vertexIndex++)
            {
                int sourceVertexIndex = startVertex + vertexIndex;
                if ((uint)sourceVertexIndex >= (uint)allVertices.Count)
                    break;

                vertices.Add(allVertices[sourceVertexIndex].Position);
            }

            portals.Add(new WmoPortalDetail(index, startVertex, vertexCount, vertices, normal, planeDistance));
        }

        return portals;
    }

    public static IReadOnlyList<WmoPortalReferenceDetail> ReadReferences(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return ReadReferences(stream, Path.GetFullPath(path));
    }

    public static IReadOnlyList<WmoPortalReferenceDetail> ReadReferences(Stream stream, string sourcePath = "<memory>")
    {
        byte[] payload = WmoPortalVertexSummaryReader.ReadPortalChunk(stream, sourcePath, WmoChunkIds.Mopr, out _);
        if (payload.Length % PortalRefEntrySize != 0)
            throw new InvalidDataException($"MOPR payload size {payload.Length} is not divisible by {PortalRefEntrySize}.");

        int entryCount = payload.Length / PortalRefEntrySize;
        List<WmoPortalReferenceDetail> refs = new(entryCount);
        for (int index = 0; index < entryCount; index++)
        {
            int offset = index * PortalRefEntrySize;
            refs.Add(new WmoPortalReferenceDetail(
                index,
                BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(offset, 2)),
                BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(offset + 2, 2)),
                BinaryPrimitives.ReadInt16LittleEndian(payload.AsSpan(offset + 4, 2))));
        }

        return refs;
    }
}