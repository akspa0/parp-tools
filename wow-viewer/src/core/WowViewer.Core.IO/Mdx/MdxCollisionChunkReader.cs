using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.IO.Mdx;

internal static class MdxCollisionChunkReader
{
    public static MdxCollisionMesh? ReadClassicCollisionMesh(Stream stream, long dataOffset, uint size, uint? version)
    {
        if (version is not null and not 1300u and not 1400u)
            return null;

        long previousPosition = stream.Position;
        try
        {
            long chunkEnd = checked(dataOffset + size);
            stream.Position = dataOffset;

            ExpectTag(stream, "VRTX", "CLID(v1300): expected VRTX.");
            int vertexCount = ReadNonNegativeCount(stream, "CLID(v1300): negative VRTX count.");

            List<Vector3> vertices = new(vertexCount);
            Vector3? boundsMin = null;
            Vector3? boundsMax = null;
            if (vertexCount > 0)
            {
                Vector3 firstVertex = ReadVector3(stream);
                vertices.Add(firstVertex);

                Vector3 min = firstVertex;
                Vector3 max = firstVertex;
                for (int index = 1; index < vertexCount; index++)
                {
                    Vector3 vertex = ReadVector3(stream);
                    vertices.Add(vertex);
                    min = Vector3.Min(min, vertex);
                    max = Vector3.Max(max, vertex);
                }

                boundsMin = min;
                boundsMax = max;
            }

            ExpectTag(stream, "TRI ", "CLID(v1300): expected TRI .");
            int triangleIndexCount = ReadNonNegativeCount(stream, "CLID(v1300): negative TRI count.");
            if (triangleIndexCount % 3 != 0)
                throw new InvalidDataException("CLID(v1300): TRI count must be divisible by 3.");

            List<ushort> triangleIndices = new(triangleIndexCount);
            int maxTriangleIndex = 0;
            for (int index = 0; index < triangleIndexCount; index++)
            {
                ushort triangleIndex = ReadUInt16(stream);
                if (triangleIndex >= vertexCount)
                    throw new InvalidDataException($"CLID(v1300): TRI index {triangleIndex} exceeded VRTX count {vertexCount}.");

                triangleIndices.Add(triangleIndex);
                maxTriangleIndex = Math.Max(maxTriangleIndex, triangleIndex);
            }

            ExpectTag(stream, "NRMS", "CLID(v1300): expected NRMS.");
            int facetNormalCount = ReadNonNegativeCount(stream, "CLID(v1300): negative NRMS count.");
            List<Vector3> facetNormals = new(facetNormalCount);
            for (int index = 0; index < facetNormalCount; index++)
                facetNormals.Add(ReadVector3(stream));

            if (stream.Position != chunkEnd)
                throw new InvalidDataException("CLID(v1300): chunk contained unexpected trailing bytes.");

            return new MdxCollisionMesh(vertices, triangleIndices, facetNormals, maxTriangleIndex, boundsMin, boundsMax);
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static void ExpectTag(Stream stream, string expectedTag, string errorMessage)
    {
        string actualTag = ReadTag(stream);
        if (!string.Equals(actualTag, expectedTag, StringComparison.Ordinal))
            throw new InvalidDataException($"{errorMessage} Found '{actualTag}'.");
    }

    private static string ReadTag(Stream stream)
    {
        Span<byte> bytes = stackalloc byte[4];
        stream.ReadExactly(bytes);
        return Encoding.ASCII.GetString(bytes);
    }

    private static int ReadNonNegativeCount(Stream stream, string errorMessage)
    {
        int count = unchecked((int)ReadUInt32(stream));
        if (count < 0)
            throw new InvalidDataException(errorMessage);

        return count;
    }

    private static ushort ReadUInt16(Stream stream)
    {
        Span<byte> buffer = stackalloc byte[sizeof(ushort)];
        stream.ReadExactly(buffer);
        return BinaryPrimitives.ReadUInt16LittleEndian(buffer);
    }

    private static uint ReadUInt32(Stream stream)
    {
        Span<byte> buffer = stackalloc byte[sizeof(uint)];
        stream.ReadExactly(buffer);
        return BinaryPrimitives.ReadUInt32LittleEndian(buffer);
    }

    private static float ReadSingle(Stream stream)
    {
        Span<byte> buffer = stackalloc byte[sizeof(float)];
        stream.ReadExactly(buffer);
        return BinaryPrimitives.ReadSingleLittleEndian(buffer);
    }

    private static Vector3 ReadVector3(Stream stream)
    {
        return new Vector3(ReadSingle(stream), ReadSingle(stream), ReadSingle(stream));
    }
}