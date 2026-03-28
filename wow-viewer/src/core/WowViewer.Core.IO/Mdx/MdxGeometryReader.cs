using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.Chunks;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.IO.Mdx;

public static class MdxGeometryReader
{
    private const int SignatureSizeBytes = 4;
    private const int ModlNameSizeBytes = 0x50;

    public static MdxGeometryFile Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static MdxGeometryFile Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        if (!stream.CanSeek)
            throw new ArgumentException("MDX geometry reading requires a seekable stream.", nameof(stream));

        if (stream.Length < SignatureSizeBytes)
            throw new InvalidDataException($"MDX file '{sourcePath}' is too small to contain a signature.");

        long previousPosition = stream.Position;
        try
        {
            stream.Position = 0;
            Span<byte> signatureBytes = stackalloc byte[SignatureSizeBytes];
            stream.ReadExactly(signatureBytes);

            string signature = Encoding.ASCII.GetString(signatureBytes);
            if (!string.Equals(signature, "MDLX", StringComparison.Ordinal))
                throw new InvalidDataException($"File '{sourcePath}' does not contain an MDLX signature. Found '{signature}'.");

            uint? version = null;
            string? modelName = null;
            List<MdxGeosetGeometry> geosets = [];
            Span<byte> headerBytes = stackalloc byte[ChunkHeader.SizeInBytes];

            while (stream.Position <= stream.Length - ChunkHeader.SizeInBytes)
            {
                long headerOffset = stream.Position;
                stream.ReadExactly(headerBytes);
                if (!TryReadMdxChunkHeader(headerBytes, out ChunkHeader header))
                    throw new InvalidDataException($"Could not decode MDX chunk header at offset {headerOffset}.");

                long dataOffset = stream.Position;
                long endOffset = checked(dataOffset + header.Size);
                if (endOffset > stream.Length)
                    throw new InvalidDataException($"MDX chunk {header.Id} at offset {headerOffset} overruns the stream length.");

                if (header.Id == MdxChunkIds.Vers && header.Size >= sizeof(uint))
                {
                    version = ReadUInt32At(stream, dataOffset);
                }
                else if (header.Id == MdxChunkIds.Modl && header.Size >= ModlNameSizeBytes)
                {
                    modelName = ReadFixedAsciiAt(stream, dataOffset, ModlNameSizeBytes);
                }
                else if (header.Id == MdxChunkIds.Geos)
                {
                    geosets = ReadClassicGeosets(stream, dataOffset, header.Size, version);
                }

                stream.Position = endOffset;
            }

            return new MdxGeometryFile(sourcePath, signature, version, modelName, geosets);
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static bool TryReadMdxChunkHeader(ReadOnlySpan<byte> data, out ChunkHeader header)
    {
        if (data.Length < ChunkHeader.SizeInBytes)
        {
            header = default;
            return false;
        }

        string idText = Encoding.ASCII.GetString(data[..4]);
        FourCC id = FourCC.FromString(idText);
        uint size = BinaryPrimitives.ReadUInt32LittleEndian(data[4..]);
        header = new ChunkHeader(id, size);
        return true;
    }

    private static List<MdxGeosetGeometry> ReadClassicGeosets(Stream stream, long dataOffset, uint size, uint? version)
    {
        if (version is not null and not 1300u and not 1400u)
            return [];

        long previousPosition = stream.Position;
        try
        {
            long chunkEnd = checked(dataOffset + size);
            stream.Position = dataOffset;
            if (chunkEnd - stream.Position < sizeof(int))
                throw new InvalidDataException("GEOS(v1300): missing geoset count.");

            int geosetCount = ReadInt32(stream);
            if (geosetCount < 0 || geosetCount > 100000)
                throw new InvalidDataException($"GEOS(v1300): invalid geoset count {geosetCount}.");

            List<MdxGeosetGeometry> geosets = new(geosetCount);
            for (int index = 0; index < geosetCount; index++)
            {
                long geosetStart = stream.Position;
                if (chunkEnd - geosetStart < sizeof(uint))
                    throw new InvalidDataException($"GEOS(v1300): truncated geoset header at index {index}.");

                uint geosetSize = ReadUInt32(stream);
                long geosetEnd = checked(geosetStart + geosetSize);
                if (geosetEnd > chunkEnd || geosetEnd <= geosetStart)
                    throw new InvalidDataException($"GEOS(v1300): invalid geoset size 0x{geosetSize:X} at index {index}.");

                ExpectTag(stream, "VRTX", $"GEOS(v1300): expected VRTX at index {index}.");
                int vertexCount = ReadNonNegativeCount(stream, "GEOS(v1300): negative VRTX count.");
                List<Vector3> vertices = new(vertexCount);
                for (int vertexIndex = 0; vertexIndex < vertexCount; vertexIndex++)
                    vertices.Add(ReadVector3(stream));

                ExpectTag(stream, "NRMS", $"GEOS(v1300): expected NRMS at index {index}.");
                int normalCount = ReadNonNegativeCount(stream, "GEOS(v1300): negative NRMS count.");
                List<Vector3> normals = new(normalCount);
                for (int normalIndex = 0; normalIndex < normalCount; normalIndex++)
                    normals.Add(ReadVector3(stream));

                List<IReadOnlyList<Vector2>> uvSets = [];
                if (TryReadTag(stream, "UVAS"))
                {
                    int uvSetCount = ReadNonNegativeCount(stream, "GEOS(v1300): negative UVAS count.");
                    for (int uvSetIndex = 0; uvSetIndex < uvSetCount; uvSetIndex++)
                    {
                        List<Vector2> uvSet = new(vertexCount);
                        for (int uvIndex = 0; uvIndex < vertexCount; uvIndex++)
                            uvSet.Add(ReadVector2(stream));

                        uvSets.Add(uvSet);
                    }
                }

                ExpectTag(stream, "PTYP", $"GEOS(v1300): expected PTYP at index {index}.");
                int primitiveTypeCount = ReadNonNegativeCount(stream, "GEOS(v1300): negative PTYP count.");
                List<byte> primitiveTypes = new(primitiveTypeCount);
                for (int primitiveIndex = 0; primitiveIndex < primitiveTypeCount; primitiveIndex++)
                {
                    int primitiveType = stream.ReadByte();
                    if (primitiveType < 0)
                        throw new EndOfStreamException("GEOS(v1300): truncated PTYP payload.");

                    if (primitiveType != 4)
                        throw new InvalidDataException($"GEOS(v1300): unsupported primitive type {primitiveType}.");

                    primitiveTypes.Add((byte)primitiveType);
                }

                ExpectTag(stream, "PCNT", $"GEOS(v1300): expected PCNT at index {index}.");
                int faceGroupCount = ReadNonNegativeCount(stream, "GEOS(v1300): negative PCNT count.");
                List<int> faceGroups = new(faceGroupCount);
                for (int faceGroupIndex = 0; faceGroupIndex < faceGroupCount; faceGroupIndex++)
                    faceGroups.Add(ReadInt32(stream));

                ExpectTag(stream, "PVTX", $"GEOS(v1300): expected PVTX at index {index}.");
                int indexCount = ReadNonNegativeCount(stream, "GEOS(v1300): negative PVTX count.");
                List<ushort> indices = new(indexCount);
                for (int indexIndex = 0; indexIndex < indexCount; indexIndex++)
                    indices.Add(ReadUInt16(stream));

                ExpectTag(stream, "GNDX", $"GEOS(v1300): expected GNDX at index {index}.");
                int vertexGroupCount = ReadNonNegativeCount(stream, "GEOS(v1300): negative GNDX count.");
                List<byte> vertexGroups = ReadByteList(stream, vertexGroupCount, "GEOS(v1300): GNDX payload overran the geoset.");

                ExpectTag(stream, "MTGC", $"GEOS(v1300): expected MTGC at index {index}.");
                int matrixGroupCount = ReadNonNegativeCount(stream, "GEOS(v1300): negative MTGC count.");
                List<uint> matrixGroups = ReadUInt32List(stream, matrixGroupCount);

                ExpectTag(stream, "MATS", $"GEOS(v1300): expected MATS at index {index}.");
                int matrixIndexCount = ReadNonNegativeCount(stream, "GEOS(v1300): negative MATS count.");
                List<uint> matrixIndices = ReadUInt32List(stream, matrixIndexCount);

                if (TryReadTag(stream, "UVBS"))
                {
                    int uvCount = ReadNonNegativeCount(stream, "GEOS(v1300): negative UVBS count.");
                    if (uvSets.Count == 0)
                    {
                        List<Vector2> uvSet = new(uvCount);
                        for (int uvIndex = 0; uvIndex < uvCount; uvIndex++)
                            uvSet.Add(ReadVector2(stream));

                        uvSets.Add(uvSet);
                    }
                    else
                    {
                        SkipBytes(stream, checked((long)uvCount * 8), geosetEnd, "GEOS(v1300): UVBS payload overran the geoset.");
                    }
                }

                ExpectTag(stream, "BIDX", $"GEOS(v1300): expected BIDX at index {index}.");
                int boneIndexCount = ReadNonNegativeCount(stream, "GEOS(v1300): negative BIDX count.");
                List<uint> boneIndices = ReadUInt32List(stream, boneIndexCount);

                ExpectTag(stream, "BWGT", $"GEOS(v1300): expected BWGT at index {index}.");
                int boneWeightCount = ReadNonNegativeCount(stream, "GEOS(v1300): negative BWGT count.");
                List<uint> boneWeights = ReadUInt32List(stream, boneWeightCount);

                int materialId = ReadInt32(stream);
                uint selectionGroup = unchecked((uint)ReadInt32(stream));
                uint flags = unchecked((uint)ReadInt32(stream));
                float boundsRadius = ReadSingle(stream);
                Vector3 boundsMin = ReadVector3(stream);
                Vector3 boundsMax = ReadVector3(stream);
                int animationExtentCount = ReadNonNegativeCount(stream, "GEOS(v1300): negative geosetAnimCount.");
                SkipBytes(stream, checked((long)animationExtentCount * 28), geosetEnd, "GEOS(v1300): geoset animation extents overran the geoset.");

                geosets.Add(new MdxGeosetGeometry(index, vertices, normals, uvSets, primitiveTypes, faceGroups, indices, vertexGroups, matrixGroups, matrixIndices, boneIndices, boneWeights, materialId, selectionGroup, flags, boundsRadius, boundsMin, boundsMax, animationExtentCount));
                stream.Position = geosetEnd;
            }

            stream.Position = chunkEnd;
            return geosets;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static uint ReadUInt32At(Stream stream, long offset)
    {
        long previousPosition = stream.Position;
        try
        {
            stream.Position = offset;
            return ReadUInt32(stream);
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static string ReadFixedAsciiAt(Stream stream, long offset, int size)
    {
        long previousPosition = stream.Position;
        try
        {
            stream.Position = offset;
            return ReadFixedAscii(stream, size);
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static List<byte> ReadByteList(Stream stream, int count, string errorMessage)
    {
        if (count == 0)
            return [];

        byte[] bytes = new byte[count];
        int read = stream.Read(bytes, 0, bytes.Length);
        if (read != bytes.Length)
            throw new EndOfStreamException(errorMessage);

        return [.. bytes];
    }

    private static List<uint> ReadUInt32List(Stream stream, int count)
    {
        List<uint> values = new(count);
        for (int index = 0; index < count; index++)
            values.Add(ReadUInt32(stream));

        return values;
    }

    private static uint ReadUInt32(Stream stream)
    {
        Span<byte> buffer = stackalloc byte[sizeof(uint)];
        stream.ReadExactly(buffer);
        return BinaryPrimitives.ReadUInt32LittleEndian(buffer);
    }

    private static ushort ReadUInt16(Stream stream)
    {
        Span<byte> buffer = stackalloc byte[sizeof(ushort)];
        stream.ReadExactly(buffer);
        return BinaryPrimitives.ReadUInt16LittleEndian(buffer);
    }

    private static int ReadInt32(Stream stream)
    {
        return unchecked((int)ReadUInt32(stream));
    }

    private static float ReadSingle(Stream stream)
    {
        Span<byte> buffer = stackalloc byte[sizeof(float)];
        stream.ReadExactly(buffer);
        return BinaryPrimitives.ReadSingleLittleEndian(buffer);
    }

    private static Vector2 ReadVector2(Stream stream)
    {
        return new Vector2(ReadSingle(stream), ReadSingle(stream));
    }

    private static Vector3 ReadVector3(Stream stream)
    {
        return new Vector3(ReadSingle(stream), ReadSingle(stream), ReadSingle(stream));
    }

    private static string ReadTag(Stream stream)
    {
        Span<byte> tagBytes = stackalloc byte[4];
        stream.ReadExactly(tagBytes);
        return Encoding.ASCII.GetString(tagBytes);
    }

    private static bool TryReadTag(Stream stream, string expected)
    {
        long previousPosition = stream.Position;
        string actual = ReadTag(stream);
        if (string.Equals(actual, expected, StringComparison.Ordinal))
            return true;

        stream.Position = previousPosition;
        return false;
    }

    private static void ExpectTag(Stream stream, string expected, string message)
    {
        string actual = ReadTag(stream);
        if (!string.Equals(actual, expected, StringComparison.Ordinal))
            throw new InvalidDataException($"{message} Found '{actual}'.");
    }

    private static int ReadNonNegativeCount(Stream stream, string message)
    {
        int count = ReadInt32(stream);
        if (count < 0)
            throw new InvalidDataException(message);

        return count;
    }

    private static void SkipBytes(Stream stream, long byteCount, long limit, string errorMessage)
    {
        if (byteCount < 0)
            throw new InvalidDataException(errorMessage);

        long endPosition = checked(stream.Position + byteCount);
        if (endPosition > limit)
            throw new InvalidDataException(errorMessage);

        stream.Position = endPosition;
    }

    private static string ReadFixedAscii(Stream stream, int size)
    {
        byte[] bytes = new byte[size];
        stream.ReadExactly(bytes);
        int terminatorIndex = Array.IndexOf(bytes, (byte)0);
        int length = terminatorIndex >= 0 ? terminatorIndex : bytes.Length;
        return Encoding.ASCII.GetString(bytes, 0, length);
    }
}