using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.Chunks;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.IO.Mdx;

public static class MdxHitTestReader
{
    private const int SignatureSizeBytes = 4;
    private const int ModlNameSizeBytes = 0x50;

    public static MdxHitTestFile Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static MdxHitTestFile Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        if (!stream.CanSeek)
            throw new ArgumentException("MDX hit-test reading requires a seekable stream.", nameof(stream));

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
            List<MdxHitTestShape> shapes = [];
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
                else if (header.Id == MdxChunkIds.Htst)
                {
                    shapes = ReadClassicHitTestShapes(stream, dataOffset, header.Size, version);
                }

                stream.Position = endOffset;
            }

            return new MdxHitTestFile(sourcePath, signature, version, modelName, shapes);
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static List<MdxHitTestShape> ReadClassicHitTestShapes(Stream stream, long dataOffset, uint size, uint? version)
    {
        if (version is not null and not 1300u and not 1400u)
            return [];

        long previousPosition = stream.Position;
        try
        {
            long chunkEnd = checked(dataOffset + size);
            stream.Position = dataOffset;
            if (chunkEnd - stream.Position < sizeof(uint))
                throw new InvalidDataException("HTST(v1300): missing hit-test shape count.");

            uint shapeCount = ReadUInt32(stream);
            if (shapeCount > 100000)
                throw new InvalidDataException($"HTST(v1300): invalid hit-test shape count {shapeCount}.");

            List<MdxHitTestShape> shapes = new(checked((int)shapeCount));
            for (int index = 0; index < shapeCount; index++)
            {
                if (chunkEnd - stream.Position < sizeof(uint) * 2)
                    throw new InvalidDataException($"HTST(v1300): truncated section header at index {index}.");

                long entryStart = stream.Position;
                uint entrySize = ReadUInt32(stream);
                long entryEnd = checked(entryStart + entrySize);
                if (entryEnd > chunkEnd || entryEnd <= entryStart)
                    throw new InvalidDataException($"HTST(v1300): invalid section size 0x{entrySize:X} at index {index}.");

                long nodeStart = stream.Position;
                uint nodeSize = ReadUInt32(stream);
                long nodeEnd = checked(nodeStart + nodeSize);
                if (nodeEnd > entryEnd || nodeEnd <= nodeStart)
                    throw new InvalidDataException($"HTST(v1300): invalid node size 0x{nodeSize:X} at index {index}.");

                (string name, int objectId, int parentId, uint flags, MdxVector3NodeTrack? translationTrack, MdxQuaternionNodeTrack? rotationTrack, MdxVector3NodeTrack? scalingTrack) =
                    ReadNodeTracks(stream, nodeEnd, index, "HTST(v1300)");

                if (entryEnd - stream.Position < 1)
                    throw new InvalidDataException($"HTST(v1300): missing shape type at index {index}.");

                MdxGeometryShapeType shapeType = ReadGeometryShapeType(stream, $"HTST(v1300): invalid shape type at index {index}.");
                Vector3? minimum = null;
                Vector3? maximum = null;
                Vector3? basePoint = null;
                float? height = null;
                float? radius = null;
                Vector3? center = null;
                float? length = null;
                float? width = null;

                switch (shapeType)
                {
                    case MdxGeometryShapeType.Box:
                        minimum = ReadVector3(stream);
                        maximum = ReadVector3(stream);
                        break;
                    case MdxGeometryShapeType.Cylinder:
                        basePoint = ReadVector3(stream);
                        height = ReadSingle(stream);
                        radius = ReadSingle(stream);
                        break;
                    case MdxGeometryShapeType.Sphere:
                        center = ReadVector3(stream);
                        radius = ReadSingle(stream);
                        break;
                    case MdxGeometryShapeType.Plane:
                        length = ReadSingle(stream);
                        width = ReadSingle(stream);
                        break;
                }

                if (stream.Position > entryEnd)
                    throw new InvalidDataException($"HTST(v1300): payload overran section {index}.");

                stream.Position = entryEnd;
                shapes.Add(new MdxHitTestShape(index, name, objectId, parentId, flags, translationTrack, rotationTrack, scalingTrack, shapeType, minimum, maximum, basePoint, height, radius, center, length, width));
            }

            stream.Position = chunkEnd;
            return shapes;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static (string Name, int ObjectId, int ParentId, uint Flags, MdxVector3NodeTrack? TranslationTrack, MdxQuaternionNodeTrack? RotationTrack, MdxVector3NodeTrack? ScalingTrack) ReadNodeTracks(Stream stream, long nodeEnd, int index, string chunkLabel)
    {
        if (nodeEnd - stream.Position < 0x50 + 12)
            throw new InvalidDataException($"{chunkLabel}: truncated node payload at index {index}.");

        string name = ReadFixedAscii(stream, 0x50);
        int objectId = ReadInt32(stream);
        int parentId = ReadInt32(stream);
        uint flags = ReadUInt32(stream);
        MdxVector3NodeTrack? translationTrack = null;
        MdxQuaternionNodeTrack? rotationTrack = null;
        MdxVector3NodeTrack? scalingTrack = null;

        while (stream.Position <= nodeEnd - 4)
        {
            string tag = ReadTag(stream);
            switch (tag)
            {
                case "KGTR":
                    translationTrack = MdxTrackReader.ReadVector3Track(stream, nodeEnd, tag, "MDLGENOBJECT(v1300)", $"{chunkLabel}: {tag} payload overran the node.");
                    break;
                case "KGRT":
                    rotationTrack = MdxTrackReader.ReadQuaternionTrack(stream, nodeEnd, tag, "MDLGENOBJECT(v1300)", $"{chunkLabel}: {tag} payload overran the node.");
                    break;
                case "KGSC":
                    scalingTrack = MdxTrackReader.ReadVector3Track(stream, nodeEnd, tag, "MDLGENOBJECT(v1300)", $"{chunkLabel}: {tag} payload overran the node.");
                    break;
                default:
                    stream.Position = nodeEnd;
                    break;
            }
        }

        stream.Position = nodeEnd;
        return (name, objectId, parentId, flags, translationTrack, rotationTrack, scalingTrack);
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

    private static string ReadFixedAscii(Stream stream, int size)
    {
        byte[] bytes = new byte[size];
        stream.ReadExactly(bytes);
        int terminatorIndex = Array.IndexOf(bytes, (byte)0);
        int count = terminatorIndex >= 0 ? terminatorIndex : bytes.Length;
        return Encoding.ASCII.GetString(bytes, 0, count);
    }

    private static string ReadTag(Stream stream)
    {
        Span<byte> bytes = stackalloc byte[4];
        stream.ReadExactly(bytes);
        return Encoding.ASCII.GetString(bytes);
    }

    private static MdxGeometryShapeType ReadGeometryShapeType(Stream stream, string invalidMessage)
    {
        byte value = ReadByte(stream);
        if (value > (byte)MdxGeometryShapeType.Plane)
            throw new InvalidDataException(invalidMessage);

        return (MdxGeometryShapeType)value;
    }

    private static uint ReadUInt32(Stream stream)
    {
        Span<byte> bytes = stackalloc byte[sizeof(uint)];
        stream.ReadExactly(bytes);
        return BinaryPrimitives.ReadUInt32LittleEndian(bytes);
    }

    private static int ReadInt32(Stream stream)
    {
        return unchecked((int)ReadUInt32(stream));
    }

    private static byte ReadByte(Stream stream)
    {
        int value = stream.ReadByte();
        if (value < 0)
            throw new EndOfStreamException();

        return (byte)value;
    }

    private static float ReadSingle(Stream stream)
    {
        Span<byte> bytes = stackalloc byte[sizeof(float)];
        stream.ReadExactly(bytes);
        return BinaryPrimitives.ReadSingleLittleEndian(bytes);
    }

    private static Vector3 ReadVector3(Stream stream)
    {
        return new Vector3(ReadSingle(stream), ReadSingle(stream), ReadSingle(stream));
    }
}