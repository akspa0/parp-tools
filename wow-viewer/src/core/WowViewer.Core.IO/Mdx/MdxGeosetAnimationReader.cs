using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.Chunks;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.IO.Mdx;

public static class MdxGeosetAnimationReader
{
    private const int SignatureSizeBytes = 4;
    private const int ModlNameSizeBytes = 0x50;

    public static MdxGeosetAnimationFile Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static MdxGeosetAnimationFile Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        if (!stream.CanSeek)
            throw new ArgumentException("MDX geoset-animation reading requires a seekable stream.", nameof(stream));

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
            List<MdxGeosetAnimation> geosetAnimations = [];
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
                else if (header.Id == MdxChunkIds.Geoa)
                {
                    geosetAnimations = ReadClassicGeosetAnimations(stream, dataOffset, header.Size, version);
                }

                stream.Position = endOffset;
            }

            return new MdxGeosetAnimationFile(sourcePath, signature, version, modelName, geosetAnimations);
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static List<MdxGeosetAnimation> ReadClassicGeosetAnimations(Stream stream, long dataOffset, uint size, uint? version)
    {
        if (version is not null and not 1300u and not 1400u)
            return [];

        long previousPosition = stream.Position;
        try
        {
            long chunkEnd = checked(dataOffset + size);
            stream.Position = dataOffset;
            if (chunkEnd - stream.Position < sizeof(uint))
                throw new InvalidDataException("GEOA(v1300): missing geoset animation count.");

            uint animationCount = ReadUInt32(stream);
            if (animationCount > 100000)
                throw new InvalidDataException($"GEOA(v1300): invalid geoset animation count {animationCount}.");

            List<MdxGeosetAnimation> geosetAnimations = new(checked((int)animationCount));
            for (int index = 0; index < animationCount; index++)
            {
                if (chunkEnd - stream.Position < sizeof(uint))
                    throw new InvalidDataException($"GEOA(v1300): truncated section header at index {index}.");

                long entryStart = stream.Position;
                uint entrySize = ReadUInt32(stream);
                long entryEnd = checked(entryStart + entrySize);
                if (entryEnd > chunkEnd || entryEnd <= entryStart)
                    throw new InvalidDataException($"GEOA(v1300): invalid section size 0x{entrySize:X} at index {index}.");

                uint geosetId = ReadUInt32(stream);
                float staticAlpha = ReadSingle(stream);
                Vector3 staticColor = ReadVector3(stream);
                uint flags = ReadUInt32(stream);

                MdxScalarTrack? alphaTrack = null;
                MdxColorTrack? colorTrack = null;
                while (stream.Position <= entryEnd - 4)
                {
                    string tag = ReadTag(stream);
                    switch (tag)
                    {
                        case "KGAO":
                            alphaTrack = ReadScalarTrack(stream, entryEnd, tag, "GEOA(v1300)", $"GEOA(v1300): {tag} payload overran the section.");
                            break;
                        case "KGAC":
                            colorTrack = ReadColorTrack(stream, entryEnd, tag, "GEOA(v1300)", $"GEOA(v1300): {tag} payload overran the section.");
                            break;
                        default:
                            stream.Position = entryEnd;
                            break;
                    }
                }

                stream.Position = entryEnd;
                geosetAnimations.Add(new MdxGeosetAnimation(index, geosetId, staticAlpha, staticColor, flags, alphaTrack, colorTrack));
            }

            stream.Position = chunkEnd;
            return geosetAnimations;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static MdxScalarTrack ReadScalarTrack(Stream stream, long limit, string tag, string contextLabel, string overrunMessage)
    {
        uint keyCount = ReadUInt32(stream);
        if (keyCount > 100000)
            throw new InvalidDataException($"{contextLabel}: invalid {tag} key count {keyCount}.");

        uint interpolationType = ReadUInt32(stream);
        int globalSequenceId = ReadInt32(stream);
        List<MdxScalarKeyframe> keys = new(checked((int)keyCount));

        for (uint keyIndex = 0; keyIndex < keyCount; keyIndex++)
        {
            int time = ReadInt32(stream);
            float value = ReadSingle(stream);
            float? inTangent = null;
            float? outTangent = null;
            if (TrackUsesTangents(interpolationType))
            {
                inTangent = ReadSingle(stream);
                outTangent = ReadSingle(stream);
            }

            keys.Add(new MdxScalarKeyframe(time, value, inTangent, outTangent));
        }

        if (stream.Position > limit)
            throw new InvalidDataException(overrunMessage);

        return new MdxScalarTrack(tag, (MdxTrackInterpolationType)interpolationType, globalSequenceId, keys);
    }

    private static MdxColorTrack ReadColorTrack(Stream stream, long limit, string tag, string contextLabel, string overrunMessage)
    {
        uint keyCount = ReadUInt32(stream);
        if (keyCount > 100000)
            throw new InvalidDataException($"{contextLabel}: invalid {tag} key count {keyCount}.");

        uint interpolationType = ReadUInt32(stream);
        int globalSequenceId = ReadInt32(stream);
        List<MdxColorKeyframe> keys = new(checked((int)keyCount));

        for (uint keyIndex = 0; keyIndex < keyCount; keyIndex++)
        {
            int time = ReadInt32(stream);
            Vector3 value = ReadVector3(stream);
            Vector3? inTangent = null;
            Vector3? outTangent = null;
            if (TrackUsesTangents(interpolationType))
            {
                inTangent = ReadVector3(stream);
                outTangent = ReadVector3(stream);
            }

            keys.Add(new MdxColorKeyframe(time, value, inTangent, outTangent));
        }

        if (stream.Position > limit)
            throw new InvalidDataException(overrunMessage);

        return new MdxColorTrack(tag, (MdxTrackInterpolationType)interpolationType, globalSequenceId, keys);
    }

    private static bool TrackUsesTangents(uint interpolationType) => interpolationType >= 2u;

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

    private static uint ReadUInt32(Stream stream)
    {
        Span<byte> bytes = stackalloc byte[sizeof(uint)];
        stream.ReadExactly(bytes);
        return BinaryPrimitives.ReadUInt32LittleEndian(bytes);
    }

    private static int ReadInt32(Stream stream) => unchecked((int)ReadUInt32(stream));

    private static float ReadSingle(Stream stream)
    {
        Span<byte> bytes = stackalloc byte[sizeof(float)];
        stream.ReadExactly(bytes);
        return BinaryPrimitives.ReadSingleLittleEndian(bytes);
    }

    private static Vector3 ReadVector3(Stream stream) => new(ReadSingle(stream), ReadSingle(stream), ReadSingle(stream));
}
