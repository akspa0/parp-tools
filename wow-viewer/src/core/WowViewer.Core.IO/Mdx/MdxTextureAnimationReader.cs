using System.Buffers.Binary;
using System.Text;
using WowViewer.Core.Chunks;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.IO.Mdx;

public static class MdxTextureAnimationReader
{
    private const int SignatureSizeBytes = 4;
    private const int ModlNameSizeBytes = 0x50;

    public static MdxTextureAnimationFile Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static MdxTextureAnimationFile Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        if (!stream.CanSeek)
            throw new ArgumentException("MDX texture animation reading requires a seekable stream.", nameof(stream));

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
            List<MdxTextureAnimation> textureAnimations = [];
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
                else if (header.Id == MdxChunkIds.Txan)
                {
                    textureAnimations = ReadClassicTextureAnimations(stream, dataOffset, header.Size, version);
                }

                stream.Position = endOffset;
            }

            return new MdxTextureAnimationFile(sourcePath, signature, version, modelName, textureAnimations);
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static List<MdxTextureAnimation> ReadClassicTextureAnimations(Stream stream, long dataOffset, uint size, uint? version)
    {
        if (version is not null and not 1300u and not 1400u)
            return [];

        long previousPosition = stream.Position;
        try
        {
            long chunkEnd = checked(dataOffset + size);
            stream.Position = dataOffset;
            if (chunkEnd - stream.Position < sizeof(uint))
                throw new InvalidDataException("TXAN(v1300): missing texture animation count.");

            uint animationCount = ReadUInt32(stream);
            if (animationCount > 100000)
                throw new InvalidDataException($"TXAN(v1300): invalid texture animation count {animationCount}.");

            List<MdxTextureAnimation> textureAnimations = new(checked((int)animationCount));
            for (int index = 0; index < animationCount; index++)
            {
                if (chunkEnd - stream.Position < sizeof(uint))
                    throw new InvalidDataException($"TXAN(v1300): truncated section header at index {index}.");

                long entryStart = stream.Position;
                uint entrySize = ReadUInt32(stream);
                long entryEnd = checked(entryStart + entrySize);
                if (entryEnd > chunkEnd || entryEnd <= entryStart)
                    throw new InvalidDataException($"TXAN(v1300): invalid section size 0x{entrySize:X} at index {index}.");

                MdxVector3NodeTrack? translationTrack = null;
                MdxQuaternionNodeTrack? rotationTrack = null;
                MdxVector3NodeTrack? scalingTrack = null;

                while (stream.Position <= entryEnd - 4)
                {
                    string tag = ReadTag(stream);
                    switch (tag)
                    {
                        case "KTAT":
                            translationTrack = MdxTrackReader.ReadVector3Track(stream, entryEnd, tag, "TXAN(v1300)", $"TXAN(v1300): {tag} payload overran the section.");
                            break;
                        case "KTAR":
                            rotationTrack = MdxTrackReader.ReadQuaternionTrack(stream, entryEnd, tag, "TXAN(v1300)", $"TXAN(v1300): {tag} payload overran the section.");
                            break;
                        case "KTAS":
                            scalingTrack = MdxTrackReader.ReadVector3Track(stream, entryEnd, tag, "TXAN(v1300)", $"TXAN(v1300): {tag} payload overran the section.");
                            break;
                        default:
                            stream.Position = entryEnd;
                            break;
                    }
                }

                stream.Position = entryEnd;
                textureAnimations.Add(new MdxTextureAnimation(index, translationTrack, rotationTrack, scalingTrack));
            }

            stream.Position = chunkEnd;
            return textureAnimations;
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
}