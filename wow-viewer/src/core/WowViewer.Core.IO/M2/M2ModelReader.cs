using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.M2;

namespace WowViewer.Core.IO.M2;

public static class M2ModelReader
{
    private const int SignatureSizeBytes = 4;
    private const int MinimumHeaderSizeBytes = 0xD0;
    private const int VersionOffset = 0x04;
    private const int NameCountOffset = 0x08;
    private const int NameOffsetOffset = 0x0C;
    private const int EmbeddedSkinProfileCountOffset = 0x4C;
    private const int EmbeddedSkinProfileOffsetOffset = 0x50;
    private const int BoundsOffset = 0xB4;
    private const int BoundsRadiusOffset = 0xCC;

    public static M2ModelDocument Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static M2ModelDocument Read(Stream stream, string sourcePath)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        if (!stream.CanSeek)
            throw new ArgumentException("M2 model reading requires a seekable stream.", nameof(stream));

        if (stream.Length < MinimumHeaderSizeBytes)
            throw new InvalidDataException($"M2 file '{sourcePath}' is too small to contain a strict MD20 header.");

        M2ModelIdentity identity = M2ModelIdentity.FromPath(sourcePath);
        long previousPosition = stream.Position;
        try
        {
            stream.Position = 0;
            Span<byte> signatureBytes = stackalloc byte[SignatureSizeBytes];
            stream.ReadExactly(signatureBytes);

            string signature = Encoding.ASCII.GetString(signatureBytes);
            if (!string.Equals(signature, "MD20", StringComparison.Ordinal))
                throw new InvalidDataException($"M2 file '{sourcePath}' does not contain a strict MD20 root. Found '{FormatSignature(signatureBytes)}'.");

            uint version = ReadUInt32At(stream, VersionOffset);
            string? modelName = TryReadName(stream, sourcePath);
            Vector3 boundsMin = ReadFiniteVector3At(stream, BoundsOffset, sourcePath, "boundsMin");
            Vector3 boundsMax = ReadFiniteVector3At(stream, BoundsOffset + 0x0C, sourcePath, "boundsMax");
            float boundsRadius = ReadFiniteSingleAt(stream, BoundsRadiusOffset, sourcePath, "boundsRadius");
            uint embeddedSkinProfileCount = ReadUInt32At(stream, EmbeddedSkinProfileCountOffset);
            uint embeddedSkinProfileOffset = ReadUInt32At(stream, EmbeddedSkinProfileOffsetOffset);

            return new M2ModelDocument(
                identity,
                signature,
                version,
                modelName,
                boundsMin,
                boundsMax,
                boundsRadius,
                embeddedSkinProfileCount,
                embeddedSkinProfileOffset);
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static string? TryReadName(Stream stream, string sourcePath)
    {
        uint nameCount = ReadUInt32At(stream, NameCountOffset);
        uint nameOffset = ReadUInt32At(stream, NameOffsetOffset);
        if (nameCount == 0 || nameOffset == 0)
            return null;

        ValidateSpan(nameCount, nameOffset, 1, stream.Length, sourcePath, "modelName");

        long previousPosition = stream.Position;
        try
        {
            stream.Position = nameOffset;
            byte[] bytes = new byte[nameCount];
            stream.ReadExactly(bytes);
            int terminator = Array.IndexOf(bytes, (byte)0);
            int length = terminator >= 0 ? terminator : bytes.Length;
            return length == 0 ? null : Encoding.UTF8.GetString(bytes, 0, length);
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static uint ReadUInt32At(Stream stream, int offset)
    {
        long previousPosition = stream.Position;
        try
        {
            stream.Position = offset;
            Span<byte> bytes = stackalloc byte[sizeof(uint)];
            stream.ReadExactly(bytes);
            return BinaryPrimitives.ReadUInt32LittleEndian(bytes);
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static float ReadFiniteSingleAt(Stream stream, int offset, string sourcePath, string label)
    {
        long previousPosition = stream.Position;
        try
        {
            stream.Position = offset;
            Span<byte> bytes = stackalloc byte[sizeof(float)];
            stream.ReadExactly(bytes);
            float value = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(bytes));
            if (!float.IsFinite(value))
                throw new InvalidDataException($"M2 file '{sourcePath}' has a non-finite {label} value.");

            return value;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static Vector3 ReadFiniteVector3At(Stream stream, int offset, string sourcePath, string label)
    {
        return new Vector3(
            ReadFiniteSingleAt(stream, offset + 0x00, sourcePath, $"{label}.x"),
            ReadFiniteSingleAt(stream, offset + 0x04, sourcePath, $"{label}.y"),
            ReadFiniteSingleAt(stream, offset + 0x08, sourcePath, $"{label}.z"));
    }

    private static void ValidateSpan(uint count, uint offset, uint stride, long length, string sourcePath, string label)
    {
        if (count == 0)
            return;

        if (offset == 0)
            throw new InvalidDataException($"M2 file '{sourcePath}' has a zero offset for non-empty span '{label}'.");

        ulong total = (ulong)count * stride;
        ulong end = (ulong)offset + total;
        if ((ulong)offset >= (ulong)length || end > (ulong)length || end < offset)
        {
            throw new InvalidDataException(
                $"M2 file '{sourcePath}' has an out-of-range span for '{label}': count={count}, offset=0x{offset:X}, stride=0x{stride:X}, length=0x{length:X}.");
        }
    }

    private static string FormatSignature(ReadOnlySpan<byte> signature)
    {
        bool isAscii = signature.ToArray().All(static value => value >= 0x20 && value <= 0x7E);
        if (isAscii)
            return Encoding.ASCII.GetString(signature);

        return Convert.ToHexString(signature);
    }
}