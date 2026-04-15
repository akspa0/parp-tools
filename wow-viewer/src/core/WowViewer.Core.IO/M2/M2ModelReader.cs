using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.M2;

namespace WowViewer.Core.IO.M2;

public static class M2ModelReader
{
    private const int SignatureSizeBytes = 4;
    private const int MinimumHeaderSizeBytes = 0x110;
    private const int VersionOffset = 0x04;
    private const int NameCountOffset = 0x08;
    private const int NameOffsetOffset = 0x0C;
    private const int FlagsOffset = 0x10;
    private const int GlobalLoopCountOffset = 0x14;
    private const int GlobalLoopOffsetOffset = 0x18;
    private const int SequenceCountOffset = 0x1C;
    private const int SequenceOffsetOffset = 0x20;
    private const int SequenceLookupCountOffset = 0x24;
    private const int SequenceLookupOffsetOffset = 0x28;
    private const int ViewCountOffset = 0x44;
    private const int ColorCountOffset = 0x48;
    private const int ColorOffsetOffset = 0x4C;
    private const int TextureWeightCountOffset = 0x58;
    private const int TextureWeightOffsetOffset = 0x5C;
    private const int TextureTransformCountOffset = 0x60;
    private const int TextureTransformOffsetOffset = 0x64;
    private const int BoundsOffset = 0xA0;
    private const int BoundsRadiusOffset = 0xB8;
    private const int LightCountOffset = 0x108;
    private const int LightOffsetOffset = 0x10C;
    private const int SequenceStride = 0x40;
    private const int ColorStride = 0x28;
    private const int TextureWeightStride = 0x14;
    private const int TextureTransformStride = 0x3C;
    private const int LightStride = 0x9C;
    private const int TrackStride = 0x14;

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

        byte[] data = ReadAllBytes(stream);
        if (data.Length < MinimumHeaderSizeBytes)
            throw new InvalidDataException($"M2 file '{sourcePath}' is too small to contain a strict MD20 header.");

        M2ModelIdentity identity = M2ModelIdentity.FromPath(sourcePath);
        using MemoryStream dataStream = new(data, writable: false);
        Span<byte> signatureBytes = stackalloc byte[SignatureSizeBytes];
        dataStream.ReadExactly(signatureBytes);

        string signature = Encoding.ASCII.GetString(signatureBytes);
        if (!string.Equals(signature, "MD20", StringComparison.Ordinal))
        {
            throw new InvalidDataException($"M2 file '{sourcePath}' does not contain a strict MD20 root. Found '{FormatSignature(signatureBytes)}'.");
        }

        uint version = ReadUInt32At(dataStream, VersionOffset);
        uint flags = ReadUInt32At(dataStream, FlagsOffset);
        uint viewCount = ReadUInt32At(dataStream, ViewCountOffset);
        string? modelName = TryReadName(dataStream, sourcePath);
        List<uint> globalLoops = ReadUInt32Table(dataStream, sourcePath, "globalLoops", GlobalLoopCountOffset, GlobalLoopOffsetOffset);
        List<M2SequenceDefinition> sequences = ReadSequences(dataStream, sourcePath);
        List<short> sequenceLookup = ReadInt16Table(dataStream, sourcePath, "sequenceLookup", SequenceLookupCountOffset, SequenceLookupOffsetOffset);
        List<M2ColorDefinition> colors = ReadColors(data, globalLoops.Count, sourcePath);
        List<M2TextureWeightDefinition> textureWeights = ReadTextureWeights(data, globalLoops.Count, sourcePath);
        List<M2TextureTransformDefinition> textureTransforms = ReadTextureTransforms(data, globalLoops.Count, sourcePath);
        List<M2LightDefinition> lights = ReadLights(data, globalLoops.Count, sourcePath);
        Vector3 boundsMin = ReadFiniteVector3At(data, BoundsOffset, sourcePath, "boundsMin");
        Vector3 boundsMax = ReadFiniteVector3At(data, BoundsOffset + 0x0C, sourcePath, "boundsMax");
        float boundsRadius = ReadFiniteSingleAt(data, BoundsRadiusOffset, sourcePath, "boundsRadius");

        return new M2ModelDocument(
            identity,
            data,
            signature,
            version,
            flags,
            viewCount,
            modelName,
            globalLoops,
            sequences,
            sequenceLookup,
            colors,
            textureWeights,
            textureTransforms,
            lights,
            boundsMin,
            boundsMax,
            boundsRadius,
            embeddedSkinProfileCount: 0,
            embeddedSkinProfileOffset: 0);
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

    private static uint ReadUInt32At(byte[] data, int offset)
    {
        EnsureReadable(data, offset, sizeof(uint), "m2 data", "uint32");
        return BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(offset, sizeof(uint)));
    }

    private static ushort ReadUInt16At(Stream stream, int offset)
    {
        long previousPosition = stream.Position;
        try
        {
            stream.Position = offset;
            Span<byte> bytes = stackalloc byte[sizeof(ushort)];
            stream.ReadExactly(bytes);
            return BinaryPrimitives.ReadUInt16LittleEndian(bytes);
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static ushort ReadUInt16At(byte[] data, int offset)
    {
        EnsureReadable(data, offset, sizeof(ushort), "m2 data", "uint16");
        return BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(offset, sizeof(ushort)));
    }

    private static short ReadInt16At(Stream stream, int offset)
    {
        long previousPosition = stream.Position;
        try
        {
            stream.Position = offset;
            Span<byte> bytes = stackalloc byte[sizeof(short)];
            stream.ReadExactly(bytes);
            return BinaryPrimitives.ReadInt16LittleEndian(bytes);
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static short ReadInt16At(byte[] data, int offset)
    {
        EnsureReadable(data, offset, sizeof(short), "m2 data", "int16");
        return BinaryPrimitives.ReadInt16LittleEndian(data.AsSpan(offset, sizeof(short)));
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

    private static float ReadFiniteSingleAt(byte[] data, int offset, string sourcePath, string label)
    {
        EnsureReadable(data, offset, sizeof(float), sourcePath, label);
        float value = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(data.AsSpan(offset, sizeof(float))));
        if (!float.IsFinite(value))
            throw new InvalidDataException($"M2 file '{sourcePath}' has a non-finite {label} value.");

        return value;
    }

    private static Vector3 ReadFiniteVector3At(Stream stream, int offset, string sourcePath, string label)
    {
        return new Vector3(
            ReadFiniteSingleAt(stream, offset + 0x00, sourcePath, $"{label}.x"),
            ReadFiniteSingleAt(stream, offset + 0x04, sourcePath, $"{label}.y"),
            ReadFiniteSingleAt(stream, offset + 0x08, sourcePath, $"{label}.z"));
    }

    private static Vector3 ReadFiniteVector3At(byte[] data, int offset, string sourcePath, string label)
    {
        return new Vector3(
            ReadFiniteSingleAt(data, offset + 0x00, sourcePath, $"{label}.x"),
            ReadFiniteSingleAt(data, offset + 0x04, sourcePath, $"{label}.y"),
            ReadFiniteSingleAt(data, offset + 0x08, sourcePath, $"{label}.z"));
    }

    private static Quaternion ReadFiniteQuaternionAt(byte[] data, int offset, string sourcePath, string label)
    {
        Quaternion value = new(
            ReadFiniteSingleAt(data, offset + 0x00, sourcePath, $"{label}.x"),
            ReadFiniteSingleAt(data, offset + 0x04, sourcePath, $"{label}.y"),
            ReadFiniteSingleAt(data, offset + 0x08, sourcePath, $"{label}.z"),
            ReadFiniteSingleAt(data, offset + 0x0C, sourcePath, $"{label}.w"));

        return Quaternion.Normalize(value);
    }

    private static byte ReadByteAt(byte[] data, int offset)
    {
        EnsureReadable(data, offset, sizeof(byte), "m2 data", "byte");
        return data[offset];
    }

    private static List<uint> ReadUInt32Table(Stream stream, string sourcePath, string label, int countOffset, int offsetOffset)
    {
        uint count = ReadUInt32At(stream, countOffset);
        uint offset = ReadUInt32At(stream, offsetOffset);
        ValidateSpan(count, offset, sizeof(uint), stream.Length, sourcePath, label);

        List<uint> values = new(checked((int)count));
        for (int index = 0; index < count; index++)
            values.Add(ReadUInt32At(stream, checked((int)offset + (index * sizeof(uint)))));

        return values;
    }

    private static List<short> ReadInt16Table(Stream stream, string sourcePath, string label, int countOffset, int offsetOffset)
    {
        uint count = ReadUInt32At(stream, countOffset);
        uint offset = ReadUInt32At(stream, offsetOffset);
        ValidateSpan(count, offset, sizeof(short), stream.Length, sourcePath, label);

        List<short> values = new(checked((int)count));
        for (int index = 0; index < count; index++)
            values.Add(ReadInt16At(stream, checked((int)offset + (index * sizeof(short)))));

        return values;
    }

    private static List<M2SequenceDefinition> ReadSequences(Stream stream, string sourcePath)
    {
        uint count = ReadUInt32At(stream, SequenceCountOffset);
        uint offset = ReadUInt32At(stream, SequenceOffsetOffset);
        ValidateSpan(count, offset, SequenceStride, stream.Length, sourcePath, "sequences");

        List<M2SequenceDefinition> values = new(checked((int)count));
        for (int index = 0; index < count; index++)
        {
            int entryOffset = checked((int)offset + (index * SequenceStride));
            values.Add(new M2SequenceDefinition(
                index,
                ReadUInt16At(stream, entryOffset + 0x00),
                ReadUInt16At(stream, entryOffset + 0x02),
                ReadUInt32At(stream, entryOffset + 0x04),
                ReadFiniteSingleAt(stream, entryOffset + 0x08, sourcePath, $"sequence[{index}].moveSpeed"),
                ReadUInt32At(stream, entryOffset + 0x0C),
                ReadInt16At(stream, entryOffset + 0x10),
                ReadUInt32At(stream, entryOffset + 0x14),
                ReadUInt32At(stream, entryOffset + 0x18),
                ReadUInt16At(stream, entryOffset + 0x1C),
                ReadUInt16At(stream, entryOffset + 0x1E),
                ReadFiniteVector3At(stream, entryOffset + 0x20, sourcePath, $"sequence[{index}].boundsMin"),
                ReadFiniteVector3At(stream, entryOffset + 0x2C, sourcePath, $"sequence[{index}].boundsMax"),
                ReadFiniteSingleAt(stream, entryOffset + 0x38, sourcePath, $"sequence[{index}].boundsRadius"),
                ReadInt16At(stream, entryOffset + 0x3C),
                ReadUInt16At(stream, entryOffset + 0x3E)));
        }

        return values;
    }

    private static List<M2ColorDefinition> ReadColors(byte[] data, int globalLoopCount, string sourcePath)
    {
        uint count = ReadUInt32At(data, ColorCountOffset);
        uint offset = ReadUInt32At(data, ColorOffsetOffset);
        ValidateSpan(count, offset, ColorStride, data.Length, sourcePath, "colors");

        List<M2ColorDefinition> values = new(checked((int)count));
        for (int index = 0; index < count; index++)
        {
            int entryOffset = checked((int)offset + (index * ColorStride));
            values.Add(new M2ColorDefinition(
                index,
                ReadTrackDefinition<Vector3>(data, entryOffset + 0x00, globalLoopCount, sourcePath, $"colors[{index}].color"),
                ReadTrackDefinition<short>(data, entryOffset + 0x14, globalLoopCount, sourcePath, $"colors[{index}].alpha")));
        }

        return values;
    }

    private static List<M2TextureWeightDefinition> ReadTextureWeights(byte[] data, int globalLoopCount, string sourcePath)
    {
        uint count = ReadUInt32At(data, TextureWeightCountOffset);
        uint offset = ReadUInt32At(data, TextureWeightOffsetOffset);
        ValidateSpan(count, offset, TextureWeightStride, data.Length, sourcePath, "textureWeights");

        List<M2TextureWeightDefinition> values = new(checked((int)count));
        for (int index = 0; index < count; index++)
        {
            int entryOffset = checked((int)offset + (index * TextureWeightStride));
            values.Add(new M2TextureWeightDefinition(
                index,
                ReadTrackDefinition<short>(data, entryOffset, globalLoopCount, sourcePath, $"textureWeights[{index}].weight")));
        }

        return values;
    }

    private static List<M2TextureTransformDefinition> ReadTextureTransforms(byte[] data, int globalLoopCount, string sourcePath)
    {
        uint count = ReadUInt32At(data, TextureTransformCountOffset);
        uint offset = ReadUInt32At(data, TextureTransformOffsetOffset);
        ValidateSpan(count, offset, TextureTransformStride, data.Length, sourcePath, "textureTransforms");

        List<M2TextureTransformDefinition> values = new(checked((int)count));
        for (int index = 0; index < count; index++)
        {
            int entryOffset = checked((int)offset + (index * TextureTransformStride));
            values.Add(new M2TextureTransformDefinition(
                index,
                ReadTrackDefinition<Vector3>(data, entryOffset + 0x00, globalLoopCount, sourcePath, $"textureTransforms[{index}].translation"),
                ReadTrackDefinition<Quaternion>(data, entryOffset + 0x14, globalLoopCount, sourcePath, $"textureTransforms[{index}].rotation"),
                ReadTrackDefinition<Vector3>(data, entryOffset + 0x28, globalLoopCount, sourcePath, $"textureTransforms[{index}].scaling")));
        }

        return values;
    }

    private static List<M2LightDefinition> ReadLights(byte[] data, int globalLoopCount, string sourcePath)
    {
        uint count = ReadUInt32At(data, LightCountOffset);
        uint offset = ReadUInt32At(data, LightOffsetOffset);
        ValidateSpan(count, offset, LightStride, data.Length, sourcePath, "lights");

        List<M2LightDefinition> values = new(checked((int)count));
        for (int index = 0; index < count; index++)
        {
            int entryOffset = checked((int)offset + (index * LightStride));
            values.Add(new M2LightDefinition(
                index,
                ReadUInt16At(data, entryOffset + 0x00),
                ReadInt16At(data, entryOffset + 0x02),
                ReadFiniteVector3At(data, entryOffset + 0x04, sourcePath, $"lights[{index}].position"),
                ReadTrackDefinition<Vector3>(data, entryOffset + 0x10, globalLoopCount, sourcePath, $"lights[{index}].ambientColor"),
                ReadTrackDefinition<float>(data, entryOffset + 0x24, globalLoopCount, sourcePath, $"lights[{index}].ambientIntensity"),
                ReadTrackDefinition<Vector3>(data, entryOffset + 0x38, globalLoopCount, sourcePath, $"lights[{index}].diffuseColor"),
                ReadTrackDefinition<float>(data, entryOffset + 0x4C, globalLoopCount, sourcePath, $"lights[{index}].diffuseIntensity"),
                ReadTrackDefinition<float>(data, entryOffset + 0x60, globalLoopCount, sourcePath, $"lights[{index}].attenuationStart"),
                ReadTrackDefinition<float>(data, entryOffset + 0x74, globalLoopCount, sourcePath, $"lights[{index}].attenuationEnd"),
                ReadTrackDefinition<byte>(data, entryOffset + 0x88, globalLoopCount, sourcePath, $"lights[{index}].visibility")));
        }

        return values;
    }

    private static M2TrackDefinition<T> ReadTrackDefinition<T>(byte[] data, int offset, int globalLoopCount, string sourcePath, string label)
    {
        EnsureReadable(data, offset, TrackStride, sourcePath, label);

        ushort firstHeader = ReadUInt16At(data, offset + 0x00);
        ushort secondHeader = ReadUInt16At(data, offset + 0x02);
        (M2TrackInterpolation interpolation, int globalSequenceIndex) = NormalizeTrackHeader(firstHeader, secondHeader, globalLoopCount);

        M2TrackArrayReference timestampArray = new(
            ReadUInt32At(data, offset + 0x04),
            ReadUInt32At(data, offset + 0x08));
        M2TrackArrayReference valueArray = new(
            ReadUInt32At(data, offset + 0x0C),
            ReadUInt32At(data, offset + 0x10));

        return new M2TrackDefinition<T>(interpolation, globalSequenceIndex, timestampArray, valueArray);
    }

    private static (M2TrackInterpolation Interpolation, int GlobalSequenceIndex) NormalizeTrackHeader(ushort first, ushort second, int globalLoopCount)
    {
        static bool IsInterpolationCandidate(ushort value) => value <= 3;
        static int NormalizeGlobalSequence(ushort value, int count) => value == ushort.MaxValue || value >= count ? -1 : value;

        bool firstInterpolation = IsInterpolationCandidate(first);
        bool secondInterpolation = IsInterpolationCandidate(second);

        if (firstInterpolation && (!secondInterpolation || second == ushort.MaxValue))
            return ((M2TrackInterpolation)first, NormalizeGlobalSequence(second, globalLoopCount));

        if (secondInterpolation && (!firstInterpolation || first == ushort.MaxValue))
            return ((M2TrackInterpolation)second, NormalizeGlobalSequence(first, globalLoopCount));

        if (firstInterpolation)
            return ((M2TrackInterpolation)first, NormalizeGlobalSequence(second, globalLoopCount));

        if (secondInterpolation)
            return ((M2TrackInterpolation)second, NormalizeGlobalSequence(first, globalLoopCount));

        return (M2TrackInterpolation.Linear, NormalizeGlobalSequence(second, globalLoopCount));
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

    private static void EnsureReadable(byte[] data, int offset, int size, string sourcePath, string label)
    {
        if (offset < 0 || size < 0 || offset > data.Length - size)
            throw new InvalidDataException($"M2 file '{sourcePath}' does not contain a readable span for '{label}' at offset 0x{offset:X} with size 0x{size:X}.");
    }

    private static byte[] ReadAllBytes(Stream stream)
    {
        long previousPosition = stream.Position;
        try
        {
            stream.Position = 0;
            byte[] data = new byte[checked((int)stream.Length)];
            stream.ReadExactly(data);
            return data;
        }
        finally
        {
            stream.Position = previousPosition;
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