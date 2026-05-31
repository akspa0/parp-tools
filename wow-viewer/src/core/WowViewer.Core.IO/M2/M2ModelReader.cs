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
    private const int BoneCountOffset = 0x2C;
    private const int BoneOffsetOffset = 0x30;
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
    private const int CameraCountOffset = 0x110;
    private const int CameraOffsetOffset = 0x114;
    private const int RibbonCountOffset = 0x120;
    private const int RibbonOffsetOffset = 0x124;
    private const int ParticleCountOffset = 0x128;
    private const int ParticleOffsetOffset = 0x12C;
    private const int SequenceStride = 0x40;
    private const int BoneStride = 0x58;
    private const int ColorStride = 0x28;
    private const int TextureWeightStride = 0x14;
    private const int TextureTransformStride = 0x3C;
    private const int LightStride = 0x9C;
    private const int CameraStrideClassic = 0x64;
    private const int CameraStrideModern = 0x74;
    private const int RibbonStrideClassic = 0xAC;
    private const int RibbonStrideModern = 0xB0;
    private const int ParticleStrideClassic = 0x1DC;
    private const int ParticleStrideModern = 0x1EC;
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
        List<M2BoneDefinition> bones = ReadBones(data, globalLoops.Count, sourcePath);
        List<M2ColorDefinition> colors = ReadColors(data, globalLoops.Count, sourcePath);
        List<M2TextureWeightDefinition> textureWeights = ReadTextureWeights(data, globalLoops.Count, sourcePath);
        List<M2TextureTransformDefinition> textureTransforms = ReadTextureTransforms(data, globalLoops.Count, sourcePath);
        List<M2LightDefinition> lights = ReadLights(data, globalLoops.Count, sourcePath);
        List<M2CameraDefinition> cameras = ReadCameras(data, globalLoops.Count, version, sourcePath);
        List<M2RibbonDefinition> ribbons = ReadRibbons(data, globalLoops.Count, sourcePath);
        List<M2ParticleDefinition> particles = ReadParticles(data, globalLoops.Count, version, flags, sourcePath);
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
            cameras,
            boundsMin,
            boundsMax,
            boundsRadius,
            embeddedSkinProfileCount: 0,
            embeddedSkinProfileOffset: 0,
            bones,
            ribbons,
            particles);
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

    private static bool TryReadUInt32At(byte[] data, int offset, out uint value)
    {
        value = 0;
        if (offset < 0 || offset > data.Length - sizeof(uint))
            return false;

        value = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(offset, sizeof(uint)));
        return true;
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

    private static sbyte ReadSByteAt(byte[] data, int offset)
    {
        EnsureReadable(data, offset, sizeof(sbyte), "m2 data", "sbyte");
        return unchecked((sbyte)data[offset]);
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
                return 0f;

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
            return 0f;

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

    private static string? ReadStringAt(byte[] data, string sourcePath, string label, uint count, uint offset)
    {
        if (count == 0)
            return null;

        ValidateSpan(count, offset, sizeof(byte), data.Length, sourcePath, label);
        ReadOnlySpan<byte> bytes = data.AsSpan(checked((int)offset), checked((int)count));
        int terminator = bytes.IndexOf((byte)0);
        int length = terminator >= 0 ? terminator : bytes.Length;
        return length == 0 ? null : Encoding.UTF8.GetString(bytes[..length]);
    }

    private static List<ushort> ReadUInt16Array(byte[] data, string sourcePath, string label, int countOffset, int offsetOffset)
    {
        uint count = ReadUInt32At(data, countOffset);
        uint offset = ReadUInt32At(data, offsetOffset);
        ValidateSpan(count, offset, sizeof(ushort), data.Length, sourcePath, label);

        List<ushort> values = new(checked((int)count));
        for (int index = 0; index < count; index++)
            values.Add(ReadUInt16At(data, checked((int)offset + (index * sizeof(ushort)))));

        return values;
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

    private static List<M2BoneDefinition> ReadBones(byte[] data, int globalLoopCount, string sourcePath)
    {
        uint count = ReadUInt32At(data, BoneCountOffset);
        uint offset = ReadUInt32At(data, BoneOffsetOffset);
        ValidateSpan(count, offset, BoneStride, data.Length, sourcePath, "bones");

        List<M2BoneDefinition> values = new(checked((int)count));
        for (int index = 0; index < count; index++)
        {
            int entryOffset = checked((int)offset + (index * BoneStride));
            values.Add(new M2BoneDefinition(
                index,
                BinaryPrimitives.ReadInt32LittleEndian(data.AsSpan(entryOffset + 0x00, sizeof(int))),
                ReadUInt32At(data, entryOffset + 0x04),
                ReadInt16At(data, entryOffset + 0x08),
                ReadUInt16At(data, entryOffset + 0x0A),
                ReadUInt32At(data, entryOffset + 0x0C),
                ReadTrackDefinition<Vector3>(data, entryOffset + 0x10, globalLoopCount, sourcePath, $"bones[{index}].translation"),
                ReadTrackDefinition<M2CompQuaternion>(data, entryOffset + 0x24, globalLoopCount, sourcePath, $"bones[{index}].rotation"),
                ReadTrackDefinition<Vector3>(data, entryOffset + 0x38, globalLoopCount, sourcePath, $"bones[{index}].scaling"),
                ReadFiniteVector3At(data, entryOffset + 0x4C, sourcePath, $"bones[{index}].pivot")));
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

    private static List<M2CameraDefinition> ReadCameras(byte[] data, int globalLoopCount, uint version, string sourcePath)
    {
        if (!TryReadUInt32At(data, CameraCountOffset, out uint count)
            || !TryReadUInt32At(data, CameraOffsetOffset, out uint offset)
            || count == 0)
        {
            return [];
        }

        int preferredStride = version > 264u ? CameraStrideModern : CameraStrideClassic;
        int fallbackStride = preferredStride == CameraStrideModern ? CameraStrideClassic : CameraStrideModern;
        int stride = ResolveAvailableStride(count, offset, data.Length, preferredStride, fallbackStride, sourcePath, "cameras");

        List<M2CameraDefinition> values = new(checked((int)count));
        for (int index = 0; index < count; index++)
        {
            int entryOffset = checked((int)offset + (index * stride));
            bool modernCamera = stride >= CameraStrideModern;
            float? staticFieldOfView = modernCamera
                ? null
                : ReadFiniteSingleAt(data, entryOffset + 0x04, sourcePath, $"cameras[{index}].fieldOfView");
            int farClipOffset = modernCamera ? 0x04 : 0x08;
            int nearClipOffset = modernCamera ? 0x08 : 0x0C;
            int positionTrackOffset = modernCamera ? 0x0C : 0x10;
            int positionBaseOffset = modernCamera ? 0x20 : 0x24;
            int targetTrackOffset = modernCamera ? 0x2C : 0x30;
            int targetBaseOffset = modernCamera ? 0x40 : 0x44;
            int rollTrackOffset = modernCamera ? 0x4C : 0x50;
            int fieldOfViewTrackOffset = modernCamera ? 0x60 : -1;

            values.Add(new M2CameraDefinition(
                index,
                unchecked((int)ReadUInt32At(data, entryOffset + 0x00)),
                staticFieldOfView,
                ReadFiniteSingleAt(data, entryOffset + farClipOffset, sourcePath, $"cameras[{index}].farClip"),
                ReadFiniteSingleAt(data, entryOffset + nearClipOffset, sourcePath, $"cameras[{index}].nearClip"),
                ReadTrackDefinition<Vector3>(data, entryOffset + positionTrackOffset, globalLoopCount, sourcePath, $"cameras[{index}].positionTrack"),
                ReadFiniteVector3At(data, entryOffset + positionBaseOffset, sourcePath, $"cameras[{index}].positionBase"),
                ReadTrackDefinition<Vector3>(data, entryOffset + targetTrackOffset, globalLoopCount, sourcePath, $"cameras[{index}].targetPositionTrack"),
                ReadFiniteVector3At(data, entryOffset + targetBaseOffset, sourcePath, $"cameras[{index}].targetPositionBase"),
                ReadTrackDefinition<float>(data, entryOffset + rollTrackOffset, globalLoopCount, sourcePath, $"cameras[{index}].rollTrack"),
                modernCamera
                    ? ReadTrackDefinition<float>(data, entryOffset + fieldOfViewTrackOffset, globalLoopCount, sourcePath, $"cameras[{index}].fieldOfViewTrack")
                    : null));
        }

        return values;
    }

    private static List<M2RibbonDefinition> ReadRibbons(byte[] data, int globalLoopCount, string sourcePath)
    {
        if (!TryReadUInt32At(data, RibbonCountOffset, out uint count)
            || !TryReadUInt32At(data, RibbonOffsetOffset, out uint offset)
            || !TryResolveOptionalStride(count, offset, data.Length, RibbonStrideModern, RibbonStrideClassic, out int stride))
        {
            return [];
        }

        List<M2RibbonDefinition> values = new(checked((int)count));
        for (int index = 0; index < count; index++)
        {
            int entryOffset = checked((int)offset + (index * stride));
            values.Add(new M2RibbonDefinition(
                index,
                ReadUInt32At(data, entryOffset + 0x00),
                ReadUInt32At(data, entryOffset + 0x04),
                ReadFiniteVector3At(data, entryOffset + 0x08, sourcePath, $"ribbons[{index}].position"),
                ReadUInt16Array(data, sourcePath, $"ribbons[{index}].textureIndices", entryOffset + 0x14, entryOffset + 0x18),
                ReadUInt16Array(data, sourcePath, $"ribbons[{index}].materialIndices", entryOffset + 0x1C, entryOffset + 0x20),
                ReadTrackDefinition<Vector3>(data, entryOffset + 0x24, globalLoopCount, sourcePath, $"ribbons[{index}].color"),
                ReadTrackDefinition<short>(data, entryOffset + 0x38, globalLoopCount, sourcePath, $"ribbons[{index}].alpha"),
                ReadTrackDefinition<float>(data, entryOffset + 0x4C, globalLoopCount, sourcePath, $"ribbons[{index}].heightAbove"),
                ReadTrackDefinition<float>(data, entryOffset + 0x60, globalLoopCount, sourcePath, $"ribbons[{index}].heightBelow"),
                ReadFiniteSingleAt(data, entryOffset + 0x74, sourcePath, $"ribbons[{index}].edgesPerSecond"),
                ReadFiniteSingleAt(data, entryOffset + 0x78, sourcePath, $"ribbons[{index}].edgeLifetime"),
                ReadFiniteSingleAt(data, entryOffset + 0x7C, sourcePath, $"ribbons[{index}].gravity"),
                ReadUInt16At(data, entryOffset + 0x80),
                ReadUInt16At(data, entryOffset + 0x82),
                ReadTrackDefinition<ushort>(data, entryOffset + 0x84, globalLoopCount, sourcePath, $"ribbons[{index}].textureSlot"),
                ReadTrackDefinition<byte>(data, entryOffset + 0x98, globalLoopCount, sourcePath, $"ribbons[{index}].visibility"),
                stride >= RibbonStrideModern ? ReadInt16At(data, entryOffset + 0xAC) : (short)0,
                stride >= RibbonStrideModern ? ReadSByteAt(data, entryOffset + 0xAE) : (sbyte)-1,
                stride >= RibbonStrideModern ? ReadSByteAt(data, entryOffset + 0xAF) : (sbyte)-1));
        }

        return values;
    }

    private static List<M2ParticleDefinition> ReadParticles(byte[] data, int globalLoopCount, uint version, uint flags, string sourcePath)
    {
        if (!TryReadUInt32At(data, ParticleCountOffset, out uint count)
            || !TryReadUInt32At(data, ParticleOffsetOffset, out uint offset))
        {
            return [];
        }

        int preferredStride = ((flags & 0x200u) != 0 || version > 271u) ? ParticleStrideModern : ParticleStrideClassic;
        int fallbackStride = preferredStride == ParticleStrideModern ? ParticleStrideClassic : ParticleStrideModern;
        if (!TryResolveOptionalStride(count, offset, data.Length, preferredStride, fallbackStride, out int stride))
            return [];

        List<M2ParticleDefinition> values = new(checked((int)count));
        for (int index = 0; index < count; index++)
        {
            int entryOffset = checked((int)offset + (index * stride));
            ushort blendingType = version >= 262u
                ? ReadByteAt(data, entryOffset + 0x28)
                : ReadUInt16At(data, entryOffset + 0x28);
            ushort emitterType = version >= 262u
                ? ReadByteAt(data, entryOffset + 0x29)
                : ReadUInt16At(data, entryOffset + 0x2A);
            ushort particleColorIndex = version >= 262u
                ? ReadUInt16At(data, entryOffset + 0x2A)
                : (ushort)0;

            values.Add(new M2ParticleDefinition(
                index,
                ReadUInt32At(data, entryOffset + 0x00),
                ReadUInt32At(data, entryOffset + 0x04),
                ReadFiniteVector3At(data, entryOffset + 0x08, sourcePath, $"particles[{index}].position"),
                ReadUInt16At(data, entryOffset + 0x14),
                ReadUInt16At(data, entryOffset + 0x16),
                ReadStringAt(data, sourcePath, $"particles[{index}].geometryModel", ReadUInt32At(data, entryOffset + 0x18), ReadUInt32At(data, entryOffset + 0x1C)),
                ReadStringAt(data, sourcePath, $"particles[{index}].recursionModel", ReadUInt32At(data, entryOffset + 0x20), ReadUInt32At(data, entryOffset + 0x24)),
                blendingType,
                emitterType,
                particleColorIndex,
                ReadByteAt(data, entryOffset + 0x2C),
                ReadByteAt(data, entryOffset + 0x2D),
                ReadInt16At(data, entryOffset + 0x2E),
                ReadUInt16At(data, entryOffset + 0x30),
                ReadUInt16At(data, entryOffset + 0x32),
                ReadTrackDefinition<float>(data, entryOffset + 0x34, globalLoopCount, sourcePath, $"particles[{index}].emissionSpeed"),
                ReadTrackDefinition<float>(data, entryOffset + 0x48, globalLoopCount, sourcePath, $"particles[{index}].speedVariation"),
                ReadTrackDefinition<float>(data, entryOffset + 0x5C, globalLoopCount, sourcePath, $"particles[{index}].verticalRange"),
                ReadTrackDefinition<float>(data, entryOffset + 0x70, globalLoopCount, sourcePath, $"particles[{index}].horizontalRange"),
                ReadTrackDefinition<float>(data, entryOffset + 0x84, globalLoopCount, sourcePath, $"particles[{index}].gravity"),
                ReadTrackDefinition<float>(data, entryOffset + 0x98, globalLoopCount, sourcePath, $"particles[{index}].lifespan"),
                ReadTrackDefinition<float>(data, entryOffset + 0xB0, globalLoopCount, sourcePath, $"particles[{index}].emissionRate"),
                ReadTrackDefinition<float>(data, entryOffset + 0xC8, globalLoopCount, sourcePath, $"particles[{index}].emissionAreaLength"),
                ReadTrackDefinition<float>(data, entryOffset + 0xDC, globalLoopCount, sourcePath, $"particles[{index}].emissionAreaWidth"),
                ReadTrackDefinition<float>(data, entryOffset + 0xF0, globalLoopCount, sourcePath, $"particles[{index}].zSource"),
                ReadTrackDefinition<byte>(data, entryOffset + 0x1C8, globalLoopCount, sourcePath, $"particles[{index}].enabled")));
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

    private static int ResolveAvailableStride(uint count, uint offset, long length, int preferredStride, int fallbackStride, string sourcePath, string label)
    {
        if (count == 0)
            return preferredStride;

        if (offset == 0)
            throw new InvalidDataException($"M2 file '{sourcePath}' has a zero offset for non-empty span '{label}'.");

        if (SpanFits(count, offset, preferredStride, length))
            return preferredStride;

        if (SpanFits(count, offset, fallbackStride, length))
            return fallbackStride;

        ValidateSpan(count, offset, (uint)preferredStride, length, sourcePath, label);
        return preferredStride;
    }

    private static bool TryResolveOptionalStride(uint count, uint offset, long length, int preferredStride, int fallbackStride, out int stride)
    {
        stride = preferredStride;
        if (count == 0)
            return false;

        if (offset == 0)
            return false;

        if (SpanFits(count, offset, preferredStride, length))
        {
            stride = preferredStride;
            return true;
        }

        if (SpanFits(count, offset, fallbackStride, length))
        {
            stride = fallbackStride;
            return true;
        }

        return false;
    }

    private static bool SpanFits(uint count, uint offset, int stride, long length)
    {
        if (count == 0)
            return true;

        if (offset == 0 || stride <= 0)
            return false;

        ulong total = (ulong)count * (uint)stride;
        ulong end = (ulong)offset + total;
        return (ulong)offset < (ulong)length && end <= (ulong)length && end >= offset;
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
