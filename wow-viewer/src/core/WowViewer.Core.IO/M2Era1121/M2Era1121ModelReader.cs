using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.M2;

namespace WowViewer.Core.IO.M2Era1121;

public static class M2Era1121ModelReader
{
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
            throw new ArgumentException("1.12.1 M2 model reading requires a seekable stream.", nameof(stream));

        byte[] data = ReadAllBytes(stream);
        if (data.Length < M2Era1121Constants.DispatchHeaderSizeBytes)
            throw new InvalidDataException($"1.12.1 M2 file '{sourcePath}' is too small to contain a magic+version pair.");

        uint magic = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(0, sizeof(uint)));
        if (magic != M2Era1121Constants.Md20Magic)
            throw new InvalidDataException($"1.12.1 M2 file '{sourcePath}' does not contain a strict MD20 root.");

        if (data.Length < M2Era1121Constants.MinimumHeaderSizeBytes)
            throw new InvalidDataException($"1.12.1 M2 file '{sourcePath}' is too small to contain a strict 1.12.1 MD20 header.");

        uint rawVersion = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(M2Era1121Constants.VersionOffset, sizeof(uint)));
        M2Era1121Version version = M2Era1121VersionExtensions.FromUInt(rawVersion);
        if (!version.Is1121())
        {
            throw new NotSupportedException(
                $"1.12.1 M2 file '{sourcePath}' has unsupported version 0x{rawVersion:X}. Expected 0x100 or 0x101.");
        }

        return ParseM2(data, sourcePath, version);
    }

    private static M2ModelDocument ParseM2(byte[] data, string sourcePath, M2Era1121Version version)
    {
        uint flags = ReadUInt32At(data, M2Era1121Constants.FlagsOffset);
        uint viewCount = ReadUInt32At(data, M2Era1121Constants.ViewCountOffset);
        string? modelName = TryReadName(data, sourcePath);
        List<uint> globalLoops = ReadUInt32Table(data, sourcePath, "globalLoops",
            M2Era1121Constants.GlobalLoopCountOffset, M2Era1121Constants.GlobalLoopOffsetOffset);

        M2ModelIdentity identity = M2ModelIdentity.FromPath(sourcePath);
        IReadOnlyList<M2SequenceDefinition> sequences = ReadSequences(data, sourcePath);
        IReadOnlyList<short> sequenceLookup = ReadInt16Table(data, sourcePath, "sequenceLookup",
            M2Era1121Constants.SequenceLookupCountOffset, M2Era1121Constants.SequenceLookupOffsetOffset);
        IReadOnlyList<M2ColorDefinition> colors = ReadColors(data, globalLoops.Count, sourcePath);
        IReadOnlyList<M2TextureWeightDefinition> textureWeights = ReadTextureWeights(data, globalLoops.Count, sourcePath);
        IReadOnlyList<M2TextureTransformDefinition> textureTransforms = ReadTextureTransforms(data, globalLoops.Count, sourcePath);
        IReadOnlyList<M2LightDefinition> lights = ReadLights(data, globalLoops.Count, sourcePath);
        IReadOnlyList<M2CameraDefinition>? cameras = ReadCameras(data, globalLoops.Count, sourcePath);
        IReadOnlyList<M2RibbonDefinition> ribbons = ReadRibbons(data, globalLoops.Count, sourcePath);
        IReadOnlyList<M2ParticleDefinition> particles = ReadParticles(data, globalLoops.Count, sourcePath);

        Vector3 boundsMin = ReadFiniteVector3At(data, M2Era1121Constants.BoundsOffset, sourcePath, "boundsMin");
        Vector3 boundsMax = ReadFiniteVector3At(data, M2Era1121Constants.BoundsOffset + 0x0C, sourcePath, "boundsMax");
        float boundsRadius = ReadFiniteSingleAt(data, M2Era1121Constants.BoundsRadiusOffset, sourcePath, "boundsRadius");

        return new M2ModelDocument(
            identity,
            data,
            "MD20",
            (uint)version,
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
            bones: [],
            ribbons: ribbons,
            particles: particles);
    }

    private static string? TryReadName(byte[] data, string sourcePath)
    {
        uint nameCount = ReadUInt32At(data, M2Era1121Constants.NameCountOffset);
        uint nameOffset = ReadUInt32At(data, M2Era1121Constants.NameOffsetOffset);
        if (nameCount == 0 || nameOffset == 0)
            return null;

        ValidateSpan(nameCount, nameOffset, M2Era1121Constants.NameStride, data.Length, sourcePath, "modelName");

        int nameLen = checked((int)nameCount);
        if (nameOffset >= (uint)data.Length || nameOffset + nameLen > (uint)data.Length)
            return null;

        ReadOnlySpan<byte> bytes = data.AsSpan(checked((int)nameOffset), nameLen);
        int terminator = bytes.IndexOf((byte)0);
        int length = terminator >= 0 ? terminator : bytes.Length;
        return length == 0 ? null : Encoding.UTF8.GetString(bytes[..length]);
    }

    private static IReadOnlyList<M2SequenceDefinition> ReadSequences(byte[] data, string sourcePath)
    {
        uint count = ReadUInt32At(data, M2Era1121Constants.SequenceCountOffset);
        uint offset = ReadUInt32At(data, M2Era1121Constants.SequenceOffsetOffset);
        ValidateSpan(count, offset, M2Era1121Constants.SequenceStride, data.Length, sourcePath, "sequences");

        List<M2SequenceDefinition> values = new(checked((int)count));
        for (int index = 0; index < count; index++)
        {
            int entryOffset = checked((int)offset + (index * M2Era1121Constants.SequenceStride));
            values.Add(new M2SequenceDefinition(
                index,
                ReadUInt16At(data, entryOffset + 0x00),
                ReadUInt16At(data, entryOffset + 0x02),
                ReadUInt32At(data, entryOffset + 0x04),
                ReadFiniteSingleAt(data, entryOffset + 0x08, sourcePath, $"sequence[{index}].moveSpeed"),
                ReadUInt32At(data, entryOffset + 0x0C),
                ReadInt16At(data, entryOffset + 0x10),
                ReadUInt32At(data, entryOffset + 0x14),
                ReadUInt32At(data, entryOffset + 0x18),
                ReadUInt16At(data, entryOffset + 0x1C),
                ReadUInt16At(data, entryOffset + 0x1E),
                ReadFiniteVector3At(data, entryOffset + 0x20, sourcePath, $"sequence[{index}].boundsMin"),
                ReadFiniteVector3At(data, entryOffset + 0x2C, sourcePath, $"sequence[{index}].boundsMax"),
                ReadFiniteSingleAt(data, entryOffset + 0x38, sourcePath, $"sequence[{index}].boundsRadius"),
                ReadInt16At(data, entryOffset + 0x3C),
                ReadUInt16At(data, entryOffset + 0x3E)));
        }

        return values;
    }

    private static IReadOnlyList<M2ColorDefinition> ReadColors(byte[] data, int globalLoopCount, string sourcePath)
    {
        uint count = ReadUInt32At(data, M2Era1121Constants.ColorCountOffset);
        uint offset = ReadUInt32At(data, M2Era1121Constants.ColorOffsetOffset);
        ValidateSpan(count, offset, M2Era1121Constants.ColorStride, data.Length, sourcePath, "colors");

        List<M2ColorDefinition> values = new(checked((int)count));
        for (int index = 0; index < count; index++)
        {
            int entryOffset = checked((int)offset + (index * M2Era1121Constants.ColorStride));
            values.Add(new M2ColorDefinition(
                index,
                ReadTrackDefinition<Vector3>(data, entryOffset + 0x00, globalLoopCount, sourcePath, $"colors[{index}].color"),
                ReadTrackDefinition<short>(data, entryOffset + 0x14, globalLoopCount, sourcePath, $"colors[{index}].alpha")));
        }

        return values;
    }

    private static IReadOnlyList<M2TextureWeightDefinition> ReadTextureWeights(byte[] data, int globalLoopCount, string sourcePath)
    {
        uint count = ReadUInt32At(data, M2Era1121Constants.TexWeightCountOffset);
        uint offset = ReadUInt32At(data, M2Era1121Constants.TexWeightOffsetOffset);
        ValidateSpan(count, offset, M2Era1121Constants.TexWeightStride, data.Length, sourcePath, "textureWeights");

        List<M2TextureWeightDefinition> values = new(checked((int)count));
        for (int index = 0; index < count; index++)
        {
            int entryOffset = checked((int)offset + (index * M2Era1121Constants.TexWeightStride));
            M2TrackDefinition<short> track = ReadTrackDefinition<short>(data, entryOffset, globalLoopCount, sourcePath, $"textureWeights[{index}].weight");
            values.Add(new M2TextureWeightDefinition(index, track));
        }

        return values;
    }

    private static IReadOnlyList<M2TextureTransformDefinition> ReadTextureTransforms(byte[] data, int globalLoopCount, string sourcePath)
    {
        uint count = ReadUInt32At(data, M2Era1121Constants.TexAnimCountOffset);
        uint offset = ReadUInt32At(data, M2Era1121Constants.TexAnimOffsetOffset);
        ValidateSpan(count, offset, M2Era1121Constants.TexAnimStride, data.Length, sourcePath, "textureTransforms");

        List<M2TextureTransformDefinition> values = new(checked((int)count));
        for (int index = 0; index < count; index++)
        {
            int entryOffset = checked((int)offset + (index * M2Era1121Constants.TexAnimStride));
            M2TrackDefinition<short> unused = ReadTrackDefinition<short>(data, entryOffset, globalLoopCount, sourcePath, $"textureTransforms[{index}]");
            _ = unused;
            M2TrackDefinition<Vector3> translation = ReadTrackDefinition<Vector3>(data, entryOffset, globalLoopCount, sourcePath, $"textureTransforms[{index}].translation");
            M2TrackDefinition<Quaternion> rotation = ReadTrackDefinition<Quaternion>(data, entryOffset, globalLoopCount, sourcePath, $"textureTransforms[{index}].rotation");
            M2TrackDefinition<Vector3> scaling = ReadTrackDefinition<Vector3>(data, entryOffset, globalLoopCount, sourcePath, $"textureTransforms[{index}].scaling");
            values.Add(new M2TextureTransformDefinition(index, translation, rotation, scaling));
        }

        return values;
    }

    private static IReadOnlyList<M2LightDefinition> ReadLights(byte[] data, int globalLoopCount, string sourcePath)
    {
        uint count = ReadUInt32At(data, M2Era1121Constants.LightCountOffset);
        uint offset = ReadUInt32At(data, M2Era1121Constants.LightOffsetOffset);
        ValidateSpan(count, offset, M2Era1121Constants.LightStride, data.Length, sourcePath, "lights");

        List<M2LightDefinition> values = new(checked((int)count));
        for (int index = 0; index < count; index++)
        {
            int entryOffset = checked((int)offset + (index * M2Era1121Constants.LightStride));
            Vector3 position = ReadFiniteVector3At(data, entryOffset, sourcePath, $"lights[{index}].position");
            M2TrackDefinition<Vector3> ambientColor = ReadTrackDefinition<Vector3>(data, entryOffset, globalLoopCount, sourcePath, $"lights[{index}].ambientColor");
            M2TrackDefinition<float> ambientIntensity = ReadTrackDefinition<float>(data, entryOffset, globalLoopCount, sourcePath, $"lights[{index}].ambientIntensity");
            M2TrackDefinition<Vector3> diffuseColor = ReadTrackDefinition<Vector3>(data, entryOffset, globalLoopCount, sourcePath, $"lights[{index}].diffuseColor");
            M2TrackDefinition<float> diffuseIntensity = ReadTrackDefinition<float>(data, entryOffset, globalLoopCount, sourcePath, $"lights[{index}].diffuseIntensity");
            M2TrackDefinition<float> attenuationStart = ReadTrackDefinition<float>(data, entryOffset, globalLoopCount, sourcePath, $"lights[{index}].attenuationStart");
            M2TrackDefinition<float> attenuationEnd = ReadTrackDefinition<float>(data, entryOffset, globalLoopCount, sourcePath, $"lights[{index}].attenuationEnd");
            M2TrackDefinition<byte> visibility = ReadTrackDefinition<byte>(data, entryOffset, globalLoopCount, sourcePath, $"lights[{index}].visibility");
            values.Add(new M2LightDefinition(
                index,
                0,
                0,
                position,
                ambientColor,
                ambientIntensity,
                diffuseColor,
                diffuseIntensity,
                attenuationStart,
                attenuationEnd,
                visibility));
        }

        return values;
    }

    private static IReadOnlyList<M2CameraDefinition>? ReadCameras(byte[] data, int globalLoopCount, string sourcePath)
    {
        if (!TryReadUInt32At(data, M2Era1121Constants.CameraCountOffset, out uint count)
            || !TryReadUInt32At(data, M2Era1121Constants.CameraOffsetOffset, out uint offset)
            || count == 0)
        {
            return [];
        }

        ValidateSpan(count, offset, M2Era1121Constants.CameraStride, data.Length, sourcePath, "cameras");

        List<M2CameraDefinition> values = new(checked((int)count));
        for (int index = 0; index < count; index++)
        {
            int entryOffset = checked((int)offset + (index * M2Era1121Constants.CameraStride));
            M2TrackDefinition<Vector3> positionTrack = ReadTrackDefinition<Vector3>(data, entryOffset, globalLoopCount, sourcePath, $"cameras[{index}].positionTrack");
            Vector3 positionBase = ReadFiniteVector3At(data, entryOffset, sourcePath, $"cameras[{index}].positionBase");
            M2TrackDefinition<Vector3> targetTrack = ReadTrackDefinition<Vector3>(data, entryOffset, globalLoopCount, sourcePath, $"cameras[{index}].targetPositionTrack");
            Vector3 targetBase = ReadFiniteVector3At(data, entryOffset, sourcePath, $"cameras[{index}].targetPositionBase");
            M2TrackDefinition<float> rollTrack = ReadTrackDefinition<float>(data, entryOffset, globalLoopCount, sourcePath, $"cameras[{index}].rollTrack");
            values.Add(new M2CameraDefinition(
                index,
                0,
                null,
                0f,
                0f,
                positionTrack,
                positionBase,
                targetTrack,
                targetBase,
                rollTrack));
        }

        return values;
    }

    private static IReadOnlyList<M2RibbonDefinition> ReadRibbons(byte[] data, int globalLoopCount, string sourcePath)
    {
        if (!TryReadUInt32At(data, M2Era1121Constants.RibbonCountOffset, out uint count)
            || !TryReadUInt32At(data, M2Era1121Constants.RibbonOffsetOffset, out uint offset)
            || count == 0)
        {
            return [];
        }

        ValidateSpan(count, offset, M2Era1121Constants.RibbonStride, data.Length, sourcePath, "ribbons");

        List<M2RibbonDefinition> values = new(checked((int)count));
        for (int index = 0; index < count; index++)
        {
            int entryOffset = checked((int)offset + (index * M2Era1121Constants.RibbonStride));
            M2TrackDefinition<Vector3> colorTrack = ReadTrackDefinition<Vector3>(data, entryOffset, globalLoopCount, sourcePath, $"ribbons[{index}].color");
            M2TrackDefinition<short> alphaTrack = ReadTrackDefinition<short>(data, entryOffset, globalLoopCount, sourcePath, $"ribbons[{index}].alpha");
            M2TrackDefinition<float> heightAboveTrack = ReadTrackDefinition<float>(data, entryOffset, globalLoopCount, sourcePath, $"ribbons[{index}].heightAbove");
            M2TrackDefinition<float> heightBelowTrack = ReadTrackDefinition<float>(data, entryOffset, globalLoopCount, sourcePath, $"ribbons[{index}].heightBelow");
            M2TrackDefinition<ushort> textureSlotTrack = ReadTrackDefinition<ushort>(data, entryOffset, globalLoopCount, sourcePath, $"ribbons[{index}].textureSlot");
            M2TrackDefinition<byte> visibilityTrack = ReadTrackDefinition<byte>(data, entryOffset, globalLoopCount, sourcePath, $"ribbons[{index}].visibility");
            values.Add(new M2RibbonDefinition(
                index,
                ReadUInt32At(data, entryOffset + 0x00),
                ReadUInt32At(data, entryOffset + 0x04),
                ReadFiniteVector3At(data, entryOffset + 0x08, sourcePath, $"ribbons[{index}].position"),
                [],
                [],
                colorTrack,
                alphaTrack,
                heightAboveTrack,
                heightBelowTrack,
                ReadFiniteSingleAt(data, entryOffset + 0x74, sourcePath, $"ribbons[{index}].edgesPerSecond"),
                ReadFiniteSingleAt(data, entryOffset + 0x78, sourcePath, $"ribbons[{index}].edgeLifetime"),
                ReadFiniteSingleAt(data, entryOffset + 0x7C, sourcePath, $"ribbons[{index}].gravity"),
                ReadUInt16At(data, entryOffset + 0x80),
                ReadUInt16At(data, entryOffset + 0x82),
                textureSlotTrack,
                visibilityTrack,
                0,
                -1,
                -1));
        }

        return values;
    }

    private static IReadOnlyList<M2ParticleDefinition> ReadParticles(byte[] data, int globalLoopCount, string sourcePath)
    {
        if (!TryReadUInt32At(data, M2Era1121Constants.ParticleCountOffset, out uint count)
            || !TryReadUInt32At(data, M2Era1121Constants.ParticleOffsetOffset, out uint offset)
            || count == 0)
        {
            return [];
        }

        ValidateSpan(count, offset, M2Era1121Constants.ParticleStride, data.Length, sourcePath, "particles");

        List<M2ParticleDefinition> values = new(checked((int)count));
        for (int index = 0; index < count; index++)
        {
            int entryOffset = checked((int)offset + (index * M2Era1121Constants.ParticleStride));
            M2TrackDefinition<float> emissionSpeed = ReadTrackDefinition<float>(data, entryOffset, globalLoopCount, sourcePath, $"particles[{index}].emissionSpeed");
            M2TrackDefinition<float> speedVariation = ReadTrackDefinition<float>(data, entryOffset, globalLoopCount, sourcePath, $"particles[{index}].speedVariation");
            M2TrackDefinition<float> verticalRange = ReadTrackDefinition<float>(data, entryOffset, globalLoopCount, sourcePath, $"particles[{index}].verticalRange");
            M2TrackDefinition<float> horizontalRange = ReadTrackDefinition<float>(data, entryOffset, globalLoopCount, sourcePath, $"particles[{index}].horizontalRange");
            M2TrackDefinition<float> gravity = ReadTrackDefinition<float>(data, entryOffset, globalLoopCount, sourcePath, $"particles[{index}].gravity");
            M2TrackDefinition<float> lifespan = ReadTrackDefinition<float>(data, entryOffset, globalLoopCount, sourcePath, $"particles[{index}].lifespan");
            M2TrackDefinition<float> emissionRate = ReadTrackDefinition<float>(data, entryOffset, globalLoopCount, sourcePath, $"particles[{index}].emissionRate");
            M2TrackDefinition<float> emissionAreaLength = ReadTrackDefinition<float>(data, entryOffset, globalLoopCount, sourcePath, $"particles[{index}].emissionAreaLength");
            M2TrackDefinition<float> emissionAreaWidth = ReadTrackDefinition<float>(data, entryOffset, globalLoopCount, sourcePath, $"particles[{index}].emissionAreaWidth");
            M2TrackDefinition<float> zSource = ReadTrackDefinition<float>(data, entryOffset, globalLoopCount, sourcePath, $"particles[{index}].zSource");
            M2TrackDefinition<byte> enabled = ReadTrackDefinition<byte>(data, entryOffset, globalLoopCount, sourcePath, $"particles[{index}].enabled");
            values.Add(new M2ParticleDefinition(
                index,
                ReadUInt32At(data, entryOffset + 0x00),
                ReadUInt32At(data, entryOffset + 0x04),
                ReadFiniteVector3At(data, entryOffset + 0x08, sourcePath, $"particles[{index}].position"),
                ReadUInt16At(data, entryOffset + 0x14),
                ReadUInt16At(data, entryOffset + 0x16),
                TryReadStringAt(data, sourcePath, $"particles[{index}].geometryModel", ReadUInt32At(data, entryOffset + 0x18), ReadUInt32At(data, entryOffset + 0x1C)),
                TryReadStringAt(data, sourcePath, $"particles[{index}].recursionModel", ReadUInt32At(data, entryOffset + 0x20), ReadUInt32At(data, entryOffset + 0x24)),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                emissionSpeed,
                speedVariation,
                verticalRange,
                horizontalRange,
                gravity,
                lifespan,
                emissionRate,
                emissionAreaLength,
                emissionAreaWidth,
                zSource,
                enabled));
        }

        return values;
    }

    private static M2TrackDefinition<T> ReadTrackDefinition<T>(byte[] data, int offset, int globalLoopCount, string sourcePath, string label)
        where T : struct
    {
        EnsureReadable(data, offset, 0x14, sourcePath, label);

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

    private static List<uint> ReadUInt32Table(byte[] data, string sourcePath, string label, int countOffset, int offsetOffset)
    {
        uint count = ReadUInt32At(data, countOffset);
        uint offset = ReadUInt32At(data, offsetOffset);
        ValidateSpan(count, offset, sizeof(uint), data.Length, sourcePath, label);

        List<uint> values = new(checked((int)count));
        for (int index = 0; index < count; index++)
            values.Add(ReadUInt32At(data, checked((int)offset + (index * sizeof(uint)))));

        return values;
    }

    private static List<short> ReadInt16Table(byte[] data, string sourcePath, string label, int countOffset, int offsetOffset)
    {
        uint count = ReadUInt32At(data, countOffset);
        uint offset = ReadUInt32At(data, offsetOffset);
        ValidateSpan(count, offset, sizeof(short), data.Length, sourcePath, label);

        List<short> values = new(checked((int)count));
        for (int index = 0; index < count; index++)
            values.Add(ReadInt16At(data, checked((int)offset + (index * sizeof(short)))));

        return values;
    }

    private static uint ReadUInt32At(byte[] data, int offset)
    {
        EnsureReadable(data, offset, sizeof(uint), "1.12.1 M2 data", "uint32");
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

    private static ushort ReadUInt16At(byte[] data, int offset)
    {
        EnsureReadable(data, offset, sizeof(ushort), "1.12.1 M2 data", "uint16");
        return BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(offset, sizeof(ushort)));
    }

    private static short ReadInt16At(byte[] data, int offset)
    {
        EnsureReadable(data, offset, sizeof(short), "1.12.1 M2 data", "int16");
        return BinaryPrimitives.ReadInt16LittleEndian(data.AsSpan(offset, sizeof(short)));
    }

    private static float ReadFiniteSingleAt(byte[] data, int offset, string sourcePath, string label)
    {
        EnsureReadable(data, offset, sizeof(float), sourcePath, label);
        float value = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(data.AsSpan(offset, sizeof(float))));
        if (!float.IsFinite(value))
            throw new InvalidDataException($"1.12.1 M2 file '{sourcePath}' has a non-finite {label} value.");

        return value;
    }

    private static Vector3 ReadFiniteVector3At(byte[] data, int offset, string sourcePath, string label)
    {
        return new Vector3(
            ReadFiniteSingleAt(data, offset + 0x00, sourcePath, $"{label}.x"),
            ReadFiniteSingleAt(data, offset + 0x04, sourcePath, $"{label}.y"),
            ReadFiniteSingleAt(data, offset + 0x08, sourcePath, $"{label}.z"));
    }

    private static string? TryReadStringAt(byte[] data, string sourcePath, string label, uint count, uint offset)
    {
        if (count == 0 || offset == 0)
            return null;

        ValidateSpan(count, offset, sizeof(byte), data.Length, sourcePath, label);
        ReadOnlySpan<byte> bytes = data.AsSpan(checked((int)offset), checked((int)count));
        int terminator = bytes.IndexOf((byte)0);
        int length = terminator >= 0 ? terminator : bytes.Length;
        return length == 0 ? null : Encoding.UTF8.GetString(bytes[..length]);
    }

    private static void ValidateSpan(uint count, uint offset, int stride, long length, string sourcePath, string label)
    {
        if (count == 0)
            return;

        if (offset == 0)
            throw new InvalidDataException($"1.12.1 M2 file '{sourcePath}' has a zero offset for non-empty span '{label}'.");

        ulong total = (ulong)count * (uint)stride;
        ulong end = (ulong)offset + total;
        if ((ulong)offset >= (ulong)length || end > (ulong)length || end < offset)
        {
            throw new InvalidDataException(
                $"1.12.1 M2 file '{sourcePath}' has an out-of-range span for '{label}': count={count}, offset=0x{offset:X}, stride=0x{stride:X}, length=0x{length:X}.");
        }
    }

    private static void EnsureReadable(byte[] data, int offset, int size, string sourcePath, string label)
    {
        if (offset < 0 || size < 0 || offset > data.Length - size)
            throw new InvalidDataException($"1.12.1 M2 file '{sourcePath}' does not contain a readable span for '{label}' at offset 0x{offset:X} with size 0x{size:X}.");
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
}
