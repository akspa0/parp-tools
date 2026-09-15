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
        M2Era1121Layout layout = new(version);

        uint flags = ReadUInt32At(data, M2Era1121Constants.FlagsOffset);
        uint viewCount = ReadUInt32At(data, layout.CameraCountOffset >= 0 ? layout.CameraCountOffset - 0x78 : M2Era1121Constants.ViewCountOffset); // viewCount is always 0x3C in both V100/V101
        string? modelName = TryReadName(data, sourcePath);
        List<uint> globalLoops = ReadUInt32Table(data, sourcePath, "globalLoops",
            M2Era1121Constants.GlobalLoopCountOffset, M2Era1121Constants.GlobalLoopOffsetOffset);

        M2ModelIdentity identity = M2ModelIdentity.FromPath(sourcePath);
        IReadOnlyList<M2SequenceDefinition> sequences = ReadSequences(data, sourcePath, layout);
        IReadOnlyList<short> sequenceLookup = ReadInt16Table(data, sourcePath, "sequenceLookup",
            M2Era1121Constants.SequenceLookupCountOffset, M2Era1121Constants.SequenceLookupOffsetOffset);
        IReadOnlyList<M2ColorDefinition> colors = ReadColors(data, globalLoops.Count, sourcePath);
        IReadOnlyList<M2TextureWeightDefinition> textureWeights = ReadTextureWeights(data, globalLoops.Count, sourcePath);
        IReadOnlyList<M2TextureTransformDefinition> textureTransforms = ReadTextureTransforms(data, globalLoops.Count, sourcePath);
        IReadOnlyList<M2LightDefinition> lights = ReadLights(data, globalLoops.Count, sourcePath);
        IReadOnlyList<M2CameraDefinition>? cameras = ReadCameras(data, globalLoops.Count, sourcePath, layout);
        IReadOnlyList<M2RibbonDefinition> ribbons = ReadRibbons(data, globalLoops.Count, sourcePath, layout);
        IReadOnlyList<M2ParticleDefinition> particles = ReadParticles(data, globalLoops.Count, sourcePath, layout);

        Vector3 boundsMin = ReadFiniteVector3At(data, layout.BoundsOffset, sourcePath, "boundsMin");
        Vector3 boundsMax = ReadFiniteVector3At(data, layout.BoundsOffset + 0x0C, sourcePath, "boundsMax");
        float boundsRadius = ReadFiniteSingleAt(data, layout.BoundsRadiusOffset, sourcePath, "boundsRadius");

        M2ModelDocument document = new(
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

        try
        {
            IReadOnlyList<M2Era1121VertexIndex> vertexIndices = ReadVertexIndices(data, sourcePath, layout);
            IReadOnlyList<Vector3> positions = ReadPositions(data, sourcePath, layout);
            IReadOnlyList<Vector3> normals = ReadNormals(data, sourcePath, layout);
            IReadOnlyList<Vector2> uvs = ReadUvs(data, sourcePath, layout);
            IReadOnlyList<ushort> triangles = ReadTriangles(data, sourcePath, layout);
            IReadOnlyList<M2Era1121Batch> batches = ReadBatches(data, sourcePath, layout);
            IReadOnlyList<M2Era1121Texture> textures = ReadTextures(data, sourcePath);
            IReadOnlyList<M2Era1121RenderFlag> renderFlags = ReadRenderFlags(data, sourcePath);
            IReadOnlyList<ushort> textureLookup = ReadTextureLookup(data, sourcePath);

            if (vertexIndices.Count > 0 && positions.Count > 0 && triangles.Count > 0)
            {
                document.InlineEra1121Geometry = new M2Era1121Geometry(
                    vertexIndices,
                    positions,
                    normals,
                    uvs,
                    triangles,
                    batches,
                    textures,
                    renderFlags,
                    textureLookup);
            }
        }
        catch (InvalidDataException)
        {
            // Geometry table offsets are speculative for 1.12.1 format.
            // If extraction fails, return document with null geometry so
            // the caller can fall through to other parsing strategies.
        }

        return document;
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

    private static IReadOnlyList<M2SequenceDefinition> ReadSequences(byte[] data, string sourcePath, M2Era1121Layout layout)
    {
        uint count = ReadUInt32At(data, M2Era1121Constants.SequenceCountOffset);
        uint offset = ReadUInt32At(data, M2Era1121Constants.SequenceOffsetOffset);
        ValidateSpan(count, offset, layout.SequenceStride, data.Length, sourcePath, "sequences");

        List<M2SequenceDefinition> values = new(checked((int)count));
        for (int index = 0; index < count; index++)
        {
            int entryOffset = checked((int)offset + (index * layout.SequenceStride));
            values.Add(new M2SequenceDefinition(
                index,
                ReadUInt16At(data, entryOffset + 0x00),
                ReadUInt16At(data, entryOffset + 0x02),
                ReadUInt32At(data, entryOffset + 0x04),
                ReadLenientSingleAt(data, entryOffset + 0x08, sourcePath, $"sequence[{index}].moveSpeed"),
                ReadUInt32At(data, entryOffset + 0x0C),
                ReadInt16At(data, entryOffset + 0x10),
                ReadUInt32At(data, entryOffset + 0x14),
                ReadUInt32At(data, entryOffset + 0x18),
                ReadUInt16At(data, entryOffset + 0x1C),
                ReadUInt16At(data, entryOffset + 0x1E),
                ReadLenientVector3At(data, entryOffset + 0x20, sourcePath, $"sequence[{index}].boundsMin"),
                ReadLenientVector3At(data, entryOffset + 0x2C, sourcePath, $"sequence[{index}].boundsMax"),
                ReadLenientSingleAt(data, entryOffset + 0x38, sourcePath, $"sequence[{index}].boundsRadius"),
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
            values.Add(new M2LightDefinition(
                index,
                0,
                0,
                position,
                CreateDummyTrack<Vector3>(),
                CreateDummyTrack<float>(),
                CreateDummyTrack<Vector3>(),
                CreateDummyTrack<float>(),
                CreateDummyTrack<float>(),
                CreateDummyTrack<float>(),
                CreateDummyTrack<byte>()));
        }

        return values;
    }

    private static M2TrackDefinition<T> CreateDummyTrack<T>() where T : struct
    {
        return new M2TrackDefinition<T>(
            M2TrackInterpolation.Linear,
            -1,
            new M2TrackArrayReference(0u, 0u),
            new M2TrackArrayReference(0u, 0u));
    }

    private static IReadOnlyList<M2CameraDefinition>? ReadCameras(byte[] data, int globalLoopCount, string sourcePath, M2Era1121Layout layout)
    {
        if (layout.CameraCountOffset < 0 || layout.CameraOffsetOffset < 0)
            return [];

        if (!TryReadUInt32At(data, layout.CameraCountOffset, out uint count)
            || !TryReadUInt32At(data, layout.CameraOffsetOffset, out uint offset)
            || count == 0
            || count > 16
            || offset == 0
            || offset >= (uint)data.Length)
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

    private static IReadOnlyList<M2RibbonDefinition> ReadRibbons(byte[] data, int globalLoopCount, string sourcePath, M2Era1121Layout layout)
    {
        if (layout.RibbonCountOffset < 0 || layout.RibbonOffsetOffset < 0)
            return [];

        if (!TryReadUInt32At(data, layout.RibbonCountOffset, out uint count)
            || !TryReadUInt32At(data, layout.RibbonOffsetOffset, out uint offset)
            || count == 0
            || count > 16
            || offset == 0
            || offset >= (uint)data.Length)
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

    private static IReadOnlyList<M2ParticleDefinition> ReadParticles(byte[] data, int globalLoopCount, string sourcePath, M2Era1121Layout layout)
    {
        if (layout.ParticleCountOffset < 0 || layout.ParticleOffsetOffset < 0)
            return [];

        if (!TryReadUInt32At(data, layout.ParticleCountOffset, out uint count)
            || !TryReadUInt32At(data, layout.ParticleOffsetOffset, out uint offset)
            || count == 0
            || count > 256
            || offset == 0
            || offset >= (uint)data.Length)
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

    private static IReadOnlyList<M2Era1121VertexIndex> ReadVertexIndices(byte[] data, string sourcePath, M2Era1121Layout layout)
    {
        if (layout.VertexCountOffset < 0 || layout.VertexOffsetOffset < 0)
            return [];

        uint count = ReadUInt32At(data, layout.VertexCountOffset);
        uint offset = ReadUInt32At(data, layout.VertexOffsetOffset);
        if (count == 0 || count > 100000 || offset == 0 || offset >= (uint)data.Length)
            return [];

        ValidateSpan(count, offset, M2Era1121Constants.VertexIndexStride, data.Length, sourcePath, "vertexIndices");

        List<M2Era1121VertexIndex> values = new(checked((int)count));
        for (int index = 0; index < count; index++)
        {
            int entryOffset = checked((int)offset + (index * M2Era1121Constants.VertexIndexStride));
            uint packed = ReadUInt32At(data, entryOffset);
            ushort positionIndex = (ushort)(packed & 0xFFFFu);
            ushort normalIndex = (ushort)((packed >> 16) & 0xFFFFu);
            values.Add(new M2Era1121VertexIndex(positionIndex, normalIndex));
        }

        return values;
    }

    private static IReadOnlyList<Vector3> ReadPositions(byte[] data, string sourcePath, M2Era1121Layout layout)
    {
        if (layout.PositionCountOffset < 0 || layout.PositionOffsetOffset < 0)
            return [];

        uint count = ReadUInt32At(data, layout.PositionCountOffset);
        uint offset = ReadUInt32At(data, layout.PositionOffsetOffset);
        if (count == 0 || count > 100000 || offset == 0 || offset >= (uint)data.Length)
            return [];

        ValidateSpan(count, offset, M2Era1121Constants.PositionStride, data.Length, sourcePath, "positions");

        List<Vector3> values = new(checked((int)count));
        for (int index = 0; index < count; index++)
        {
            int entryOffset = checked((int)offset + (index * M2Era1121Constants.PositionStride));
            values.Add(ReadLenientVector3At(data, entryOffset, sourcePath, $"positions[{index}]"));
        }

        return values;
    }

    private static IReadOnlyList<Vector3> ReadNormals(byte[] data, string sourcePath, M2Era1121Layout layout)
    {
        if (layout.NormalCountOffset < 0 || layout.NormalOffsetOffset < 0)
            return [];

        uint count = ReadUInt32At(data, layout.NormalCountOffset);
        uint offset = ReadUInt32At(data, layout.NormalOffsetOffset);
        if (count == 0 || count > 100000 || offset == 0 || offset >= (uint)data.Length)
            return [];

        ValidateSpan(count, offset, M2Era1121Constants.NormalStride, data.Length, sourcePath, "normals");

        List<Vector3> values = new(checked((int)count));
        for (int index = 0; index < count; index++)
        {
            int entryOffset = checked((int)offset + (index * M2Era1121Constants.NormalStride));
            values.Add(ReadLenientVector3At(data, entryOffset, sourcePath, $"normals[{index}]"));
        }

        return values;
    }

    private static IReadOnlyList<Vector2> ReadUvs(byte[] data, string sourcePath, M2Era1121Layout layout)
    {
        if (layout.UvCountOffset < 0 || layout.UvOffsetOffset < 0)
            return [];

        uint count = ReadUInt32At(data, layout.UvCountOffset);
        uint offset = ReadUInt32At(data, layout.UvOffsetOffset);
        if (count == 0 || count > 100000 || offset == 0 || offset >= (uint)data.Length)
            return [];

        ValidateSpan(count, offset, M2Era1121Constants.UvStride, data.Length, sourcePath, "uvs");

        List<Vector2> values = new(checked((int)count));
        for (int index = 0; index < count; index++)
        {
            int entryOffset = checked((int)offset + (index * M2Era1121Constants.UvStride));
            values.Add(new Vector2(
                ReadLenientSingleAt(data, entryOffset + 0x00, sourcePath, $"uvs[{index}].u"),
                ReadLenientSingleAt(data, entryOffset + 0x04, sourcePath, $"uvs[{index}].v")));
        }

        return values;
    }

    private static IReadOnlyList<ushort> ReadTriangles(byte[] data, string sourcePath, M2Era1121Layout layout)
    {
        if (layout.TriangleCountOffset < 0 || layout.TriangleOffsetOffset < 0)
            return [];

        uint count = ReadUInt32At(data, layout.TriangleCountOffset);
        uint offset = ReadUInt32At(data, layout.TriangleOffsetOffset);
        if (count == 0 || count > 100000 || offset == 0 || offset >= (uint)data.Length)
            return [];

        ValidateSpan(count, offset, M2Era1121Constants.TriangleStride, data.Length, sourcePath, "triangles");

        List<ushort> values = new(checked((int)count));
        for (int index = 0; index < count; index++)
        {
            int entryOffset = checked((int)offset + (index * M2Era1121Constants.TriangleStride));
            values.Add(ReadUInt16At(data, entryOffset));
        }

        return values;
    }

    private static IReadOnlyList<M2Era1121Batch> ReadBatches(byte[] data, string sourcePath, M2Era1121Layout layout)
    {
        if (layout.BatchCountOffset < 0 || layout.BatchOffsetOffset < 0)
            return [];

        uint count = ReadUInt32At(data, layout.BatchCountOffset);
        uint offset = ReadUInt32At(data, layout.BatchOffsetOffset);
        if (count == 0 || count > 10000 || offset == 0 || offset >= (uint)data.Length)
            return [];

        ValidateSpan(count, offset, M2Era1121Constants.BatchStride, data.Length, sourcePath, "batches");

        List<M2Era1121Batch> values = new(checked((int)count));
        for (int index = 0; index < count; index++)
        {
            int entryOffset = checked((int)offset + (index * M2Era1121Constants.BatchStride));
            values.Add(new M2Era1121Batch(
                ReadUInt16At(data, entryOffset + 0x00),
                ReadUInt16At(data, entryOffset + 0x02),
                ReadUInt16At(data, entryOffset + 0x04),
                ReadUInt16At(data, entryOffset + 0x06),
                ReadUInt16At(data, entryOffset + 0x08),
                ReadUInt16At(data, entryOffset + 0x0A),
                ReadUInt16At(data, entryOffset + 0x0C),
                ReadUInt16At(data, entryOffset + 0x0E),
                ReadUInt16At(data, entryOffset + 0x10),
                ReadUInt16At(data, entryOffset + 0x12),
                ReadUInt16At(data, entryOffset + 0x14),
                ReadUInt16At(data, entryOffset + 0x16),
                ReadUInt16At(data, entryOffset + 0x18),
                ReadUInt16At(data, entryOffset + 0x1A),
                ReadUInt16At(data, entryOffset + 0x1C)));
        }

        return values;
    }

    private static IReadOnlyList<M2Era1121Texture> ReadTextures(byte[] data, string sourcePath)
    {
        uint count = ReadUInt32At(data, M2Era1121Constants.TextureCountOffset);
        uint offset = ReadUInt32At(data, M2Era1121Constants.TextureOffsetOffset);
        const int stride = 16;
        ValidateSpan(count, offset, stride, data.Length, sourcePath, "textures");

        List<M2Era1121Texture> values = new(checked((int)count));
        for (int index = 0; index < count; index++)
        {
            int entryOffset = checked((int)offset + (index * stride));
            uint type = ReadUInt32At(data, entryOffset + 0x00);
            uint flags = ReadUInt32At(data, entryOffset + 0x04);
            uint nameLen = ReadUInt32At(data, entryOffset + 0x08);
            uint nameOff = ReadUInt32At(data, entryOffset + 0x0C);

            string? filename = TryReadStringAt(data, sourcePath, $"textures[{index}].filename", nameLen, nameOff);
            values.Add(new M2Era1121Texture(type, flags, filename ?? string.Empty));
        }

        return values;
    }

    private static IReadOnlyList<M2Era1121RenderFlag> ReadRenderFlags(byte[] data, string sourcePath)
    {
        uint count = ReadUInt32At(data, M2Era1121Constants.RenderFlagCountOffset);
        uint offset = ReadUInt32At(data, M2Era1121Constants.RenderFlagOffsetOffset);
        const int stride = 16;
        ValidateSpan(count, offset, stride, data.Length, sourcePath, "renderFlags");

        List<M2Era1121RenderFlag> values = new(checked((int)count));
        for (int index = 0; index < count; index++)
        {
            int entryOffset = checked((int)offset + (index * stride));
            ushort flags = ReadUInt16At(data, entryOffset + 0x00);
            ushort blendMode = ReadUInt16At(data, entryOffset + 0x02);
            values.Add(new M2Era1121RenderFlag(flags, blendMode));
        }

        return values;
    }

    private static IReadOnlyList<ushort> ReadTextureLookup(byte[] data, string sourcePath)
    {
        uint count = ReadUInt32At(data, M2Era1121Constants.TexLookupCountOffset);
        uint offset = ReadUInt32At(data, M2Era1121Constants.TexLookupOffsetOffset);
        const int stride = 4;
        ValidateSpan(count, offset, stride, data.Length, sourcePath, "textureLookup");

        List<ushort> values = new(checked((int)count));
        for (int index = 0; index < count; index++)
        {
            int entryOffset = checked((int)offset + (index * stride));
            uint val = ReadUInt32At(data, entryOffset);
            values.Add((ushort)val);
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

    private static float ReadLenientSingleAt(byte[] data, int offset, string sourcePath, string label)
    {
        EnsureReadable(data, offset, sizeof(float), sourcePath, label);
        return BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(data.AsSpan(offset, sizeof(float))));
    }

    private static Vector3 ReadFiniteVector3At(byte[] data, int offset, string sourcePath, string label)
    {
        return new Vector3(
            ReadFiniteSingleAt(data, offset + 0x00, sourcePath, $"{label}.x"),
            ReadFiniteSingleAt(data, offset + 0x04, sourcePath, $"{label}.y"),
            ReadFiniteSingleAt(data, offset + 0x08, sourcePath, $"{label}.z"));
    }

    private static Vector3 ReadLenientVector3At(byte[] data, int offset, string sourcePath, string label)
    {
        return new Vector3(
            ReadLenientSingleAt(data, offset + 0x00, sourcePath, $"{label}.x"),
            ReadLenientSingleAt(data, offset + 0x04, sourcePath, $"{label}.y"),
            ReadLenientSingleAt(data, offset + 0x08, sourcePath, $"{label}.z"));
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
