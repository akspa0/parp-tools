using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.Chunks;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.IO.Mdx;

public static class MdxParticleEmitter2Reader
{
    private const int SignatureSizeBytes = 4;
    private const int ModlNameSizeBytes = 0x50;
    private const int PivtEntrySizeBytes = 12;
    private const int Pre2ModelPathSizeBytes = 0x104;
    private const int Pre2ClassicEmitterPayloadSizeMinBytes = 791;

    public static MdxParticleEmitter2File Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static MdxParticleEmitter2File Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        if (!stream.CanSeek)
            throw new ArgumentException("MDX particle emitter reading requires a seekable stream.", nameof(stream));

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
            List<MdxPivotPointSummary> pivotPoints = [];
            List<MdxParticleEmitter2> particleEmitters = [];
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
                else if (header.Id == MdxChunkIds.Pivt)
                {
                    pivotPoints = ReadPivtSummary(stream, dataOffset, header.Size);
                }
                else if (header.Id == MdxChunkIds.Pre2)
                {
                    particleEmitters = ReadClassicParticleEmitters(stream, dataOffset, header.Size, version, pivotPoints);
                }

                stream.Position = endOffset;
            }

            if (particleEmitters.Count > 0 && pivotPoints.Count > 0)
                particleEmitters = AssignDeferredPivots(particleEmitters, pivotPoints);

            return new MdxParticleEmitter2File(sourcePath, signature, version, modelName, particleEmitters);
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static List<MdxParticleEmitter2> ReadClassicParticleEmitters(Stream stream, long dataOffset, uint size, uint? version, IReadOnlyList<MdxPivotPointSummary> pivotPoints)
    {
        if (version is not null and not 1300u and not 1400u)
            return [];

        long previousPosition = stream.Position;
        try
        {
            long chunkEnd = checked(dataOffset + size);
            stream.Position = dataOffset;
            if (chunkEnd - stream.Position < sizeof(uint))
                throw new InvalidDataException("PRE2(v1300): missing particle emitter count.");

            uint emitterCount = ReadUInt32(stream);
            if (emitterCount > 100000)
                throw new InvalidDataException($"PRE2(v1300): invalid particle emitter count {emitterCount}.");

            List<MdxParticleEmitter2> particleEmitters = new(checked((int)emitterCount));
            for (int index = 0; index < emitterCount; index++)
            {
                if (chunkEnd - stream.Position < sizeof(uint) * 2)
                    throw new InvalidDataException($"PRE2(v1300): truncated emitter header at index {index}.");

                long entryStart = stream.Position;
                uint entrySize = ReadUInt32(stream);
                long entryEnd = checked(entryStart + entrySize);
                if (entryEnd > chunkEnd || entryEnd <= entryStart)
                    throw new InvalidDataException($"PRE2(v1300): invalid emitter size 0x{entrySize:X} at index {index}.");

                long nodeStart = stream.Position;
                uint nodeSize = ReadUInt32(stream);
                long nodeEnd = checked(nodeStart + nodeSize);
                if (nodeEnd > entryEnd || nodeEnd <= nodeStart)
                    throw new InvalidDataException($"PRE2(v1300): invalid node size 0x{nodeSize:X} at index {index}.");

                (string name, int objectId, int parentId, uint flags, MdxVector3NodeTrack? translationTrack, MdxQuaternionNodeTrack? rotationTrack, MdxVector3NodeTrack? scalingTrack) =
                    ReadNodeTracks(stream, nodeEnd, index, "PRE2(v1300)");

                if (entryEnd - stream.Position < sizeof(uint))
                    throw new InvalidDataException($"PRE2(v1300): missing emitter payload size at index {index}.");

                uint emitterPayloadSize = ReadUInt32(stream);
                if (emitterPayloadSize < Pre2ClassicEmitterPayloadSizeMinBytes)
                    throw new InvalidDataException($"PRE2(v1300): invalid emitter payload size 0x{emitterPayloadSize:X} at index {index}.");

                long emitterPayloadEnd = checked(stream.Position + emitterPayloadSize);
                if (emitterPayloadEnd > entryEnd)
                    throw new InvalidDataException($"PRE2(v1300): emitter payload size 0x{emitterPayloadSize:X} overran entry {index}.");

                int emitterType = ReadInt32(stream);
                float staticSpeed = ReadSingle(stream);
                float staticVariation = ReadSingle(stream);
                float staticLatitude = ReadSingle(stream);
                float staticLongitude = ReadSingle(stream);
                float staticGravity = ReadSingle(stream);
                float staticZSource = ReadSingle(stream);
                float staticLife = ReadSingle(stream);
                float staticEmissionRate = ReadSingle(stream);
                float staticLength = ReadSingle(stream);
                float staticWidth = ReadSingle(stream);
                uint rows = ReadUInt32(stream);
                uint columns = ReadUInt32(stream);
                uint particleType = ReadUInt32(stream);
                float tailLength = ReadSingle(stream);
                float middleTime = ReadSingle(stream);
                Vector3 startColor = ReadVector3(stream);
                Vector3 middleColor = ReadVector3(stream);
                Vector3 endColor = ReadVector3(stream);
                byte startAlpha = ReadByte(stream);
                byte middleAlpha = ReadByte(stream);
                byte endAlpha = ReadByte(stream);
                float startScale = ReadSingle(stream);
                float middleScale = ReadSingle(stream);
                float endScale = ReadSingle(stream);

                List<uint> unknownIntervals = new(12);
                for (int intervalIndex = 0; intervalIndex < 12; intervalIndex++)
                    unknownIntervals.Add(ReadUInt32(stream));

                uint blendMode = ReadUInt32(stream);
                int textureId = ReadInt32(stream);
                int priorityPlane = ReadInt32(stream);
                uint replaceableId = ReadUInt32(stream);
                string? geometryModel = ReadNullTerminatedAscii(ReadBytes(stream, Pre2ModelPathSizeBytes));
                string? recursionModel = ReadNullTerminatedAscii(ReadBytes(stream, Pre2ModelPathSizeBytes));

                List<float> unknownFloatBlockA = ReadSingleBlock(stream, 5);
                List<float> unknownTumbleValues = ReadSingleBlock(stream, 6);
                List<float> unknownFloatBlockB = ReadSingleBlock(stream, 2);
                Vector3 unknownVector = ReadVector3(stream);
                List<float> unknownFloatBlockC = ReadSingleBlock(stream, 5);

                uint splineCount = ReadUInt32(stream);
                if (splineCount > 100000)
                    throw new InvalidDataException($"PRE2(v1300): invalid spline count {splineCount} at index {index}.");

                List<Vector3> splinePoints = new(checked((int)splineCount));
                for (uint splineIndex = 0; splineIndex < splineCount; splineIndex++)
                    splinePoints.Add(ReadVector3(stream));

                int squirts = ReadInt32(stream);

                if (stream.Position > emitterPayloadEnd)
                    throw new InvalidDataException($"PRE2(v1300): static emitter payload overran entry {index}.");

                stream.Position = emitterPayloadEnd;

                MdxScalarTrack? visibilityTrack = null;
                MdxScalarTrack? speedTrack = null;
                MdxScalarTrack? variationTrack = null;
                MdxScalarTrack? latitudeTrack = null;
                MdxScalarTrack? longitudeTrack = null;
                MdxScalarTrack? gravityTrack = null;
                MdxScalarTrack? lifeTrack = null;
                MdxScalarTrack? emissionRateTrack = null;
                MdxScalarTrack? widthTrack = null;
                MdxScalarTrack? lengthTrack = null;
                MdxScalarTrack? zSourceTrack = null;

                while (stream.Position <= entryEnd - 4)
                {
                    string tag = ReadTag(stream);
                    switch (tag)
                    {
                        case "KVIS":
                        case "KP2V":
                            visibilityTrack = MdxTrackReader.ReadScalarTrack(stream, entryEnd, tag, "PRE2(v1300)", $"PRE2(v1300): {tag} payload overran the emitter.");
                            break;
                        case "KP2S":
                            speedTrack = MdxTrackReader.ReadScalarTrack(stream, entryEnd, tag, "PRE2(v1300)", $"PRE2(v1300): {tag} payload overran the emitter.");
                            break;
                        case "KP2R":
                            variationTrack = MdxTrackReader.ReadScalarTrack(stream, entryEnd, tag, "PRE2(v1300)", $"PRE2(v1300): {tag} payload overran the emitter.");
                            break;
                        case "KP2L":
                            latitudeTrack = MdxTrackReader.ReadScalarTrack(stream, entryEnd, tag, "PRE2(v1300)", $"PRE2(v1300): {tag} payload overran the emitter.");
                            break;
                        case "KPLN":
                            longitudeTrack = MdxTrackReader.ReadScalarTrack(stream, entryEnd, tag, "PRE2(v1300)", $"PRE2(v1300): {tag} payload overran the emitter.");
                            break;
                        case "KP2G":
                            gravityTrack = MdxTrackReader.ReadScalarTrack(stream, entryEnd, tag, "PRE2(v1300)", $"PRE2(v1300): {tag} payload overran the emitter.");
                            break;
                        case "KLIF":
                            lifeTrack = MdxTrackReader.ReadScalarTrack(stream, entryEnd, tag, "PRE2(v1300)", $"PRE2(v1300): {tag} payload overran the emitter.");
                            break;
                        case "KP2E":
                            emissionRateTrack = MdxTrackReader.ReadScalarTrack(stream, entryEnd, tag, "PRE2(v1300)", $"PRE2(v1300): {tag} payload overran the emitter.");
                            break;
                        case "KP2W":
                            widthTrack = MdxTrackReader.ReadScalarTrack(stream, entryEnd, tag, "PRE2(v1300)", $"PRE2(v1300): {tag} payload overran the emitter.");
                            break;
                        case "KP2N":
                            lengthTrack = MdxTrackReader.ReadScalarTrack(stream, entryEnd, tag, "PRE2(v1300)", $"PRE2(v1300): {tag} payload overran the emitter.");
                            break;
                        case "KP2Z":
                            zSourceTrack = MdxTrackReader.ReadScalarTrack(stream, entryEnd, tag, "PRE2(v1300)", $"PRE2(v1300): {tag} payload overran the emitter.");
                            break;
                        default:
                            stream.Position = entryEnd;
                            break;
                    }
                }

                Vector3 pivotPoint = objectId >= 0 && objectId < pivotPoints.Count
                    ? pivotPoints[objectId].Position
                    : Vector3.Zero;

                particleEmitters.Add(new MdxParticleEmitter2(
                    index,
                    name,
                    objectId,
                    parentId,
                    flags,
                    pivotPoint,
                    emitterType,
                    staticSpeed,
                    staticVariation,
                    staticLatitude,
                    staticLongitude,
                    staticGravity,
                    staticZSource,
                    staticLife,
                    staticEmissionRate,
                    staticLength,
                    staticWidth,
                    rows,
                    columns,
                    particleType,
                    tailLength,
                    middleTime,
                    startColor,
                    middleColor,
                    endColor,
                    startAlpha,
                    middleAlpha,
                    endAlpha,
                    startScale,
                    middleScale,
                    endScale,
                    unknownIntervals,
                    blendMode,
                    textureId,
                    priorityPlane,
                    replaceableId,
                    geometryModel,
                    recursionModel,
                    unknownFloatBlockA,
                    unknownTumbleValues,
                    unknownFloatBlockB,
                    unknownVector,
                    unknownFloatBlockC,
                    splineCount,
                    splinePoints,
                    squirts,
                    translationTrack,
                    rotationTrack,
                    scalingTrack,
                    visibilityTrack,
                    speedTrack,
                    variationTrack,
                    latitudeTrack,
                    longitudeTrack,
                    gravityTrack,
                    lifeTrack,
                    emissionRateTrack,
                    widthTrack,
                    lengthTrack,
                    zSourceTrack));
                stream.Position = entryEnd;
            }

            stream.Position = chunkEnd;
            return particleEmitters;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static List<MdxParticleEmitter2> AssignDeferredPivots(IReadOnlyList<MdxParticleEmitter2> particleEmitters, IReadOnlyList<MdxPivotPointSummary> pivotPoints)
    {
        List<MdxParticleEmitter2> reassigned = new(particleEmitters.Count);
        foreach (MdxParticleEmitter2 particleEmitter in particleEmitters)
        {
            Vector3 pivotPoint = particleEmitter.ObjectId >= 0 && particleEmitter.ObjectId < pivotPoints.Count
                ? pivotPoints[particleEmitter.ObjectId].Position
                : particleEmitter.PivotPoint;

            reassigned.Add(new MdxParticleEmitter2(
                particleEmitter.Index,
                particleEmitter.Name,
                particleEmitter.ObjectId,
                particleEmitter.ParentId,
                particleEmitter.Flags,
                pivotPoint,
                particleEmitter.EmitterType,
                particleEmitter.StaticSpeed,
                particleEmitter.StaticVariation,
                particleEmitter.StaticLatitude,
                particleEmitter.StaticLongitude,
                particleEmitter.StaticGravity,
                particleEmitter.StaticZSource,
                particleEmitter.StaticLife,
                particleEmitter.StaticEmissionRate,
                particleEmitter.StaticLength,
                particleEmitter.StaticWidth,
                particleEmitter.Rows,
                particleEmitter.Columns,
                particleEmitter.ParticleType,
                particleEmitter.TailLength,
                particleEmitter.MiddleTime,
                particleEmitter.StartColor,
                particleEmitter.MiddleColor,
                particleEmitter.EndColor,
                particleEmitter.StartAlpha,
                particleEmitter.MiddleAlpha,
                particleEmitter.EndAlpha,
                particleEmitter.StartScale,
                particleEmitter.MiddleScale,
                particleEmitter.EndScale,
                particleEmitter.UnknownIntervals,
                particleEmitter.BlendMode,
                particleEmitter.TextureId,
                particleEmitter.PriorityPlane,
                particleEmitter.ReplaceableId,
                particleEmitter.GeometryModel,
                particleEmitter.RecursionModel,
                particleEmitter.UnknownFloatBlockA,
                particleEmitter.UnknownTumbleValues,
                particleEmitter.UnknownFloatBlockB,
                particleEmitter.UnknownVector,
                particleEmitter.UnknownFloatBlockC,
                particleEmitter.SplineCount,
                particleEmitter.SplinePoints,
                particleEmitter.Squirts,
                particleEmitter.TranslationTrack,
                particleEmitter.RotationTrack,
                particleEmitter.ScalingTrack,
                particleEmitter.VisibilityTrack,
                particleEmitter.SpeedTrack,
                particleEmitter.VariationTrack,
                particleEmitter.LatitudeTrack,
                particleEmitter.LongitudeTrack,
                particleEmitter.GravityTrack,
                particleEmitter.LifeTrack,
                particleEmitter.EmissionRateTrack,
                particleEmitter.WidthTrack,
                particleEmitter.LengthTrack,
                particleEmitter.ZSourceTrack));
        }

        return reassigned;
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

    private static List<MdxPivotPointSummary> ReadPivtSummary(Stream stream, long dataOffset, uint size)
    {
        if (size % PivtEntrySizeBytes != 0)
            throw new InvalidDataException($"Invalid PIVT size 0x{size:X}: expected multiple of {PivtEntrySizeBytes}.");

        long previousPosition = stream.Position;
        try
        {
            stream.Position = dataOffset;
            int count = checked((int)(size / PivtEntrySizeBytes));
            List<MdxPivotPointSummary> pivotPoints = new(count);
            for (int index = 0; index < count; index++)
                pivotPoints.Add(new MdxPivotPointSummary(index, ReadVector3(stream)));

            return pivotPoints;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static List<float> ReadSingleBlock(Stream stream, int count)
    {
        List<float> values = new(count);
        for (int index = 0; index < count; index++)
            values.Add(ReadSingle(stream));

        return values;
    }

    private static byte[] ReadBytes(Stream stream, int count)
    {
        byte[] bytes = new byte[count];
        stream.ReadExactly(bytes);
        return bytes;
    }

    private static string? ReadNullTerminatedAscii(byte[] bytes)
    {
        int terminatorIndex = Array.IndexOf(bytes, (byte)0);
        int count = terminatorIndex >= 0 ? terminatorIndex : bytes.Length;
        if (count == 0)
            return null;

        return Encoding.ASCII.GetString(bytes, 0, count);
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

    private static byte ReadByte(Stream stream)
    {
        int value = stream.ReadByte();
        if (value < 0)
            throw new EndOfStreamException();

        return (byte)value;
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