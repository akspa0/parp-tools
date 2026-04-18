using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.Chunks;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.IO.Mdx;

public static class MdxRibbonEmitterReader
{
    private const int SignatureSizeBytes = 4;
    private const int ModlNameSizeBytes = 0x50;
    private const int PivtEntrySizeBytes = 12;

    public static MdxRibbonEmitterFile Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static MdxRibbonEmitterFile Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        if (!stream.CanSeek)
            throw new ArgumentException("MDX ribbon reading requires a seekable stream.", nameof(stream));

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
            List<MdxRibbonEmitter> ribbons = [];
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
                else if (header.Id == MdxChunkIds.Ribb)
                {
                    ribbons = ReadClassicRibbons(stream, dataOffset, header.Size, version, pivotPoints);
                }

                stream.Position = endOffset;
            }

            if (ribbons.Count > 0 && pivotPoints.Count > 0)
                ribbons = AssignDeferredPivots(ribbons, pivotPoints);

            return new MdxRibbonEmitterFile(sourcePath, signature, version, modelName, ribbons);
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static List<MdxRibbonEmitter> ReadClassicRibbons(Stream stream, long dataOffset, uint size, uint? version, IReadOnlyList<MdxPivotPointSummary> pivotPoints)
    {
        if (version is not null and not 1300u and not 1400u)
            return [];

        long previousPosition = stream.Position;
        try
        {
            long chunkEnd = checked(dataOffset + size);
            stream.Position = dataOffset;
            if (chunkEnd - stream.Position < sizeof(uint))
                throw new InvalidDataException("RIBB(v1300): missing ribbon emitter count.");

            uint ribbonCount = ReadUInt32(stream);
            if (ribbonCount > 100000)
                throw new InvalidDataException($"RIBB(v1300): invalid ribbon emitter count {ribbonCount}.");

            List<MdxRibbonEmitter> ribbons = new(checked((int)ribbonCount));
            for (int index = 0; index < ribbonCount; index++)
            {
                if (chunkEnd - stream.Position < sizeof(uint) * 2)
                    throw new InvalidDataException($"RIBB(v1300): truncated emitter header at index {index}.");

                long entryStart = stream.Position;
                uint entrySize = ReadUInt32(stream);
                long entryEnd = checked(entryStart + entrySize);
                if (entryEnd > chunkEnd || entryEnd <= entryStart)
                    throw new InvalidDataException($"RIBB(v1300): invalid emitter size 0x{entrySize:X} at index {index}.");

                long nodeStart = stream.Position;
                uint nodeSize = ReadUInt32(stream);
                long nodeEnd = checked(nodeStart + nodeSize);
                if (nodeEnd > entryEnd || nodeEnd <= nodeStart)
                    throw new InvalidDataException($"RIBB(v1300): invalid node size 0x{nodeSize:X} at index {index}.");

                (string name, int objectId, int parentId, uint flags, MdxVector3NodeTrack? translationTrack, MdxQuaternionNodeTrack? rotationTrack, MdxVector3NodeTrack? scalingTrack) =
                    ReadNodeTracks(stream, nodeEnd, index, "RIBB(v1300)");

                if (entryEnd - stream.Position < 56)
                    throw new InvalidDataException($"RIBB(v1300): missing emitter fields at index {index}.");

                uint emitterPayloadSize = ReadUInt32(stream);
                if (emitterPayloadSize < 56)
                    throw new InvalidDataException($"RIBB(v1300): invalid emitter payload size 0x{emitterPayloadSize:X} at index {index}.");

                float staticHeightAbove = ReadSingle(stream);
                float staticHeightBelow = ReadSingle(stream);
                float staticAlpha = ReadSingle(stream);
                Vector3 staticColor = ReadVector3(stream);
                float edgeLifetime = ReadSingle(stream);
                uint staticTextureSlot = ReadUInt32(stream);
                uint edgesPerSecond = ReadUInt32(stream);
                uint textureRows = ReadUInt32(stream);
                uint textureColumns = ReadUInt32(stream);
                uint materialId = ReadUInt32(stream);
                float gravity = ReadSingle(stream);

                MdxScalarTrack? heightAboveTrack = null;
                MdxScalarTrack? heightBelowTrack = null;
                MdxScalarTrack? alphaTrack = null;
                MdxColorTrack? colorTrack = null;
                MdxIntTrack? textureSlotTrack = null;
                MdxScalarTrack? visibilityTrack = null;

                while (stream.Position <= entryEnd - 4)
                {
                    string tag = ReadTag(stream);
                    switch (tag)
                    {
                        case "KRHA":
                        case "KRHB":
                        case "KRAL":
                            MdxScalarTrack scalarTrack = MdxTrackReader.ReadScalarTrack(stream, entryEnd, tag, "RIBB(v1300)", $"RIBB(v1300): {tag} payload overran the emitter.");
                            if (tag == "KRHA")
                                heightAboveTrack = scalarTrack;
                            else if (tag == "KRHB")
                                heightBelowTrack = scalarTrack;
                            else
                                alphaTrack = scalarTrack;
                            break;
                        case "KRCO":
                            colorTrack = MdxTrackReader.ReadColorTrack(stream, entryEnd, tag, "RIBB(v1300)", $"RIBB(v1300): {tag} payload overran the emitter.");
                            break;
                        case "KRTX":
                            textureSlotTrack = MdxTrackReader.ReadIntTrack(stream, entryEnd, tag, "RIBB(v1300)", $"RIBB(v1300): {tag} payload overran the emitter.");
                            break;
                        case "KVIS":
                        case "KATV":
                            visibilityTrack = MdxTrackReader.ReadScalarTrack(stream, entryEnd, tag, "RIBB(v1300)", $"RIBB(v1300): {tag} payload overran the emitter.");
                            break;
                        default:
                            stream.Position = entryEnd;
                            break;
                    }
                }

                Vector3 pivotPoint = objectId >= 0 && objectId < pivotPoints.Count
                    ? pivotPoints[objectId].Position
                    : Vector3.Zero;

                ribbons.Add(new MdxRibbonEmitter(
                    index,
                    name,
                    objectId,
                    parentId,
                    flags,
                    pivotPoint,
                    staticHeightAbove,
                    staticHeightBelow,
                    staticAlpha,
                    staticColor,
                    edgeLifetime,
                    staticTextureSlot,
                    edgesPerSecond,
                    textureRows,
                    textureColumns,
                    materialId,
                    gravity,
                    translationTrack,
                    rotationTrack,
                    scalingTrack,
                    heightAboveTrack,
                    heightBelowTrack,
                    alphaTrack,
                    colorTrack,
                    textureSlotTrack,
                    visibilityTrack));
                stream.Position = entryEnd;
            }

            stream.Position = chunkEnd;
            return ribbons;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static List<MdxRibbonEmitter> AssignDeferredPivots(IReadOnlyList<MdxRibbonEmitter> ribbons, IReadOnlyList<MdxPivotPointSummary> pivotPoints)
    {
        List<MdxRibbonEmitter> reassigned = new(ribbons.Count);
        foreach (MdxRibbonEmitter ribbon in ribbons)
        {
            Vector3 pivotPoint = ribbon.ObjectId >= 0 && ribbon.ObjectId < pivotPoints.Count
                ? pivotPoints[ribbon.ObjectId].Position
                : ribbon.PivotPoint;

            reassigned.Add(new MdxRibbonEmitter(
                ribbon.Index,
                ribbon.Name,
                ribbon.ObjectId,
                ribbon.ParentId,
                ribbon.Flags,
                pivotPoint,
                ribbon.StaticHeightAbove,
                ribbon.StaticHeightBelow,
                ribbon.StaticAlpha,
                ribbon.StaticColor,
                ribbon.EdgeLifetime,
                ribbon.StaticTextureSlot,
                ribbon.EdgesPerSecond,
                ribbon.TextureRows,
                ribbon.TextureColumns,
                ribbon.MaterialId,
                ribbon.Gravity,
                ribbon.TranslationTrack,
                ribbon.RotationTrack,
                ribbon.ScalingTrack,
                ribbon.HeightAboveTrack,
                ribbon.HeightBelowTrack,
                ribbon.AlphaTrack,
                ribbon.ColorTrack,
                ribbon.TextureSlotTrack,
                ribbon.VisibilityTrack));
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
