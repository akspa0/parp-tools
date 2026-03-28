using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.Chunks;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.IO.Mdx;

public static class MdxSummaryReader
{
    private const int SignatureSizeBytes = 4;
    private const int ModlNameSizeBytes = 0x50;
    private const int ModlBoundsAndBlendSizeBytes = 0x18 + sizeof(uint);
    private const int ModlSummarySizeBytes = ModlNameSizeBytes + ModlBoundsAndBlendSizeBytes;
    private const int PivtEntrySizeBytes = 12;
    private const int SeqsNameSizeBytes = 0x50;
    private const int SeqsCountedNamedRecordSizeBytes = 0x8C;
    private const int TexsEntrySizeLegacy = 0x108;
    private const int TexsEntrySizeExtended = 0x10C;
    private const int TexsPathSizeLegacy = 0x100;
    private const int TexsPathSizeExtended = 0x104;
    private const int CamsNameSizeBytes = 0x50;
    private const int Pre2ModelPathSizeBytes = 0x104;
    private const int Pre2ClassicEmitterPayloadSizeMinBytes = 791;

    private static readonly uint[] LegacySeqsEntrySizes = [140u, 136u, 132u, 128u];

    private static readonly HashSet<FourCC> KnownChunkIds =
    [
        MdxChunkIds.Vers,
        MdxChunkIds.Modl,
        MdxChunkIds.Seqs,
        MdxChunkIds.Glbs,
        MdxChunkIds.Mtls,
        MdxChunkIds.Texs,
        MdxChunkIds.Geos,
        MdxChunkIds.Geoa,
        MdxChunkIds.Bone,
        MdxChunkIds.Help,
        MdxChunkIds.Pivt,
        MdxChunkIds.Atch,
        MdxChunkIds.Lite,
        MdxChunkIds.Prem,
        MdxChunkIds.Pre2,
        MdxChunkIds.Ribb,
        MdxChunkIds.Evts,
        MdxChunkIds.Cams,
        MdxChunkIds.Clid,
        MdxChunkIds.Htst,
        MdxChunkIds.Txan,
        MdxChunkIds.Corn,
    ];

    public static MdxSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static MdxSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        if (!stream.CanSeek)
            throw new ArgumentException("MDX summary reading requires a seekable stream.", nameof(stream));

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

            List<MdxChunkSummary> chunks = [];
            uint? version = null;
            string? modelName = null;
            uint? blendTime = null;
            Vector3? boundsMin = null;
            Vector3? boundsMax = null;
            List<MdxGlobalSequenceSummary> globalSequences = [];
            List<MdxSequenceSummary> sequences = [];
            List<MdxGeosetSummary> geosets = [];
            List<MdxGeosetAnimationSummary> geosetAnimations = [];
            List<MdxBoneSummary> bones = [];
            List<MdxHelperSummary> helpers = [];
            List<MdxAttachmentSummary> attachments = [];
            List<MdxParticleEmitter2Summary> particleEmitters2 = [];
            List<MdxRibbonEmitterSummary> ribbons = [];
            List<MdxCameraSummary> cameras = [];
            List<MdxEventSummary> events = [];
            List<MdxHitTestShapeSummary> hitTestShapes = [];
            MdxCollisionSummary? collision = null;
            List<MdxPivotPointSummary> pivotPoints = [];
            List<MdxTextureSummary> textures = [];
            List<MdxMaterialSummary> materials = [];
            int knownChunkCount = 0;
            int unknownChunkCount = 0;
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

                bool isKnownChunk = KnownChunkIds.Contains(header.Id);
                if (isKnownChunk)
                    knownChunkCount++;
                else
                    unknownChunkCount++;

                chunks.Add(new MdxChunkSummary(header.Id, header.Size, headerOffset, dataOffset, isKnownChunk));

                if (header.Id == MdxChunkIds.Vers && header.Size >= sizeof(uint))
                {
                    version = ReadUInt32At(stream, dataOffset);
                }
                else if (header.Id == MdxChunkIds.Modl)
                {
                    ReadModlSummary(stream, dataOffset, header.Size, out modelName, out blendTime, out boundsMin, out boundsMax);
                }
                else if (header.Id == MdxChunkIds.Glbs)
                {
                    globalSequences = ReadGlbsSummary(stream, dataOffset, header.Size);
                }
                else if (header.Id == MdxChunkIds.Seqs)
                {
                    sequences = ReadSeqsSummary(stream, dataOffset, header.Size);
                }
                else if (header.Id == MdxChunkIds.Geos)
                {
                    geosets = ReadGeosSummary(stream, dataOffset, header.Size, version);
                }
                else if (header.Id == MdxChunkIds.Geoa)
                {
                    geosetAnimations = ReadGeoaSummary(stream, dataOffset, header.Size, version);
                }
                else if (header.Id == MdxChunkIds.Bone)
                {
                    bones = ReadBoneSummary(stream, dataOffset, header.Size, version);
                }
                else if (header.Id == MdxChunkIds.Help)
                {
                    helpers = ReadHelpSummary(stream, dataOffset, header.Size, version);
                }
                else if (header.Id == MdxChunkIds.Atch)
                {
                    attachments = ReadAtchSummary(stream, dataOffset, header.Size, version);
                }
                else if (header.Id == MdxChunkIds.Pre2)
                {
                    particleEmitters2 = ReadPre2Summary(stream, dataOffset, header.Size, version);
                }
                else if (header.Id == MdxChunkIds.Ribb)
                {
                    ribbons = ReadRibbSummary(stream, dataOffset, header.Size, version);
                }
                else if (header.Id == MdxChunkIds.Cams)
                {
                    cameras = ReadCamsSummary(stream, dataOffset, header.Size, version);
                }
                else if (header.Id == MdxChunkIds.Evts)
                {
                    events = ReadEvtsSummary(stream, dataOffset, header.Size, version);
                }
                else if (header.Id == MdxChunkIds.Htst)
                {
                    hitTestShapes = ReadHtstSummary(stream, dataOffset, header.Size, version);
                }
                else if (header.Id == MdxChunkIds.Clid)
                {
                    collision = ReadClidSummary(stream, dataOffset, header.Size, version);
                }
                else if (header.Id == MdxChunkIds.Pivt)
                {
                    pivotPoints = ReadPivtSummary(stream, dataOffset, header.Size);
                }
                else if (header.Id == MdxChunkIds.Texs)
                {
                    textures = ReadTexsSummary(stream, dataOffset, header.Size);
                }
                else if (header.Id == MdxChunkIds.Mtls)
                {
                    materials = ReadMtlsSummary(stream, dataOffset, header.Size);
                }

                stream.Position = endOffset;
            }

            return new MdxSummary(sourcePath, signature, version, modelName, blendTime, boundsMin, boundsMax, globalSequences, sequences, geosets, geosetAnimations, bones, helpers, attachments, particleEmitters2, ribbons, cameras, events, hitTestShapes, collision, pivotPoints, textures, materials, chunks, knownChunkCount, unknownChunkCount);
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static List<MdxGlobalSequenceSummary> ReadGlbsSummary(Stream stream, long dataOffset, uint size)
    {
        if (size % sizeof(uint) != 0)
            throw new InvalidDataException("GLBS: payload size must be divisible by 4.");

        long previousPosition = stream.Position;
        try
        {
            stream.Position = dataOffset;
            int count = checked((int)(size / sizeof(uint)));
            List<MdxGlobalSequenceSummary> globalSequences = new(count);

            for (int index = 0; index < count; index++)
                globalSequences.Add(new MdxGlobalSequenceSummary(index, ReadUInt32(stream)));

            return globalSequences;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static List<MdxBoneSummary> ReadBoneSummary(Stream stream, long dataOffset, uint size, uint? version)
    {
        if (version is not null and not 1300u and not 1400u)
            return [];

        long previousPosition = stream.Position;
        try
        {
            long chunkEnd = checked(dataOffset + size);
            stream.Position = dataOffset;
            if (chunkEnd - stream.Position < sizeof(uint))
                throw new InvalidDataException("BONE(v1300): missing bone count.");

            uint boneCount = ReadUInt32(stream);
            if (boneCount > 100000)
                throw new InvalidDataException($"BONE(v1300): invalid bone count {boneCount}.");

            List<MdxBoneSummary> bones = new(checked((int)boneCount));
            for (int index = 0; index < boneCount; index++)
            {
                if (chunkEnd - stream.Position < sizeof(uint))
                    throw new InvalidDataException($"BONE(v1300): truncated node header at index {index}.");

                long nodeStart = stream.Position;
                uint nodeSize = ReadUInt32(stream);
                long nodeEnd = checked(nodeStart + nodeSize);
                if (nodeEnd > chunkEnd || nodeEnd <= nodeStart)
                    throw new InvalidDataException($"BONE(v1300): invalid node size 0x{nodeSize:X} at index {index}.");

                (string name, int objectId, int parentId, uint flags, MdxNodeTrackSummary? translationTrack, MdxNodeTrackSummary? rotationTrack, MdxNodeTrackSummary? scalingTrack) =
                    ReadNodeTrackSummary(stream, nodeEnd, index, "BONE(v1300)");

                if (chunkEnd - stream.Position < 8)
                    throw new InvalidDataException($"BONE(v1300): missing geoset fields at index {index}.");

                uint geosetId = ReadUInt32(stream);
                uint geosetAnimationId = ReadUInt32(stream);
                bones.Add(new MdxBoneSummary(index, name, objectId, parentId, flags, geosetId, geosetAnimationId, translationTrack, rotationTrack, scalingTrack));
            }

            stream.Position = chunkEnd;
            return bones;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static List<MdxHelperSummary> ReadHelpSummary(Stream stream, long dataOffset, uint size, uint? version)
    {
        if (version is not null and not 1300u and not 1400u)
            return [];

        long previousPosition = stream.Position;
        try
        {
            long chunkEnd = checked(dataOffset + size);
            stream.Position = dataOffset;
            if (chunkEnd - stream.Position < sizeof(uint))
                throw new InvalidDataException("HELP(v1300): missing helper count.");

            uint helperCount = ReadUInt32(stream);
            if (helperCount > 100000)
                throw new InvalidDataException($"HELP(v1300): invalid helper count {helperCount}.");

            List<MdxHelperSummary> helpers = new(checked((int)helperCount));
            for (int index = 0; index < helperCount; index++)
            {
                if (chunkEnd - stream.Position < sizeof(uint))
                    throw new InvalidDataException($"HELP(v1300): truncated node header at index {index}.");

                long nodeStart = stream.Position;
                uint nodeSize = ReadUInt32(stream);
                long nodeEnd = checked(nodeStart + nodeSize);
                if (nodeEnd > chunkEnd || nodeEnd <= nodeStart)
                    throw new InvalidDataException($"HELP(v1300): invalid node size 0x{nodeSize:X} at index {index}.");

                (string name, int objectId, int parentId, uint flags, MdxNodeTrackSummary? translationTrack, MdxNodeTrackSummary? rotationTrack, MdxNodeTrackSummary? scalingTrack) =
                    ReadNodeTrackSummary(stream, nodeEnd, index, "HELP(v1300)");

                helpers.Add(new MdxHelperSummary(index, name, objectId, parentId, flags, translationTrack, rotationTrack, scalingTrack));
            }

            stream.Position = chunkEnd;
            return helpers;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static List<MdxAttachmentSummary> ReadAtchSummary(Stream stream, long dataOffset, uint size, uint? version)
    {
        if (version is not null and not 1300u and not 1400u)
            return [];

        long previousPosition = stream.Position;
        try
        {
            long chunkEnd = checked(dataOffset + size);
            stream.Position = dataOffset;
            if (chunkEnd - stream.Position < 8)
                throw new InvalidDataException("ATCH(v1300): missing attachment count or unused field.");

            uint attachmentCount = ReadUInt32(stream);
            if (attachmentCount > 100000)
                throw new InvalidDataException($"ATCH(v1300): invalid attachment count {attachmentCount}.");

            _ = ReadUInt32(stream);

            List<MdxAttachmentSummary> attachments = new(checked((int)attachmentCount));
            for (int index = 0; index < attachmentCount; index++)
            {
                if (chunkEnd - stream.Position < sizeof(uint) * 2)
                    throw new InvalidDataException($"ATCH(v1300): truncated section header at index {index}.");

                long entryStart = stream.Position;
                uint entrySize = ReadUInt32(stream);
                long entryEnd = checked(entryStart + entrySize);
                if (entryEnd > chunkEnd || entryEnd <= entryStart)
                    throw new InvalidDataException($"ATCH(v1300): invalid section size 0x{entrySize:X} at index {index}.");

                long nodeStart = stream.Position;
                uint nodeSize = ReadUInt32(stream);
                long nodeEnd = checked(nodeStart + nodeSize);
                if (nodeEnd > entryEnd || nodeEnd <= nodeStart)
                    throw new InvalidDataException($"ATCH(v1300): invalid node size 0x{nodeSize:X} at index {index}.");

                (string name, int objectId, int parentId, uint flags, MdxNodeTrackSummary? translationTrack, MdxNodeTrackSummary? rotationTrack, MdxNodeTrackSummary? scalingTrack) =
                    ReadNodeTrackSummary(stream, nodeEnd, index, "ATCH(v1300)");

                if (entryEnd - stream.Position < 4 + 1 + 0x104)
                    throw new InvalidDataException($"ATCH(v1300): missing attachment fields at index {index}.");

                uint attachmentId = ReadUInt32(stream);
                _ = ReadBytes(stream, 1);
                string? path = ReadNullTerminatedAscii(ReadBytes(stream, 0x104));

                MdxVisibilityTrackSummary? visibilityTrack = null;
                while (stream.Position <= entryEnd - 4)
                {
                    string tag = ReadTag(stream);
                    switch (tag)
                    {
                        case "KVIS":
                        case "KATV":
                            visibilityTrack = ReadVisibilityTrackSummary(stream, entryEnd, tag, $"ATCH(v1300): {tag} payload overran the section.");
                            break;
                        default:
                            stream.Position -= 4;
                            goto AttachmentDone;
                    }
                }

            AttachmentDone:
                stream.Position = entryEnd;
                attachments.Add(new MdxAttachmentSummary(index, name, objectId, parentId, flags, attachmentId, path, translationTrack, rotationTrack, scalingTrack, visibilityTrack));
            }

            stream.Position = chunkEnd;
            return attachments;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static List<MdxRibbonEmitterSummary> ReadRibbSummary(Stream stream, long dataOffset, uint size, uint? version)
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

            List<MdxRibbonEmitterSummary> ribbons = new(checked((int)ribbonCount));
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

                (string name, int objectId, int parentId, uint flags, MdxNodeTrackSummary? translationTrack, MdxNodeTrackSummary? rotationTrack, MdxNodeTrackSummary? scalingTrack) =
                    ReadNodeTrackSummary(stream, nodeEnd, index, "RIBB(v1300)");

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

                MdxTrackSummary? heightAboveTrack = null;
                MdxTrackSummary? heightBelowTrack = null;
                MdxTrackSummary? alphaTrack = null;
                MdxTrackSummary? colorTrack = null;
                MdxTrackSummary? textureSlotTrack = null;
                MdxVisibilityTrackSummary? visibilityTrack = null;

                while (stream.Position <= entryEnd - 4)
                {
                    string tag = ReadTag(stream);
                    switch (tag)
                    {
                        case "KRHA":
                        case "KRHB":
                        case "KRAL":
                            MdxTrackSummary scalarTrack = ReadTrackSummary(stream, entryEnd, tag, sizeof(float), $"RIBB(v1300): {tag} payload overran the emitter.");
                            if (tag == "KRHA")
                                heightAboveTrack = scalarTrack;
                            else if (tag == "KRHB")
                                heightBelowTrack = scalarTrack;
                            else
                                alphaTrack = scalarTrack;
                            break;
                        case "KRCO":
                            colorTrack = ReadTrackSummary(stream, entryEnd, tag, sizeof(float) * 3, $"RIBB(v1300): {tag} payload overran the emitter.");
                            break;
                        case "KRTX":
                            textureSlotTrack = ReadTrackSummary(stream, entryEnd, tag, sizeof(int), $"RIBB(v1300): {tag} payload overran the emitter.");
                            break;
                        case "KVIS":
                        case "KATV":
                            visibilityTrack = ReadVisibilityTrackSummary(stream, entryEnd, tag, $"RIBB(v1300): {tag} payload overran the emitter.");
                            break;
                        default:
                            stream.Position -= 4;
                            goto RibbonDone;
                    }
                }

            RibbonDone:
                stream.Position = entryEnd;
                ribbons.Add(new MdxRibbonEmitterSummary(index, name, objectId, parentId, flags, staticHeightAbove, staticHeightBelow, staticAlpha, staticColor, edgeLifetime, staticTextureSlot, edgesPerSecond, textureRows, textureColumns, materialId, gravity, translationTrack, rotationTrack, scalingTrack, heightAboveTrack, heightBelowTrack, alphaTrack, colorTrack, textureSlotTrack, visibilityTrack));
            }

            stream.Position = chunkEnd;
            return ribbons;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static List<MdxCameraSummary> ReadCamsSummary(Stream stream, long dataOffset, uint size, uint? version)
    {
        if (version is not null and not 1300u and not 1400u)
            return [];

        long previousPosition = stream.Position;
        try
        {
            long chunkEnd = checked(dataOffset + size);
            stream.Position = dataOffset;
            if (chunkEnd - stream.Position < sizeof(uint))
                throw new InvalidDataException("CAMS(v1300): missing camera count.");

            uint cameraCount = ReadUInt32(stream);
            if (cameraCount > 100000)
                throw new InvalidDataException($"CAMS(v1300): invalid camera count {cameraCount}.");

            List<MdxCameraSummary> cameras = new(checked((int)cameraCount));
            for (int index = 0; index < cameraCount; index++)
            {
                long entryStart = stream.Position;
                if (chunkEnd - entryStart < sizeof(uint))
                    throw new InvalidDataException($"CAMS(v1300): truncated camera header at index {index}.");

                uint entrySize = ReadUInt32(stream);
                long entryEnd = checked(entryStart + entrySize);
                if (entryEnd > chunkEnd || entryEnd <= entryStart)
                    throw new InvalidDataException($"CAMS(v1300): invalid camera size 0x{entrySize:X} at index {index}.");

                if (entryEnd - stream.Position < CamsNameSizeBytes + 36)
                    throw new InvalidDataException($"CAMS(v1300): truncated camera payload at index {index}.");

                string name = ReadFixedAscii(stream, CamsNameSizeBytes);
                Vector3 pivotPoint = ReadVector3(stream);
                float fieldOfView = ReadSingle(stream);
                float farClip = ReadSingle(stream);
                float nearClip = ReadSingle(stream);
                Vector3 targetPivotPoint = ReadVector3(stream);

                MdxTrackSummary? positionTrack = null;
                MdxTrackSummary? rollTrack = null;
                MdxVisibilityTrackSummary? visibilityTrack = null;
                MdxTrackSummary? targetPositionTrack = null;

                while (stream.Position <= entryEnd - 4)
                {
                    string tag = ReadTag(stream);
                    switch (tag)
                    {
                        case "KCTR":
                            positionTrack = ReadTrackSummary(stream, entryEnd, tag, sizeof(float) * 3, $"CAMS(v1300): {tag} payload overran the camera.");
                            break;
                        case "KCRL":
                            rollTrack = ReadTrackSummary(stream, entryEnd, tag, sizeof(float), $"CAMS(v1300): {tag} payload overran the camera.");
                            break;
                        case "KVIS":
                            visibilityTrack = ReadVisibilityTrackSummary(stream, entryEnd, tag, $"CAMS(v1300): {tag} payload overran the camera.");
                            break;
                        case "KTTR":
                            targetPositionTrack = ReadTrackSummary(stream, entryEnd, tag, sizeof(float) * 3, $"CAMS(v1300): {tag} payload overran the camera.");
                            break;
                        default:
                            stream.Position -= 4;
                            stream.Position = entryEnd;
                            break;
                    }
                }

                cameras.Add(new MdxCameraSummary(index, name, pivotPoint, fieldOfView, farClip, nearClip, targetPivotPoint, positionTrack, rollTrack, visibilityTrack, targetPositionTrack));
                stream.Position = entryEnd;
            }

            stream.Position = chunkEnd;
            return cameras;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static List<MdxEventSummary> ReadEvtsSummary(Stream stream, long dataOffset, uint size, uint? version)
    {
        if (version is not null and not 1300u and not 1400u)
            return [];

        long previousPosition = stream.Position;
        try
        {
            long chunkEnd = checked(dataOffset + size);
            stream.Position = dataOffset;
            if (chunkEnd - stream.Position < sizeof(uint))
                throw new InvalidDataException("EVTS(v1300): missing event count.");

            uint eventCount = ReadUInt32(stream);
            if (eventCount > 100000)
                throw new InvalidDataException($"EVTS(v1300): invalid event count {eventCount}.");

            List<MdxEventSummary> events = new(checked((int)eventCount));
            for (int index = 0; index < eventCount; index++)
            {
                if (chunkEnd - stream.Position < sizeof(uint) * 2)
                    throw new InvalidDataException($"EVTS(v1300): truncated section header at index {index}.");

                long entryStart = stream.Position;
                uint entrySize = ReadUInt32(stream);
                long entryEnd = checked(entryStart + entrySize);
                if (entryEnd > chunkEnd || entryEnd <= entryStart)
                    throw new InvalidDataException($"EVTS(v1300): invalid section size 0x{entrySize:X} at index {index}.");

                long nodeStart = stream.Position;
                uint nodeSize = ReadUInt32(stream);
                long nodeEnd = checked(nodeStart + nodeSize);
                if (nodeEnd > entryEnd || nodeEnd <= nodeStart)
                    throw new InvalidDataException($"EVTS(v1300): invalid node size 0x{nodeSize:X} at index {index}.");

                (string name, int objectId, int parentId, uint flags, MdxNodeTrackSummary? translationTrack, MdxNodeTrackSummary? rotationTrack, MdxNodeTrackSummary? scalingTrack) =
                    ReadNodeTrackSummary(stream, nodeEnd, index, "EVTS(v1300)");

                MdxEventTrackSummary? eventTrack = null;
                while (stream.Position <= entryEnd - 4)
                {
                    string tag = ReadTag(stream);
                    switch (tag)
                    {
                        case "KEVT":
                            eventTrack = ReadEventTrackSummary(stream, entryEnd, tag, $"EVTS(v1300): {tag} payload overran the section.");
                            break;
                        default:
                            stream.Position -= 4;
                            stream.Position = entryEnd;
                            break;
                    }
                }

                events.Add(new MdxEventSummary(index, name, objectId, parentId, flags, translationTrack, rotationTrack, scalingTrack, eventTrack));
                stream.Position = entryEnd;
            }

            stream.Position = chunkEnd;
            return events;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static List<MdxParticleEmitter2Summary> ReadPre2Summary(Stream stream, long dataOffset, uint size, uint? version)
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

            List<MdxParticleEmitter2Summary> particleEmitters2 = new(checked((int)emitterCount));
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

                (string name, int objectId, int parentId, uint flags, MdxNodeTrackSummary? translationTrack, MdxNodeTrackSummary? rotationTrack, MdxNodeTrackSummary? scalingTrack) =
                    ReadNodeTrackSummary(stream, nodeEnd, index, "PRE2(v1300)");

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

                for (int intervalIndex = 0; intervalIndex < 12; intervalIndex++)
                    _ = ReadUInt32(stream);

                uint blendMode = ReadUInt32(stream);
                int textureId = ReadInt32(stream);
                int priorityPlane = ReadInt32(stream);
                uint replaceableId = ReadUInt32(stream);
                string? geometryModel = ReadNullTerminatedAscii(ReadBytes(stream, Pre2ModelPathSizeBytes));
                string? recursionModel = ReadNullTerminatedAscii(ReadBytes(stream, Pre2ModelPathSizeBytes));

                _ = ReadSingle(stream);
                _ = ReadSingle(stream);
                _ = ReadSingle(stream);
                _ = ReadSingle(stream);
                _ = ReadSingle(stream);

                for (int tumbleIndex = 0; tumbleIndex < 6; tumbleIndex++)
                    _ = ReadSingle(stream);

                _ = ReadSingle(stream);
                _ = ReadSingle(stream);
                _ = ReadVector3(stream);
                _ = ReadSingle(stream);
                _ = ReadSingle(stream);
                _ = ReadSingle(stream);
                _ = ReadSingle(stream);
                _ = ReadSingle(stream);

                uint splineCount = ReadUInt32(stream);
                if (splineCount > 100000)
                    throw new InvalidDataException($"PRE2(v1300): invalid spline count {splineCount} at index {index}.");

                SkipBytes(stream, checked((long)splineCount * sizeof(float) * 3), emitterPayloadEnd, $"PRE2(v1300): spline payload overran emitter {index}.");
                int squirts = ReadInt32(stream);

                if (stream.Position > emitterPayloadEnd)
                    throw new InvalidDataException($"PRE2(v1300): static emitter payload overran entry {index}.");

                stream.Position = emitterPayloadEnd;

                MdxVisibilityTrackSummary? visibilityTrack = null;
                MdxTrackSummary? speedTrack = null;
                MdxTrackSummary? variationTrack = null;
                MdxTrackSummary? latitudeTrack = null;
                MdxTrackSummary? longitudeTrack = null;
                MdxTrackSummary? gravityTrack = null;
                MdxTrackSummary? lifeTrack = null;
                MdxTrackSummary? emissionRateTrack = null;
                MdxTrackSummary? widthTrack = null;
                MdxTrackSummary? lengthTrack = null;
                MdxTrackSummary? zSourceTrack = null;

                while (stream.Position <= entryEnd - 4)
                {
                    string tag = ReadTag(stream);
                    switch (tag)
                    {
                        case "KVIS":
                        case "KP2V":
                            visibilityTrack = ReadVisibilityTrackSummary(stream, entryEnd, tag, $"PRE2(v1300): {tag} payload overran the emitter.");
                            break;
                        case "KP2S":
                            speedTrack = ReadTrackSummary(stream, entryEnd, tag, sizeof(float), $"PRE2(v1300): {tag} payload overran the emitter.");
                            break;
                        case "KP2R":
                            variationTrack = ReadTrackSummary(stream, entryEnd, tag, sizeof(float), $"PRE2(v1300): {tag} payload overran the emitter.");
                            break;
                        case "KP2L":
                            latitudeTrack = ReadTrackSummary(stream, entryEnd, tag, sizeof(float), $"PRE2(v1300): {tag} payload overran the emitter.");
                            break;
                        case "KPLN":
                            longitudeTrack = ReadTrackSummary(stream, entryEnd, tag, sizeof(float), $"PRE2(v1300): {tag} payload overran the emitter.");
                            break;
                        case "KP2G":
                            gravityTrack = ReadTrackSummary(stream, entryEnd, tag, sizeof(float), $"PRE2(v1300): {tag} payload overran the emitter.");
                            break;
                        case "KLIF":
                            lifeTrack = ReadTrackSummary(stream, entryEnd, tag, sizeof(float), $"PRE2(v1300): {tag} payload overran the emitter.");
                            break;
                        case "KP2E":
                            emissionRateTrack = ReadTrackSummary(stream, entryEnd, tag, sizeof(float), $"PRE2(v1300): {tag} payload overran the emitter.");
                            break;
                        case "KP2W":
                            widthTrack = ReadTrackSummary(stream, entryEnd, tag, sizeof(float), $"PRE2(v1300): {tag} payload overran the emitter.");
                            break;
                        case "KP2N":
                            lengthTrack = ReadTrackSummary(stream, entryEnd, tag, sizeof(float), $"PRE2(v1300): {tag} payload overran the emitter.");
                            break;
                        case "KP2Z":
                            zSourceTrack = ReadTrackSummary(stream, entryEnd, tag, sizeof(float), $"PRE2(v1300): {tag} payload overran the emitter.");
                            break;
                        default:
                            stream.Position -= 4;
                            goto Pre2Done;
                    }
                }

            Pre2Done:
                stream.Position = entryEnd;
                particleEmitters2.Add(new MdxParticleEmitter2Summary(index, name, objectId, parentId, flags, emitterType, staticSpeed, staticVariation, staticLatitude, staticLongitude, staticGravity, staticZSource, staticLife, staticEmissionRate, staticLength, staticWidth, rows, columns, particleType, tailLength, middleTime, startColor, middleColor, endColor, startAlpha, middleAlpha, endAlpha, startScale, middleScale, endScale, blendMode, textureId, priorityPlane, replaceableId, geometryModel, recursionModel, splineCount, squirts, translationTrack, rotationTrack, scalingTrack, visibilityTrack, speedTrack, variationTrack, latitudeTrack, longitudeTrack, gravityTrack, lifeTrack, emissionRateTrack, widthTrack, lengthTrack, zSourceTrack));
            }

            stream.Position = chunkEnd;
            return particleEmitters2;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static MdxCollisionSummary? ReadClidSummary(Stream stream, long dataOffset, uint size, uint? version)
    {
        if (version is not null and not 1300u and not 1400u)
            return null;

        long previousPosition = stream.Position;
        try
        {
            long chunkEnd = checked(dataOffset + size);
            stream.Position = dataOffset;

            ExpectTag(stream, "VRTX", "CLID(v1300): expected VRTX.");
            int vertexCount = ReadNonNegativeCount(stream, "CLID(v1300): negative VRTX count.");

            Vector3? boundsMin = null;
            Vector3? boundsMax = null;
            if (vertexCount > 0)
            {
                Vector3 firstVertex = ReadVector3(stream);
                Vector3 min = firstVertex;
                Vector3 max = firstVertex;
                for (int index = 1; index < vertexCount; index++)
                {
                    Vector3 vertex = ReadVector3(stream);
                    min = Vector3.Min(min, vertex);
                    max = Vector3.Max(max, vertex);
                }

                boundsMin = min;
                boundsMax = max;
            }

            ExpectTag(stream, "TRI ", "CLID(v1300): expected TRI .");
            int triangleIndexCount = ReadNonNegativeCount(stream, "CLID(v1300): negative TRI count.");
            if (triangleIndexCount % 3 != 0)
                throw new InvalidDataException("CLID(v1300): TRI count must be divisible by 3.");

            int maxTriangleIndex = 0;
            for (int index = 0; index < triangleIndexCount; index++)
            {
                int triangleIndex = ReadUInt16(stream);

                if (triangleIndex >= vertexCount)
                    throw new InvalidDataException($"CLID(v1300): TRI index {triangleIndex} exceeded VRTX count {vertexCount}.");

                maxTriangleIndex = Math.Max(maxTriangleIndex, triangleIndex);
            }

            ExpectTag(stream, "NRMS", "CLID(v1300): expected NRMS.");
            int facetNormalCount = ReadNonNegativeCount(stream, "CLID(v1300): negative NRMS count.");
            SkipBytes(stream, checked((long)facetNormalCount * sizeof(float) * 3), chunkEnd, "CLID(v1300): NRMS payload overran the chunk.");

            if (stream.Position != chunkEnd)
                throw new InvalidDataException("CLID(v1300): chunk contained unexpected trailing bytes.");

            return new MdxCollisionSummary(vertexCount, triangleIndexCount, triangleIndexCount / 3, facetNormalCount, maxTriangleIndex, boundsMin, boundsMax);
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static List<MdxHitTestShapeSummary> ReadHtstSummary(Stream stream, long dataOffset, uint size, uint? version)
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

            List<MdxHitTestShapeSummary> hitTestShapes = new(checked((int)shapeCount));
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

                (string name, int objectId, int parentId, uint flags, MdxNodeTrackSummary? translationTrack, MdxNodeTrackSummary? rotationTrack, MdxNodeTrackSummary? scalingTrack) =
                    ReadNodeTrackSummary(stream, nodeEnd, index, "HTST(v1300)");

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
                        if (entryEnd - stream.Position < sizeof(float) * 6)
                            throw new InvalidDataException($"HTST(v1300): truncated box payload at index {index}.");

                        minimum = ReadVector3(stream);
                        maximum = ReadVector3(stream);
                        break;
                    case MdxGeometryShapeType.Cylinder:
                        if (entryEnd - stream.Position < sizeof(float) * 5)
                            throw new InvalidDataException($"HTST(v1300): truncated cylinder payload at index {index}.");

                        basePoint = ReadVector3(stream);
                        height = ReadSingle(stream);
                        radius = ReadSingle(stream);
                        break;
                    case MdxGeometryShapeType.Sphere:
                        if (entryEnd - stream.Position < sizeof(float) * 4)
                            throw new InvalidDataException($"HTST(v1300): truncated sphere payload at index {index}.");

                        center = ReadVector3(stream);
                        radius = ReadSingle(stream);
                        break;
                    case MdxGeometryShapeType.Plane:
                        if (entryEnd - stream.Position < sizeof(float) * 2)
                            throw new InvalidDataException($"HTST(v1300): truncated plane payload at index {index}.");

                        length = ReadSingle(stream);
                        width = ReadSingle(stream);
                        break;
                }

                if (stream.Position > entryEnd)
                    throw new InvalidDataException($"HTST(v1300): payload overran section {index}.");

                stream.Position = entryEnd;
                hitTestShapes.Add(new MdxHitTestShapeSummary(index, name, objectId, parentId, flags, translationTrack, rotationTrack, scalingTrack, shapeType, minimum, maximum, basePoint, height, radius, center, length, width));
            }

            stream.Position = chunkEnd;
            return hitTestShapes;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static (string Name, int ObjectId, int ParentId, uint Flags, MdxNodeTrackSummary? TranslationTrack, MdxNodeTrackSummary? RotationTrack, MdxNodeTrackSummary? ScalingTrack) ReadNodeTrackSummary(Stream stream, long nodeEnd, int index, string chunkLabel)
    {
        if (nodeEnd - stream.Position < 0x50 + 12)
            throw new InvalidDataException($"{chunkLabel}: truncated node payload at index {index}.");

        byte[] nameBytes = ReadBytes(stream, 0x50);
        string name = ReadNullTerminatedAscii(nameBytes);
        int objectId = ReadInt32(stream);
        int parentId = ReadInt32(stream);
        uint flags = ReadUInt32(stream);

        MdxNodeTrackSummary? translationTrack = null;
        MdxNodeTrackSummary? rotationTrack = null;
        MdxNodeTrackSummary? scalingTrack = null;

        while (stream.Position <= nodeEnd - 4)
        {
            string tag = ReadTag(stream);
            switch (tag)
            {
                case "KGTR":
                    translationTrack = ReadNodeVectorTrackSummary(stream, nodeEnd, tag, $"{chunkLabel}: KGTR payload overran the node.");
                    break;
                case "KGRT":
                    rotationTrack = ReadNodeQuaternionTrackSummary(stream, nodeEnd, tag, $"{chunkLabel}: KGRT payload overran the node.");
                    break;
                case "KGSC":
                    scalingTrack = ReadNodeVectorTrackSummary(stream, nodeEnd, tag, $"{chunkLabel}: KGSC payload overran the node.");
                    break;
                default:
                    stream.Position -= 4;
                    stream.Position = nodeEnd;
                    break;
            }
        }

        stream.Position = nodeEnd;
        return (name, objectId, parentId, flags, translationTrack, rotationTrack, scalingTrack);
    }

    private static List<MdxGeosetAnimationSummary> ReadGeoaSummary(Stream stream, long dataOffset, uint size, uint? version)
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

            List<MdxGeosetAnimationSummary> geosetAnimations = new(checked((int)animationCount));
            for (int index = 0; index < animationCount; index++)
            {
                long entryStart = stream.Position;
                if (chunkEnd - entryStart < sizeof(uint))
                    throw new InvalidDataException($"GEOA(v1300): truncated entry header at index {index}.");

                uint entrySize = ReadUInt32(stream);
                long entryEnd = checked(entryStart + entrySize);
                if (entryEnd > chunkEnd || entryEnd <= entryStart)
                    throw new InvalidDataException($"GEOA(v1300): invalid entry size 0x{entrySize:X} at index {index}.");

                if (entryEnd - stream.Position < 24)
                    throw new InvalidDataException($"GEOA(v1300): truncated entry payload at index {index}.");

                uint geosetId = ReadUInt32(stream);
                float staticAlpha = ReadSingle(stream);
                Vector3 staticColor = ReadVector3(stream);
                uint flags = ReadUInt32(stream);

                MdxGeosetAnimationTrackSummary? alphaTrack = null;
                MdxGeosetAnimationTrackSummary? colorTrack = null;

                while (stream.Position <= entryEnd - 4)
                {
                    string tag = ReadTag(stream);
                    switch (tag)
                    {
                        case "KGAO":
                            alphaTrack = ReadGeosetAnimationAlphaTrackSummary(stream, entryEnd);
                            break;
                        case "KGAC":
                            colorTrack = ReadGeosetAnimationColorTrackSummary(stream, entryEnd);
                            break;
                        default:
                            stream.Position -= 4;
                            stream.Position = entryEnd;
                            break;
                    }
                }

                geosetAnimations.Add(new MdxGeosetAnimationSummary(index, geosetId, staticAlpha, staticColor, flags, alphaTrack, colorTrack));
                stream.Position = entryEnd;
            }

            stream.Position = chunkEnd;
            return geosetAnimations;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static List<MdxGeosetSummary> ReadGeosSummary(Stream stream, long dataOffset, uint size, uint? version)
    {
        if (version is not null and not 1300u and not 1400u)
            return [];

        long previousPosition = stream.Position;
        try
        {
            long chunkEnd = checked(dataOffset + size);
            stream.Position = dataOffset;
            if (chunkEnd - stream.Position < sizeof(int))
                throw new InvalidDataException("GEOS(v1300): missing geoset count.");

            int geosetCount = ReadInt32(stream);
            if (geosetCount < 0 || geosetCount > 100000)
                throw new InvalidDataException($"GEOS(v1300): invalid geoset count {geosetCount}.");

            List<MdxGeosetSummary> geosets = new(geosetCount);
            for (int index = 0; index < geosetCount; index++)
            {
                long geosetStart = stream.Position;
                if (chunkEnd - geosetStart < sizeof(uint))
                    throw new InvalidDataException($"GEOS(v1300): truncated geoset header at index {index}.");

                uint geosetSize = ReadUInt32(stream);
                long geosetEnd = checked(geosetStart + geosetSize);
                if (geosetEnd > chunkEnd || geosetEnd <= geosetStart)
                    throw new InvalidDataException($"GEOS(v1300): invalid geoset size 0x{geosetSize:X} at index {index}.");

                ExpectTag(stream, "VRTX", $"GEOS(v1300): expected VRTX at index {index}.");
                int vertexCount = ReadNonNegativeCount(stream, "GEOS(v1300): negative VRTX count.");
                SkipBytes(stream, checked((long)vertexCount * 12), geosetEnd, "GEOS(v1300): VRTX payload overran the geoset.");

                ExpectTag(stream, "NRMS", $"GEOS(v1300): expected NRMS at index {index}.");
                int normalCount = ReadNonNegativeCount(stream, "GEOS(v1300): negative NRMS count.");
                SkipBytes(stream, checked((long)normalCount * 12), geosetEnd, "GEOS(v1300): NRMS payload overran the geoset.");

                int uvSetCount = 0;
                int primaryUvCount = 0;
                if (TryReadTag(stream, "UVAS"))
                {
                    uvSetCount = ReadNonNegativeCount(stream, "GEOS(v1300): negative UVAS count.");
                    primaryUvCount = vertexCount;
                    SkipBytes(stream, checked((long)uvSetCount * vertexCount * 8), geosetEnd, "GEOS(v1300): UVAS payload overran the geoset.");
                }

                ExpectTag(stream, "PTYP", $"GEOS(v1300): expected PTYP at index {index}.");
                int primitiveTypeCount = ReadNonNegativeCount(stream, "GEOS(v1300): negative PTYP count.");
                for (int primitiveIndex = 0; primitiveIndex < primitiveTypeCount; primitiveIndex++)
                {
                    int primitiveType = stream.ReadByte();
                    if (primitiveType < 0)
                        throw new EndOfStreamException("GEOS(v1300): truncated PTYP payload.");

                    if (primitiveType != 4)
                        throw new InvalidDataException($"GEOS(v1300): unsupported primitive type {primitiveType}.");
                }

                ExpectTag(stream, "PCNT", $"GEOS(v1300): expected PCNT at index {index}.");
                int faceGroupCount = ReadNonNegativeCount(stream, "GEOS(v1300): negative PCNT count.");
                SkipBytes(stream, checked((long)faceGroupCount * sizeof(int)), geosetEnd, "GEOS(v1300): PCNT payload overran the geoset.");

                ExpectTag(stream, "PVTX", $"GEOS(v1300): expected PVTX at index {index}.");
                int indexCount = ReadNonNegativeCount(stream, "GEOS(v1300): negative PVTX count.");
                SkipBytes(stream, checked((long)indexCount * sizeof(ushort)), geosetEnd, "GEOS(v1300): PVTX payload overran the geoset.");

                ExpectTag(stream, "GNDX", $"GEOS(v1300): expected GNDX at index {index}.");
                int vertexGroupCount = ReadNonNegativeCount(stream, "GEOS(v1300): negative GNDX count.");
                SkipBytes(stream, vertexGroupCount, geosetEnd, "GEOS(v1300): GNDX payload overran the geoset.");

                ExpectTag(stream, "MTGC", $"GEOS(v1300): expected MTGC at index {index}.");
                int matrixGroupCount = ReadNonNegativeCount(stream, "GEOS(v1300): negative MTGC count.");
                SkipBytes(stream, checked((long)matrixGroupCount * sizeof(uint)), geosetEnd, "GEOS(v1300): MTGC payload overran the geoset.");

                ExpectTag(stream, "MATS", $"GEOS(v1300): expected MATS at index {index}.");
                int matrixIndexCount = ReadNonNegativeCount(stream, "GEOS(v1300): negative MATS count.");
                SkipBytes(stream, checked((long)matrixIndexCount * sizeof(uint)), geosetEnd, "GEOS(v1300): MATS payload overran the geoset.");

                if (TryReadTag(stream, "UVBS"))
                {
                    int uvCount = ReadNonNegativeCount(stream, "GEOS(v1300): negative UVBS count.");
                    if (primaryUvCount == 0)
                    {
                        primaryUvCount = uvCount;
                        uvSetCount = Math.Max(uvSetCount, 1);
                    }

                    SkipBytes(stream, checked((long)uvCount * 8), geosetEnd, "GEOS(v1300): UVBS payload overran the geoset.");
                }

                ExpectTag(stream, "BIDX", $"GEOS(v1300): expected BIDX at index {index}.");
                int boneIndexCount = ReadNonNegativeCount(stream, "GEOS(v1300): negative BIDX count.");
                SkipBytes(stream, checked((long)boneIndexCount * sizeof(uint)), geosetEnd, "GEOS(v1300): BIDX payload overran the geoset.");

                ExpectTag(stream, "BWGT", $"GEOS(v1300): expected BWGT at index {index}.");
                int boneWeightCount = ReadNonNegativeCount(stream, "GEOS(v1300): negative BWGT count.");
                SkipBytes(stream, checked((long)boneWeightCount * sizeof(uint)), geosetEnd, "GEOS(v1300): BWGT payload overran the geoset.");

                int materialId = ReadInt32(stream);
                uint selectionGroup = unchecked((uint)ReadInt32(stream));
                uint flags = unchecked((uint)ReadInt32(stream));
                float boundsRadius = ReadSingle(stream);
                Vector3 boundsMin = ReadVector3(stream);
                Vector3 boundsMax = ReadVector3(stream);
                int animationExtentCount = ReadNonNegativeCount(stream, "GEOS(v1300): negative geosetAnimCount.");
                SkipBytes(stream, checked((long)animationExtentCount * 28), geosetEnd, "GEOS(v1300): geoset animation extents overran the geoset.");

                geosets.Add(new MdxGeosetSummary(index, vertexCount, normalCount, uvSetCount, primaryUvCount, primitiveTypeCount, faceGroupCount, indexCount, vertexGroupCount, matrixGroupCount, matrixIndexCount, boneIndexCount, boneWeightCount, materialId, selectionGroup, flags, boundsRadius, boundsMin, boundsMax, animationExtentCount));
                stream.Position = geosetEnd;
            }

            stream.Position = chunkEnd;
            return geosets;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static List<MdxPivotPointSummary> ReadPivtSummary(Stream stream, long dataOffset, uint size)
    {
        if (size % PivtEntrySizeBytes != 0)
            throw new InvalidDataException($"Invalid PIVT size 0x{size:X}: expected multiple of {PivtEntrySizeBytes}.");

        long previousPosition = stream.Position;
        try
        {
            int count = checked((int)(size / PivtEntrySizeBytes));
            List<MdxPivotPointSummary> pivotPoints = new(count);

            stream.Position = dataOffset;
            for (int index = 0; index < count; index++)
            {
                pivotPoints.Add(new MdxPivotPointSummary(index, ReadVector3(stream)));
            }

            return pivotPoints;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static List<MdxSequenceSummary> ReadSeqsSummary(Stream stream, long dataOffset, uint size)
    {
        long previousPosition = stream.Position;
        try
        {
            List<MdxSequenceSummary> sequences = [];
            long chunkEnd = checked(dataOffset + size);
            stream.Position = dataOffset;

            if (size >= sizeof(uint))
            {
                uint count = ReadUInt32(stream);
                uint remaining = size - sizeof(uint);
                long seqDataStart = checked(dataOffset + sizeof(uint));

                if (count > 0 && remaining == count * SeqsCountedNamedRecordSizeBytes)
                {
                    int sampleCount = (int)Math.Min(count, 2u);
                    if (AllCountedNamedSeqsLookSane(stream, seqDataStart, sampleCount, chunkEnd))
                    {
                        stream.Position = seqDataStart;
                        for (int index = 0; index < count; index++)
                        {
                            sequences.Add(ParseCountedNamedSeqRecord8C(stream, index));
                        }

                        return sequences;
                    }

                    if (AllSeq090RecordsLookSane(stream, seqDataStart, sampleCount, chunkEnd))
                    {
                        stream.Position = seqDataStart;
                        for (int index = 0; index < count; index++)
                        {
                            sequences.Add(ParseSeq090Record(stream, index));
                        }

                        return sequences;
                    }
                }

                if (count > 0 && remaining % count == 0)
                {
                    uint entrySize = remaining / count;
                    if (entrySize is 128u or 132u or 136u or 140u)
                    {
                        stream.Position = seqDataStart;
                        for (int index = 0; index < count; index++)
                        {
                            sequences.Add(ParseLegacyNamedSeqRecord(stream, entrySize, index));
                        }

                        return sequences;
                    }
                }
            }

            foreach (uint entrySize in LegacySeqsEntrySizes)
            {
                if (size < entrySize)
                    continue;

                uint remainder = size % entrySize;
                if (remainder > 12)
                    continue;

                uint legacyCount = size / entrySize;
                if (legacyCount == 0)
                    continue;

                int sampleCount = (int)Math.Min(legacyCount, 2u);
                if (!AllLegacyNamedSeqsLookSane(stream, dataOffset, entrySize, sampleCount, chunkEnd))
                    continue;

                stream.Position = dataOffset;
                for (int index = 0; index < legacyCount; index++)
                {
                    sequences.Add(ParseLegacyNamedSeqRecord(stream, entrySize, index));
                }

                return sequences;
            }

            stream.Position = dataOffset;
            uint fallbackCount = size / 132u;
            for (int index = 0; index < fallbackCount; index++)
            {
                sequences.Add(ParseLegacyNamedSeqRecord(stream, 132u, index));
            }

            return sequences;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static uint ReadUInt32At(Stream stream, long offset)
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

    private static List<MdxTextureSummary> ReadTexsSummary(Stream stream, long dataOffset, uint size)
    {
        long previousPosition = stream.Position;
        try
        {
            (int entrySize, int pathSize) = ResolveTexsLayout(size);
            int count = checked((int)(size / entrySize));
            List<MdxTextureSummary> textures = new(count);
            byte[] replaceableBytes = new byte[sizeof(uint)];
            byte[] flagsBytes = new byte[sizeof(uint)];

            stream.Position = dataOffset;
            for (int index = 0; index < count; index++)
            {
                stream.ReadExactly(replaceableBytes);
                uint replaceableId = BinaryPrimitives.ReadUInt32LittleEndian(replaceableBytes);

                byte[] pathBytes = new byte[pathSize];
                stream.ReadExactly(pathBytes);
                string path = ReadNullTerminatedAscii(pathBytes);

                stream.ReadExactly(flagsBytes);
                uint flags = BinaryPrimitives.ReadUInt32LittleEndian(flagsBytes);

                textures.Add(new MdxTextureSummary(index, replaceableId, path, flags));
            }

            return textures;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static bool AllCountedNamedSeqsLookSane(Stream stream, long seqDataStart, int sampleCount, long chunkEnd)
    {
        for (int sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
        {
            long recordStart = checked(seqDataStart + (sampleIndex * SeqsCountedNamedRecordSizeBytes));
            if (!LooksLikeLegacyNamedSeqRecord(stream, recordStart, SeqsCountedNamedRecordSizeBytes, chunkEnd))
                return false;
        }

        return true;
    }

    private static bool AllSeq090RecordsLookSane(Stream stream, long seqDataStart, int sampleCount, long chunkEnd)
    {
        for (int sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
        {
            long recordStart = checked(seqDataStart + (sampleIndex * SeqsCountedNamedRecordSizeBytes));
            if (!LooksLikeSeq090Record(stream, recordStart, chunkEnd))
                return false;
        }

        return true;
    }

    private static bool AllLegacyNamedSeqsLookSane(Stream stream, long dataOffset, uint entrySize, int sampleCount, long chunkEnd)
    {
        for (int sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
        {
            long recordStart = checked(dataOffset + (sampleIndex * entrySize));
            if (!LooksLikeLegacyNamedSeqRecord(stream, recordStart, entrySize, chunkEnd))
                return false;
        }

        return true;
    }

    private static bool LooksLikeLegacyNamedSeqRecord(Stream stream, long recordStart, uint entrySize, long chunkEnd)
    {
        if (recordStart < 0 || checked(recordStart + entrySize) > chunkEnd)
            return false;

        long previousPosition = stream.Position;
        try
        {
            stream.Position = checked(recordStart + SeqsNameSizeBytes);
            uint startTime = ReadUInt32(stream);
            uint endTime = ReadUInt32(stream);
            float moveSpeed = ReadSingle(stream);

            bool intervalLooksRight = endTime >= startTime && (endTime - startTime) <= 0x0FFFFFFF;
            bool moveSpeedLooksRight = !float.IsNaN(moveSpeed) && !float.IsInfinity(moveSpeed) && moveSpeed >= 0f && moveSpeed < 10000f;
            return intervalLooksRight && moveSpeedLooksRight;
        }
        catch
        {
            return false;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static bool LooksLikeSeq090Record(Stream stream, long recordStart, long chunkEnd)
    {
        if (recordStart < 0 || checked(recordStart + SeqsCountedNamedRecordSizeBytes) > chunkEnd)
            return false;

        long previousPosition = stream.Position;
        try
        {
            stream.Position = recordStart;
            byte[] head = new byte[0x20];
            stream.ReadExactly(head);
            int printable = 0;
            for (int index = 0; index < head.Length; index++)
            {
                if (head[index] >= 32 && head[index] <= 126)
                    printable++;
            }

            if (printable >= 10)
                return false;

            stream.Position = checked(recordStart + 0x08);
            uint reserved0 = ReadUInt32(stream);
            uint reserved1 = ReadUInt32(stream);

            stream.Position = checked(recordStart + SeqsNameSizeBytes);
            uint startTime = ReadUInt32(stream);
            uint endTime = ReadUInt32(stream);
            float moveSpeed = ReadSingle(stream);

            bool reservedLooksRight = reserved0 == 0 && reserved1 == 0;
            bool intervalLooksRight = endTime >= startTime && (endTime - startTime) <= 0x0FFFFFFF;
            bool moveSpeedLooksRight = !float.IsNaN(moveSpeed) && !float.IsInfinity(moveSpeed) && moveSpeed >= 0f && moveSpeed < 10000f;
            return reservedLooksRight && intervalLooksRight && moveSpeedLooksRight;
        }
        catch
        {
            return false;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static MdxSequenceSummary ParseLegacyNamedSeqRecord(Stream stream, uint entrySize, int index)
    {
        long entryStart = stream.Position;

        string name = ReadFixedAscii(stream, SeqsNameSizeBytes);
        int startTime = ReadInt32(stream);
        int endTime = ReadInt32(stream);
        float moveSpeed = ReadSingle(stream);
        uint flags = ReadUInt32(stream);
        float frequency = ReadSingle(stream);

        int replayStart;
        int replayEnd;
        uint? blendTime = null;
        if (entrySize is 128u or 132u)
        {
            replayStart = ReadInt32(stream);
            replayEnd = 0;
        }
        else
        {
            replayStart = ReadInt32(stream);
            replayEnd = ReadInt32(stream);
            if (entrySize >= 140u)
                blendTime = ReadUInt32(stream);
        }

        float? boundsRadius;
        Vector3 boundsMin;
        Vector3 boundsMax;
        if (entrySize == 128u)
        {
            boundsMin = ReadVector3(stream);
            boundsMax = ReadVector3(stream);
            boundsRadius = ReadSingle(stream);
        }
        else
        {
            boundsRadius = ReadSingle(stream);
            boundsMin = ReadVector3(stream);
            boundsMax = ReadVector3(stream);
        }

        stream.Position = checked(entryStart + entrySize);
        return new MdxSequenceSummary(index, name, startTime, endTime, moveSpeed, flags, frequency, replayStart, replayEnd, blendTime, boundsMin, boundsMax, boundsRadius);
    }

    private static MdxSequenceSummary ParseCountedNamedSeqRecord8C(Stream stream, int index)
    {
        long entryStart = stream.Position;

        string name = ReadFixedAscii(stream, SeqsNameSizeBytes);
        int startTime = ReadInt32(stream);
        int endTime = ReadInt32(stream);
        float moveSpeed = ReadSingle(stream);
        uint flags = ReadUInt32(stream);
        Vector3 boundsMin = ReadVector3(stream);
        Vector3 boundsMax = ReadVector3(stream);
        _ = ReadSingle(stream);
        _ = ReadUInt32(stream);
        int replayStart = ReadInt32(stream);
        int replayEnd = ReadInt32(stream);
        uint blendTime = ReadUInt32(stream);

        stream.Position = entryStart + SeqsCountedNamedRecordSizeBytes;
        return new MdxSequenceSummary(index, name, startTime, endTime, moveSpeed, flags, 1.0f, replayStart, replayEnd, blendTime, boundsMin, boundsMax, null);
    }

    private static MdxSequenceSummary ParseSeq090Record(Stream stream, int index)
    {
        long entryStart = stream.Position;

        uint animId = ReadUInt32(stream);
        _ = ReadUInt32(stream);
        stream.Position = checked(entryStart + 0x10);
        stream.Position += 0x40;

        int startTime = ReadInt32(stream);
        int endTime = ReadInt32(stream);
        float moveSpeed = ReadSingle(stream);
        uint flags = ReadUInt32(stream);
        Vector3 boundsMin = ReadVector3(stream);
        Vector3 boundsMax = ReadVector3(stream);
        _ = ReadSingle(stream);
        _ = ReadUInt32(stream);
        float frequency = ReadUInt32(stream);
        _ = ReadUInt32(stream);
        _ = ReadUInt32(stream);

        stream.Position = entryStart + SeqsCountedNamedRecordSizeBytes;
        return new MdxSequenceSummary(index, $"Seq_{animId}", startTime, endTime, moveSpeed, flags, frequency, 0, 0, null, boundsMin, boundsMax, null);
    }

    private static (int EntrySize, int PathSize) ResolveTexsLayout(uint size)
    {
        if (size % TexsEntrySizeExtended == 0)
            return (TexsEntrySizeExtended, TexsPathSizeExtended);

        if (size % TexsEntrySizeLegacy == 0)
            return (TexsEntrySizeLegacy, TexsPathSizeLegacy);

        throw new InvalidDataException($"Invalid TEXS size 0x{size:X}: expected divisibility by 0x{TexsEntrySizeLegacy:X} or 0x{TexsEntrySizeExtended:X}.");
    }

    private static List<MdxMaterialSummary> ReadMtlsSummary(Stream stream, long dataOffset, uint size)
    {
        long previousPosition = stream.Position;
        try
        {
            stream.Position = dataOffset;
            if (size < 8)
                return [];

            uint materialCount = ReadUInt32(stream);
            _ = ReadUInt32(stream);

            List<MdxMaterialSummary> materials = new(checked((int)materialCount));
            for (int materialIndex = 0; materialIndex < materialCount; materialIndex++)
            {
                long materialSizeOffset = stream.Position;
                uint materialSize = ReadUInt32(stream);
                long materialEnd = checked(materialSizeOffset + materialSize);
                if (materialEnd > dataOffset + size)
                    throw new InvalidDataException($"MTLS material {materialIndex} overruns the MTLS payload.");

                int priorityPlane = ReadInt32(stream);
                uint layerCount = ReadUInt32(stream);
                List<MdxMaterialLayerSummary> layers = new(checked((int)layerCount));

                for (int layerIndex = 0; layerIndex < layerCount; layerIndex++)
                {
                    long layerSizeOffset = stream.Position;
                    uint layerSize = ReadUInt32(stream);
                    long layerEnd = checked(layerSizeOffset + layerSize);
                    if (layerEnd > materialEnd)
                        throw new InvalidDataException($"MTLS layer {layerIndex} in material {materialIndex} overruns the material payload.");

                    uint blendMode = ReadUInt32(stream);
                    uint flags = ReadUInt32(stream);
                    int textureId = ReadInt32(stream);
                    int transformId = ReadInt32(stream);
                    int coordId = ReadInt32(stream);
                    float staticAlpha = ReadSingle(stream);

                    layers.Add(new MdxMaterialLayerSummary(layerIndex, blendMode, flags, textureId, transformId, coordId, staticAlpha));
                    stream.Position = layerEnd;
                }

                materials.Add(new MdxMaterialSummary(materialIndex, priorityPlane, layers));
                stream.Position = materialEnd;
            }

            return materials;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static MdxGeosetAnimationTrackSummary ReadGeosetAnimationAlphaTrackSummary(Stream stream, long limit)
    {
        uint keyCount = ReadUInt32(stream);
        if (keyCount > 100000)
            throw new InvalidDataException($"GEOA(v1300): invalid KGAO key count {keyCount}.");

        uint interpolationType = ReadUInt32(stream);
        int globalSequenceId = ReadInt32(stream);
        int? firstKeyTime = null;
        int? lastKeyTime = null;

        for (uint keyIndex = 0; keyIndex < keyCount; keyIndex++)
        {
            int keyTime = ReadInt32(stream);
            firstKeyTime ??= keyTime;
            lastKeyTime = keyTime;

            _ = ReadSingle(stream);
            if (TrackUsesTangents(interpolationType))
            {
                _ = ReadSingle(stream);
                _ = ReadSingle(stream);
            }
        }

        if (stream.Position > limit)
            throw new InvalidDataException("GEOA(v1300): KGAO payload overran its subchunk.");

        return new MdxGeosetAnimationTrackSummary("KGAO", checked((int)keyCount), interpolationType, globalSequenceId, firstKeyTime, lastKeyTime);
    }

    private static MdxGeosetAnimationTrackSummary ReadGeosetAnimationColorTrackSummary(Stream stream, long limit)
    {
        uint keyCount = ReadUInt32(stream);
        if (keyCount > 100000)
            throw new InvalidDataException($"GEOA(v1300): invalid KGAC key count {keyCount}.");

        uint interpolationType = ReadUInt32(stream);
        int globalSequenceId = ReadInt32(stream);
        int? firstKeyTime = null;
        int? lastKeyTime = null;

        for (uint keyIndex = 0; keyIndex < keyCount; keyIndex++)
        {
            int keyTime = ReadInt32(stream);
            firstKeyTime ??= keyTime;
            lastKeyTime = keyTime;

            _ = ReadVector3(stream);
            if (TrackUsesTangents(interpolationType))
            {
                _ = ReadVector3(stream);
                _ = ReadVector3(stream);
            }
        }

        if (stream.Position > limit)
            throw new InvalidDataException("GEOA(v1300): KGAC payload overran its subchunk.");

        return new MdxGeosetAnimationTrackSummary("KGAC", checked((int)keyCount), interpolationType, globalSequenceId, firstKeyTime, lastKeyTime);
    }

    private static MdxEventTrackSummary ReadEventTrackSummary(Stream stream, long limit, string tag, string overrunMessage)
    {
        uint keyCount = ReadUInt32(stream);
        if (keyCount > 100000)
            throw new InvalidDataException($"EVTS(v1300): invalid {tag} key count {keyCount}.");

        int globalSequenceId = ReadInt32(stream);
        int? firstKeyTime = null;
        int? lastKeyTime = null;

        for (uint keyIndex = 0; keyIndex < keyCount; keyIndex++)
        {
            int keyTime = ReadInt32(stream);
            firstKeyTime ??= keyTime;
            lastKeyTime = keyTime;
        }

        if (stream.Position > limit)
            throw new InvalidDataException(overrunMessage);

        return new MdxEventTrackSummary(tag, checked((int)keyCount), globalSequenceId, firstKeyTime, lastKeyTime);
    }

    private static MdxGeometryShapeType ReadGeometryShapeType(Stream stream, string invalidMessage)
    {
        byte value = ReadByte(stream);
        if (value > (byte)MdxGeometryShapeType.Plane)
            throw new InvalidDataException(invalidMessage);

        return (MdxGeometryShapeType)value;
    }

    private static MdxNodeTrackSummary ReadNodeVectorTrackSummary(Stream stream, long limit, string tag, string overrunMessage)
    {
        return ReadNodeTrackSummary(stream, limit, tag, sizeof(float) * 3, overrunMessage);
    }

    private static MdxNodeTrackSummary ReadNodeQuaternionTrackSummary(Stream stream, long limit, string tag, string overrunMessage)
    {
        return ReadNodeTrackSummary(stream, limit, tag, sizeof(uint) * 2, overrunMessage);
    }

    private static MdxNodeTrackSummary ReadNodeTrackSummary(Stream stream, long limit, string tag, int valueSizeBytes, string overrunMessage)
    {
        uint keyCount = ReadUInt32(stream);
        if (keyCount > 100000)
            throw new InvalidDataException($"MDLGENOBJECT(v1300): invalid {tag} key count {keyCount}.");

        uint interpolationType = ReadUInt32(stream);
        int globalSequenceId = ReadInt32(stream);
        int? firstKeyTime = null;
        int? lastKeyTime = null;

        for (uint keyIndex = 0; keyIndex < keyCount; keyIndex++)
        {
            int keyTime = ReadInt32(stream);
            firstKeyTime ??= keyTime;
            lastKeyTime = keyTime;

            SkipBytes(stream, valueSizeBytes, limit, overrunMessage);
            if (TrackUsesTangents(interpolationType))
            {
                SkipBytes(stream, valueSizeBytes, limit, overrunMessage);
                SkipBytes(stream, valueSizeBytes, limit, overrunMessage);
            }
        }

        if (stream.Position > limit)
            throw new InvalidDataException(overrunMessage);

        return new MdxNodeTrackSummary(tag, checked((int)keyCount), interpolationType, globalSequenceId, firstKeyTime, lastKeyTime);
    }

    private static MdxVisibilityTrackSummary ReadVisibilityTrackSummary(Stream stream, long limit, string tag, string overrunMessage)
    {
        uint keyCount = ReadUInt32(stream);
        if (keyCount > 100000)
            throw new InvalidDataException($"MDLVISIBILITY(v1300): invalid {tag} key count {keyCount}.");

        uint interpolationType = ReadUInt32(stream);
        int globalSequenceId = ReadInt32(stream);
        int? firstKeyTime = null;
        int? lastKeyTime = null;

        for (uint keyIndex = 0; keyIndex < keyCount; keyIndex++)
        {
            int keyTime = ReadInt32(stream);
            firstKeyTime ??= keyTime;
            lastKeyTime = keyTime;

            SkipBytes(stream, sizeof(float), limit, overrunMessage);
            if (TrackUsesTangents(interpolationType))
            {
                SkipBytes(stream, sizeof(float), limit, overrunMessage);
                SkipBytes(stream, sizeof(float), limit, overrunMessage);
            }
        }

        if (stream.Position > limit)
            throw new InvalidDataException(overrunMessage);

        return new MdxVisibilityTrackSummary(tag, checked((int)keyCount), interpolationType, globalSequenceId, firstKeyTime, lastKeyTime);
    }

    private static MdxTrackSummary ReadTrackSummary(Stream stream, long limit, string tag, int valueSizeBytes, string overrunMessage)
    {
        uint keyCount = ReadUInt32(stream);
        if (keyCount > 100000)
            throw new InvalidDataException($"MDLANIMATION(v1300): invalid {tag} key count {keyCount}.");

        uint interpolationType = ReadUInt32(stream);
        int globalSequenceId = ReadInt32(stream);
        int? firstKeyTime = null;
        int? lastKeyTime = null;

        for (uint keyIndex = 0; keyIndex < keyCount; keyIndex++)
        {
            int keyTime = ReadInt32(stream);
            firstKeyTime ??= keyTime;
            lastKeyTime = keyTime;

            SkipBytes(stream, valueSizeBytes, limit, overrunMessage);
            if (TrackUsesTangents(interpolationType))
            {
                SkipBytes(stream, valueSizeBytes, limit, overrunMessage);
                SkipBytes(stream, valueSizeBytes, limit, overrunMessage);
            }
        }

        if (stream.Position > limit)
            throw new InvalidDataException(overrunMessage);

        return new MdxTrackSummary(tag, checked((int)keyCount), interpolationType, globalSequenceId, firstKeyTime, lastKeyTime);
    }

    private static uint ReadUInt32(Stream stream)
    {
        Span<byte> bytes = stackalloc byte[sizeof(uint)];
        stream.ReadExactly(bytes);
        return BinaryPrimitives.ReadUInt32LittleEndian(bytes);
    }

    private static ushort ReadUInt16(Stream stream)
    {
        Span<byte> bytes = stackalloc byte[sizeof(ushort)];
        stream.ReadExactly(bytes);
        return BinaryPrimitives.ReadUInt16LittleEndian(bytes);
    }

    private static byte ReadByte(Stream stream)
    {
        int value = stream.ReadByte();
        if (value < 0)
            throw new EndOfStreamException();

        return (byte)value;
    }

    private static int ReadInt32(Stream stream)
    {
        Span<byte> bytes = stackalloc byte[sizeof(int)];
        stream.ReadExactly(bytes);
        return BinaryPrimitives.ReadInt32LittleEndian(bytes);
    }

    private static float ReadSingle(Stream stream)
    {
        Span<byte> bytes = stackalloc byte[sizeof(float)];
        stream.ReadExactly(bytes);
        return BinaryPrimitives.ReadSingleLittleEndian(bytes);
    }

    private static string ReadTag(Stream stream)
    {
        Span<byte> bytes = stackalloc byte[4];
        stream.ReadExactly(bytes);
        return Encoding.ASCII.GetString(bytes);
    }

    private static byte[] ReadBytes(Stream stream, int byteCount)
    {
        byte[] bytes = new byte[byteCount];
        stream.ReadExactly(bytes);
        return bytes;
    }

    private static void ExpectTag(Stream stream, string expected, string message)
    {
        string actual = ReadTag(stream);
        if (!string.Equals(actual, expected, StringComparison.Ordinal))
            throw new InvalidDataException($"{message} Found '{actual}'.");
    }

    private static bool TryReadTag(Stream stream, string expected)
    {
        long previousPosition = stream.Position;
        string actual = ReadTag(stream);
        if (string.Equals(actual, expected, StringComparison.Ordinal))
            return true;

        stream.Position = previousPosition;
        return false;
    }

    private static int ReadNonNegativeCount(Stream stream, string errorMessage)
    {
        int count = ReadInt32(stream);
        if (count < 0)
            throw new InvalidDataException(errorMessage);

        return count;
    }

    private static bool TrackUsesTangents(uint interpolationType)
    {
        return interpolationType >= 2u;
    }

    private static void SkipBytes(Stream stream, long byteCount, long limit, string errorMessage)
    {
        if (byteCount < 0)
            throw new InvalidDataException(errorMessage);

        long nextPosition = checked(stream.Position + byteCount);
        if (nextPosition > limit)
            throw new InvalidDataException(errorMessage);

        stream.Position = nextPosition;
    }

    private static Vector3 ReadVector3(Stream stream)
    {
        return new Vector3(ReadSingle(stream), ReadSingle(stream), ReadSingle(stream));
    }

    private static void ReadModlSummary(Stream stream, long dataOffset, uint size, out string? modelName, out uint? blendTime, out Vector3? boundsMin, out Vector3? boundsMax)
    {
        long previousPosition = stream.Position;
        try
        {
            stream.Position = dataOffset;
            int nameBytesToRead = checked((int)Math.Min(ModlNameSizeBytes, size));
            byte[] nameBytes = new byte[nameBytesToRead];
            if (nameBytesToRead > 0)
                stream.ReadExactly(nameBytes);

            string rawName = ReadNullTerminatedAscii(nameBytes);
            modelName = string.IsNullOrWhiteSpace(rawName) ? null : rawName;

            blendTime = null;
            boundsMin = null;
            boundsMax = null;
            if (size < ModlSummarySizeBytes)
                return;

            Span<byte> boundsAndBlendBytes = stackalloc byte[ModlBoundsAndBlendSizeBytes];
            stream.ReadExactly(boundsAndBlendBytes);
            boundsMin = new Vector3(
                BinaryPrimitives.ReadSingleLittleEndian(boundsAndBlendBytes[0x00..0x04]),
                BinaryPrimitives.ReadSingleLittleEndian(boundsAndBlendBytes[0x04..0x08]),
                BinaryPrimitives.ReadSingleLittleEndian(boundsAndBlendBytes[0x08..0x0C]));
            boundsMax = new Vector3(
                BinaryPrimitives.ReadSingleLittleEndian(boundsAndBlendBytes[0x0C..0x10]),
                BinaryPrimitives.ReadSingleLittleEndian(boundsAndBlendBytes[0x10..0x14]),
                BinaryPrimitives.ReadSingleLittleEndian(boundsAndBlendBytes[0x14..0x18]));
            blendTime = BinaryPrimitives.ReadUInt32LittleEndian(boundsAndBlendBytes[0x18..0x1C]);
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static string ReadNullTerminatedAscii(byte[] bytes)
    {
        int length = Array.IndexOf(bytes, (byte)0);
        if (length < 0)
            length = bytes.Length;

        return Encoding.ASCII.GetString(bytes, 0, length);
    }

    private static string ReadFixedAscii(Stream stream, int size)
    {
        byte[] bytes = new byte[size];
        stream.ReadExactly(bytes);
        return ReadNullTerminatedAscii(bytes);
    }
}