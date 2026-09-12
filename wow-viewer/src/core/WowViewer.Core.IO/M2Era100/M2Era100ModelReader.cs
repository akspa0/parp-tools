using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.M2;

namespace WowViewer.Core.IO.M2Era100;

/// <summary>
/// Reader for WoW 1.0.0 (build 3980, beta-3) M2/MDX models — version 0x100 with the
/// "classic" header layout (M2Vertex records + M2Division embedded skin profiles).
///
/// All header offsets and element sizes are from the Ghidra static trace of
/// FUN_0071e190 (MD20 parser/relocator):
/// specs/104-legacy-m2-rendering/research-1.0.0-ghidra-trace.md §4.
///
/// CRITICAL: 1.0.0 and 1.12.1 both use version 0x100 but have completely different
/// header layouts. This reader handles 1.0.0; the 1.12.1 reader (M2Era1121ModelReader)
/// handles 1.12.1. The dispatcher distinguishes them via layout validation.
/// </summary>
public static class M2Era100ModelReader
{
    public static M2ModelDocument Read(Stream stream, string sourcePath)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        if (!stream.CanSeek)
            throw new ArgumentException("1.0.0 M2 model reading requires a seekable stream.", nameof(stream));

        byte[] data = ReadAllBytes(stream);
        if (data.Length < M2Era100Constants.DispatchHeaderSizeBytes)
            throw new InvalidDataException($"1.0.0 M2 file '{sourcePath}' is too small to contain a magic+version pair.");

        uint magic = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(0, sizeof(uint)));
        if (magic != M2Era100Constants.Md20Magic)
            throw new InvalidDataException($"1.0.0 M2 file '{sourcePath}' does not contain a strict MD20 root.");

        if (data.Length < M2Era100Constants.MinimumHeaderSizeBytes)
            throw new InvalidDataException($"1.0.0 M2 file '{sourcePath}' is too small to contain a 1.0.0 MD20 header (≥ 0x144 bytes).");

        uint rawVersion = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(M2Era100Constants.VersionOffset, sizeof(uint)));
        if (rawVersion < 0x100u || rawVersion > 0x107u)
            throw new NotSupportedException($"Legacy M2 file '{sourcePath}' has version 0x{rawVersion:X}. Expected 0x100-0x107.");

        return ParseM2(data, sourcePath, rawVersion);
    }

    private static M2ModelDocument ParseM2(byte[] data, string sourcePath, uint rawVersion)
    {
        uint flags = ReadUInt32At(data, M2Era100Constants.FlagsOffset);
        uint viewCount = ReadUInt32At(data, M2Era100Constants.DivisionCountOffset); // divisions == views
        string? modelName = TryReadName(data, sourcePath);

        List<uint> globalLoops = ReadUInt32Table(data, sourcePath, "globalLoops",
            M2Era100Constants.GlobalLoopCountOffset, M2Era100Constants.GlobalLoopOffsetOffset);

        List<M2SequenceDefinition> sequences = ReadSequences(data, sourcePath);
        List<short> sequenceLookup = ReadInt16Table(data, sourcePath, "sequenceLookup",
            M2Era100Constants.SequenceLookupCountOffset, M2Era100Constants.SequenceLookupOffsetOffset);

        // Bounds (at 0xB4, matching the gap in the 1.0.0 header).
        Vector3 boundsMin = ReadLenientVector3At(data, M2Era100Constants.BoundsOffset, sourcePath, "boundsMin");
        Vector3 boundsMax = ReadLenientVector3At(data, M2Era100Constants.BoundsOffset + 0x0C, sourcePath, "boundsMax");
        float boundsRadius = ReadLenientSingleAt(data, M2Era100Constants.BoundsRadiusOffset, sourcePath, "boundsRadius");

        // --- Geometry & Embedded Skins ---
        (M2Era100Geometry? geometry, List<M2SkinDocument> embeddedSkins) = ReadGeometry(data, sourcePath);

        // --- Textures ---

        List<M2Era100Texture> textures = ReadTextures(data, sourcePath);
        List<short> textureLookup = ReadInt16Lookup(data, sourcePath, "textureLookup",
            M2Era100Constants.TextureLookupCountOffset, M2Era100Constants.TextureLookupOffsetOffset);
        List<M2Era100Material> materials = ReadMaterials(data, sourcePath);

        if (geometry != null)
        {
            geometry = new M2Era100Geometry(
                geometry.RenderVertices,
                geometry.Triangles,
                geometry.Sections,
                geometry.Batches,
                textures,
                textureLookup.Count > 0 ? textureLookup : geometry.TextureLookup,
                materials,
                geometry.GlobalVertices);
        }

        // --- Animation & Tracks ---
        M2Era100PayloadAppender appender = new(data);
        List<M2BoneDefinition> bones = ReadBones(data, rawVersion, sequences.Count, globalLoops.Count, appender, sourcePath);
        List<M2CameraDefinition> cameras = ReadCameras(data, sequences.Count, globalLoops.Count, appender, sourcePath);

        M2ModelIdentity identity = M2ModelIdentity.FromPath(sourcePath);

        M2ModelDocument document = new(
            identity,
            appender.ToPayload(),
            "MD20",
            rawVersion,
            flags,
            viewCount,
            modelName,
            globalLoops,
            sequences,
            sequenceLookup,
            colors: [],
            textureWeights: [],
            textureTransforms: [],
            lights: [],
            cameras: cameras,
            boundsMin,
            boundsMax,
            boundsRadius,
            embeddedSkinProfileCount: viewCount,
            embeddedSkinProfileOffset: ReadUInt32At(data, M2Era100Constants.DivisionOffsetOffset),
            bones: bones.Count > 0 ? bones : null,
            ribbons: null,
            particles: null);

        document.EmbeddedSkinDocuments = embeddedSkins;
        if (geometry != null)
            document.InlineEra100Geometry = geometry;

        return document;
    }

    private static List<M2CameraDefinition> ReadCameras(byte[] data, int sequenceCount, int globalLoopCount, M2Era100PayloadAppender appender, string sourcePath)
    {
        uint count = ReadUInt32At(data, M2Era100Constants.CameraCountOffset);
        uint offset = ReadUInt32At(data, M2Era100Constants.CameraOffsetOffset);
        if (count == 0)
            return [];

        ValidateSpan(count, offset, M2Era100Constants.CameraStride, data.Length, sourcePath, "cameras");

        // Validate the native id -> record lookup even though the shared document exposes
        // cameras by record index. A broken lookup must not be mistaken for a cameraIndex
        // problem later in the importer.
        _ = ReadInt16Table(data, sourcePath, "cameraLookup",
            M2Era100Constants.CameraLookupCountOffset, M2Era100Constants.CameraLookupOffsetOffset);

        List<M2CameraDefinition> cameras = new(checked((int)count));
        for (int index = 0; index < count; index++)
        {
            int entryOffset = checked((int)offset + (index * M2Era100Constants.CameraStride));
            OldTrack positionTrack = ReadOldTrack(data, entryOffset + 0x10, 12, globalLoopCount, sourcePath, $"cameras[{index}].positionTrack");
            OldTrack targetTrack = ReadOldTrack(data, entryOffset + 0x38, 12, globalLoopCount, sourcePath, $"cameras[{index}].targetPositionTrack");
            OldTrack rollTrack = ReadOldTrack(data, entryOffset + 0x60, 4, globalLoopCount, sourcePath, $"cameras[{index}].rollTrack");

            cameras.Add(new M2CameraDefinition(
                index,
                unchecked((int)ReadUInt32At(data, entryOffset + 0x00)),
                ReadLenientSingleAt(data, entryOffset + 0x04, sourcePath, $"cameras[{index}].fieldOfView"),
                ReadLenientSingleAt(data, entryOffset + 0x08, sourcePath, $"cameras[{index}].farClip"),
                ReadLenientSingleAt(data, entryOffset + 0x0C, sourcePath, $"cameras[{index}].nearClip"),
                appender.NormalizeVectorTrack(positionTrack, sequenceCount),
                ReadLenientVector3At(data, entryOffset + 0x2C, sourcePath, $"cameras[{index}].positionBase"),
                appender.NormalizeVectorTrack(targetTrack, sequenceCount),
                ReadLenientVector3At(data, entryOffset + 0x54, sourcePath, $"cameras[{index}].targetPositionBase"),
                appender.NormalizeFloatTrack(rollTrack, sequenceCount)));
        }

        return cameras;
    }

    private static List<M2BoneDefinition> ReadBones(
        byte[] data,
        uint rawVersion,
        int sequenceCount,
        int globalLoopCount,
        M2Era100PayloadAppender appender,
        string sourcePath)
    {
        uint count = ReadUInt32At(data, M2Era100Constants.BoneCountOffset);
        uint offset = ReadUInt32At(data, M2Era100Constants.BoneOffsetOffset);
        if (count == 0 || offset == 0)
            return [];

        // Detect whether bones include boneNameCrc (0x70 / 112 bytes) or legacy (0x6C / 108 bytes).
        // 0x100 uses 108 bytes (track at +0x0C). 0x104–0x107 uses 112 bytes (boneNameCrc at +0x0C, track at +0x10).
        bool hasNameCrc = rawVersion >= 0x104;
        if (count > 0 && checked((int)offset + 0x14) <= data.Length)
        {
            ushort interpAt0C = ReadUInt16At(data, (int)offset + 0x0C);
            ushort interpAt10 = ReadUInt16At(data, (int)offset + 0x10);
            if (interpAt10 <= (ushort)M2TrackInterpolation.Bezier && interpAt0C > (ushort)M2TrackInterpolation.Bezier)
            {
                hasNameCrc = true;
            }
            else if (interpAt0C <= (ushort)M2TrackInterpolation.Bezier && interpAt10 > (ushort)M2TrackInterpolation.Bezier)
            {
                hasNameCrc = false;
            }
        }

        int boneStride = hasNameCrc ? M2Era100Constants.BoneStrideEra104 : M2Era100Constants.BoneStrideEra100;
        ValidateSpan(count, offset, boneStride, data.Length, sourcePath, "bones");

        List<M2BoneDefinition> bones = new(checked((int)count));
        for (int index = 0; index < count; index++)
        {
            int entryOffset = checked((int)offset + (index * boneStride));
            int keyBoneId = BinaryPrimitives.ReadInt32LittleEndian(data.AsSpan(entryOffset + 0x00, sizeof(int)));
            uint flags = ReadUInt32At(data, entryOffset + 0x04);
            short parentBone = ReadInt16At(data, entryOffset + 0x08);
            ushort submeshId = ReadUInt16At(data, entryOffset + 0x0A);
            uint boneNameCrc = hasNameCrc ? ReadUInt32At(data, entryOffset + 0x0C) : 0u;

            int trackOffset = hasNameCrc ? 0x10 : 0x0C;
            OldTrack translationTrack = ReadOldTrack(data, entryOffset + trackOffset, 12, globalLoopCount, sourcePath, $"bones[{index}].translation");
            OldTrack rotationTrack = ReadOldTrack(data, entryOffset + trackOffset + 0x1C, 8, globalLoopCount, sourcePath, $"bones[{index}].rotation");
            OldTrack scalingTrack = ReadOldTrack(data, entryOffset + trackOffset + 0x38, 12, globalLoopCount, sourcePath, $"bones[{index}].scaling");
            Vector3 pivot = ReadLenientVector3At(data, entryOffset + trackOffset + 0x54, sourcePath, $"bones[{index}].pivot");

            bones.Add(new M2BoneDefinition(
                index,
                keyBoneId,
                flags,
                parentBone,
                submeshId,
                boneNameCrc,
                appender.NormalizeVectorTrack(translationTrack, sequenceCount),
                appender.NormalizeQuaternionTrack(rotationTrack, sequenceCount),
                appender.NormalizeVectorTrack(scalingTrack, sequenceCount),
                pivot));
        }

        return bones;
    }

    private static OldTrack ReadOldTrack(byte[] data, int offset, int scalarSize, int globalLoopCount, string sourcePath, string label)
    {
        EnsureReadable(data, offset, M2Era100Constants.TrackStride, sourcePath, label);

        ushort interpolationValue = ReadUInt16At(data, offset + 0x00);
        ushort globalSequenceValue = ReadUInt16At(data, offset + 0x02);
        if (interpolationValue > (ushort)M2TrackInterpolation.Bezier)
            throw new InvalidDataException($"Legacy M2 file '{sourcePath}' has unsupported interpolation {interpolationValue} in '{label}'.");

        M2TrackInterpolation interpolation = (M2TrackInterpolation)interpolationValue;
        int globalSequence = globalSequenceValue == ushort.MaxValue || globalSequenceValue >= globalLoopCount
            ? -1
            : globalSequenceValue;
        uint rangeCount = ReadUInt32At(data, offset + 0x04);
        uint rangeOffset = ReadUInt32At(data, offset + 0x08);
        uint timestampCount = ReadUInt32At(data, offset + 0x0C);
        uint timestampOffset = ReadUInt32At(data, offset + 0x10);
        uint valueCount = ReadUInt32At(data, offset + 0x14);
        uint valueOffset = ReadUInt32At(data, offset + 0x18);

        ValidateSpan(rangeCount, rangeOffset, 0x08, data.Length, sourcePath, $"{label}.ranges");
        ValidateSpan(timestampCount, timestampOffset, sizeof(uint), data.Length, sourcePath, $"{label}.timestamps");
        int valueStride = interpolation is M2TrackInterpolation.Hermite or M2TrackInterpolation.Bezier
            ? checked(scalarSize * 3)
            : scalarSize;
        ValidateSpan(valueCount, valueOffset, valueStride, data.Length, sourcePath, $"{label}.values");

        return new OldTrack(interpolation, globalSequence, rangeCount, rangeOffset, timestampCount, timestampOffset, valueCount, valueOffset);
    }

    private readonly record struct OldTrack(
        M2TrackInterpolation Interpolation,
        int GlobalSequenceIndex,
        uint RangeCount,
        uint RangeOffset,
        uint TimestampCount,
        uint TimestampOffset,
        uint ValueCount,
        uint ValueOffset);

    private sealed class M2Era100PayloadAppender
    {
        private readonly byte[] _source;
        private readonly List<byte> _extension = [];

        public M2Era100PayloadAppender(byte[] source) => _source = source;

        public M2TrackDefinition<Vector3> NormalizeVectorTrack(OldTrack track, int sequenceCount)
            => Normalize<Vector3>(track, sequenceCount, 12, static normalized => new M2TrackDefinition<Vector3>(normalized.Interpolation, normalized.GlobalSequenceIndex, normalized.TimestampReferences, normalized.ValueReferences));

        public M2TrackDefinition<M2CompQuaternion> NormalizeQuaternionTrack(OldTrack track, int sequenceCount)
            => Normalize<M2CompQuaternion>(track, sequenceCount, 8, static normalized => new M2TrackDefinition<M2CompQuaternion>(normalized.Interpolation, normalized.GlobalSequenceIndex, normalized.TimestampReferences, normalized.ValueReferences));

        public M2TrackDefinition<float> NormalizeFloatTrack(OldTrack track, int sequenceCount)
            => Normalize<float>(track, sequenceCount, 4, static normalized => new M2TrackDefinition<float>(normalized.Interpolation, normalized.GlobalSequenceIndex, normalized.TimestampReferences, normalized.ValueReferences));

        private M2TrackDefinition<T> Normalize<T>(OldTrack track, int sequenceCount, int scalarSize, Func<NormalizedTrack<T>, M2TrackDefinition<T>> create)
        {
            int referenceCount = track.GlobalSequenceIndex >= 0 ? 1 : Math.Max(sequenceCount, 1);
            List<M2TrackArrayReference> timestamps = new(referenceCount);
            List<M2TrackArrayReference> values = new(referenceCount);
            uint availableCount = Math.Min(track.TimestampCount, track.ValueCount);

            for (int index = 0; index < referenceCount; index++)
            {
                uint first = 0;
                uint last = availableCount == 0 ? 0 : availableCount - 1;
                if (track.RangeCount > 0 && availableCount > 0)
                {
                    int rangeIndex = track.GlobalSequenceIndex >= 0 ? 0 : Math.Min(index, checked((int)track.RangeCount - 1));
                    int rangeOffset = checked((int)track.RangeOffset + (rangeIndex * 0x08));
                    first = ReadUInt32At(_source, rangeOffset + 0x00);
                    last = ReadUInt32At(_source, rangeOffset + 0x04);
                    if (first > last || last >= availableCount)
                        throw new InvalidDataException($"Legacy M2 track range [{first}, {last}] is outside {availableCount} keys.");
                }

                uint keyCount = availableCount == 0 ? 0 : last - first + 1;
                timestamps.Add(new M2TrackArrayReference(
                    keyCount,
                    keyCount == 0 ? 0 : checked(track.TimestampOffset + (first * sizeof(uint)))));
                int valueStride = track.Interpolation is M2TrackInterpolation.Hermite or M2TrackInterpolation.Bezier
                    ? checked(scalarSize * 3)
                    : scalarSize;
                values.Add(new M2TrackArrayReference(
                    keyCount,
                    keyCount == 0 ? 0 : checked(track.ValueOffset + (first * (uint)valueStride))));
            }

            uint timestampReferencesOffset = AppendReferences(timestamps);
            uint valueReferencesOffset = AppendReferences(values);
            return create(new NormalizedTrack<T>(
                track.Interpolation,
                track.GlobalSequenceIndex,
                new M2TrackArrayReference((uint)referenceCount, timestampReferencesOffset),
                new M2TrackArrayReference((uint)referenceCount, valueReferencesOffset)));
        }

        public byte[] ToPayload()
        {
            byte[] payload = new byte[_source.Length + _extension.Count];
            Buffer.BlockCopy(_source, 0, payload, 0, _source.Length);
            _extension.CopyTo(payload, _source.Length);
            return payload;
        }

        private uint AppendReferences(IReadOnlyList<M2TrackArrayReference> references)
        {
            while ((_extension.Count & 3) != 0)
                _extension.Add(0);

            uint offset = checked((uint)(_source.Length + _extension.Count));
            foreach (M2TrackArrayReference reference in references)
            {
                AppendUInt32(reference.Count);
                AppendUInt32(reference.Offset);
            }

            return offset;
        }

        private void AppendUInt32(uint value)
        {
            Span<byte> bytes = stackalloc byte[sizeof(uint)];
            BinaryPrimitives.WriteUInt32LittleEndian(bytes, value);
            _extension.AddRange(bytes.ToArray());
        }

        private readonly record struct NormalizedTrack<T>(
            M2TrackInterpolation Interpolation,
            int GlobalSequenceIndex,
            M2TrackArrayReference TimestampReferences,
            M2TrackArrayReference ValueReferences);
    }

    // ─── Geometry: M2Vertex + M2Division ─────────────────────────────────────

    private static (M2Era100Geometry? Geometry, List<M2SkinDocument> EmbeddedSkins) ReadGeometry(byte[] data, string sourcePath)
    {
        // Read global M2Vertex[] from header 0x44.
        uint vertexCount = ReadUInt32At(data, M2Era100Constants.VertexCountOffset);
        uint vertexOffset = ReadUInt32At(data, M2Era100Constants.VertexOffsetOffset);
        if (vertexCount == 0 || vertexOffset == 0)
            return (null, []);

        ValidateSpan(vertexCount, vertexOffset, M2Era100Constants.VertexStride, data.Length, sourcePath, "vertices");
        List<M2Era100Vertex> globalVertices = new(checked((int)vertexCount));
        for (int i = 0; i < vertexCount; i++)
        {
            int ofs = checked((int)vertexOffset + (i * M2Era100Constants.VertexStride));
            globalVertices.Add(ReadM2Vertex(data, ofs, sourcePath, i));
        }

        // Read divisions from header 0x4C.
        uint divisionCount = ReadUInt32At(data, M2Era100Constants.DivisionCountOffset);
        uint divisionOffset = ReadUInt32At(data, M2Era100Constants.DivisionOffsetOffset);
        if (divisionCount == 0 || divisionOffset == 0)
            return (null, []);

        ValidateSpan(divisionCount, divisionOffset, M2Era100Constants.DivisionStride, data.Length, sourcePath, "divisions");

        List<M2SkinDocument> embeddedSkins = new(checked((int)divisionCount));
        M2Era100Geometry? primaryGeometry = null;

        for (int divIndex = 0; divIndex < divisionCount; divIndex++)
        {
            int divBase = checked((int)divisionOffset + (divIndex * M2Era100Constants.DivisionStride));
            uint vtxLookupCount = ReadUInt32At(data, divBase + M2Era100Constants.DivisionVertexLookupCountOffset);
            uint vtxLookupOfs = ReadUInt32At(data, divBase + M2Era100Constants.DivisionVertexLookupOffsetOffset);
            uint indicesCount = ReadUInt32At(data, divBase + M2Era100Constants.DivisionIndicesCountOffset);
            uint indicesOfs = ReadUInt32At(data, divBase + M2Era100Constants.DivisionIndicesOffsetOffset);
            uint boneEntriesCount = ReadUInt32At(data, divBase + M2Era100Constants.DivisionUint32ArrayCountOffset);
            uint boneEntriesOfs = ReadUInt32At(data, divBase + M2Era100Constants.DivisionUint32ArrayOffsetOffset);
            uint sectionsCount = ReadUInt32At(data, divBase + M2Era100Constants.DivisionSectionsCountOffset);
            uint sectionsOfs = ReadUInt32At(data, divBase + M2Era100Constants.DivisionSectionsOffsetOffset);
            uint batchesCount = ReadUInt32At(data, divBase + M2Era100Constants.DivisionBatchesCountOffset);
            uint batchesOfs = ReadUInt32At(data, divBase + M2Era100Constants.DivisionBatchesOffsetOffset);

            // Read vertexLookup (uint16[] — local → global vertex index).
            List<ushort> vertexLookup = new();
            if (vtxLookupCount > 0 && vtxLookupOfs > 0)
            {
                ValidateSpan(vtxLookupCount, vtxLookupOfs, sizeof(ushort), data.Length, sourcePath, $"division[{divIndex}].vertexLookup");
                for (int i = 0; i < vtxLookupCount; i++)
                {
                    int ofs = checked((int)vtxLookupOfs + (i * sizeof(ushort)));
                    vertexLookup.Add(ReadUInt16At(data, ofs));
                }
            }

            // Read triangle indices (uint16[]).
            List<ushort> triangles = new();
            if (indicesCount > 0 && indicesOfs > 0)
            {
                ValidateSpan(indicesCount, indicesOfs, sizeof(ushort), data.Length, sourcePath, $"division[{divIndex}].indices");
                for (int i = 0; i < indicesCount; i++)
                {
                    int ofs = checked((int)indicesOfs + (i * sizeof(ushort)));
                    triangles.Add(ReadUInt16At(data, ofs));
                }
            }

            // Read bone entries (uint32 / 4-byte records).
            List<M2SkinBoneEntry> boneEntries = new();
            if (boneEntriesCount > 0 && boneEntriesOfs > 0)
            {
                ValidateSpan(boneEntriesCount, boneEntriesOfs, 4, data.Length, sourcePath, $"division[{divIndex}].boneEntries");
                for (int i = 0; i < boneEntriesCount; i++)
                {
                    int ofs = checked((int)boneEntriesOfs + (i * 4));
                    boneEntries.Add(new M2SkinBoneEntry(data[ofs], data[ofs + 1], data[ofs + 2], data[ofs + 3]));
                }
            }

            // Read sections (0x20 B each).
            List<M2Era100Section> eraSections = new();
            List<M2SkinSubmesh> submeshes = new();
            if (sectionsCount > 0 && sectionsOfs > 0)
            {
                ValidateSpan(sectionsCount, sectionsOfs, M2Era100Constants.SectionStride, data.Length, sourcePath, $"division[{divIndex}].sections");
                for (int i = 0; i < sectionsCount; i++)
                {
                    int ofs = checked((int)sectionsOfs + (i * M2Era100Constants.SectionStride));
                    ushort submeshId = ReadUInt16At(data, ofs + M2Era100Constants.SectionSubmeshIdOffset);
                    ushort level = ReadUInt16At(data, ofs + M2Era100Constants.SectionLevelOffset);
                    ushort vStart = ReadUInt16At(data, ofs + M2Era100Constants.SectionVertexStartOffset);
                    ushort vCount = ReadUInt16At(data, ofs + M2Era100Constants.SectionVertexCountOffset);
                    ushort iStart = ReadUInt16At(data, ofs + M2Era100Constants.SectionIndexStartOffset);
                    ushort iCount = ReadUInt16At(data, ofs + M2Era100Constants.SectionIndexCountOffset);
                    ushort bCount = ReadUInt16At(data, ofs + M2Era100Constants.SectionBoneCountOffset);
                    ushort bCombo = ReadUInt16At(data, ofs + M2Era100Constants.SectionBoneComboIndexOffset);
                    ushort bInfl = ReadUInt16At(data, ofs + M2Era100Constants.SectionBoneInfluencesOffset);
                    ushort centerBone = ReadUInt16At(data, ofs + M2Era100Constants.SectionCenterBoneIndexOffset);

                    uint levelHighBits = (uint)level << 16;
                    eraSections.Add(new M2Era100Section(
                        submeshId,
                        level,
                        vStart | levelHighBits,
                        vCount,
                        iStart | levelHighBits,
                        iCount));

                    submeshes.Add(new M2SkinSubmesh(
                        submeshId,
                        level,
                        vStart,
                        vCount,
                        iStart,
                        iCount,
                        bCount,
                        bCombo,
                        bInfl,
                        centerBone));
                }
            }

            // Read batches (0x18 B each).
            List<M2Era100Batch> eraBatches = new();
            List<M2SkinBatch> skinBatches = new();
            if (batchesCount > 0 && batchesOfs > 0)
            {
                ValidateSpan(batchesCount, batchesOfs, M2Era100Constants.BatchStride, data.Length, sourcePath, $"division[{divIndex}].batches");
                for (int i = 0; i < batchesCount; i++)
                {
                    int ofs = checked((int)batchesOfs + (i * M2Era100Constants.BatchStride));
                    eraBatches.Add(ReadM2Batch(data, ofs));

                    byte bFlags = ReadByteAt(data, ofs + M2Era100Constants.BatchFlagsOffset);
                    byte priorityPlane = ReadByteAt(data, ofs + M2Era100Constants.BatchPriorityPlaneOffset);
                    ushort shaderId = ReadUInt16At(data, ofs + M2Era100Constants.BatchShaderIdOffset);
                    ushort skinSectionIndex = ReadUInt16At(data, ofs + M2Era100Constants.BatchSkinSectionIndexOffset);
                    ushort geosetIndex = ReadUInt16At(data, ofs + M2Era100Constants.BatchGeosetIndexOffset);
                    short colorIndex = (short)ReadUInt16At(data, ofs + M2Era100Constants.BatchColorIndexOffset);
                    ushort renderFlagsIndex = ReadUInt16At(data, ofs + M2Era100Constants.BatchMaterialIndexOffset);
                    ushort materialLayer = ReadUInt16At(data, ofs + M2Era100Constants.BatchMaterialLayerOffset);
                    ushort textureCount = ReadUInt16At(data, ofs + M2Era100Constants.BatchTextureCountOffset);
                    ushort textureComboIndex = ReadUInt16At(data, ofs + M2Era100Constants.BatchTextureComboIndexOffset);
                    ushort textureCoordComboIndex = ReadUInt16At(data, ofs + M2Era100Constants.BatchTextureCoordComboIndexOffset);
                    ushort transparencyComboIndex = ReadUInt16At(data, ofs + M2Era100Constants.BatchTextureWeightComboIndexOffset);
                    ushort textureTransformComboIndex = ReadUInt16At(data, ofs + M2Era100Constants.BatchTextureTransformComboIndexOffset);

                    skinBatches.Add(new M2SkinBatch(
                        bFlags,
                        priorityPlane,
                        shaderId,
                        skinSectionIndex,
                        geosetIndex,
                        colorIndex,
                        renderFlagsIndex,
                        materialLayer,
                        textureCount,
                        textureComboIndex,
                        textureCoordComboIndex,
                        transparencyComboIndex,
                        textureTransformComboIndex));
                }
            }

            string skinSourcePath = divisionCount > 1
                ? $"{sourcePath}#{divIndex:D2}"
                : $"{sourcePath}#00";

            M2SkinDocument skinDoc = new(
                skinSourcePath,
                "SKIN",
                vertexLookup,
                vtxLookupOfs,
                triangles,
                indicesOfs,
                boneEntries,
                boneEntriesOfs,
                submeshes,
                sectionsOfs,
                skinBatches,
                batchesOfs,
                globalVertexOffset: 0,
                shadowBatchCount: 0,
                shadowBatchOffset: 0);

            embeddedSkins.Add(skinDoc);

            if (divIndex == 0)
            {
                // Resolve render vertices: walk vertexLookup → global M2Vertex.
                List<M2Era100Vertex> renderVertices = new(vertexLookup.Count);
                for (int i = 0; i < vertexLookup.Count; i++)
                {
                    ushort globalIndex = vertexLookup[i];
                    M2Era100Vertex vertex = globalIndex < globalVertices.Count
                        ? globalVertices[globalIndex]
                        : default;
                    renderVertices.Add(vertex);
                }

                primaryGeometry = new M2Era100Geometry(
                    renderVertices,
                    triangles,
                    eraSections,
                    eraBatches,
                    textures: [],
                    textureLookup: [],
                    materials: null,
                    globalVertices: globalVertices);
            }
        }

        return (primaryGeometry, embeddedSkins);
    }

    private static M2Era100Vertex ReadM2Vertex(byte[] data, int ofs, string sourcePath, int index)
    {
        Vector3 position = ReadLenientVector3At(data, ofs + M2Era100Constants.VertexPositionOffset, sourcePath, $"vertices[{index}].position");
        Vector3 normal = ReadLenientVector3At(data, ofs + M2Era100Constants.VertexNormalOffset, sourcePath, $"vertices[{index}].normal");
        Vector2 uv0 = new(
            ReadLenientSingleAt(data, ofs + M2Era100Constants.VertexTexCoords0Offset + 0x00, sourcePath, $"vertices[{index}].uv0.x"),
            ReadLenientSingleAt(data, ofs + M2Era100Constants.VertexTexCoords0Offset + 0x04, sourcePath, $"vertices[{index}].uv0.y"));
        Vector2 uv1 = new(
            ReadLenientSingleAt(data, ofs + M2Era100Constants.VertexTexCoords1Offset + 0x00, sourcePath, $"vertices[{index}].uv1.x"),
            ReadLenientSingleAt(data, ofs + M2Era100Constants.VertexTexCoords1Offset + 0x04, sourcePath, $"vertices[{index}].uv1.y"));

        // Bone weights/indices: 4 bytes each, packed as uint8[4].
        uint weightsPacked = ReadUInt32At(data, ofs + M2Era100Constants.VertexBoneWeightsOffset);
        uint indicesPacked = ReadUInt32At(data, ofs + M2Era100Constants.VertexBoneIndicesOffset);

        return new M2Era100Vertex(
            position, normal, uv0, uv1,
            (byte)(weightsPacked & 0xFF),
            (byte)((weightsPacked >> 8) & 0xFF),
            (byte)((weightsPacked >> 16) & 0xFF),
            (byte)((weightsPacked >> 24) & 0xFF),
            (byte)(indicesPacked & 0xFF),
            (byte)((indicesPacked >> 8) & 0xFF),
            (byte)((indicesPacked >> 16) & 0xFF),
            (byte)((indicesPacked >> 24) & 0xFF));
    }

    private static M2Era100Batch ReadM2Batch(byte[] data, int ofs)
    {
        return new M2Era100Batch(
            ReadByteAt(data, ofs + M2Era100Constants.BatchFlagsOffset),
            ReadByteAt(data, ofs + M2Era100Constants.BatchPriorityPlaneOffset),
            ReadUInt16At(data, ofs + M2Era100Constants.BatchShaderIdOffset),
            ReadUInt16At(data, ofs + M2Era100Constants.BatchSkinSectionIndexOffset),
            ReadUInt16At(data, ofs + M2Era100Constants.BatchGeosetIndexOffset),
            ReadUInt16At(data, ofs + M2Era100Constants.BatchColorIndexOffset),
            ReadUInt16At(data, ofs + M2Era100Constants.BatchMaterialIndexOffset),
            ReadUInt16At(data, ofs + M2Era100Constants.BatchMaterialLayerOffset),
            ReadUInt16At(data, ofs + M2Era100Constants.BatchTextureCountOffset),
            ReadUInt16At(data, ofs + M2Era100Constants.BatchTextureComboIndexOffset),
            ReadUInt16At(data, ofs + M2Era100Constants.BatchTextureCoordComboIndexOffset),
            ReadUInt16At(data, ofs + M2Era100Constants.BatchTextureWeightComboIndexOffset),
            ReadUInt16At(data, ofs + M2Era100Constants.BatchTextureTransformComboIndexOffset));
    }

    // ─── Textures ────────────────────────────────────────────────────────────

    private static List<M2Era100Texture> ReadTextures(byte[] data, string sourcePath)
    {
        uint count = ReadUInt32At(data, M2Era100Constants.TextureCountOffset);
        uint offset = ReadUInt32At(data, M2Era100Constants.TextureOffsetOffset);
        if (count == 0 || offset == 0)
            return [];

        ValidateSpan(count, offset, M2Era100Constants.TextureStride, data.Length, sourcePath, "textures");
        List<M2Era100Texture> values = new(checked((int)count));
        for (int i = 0; i < count; i++)
        {
            int ofs = checked((int)offset + (i * M2Era100Constants.TextureStride));
            uint type = ReadUInt32At(data, ofs + M2Era100Constants.TextureTypeOffset);
            uint texFlags = ReadUInt32At(data, ofs + M2Era100Constants.TextureFlagsOffset);
            uint nameLen = ReadUInt32At(data, ofs + M2Era100Constants.TextureNameLenOffset);
            uint nameOfs = ReadUInt32At(data, ofs + M2Era100Constants.TextureNameOfsOffset);
            string? filename = TryReadStringAt(data, sourcePath, $"textures[{i}].filename", nameLen, nameOfs);
            values.Add(new M2Era100Texture(type, texFlags, filename ?? string.Empty));
        }

        return values;
    }

    // ─── Sequences ───────────────────────────────────────────────────────────

    private static List<M2SequenceDefinition> ReadSequences(byte[] data, string sourcePath)
    {
        uint count = ReadUInt32At(data, M2Era100Constants.SequenceCountOffset);
        uint offset = ReadUInt32At(data, M2Era100Constants.SequenceOffsetOffset);
        if (count == 0 || offset == 0)
            return [];

        ValidateSpan(count, offset, M2Era100Constants.SequenceStride, data.Length, sourcePath, "sequences");
        List<M2SequenceDefinition> values = new(checked((int)count));
        for (int i = 0; i < count; i++)
        {
            int ofs = checked((int)offset + (i * M2Era100Constants.SequenceStride));
            values.Add(new M2SequenceDefinition(
                i,
                ReadUInt16At(data, ofs + 0x00),
                ReadUInt16At(data, ofs + 0x02),
                ReadUInt32At(data, ofs + 0x04),
                ReadLenientSingleAt(data, ofs + 0x08, sourcePath, $"sequences[{i}].moveSpeed"),
                ReadUInt32At(data, ofs + 0x0C),
                ReadInt16At(data, ofs + 0x10),
                ReadUInt32At(data, ofs + 0x14),
                ReadUInt32At(data, ofs + 0x18),
                ReadUInt16At(data, ofs + 0x1C),
                ReadUInt16At(data, ofs + 0x1E),
                ReadLenientVector3At(data, ofs + 0x20, sourcePath, $"sequences[{i}].boundsMin"),
                ReadLenientVector3At(data, ofs + 0x2C, sourcePath, $"sequences[{i}].boundsMax"),
                ReadLenientSingleAt(data, ofs + 0x38, sourcePath, $"sequences[{i}].boundsRadius"),
                ReadInt16At(data, ofs + 0x3C),
                ReadUInt16At(data, ofs + 0x3E)));
        }

        return values;
    }

    // ─── Primitive readers ───────────────────────────────────────────────────

    private static byte[] ReadAllBytes(Stream stream)
    {
        if (stream is MemoryStream ms && ms.TryGetBuffer(out ArraySegment<byte> segment))
            return segment.ToArray();

        long length = stream.Length;
        if (length > int.MaxValue)
            throw new InvalidDataException("M2 file exceeds 2 GiB.");

        stream.Position = 0;
        byte[] buffer = new byte[length];
        int totalRead = 0;
        while (totalRead < buffer.Length)
        {
            int read = stream.Read(buffer, totalRead, buffer.Length - totalRead);
            if (read == 0)
                break;
            totalRead += read;
        }

        return totalRead == buffer.Length ? buffer : buffer.AsSpan(0, totalRead).ToArray();
    }

    private static uint ReadUInt32At(byte[] data, int offset)
    {
        EnsureReadable(data, offset, sizeof(uint), "m2 data", "uint32");
        return BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(offset, sizeof(uint)));
    }

    private static ushort ReadUInt16At(byte[] data, int offset)
    {
        EnsureReadable(data, offset, sizeof(ushort), "m2 data", "uint16");
        return BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(offset, sizeof(ushort)));
    }

    private static short ReadInt16At(byte[] data, int offset)
    {
        EnsureReadable(data, offset, sizeof(short), "m2 data", "int16");
        return BinaryPrimitives.ReadInt16LittleEndian(data.AsSpan(offset, sizeof(short)));
    }

    private static byte ReadByteAt(byte[] data, int offset)
    {
        EnsureReadable(data, offset, sizeof(byte), "m2 data", "byte");
        return data[offset];
    }

    private static float ReadLenientSingleAt(byte[] data, int offset, string sourcePath, string label)
    {
        EnsureReadable(data, offset, sizeof(float), sourcePath, label);
        float value = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(data.AsSpan(offset, sizeof(float))));
        return float.IsFinite(value) ? value : 0f;
    }

    private static Vector3 ReadLenientVector3At(byte[] data, int offset, string sourcePath, string label)
    {
        return new Vector3(
            ReadLenientSingleAt(data, offset + 0x00, sourcePath, $"{label}.x"),
            ReadLenientSingleAt(data, offset + 0x04, sourcePath, $"{label}.y"),
            ReadLenientSingleAt(data, offset + 0x08, sourcePath, $"{label}.z"));
    }

    private static string? TryReadName(byte[] data, string sourcePath)
    {
        uint nameCount = ReadUInt32At(data, M2Era100Constants.NameCountOffset);
        uint nameOffset = ReadUInt32At(data, M2Era100Constants.NameOffsetOffset);
        if (nameCount == 0 || nameOffset == 0)
            return null;

        ValidateSpan(nameCount, nameOffset, 1, data.Length, sourcePath, "modelName");
        ReadOnlySpan<byte> bytes = data.AsSpan(checked((int)nameOffset), checked((int)nameCount));
        int terminator = bytes.IndexOf((byte)0);
        int length = terminator >= 0 ? terminator : bytes.Length;
        return length == 0 ? null : Encoding.UTF8.GetString(bytes[..length]);
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

    private static List<uint> ReadUInt32Table(byte[] data, string sourcePath, string label, int countOffset, int offsetOffset)
    {
        uint count = ReadUInt32At(data, countOffset);
        uint offset = ReadUInt32At(data, offsetOffset);
        if (count == 0 || offset == 0)
            return [];

        ValidateSpan(count, offset, sizeof(uint), data.Length, sourcePath, label);
        List<uint> values = new(checked((int)count));
        for (int i = 0; i < count; i++)
            values.Add(ReadUInt32At(data, checked((int)offset + (i * sizeof(uint)))));

        return values;
    }

    private static List<short> ReadInt16Table(byte[] data, string sourcePath, string label, int countOffset, int offsetOffset)
    {
        uint count = ReadUInt32At(data, countOffset);
        uint offset = ReadUInt32At(data, offsetOffset);
        if (count == 0 || offset == 0)
            return [];

        ValidateSpan(count, offset, sizeof(short), data.Length, sourcePath, label);
        List<short> values = new(checked((int)count));
        for (int i = 0; i < count; i++)
            values.Add(ReadInt16At(data, checked((int)offset + (i * sizeof(short)))));

        return values;
    }

    private static List<short> ReadInt16Lookup(byte[] data, string sourcePath, string label, int countOffset, int offsetOffset)
    {
        uint count = ReadUInt32At(data, countOffset);
        uint offset = ReadUInt32At(data, offsetOffset);
        if (count == 0 || offset == 0)
            return [];

        ValidateSpan(count, offset, sizeof(short), data.Length, sourcePath, label);
        List<short> values = new(checked((int)count));
        for (int i = 0; i < count; i++)
            values.Add(unchecked((short)ReadUInt16At(data, checked((int)offset + (i * sizeof(short))))));

        return values;
    }

    /// <summary>Reads M2Material[] at header 0x84 — {uint16 flags, uint16 blendMode}, stride 4.</summary>
    private static List<M2Era100Material> ReadMaterials(byte[] data, string sourcePath)
    {
        uint count = ReadUInt32At(data, M2Era100Constants.MaterialCountOffset);
        uint offset = ReadUInt32At(data, M2Era100Constants.MaterialOffsetOffset);
        if (count == 0 || offset == 0)
            return [];

        ValidateSpan(count, offset, M2Era100Constants.MaterialStride, data.Length, sourcePath, "materials");
        List<M2Era100Material> materials = new(checked((int)count));
        for (int i = 0; i < count; i++)
        {
            int ofs = checked((int)offset + (i * M2Era100Constants.MaterialStride));
            materials.Add(new M2Era100Material(ReadUInt16At(data, ofs), ReadUInt16At(data, ofs + 2)));
        }

        return materials;
    }

    // ─── Validation ──────────────────────────────────────────────────────────

    private static void EnsureReadable(byte[] data, int offset, int size, string sourcePath, string label)
    {
        if (offset < 0 || offset > data.Length - size)
            throw new InvalidDataException($"M2 file '{sourcePath}' is truncated at {label} (offset 0x{offset:X}, need {size} bytes, have {data.Length - Math.Max(0, offset)}).");
    }

    private static void ValidateSpan(uint count, uint offset, int stride, long fileSize, string sourcePath, string label)
    {
        if (count == 0)
            return;
        if (offset == 0)
            throw new InvalidDataException($"M2 file '{sourcePath}' has zero offset for non-empty {label}.");

        // Check for overflow: count * stride
        if (count > int.MaxValue / stride)
            throw new InvalidDataException($"M2 file '{sourcePath}' has impossibly large {label} count ({count}).");

        long end = checked((long)offset + (long)count * stride);
        if (end > fileSize)
            throw new InvalidDataException($"M2 file '{sourcePath}' has {label} span (offset 0x{offset:X}, count {count}, stride {stride}) exceeding file size (0x{fileSize:X}).");
    }

    // ─── Layout validation (used by the dispatcher to distinguish 1.0.0 from 1.12.1) ───

    /// <summary>
    /// Validates whether the header bytes are consistent with the 1.0.0 layout.
    /// Checks that the vertices and divisions M2Array fields at the 1.0.0 header
    /// positions produce sane offsets within the file.
    /// </summary>
    public static bool ValidateLayout(ReadOnlySpan<byte> data, string sourcePath)
    {
        if (data.Length < M2Era100Constants.MinimumHeaderSizeBytes)
            return false;

        // Check bones at 0x34 (count) / 0x38 (offset) — M2Bone stride 0x6C.
        if (!TryValidateArray(data, M2Era100Constants.BoneCountOffset, M2Era100Constants.BoneOffsetOffset,
            M2Era100Constants.BoneStride, "bones"))
            return false;

        // Check vertices at 0x44 (count) / 0x48 (offset) — M2Vertex stride 0x30.
        if (!TryValidateArray(data, M2Era100Constants.VertexCountOffset, M2Era100Constants.VertexOffsetOffset,
            M2Era100Constants.VertexStride, "vertices"))
            return false;

        // Check divisions at 0x4C (count) / 0x50 (offset) — M2Division stride 0x2C.
        if (!TryValidateArray(data, M2Era100Constants.DivisionCountOffset, M2Era100Constants.DivisionOffsetOffset,
            M2Era100Constants.DivisionStride, "divisions"))
            return false;

        // Check textures at 0x5C (count) / 0x60 (offset) — M2Texture stride 0x10.
        if (!TryValidateArray(data, M2Era100Constants.TextureCountOffset, M2Era100Constants.TextureOffsetOffset,
            M2Era100Constants.TextureStride, "textures"))
            return false;

        return true;
    }

    private static bool TryValidateArray(ReadOnlySpan<byte> data, int countOffset, int offsetOffset, int stride, string label)
    {
        if (countOffset + sizeof(uint) > data.Length || offsetOffset + sizeof(uint) > data.Length)
            return false;

        uint count = BinaryPrimitives.ReadUInt32LittleEndian(data.Slice(countOffset, sizeof(uint)));
        uint offset = BinaryPrimitives.ReadUInt32LittleEndian(data.Slice(offsetOffset, sizeof(uint)));

        if (count == 0)
            return true; // Empty arrays are valid.
        if (offset == 0 || offset >= (uint)data.Length)
            return false;
        if (count > int.MaxValue / (uint)stride)
            return false;

        long end = checked((long)offset + (long)count * stride);
        return end <= data.Length;
    }
}
