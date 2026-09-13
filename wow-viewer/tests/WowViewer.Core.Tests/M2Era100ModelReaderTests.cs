using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.IO.M2Chunked;
using WowViewer.Core.IO.M2Era100;
using WowViewer.Core.IO.M2Era1121;
using WowViewer.Core.M2;
using WowViewer.Core.Runtime.M2;

namespace WowViewer.Core.Tests;

/// <summary>
/// Era-100 (1.0.0, MD20 version 0x100 classic layout) reader tests built on synthetic
/// fixtures, so they run without a staged client.
/// </summary>
public sealed class M2Era100ModelReaderTests
{
    [Fact]
    public void Era100Reader_ReadsSectionIndexFields_AsUint16_NotUint32()
    {
        // indexCount lives at section+0x0A as a uint16. Reading indexStart as a uint32 at
        // section+0x08 would fold indexCount into its high bits (0 | 6 << 16 = 393216).
        byte[] m2 = CreateSyntheticEra100M2(indexCount: 6, level: 0);

        using MemoryStream stream = new(m2, writable: false);
        M2ModelDocument document = M2Era100ModelReader.Read(stream, "Character\\Synthetic\\Era100.m2");

        M2Era100Geometry geometry = Assert.IsType<M2Era100Geometry>(document.InlineEra100Geometry);
        M2Era100Section section = Assert.Single(geometry.Sections);

        Assert.Equal(0u, section.IndexStart);
        Assert.Equal(6u, section.IndexCount);
        Assert.Equal(0u, section.VertexStart);
        Assert.Equal(4, section.VertexCount);
    }

    [Fact]
    public void Era100Reader_ProducesDrawableGeometry_ForSyntheticSection()
    {
        byte[] m2 = CreateSyntheticEra100M2(indexCount: 6, level: 0);

        using MemoryStream stream = new(m2, writable: false);
        M2ModelDocument document = M2Era100ModelReader.Read(stream, "Character\\Synthetic\\Era100.m2");

        M2Era100Geometry geometry = Assert.IsType<M2Era100Geometry>(document.InlineEra100Geometry);
        Assert.Equal(4, geometry.RenderVertices.Count);
        Assert.Equal(6, geometry.Triangles.Count);

        // The section must address a real, in-bounds slice of the index buffer; this is the
        // bounds check the runtime bridge applies before it will draw the section at all.
        M2Era100Section section = Assert.Single(geometry.Sections);
        Assert.True(section.IndexStart + section.IndexCount <= (uint)geometry.Triangles.Count);
    }

    [Fact]
    public void Era100Reader_ReadsVertexAttributes_WithStandardM2LayoutOffsets()
    {
        byte[] m2 = CreateSyntheticEra100M2(indexCount: 6, level: 0);

        using MemoryStream stream = new(m2, writable: false);
        M2ModelDocument document = M2Era100ModelReader.Read(stream, "Character\\Synthetic\\Era100.m2");

        M2Era100Geometry geometry = Assert.IsType<M2Era100Geometry>(document.InlineEra100Geometry);
        Assert.NotEmpty(geometry.RenderVertices);

        M2Era100Vertex v0 = geometry.RenderVertices[0];
        Assert.Equal(Vector3.Zero, v0.Position);
        Assert.Equal(new Vector3(0f, 0f, 1f), v0.Normal);
        Assert.Equal(new Vector2(0f, 0f), v0.TexCoord0);
        Assert.Equal(255, v0.BoneWeight0);
        Assert.Equal(0, v0.BoneIndex0);

        M2Era100Vertex v1 = geometry.RenderVertices[1];
        Assert.Equal(new Vector3(1f, 1f, 0f), v1.Position);
        Assert.Equal(new Vector3(0f, 0f, 1f), v1.Normal);
        Assert.Equal(new Vector2(0.25f, 0.75f), v1.TexCoord0);
        Assert.Equal(new Vector2(0.1f, 0.2f), v1.TexCoord1);
        Assert.Equal(255, v1.BoneWeight0);
        Assert.Equal(1, v1.BoneIndex0);
    }

    [Fact]
    public void Era100Reader_AppliesLevelHighBits_ToVertexAndIndexStart()
    {
        // Level carries the high 16 bits of vertexStart/indexStart so a division may exceed
        // 65535 entries. level 1 + low 5 must resolve to 0x10005, not 5.
        byte[] m2 = CreateSyntheticEra100M2(indexCount: 6, level: 1);

        using MemoryStream stream = new(m2, writable: false);
        M2ModelDocument document = M2Era100ModelReader.Read(stream, "Character\\Synthetic\\Era100Level.m2");

        M2Era100Geometry geometry = Assert.IsType<M2Era100Geometry>(document.InlineEra100Geometry);
        M2Era100Section section = Assert.Single(geometry.Sections);

        Assert.Equal(1, section.Level);
        Assert.Equal(0x10000u, section.IndexStart);
        Assert.Equal(0x10000u, section.VertexStart);
    }

    [Fact]
    public void Dispatcher_ClassicLayout_V100_RoutesToEra100Reader()
    {
        byte[] m2 = CreateSyntheticEra100M2(indexCount: 6, level: 0);

        using MemoryStream stream = new(m2, writable: false);
        M2DispatchResult result = M2ModelReaderDispatcher.ReadDetailed(stream, "Character\\Synthetic\\Era100.m2");

        Assert.Equal(M2Era1121EraTag.Md20_1X_V100_Era100, result.Era);
        Assert.Equal(0x100u, result.Document.Version);
        Assert.NotNull(result.Document.InlineEra100Geometry);
    }

    [Fact]
    public void Era100Reader_NormalizesOldCameraTracksForSharedImporter()
    {
        byte[] m2 = CreateSyntheticEra100CameraM2();

        using MemoryStream stream = new(m2, writable: false);
        M2DispatchResult result = M2ModelReaderDispatcher.ReadDetailed(stream, "Cameras\\SyntheticEra100.m2");

        M2CameraDefinition camera = Assert.Single(result.Document.Cameras);
        Assert.Equal(0x100u, result.Document.Version);
        Assert.Equal(1, result.Document.SequenceCount);
        Assert.Equal(1, camera.Type);

        M2CameraPathDocument imported = M2CameraPathImporter.Import(result.Document, sampleIntervalMs: 500);

        Assert.Equal(2, imported.Keyframes.Count);
        Assert.Equal(new Vector3(1f, 2f, 3f), imported.Keyframes[0].Position);
        Assert.Equal(new Vector3(14f, 15f, 16f), imported.Keyframes[1].Target, new Vector3EqualityComparer(0.02f));
        Assert.Equal(1f * (180f / MathF.PI), imported.Keyframes[0].FovDegrees, 3);
        Assert.Equal(0.5f * (180f / MathF.PI), imported.Keyframes[1].RollDegrees, 0.05f);
    }

    /// <summary>
    /// Builds a minimal but structurally valid 1.0.0 M2: a 0x144 header, four M2Vertex
    /// records, one division whose vertexLookup/indices/sections/batches describe two
    /// triangles over a quad.
    /// </summary>
    private static byte[] CreateSyntheticEra100M2(ushort indexCount, ushort level)
    {
        const int headerSize = 0x144;
        const int vertexCount = 4;
        ushort[] lookup = [0, 1, 2, 3];
        ushort[] indices = [0, 1, 2, 0, 2, 3];

        int verticesOfs = headerSize;
        int lookupOfs = verticesOfs + (vertexCount * M2Era100Constants.VertexStride);
        int indicesOfs = lookupOfs + (lookup.Length * sizeof(ushort));
        int sectionsOfs = indicesOfs + (indices.Length * sizeof(ushort));
        int batchesOfs = sectionsOfs + M2Era100Constants.SectionStride;
        int divisionOfs = batchesOfs + M2Era100Constants.BatchStride;
        int total = divisionOfs + M2Era100Constants.DivisionStride;

        byte[] data = new byte[total];
        Span<byte> span = data;

        BinaryPrimitives.WriteUInt32LittleEndian(span[..4], M2Era100Constants.Md20Magic);
        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(M2Era100Constants.VersionOffset, 4), 0x100u);

        WriteArray(span, M2Era100Constants.VertexCountOffset, vertexCount, verticesOfs);
        WriteArray(span, M2Era100Constants.DivisionCountOffset, 1, divisionOfs);
        // Textures stay empty; ValidateLayout treats a zero-count array as valid.

        for (int i = 0; i < vertexCount; i++)
        {
            int ofs = verticesOfs + (i * M2Era100Constants.VertexStride);
            WriteVector3(span, ofs + M2Era100Constants.VertexPositionOffset, i, i % 2, 0f);
            span[ofs + M2Era100Constants.VertexBoneWeightsOffset] = 255;
            span[ofs + M2Era100Constants.VertexBoneIndicesOffset] = (byte)i;
            WriteVector3(span, ofs + M2Era100Constants.VertexNormalOffset, 0f, 0f, 1f);
            WriteVector2(span, ofs + M2Era100Constants.VertexTexCoords0Offset, 0.25f * i, 0.75f * i);
            WriteVector2(span, ofs + M2Era100Constants.VertexTexCoords1Offset, 0.1f * i, 0.2f * i);
        }

        for (int i = 0; i < lookup.Length; i++)
            BinaryPrimitives.WriteUInt16LittleEndian(span.Slice(lookupOfs + (i * sizeof(ushort)), 2), lookup[i]);

        for (int i = 0; i < indices.Length; i++)
            BinaryPrimitives.WriteUInt16LittleEndian(span.Slice(indicesOfs + (i * sizeof(ushort)), 2), indices[i]);

        // Section: every field is uint16. With level != 0 the start fields hold only low bits.
        BinaryPrimitives.WriteUInt16LittleEndian(span.Slice(sectionsOfs + M2Era100Constants.SectionSubmeshIdOffset, 2), 0);
        BinaryPrimitives.WriteUInt16LittleEndian(span.Slice(sectionsOfs + M2Era100Constants.SectionLevelOffset, 2), level);
        BinaryPrimitives.WriteUInt16LittleEndian(span.Slice(sectionsOfs + M2Era100Constants.SectionVertexStartOffset, 2), 0);
        BinaryPrimitives.WriteUInt16LittleEndian(span.Slice(sectionsOfs + M2Era100Constants.SectionVertexCountOffset, 2), vertexCount);
        BinaryPrimitives.WriteUInt16LittleEndian(span.Slice(sectionsOfs + M2Era100Constants.SectionIndexStartOffset, 2), 0);
        BinaryPrimitives.WriteUInt16LittleEndian(span.Slice(sectionsOfs + M2Era100Constants.SectionIndexCountOffset, 2), indexCount);

        // Batch: bind section 0, no textures.
        BinaryPrimitives.WriteUInt16LittleEndian(span.Slice(batchesOfs + M2Era100Constants.BatchSkinSectionIndexOffset, 2), 0);

        // Division (0x2C): vertexLookup, indices, an unused uint32 array, sections, batches.
        WriteArray(span, divisionOfs + M2Era100Constants.DivisionVertexLookupCountOffset, lookup.Length, lookupOfs);
        WriteArray(span, divisionOfs + M2Era100Constants.DivisionIndicesCountOffset, indices.Length, indicesOfs);
        WriteArray(span, divisionOfs + M2Era100Constants.DivisionUint32ArrayCountOffset, 0, 0);
        WriteArray(span, divisionOfs + M2Era100Constants.DivisionSectionsCountOffset, 1, sectionsOfs);
        WriteArray(span, divisionOfs + M2Era100Constants.DivisionBatchesCountOffset, 1, batchesOfs);

        return data;
    }

    private static byte[] CreateSyntheticEra100CameraM2()
    {
        const int headerSize = 0x144;
        const int sequenceOffset = headerSize;
        const int cameraOffset = sequenceOffset + M2Era100Constants.SequenceStride;
        int cursor = cameraOffset + M2Era100Constants.CameraStride;

        int positionRanges = cursor; cursor += 0x08;
        int positionTimes = cursor; cursor += 0x08;
        int positionValues = cursor; cursor += 0x18;
        int targetRanges = cursor; cursor += 0x08;
        int targetTimes = cursor; cursor += 0x08;
        int targetValues = cursor; cursor += 0x18;
        int rollRanges = cursor; cursor += 0x08;
        int rollTimes = cursor; cursor += 0x08;
        int rollValues = cursor; cursor += 0x08;
        int cameraLookup = cursor; cursor += sizeof(short);

        byte[] data = new byte[cursor];
        Span<byte> span = data;
        BinaryPrimitives.WriteUInt32LittleEndian(span[..4], M2Era100Constants.Md20Magic);
        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(M2Era100Constants.VersionOffset, 4), 0x100u);
        WriteArray(span, M2Era100Constants.SequenceCountOffset, 1, sequenceOffset);
        WriteArray(span, M2Era100Constants.CameraCountOffset, 1, cameraOffset);
        WriteArray(span, M2Era100Constants.CameraLookupCountOffset, 1, cameraLookup);
        BinaryPrimitives.WriteUInt16LittleEndian(span.Slice(sequenceOffset + 0x00, 2), 0);
        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(sequenceOffset + 0x04, 4), 1000);

        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(cameraOffset + 0x00, 4), 1);
        WriteSingle(span, cameraOffset + 0x04, 1f);
        WriteSingle(span, cameraOffset + 0x08, 1000f);
        WriteSingle(span, cameraOffset + 0x0C, 1f);
        WriteOldTrack(span, cameraOffset + 0x10, positionRanges, positionTimes, positionValues, valueCount: 2);
        WriteVector3(span, cameraOffset + 0x2C, 0f, 0f, 0f);
        WriteOldTrack(span, cameraOffset + 0x38, targetRanges, targetTimes, targetValues, valueCount: 2);
        WriteVector3(span, cameraOffset + 0x54, 0f, 0f, 0f);
        WriteOldTrack(span, cameraOffset + 0x60, rollRanges, rollTimes, rollValues, valueCount: 2);

        WriteRange(span, positionRanges); WriteRange(span, targetRanges); WriteRange(span, rollRanges);
        WriteTimes(span, positionTimes); WriteTimes(span, targetTimes); WriteTimes(span, rollTimes);
        WriteVector3(span, positionValues, 1f, 2f, 3f);
        WriteVector3(span, positionValues + 0x0C, 11f, 12f, 13f);
        WriteVector3(span, targetValues, 4f, 5f, 6f);
        WriteVector3(span, targetValues + 0x0C, 14f, 15f, 16f);
        WriteSingle(span, rollValues, 0f); WriteSingle(span, rollValues + 4, 0.5f);
        BinaryPrimitives.WriteInt16LittleEndian(span.Slice(cameraLookup, 2), 0);
        return data;
    }

    private static void WriteOldTrack(Span<byte> span, int offset, int ranges, int times, int values, int valueCount)
    {
        BinaryPrimitives.WriteUInt16LittleEndian(span.Slice(offset + 0x00, 2), 1);
        BinaryPrimitives.WriteUInt16LittleEndian(span.Slice(offset + 0x02, 2), ushort.MaxValue);
        WriteArray(span, offset + 0x04, 1, ranges);
        WriteArray(span, offset + 0x0C, 2, times);
        WriteArray(span, offset + 0x14, valueCount, values);
    }

    private static void WriteRange(Span<byte> span, int offset)
    {
        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(offset, 4), 0);
        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(offset + 4, 4), 1);
    }

    private static void WriteTimes(Span<byte> span, int offset)
    {
        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(offset, 4), 0);
        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(offset + 4, 4), 1000);
    }

    private static void WriteSingle(Span<byte> span, int offset, float value)
        => BinaryPrimitives.WriteInt32LittleEndian(span.Slice(offset, 4), BitConverter.SingleToInt32Bits(value));

    private sealed class Vector3EqualityComparer(float tolerance) : IEqualityComparer<Vector3>
    {
        public bool Equals(Vector3 left, Vector3 right)
            => Vector3.DistanceSquared(left, right) <= tolerance * tolerance;

        public int GetHashCode(Vector3 value) => value.GetHashCode();
    }

    private static void WriteArray(Span<byte> span, int countOffset, int count, int offset)
    {
        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(countOffset, 4), (uint)count);
        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(countOffset + 4, 4), (uint)offset);
    }

    private static void WriteVector3(Span<byte> span, int offset, float x, float y, float z)
    {
        BinaryPrimitives.WriteSingleLittleEndian(span.Slice(offset, 4), x);
        BinaryPrimitives.WriteSingleLittleEndian(span.Slice(offset + 4, 4), y);
        BinaryPrimitives.WriteSingleLittleEndian(span.Slice(offset + 8, 4), z);
    }

    private static void WriteVector2(Span<byte> span, int offset, float x, float y)
    {
        BinaryPrimitives.WriteSingleLittleEndian(span.Slice(offset, 4), x);
        BinaryPrimitives.WriteSingleLittleEndian(span.Slice(offset + 4, 4), y);
    }

    [Fact]
    public void Era100Reader_Reads108ByteBones_AndPopulatesDocumentBones()
    {
        byte[] m2 = CreateSyntheticEra100M2WithBone();

        using MemoryStream stream = new(m2, writable: false);
        M2ModelDocument document = M2Era100ModelReader.Read(stream, "Character\\Synthetic\\Era100Bone.m2");

        M2BoneDefinition bone = Assert.Single(document.Bones);
        Assert.Equal(5, bone.KeyBoneId);
        Assert.Equal(8u, bone.Flags);
        Assert.Equal(-1, bone.ParentBone);
        Assert.Equal(0, bone.SubmeshId);
        Assert.Equal(new Vector3(1f, 2f, 3f), bone.Pivot);
        Assert.Equal(0u, bone.BoneNameCrc);
        Assert.Equal(M2TrackInterpolation.Linear, bone.TranslationTrack.Interpolation);
        Assert.Equal(M2TrackInterpolation.Linear, bone.RotationTrack.Interpolation);
        Assert.Equal(M2TrackInterpolation.Linear, bone.ScalingTrack.Interpolation);
    }

    [Fact]
    public void Era100Reader_Reads112ByteBones_WithBoneNameCrc()
    {
        byte[] m2 = CreateSyntheticEra104M2WithBone();

        using MemoryStream stream = new(m2, writable: false);
        M2ModelDocument document = M2Era100ModelReader.Read(stream, "Character\\Synthetic\\Era104Bone.m2");

        M2BoneDefinition bone = Assert.Single(document.Bones);
        Assert.Equal(7, bone.KeyBoneId);
        Assert.Equal(16u, bone.Flags);
        Assert.Equal(-1, bone.ParentBone);
        Assert.Equal(0, bone.SubmeshId);
        Assert.Equal(0x1F1A625Au, bone.BoneNameCrc);
        Assert.Equal(new Vector3(4f, 5f, 6f), bone.Pivot);
        Assert.Equal(M2TrackInterpolation.Linear, bone.TranslationTrack.Interpolation);
        Assert.Equal(M2TrackInterpolation.Linear, bone.RotationTrack.Interpolation);
        Assert.Equal(M2TrackInterpolation.Linear, bone.ScalingTrack.Interpolation);
    }

    [Fact]
    public void Era100Reader_ExtractsEmbeddedSkinDocuments_AndPreservesGlobalVertices()
    {
        byte[] m2 = CreateSyntheticEra100M2(indexCount: 6, level: 0);

        using MemoryStream stream = new(m2, writable: false);
        M2ModelDocument document = M2Era100ModelReader.Read(stream, "Character\\Synthetic\\Era100Skin.m2");

        M2SkinDocument skin = Assert.Single(document.EmbeddedSkinDocuments);
        Assert.Single(skin.Submeshes);
        Assert.Equal(6, skin.TriangleIndices.Count);
        Assert.Single(skin.Batches);

        Assert.NotNull(document.InlineEra100Geometry);
        Assert.Equal(4, document.InlineEra100Geometry.GlobalVertices.Count);
        Assert.Equal(4, document.InlineEra100Geometry.RenderVertices.Count);
    }

    [Theory]
    [InlineData(0x100u)]
    [InlineData(0x104u)]
    [InlineData(0x107u)]
    public void Era100Reader_AcceptsLegacyVersions_0x100_Through_0x107(uint version)
    {
        byte[] m2 = CreateSyntheticEra100MultiVersionM2(indexCount: 6, level: 0, version: version);

        using MemoryStream stream = new(m2, writable: false);
        M2DispatchResult result = M2ModelReaderDispatcher.ReadDetailed(stream, $"Character\\Synthetic\\Era100_{version:X}.m2");

        Assert.Equal(M2Era1121EraTag.Md20_1X_V100_Era100, result.Era);
        Assert.Equal(version, result.Document.Version);
        Assert.NotNull(result.Document.InlineEra100Geometry);
    }

    [Fact]
    public void Era100Reader_ZeroVertexModel_HasNullInlineGeometry()
    {
        byte[] m2 = CreateSyntheticEra100CameraM2();
        using MemoryStream stream = new(m2, writable: false);
        M2DispatchResult result = M2ModelReaderDispatcher.ReadDetailed(stream, @"Cameras\TestCamera.m2");

        Assert.Equal(M2Era1121EraTag.Md20_1X_V100_Era100, result.Era);
        Assert.Null(result.Document.InlineEra100Geometry);
    }

    [Fact]
    public void ReadDetailed_ZangarPlantGroup05_ParsesMaterialsAndEmbeddedSectionsCorrectly()
    {
        string clientPath = @"H:\CLIENTS\TBC\2.X_Retail_Windows_enUS_2.4.3.8606\World of Warcraft\Data";
        if (!Directory.Exists(clientPath)) return;
        byte[] m2Bytes = WowViewer.Core.IO.Files.ArchiveVirtualFileReader.ReadVirtualFile(
            @"world\expansion01\doodads\zangar\plantgroups\zangarplantgroup05.m2",
            [clientPath],
            (string?)null);

        using MemoryStream stream = new(m2Bytes, writable: false);
        var dispatch = M2ModelReaderDispatcher.ReadDetailed(stream, @"world\expansion01\doodads\zangar\plantgroups\zangarplantgroup05.m2");
        var geom = dispatch.Document.InlineEra100Geometry;
        Assert.NotNull(geom);

        // Verify material blend modes: material 0 must be AlphaKey (1) for transparent cutout foliage
        Assert.Equal(3, geom.Materials.Count);
        Assert.Equal(1, geom.Materials[0].BlendMode); // AlphaKey
        Assert.Equal(0, geom.Materials[1].BlendMode); // Opaque
        Assert.Equal(4, geom.Materials[2].BlendMode); // Add

        // Verify embedded division sections parsed with 48-byte stride (not 32-byte stride)
        Assert.Equal(2, geom.Sections.Count);
        Assert.Equal(0u, geom.Sections[0].VertexStart);
        Assert.Equal(233u, geom.Sections[0].VertexCount);
        Assert.Equal(0u, geom.Sections[0].IndexStart);
        Assert.Equal(960u, geom.Sections[0].IndexCount);

        Assert.Equal(233u, geom.Sections[1].VertexStart);
        Assert.Equal(417u, geom.Sections[1].VertexCount);
        Assert.Equal(960u, geom.Sections[1].IndexStart);
        Assert.Equal(1908u, geom.Sections[1].IndexCount);

        // Total vertices across both sections must match global vertex count
        Assert.Equal(650, geom.RenderVertices.Count);

        // Batches mapping
        Assert.Equal(3, geom.Batches.Count);
        Assert.Equal(0, geom.Batches[0].SkinSectionIndex);
        Assert.Equal(0, geom.Batches[0].MaterialIndex);
        Assert.Equal(1, geom.Batches[1].SkinSectionIndex);
        Assert.Equal(1, geom.Batches[1].MaterialIndex);
        Assert.Equal(1, geom.Batches[2].SkinSectionIndex);
        Assert.Equal(2, geom.Batches[2].MaterialIndex);
    }

    [Fact]
    public void Era100_TrollFemale_QuaternionNormalizationAndSampling()
    {
        string path = @"I:\parp\parp-tools\wow-viewer\src\viewer\WoWViewer\bin\Debug\net10.0\output\cache\3391d25045a9cd6dbad6f7bd1c4487d2a68e8704\TrollFemale.m2";
        if (!File.Exists(path)) return;

        byte[] bytes = File.ReadAllBytes(path);
        using MemoryStream ms = new(bytes, writable: false);
        var dispatch = M2ModelReaderDispatcher.ReadDetailed(ms, path);
        var model = dispatch.Document;

        Assert.Equal(M2Era1121EraTag.Md20_1X_V100_Era100, dispatch.Era);
        Assert.True(model.Bones.Count > 0);

        var b0 = model.Bones[0];
        var rot0 = WowViewer.Core.Runtime.M2.M2TrackSampler.SampleCompressedQuaternion(model.RawBytes, model, 0, 0, b0.RotationTrack, Quaternion.Identity);
        Assert.InRange(rot0.X, -0.05f, 0.05f);
        Assert.InRange(rot0.Y, -0.05f, 0.05f);
        Assert.InRange(rot0.Z, -0.05f, 0.05f);
        Assert.InRange(rot0.W, 0.95f, 1.05f);

        var b66 = model.Bones[66];
        var rot66 = WowViewer.Core.Runtime.M2.M2TrackSampler.SampleCompressedQuaternion(model.RawBytes, model, 1, 0, b66.RotationTrack, Quaternion.Identity);
        float lenSq66 = (rot66.X * rot66.X) + (rot66.Y * rot66.Y) + (rot66.Z * rot66.Z) + (rot66.W * rot66.W);
        Assert.InRange(lenSq66, 0.95f, 1.05f);
    }

    [Fact]
    public void Era100_Synthetic_UncompressedQuaternion_NormalizedAndSampled()
    {
        byte[] data = CreateSyntheticEra100M2WithUncompressedBone();

        using MemoryStream stream = new(data, writable: false);
        var dispatch = M2ModelReaderDispatcher.ReadDetailed(stream, "SyntheticUncompressed.m2");
        var model = dispatch.Document;

        Assert.Equal(M2Era1121EraTag.Md20_1X_V100_Era100, dispatch.Era);
        Assert.NotNull(model.Bones);
        Assert.Single(model.Bones);

        var b0 = model.Bones[0];
        var rot0 = WowViewer.Core.Runtime.M2.M2TrackSampler.SampleCompressedQuaternion(
            model.RawBytes, model, 0, 0, b0.RotationTrack, Quaternion.Identity);

        Assert.InRange(rot0.X, 0.70f, 0.71f);
        Assert.InRange(rot0.Y, -0.01f, 0.01f);
        Assert.InRange(rot0.Z, -0.01f, 0.01f);
        Assert.InRange(rot0.W, 0.70f, 0.71f);
        float lenSq = (rot0.X * rot0.X) + (rot0.Y * rot0.Y) + (rot0.Z * rot0.Z) + (rot0.W * rot0.W);
        Assert.InRange(lenSq, 0.99f, 1.01f);
    }

    [Fact]
    public void Era100_Synthetic_MaterialsWithAlphaKey_ParsedCorrectly()
    {
        byte[] data = CreateSyntheticEra100M2WithMaterial(blendMode: 1);

        using MemoryStream stream = new(data, writable: false);
        var dispatch = M2ModelReaderDispatcher.ReadDetailed(stream, "SyntheticAlphaKey.m2");
        var geom = dispatch.Document.InlineEra100Geometry;

        Assert.NotNull(geom);
        Assert.Single(geom.Materials);
        Assert.Equal(1, geom.Materials[0].BlendMode);
        Assert.Single(geom.Batches);
        Assert.Equal(0, geom.Batches[0].MaterialIndex);
    }

    private static byte[] CreateSyntheticEra100MultiVersionM2(ushort indexCount, ushort level, uint version)
    {
        byte[] data = CreateSyntheticEra100M2(indexCount, level);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(M2Era100Constants.VersionOffset, 4), version);
        return data;
    }

    private static byte[] CreateSyntheticEra100M2WithBone()
    {
        const int headerSize = 0x144;
        int boneOffset = headerSize;
        int cursor = boneOffset + M2Era100Constants.BoneStride;

        int transRanges = cursor; cursor += 8;
        int transTimes = cursor; cursor += 4;
        int transValues = cursor; cursor += 12;

        int rotRanges = cursor; cursor += 8;
        int rotTimes = cursor; cursor += 4;
        int rotValues = cursor; cursor += 8;

        int scaleRanges = cursor; cursor += 8;
        int scaleTimes = cursor; cursor += 4;
        int scaleValues = cursor; cursor += 12;

        byte[] data = new byte[cursor];
        Span<byte> span = data;

        BinaryPrimitives.WriteUInt32LittleEndian(span[..4], M2Era100Constants.Md20Magic);
        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(M2Era100Constants.VersionOffset, 4), 0x100u);

        WriteArray(span, M2Era100Constants.BoneCountOffset, 1, boneOffset);

        BinaryPrimitives.WriteInt32LittleEndian(span.Slice(boneOffset + 0x00, 4), 5);
        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(boneOffset + 0x04, 4), 8u);
        BinaryPrimitives.WriteInt16LittleEndian(span.Slice(boneOffset + 0x08, 2), -1);
        BinaryPrimitives.WriteUInt16LittleEndian(span.Slice(boneOffset + 0x0A, 2), 0);

        WriteOldTrackSingle(span, boneOffset + 0x0C, transRanges, transTimes, transValues, valueCount: 1);
        WriteOldTrackSingle(span, boneOffset + 0x28, rotRanges, rotTimes, rotValues, valueCount: 1);
        WriteOldTrackSingle(span, boneOffset + 0x44, scaleRanges, scaleTimes, scaleValues, valueCount: 1);
        WriteVector3(span, boneOffset + 0x60, 1f, 2f, 3f);

        WriteRangeSingle(span, transRanges);
        WriteRangeSingle(span, rotRanges);
        WriteRangeSingle(span, scaleRanges);

        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(transTimes, 4), 0);
        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(rotTimes, 4), 0);
        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(scaleTimes, 4), 0);

        WriteVector3(span, transValues, 10f, 20f, 30f);
        BinaryPrimitives.WriteInt16LittleEndian(span.Slice(rotValues + 0, 2), 0);
        BinaryPrimitives.WriteInt16LittleEndian(span.Slice(rotValues + 2, 2), 0);
        BinaryPrimitives.WriteInt16LittleEndian(span.Slice(rotValues + 4, 2), 0);
        BinaryPrimitives.WriteInt16LittleEndian(span.Slice(rotValues + 6, 2), short.MaxValue);
        WriteVector3(span, scaleValues, 1f, 1f, 1f);

        return data;
    }

    private static byte[] CreateSyntheticEra104M2WithBone()
    {
        const int headerSize = 0x144;
        int boneOffset = headerSize;
        int cursor = boneOffset + M2Era100Constants.BoneStrideEra104;

        int transRanges = cursor; cursor += 8;
        int transTimes = cursor; cursor += 4;
        int transValues = cursor; cursor += 12;

        int rotRanges = cursor; cursor += 8;
        int rotTimes = cursor; cursor += 4;
        int rotValues = cursor; cursor += 8;

        int scaleRanges = cursor; cursor += 8;
        int scaleTimes = cursor; cursor += 4;
        int scaleValues = cursor; cursor += 12;

        byte[] data = new byte[cursor];
        Span<byte> span = data;

        BinaryPrimitives.WriteUInt32LittleEndian(span[..4], M2Era100Constants.Md20Magic);
        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(M2Era100Constants.VersionOffset, 4), 0x107u);

        WriteArray(span, M2Era100Constants.BoneCountOffset, 1, boneOffset);

        BinaryPrimitives.WriteInt32LittleEndian(span.Slice(boneOffset + 0x00, 4), 7);
        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(boneOffset + 0x04, 4), 16u);
        BinaryPrimitives.WriteInt16LittleEndian(span.Slice(boneOffset + 0x08, 2), -1);
        BinaryPrimitives.WriteUInt16LittleEndian(span.Slice(boneOffset + 0x0A, 2), 0);
        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(boneOffset + 0x0C, 4), 0x1F1A625Au);

        WriteOldTrackSingle(span, boneOffset + 0x10, transRanges, transTimes, transValues, valueCount: 1);
        WriteOldTrackSingle(span, boneOffset + 0x2C, rotRanges, rotTimes, rotValues, valueCount: 1);
        WriteOldTrackSingle(span, boneOffset + 0x48, scaleRanges, scaleTimes, scaleValues, valueCount: 1);
        WriteVector3(span, boneOffset + 0x64, 4f, 5f, 6f);

        WriteRangeSingle(span, transRanges);
        WriteRangeSingle(span, rotRanges);
        WriteRangeSingle(span, scaleRanges);

        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(transTimes, 4), 0);
        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(rotTimes, 4), 0);
        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(scaleTimes, 4), 0);

        WriteVector3(span, transValues, 10f, 20f, 30f);
        BinaryPrimitives.WriteInt16LittleEndian(span.Slice(rotValues + 0, 2), 0);
        BinaryPrimitives.WriteInt16LittleEndian(span.Slice(rotValues + 2, 2), 0);
        BinaryPrimitives.WriteInt16LittleEndian(span.Slice(rotValues + 4, 2), 0);
        BinaryPrimitives.WriteInt16LittleEndian(span.Slice(rotValues + 6, 2), short.MaxValue);
        WriteVector3(span, scaleValues, 1f, 1f, 1f);

        return data;
    }

    private static void WriteRangeSingle(Span<byte> span, int offset)
    {
        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(offset, 4), 0);
        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(offset + 4, 4), 0);
    }

    private static void WriteOldTrackSingle(Span<byte> span, int offset, int ranges, int times, int values, int valueCount)
    {
        BinaryPrimitives.WriteUInt16LittleEndian(span.Slice(offset + 0x00, 2), 1);
        BinaryPrimitives.WriteUInt16LittleEndian(span.Slice(offset + 0x02, 2), ushort.MaxValue);
        WriteArray(span, offset + 0x04, 1, ranges);
        WriteArray(span, offset + 0x0C, 1, times);
        WriteArray(span, offset + 0x14, valueCount, values);
    }

    private static byte[] CreateSyntheticEra100M2WithUncompressedBone()
    {
        const int headerSize = 0x144;
        const int sequenceOffset = headerSize;
        int boneOffset = sequenceOffset + M2Era100Constants.SequenceStride;
        int cursor = boneOffset + M2Era100Constants.BoneStride;

        int transRanges = cursor; cursor += 8;
        int transTimes = cursor; cursor += 4;
        int transValues = cursor; cursor += 12;

        int rotRanges = cursor; cursor += 8;
        int rotTimes = cursor; cursor += 4;
        int rotValues = cursor; cursor += 16;

        int scaleRanges = cursor; cursor += 8;
        int scaleTimes = cursor; cursor += 4;
        int scaleValues = cursor; cursor += 12;

        byte[] data = new byte[cursor];
        Span<byte> span = data;

        BinaryPrimitives.WriteUInt32LittleEndian(span[..4], M2Era100Constants.Md20Magic);
        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(M2Era100Constants.VersionOffset, 4), 0x100u);

        WriteArray(span, M2Era100Constants.SequenceCountOffset, 1, sequenceOffset);
        WriteArray(span, M2Era100Constants.BoneCountOffset, 1, boneOffset);

        BinaryPrimitives.WriteUInt16LittleEndian(span.Slice(sequenceOffset + 0x00, 2), 0);
        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(sequenceOffset + 0x04, 4), 1000);

        BinaryPrimitives.WriteInt32LittleEndian(span.Slice(boneOffset + 0x00, 4), 5);
        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(boneOffset + 0x04, 4), 8u);
        BinaryPrimitives.WriteInt16LittleEndian(span.Slice(boneOffset + 0x08, 2), -1);
        BinaryPrimitives.WriteUInt16LittleEndian(span.Slice(boneOffset + 0x0A, 2), 0);

        WriteOldTrackSingle(span, boneOffset + 0x0C, transRanges, transTimes, transValues, valueCount: 1);
        WriteOldTrackSingle(span, boneOffset + 0x28, rotRanges, rotTimes, rotValues, valueCount: 1);
        WriteOldTrackSingle(span, boneOffset + 0x44, scaleRanges, scaleTimes, scaleValues, valueCount: 1);
        WriteVector3(span, boneOffset + 0x60, 1f, 2f, 3f);

        WriteRangeSingle(span, transRanges);
        WriteRangeSingle(span, rotRanges);
        WriteRangeSingle(span, scaleRanges);

        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(transTimes, 4), 0);
        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(rotTimes, 4), 0);
        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(scaleTimes, 4), 0);

        WriteVector3(span, transValues, 10f, 20f, 30f);
        WriteSingle(span, rotValues + 0, 0.70710678f);
        WriteSingle(span, rotValues + 4, 0f);
        WriteSingle(span, rotValues + 8, 0f);
        WriteSingle(span, rotValues + 12, 0.70710678f);
        WriteVector3(span, scaleValues, 1f, 1f, 1f);

        return data;
    }

    public static byte[] CreateSyntheticEra100M2WithMaterial(ushort blendMode)
    {
        const int headerSize = 0x144;
        const int vertexCount = 4;
        ushort[] lookup = [0, 1, 2, 3];
        ushort[] indices = [0, 1, 2, 0, 2, 3];

        int verticesOfs = headerSize;
        int lookupOfs = verticesOfs + (vertexCount * M2Era100Constants.VertexStride);
        int indicesOfs = lookupOfs + (lookup.Length * sizeof(ushort));
        int sectionsOfs = indicesOfs + (indices.Length * sizeof(ushort));
        int batchesOfs = sectionsOfs + M2Era100Constants.SectionStride;
        int divisionOfs = batchesOfs + M2Era100Constants.BatchStride;
        int materialsOfs = divisionOfs + M2Era100Constants.DivisionStride;
        int texturesOfs = materialsOfs + 4;
        int textureFileNameOfs = texturesOfs + 16;
        int total = textureFileNameOfs + 16;

        byte[] data = new byte[total];
        Span<byte> span = data;

        BinaryPrimitives.WriteUInt32LittleEndian(span[..4], M2Era100Constants.Md20Magic);
        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(M2Era100Constants.VersionOffset, 4), 0x100u);

        WriteArray(span, M2Era100Constants.VertexCountOffset, vertexCount, verticesOfs);
        WriteArray(span, M2Era100Constants.DivisionCountOffset, 1, divisionOfs);
        WriteArray(span, M2Era100Constants.MaterialCountOffset, 1, materialsOfs);
        WriteArray(span, M2Era100Constants.TextureCountOffset, 1, texturesOfs);

        for (int i = 0; i < vertexCount; i++)
        {
            int ofs = verticesOfs + (i * M2Era100Constants.VertexStride);
            WriteVector3(span, ofs + M2Era100Constants.VertexPositionOffset, i, i % 2, 0f);
            span[ofs + M2Era100Constants.VertexBoneWeightsOffset] = 255;
            span[ofs + M2Era100Constants.VertexBoneIndicesOffset] = (byte)i;
            WriteVector3(span, ofs + M2Era100Constants.VertexNormalOffset, 0f, 0f, 1f);
            WriteVector2(span, ofs + M2Era100Constants.VertexTexCoords0Offset, 0.25f * i, 0.75f * i);
            WriteVector2(span, ofs + M2Era100Constants.VertexTexCoords1Offset, 0.1f * i, 0.2f * i);
        }

        for (int i = 0; i < lookup.Length; i++)
            BinaryPrimitives.WriteUInt16LittleEndian(span.Slice(lookupOfs + (i * sizeof(ushort)), 2), lookup[i]);

        for (int i = 0; i < indices.Length; i++)
            BinaryPrimitives.WriteUInt16LittleEndian(span.Slice(indicesOfs + (i * sizeof(ushort)), 2), indices[i]);

        // Section
        BinaryPrimitives.WriteUInt16LittleEndian(span.Slice(sectionsOfs + M2Era100Constants.SectionSubmeshIdOffset, 2), 0);
        BinaryPrimitives.WriteUInt16LittleEndian(span.Slice(sectionsOfs + M2Era100Constants.SectionLevelOffset, 2), 0);
        BinaryPrimitives.WriteUInt16LittleEndian(span.Slice(sectionsOfs + M2Era100Constants.SectionVertexStartOffset, 2), 0);
        BinaryPrimitives.WriteUInt16LittleEndian(span.Slice(sectionsOfs + M2Era100Constants.SectionVertexCountOffset, 2), vertexCount);
        BinaryPrimitives.WriteUInt16LittleEndian(span.Slice(sectionsOfs + M2Era100Constants.SectionIndexStartOffset, 2), 0);
        BinaryPrimitives.WriteUInt16LittleEndian(span.Slice(sectionsOfs + M2Era100Constants.SectionIndexCountOffset, 2), (ushort)indices.Length);

        // Batch: bind section 0, material 0
        BinaryPrimitives.WriteUInt16LittleEndian(span.Slice(batchesOfs + M2Era100Constants.BatchSkinSectionIndexOffset, 2), 0);
        BinaryPrimitives.WriteUInt16LittleEndian(span.Slice(batchesOfs + M2Era100Constants.BatchMaterialIndexOffset, 2), 0);
        BinaryPrimitives.WriteUInt16LittleEndian(span.Slice(batchesOfs + M2Era100Constants.BatchTextureCountOffset, 2), 1);

        // Division
        WriteArray(span, divisionOfs + M2Era100Constants.DivisionVertexLookupCountOffset, lookup.Length, lookupOfs);
        WriteArray(span, divisionOfs + M2Era100Constants.DivisionIndicesCountOffset, indices.Length, indicesOfs);
        WriteArray(span, divisionOfs + M2Era100Constants.DivisionUint32ArrayCountOffset, 0, 0);
        WriteArray(span, divisionOfs + M2Era100Constants.DivisionSectionsCountOffset, 1, sectionsOfs);
        WriteArray(span, divisionOfs + M2Era100Constants.DivisionBatchesCountOffset, 1, batchesOfs);

        // Material at materialsOfs: {uint16 flags, uint16 blendMode}
        BinaryPrimitives.WriteUInt16LittleEndian(span.Slice(materialsOfs + 0, 2), 0);
        BinaryPrimitives.WriteUInt16LittleEndian(span.Slice(materialsOfs + 2, 2), blendMode);

        // Texture at texturesOfs: {uint32 type, uint32 flags, uint32 nameLen, uint32 nameOfs}
        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(texturesOfs + 0, 4), 0);
        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(texturesOfs + 4, 4), 0);
        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(texturesOfs + 8, 4), 4);
        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(texturesOfs + 12, 4), (uint)textureFileNameOfs);
        Encoding.ASCII.GetBytes("test").CopyTo(span.Slice(textureFileNameOfs, 4));

        return data;
    }
}
