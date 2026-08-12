using System.Buffers.Binary;
using System.Numerics;
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
            WriteVector3(span, ofs + M2Era100Constants.VertexNormalOffset, 0f, 0f, 1f);
            span[ofs + M2Era100Constants.VertexBoneWeightsOffset] = 255;
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
}
