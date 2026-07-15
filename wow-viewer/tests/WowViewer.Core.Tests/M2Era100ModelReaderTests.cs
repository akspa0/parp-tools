using System.Buffers.Binary;
using WowViewer.Core.IO.M2Chunked;
using WowViewer.Core.IO.M2Era100;
using WowViewer.Core.IO.M2Era1121;
using WowViewer.Core.M2;

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
