using System.Buffers.Binary;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoMaterialDetailReaderTests
{
    [Fact]
    public void Read_V17MomtBuffer_ProducesResolvedTextureNames()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", CreateMohd(materialCount: 2)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOTX", CreateStringBlock("tex_a.blp", "tex_b.blp", "tex_c.blp")),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOMT", CreateMomtEntries(64,
                CreateMomtEntry(64, flags: 1, shader: 2, blendMode: 3, texture1Offset: 0, texture2Offset: 10, texture3Offset: 20),
                CreateMomtEntry(64, flags: 4, shader: 5, blendMode: 6, texture1Offset: 20, texture2Offset: 0, texture3Offset: 10))),
        ];

        using MemoryStream stream = new(bytes);
        IReadOnlyList<WmoMaterialDetail> details = WmoMaterialDetailReader.Read(stream, "synthetic_v17_materials.wmo");

        Assert.Equal(2, details.Count);
        Assert.Equal(64, details[0].EntrySizeBytes);
        Assert.Equal((uint)1, details[0].Flags);
        Assert.Equal((uint)2, details[0].Shader);
        Assert.Equal((uint)3, details[0].BlendMode);
        Assert.Equal("tex_a.blp", details[0].Texture1Name);
        Assert.Equal("tex_b.blp", details[0].Texture2Name);
        Assert.Equal("tex_c.blp", details[0].Texture3Name);
        Assert.Equal(64, details[1].PayloadOffset);
        Assert.Equal("tex_c.blp", details[1].Texture1Name);
    }

    [Fact]
    public void Read_V14MomtBuffer_UsesLegacyEntrySize()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(14)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOMO", MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", CreateMohd(materialCount: 1))),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOTX", CreateStringBlock("alpha_a.blp", "alpha_b.blp", "alpha_c.blp")),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOMT", CreateMomtEntries(48,
                CreateMomtEntry(48, flags: 7, shader: 8, blendMode: 9, texture1Offset: 0, texture2Offset: 12, texture3Offset: 24))),
        ];

        using MemoryStream stream = new(bytes);
        IReadOnlyList<WmoMaterialDetail> details = WmoMaterialDetailReader.Read(stream, "synthetic_v14_materials.wmo");

        WmoMaterialDetail detail = Assert.Single(details);
        Assert.Equal(48, detail.EntrySizeBytes);
        Assert.Equal((uint)7, detail.Flags);
        Assert.Equal((uint)8, detail.Shader);
        Assert.Equal((uint)9, detail.BlendMode);
        Assert.Equal("alpha_a.blp", detail.Texture1Name);
        Assert.Equal("alpha_b.blp", detail.Texture2Name);
        Assert.Equal("alpha_c.blp", detail.Texture3Name);
    }

    private static byte[] CreateMohd(uint materialCount)
    {
        byte[] bytes = new byte[64];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(0, 4), materialCount);
        return bytes;
    }

    private static byte[] CreateMomtEntries(int entrySize, params byte[][] entries)
    {
        using MemoryStream stream = new();
        foreach (byte[] entry in entries)
        {
            Assert.Equal(entrySize, entry.Length);
            stream.Write(entry, 0, entry.Length);
        }

        return stream.ToArray();
    }

    private static byte[] CreateMomtEntry(int entrySize, uint flags, uint shader, uint blendMode, uint texture1Offset, uint texture2Offset, uint texture3Offset)
    {
        byte[] bytes = new byte[entrySize];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(0, 4), flags);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4, 4), shader);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(8, 4), blendMode);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(12, 4), texture1Offset);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(24, 4), texture2Offset);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(36, 4), texture3Offset);
        return bytes;
    }

    private static byte[] CreateStringBlock(params string[] entries)
    {
        using MemoryStream stream = new();
        foreach (string entry in entries)
        {
            byte[] bytes = System.Text.Encoding.UTF8.GetBytes(entry);
            stream.Write(bytes, 0, bytes.Length);
            stream.WriteByte(0);
        }

        if ((stream.Length & 1) != 0)
            stream.WriteByte(0);

        return stream.ToArray();
    }
}