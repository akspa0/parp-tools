using System.Buffers.Binary;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class AdtSummaryReaderTests
{
    [Fact]
    public void Read_RootAdtBuffer_ProducesSemanticSummary()
    {
        byte[] bytes =
        [
            .. CreateChunk("MVER", CreateUInt32Payload(18)),
            .. CreateChunk("MHDR", new byte[64]),
            .. CreateChunk("MFBO", new byte[36]),
            .. CreateChunk("MH2O", new byte[128]),
            .. CreateChunk("MCNK", new byte[256]),
            .. CreateChunk("MCNK", new byte[256]),
        ];

        using MemoryStream stream = new(bytes);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, "synthetic_0_0.adt");
        AdtSummary summary = AdtSummaryReader.Read(stream, fileSummary);

        Assert.Equal(MapFileKind.Adt, summary.Kind);
        Assert.Equal(2, summary.TerrainChunkCount);
        Assert.Equal(0, summary.TextureNameCount);
        Assert.Equal(0, summary.ModelNameCount);
        Assert.Equal(0, summary.WorldModelNameCount);
        Assert.Equal(0, summary.ModelPlacementCount);
        Assert.Equal(0, summary.WorldModelPlacementCount);
        Assert.True(summary.HasFlightBounds);
        Assert.True(summary.HasWater);
        Assert.False(summary.HasTextureParams);
        Assert.False(summary.HasTextureFlags);
    }

    [Fact]
    public void Read_TexAdtBuffer_ProducesSemanticSummary()
    {
        byte[] bytes =
        [
            .. CreateChunk("MVER", CreateUInt32Payload(18)),
            .. CreateChunk("MAMP", CreateUInt32Payload(7)),
            .. CreateChunk("MTEX", CreateStringBlock("foo.blp", "bar.blp", "baz.blp")),
            .. CreateChunk("MTXF", new byte[12]),
            .. CreateChunk("MCNK", new byte[24]),
            .. CreateChunk("MCNK", new byte[24]),
            .. CreateChunk("MCNK", new byte[24]),
        ];

        using MemoryStream stream = new(bytes);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, "synthetic_0_0_tex0.adt");
        AdtSummary summary = AdtSummaryReader.Read(stream, fileSummary);

        Assert.Equal(MapFileKind.AdtTex, summary.Kind);
        Assert.Equal(3, summary.TerrainChunkCount);
        Assert.Equal(3, summary.TextureNameCount);
        Assert.Equal(0, summary.ModelNameCount);
        Assert.Equal(0, summary.WorldModelNameCount);
        Assert.Equal(0, summary.ModelPlacementCount);
        Assert.Equal(0, summary.WorldModelPlacementCount);
        Assert.False(summary.HasFlightBounds);
        Assert.False(summary.HasWater);
        Assert.True(summary.HasTextureParams);
        Assert.True(summary.HasTextureFlags);
    }

    [Fact]
    public void Read_ObjAdtBuffer_ProducesSemanticSummary()
    {
        byte[] bytes =
        [
            .. CreateChunk("MVER", CreateUInt32Payload(18)),
            .. CreateChunk("MMDX", CreateStringBlock("foo.mdx", "bar.mdx")),
            .. CreateChunk("MMID", new byte[8]),
            .. CreateChunk("MWMO", CreateStringBlock("a.wmo", "b.wmo", "c.wmo")),
            .. CreateChunk("MWID", new byte[12]),
            .. CreateChunk("MDDF", new byte[36 * 2]),
            .. CreateChunk("MODF", new byte[64]),
            .. CreateChunk("MCNK", Array.Empty<byte>()),
            .. CreateChunk("MCNK", Array.Empty<byte>()),
        ];

        using MemoryStream stream = new(bytes);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, "synthetic_0_0_obj0.adt");
        AdtSummary summary = AdtSummaryReader.Read(stream, fileSummary);

        Assert.Equal(MapFileKind.AdtObj, summary.Kind);
        Assert.Equal(2, summary.TerrainChunkCount);
        Assert.Equal(0, summary.TextureNameCount);
        Assert.Equal(2, summary.ModelNameCount);
        Assert.Equal(3, summary.WorldModelNameCount);
        Assert.Equal(2, summary.ModelPlacementCount);
        Assert.Equal(1, summary.WorldModelPlacementCount);
        Assert.False(summary.HasFlightBounds);
        Assert.False(summary.HasWater);
        Assert.False(summary.HasTextureParams);
        Assert.False(summary.HasTextureFlags);
    }

    [Fact]
    public void Read_DevelopmentRootAdt_ProducesExpectedSemanticSummary()
    {
        AdtSummary summary = AdtSummaryReader.Read(MapTestPaths.DevelopmentRootAdtPath);

        Assert.Equal(MapFileKind.Adt, summary.Kind);
        Assert.Equal(256, summary.TerrainChunkCount);
        Assert.Equal(0, summary.TextureNameCount);
        Assert.Equal(0, summary.ModelNameCount);
        Assert.Equal(0, summary.WorldModelNameCount);
        Assert.Equal(0, summary.ModelPlacementCount);
        Assert.Equal(0, summary.WorldModelPlacementCount);
        Assert.True(summary.HasFlightBounds);
        Assert.False(summary.HasWater);
        Assert.False(summary.HasTextureParams);
        Assert.False(summary.HasTextureFlags);
    }

    [Fact]
    public void Read_DevelopmentTexAdt_ProducesExpectedSemanticSummary()
    {
        AdtSummary summary = AdtSummaryReader.Read(MapTestPaths.DevelopmentTexAdtPath);

        Assert.Equal(MapFileKind.AdtTex, summary.Kind);
        Assert.Equal(256, summary.TerrainChunkCount);
        Assert.Equal(5, summary.TextureNameCount);
        Assert.Equal(0, summary.ModelNameCount);
        Assert.Equal(0, summary.WorldModelNameCount);
        Assert.Equal(0, summary.ModelPlacementCount);
        Assert.Equal(0, summary.WorldModelPlacementCount);
        Assert.False(summary.HasFlightBounds);
        Assert.False(summary.HasWater);
        Assert.True(summary.HasTextureParams);
        Assert.False(summary.HasTextureFlags);
    }

    [Fact]
    public void Read_DevelopmentObjAdt_ProducesExpectedSemanticSummary()
    {
        AdtSummary summary = AdtSummaryReader.Read(MapTestPaths.DevelopmentObjAdtPath);

        Assert.Equal(MapFileKind.AdtObj, summary.Kind);
        Assert.Equal(256, summary.TerrainChunkCount);
        Assert.Equal(0, summary.TextureNameCount);
        Assert.Equal(6, summary.ModelNameCount);
        Assert.Equal(12, summary.WorldModelNameCount);
        Assert.Equal(10, summary.ModelPlacementCount);
        Assert.Equal(15, summary.WorldModelPlacementCount);
        Assert.False(summary.HasFlightBounds);
        Assert.False(summary.HasWater);
        Assert.False(summary.HasTextureParams);
        Assert.False(summary.HasTextureFlags);
    }

    private static byte[] CreateChunk(string id, byte[] payload)
    {
        byte[] bytes = new byte[8 + payload.Length];
        Array.Copy(WowViewer.Core.Chunks.FourCC.FromString(id).ToFileBytes(), 0, bytes, 0, 4);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4), (uint)payload.Length);
        Array.Copy(payload, 0, bytes, 8, payload.Length);
        return bytes;
    }

    private static byte[] CreateUInt32Payload(uint value)
    {
        byte[] bytes = new byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes, value);
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

        return stream.ToArray();
    }
}
