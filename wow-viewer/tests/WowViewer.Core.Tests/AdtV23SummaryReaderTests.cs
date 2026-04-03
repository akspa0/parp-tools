using System.Buffers.Binary;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class AdtV23SummaryReaderTests
{
    [Fact]
    public void Read_AdtV23Buffer_ProducesSemanticSummary()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("AHDR", CreateAhdrPayload(version: 23, verticesX: 129, verticesY: 129, chunksX: 16, chunksY: 16)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("AVTX", new byte[32]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("ANRM", new byte[24]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("ATEX", CreateStringBlock("grass.blp")),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("ATEX", CreateStringBlock("rock.blp")),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("ADOO", CreateStringBlock("tree.m2")),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("ACNK", new byte[64]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("ACNK", new byte[64]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("AFBO", new byte[0x48]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("ACVT", new byte[64]),
        ];

        using MemoryStream stream = new(bytes);
        MapFileSummary summary = MapFileSummaryReader.Read(stream, "synthetic.error");
        AdtV23Summary semantic = AdtV23SummaryReader.Read(stream, summary);

        Assert.Equal(MapFileKind.AdtV23Error, semantic.Kind);
        Assert.Equal(23u, semantic.HeaderVersion);
        Assert.Equal(129, semantic.VerticesX);
        Assert.Equal(129, semantic.VerticesY);
        Assert.Equal(16, semantic.ChunksX);
        Assert.Equal(16, semantic.ChunksY);
        Assert.Equal(2, semantic.TerrainChunkCount);
        Assert.Equal(2, semantic.TextureNameCount);
        Assert.Equal(1, semantic.ObjectNameCount);
        Assert.True(semantic.HasVertexHeights);
        Assert.True(semantic.HasNormals);
        Assert.True(semantic.HasFlightBounds);
        Assert.True(semantic.HasVertexShading);
    }

    private static byte[] CreateAhdrPayload(uint version, uint verticesX, uint verticesY, uint chunksX, uint chunksY)
    {
        byte[] bytes = new byte[0x40];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(0x00, 4), version);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(0x04, 4), verticesX);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(0x08, 4), verticesY);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(0x0C, 4), chunksX);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(0x10, 4), chunksY);
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