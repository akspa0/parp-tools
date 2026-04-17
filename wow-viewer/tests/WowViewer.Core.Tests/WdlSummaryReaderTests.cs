using System.Buffers.Binary;
using System.Text;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class WdlSummaryReaderTests
{
    [Fact]
    public void Read_DevelopmentWdl_ProducesExpectedTileSignals()
    {
        WdlSummary summary = WdlSummaryReader.Read(MapTestPaths.DevelopmentWdlPath);

        Assert.Equal((uint)18, summary.Version);
        Assert.True(summary.TileCount > 0);
        Assert.True(summary.TryGetTile(0, 0, out WdlTileSummary? tile));
        Assert.NotNull(tile);
        Assert.Equal(0, tile!.TileX);
        Assert.Equal(0, tile.TileY);
        Assert.Equal(WdlTileSummary.OuterHeightCount, tile.OuterHeights.Length);
        Assert.Equal(WdlTileSummary.InnerHeightCount, tile.InnerHeights.Length);
        Assert.True(tile.MinHeight <= tile.MaxHeight);
    }

    [Fact]
    public void Read_SyntheticReadableChunkTags_ProducesExpectedHeights()
    {
        byte[] bytes = WdlTestDataBuilder.CreateSingleTileWdl(tileX: 5, tileY: 7, useReadableTags: true, startHeight: 10);

        using MemoryStream stream = new(bytes, writable: false);
        WdlSummary summary = WdlSummaryReader.Read(stream, "synthetic.wdl");

        Assert.Equal((uint)18, summary.Version);
        Assert.Equal(1, summary.TileCount);
        Assert.True(summary.TryGetTile(5, 7, out WdlTileSummary? tile));
        Assert.NotNull(tile);
        Assert.Equal((short)10, tile!.GetOuterHeight(0, 0));
        Assert.Equal((short)298, tile.GetOuterHeight(16, 16));
        Assert.Equal((short)299, tile.GetInnerHeight(0, 0));
        Assert.Equal((short)554, tile.GetInnerHeight(15, 15));
        Assert.Equal((short)10, tile.MinHeight);
        Assert.Equal((short)554, tile.MaxHeight);
    }
}

internal static class WdlTestDataBuilder
{
    private const int GridSize = 64;
    private const int ChunkHeaderBytes = 8;

    public static byte[] CreateSingleTileWdl(int tileX, int tileY, bool useReadableTags, short startHeight)
    {
        byte[] mver = CreateChunk(useReadableTags ? "MVER" : "REVM", CreateUInt32Payload(18));

        byte[] maofPayload = new byte[GridSize * GridSize * sizeof(uint)];
        uint mareOffset = (uint)(mver.Length + ChunkHeaderBytes + maofPayload.Length);
        BinaryPrimitives.WriteUInt32LittleEndian(maofPayload.AsSpan(((tileY * GridSize) + tileX) * sizeof(uint), sizeof(uint)), mareOffset);

        byte[] maof = CreateChunk(useReadableTags ? "MAOF" : "FOAM", maofPayload);
        byte[] mare = CreateChunk(useReadableTags ? "MARE" : "ERAM", CreateMarePayload(startHeight));
        return [.. mver, .. maof, .. mare];
    }

    private static byte[] CreateChunk(string tag, byte[] payload)
    {
        byte[] bytes = new byte[ChunkHeaderBytes + payload.Length];
        Encoding.ASCII.GetBytes(tag, bytes.AsSpan(0, 4));
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4, 4), (uint)payload.Length);
        payload.CopyTo(bytes.AsSpan(ChunkHeaderBytes));
        return bytes;
    }

    private static byte[] CreateUInt32Payload(uint value)
    {
        byte[] bytes = new byte[sizeof(uint)];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes, value);
        return bytes;
    }

    private static byte[] CreateMarePayload(short startHeight)
    {
        short[] heights = Enumerable.Range(0, WdlTileSummary.OuterHeightCount + WdlTileSummary.InnerHeightCount)
            .Select(index => checked((short)(startHeight + index)))
            .ToArray();

        byte[] bytes = new byte[heights.Length * sizeof(short)];
        for (int index = 0; index < heights.Length; index++)
            BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(index * sizeof(short), sizeof(short)), heights[index]);

        return bytes;
    }
}