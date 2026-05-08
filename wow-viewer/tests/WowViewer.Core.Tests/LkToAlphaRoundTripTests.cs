using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class LkToAlphaRoundTripTests
{
    [Fact]
    public void ConvertTile_AndWriteAlphaWdt_RoundTripsChunkHeightsAlphaAndLiquid()
    {
        List<LkMcnkData> chunks = [];
        for (int index = 0; index < 256; index++)
        {
            int chunkX = index % 16;
            int chunkY = index / 16;
            chunks.Add(new LkMcnkData
            {
                IndexX = chunkX,
                IndexY = chunkY,
                Flags = 0,
                BaseHeight = 0f,
                Heights = [],
                Normals = [],
                Layers = []
            });
        }

        chunks[0] = CreateChunk(0, 0, 10f, 0f, flags: 0, withAlpha: false);
        chunks[(7 * 16) + 5] = CreateChunk(5, 7, 120f, 0.25f, flags: 0, withAlpha: false);
        chunks[(12 * 16) + 10] = CreateChunk(10, 12, 80f, 0.5f, flags: 0, withAlpha: true);
        chunks[(4 * 16) + 3] = CreateChunk(3, 4, 33f, 0f, flags: 0x04, withAlpha: false);

        LkAdtData adt = new()
        {
            TileX = 0,
            TileY = 0,
            TextureNames = ["terrain_a.blp", "terrain_b.blp"],
            Chunks = chunks
        };

        AlphaTileData tile = LkToAlphaConverter.ConvertTile(adt, 0, 0);
        byte[] wdt = AlphaWdtWriter.Build("roundtrip", new Dictionary<(int tileX, int tileY), AlphaTileData>
        {
            [(0, 0)] = tile
        });

        Assert.True(AlphaWdtReader.IsAlphaWdt(wdt));
        Assert.Contains((0, 0), AlphaWdtReader.ReadExistingTiles(wdt));

        int mainChunkOffset = 36;
        int adtOffset = BitConverter.ToInt32(wdt, mainChunkOffset + 8);
        int mcinRelativeOffset = BitConverter.ToInt32(wdt, adtOffset + 8 + 0x00);
        int mtexRelativeOffset = BitConverter.ToInt32(wdt, adtOffset + 8 + 0x04);
        int firstMcinOffset = BitConverter.ToInt32(wdt, adtOffset + 8 + mcinRelativeOffset + 8);

        bool success = AlphaWdtReader.TryReadTile(wdt, 0, 0, out AlphaTileData? roundTrip);
        Assert.True(success, $"adtOffset={adtOffset} mcinRel={mcinRelativeOffset} mtexRel={mtexRelativeOffset} firstMcin={firstMcinOffset}");
        Assert.NotNull(roundTrip);

        Assert.Equal(tile.Heightmap[7 * 16, 5 * 16], roundTrip.Heightmap[7 * 16, 5 * 16], 3);
        Assert.Equal(tile.Heightmap[(7 * 16) + 1, (5 * 16) + 1], roundTrip.Heightmap[(7 * 16) + 1, (5 * 16) + 1], 3);

        Assert.NotNull(roundTrip.McalAlphaPack);
        Assert.True(roundTrip.McalAlphaPack![12 * 16 + 8, 10 * 16 + 8, 1] > 0.90f);

        AlphaLiquidChunk liquidChunk = Assert.Single(roundTrip.LiquidChunks);
        Assert.Equal(3, liquidChunk.IndexX);
        Assert.Equal(4, liquidChunk.IndexY);
        Assert.Equal(33f, liquidChunk.MinHeight, 3);
        Assert.Equal(33f, liquidChunk.MaxHeight, 3);
        Assert.Equal(0x04u, liquidChunk.McnkFlags & 0x3Cu);
    }

    [Fact]
    public void ConvertTile_ThroughAlphaWdt_BackToLkAdt_RoundTripsLiquidIntoMh2o()
    {
        List<LkMcnkData> chunks = [];
        for (int index = 0; index < 256; index++)
        {
            int chunkX = index % 16;
            int chunkY = index / 16;
            chunks.Add(new LkMcnkData
            {
                IndexX = chunkX,
                IndexY = chunkY,
                Flags = 0,
                BaseHeight = 0f,
                Heights = [],
                Normals = [],
                Layers = []
            });
        }

        float[] liquidHeights = new float[81];
        for (int index = 0; index < liquidHeights.Length; index++)
            liquidHeights[index] = 40f + (index * 0.125f);

        chunks[(4 * 16) + 3] = CreateChunk(
            3,
            4,
            33f,
            0f,
            flags: 0x04,
            withAlpha: false,
            liquidData: new AdtLiquidChunk(
                (4 * 16) + 3,
                null,
                null,
                [new AdtLiquidLayer(
                    0,
                    AdtLiquidBasicType.Water,
                    AdtLiquidVertexFormat.HeightDepth,
                    liquidHeights.Min(),
                    liquidHeights.Max(),
                    0,
                    0,
                    8,
                    8,
                    null,
                    liquidHeights,
                    new byte[81],
                    null)]));

        LkAdtData adt = new()
        {
            TileX = 0,
            TileY = 0,
            TextureNames = ["terrain_a.blp"],
            Chunks = chunks
        };

        AlphaTileData alphaTile = LkToAlphaConverter.ConvertTile(adt, 0, 0);
        byte[] wdt = AlphaWdtWriter.Build("roundtrip_liquid", new Dictionary<(int tileX, int tileY), AlphaTileData>
        {
            [(0, 0)] = alphaTile
        });

        Assert.True(AlphaWdtReader.TryReadTile(wdt, 0, 0, out AlphaTileData? alphaRoundTrip));
        Assert.NotNull(alphaRoundTrip);

        AlphaLiquidChunk alphaLiquid = Assert.Single(alphaRoundTrip.LiquidChunks);
        Assert.NotNull(alphaLiquid.Heights);
        Assert.Equal(81, alphaLiquid.Heights!.Length);
        Assert.Equal(liquidHeights[0], alphaLiquid.Heights[0], 3);
        Assert.Equal(liquidHeights[80], alphaLiquid.Heights[80], 3);

        LkAdtData lkRoundTrip = AlphaToLkConverter.ConvertTile(alphaRoundTrip, 0, 0);
        byte[] adtBytes = LkAdtWriter.Build(lkRoundTrip);

        using var stream = new MemoryStream(adtBytes, writable: false);
        MapFileSummary summary = MapFileSummaryReader.Read(stream, "roundtrip_liquid_0_0.adt");
        AdtSummary adtSummary = AdtSummaryReader.Read(stream, summary);
        AdtLiquidFile liquidFile = AdtLiquidReader.Read(stream, summary);

        Assert.Equal(MapFileKind.Adt, summary.Kind);
        Assert.True(adtSummary.HasWater);
        Assert.Equal(256, adtSummary.TerrainChunkCount);

        AdtLiquidChunk mh2oChunk = Assert.Single(liquidFile.Chunks, static chunk => chunk.Layers.Count > 0);
        Assert.Equal((4 * 16) + 3, mh2oChunk.ChunkIndex);

        AdtLiquidLayer mh2oLayer = Assert.Single(mh2oChunk.Layers);
        Assert.Equal(8, mh2oLayer.Width);
        Assert.Equal(8, mh2oLayer.Height);
        Assert.NotNull(mh2oLayer.Heights);
        Assert.Equal(81, mh2oLayer.Heights!.Length);
        Assert.Equal(liquidHeights[0], mh2oLayer.Heights[0], 3);
        Assert.Equal(liquidHeights[80], mh2oLayer.Heights[80], 3);
    }

    private static LkMcnkData CreateChunk(int chunkX, int chunkY, float baseHeight, float slope, int flags, bool withAlpha, AdtLiquidChunk? liquidData = null)
    {
        float[] heights = new float[145];
        for (int index = 0; index < heights.Length; index++)
            heights[index] = slope * index;

        List<LkMclyEntry> layers =
        [
            new LkMclyEntry(0, 0, 0, 0)
        ];

        byte[]? alphaMapData = null;
        if (withAlpha)
        {
            layers.Add(new LkMclyEntry(1, 0x200, 0, 0));
            alphaMapData = new byte[64 * 64];
            Array.Fill(alphaMapData, (byte)255);
        }

        return new LkMcnkData
        {
            IndexX = chunkX,
            IndexY = chunkY,
            Flags = flags,
            BaseHeight = baseHeight,
            Heights = heights,
            Normals = [],
            Layers = layers,
            AlphaMapData = alphaMapData,
            AlphaMapSize = alphaMapData?.Length ?? 0,
            LiquidData = liquidData
        };
    }
}