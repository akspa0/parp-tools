using System.Numerics;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class LkToAlphaRoundTripTests
{
    [Fact]
    public void WriteAlphaWdt_UsesClientMainOrderAndMcnkSubchunkContract()
    {
        List<LkMcnkData> chunks = [];
        for (int index = 0; index < 256; index++)
        {
            int chunkX = index % 16;
            int chunkY = index / 16;
            chunks.Add(CreateChunk(chunkX, chunkY, 25f + index, 0.01f, flags: 0, withAlpha: false));
        }

        LkAdtData adt = new()
        {
            TileX = 3,
            TileY = 5,
            TextureNames = ["terrain_a.blp"],
            Chunks = chunks
        };

        AlphaTileData tile = LkToAlphaConverter.ConvertTile(adt, 3, 5);
        byte[] wdt = AlphaWdtWriter.Build("legacy_contract", new Dictionary<(int tileX, int tileY), AlphaTileData>
        {
            [(3, 5)] = tile
        });

        Assert.Contains((3, 5), AlphaWdtReader.ReadExistingTiles(wdt));
        Assert.DoesNotContain((5, 3), AlphaWdtReader.ReadExistingTiles(wdt));

        int mphdOffset = 12;
        Assert.Equal("MPHD", ReadChunkId(wdt, mphdOffset));
        Assert.Equal(128, BitConverter.ToInt32(wdt, mphdOffset + 4));

        int mainPayloadOffset = mphdOffset + 8 + 128 + 8;
        int rowMajorIndex = (5 * 64) + 3;
        int transposedIndex = (3 * 64) + 5;
        int adtOffset = BitConverter.ToInt32(wdt, mainPayloadOffset + rowMajorIndex * 16);
        int adtHeaderSize = BitConverter.ToInt32(wdt, mainPayloadOffset + rowMajorIndex * 16 + 4);

        Assert.True(adtOffset > 0);
        Assert.True(adtHeaderSize > 0 && adtHeaderSize < 0x28000);
        Assert.Equal(0, BitConverter.ToInt32(wdt, mainPayloadOffset + transposedIndex * 16));

        int mhdrDataOffset = adtOffset + 8;
        int mcinRelativeOffset = BitConverter.ToInt32(wdt, mhdrDataOffset + 0x00);
        int mddfRelativeOffset = BitConverter.ToInt32(wdt, mhdrDataOffset + 0x0C);
        int modfRelativeOffset = BitConverter.ToInt32(wdt, mhdrDataOffset + 0x14);
        int mcinOffset = mhdrDataOffset + mcinRelativeOffset;
        Assert.Equal("MCIN", ReadChunkId(wdt, mcinOffset));
        Assert.Equal("MDDF", ReadChunkId(wdt, mhdrDataOffset + mddfRelativeOffset));
        Assert.Equal(0, BitConverter.ToInt32(wdt, mhdrDataOffset + mddfRelativeOffset + 4));
        Assert.Equal("MODF", ReadChunkId(wdt, mhdrDataOffset + modfRelativeOffset));
        Assert.Equal(0, BitConverter.ToInt32(wdt, mhdrDataOffset + modfRelativeOffset + 4));

        for (int chunkIndex = 0; chunkIndex < 256; chunkIndex++)
        {
            int entryOffset = mcinOffset + 8 + chunkIndex * 16;
            int mcnkOffset = BitConverter.ToInt32(wdt, entryOffset);
            int mcinSize = BitConverter.ToInt32(wdt, entryOffset + 4);

            Assert.True(mcnkOffset > 0, $"Missing MCNK offset for chunk {chunkIndex}");
            Assert.Equal("MCNK", ReadChunkId(wdt, mcnkOffset));

            int mcnkSize = BitConverter.ToInt32(wdt, mcnkOffset + 4);
            Assert.Equal(mcnkSize + 8, mcinSize);

            int headerOffset = mcnkOffset + 8;
            int dataBase = headerOffset + 128;
            int dataEnd = mcnkOffset + 8 + mcnkSize;

            int mcvtOffset = BitConverter.ToInt32(wdt, headerOffset + 0x18);
            int mcnrOffset = BitConverter.ToInt32(wdt, headerOffset + 0x1C);
            int mclyOffset = BitConverter.ToInt32(wdt, headerOffset + 0x20);
            int mcrfOffset = BitConverter.ToInt32(wdt, headerOffset + 0x24);
            int mcalOffset = BitConverter.ToInt32(wdt, headerOffset + 0x28);
            int mcalSize = BitConverter.ToInt32(wdt, headerOffset + 0x2C);
            int mcnkChunksSize = BitConverter.ToInt32(wdt, headerOffset + 0x5C);

            Assert.True(dataBase + mcvtOffset + 580 <= dataEnd, $"MCVT overrun in chunk {chunkIndex}");
            Assert.True(dataBase + mcnrOffset + 448 <= dataEnd, $"MCNR overrun in chunk {chunkIndex}");
            Assert.Equal("MCLY", ReadChunkId(wdt, dataBase + mclyOffset));
            Assert.Equal("MCRF", ReadChunkId(wdt, dataBase + mcrfOffset));
            Assert.True(mcalOffset >= 0 && mcalOffset + mcalSize <= mcnkChunksSize, $"MCAL overrun in chunk {chunkIndex}");
        }
    }

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

        int mainChunkOffset = 12 + 8 + 128;
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

    [Fact]
    public void LkAdtWriter_RoundTripsModfBoundsWithReaderOrientation()
    {
        LkModfEntry placement = new(
            0,
            88,
            new Vector3(100f, 200f, 30f),
            new Vector3(1f, 2f, 3f),
            new Vector3(90f, 180f, 10f),
            new Vector3(110f, 220f, 50f),
            0x1234,
            7,
            8,
            1.5f);

        LkAdtData adt = new()
        {
            TextureNames = ["terrain_a.blp"],
            WorldModelNames = ["world.wmo"],
            WorldModelPlacements = [placement]
        };

        byte[] adtBytes = LkAdtWriter.Build(adt);
        using MemoryStream stream = new(adtBytes, writable: false);
        MapFileSummary summary = MapFileSummaryReader.Read(stream, "roundtrip_modf_0_0.adt");
        AdtPlacementCatalog catalog = AdtPlacementReader.Read(stream, summary);

        AdtWorldModelPlacement roundTrip = Assert.Single(catalog.WorldModelPlacements);
        Assert.Equal(placement.Position, roundTrip.Position);
        Assert.Equal(placement.Rotation, roundTrip.Rotation);
        Assert.Equal(placement.BoundsMin, roundTrip.BoundsMin);
        Assert.Equal(placement.BoundsMax, roundTrip.BoundsMax);
        Assert.Equal(placement.Flags, roundTrip.Flags);
    }

    [Fact]
    public void AlphaWdt_RoundTripsMddfAndModfPlacements()
    {
        var modelPlacement = new AlphaModelPlacement(
            0, "world\\azeroth\\tree.mdx", 42,
            new Vector3(17000f, 16900f, 45.5f),
            new Vector3(15f, 90f, 270f),
            1.0f);

        var worldModelPlacement = new AlphaWorldModelPlacement(
            0, "world\\azeroth\\barracks.wmo", 100,
            new Vector3(16950f, 16900f, 40.2f),
            new Vector3(30f, 45f, 180f),
            new Vector3(16800f, 16700f, 35f),
            new Vector3(17100f, 17100f, 50f),
            0x0001);

        float[,] heightmap = new float[257, 257];
        int[,,] texIds = new int[16, 16, 4];
        bool[,,] layerMask = new bool[16, 16, 4];
        for (int cy = 0; cy < 16; cy++)
            for (int cx = 0; cx < 16; cx++)
                layerMask[cx, cy, 0] = true;

        AlphaTileData tile = new(
            sourcePath: "placement_roundtrip",
            heightmap: heightmap,
            mcalAlphaPack: null,
            mclyTextureIds: texIds,
            mclyLayerMask: layerMask,
            holeMask: new bool[16, 16],
            modelPlacements: [modelPlacement],
            worldModelPlacements: [worldModelPlacement],
            liquidChunks: [],
            textureNames: ["terrain_a.blp"]);

        byte[] wdt = AlphaWdtWriter.Build("placement_roundtrip", new Dictionary<(int, int), AlphaTileData>
        {
            [(0, 0)] = tile
        });

        var refs = CountAlphaPlacementReferences(wdt, 0, 0);
        Assert.Equal(1, refs.DoodadRefs);
        Assert.True(refs.MapObjRefs > 0);

        Assert.True(AlphaWdtReader.TryReadTile(wdt, 0, 0, out AlphaTileData? roundTrip));
        Assert.NotNull(roundTrip);

        AlphaModelPlacement mddfResult = Assert.Single(roundTrip.ModelPlacements);
        Assert.Equal(modelPlacement.NameId, mddfResult.NameId);
        Assert.Equal(modelPlacement.UniqueId, mddfResult.UniqueId);
        Assert.Equal(modelPlacement.Position.X, mddfResult.Position.X, 2);
        Assert.Equal(modelPlacement.Position.Y, mddfResult.Position.Y, 2);
        Assert.Equal(modelPlacement.Position.Z, mddfResult.Position.Z, 2);
        Assert.Equal(modelPlacement.Rotation.X, mddfResult.Rotation.X, 2);
        Assert.Equal(modelPlacement.Rotation.Y, mddfResult.Rotation.Y, 2);
        Assert.Equal(modelPlacement.Rotation.Z, mddfResult.Rotation.Z, 2);
        Assert.Equal(modelPlacement.Scale, mddfResult.Scale, 3);

        AlphaWorldModelPlacement modfResult = Assert.Single(roundTrip.WorldModelPlacements);
        Assert.Equal(worldModelPlacement.NameId, modfResult.NameId);
        Assert.Equal(worldModelPlacement.UniqueId, modfResult.UniqueId);
        Assert.Equal(worldModelPlacement.Position.X, modfResult.Position.X, 2);
        Assert.Equal(worldModelPlacement.Position.Y, modfResult.Position.Y, 2);
        Assert.Equal(worldModelPlacement.Position.Z, modfResult.Position.Z, 2);
        Assert.Equal(worldModelPlacement.Rotation.X, modfResult.Rotation.X, 2);
        Assert.Equal(worldModelPlacement.Rotation.Y, modfResult.Rotation.Y, 2);
        Assert.Equal(worldModelPlacement.Rotation.Z, modfResult.Rotation.Z, 2);
        Assert.Equal(worldModelPlacement.BoundsMin.X, modfResult.BoundsMin.X, 2);
        Assert.Equal(worldModelPlacement.BoundsMin.Y, modfResult.BoundsMin.Y, 2);
        Assert.Equal(worldModelPlacement.BoundsMin.Z, modfResult.BoundsMin.Z, 2);
        Assert.Equal(worldModelPlacement.BoundsMax.X, modfResult.BoundsMax.X, 2);
        Assert.Equal(worldModelPlacement.BoundsMax.Y, modfResult.BoundsMax.Y, 2);
        Assert.Equal(worldModelPlacement.BoundsMax.Z, modfResult.BoundsMax.Z, 2);
        Assert.Equal(worldModelPlacement.Flags, modfResult.Flags);
    }

    private static (int DoodadRefs, int MapObjRefs) CountAlphaPlacementReferences(byte[] wdt, int tileX, int tileY)
    {
        int mainPayloadOffset = 12 + 8 + 128 + 8;
        int mainIndex = tileY * 64 + tileX;
        int adtOffset = BitConverter.ToInt32(wdt, mainPayloadOffset + mainIndex * 16);
        int mhdrDataOffset = adtOffset + 8;
        int mcinRelativeOffset = BitConverter.ToInt32(wdt, mhdrDataOffset + 0x00);
        int mcinOffset = mhdrDataOffset + mcinRelativeOffset;

        int doodadRefs = 0;
        int mapObjRefs = 0;
        for (int chunkIndex = 0; chunkIndex < 256; chunkIndex++)
        {
            int entryOffset = mcinOffset + 8 + chunkIndex * 16;
            int mcnkOffset = BitConverter.ToInt32(wdt, entryOffset);
            if (mcnkOffset <= 0)
                continue;

            int headerOffset = mcnkOffset + 8;
            doodadRefs += BitConverter.ToInt32(wdt, headerOffset + 0x14);
            mapObjRefs += BitConverter.ToInt32(wdt, headerOffset + 0x3C);
        }

        return (doodadRefs, mapObjRefs);
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

    private static string ReadChunkId(byte[] data, int offset)
    {
        return new string(new[]
        {
            (char)data[offset + 3],
            (char)data[offset + 2],
            (char)data[offset + 1],
            (char)data[offset]
        });
    }
}
