using System.Numerics;
using WowViewer.Core.IO.Dbc;
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
        Assert.Equal(0x3Cu, liquidChunk.McnkFlags & 0x3Cu);
        Assert.NotNull(liquidChunk.TileFlags);
        Assert.Equal(0x01, liquidChunk.TileFlags![0] & 0x0F);
    }

    [Fact]
    public void ConvertTile_CollapsesPreservedDoodadRefsToContainingChunkWhileKeepingWorldRefs()
    {
        List<LkMcnkData> chunks = [];
        for (int index = 0; index < 256; index++)
        {
            int chunkX = index % 16;
            int chunkY = index / 16;
            IReadOnlyList<int> doodadRefs = index == 37 ? [0] : [];
            IReadOnlyList<int> worldModelRefs = index == 115 ? [0] : [];
            chunks.Add(CreateChunk(chunkX, chunkY, 25f + index, 0.01f, flags: 0, withAlpha: false, doodadRefs: doodadRefs, worldModelRefs: worldModelRefs));
        }

        const float mapOrigin = 17066.666f;
        const float chunkWorldSize = 533.3333f / 16f;
        Vector3 placementPosition = new(mapOrigin - (0.5f * chunkWorldSize), mapOrigin - (0.5f * chunkWorldSize), 50f);

        LkAdtData adt = new()
        {
            TileX = 0,
            TileY = 0,
            TextureNames = ["terrain_a.blp"],
            ModelNames = ["test.mdx"],
            WorldModelNames = ["test.wmo"],
            ModelPlacements = [new LkMddfEntry(0, 111, placementPosition, Vector3.Zero, 1f)],
            WorldModelPlacements = [new LkModfEntry(0, 222, placementPosition, Vector3.Zero, placementPosition - Vector3.One, placementPosition + Vector3.One, 0, 0, 0, 1f)],
            Chunks = chunks
        };

        AlphaTileData tile = LkToAlphaConverter.ConvertTile(adt, 0, 0);
        byte[] wdt = AlphaWdtWriter.Build("preserve_mcrf", new Dictionary<(int tileX, int tileY), AlphaTileData>
        {
            [(0, 0)] = tile
        });

        var chunk37Refs = ReadMcrfRefs(wdt, 0, 0, 37);
        Assert.Empty(chunk37Refs.DoodadRefs);
        Assert.Empty(chunk37Refs.WorldModelRefs);

        var chunk115Refs = ReadMcrfRefs(wdt, 0, 0, 115);
        Assert.Empty(chunk115Refs.DoodadRefs);
        Assert.Equal([0], chunk115Refs.WorldModelRefs);

        var chunk0Refs = ReadMcrfRefs(wdt, 0, 0, 0);
        Assert.Equal([0], chunk0Refs.DoodadRefs);
        Assert.Empty(chunk0Refs.WorldModelRefs);

        int totalDoodadRefs = 0;
        for (int chunkIndex = 0; chunkIndex < 256; chunkIndex++)
            totalDoodadRefs += ReadMcrfRefs(wdt, 0, 0, chunkIndex).DoodadRefs.Length;

        Assert.Equal(1, totalDoodadRefs);
    }

    [Fact]
    public void ConvertTile_WithEmptyPreservedChunkRefs_FallsBackToHeuristicMcrfPopulation()
    {
        List<LkMcnkData> chunks = [];
        for (int index = 0; index < 256; index++)
        {
            int chunkX = index % 16;
            int chunkY = index / 16;
            chunks.Add(CreateChunk(chunkX, chunkY, 25f + index, 0.01f, flags: 0, withAlpha: false));
        }

        const float mapOrigin = 17066.666f;
        const float chunkWorldSize = 533.3333f / 16f;
        Vector3 placementPosition = new(mapOrigin - (0.5f * chunkWorldSize), mapOrigin - (0.5f * chunkWorldSize), 50f);

        LkAdtData adt = new()
        {
            TileX = 0,
            TileY = 0,
            TextureNames = ["terrain_a.blp"],
            ModelNames = ["test.mdx"],
            ModelPlacements = [new LkMddfEntry(0, 111, placementPosition, Vector3.Zero, 1f)],
            Chunks = chunks
        };

        AlphaTileData tile = LkToAlphaConverter.ConvertTile(adt, 0, 0);
        byte[] wdt = AlphaWdtWriter.Build("heuristic_mcrf_fallback", new Dictionary<(int tileX, int tileY), AlphaTileData>
        {
            [(0, 0)] = tile
        });

        int chunksWithRefs = 0;
        for (int chunkIndex = 0; chunkIndex < 256; chunkIndex++)
        {
            var refs = ReadMcrfRefs(wdt, 0, 0, chunkIndex);
            if (refs.DoodadRefs.Length > 0 || refs.WorldModelRefs.Length > 0)
                chunksWithRefs++;
        }

        Assert.True(chunksWithRefs > 0);
    }

    [Fact]
    public void ConvertTile_MapsPostAlphaAreaIdsBackToAlphaUsingCrosswalk()
    {
        string tempDirectory = Path.Combine(Path.GetTempPath(), $"wowviewer-area-crosswalk-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDirectory);

        try
        {
            File.WriteAllText(
                Path.Combine(tempDirectory, "crosswalk.csv"),
                "src_mapId,src_areaId,src_parentId,src_isZone,src_name,match_count,matches\n" +
                "0,500,0,1,Dun Morogh,1,0:77:1:map_name:active:Anvilmar\n");

            AreaIdMapper mapper = new();
            mapper.LoadCrosswalkCsv(tempDirectory);

            LkAdtData adt = new()
            {
                TileX = 0,
                TileY = 0,
                TextureNames = ["terrain_a.blp"],
                Chunks = [CreateChunk(0, 0, 10f, 0f, flags: 0, withAlpha: false, areaId: 77)]
            };

            AlphaTileData tile = LkToAlphaConverter.ConvertTile(adt, 0, 0, mapper, "Azeroth");

            Assert.NotNull(tile.AreaIds);
            Assert.Equal(500, tile.AreaIds![0, 0]);
        }
        finally
        {
            Directory.Delete(tempDirectory, recursive: true);
        }
    }

    [Fact]
    public void ConvertTile_ThroughAlphaWdt_BackToLkAdt_PreservesChunkFlagsAndMapsAreaIdsToLk()
    {
        string tempDirectory = Path.Combine(Path.GetTempPath(), $"wowviewer-alpha-to-lk-crosswalk-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDirectory);

        try
        {
            File.WriteAllText(
                Path.Combine(tempDirectory, "crosswalk.csv"),
                "src_mapId,src_areaId,src_parentId,src_isZone,src_name,match_count,matches\n" +
                "0,500,0,1,Dun Morogh,1,0:77:1:map_name:active:Anvilmar\n");

            AreaIdMapper mapper = new();
            mapper.LoadCrosswalkCsv(tempDirectory);

            List<LkMcnkData> chunks = [];
            for (int index = 0; index < 256; index++)
            {
                int chunkX = index % 16;
                int chunkY = index / 16;
                chunks.Add(CreateChunk(chunkX, chunkY, 25f + index, 0.01f, flags: 0, withAlpha: false));
            }

            chunks[(1 * 16) + 2] = CreateChunk(2, 1, 42f, 0.02f, flags: 0x40, withAlpha: false, areaId: 500);

            LkAdtData adt = new()
            {
                TileX = 0,
                TileY = 0,
                TextureNames = ["terrain_a.blp"],
                Chunks = chunks
            };

            AlphaTileData alphaTile = LkToAlphaConverter.ConvertTile(adt, 0, 0);
            byte[] wdt = AlphaWdtWriter.Build("area_flag_roundtrip", new Dictionary<(int tileX, int tileY), AlphaTileData>
            {
                [(0, 0)] = alphaTile
            });

            Assert.True(AlphaWdtReader.TryReadTile(wdt, 0, 0, out AlphaTileData? alphaRoundTrip));
            Assert.NotNull(alphaRoundTrip);
            Assert.NotNull(alphaRoundTrip.AreaIds);
            Assert.NotNull(alphaRoundTrip.McnkFlagsByChunk);
            Assert.Equal(500, alphaRoundTrip.AreaIds![2, 1]);
            Assert.Equal(0x40u, alphaRoundTrip.McnkFlagsByChunk![2, 1] & 0x40u);

            LkAdtData lkRoundTrip = AlphaToLkConverter.ConvertTile(alphaRoundTrip, 0, 0, mapper, "Azeroth");
            LkMcnkData roundTripChunk = lkRoundTrip.Chunks[(1 * 16) + 2];

            Assert.Equal(77, roundTripChunk.AreaId);
            Assert.Equal(0x40, roundTripChunk.Flags & 0x40);
        }
        finally
        {
            Directory.Delete(tempDirectory, recursive: true);
        }
    }

    [Fact]
    public void WriteAlphaWdt_RemapPreservedChunkRefsByUniqueIdAgainstFilteredPlacementTables()
    {
        float[,] heightmap = new float[257, 257];
        int[,,] texIds = new int[16, 16, 4];
        bool[,,] layerMask = new bool[16, 16, 4];
        for (int cy = 0; cy < 16; cy++)
            for (int cx = 0; cx < 16; cx++)
                layerMask[cx, cy, 0] = true;

        IReadOnlyList<int>[] doodadUniqueIdsByChunk = CreateEmptyChunkRefSet();
        IReadOnlyList<int>[] worldUniqueIdsByChunk = CreateEmptyChunkRefSet();
        IReadOnlyList<int>[] staleDoodadRefsByChunk = CreateEmptyChunkRefSet();
        IReadOnlyList<int>[] staleWorldRefsByChunk = CreateEmptyChunkRefSet();
        doodadUniqueIdsByChunk[37] = [222];
        worldUniqueIdsByChunk[115] = [333];
        staleDoodadRefsByChunk[37] = [7];
        staleWorldRefsByChunk[115] = [9];

        AlphaTileData tile = new(
            sourcePath: "filtered_ref_remap",
            heightmap: heightmap,
            mcalAlphaPack: null,
            mclyTextureIds: texIds,
            mclyLayerMask: layerMask,
            holeMask: new bool[16, 16],
            textureNames: ["terrain_a.blp"],
            modelPlacements:
            [
                new AlphaModelPlacement(0, "world\\azeroth\\tree.mdx", 222, new Vector3(17000f, 16900f, 45.5f), Vector3.Zero, 1.0f)
            ],
            worldModelPlacements:
            [
                new AlphaWorldModelPlacement(0, "world\\azeroth\\barracks.wmo", 333, new Vector3(16950f, 16900f, 40.2f), Vector3.Zero, new Vector3(16800f, 16700f, 35f), new Vector3(17100f, 17100f, 50f), 0x0001)
            ],
            liquidChunks: [],
            mcrfDoodadRefsByChunk: staleDoodadRefsByChunk,
            mcrfWorldModelRefsByChunk: staleWorldRefsByChunk,
            mcrfDoodadUniqueIdsByChunk: doodadUniqueIdsByChunk,
            mcrfWorldModelUniqueIdsByChunk: worldUniqueIdsByChunk);

        byte[] wdt = AlphaWdtWriter.Build("filtered_ref_remap", new Dictionary<(int tileX, int tileY), AlphaTileData>
        {
            [(0, 0)] = tile
        });

        var chunk37Refs = ReadMcrfRefs(wdt, 0, 0, 37);
        Assert.Equal([0], chunk37Refs.DoodadRefs);
        Assert.Empty(chunk37Refs.WorldModelRefs);

        var chunk115Refs = ReadMcrfRefs(wdt, 0, 0, 115);
        Assert.Empty(chunk115Refs.DoodadRefs);
        Assert.Equal([0], chunk115Refs.WorldModelRefs);

        int totalDoodadRefs = 0;
        for (int chunkIndex = 0; chunkIndex < 256; chunkIndex++)
            totalDoodadRefs += ReadMcrfRefs(wdt, 0, 0, chunkIndex).DoodadRefs.Length;

        Assert.Equal(1, totalDoodadRefs);
    }

    [Fact]
    public void WriteAlphaWdt_PreservesDoodadChunkOwnershipFromSourceRefs()
    {
        float[,] heightmap = new float[257, 257];
        int[,,] texIds = new int[16, 16, 4];
        bool[,,] layerMask = new bool[16, 16, 4];
        for (int cy = 0; cy < 16; cy++)
            for (int cx = 0; cx < 16; cx++)
                layerMask[cx, cy, 0] = true;

        IReadOnlyList<int>[] doodadUniqueIdsByChunk = CreateEmptyChunkRefSet();
        IReadOnlyList<int>[] doodadRefsByChunk = CreateEmptyChunkRefSet();
        doodadUniqueIdsByChunk[37] = [222];
        doodadUniqueIdsByChunk[115] = [222];
        doodadRefsByChunk[37] = [0, 0];
        doodadRefsByChunk[115] = [0];

        const float mapOrigin = 17066.666f;
        const float chunkWorldSize = 533.3333f / 16f;
        Vector3 placementPosition = new(mapOrigin - (0.5f * chunkWorldSize), mapOrigin - (0.5f * chunkWorldSize), 50f);

        AlphaTileData tile = new(
            sourcePath: "single_doodad_owner",
            heightmap: heightmap,
            mcalAlphaPack: null,
            mclyTextureIds: texIds,
            mclyLayerMask: layerMask,
            holeMask: new bool[16, 16],
            textureNames: ["terrain_a.blp"],
            modelPlacements:
            [
                new AlphaModelPlacement(0, "world\\azeroth\\tree.mdx", 222, placementPosition, Vector3.Zero, 1.0f)
            ],
            worldModelPlacements: [],
            liquidChunks: [],
            mcrfDoodadRefsByChunk: doodadRefsByChunk,
            mcrfDoodadUniqueIdsByChunk: doodadUniqueIdsByChunk);

        byte[] wdt = AlphaWdtWriter.Build("single_doodad_owner", new Dictionary<(int tileX, int tileY), AlphaTileData>
        {
            [(0, 0)] = tile
        });

        var chunk0Refs = ReadMcrfRefs(wdt, 0, 0, 0);
        Assert.Equal([0], chunk0Refs.DoodadRefs);

        var chunk37Refs = ReadMcrfRefs(wdt, 0, 0, 37);
        Assert.Empty(chunk37Refs.DoodadRefs);

        var chunk115Refs = ReadMcrfRefs(wdt, 0, 0, 115);
        Assert.Empty(chunk115Refs.DoodadRefs);

        int totalDoodadRefs = 0;
        for (int chunkIndex = 0; chunkIndex < 256; chunkIndex++)
            totalDoodadRefs += ReadMcrfRefs(wdt, 0, 0, chunkIndex).DoodadRefs.Length;

        Assert.Equal(1, totalDoodadRefs);
    }

    [Fact]
    public void WriteAlphaWdt_TrimsDuplicateWorldRefsToRespectAlphaClientMcnkLimit()
    {
        float[,] heightmap = new float[257, 257];
        int[,,] texIds = new int[16, 16, 4];
        bool[,,] layerMask = new bool[16, 16, 4];
        for (int cy = 0; cy < 16; cy++)
            for (int cx = 0; cx < 16; cx++)
                layerMask[cx, cy, 0] = true;

        const float mapOrigin = 17066.666f;
        const float tileWorldSize = 533.33333f;
        const float chunkWorldSize = tileWorldSize / 16f;

        const int worldModelCount = 4096;
        List<AlphaWorldModelPlacement> placements = new(worldModelCount);
        for (int index = 0; index < worldModelCount; index++)
        {
            int anchorChunk = (index % 255) + 1;
            int anchorCx = anchorChunk % 16;
            int anchorCy = anchorChunk / 16;

            float chunkMaxX = mapOrigin - anchorCy * chunkWorldSize;
            float chunkMinX = chunkMaxX - chunkWorldSize;
            float chunkMaxY = mapOrigin - anchorCx * chunkWorldSize;
            float chunkMinY = chunkMaxY - chunkWorldSize;
            Vector3 position = new((chunkMinX + chunkMaxX) * 0.5f, (chunkMinY + chunkMaxY) * 0.5f, 50f);

            placements.Add(new AlphaWorldModelPlacement(
                NameId: 0,
                ModelPath: "world\\azeroth\\stormwind_keep.wmo",
                UniqueId: index + 1,
                Position: position,
                Rotation: Vector3.Zero,
                BoundsMin: new Vector3(mapOrigin - tileWorldSize - 10f, mapOrigin - tileWorldSize - 10f, 0f),
                BoundsMax: new Vector3(mapOrigin + 10f, mapOrigin + 10f, 100f),
                Flags: 0x0001));
        }

        AlphaTileData tile = new(
            sourcePath: "mcnk_world_ref_budget",
            heightmap: heightmap,
            mcalAlphaPack: null,
            mclyTextureIds: texIds,
            mclyLayerMask: layerMask,
            holeMask: new bool[16, 16],
            textureNames: ["terrain_a.blp"],
            modelPlacements: [],
            worldModelPlacements: placements,
            liquidChunks: []);

        byte[] wdt = AlphaWdtWriter.Build("mcnk_world_ref_budget", new Dictionary<(int tileX, int tileY), AlphaTileData>
        {
            [(0, 0)] = tile
        });

        int chunk0DeclaredSize = ReadMcnkDeclaredSize(wdt, 0, 0, 0);
        var chunk0Refs = ReadMcrfRefs(wdt, 0, 0, 0);

        Assert.True(chunk0DeclaredSize < 15000, $"Expected MCNK payload below Alpha client limit, found {chunk0DeclaredSize}.");
        Assert.True(chunk0Refs.WorldModelRefs.Length < worldModelCount, "Expected duplicate WMO refs to be trimmed for the stressed chunk.");

        int totalWorldRefs = 0;
        for (int chunkIndex = 0; chunkIndex < 256; chunkIndex++)
            totalWorldRefs += ReadMcrfRefs(wdt, 0, 0, chunkIndex).WorldModelRefs.Length;

        Assert.True(totalWorldRefs >= worldModelCount, "Expected every WMO to keep at least one owning chunk after trimming.");
    }

    [Fact]
    public void WriteAlphaWdt_ReassignsAdjacentDoodadOwnersWhenChunkWouldExceedAlphaClientLimit()
    {
        float[,] heightmap = new float[257, 257];
        int[,,] texIds = new int[16, 16, 4];
        bool[,,] layerMask = new bool[16, 16, 4];
        for (int cy = 0; cy < 16; cy++)
            for (int cx = 0; cx < 16; cx++)
                layerMask[cx, cy, 0] = true;

        const float mapOrigin = 17066.666f;
        const float tileWorldSize = 533.33333f;
        const float chunkWorldSize = tileWorldSize / 16f;
        float chunk0MaxX = mapOrigin;
        float chunk0MinX = chunk0MaxX - chunkWorldSize;
        float chunk0MaxY = mapOrigin;
        float chunk0MinY = chunk0MaxY - chunkWorldSize;

        const int doodadCount = 4000;
        List<AlphaModelPlacement> placements = new(doodadCount);
        for (int index = 0; index < doodadCount; index++)
        {
            float xBias = (index % 5) * 0.25f;
            float yBias = ((index / 5) % 5) * 0.25f;
            placements.Add(new AlphaModelPlacement(
                NameId: 0,
                ModelPath: "world\\azeroth\\tree.mdx",
                UniqueId: index + 1,
                Position: new Vector3(chunk0MinX + 1.5f + xBias, chunk0MinY + 1.5f + yBias, 40f),
                Rotation: Vector3.Zero,
                Scale: 1.0f));
        }

        AlphaTileData tile = new(
            sourcePath: "mcnk_doodad_spill_budget",
            heightmap: heightmap,
            mcalAlphaPack: null,
            mclyTextureIds: texIds,
            mclyLayerMask: layerMask,
            holeMask: new bool[16, 16],
            textureNames: ["terrain_a.blp"],
            modelPlacements: placements,
            worldModelPlacements: [],
            liquidChunks: []);

        byte[] wdt = AlphaWdtWriter.Build("mcnk_doodad_spill_budget", new Dictionary<(int tileX, int tileY), AlphaTileData>
        {
            [(0, 0)] = tile
        });

        int chunk0DeclaredSize = ReadMcnkDeclaredSize(wdt, 0, 0, 0);
        var chunk0Refs = ReadMcrfRefs(wdt, 0, 0, 0);
        var chunk1Refs = ReadMcrfRefs(wdt, 0, 0, 1);
        var chunk16Refs = ReadMcrfRefs(wdt, 0, 0, 16);
        var chunk17Refs = ReadMcrfRefs(wdt, 0, 0, 17);

        Assert.True(chunk0DeclaredSize < 15000, $"Expected MCNK payload below Alpha client limit, found {chunk0DeclaredSize}.");
        Assert.True(chunk0Refs.DoodadRefs.Length < doodadCount, "Expected overloaded doodad ownership to spill into adjacent chunks.");
        Assert.True(chunk1Refs.DoodadRefs.Length > 0 || chunk16Refs.DoodadRefs.Length > 0 || chunk17Refs.DoodadRefs.Length > 0,
            "Expected at least one adjacent chunk to receive reassigned doodad ownership.");

        int totalDoodadRefs = 0;
        for (int chunkIndex = 0; chunkIndex < 256; chunkIndex++)
            totalDoodadRefs += ReadMcrfRefs(wdt, 0, 0, chunkIndex).DoodadRefs.Length;

        Assert.Equal(doodadCount, totalDoodadRefs);
    }

    [Fact]
    public void WriteAlphaWdt_PreservesSingleSourceChunkForDoodadOwnership()
    {
        float[,] heightmap = new float[257, 257];
        int[,,] texIds = new int[16, 16, 4];
        bool[,,] layerMask = new bool[16, 16, 4];
        for (int cy = 0; cy < 16; cy++)
            for (int cx = 0; cx < 16; cx++)
                layerMask[cx, cy, 0] = true;

        Vector3 placementPosition = new(17050f, 17050f, 42f);

        IReadOnlyList<int>[] doodadUniqueIdsByChunk = CreateEmptyChunkRefSet();
        doodadUniqueIdsByChunk[73] = [444];

        AlphaTileData tile = new(
            sourcePath: "mcnk_doodad_ref_budget",
            heightmap: heightmap,
            mcalAlphaPack: null,
            mclyTextureIds: texIds,
            mclyLayerMask: layerMask,
            holeMask: new bool[16, 16],
            textureNames: ["terrain_a.blp"],
            modelPlacements:
            [
                new AlphaModelPlacement(0, "world\\azeroth\\tree.mdx", 444, placementPosition, Vector3.Zero, 1.0f)
            ],
            worldModelPlacements: [],
            liquidChunks: [],
            mcrfDoodadUniqueIdsByChunk: doodadUniqueIdsByChunk);

        byte[] wdt = AlphaWdtWriter.Build("mcnk_doodad_ref_budget", new Dictionary<(int tileX, int tileY), AlphaTileData>
        {
            [(0, 0)] = tile
        });

        var chunk73Refs = ReadMcrfRefs(wdt, 0, 0, 73);
        Assert.Empty(chunk73Refs.DoodadRefs);

        var chunk0Refs = ReadMcrfRefs(wdt, 0, 0, 0);
        Assert.Equal([0], chunk0Refs.DoodadRefs);

        int totalDoodadRefs = 0;
        for (int chunkIndex = 0; chunkIndex < 256; chunkIndex++)
            totalDoodadRefs += ReadMcrfRefs(wdt, 0, 0, chunkIndex).DoodadRefs.Length;

        Assert.Equal(1, totalDoodadRefs);
    }

    [Fact]
    public void ConvertTile_AndWriteAlphaWdt_UsesSingleFixedSizeLiquidBlock()
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
        Array.Fill(liquidHeights, 40f);
        chunks[(4 * 16) + 3] = CreateChunk(
            3,
            4,
            33f,
            0f,
            flags: 0x3C,
            withAlpha: false,
            liquidData: new AdtLiquidChunk(
                (4 * 16) + 3,
                null,
                null,
                [new AdtLiquidLayer(
                    0,
                    AdtLiquidBasicType.Water,
                    AdtLiquidVertexFormat.HeightDepth,
                    40f,
                    40f,
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

        AlphaTileData tile = LkToAlphaConverter.ConvertTile(adt, 0, 0);
        byte[] wdt = AlphaWdtWriter.Build("liquid_contract", new Dictionary<(int tileX, int tileY), AlphaTileData>
        {
            [(0, 0)] = tile
        });

        int mainChunkOffset = 12 + 8 + 128;
        int adtOffset = BitConverter.ToInt32(wdt, mainChunkOffset + 8);
        int mcinRelativeOffset = BitConverter.ToInt32(wdt, adtOffset + 8 + 0x00);
        int mcinOffset = adtOffset + 8 + mcinRelativeOffset;
        int chunkIndex = (4 * 16) + 3;
        int mcnkOffset = BitConverter.ToInt32(wdt, mcinOffset + 8 + (chunkIndex * 16));
        int headerOffset = mcnkOffset + 8;
        int chunkDataSize = BitConverter.ToInt32(wdt, headerOffset + 0x5C);
        int mclqOffset = BitConverter.ToInt32(wdt, headerOffset + 0x64);
        uint flags = BitConverter.ToUInt32(wdt, headerOffset + 0x00) & 0x3Cu;

        Assert.Equal(0x3Cu, flags);
        Assert.Equal(0x324, chunkDataSize - mclqOffset);
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
    public void ConvertTile_ThroughAlphaWdt_BackToLkAdt_PreservesOceanLiquidTypeViaTileFlags()
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
        Array.Fill(liquidHeights, 52f);

        chunks[(6 * 16) + 5] = CreateChunk(
            5,
            6,
            10f,
            0f,
            flags: 0x08,
            withAlpha: false,
            liquidData: new AdtLiquidChunk(
                (6 * 16) + 5,
                null,
                null,
                [new AdtLiquidLayer(
                    17,
                    AdtLiquidBasicType.Ocean,
                    AdtLiquidVertexFormat.HeightDepth,
                    52f,
                    52f,
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
        byte[] wdt = AlphaWdtWriter.Build("ocean_roundtrip_liquid", new Dictionary<(int tileX, int tileY), AlphaTileData>
        {
            [(0, 0)] = alphaTile
        });

        Assert.True(AlphaWdtReader.TryReadTile(wdt, 0, 0, out AlphaTileData? alphaRoundTrip));
        Assert.NotNull(alphaRoundTrip);

        AlphaLiquidChunk alphaLiquid = Assert.Single(alphaRoundTrip.LiquidChunks, static chunk => chunk.IndexX == 5 && chunk.IndexY == 6);
        Assert.NotNull(alphaLiquid.TileFlags);
        Assert.Equal(0x02, alphaLiquid.TileFlags![0] & 0x0F);
        Assert.Equal(0x3Cu, alphaLiquid.McnkFlags & 0x3Cu);

        LkAdtData lkRoundTrip = AlphaToLkConverter.ConvertTile(alphaRoundTrip, 0, 0);
        AdtLiquidChunk mh2oChunk = Assert.Single(lkRoundTrip.Chunks.Where(static chunk => chunk.LiquidData?.Layers.Count > 0).Select(static chunk => chunk.LiquidData!));
        AdtLiquidLayer layer = Assert.Single(mh2oChunk.Layers);

        Assert.Equal(AdtLiquidBasicType.Ocean, layer.BasicType);
        Assert.Equal((ushort)17, layer.LiquidTypeId);
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

    [Fact]
    public void AlphaWdt_WithOddSizedNameTable_WritesContiguousMonmHeader()
    {
        float[,] heightmap = new float[257, 257];
        int[,,] texIds = new int[16, 16, 4];
        bool[,,] layerMask = new bool[16, 16, 4];
        for (int cy = 0; cy < 16; cy++)
            for (int cx = 0; cx < 16; cx++)
                layerMask[cx, cy, 0] = true;

        AlphaTileData tile = new(
            sourcePath: "odd_name_table",
            heightmap: heightmap,
            mcalAlphaPack: null,
            mclyTextureIds: texIds,
            mclyLayerMask: layerMask,
            holeMask: new bool[16, 16],
            textureNames: ["terrain_a.blp"],
            modelPlacements:
            [
                new AlphaModelPlacement(0, "a.mdx", 1, new Vector3(17000f, 17000f, 10f), Vector3.Zero, 1f)
            ],
            worldModelPlacements:
            [
                new AlphaWorldModelPlacement(0, "b.wmo", 2, new Vector3(17000f, 17000f, 10f), Vector3.Zero, new Vector3(16999f, 16999f, 9f), new Vector3(17001f, 17001f, 11f), 0)
            ],
            liquidChunks: []);

        byte[] wdt = AlphaWdtWriter.Build("odd_name_table", new Dictionary<(int tileX, int tileY), AlphaTileData>
        {
            [(0, 0)] = tile
        });

        int offset = 0;
        Assert.Equal("MVER", ReadChunkId(wdt, offset));
        offset += 8 + BitConverter.ToInt32(wdt, offset + 4);
        Assert.Equal("MPHD", ReadChunkId(wdt, offset));
        offset += 8 + BitConverter.ToInt32(wdt, offset + 4);
        Assert.Equal("MAIN", ReadChunkId(wdt, offset));
        offset += 8 + BitConverter.ToInt32(wdt, offset + 4);
        Assert.Equal("MDNM", ReadChunkId(wdt, offset));
        offset += 8 + BitConverter.ToInt32(wdt, offset + 4);
        Assert.Equal("MONM", ReadChunkId(wdt, offset));
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

    private static LkMcnkData CreateChunk(int chunkX, int chunkY, float baseHeight, float slope, int flags, bool withAlpha, AdtLiquidChunk? liquidData = null, IReadOnlyList<int>? doodadRefs = null, IReadOnlyList<int>? worldModelRefs = null, int areaId = 0)
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
            AreaId = areaId,
            BaseHeight = baseHeight,
            Heights = heights,
            Normals = [],
            Layers = layers,
            AlphaMapData = alphaMapData,
            AlphaMapSize = alphaMapData?.Length ?? 0,
            LiquidData = liquidData,
            DoodadRefs = doodadRefs ?? [],
            WorldModelRefs = worldModelRefs ?? []
        };
    }

    private static (int[] DoodadRefs, int[] WorldModelRefs) ReadMcrfRefs(byte[] wdt, int tileX, int tileY, int chunkIndex)
    {
        int mainPayloadOffset = 12 + 8 + 128 + 8;
        int tileOffset = BitConverter.ToInt32(wdt, mainPayloadOffset + (((tileY * 64) + tileX) * 16));
        int mhdrDataOffset = tileOffset + 8;
        int mcinRelativeOffset = BitConverter.ToInt32(wdt, mhdrDataOffset + 0x00);
        int mcinOffset = mhdrDataOffset + mcinRelativeOffset;
        int mcnkOffset = BitConverter.ToInt32(wdt, mcinOffset + 8 + (chunkIndex * 16));
        int headerOffset = mcnkOffset + 8;
        int dataBase = headerOffset + 128;

        int nDoodadRefs = BitConverter.ToInt32(wdt, headerOffset + 0x14);
        int nWorldModelRefs = BitConverter.ToInt32(wdt, headerOffset + 0x3C);
        int mcrfOffset = BitConverter.ToInt32(wdt, headerOffset + 0x24);
        int mcrfChunkOffset = dataBase + mcrfOffset;
        Assert.Equal("MCRF", ReadChunkId(wdt, mcrfChunkOffset));
        Assert.Equal((nDoodadRefs + nWorldModelRefs) * 4, BitConverter.ToInt32(wdt, mcrfChunkOffset + 4));

        int payloadOffset = mcrfChunkOffset + 8;
        int[] doodadRefs = new int[nDoodadRefs];
        for (int i = 0; i < nDoodadRefs; i++)
            doodadRefs[i] = BitConverter.ToInt32(wdt, payloadOffset + (i * 4));

        int[] worldModelRefs = new int[nWorldModelRefs];
        for (int i = 0; i < nWorldModelRefs; i++)
            worldModelRefs[i] = BitConverter.ToInt32(wdt, payloadOffset + ((nDoodadRefs + i) * 4));

        return (doodadRefs, worldModelRefs);
    }

    private static int ReadMcnkDeclaredSize(byte[] wdt, int tileX, int tileY, int chunkIndex)
    {
        int mainPayloadOffset = 12 + 8 + 128 + 8;
        int tileOffset = BitConverter.ToInt32(wdt, mainPayloadOffset + (((tileY * 64) + tileX) * 16));
        int mhdrDataOffset = tileOffset + 8;
        int mcinRelativeOffset = BitConverter.ToInt32(wdt, mhdrDataOffset + 0x00);
        int mcinOffset = mhdrDataOffset + mcinRelativeOffset;
        int mcnkOffset = BitConverter.ToInt32(wdt, mcinOffset + 8 + (chunkIndex * 16));
        return BitConverter.ToInt32(wdt, mcnkOffset + 4);
    }

    private static IReadOnlyList<int>[] CreateEmptyChunkRefSet()
    {
        IReadOnlyList<int>[] refsByChunk = new IReadOnlyList<int>[256];
        for (int chunkIndex = 0; chunkIndex < refsByChunk.Length; chunkIndex++)
            refsByChunk[chunkIndex] = [];

        return refsByChunk;
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
