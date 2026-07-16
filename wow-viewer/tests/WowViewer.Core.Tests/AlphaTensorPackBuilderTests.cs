using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using System.Numerics;

namespace WowViewer.Core.Tests;

public sealed class AlphaTensorPackBuilderTests
{
    [Fact]
    public void Build_NormalizesAlphaChunkLayersToRowMajorTensorPackLayout()
    {
        var textureIds = new int[16, 16, 4];
        var layerMask = new bool[16, 16, 4];
        textureIds[3, 7, 2] = 42;
        layerMask[3, 7, 2] = true;

        AlphaTileData tile = new(
            sourcePath: "alpha.wdt#alpha-tile(3,7)",
            heightmap: new float[257, 257],
            mcalAlphaPack: new float[256, 256, 4],
            mclyTextureIds: textureIds,
            mclyLayerMask: layerMask,
            holeMask: new bool[16, 16],
            textureNames: Array.Empty<string>(),
            modelPlacements: Array.Empty<AlphaModelPlacement>(),
            worldModelPlacements: Array.Empty<AlphaWorldModelPlacement>(),
            liquidChunks: Array.Empty<AlphaLiquidChunk>());

        TerrainTileTensorPack pack = AlphaTensorPackBuilder.Build(tile, 3, 7);

        Assert.Equal(42, pack.MclyTextureIds![7, 3, 2]);
        Assert.True(pack.MclyLayerMask![7, 3, 2]);
    }

    [Fact]
    public void Build_PreservesRawAlphaChunksIntoTensorPack()
    {
        AlphaTileData tile = new(
            sourcePath: "alpha.wdt#alpha-tile(0,0)",
            heightmap: new float[257, 257],
            mcalAlphaPack: null,
            mclyTextureIds: new int[16, 16, 4],
            mclyLayerMask: new bool[16, 16, 4],
            holeMask: new bool[16, 16],
            textureNames: Array.Empty<string>(),
            modelPlacements: Array.Empty<AlphaModelPlacement>(),
            worldModelPlacements: Array.Empty<AlphaWorldModelPlacement>(),
            liquidChunks: Array.Empty<AlphaLiquidChunk>(),
            rawChunks:
            [
                new TerrainRawChunkBlob
                {
                    EntryName = "raw_chunks/alpha/top/MHDR_000",
                    SourceKind = "alpha",
                    SourcePath = "alpha.wdt#alpha-tile(0,0)",
                    Scope = "top-level",
                    ChunkId = "MHDR",
                    Data = [1, 2, 3, 4],
                },
            ]);

        TerrainTileTensorPack pack = AlphaTensorPackBuilder.Build(tile, 0, 0);

        Assert.Single(pack.RawChunks);
        Assert.Contains("raw_alpha_chunks", pack.AvailableSignals);
        Assert.Equal("MHDR", pack.RawChunks[0].ChunkId);
    }

    [Fact]
    public void Build_EmbeddedAlphaPlacements_ProjectIntoObjectMasks()
    {
        AlphaTileData tile = new(
            sourcePath: "Azeroth.wdt#alpha-tile(36,35)",
            heightmap: new float[257, 257],
            mcalAlphaPack: null,
            mclyTextureIds: new int[16, 16, 4],
            mclyLayerMask: new bool[16, 16, 4],
            holeMask: new bool[16, 16],
            textureNames: Array.Empty<string>(),
            modelPlacements:
            [
                new AlphaModelPlacement(
                    0,
                    "World\\Generic\\Human\\PassiveDoodads\\Barrel\\Barrel01.m2",
                    1001,
                    new Vector3(-1599.2383f, -2381.5684f, 100.70325f),
                    Vector3.Zero,
                    1.0f),
            ],
            worldModelPlacements:
            [
                new AlphaWorldModelPlacement(
                    0,
                    "World\\wmo\\Azeroth\\Buildings\\HumanFarm\\HumanFarm.wmo",
                    2001,
                    new Vector3(-1599.2383f, -2381.5684f, 100.70325f),
                    Vector3.Zero,
                    new Vector3(-1604.0f, -2386.0f, 95.0f),
                    new Vector3(-1594.0f, -2376.0f, 108.0f),
                    0),
            ],
            liquidChunks: Array.Empty<AlphaLiquidChunk>());

        TerrainTileTensorPack pack = AlphaTensorPackBuilder.Build(tile, 36, 35);

        Assert.NotNull(pack.ObjectMask257);
        Assert.NotNull(pack.ObjectPreciseMask257);
        Assert.NotNull(pack.ObjectInstanceMask257);
        Assert.Contains(pack.ObjectMask257!.Cast<float>(), static value => value > 0f);
        Assert.Contains(pack.ObjectPreciseMask257!.Cast<float>(), static value => value > 0f);
        Assert.Contains(pack.ObjectInstanceMask257!.Cast<int>(), static value => value > 0);
    }

    [Fact]
    public void Build_ClipsRoofPlacementCirclesToThe256PixelRoofBuffers()
    {
        const float tileSize = 533.33333f;
        AlphaTileData tile = new(
            sourcePath: "alpha.wdt#alpha-tile(0,0)",
            heightmap: new float[257, 257],
            mcalAlphaPack: null,
            mclyTextureIds: new int[16, 16, 4],
            mclyLayerMask: new bool[16, 16, 4],
            holeMask: new bool[16, 16],
            textureNames: Array.Empty<string>(),
            modelPlacements: Array.Empty<AlphaModelPlacement>(),
            worldModelPlacements:
            [
                new AlphaWorldModelPlacement(
                    0,
                    "World\\wmo\\Azeroth\\Buildings\\EdgeHouse.wmo",
                    2001,
                    new Vector3(tileSize, 0f, tileSize),
                    Vector3.Zero,
                    Vector3.Zero,
                    Vector3.Zero,
                    0),
            ],
            liquidChunks: Array.Empty<AlphaLiquidChunk>());

        TerrainTileTensorPack pack = AlphaTensorPackBuilder.Build(tile, 0, 0);

        Assert.NotNull(pack.ObjectRoofMask256);
        Assert.NotNull(pack.ObjectRoofConfidence256);
        Assert.Equal(1f, pack.ObjectRoofMask256![255, 255]);
        Assert.Equal(0.95f, pack.ObjectRoofConfidence256![255, 255]);
    }

    [Fact]
    public void Build_NormalizesChunkLiquidTypesAndProducesAlignedUnifiedLiquid()
    {
        var liquidTypes = new int[16, 16];
        for (int y = 0; y < 16; y++)
            for (int x = 0; x < 16; x++)
                liquidTypes[y, x] = -1;
        liquidTypes[2, 3] = (int)AdtLiquidBasicType.Magma;

        AlphaTileData tile = new(
            sourcePath: "alpha.wdt#alpha-tile(3,2)",
            heightmap: new float[257, 257],
            mcalAlphaPack: new float[256, 256, 4],
            mclyTextureIds: new int[16, 16, 4],
            mclyLayerMask: new bool[16, 16, 4],
            holeMask: new bool[16, 16],
            textureNames: Array.Empty<string>(),
            modelPlacements: Array.Empty<AlphaModelPlacement>(),
            worldModelPlacements: Array.Empty<AlphaWorldModelPlacement>(),
            liquidChunks: Array.Empty<AlphaLiquidChunk>(),
            mclqSurfaceHeight: new float[257, 257],
            mclqTypeMask: liquidTypes);

        TerrainTileTensorPack pack = AlphaTensorPackBuilder.Build(tile, 3, 2);

        Assert.Equal((257, 257), (pack.MclqTypeMask!.GetLength(0), pack.MclqTypeMask.GetLength(1)));
        Assert.True(pack.MclqPresenceMask![40, 56]);
        Assert.Equal(1f, pack.UnifiedLiquidMask![40, 56]);
        Assert.Equal((byte)AdtLiquidBasicType.Magma, pack.LiquidBasicType257![40, 56]);
        Assert.Contains("unified_liquid_mask", pack.AvailableSignals);
        Assert.Contains("unified_liquid_height", pack.AvailableSignals);
    }

    [Fact]
    public void Build_UsesAlphaLiquidCellFlagsInsteadOfPaintingAnEntireChunk()
    {
        var liquidTypes = new int[16, 16];
        for (int y = 0; y < 16; y++)
            for (int x = 0; x < 16; x++)
                liquidTypes[y, x] = -1;
        liquidTypes[0, 0] = (int)AdtLiquidBasicType.Ocean;

        byte[] flags = Enumerable.Repeat((byte)0x0F, 64).ToArray();
        flags[0] = 0x00;
        AlphaTileData tile = new(
            sourcePath: "alpha.wdt#alpha-tile(0,0)",
            heightmap: new float[257, 257],
            mcalAlphaPack: new float[256, 256, 4],
            mclyTextureIds: new int[16, 16, 4],
            mclyLayerMask: new bool[16, 16, 4],
            holeMask: new bool[16, 16],
            textureNames: Array.Empty<string>(),
            modelPlacements: Array.Empty<AlphaModelPlacement>(),
            worldModelPlacements: Array.Empty<AlphaWorldModelPlacement>(),
            liquidChunks:
            [
                new AlphaLiquidChunk(0, 0, 0, 12f, 12f, flags, 0x08u, null)
            ],
            mclqSurfaceHeight: new float[257, 257],
            mclqTypeMask: liquidTypes);

        TerrainTileTensorPack pack = AlphaTensorPackBuilder.Build(tile, 0, 0);

        Assert.True(pack.MclqPresenceMask![1, 1]);
        Assert.False(pack.MclqPresenceMask[3, 1]);
        Assert.Equal(1f, pack.UnifiedLiquidMask![1, 1]);
        Assert.Equal(0f, pack.UnifiedLiquidMask[3, 1]);
    }

    [Fact]
    public void Build_UsesVisibleMclqCellTypeBeforeItsMcnkChunkFallback()
    {
        var liquidTypes = new int[16, 16];
        for (int y = 0; y < 16; y++)
            for (int x = 0; x < 16; x++)
                liquidTypes[y, x] = -1;
        liquidTypes[0, 0] = (int)AdtLiquidBasicType.Magma;

        byte[] flags = Enumerable.Repeat((byte)0x0F, 64).ToArray();
        flags[0] = 0x02; // Ocean; the containing MCNK declares magma.
        AlphaTileData tile = new(
            sourcePath: "alpha.wdt#alpha-tile(0,0)",
            heightmap: new float[257, 257],
            mcalAlphaPack: new float[256, 256, 4],
            mclyTextureIds: new int[16, 16, 4],
            mclyLayerMask: new bool[16, 16, 4],
            holeMask: new bool[16, 16],
            textureNames: Array.Empty<string>(),
            modelPlacements: Array.Empty<AlphaModelPlacement>(),
            worldModelPlacements: Array.Empty<AlphaWorldModelPlacement>(),
            liquidChunks:
            [
                new AlphaLiquidChunk(0, 0, 0, 12f, 12f, flags, 0x10u, null)
            ],
            mclqSurfaceHeight: new float[257, 257],
            mclqTypeMask: liquidTypes);

        TerrainTileTensorPack pack = AlphaTensorPackBuilder.Build(tile, 0, 0);

        Assert.True(pack.MclqPresenceMask![1, 1]);
        Assert.Equal((int)AdtLiquidBasicType.Ocean, pack.MclqTypeMask![1, 1]);
        Assert.Equal((byte)AdtLiquidBasicType.Ocean, pack.LiquidBasicType257![1, 1]);
    }
}
