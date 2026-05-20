using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using System.Numerics;

namespace WowViewer.Core.Tests;

public sealed class AlphaTensorPackBuilderTests
{
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
}
