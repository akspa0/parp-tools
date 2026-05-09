using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

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
}