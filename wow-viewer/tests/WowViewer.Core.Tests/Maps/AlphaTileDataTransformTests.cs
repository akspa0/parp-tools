using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests.Maps;

/// <summary>
/// Spec 232 T015b: full-tile rotation has to agree with the already-established MCNK slot map
/// before the viewer routes transformed Alpha tiles through this path.
/// </summary>
public sealed class AlphaTileDataTransformTests
{
    [Fact]
    public void RotateQuarterTurn_SlicesTargetChunkFromTheSlotNamedByThePolicy()
    {
        var heightmap = new float[257, 257];
        for (int y = 0; y < 257; y++)
        {
            for (int x = 0; x < 257; x++)
                heightmap[y, x] = (y * 1000) + x;
        }

        AlphaTileData source = CreateTile(heightmap);
        AlphaTileData rotated = source.RotateQuarterTurn(1, mirrorH: false, mirrorV: false);

        (int sourceChunkX, int sourceChunkY) = (0, 15);
        (int targetChunkX, int targetChunkY) = TileContentTransform.TransformChunkSlot(
            sourceChunkX, sourceChunkY, TileTransformKind.Rotate90CW);
        Assert.Equal((0, 0), (targetChunkX, targetChunkY));

        TerrainChunkData expected = TileContentTransform.TransformChunk(
            FindChunk(source.ToTileLoadResult(20, 20), sourceChunkX, sourceChunkY),
            TileTransformKind.Rotate90CW);
        TerrainChunkData actual = FindChunk(rotated.ToTileLoadResult(20, 20), targetChunkX, targetChunkY);

        Assert.Equal(expected.Heights, actual.Heights);
        Assert.Equal(heightmap[256, 0], actual.Heights[0]);
    }

    [Fact]
    public void RotateQuarterTurn_TransformsNormalsAndChunkMetadataWithTheSameMap()
    {
        var heightmap = new float[257, 257];
        var normals = new float[257, 257, 3];
        var areaIds = new int[16, 16];
        normals[0, 0, 0] = 2f;
        normals[0, 0, 1] = 3f;
        normals[0, 0, 2] = 4f;
        areaIds[0, 0] = 1234;

        AlphaTileData rotated = CreateTile(heightmap, normals, areaIds).RotateQuarterTurn(1, false, false);

        Assert.Equal(1234, rotated.AreaIds![15, 0]);
        Assert.Equal(3f, rotated.McnrNormalXyz![0, 256, 0]);
        Assert.Equal(-2f, rotated.McnrNormalXyz[0, 256, 1]);
        Assert.Equal(4f, rotated.McnrNormalXyz[0, 256, 2]);
    }

    [Fact]
    public void RotateQuarterTurn_PreservesLiquidGridsAndMcnkFlagsAtTheTransformedSlot()
    {
        var liquidHeights = new float[81];
        var liquidFlags = new byte[64];
        for (int index = 0; index < liquidHeights.Length; index++)
        {
            liquidHeights[index] = index;
        }
        for (int index = 0; index < liquidFlags.Length; index++)
            liquidFlags[index] = (byte)index;

        var mcnkFlags = new int[16, 16];
        mcnkFlags[15, 0] = 0x3C;
        AlphaTileData rotated = CreateTile(
            new float[257, 257],
            liquidChunks:
            [
                new AlphaLiquidChunk(240, 0, 15, 10f, 12f, liquidFlags, 0x3C, liquidHeights),
            ],
            mcnkFlags16: mcnkFlags).RotateQuarterTurn(1, false, false);

        AlphaLiquidChunk liquid = Assert.Single(rotated.LiquidChunks);
        Assert.Equal((0, 0), (liquid.IndexX, liquid.IndexY));
        Assert.Equal(72f, liquid.Heights![0]);
        Assert.Equal(56, liquid.TileFlags![0]);
        Assert.Equal(0x3C, FindChunk(rotated.ToTileLoadResult(8, 9), 0, 0).McnkFlags);
    }

    [Fact]
    public void ToTileLoadResult_RehomesChunkWorldPositionsAtTheTargetTile()
    {
        TileLoadResult result = CreateTile(new float[257, 257]).ToTileLoadResult(8, 9);
        TerrainChunkData firstChunk = FindChunk(result, 0, 0);

        Assert.Equal(17066.66666f - (8 * 533.33333f), firstChunk.WorldPosition.X, 3);
        Assert.Equal(17066.66666f - (9 * 533.33333f), firstChunk.WorldPosition.Y, 3);
    }

    [Fact]
    public void ToTileLoadResult_SlicesAlphaFromTheFullResolutionPackAtEveryChunk()
    {
        // Spec 232 T015d regression: the reader's 256x256 downsampled pack cannot satisfy a
        // 64-px-per-chunk stride. Real tiles must slice from the 1024x1024 plane or chunks past
        // (3,3) silently decode zero alpha and every upper texture layer disappears.
        var alphaFull = new float[1024, 1024, 4];
        alphaFull[7 * 64 + 10, 5 * 64 + 20, 2] = 0.75f;

        AlphaTileData tile = CreateTile(new float[257, 257], mclyLayerMask3Layers: true, mcalAlphaPackFull: alphaFull);
        TerrainChunkData chunk = FindChunk(tile.ToTileLoadResult(3, 4), 5, 7);

        Assert.Equal(3, chunk.Layers.Length);
        Assert.True(chunk.AlphaMaps.ContainsKey(1));
        Assert.True(chunk.AlphaMaps.ContainsKey(2));
        Assert.Equal(191, chunk.AlphaMaps[2][10 * 64 + 20]);
        Assert.Equal(0, chunk.AlphaMaps[2][0]);
    }

    [Fact]
    public void ToTileLoadResult_UpsamplesThePacked256PlaneWhenFullResolutionIsAbsent()
    {
        var alphaPacked = new float[256, 256, 4];
        alphaPacked[7 * 16 + 2, 5 * 16 + 5, 1] = 0.5f;

        AlphaTileData tile = CreateTile(new float[257, 257], mclyLayerMask3Layers: true, mcalAlphaPack: alphaPacked);
        TerrainChunkData chunk = FindChunk(tile.ToTileLoadResult(3, 4), 5, 7);

        Assert.Equal(127, chunk.AlphaMaps[1][8 * 64 + 20]);
    }

    [Fact]
    public void RotateQuarterTurn_MovesLayerAlphaWithTheSameSlotMapAsHeights()
    {
        var alphaFull = new float[1024, 1024, 4];
        for (int y = 0; y < 1024; y++)
            for (int x = 0; x < 1024; x++)
                alphaFull[y, x, 1] = ((y * 1024) + x) / 1048576f;

        AlphaTileData source = CreateTile(new float[257, 257], mclyLayerMask3Layers: true, mcalAlphaPackFull: alphaFull);
        AlphaTileData rotated = source.RotateQuarterTurn(1, mirrorH: false, mirrorV: false);

        (int sourceChunkX, int sourceChunkY) = (0, 15);
        (int targetChunkX, int targetChunkY) = TileContentTransform.TransformChunkSlot(
            sourceChunkX, sourceChunkY, TileTransformKind.Rotate90CW);

        TerrainChunkData expected = TileContentTransform.TransformChunk(
            FindChunk(source.ToTileLoadResult(20, 20), sourceChunkX, sourceChunkY),
            TileTransformKind.Rotate90CW);
        TerrainChunkData actual = FindChunk(rotated.ToTileLoadResult(20, 20), targetChunkX, targetChunkY);

        Assert.Equal(expected.AlphaMaps[1], actual.AlphaMaps[1]);
    }

    private static AlphaTileData CreateTile(
        float[,] heightmap,
        float[,,]? normals = null,
        int[,]? areaIds = null,
        IReadOnlyList<AlphaLiquidChunk>? liquidChunks = null,
        int[,]? mcnkFlags16 = null,
        bool mclyLayerMask3Layers = false,
        float[,,]? mcalAlphaPack = null,
        float[,,]? mcalAlphaPackFull = null)
    {
        var textureIds = new int[16, 16, 4];
        var layerMask = new bool[16, 16, 4];
        if (mclyLayerMask3Layers)
        {
            for (int cx = 0; cx < 16; cx++)
            {
                for (int cy = 0; cy < 16; cy++)
                {
                    for (int l = 0; l < 3; l++)
                    {
                        textureIds[cx, cy, l] = l;
                        layerMask[cx, cy, l] = true;
                    }
                }
            }
        }

        return new AlphaTileData(
            sourcePath: "synthetic",
            heightmap: heightmap,
            mcalAlphaPack: mcalAlphaPack,
            mcalAlphaPackFull: mcalAlphaPackFull,
            mclyTextureIds: textureIds,
            mclyLayerMask: layerMask,
            holeMask: new bool[16, 16],
            textureNames: Array.Empty<string>(),
            modelPlacements: Array.Empty<AlphaModelPlacement>(),
            worldModelPlacements: Array.Empty<AlphaWorldModelPlacement>(),
            liquidChunks: liquidChunks ?? Array.Empty<AlphaLiquidChunk>(),
            mcnrNormalXyz: normals,
            areaIds: areaIds,
            mcnkFlags16: mcnkFlags16);
    }

    private static TerrainChunkData FindChunk(TileLoadResult tile, int chunkX, int chunkY)
        => Assert.Single(tile.Chunks, chunk => chunk.ChunkX == chunkX && chunk.ChunkY == chunkY);
}
