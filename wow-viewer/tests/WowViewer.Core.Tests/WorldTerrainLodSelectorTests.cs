using WowViewer.Core.Runtime.World.Terrain;

namespace WowViewer.Core.Tests;

public sealed class WorldTerrainLodSelectorTests
{
    [Fact]
    public void Select_CloseDistance_UsesFullDetailAndAllLayers()
    {
        WorldTerrainChunkData chunk = CreateChunk(layerCount: 4, holeMask: 0x0000);

        WorldTerrainLodSelection selection = WorldTerrainLodSelector.Select(chunk, distance: 128f, textureLodDistance: 512f, fogEndDistance: 2048f);

        Assert.Equal(WorldTerrainLodLevel.FullDetail, selection.Level);
        Assert.Equal(4, selection.ActiveTextureLayerCount);
        Assert.Equal(1f, selection.OverlayFadeFactor);
        Assert.Equal(64, selection.RenderableCellCount);
        Assert.False(selection.UsesLowDetailMesh);
    }

    [Fact]
    public void Select_MidDistance_FadesOverlayLayers()
    {
        WorldTerrainChunkData chunk = CreateChunk(layerCount: 4, holeMask: 0x0000);

        WorldTerrainLodSelection selection = WorldTerrainLodSelector.Select(chunk, distance: 640f, textureLodDistance: 512f, fogEndDistance: 2048f);

        Assert.Equal(WorldTerrainLodLevel.FadeToBaseLayer, selection.Level);
        Assert.Equal(4, selection.ActiveTextureLayerCount);
        Assert.Equal(0.5f, selection.OverlayFadeFactor, 3);
        Assert.False(selection.UsesLowDetailMesh);
    }

    [Fact]
    public void Select_FarDistance_BeyondFadeWindow_UsesBaseLayerOnly()
    {
        WorldTerrainChunkData chunk = CreateChunk(layerCount: 3, holeMask: 0x0000);

        WorldTerrainLodSelection selection = WorldTerrainLodSelector.Select(chunk, distance: 800f, textureLodDistance: 512f, fogEndDistance: 2048f);

        Assert.Equal(WorldTerrainLodLevel.BaseLayerOnly, selection.Level);
        Assert.Equal(1, selection.ActiveTextureLayerCount);
        Assert.Equal(0f, selection.OverlayFadeFactor);
        Assert.False(selection.UsesLowDetailMesh);
    }

    [Fact]
    public void Select_BeyondFogEnd_UsesLowDetailAndCountsRenderableCellsFromHoleMask()
    {
        WorldTerrainChunkData chunk = CreateChunk(layerCount: 4, holeMask: 0x0001);

        WorldTerrainLodSelection selection = WorldTerrainLodSelector.Select(chunk, distance: 2048f, textureLodDistance: 512f, fogEndDistance: 1536f);

        Assert.Equal(WorldTerrainLodLevel.LowDetail, selection.Level);
        Assert.Equal(0, selection.ActiveTextureLayerCount);
        Assert.Equal(0f, selection.OverlayFadeFactor);
        Assert.Equal(60, selection.RenderableCellCount);
        Assert.True(selection.UsesLowDetailMesh);
    }

    [Fact]
    public void Select_SingleLayerChunk_StaysFullDetailUntilLowDetailCutover()
    {
        WorldTerrainChunkData chunk = CreateChunk(layerCount: 1, holeMask: 0x0000);

        WorldTerrainLodSelection selection = WorldTerrainLodSelector.Select(chunk, distance: 900f, textureLodDistance: 512f, fogEndDistance: 2048f);

        Assert.Equal(WorldTerrainLodLevel.FullDetail, selection.Level);
        Assert.Equal(1, selection.ActiveTextureLayerCount);
        Assert.Equal(1f, selection.OverlayFadeFactor);
        Assert.False(selection.UsesLowDetailMesh);
    }

    private static WorldTerrainChunkData CreateChunk(int layerCount, ushort holeMask)
    {
        return new WorldTerrainChunkData(
            chunkIndex: 0,
            indexX: 0,
            indexY: 0,
            areaId: 0,
            flags: 0,
            layerCount: layerCount,
            holeMask: holeMask,
            hasLiquidFlags: false,
            hasVertexColors: false,
            heights: new float[145]);
    }
}
