using WowViewer.Core.Runtime.World.Passes;
using WowViewer.Core.Runtime.World.Validation;
using WowViewer.Tools.ValidationCapture;

namespace WowViewer.Core.Tests;

public sealed class ValidationWorldSceneAdapterTests
{
    [Fact]
    public void BuildPassOptions_ObjectsOnlyVariant_DisablesTerrainAndLiquidsButKeepsObjects()
    {
        WorldFramePassOptions options = ValidationWorldSceneAdapter.BuildPassOptions(new ValidationWorldScenePolicyState
        {
            ShowObjects = true,
            ShowWmos = true,
            ShowDoodads = true,
            ShowSky = false,
            ShowWdl = false,
            ShowTerrain = false,
            ShowTerrainLiquids = false,
        });

        Assert.True(options.ObjectsVisible);
        Assert.True(options.WmosVisible);
        Assert.True(options.DoodadsVisible);
        Assert.False(options.SkyVisible);
        Assert.False(options.WdlVisible);
        Assert.False(options.TerrainVisible);
        Assert.False(options.LiquidVisible);
        Assert.False(options.OverlayVisible);
    }

    [Fact]
    public void BuildPassOptions_HidesObjectPhaseWhenNoObjectLayersRemain()
    {
        WorldFramePassOptions options = ValidationWorldSceneAdapter.BuildPassOptions(new ValidationWorldScenePolicyState
        {
            ShowObjects = false,
            ShowWmos = false,
            ShowDoodads = false,
            ShowSky = true,
            ShowWdl = true,
            ShowTerrain = true,
            ShowTerrainLiquids = true,
        });

        Assert.False(options.ObjectsVisible);
        Assert.False(options.WmosVisible);
        Assert.False(options.DoodadsVisible);
        Assert.True(options.SkyVisible);
        Assert.True(options.WdlVisible);
        Assert.True(options.TerrainVisible);
        Assert.True(options.LiquidVisible);
    }

    [Fact]
    public void CreateSnapshot_MapsTargetTileAndPendingLoads()
    {
        ValidationWorldSceneSnapshot snapshot = ValidationWorldSceneAdapter.CreateSnapshot(
            512,
            new ValidationCaptureTileRequest("Azeroth_30_48", 30, 48, ValidationCaptureVariant.Primary, "primary.png"),
            selectedTileX: 30,
            selectedTileY: 48,
            activeTerrainTileCount: 9,
            wmoInstanceCount: 14,
            mdxInstanceCount: 5,
            pendingWorldObjectLoadCount: 3);

        Assert.True(snapshot.HasSceneContent);
        Assert.Equal(512, snapshot.FramebufferWidth);
        Assert.Equal(512, snapshot.FramebufferHeight);
        Assert.True(snapshot.TargetTileLoaded);
        Assert.False(snapshot.TerrainStreaming);
        Assert.Equal(3, snapshot.PendingWorldObjectLoadCount);
    }

}
