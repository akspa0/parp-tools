using WowViewer.Core.Runtime.World.Passes;

namespace WowViewer.Core.Tests;

public sealed class WorldFramePassCoordinatorTests
{
    [Fact]
    public void Execute_RunsCurrentFrameOrder_WhenAllStagesAreEnabled()
    {
        List<string> stages = new();

        bool continued = WorldFramePassCoordinator.Execute(
            new WorldFramePassOptions(objectsVisible: true, wmosVisible: true, doodadsVisible: true),
            CreatePasses(stages));

        Assert.True(continued);
        Assert.Equal(
        [
            "lighting",
            "sky",
            "skybox",
            "wdl",
            "terrain",
            "prepare",
            "wmo",
            "mdx-opaque",
            "liquid",
            "mdx-transparent",
            "overlay"
        ],
        stages);
    }

    [Fact]
    public void Execute_StopsAfterTerrain_WhenObjectsAreHidden()
    {
        List<string> stages = new();

        bool continued = WorldFramePassCoordinator.Execute(
            new WorldFramePassOptions(objectsVisible: false, wmosVisible: true, doodadsVisible: true),
            CreatePasses(stages));

        Assert.False(continued);
        Assert.Equal(["lighting", "sky", "skybox", "wdl", "terrain"], stages);
    }

    [Fact]
    public void Execute_SkipsDisabledWorldLayers_WhileKeepingObjectPhaseOrder()
    {
        List<string> stages = new();

        bool continued = WorldFramePassCoordinator.Execute(
            new WorldFramePassOptions(
                objectsVisible: true,
                wmosVisible: true,
                doodadsVisible: true,
                skyVisible: false,
                wdlVisible: false,
                terrainVisible: true,
                liquidVisible: false,
                overlayVisible: false),
            CreatePasses(stages));

        Assert.True(continued);
        Assert.Equal(
        [
            "lighting",
            "terrain",
            "prepare",
            "wmo",
            "mdx-opaque",
            "mdx-transparent"
        ],
        stages);
    }

    [Fact]
    public void Execute_SkipsOptionalObjectFamilies_ButStillRunsLiquidAndOverlay()
    {
        List<string> stages = new();

        bool continued = WorldFramePassCoordinator.Execute(
            new WorldFramePassOptions(objectsVisible: true, wmosVisible: false, doodadsVisible: false),
            CreatePasses(stages));

        Assert.True(continued);
        Assert.Equal(
        [
            "lighting",
            "sky",
            "skybox",
            "wdl",
            "terrain",
            "prepare",
            "liquid",
            "overlay"
        ],
        stages);
    }

    private static WorldFramePasses CreatePasses(List<string> stages)
    {
        return new WorldFramePasses(
            () => stages.Add("lighting"),
            () => stages.Add("sky"),
            () => stages.Add("skybox"),
            () => stages.Add("wdl"),
            () => stages.Add("terrain"),
            () => stages.Add("prepare"),
            () => stages.Add("wmo"),
            () => stages.Add("mdx-opaque"),
            () => stages.Add("liquid"),
            () => stages.Add("mdx-transparent"),
            () => stages.Add("overlay"));
    }
}