using WowViewer.Core.Runtime.World.Passes;

namespace WowViewer.Core.Tests;

public sealed class WorldFramePassCoordinatorTests
{
    [Fact]
    public void Execute_RunsCurrentFrameOrder_WhenAllStagesAreEnabled()
    {
        List<string> stages = new();

        bool continued = WorldFramePassCoordinator.Execute(
            new WorldFramePassOptions(ObjectsVisible: true, WmosVisible: true, DoodadsVisible: true),
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
            new WorldFramePassOptions(ObjectsVisible: false, WmosVisible: true, DoodadsVisible: true),
            CreatePasses(stages));

        Assert.False(continued);
        Assert.Equal(["lighting", "sky", "skybox", "wdl", "terrain"], stages);
    }

    [Fact]
    public void Execute_SkipsOptionalObjectFamilies_ButStillRunsLiquidAndOverlay()
    {
        List<string> stages = new();

        bool continued = WorldFramePassCoordinator.Execute(
            new WorldFramePassOptions(ObjectsVisible: true, WmosVisible: false, DoodadsVisible: false),
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