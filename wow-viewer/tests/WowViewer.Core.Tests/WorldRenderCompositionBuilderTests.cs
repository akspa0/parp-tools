using WowViewer.Core.Maps;
using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.Liquid;
using WowViewer.Core.Runtime.World.Passes;
using WowViewer.Core.Runtime.World.Terrain;
using WowViewer.Core.Runtime.World.Wdl;

namespace WowViewer.Core.Tests;

public sealed class WorldRenderCompositionBuilderTests
{
    [Fact]
    public void Build_OrdersWorldLayersAroundSkyAndTerrain()
    {
        WorldRenderCompositionFrame composition = WorldRenderCompositionBuilder.Build(
            new WorldFramePassOptions(objectsVisible: true, wmosVisible: true, doodadsVisible: true),
            CreateWdl(hasData: true),
            CreateTerrain(chunkCount: 2),
            CreateLiquid(),
            wmoSourceCount: 3,
            mdxSourceCount: 4,
            CreateStats());

        Assert.Equal(
        [
            WorldRenderLayerKind.Sky,
            WorldRenderLayerKind.SkyboxBackdrop,
            WorldRenderLayerKind.Wdl,
            WorldRenderLayerKind.Terrain,
            WorldRenderLayerKind.Liquid,
            WorldRenderLayerKind.Wmo,
            WorldRenderLayerKind.Doodad,
            WorldRenderLayerKind.Overlay
        ],
        composition.Layers.Select(static layer => layer.Kind));

        Assert.True(composition.HasSubmittedSkyLayer);
        Assert.Equal("ADT Terrain Quilt", composition.Layers.Single(static layer => layer.Kind == WorldRenderLayerKind.Terrain).DisplayName);
        Assert.Equal(2, composition.Layers.Single(static layer => layer.Kind == WorldRenderLayerKind.Terrain).SourceCount);
    }

    [Fact]
    public void Build_LeavesBackdropSlotPendingUntilDecodedSkyboxAssetsExist()
    {
        WorldRenderCompositionFrame composition = WorldRenderCompositionBuilder.Build(
            new WorldFramePassOptions(objectsVisible: false, wmosVisible: false, doodadsVisible: false, skyVisible: true),
            CreateWdl(hasData: false),
            CreateTerrain(chunkCount: 0),
            CreateLiquid(),
            wmoSourceCount: 0,
            mdxSourceCount: 0,
            WorldRenderFrameStats.Empty);

        WorldRenderLayerState skybox = composition.Layers.Single(static layer => layer.Kind == WorldRenderLayerKind.SkyboxBackdrop);
        Assert.True(skybox.Enabled);
        Assert.False(skybox.Ready);
        Assert.Contains("Reserved", skybox.Note);
    }

    [Fact]
    public void Build_MarksBackdropReady_WhenBackdropPlacementsAreClassified()
    {
        WorldRenderCompositionFrame composition = WorldRenderCompositionBuilder.Build(
            new WorldFramePassOptions(objectsVisible: false, wmosVisible: false, doodadsVisible: false, skyVisible: true),
            CreateWdl(hasData: false),
            CreateTerrain(chunkCount: 0),
            CreateLiquid(),
            wmoSourceCount: 0,
            mdxSourceCount: 2,
            WorldRenderFrameStats.Empty,
            skyboxBackdropSourceCount: 2);

        WorldRenderLayerState skybox = composition.Layers.Single(static layer => layer.Kind == WorldRenderLayerKind.SkyboxBackdrop);
        Assert.True(skybox.Enabled);
        Assert.True(skybox.Ready);
        Assert.Equal(2, skybox.SourceCount);
        Assert.Equal(0, skybox.SubmittedCount);
        Assert.Contains("classified", skybox.Note);
    }

    [Fact]
    public void Build_CountsProceduralBackdropSubmission_WhenFrameStatsSubmitClassifiedBackdrop()
    {
        WorldRenderFrameStats stats = WorldRenderFrameStats.Empty with
        {
            SkyboxBackdrop = new WorldRenderStageStats(0, VisibleCount: 2, SubmittedCount: 2),
        };

        WorldRenderCompositionFrame composition = WorldRenderCompositionBuilder.Build(
            new WorldFramePassOptions(objectsVisible: false, wmosVisible: false, doodadsVisible: false, skyVisible: true),
            CreateWdl(hasData: false),
            CreateTerrain(chunkCount: 0),
            CreateLiquid(),
            wmoSourceCount: 0,
            mdxSourceCount: 2,
            stats,
            skyboxBackdropSourceCount: 2);

        WorldRenderLayerState skybox = composition.Layers.Single(static layer => layer.Kind == WorldRenderLayerKind.SkyboxBackdrop);
        Assert.True(skybox.Ready);
        Assert.Equal(2, skybox.SourceCount);
        Assert.Equal(2, skybox.SubmittedCount);
        Assert.Contains("decoded client sky assets", skybox.Note);
    }

    [Theory]
    [InlineData(@"Environments\Stars\stars.mdl")]
    [InlineData(@"World\Environments\Stars\AzerothStars.mdx")]
    [InlineData(@"World\Generic\Skybox\BlueSkyBox.m2")]
    [InlineData(@"World\Expansion01\SkyBowl\OutlandSkyBowl.mdx")]
    public void IsBackdropModelPath_AcceptsSkyBackdropFamilies(string modelPath)
    {
        Assert.True(WorldSkyboxBackdropClassifier.IsBackdropModelPath(modelPath));
    }

    [Theory]
    [InlineData(@"World\Generic\PassiveDoodads\Skylight\Skylight01.mdx")]
    [InlineData(@"World\Generic\PassiveDoodads\Trees\Oak01.mdx")]
    [InlineData(@"World\Generic\Skybox\Readme.txt")]
    public void IsBackdropModelPath_RejectsNonBackdropFamilies(string modelPath)
    {
        Assert.False(WorldSkyboxBackdropClassifier.IsBackdropModelPath(modelPath));
    }

    private static WorldWdlTileData CreateWdl(bool hasData)
    {
        return new WorldWdlTileData(
            "wdl",
            version: 18,
            tileX: 32,
            tileY: 28,
            sourceFound: true,
            hasData,
            minHeight: hasData ? (short)1 : null,
            maxHeight: hasData ? (short)9 : null,
            centerHeight: hasData ? (short)5 : null,
            northWestHeight: null,
            northEastHeight: null,
            southWestHeight: null,
            southEastHeight: null,
            outerHeightCount: hasData ? 289 : 0,
            innerHeightCount: hasData ? 256 : 0);
    }

    private static WorldTerrainTileData CreateTerrain(int chunkCount)
    {
        WorldTerrainChunkData[] chunks = Enumerable.Range(0, chunkCount)
            .Select(static index => new WorldTerrainChunkData(index, index % 16, index / 16, 0, 0, 1, 0, false, false, new float[145]))
            .ToArray();

        return new WorldTerrainTileData("adt", MapFileKind.Adt, chunks, heightmap: null);
    }

    private static WorldLiquidTileData CreateLiquid()
    {
        return new WorldLiquidTileData("adt", MapFileKind.Adt, []);
    }

    private static WorldRenderFrameStats CreateStats()
    {
        return WorldRenderFrameStats.Empty with
        {
            Wdl = new WorldRenderStageStats(0, VisibleCount: 1, SubmittedCount: 1),
            Terrain = new WorldRenderStageStats(0, VisibleCount: 2, SubmittedCount: 2),
            WmoSubmission = new WorldRenderStageStats(0, VisibleCount: 3, SubmittedCount: 3),
            MdxOpaqueSubmission = new WorldRenderStageStats(0, VisibleCount: 4, SubmittedCount: 4),
        };
    }
}
