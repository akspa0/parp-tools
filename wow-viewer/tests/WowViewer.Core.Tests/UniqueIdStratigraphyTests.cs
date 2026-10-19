using System.Numerics;
using WowViewer.Core.Archaeology;
using Xunit;

namespace WowViewer.Core.Tests;

public sealed class UniqueIdStratigraphyTests
{
    [Fact]
    public void ComputeQuantile_ValidDistribution_ReturnsExpectedValues()
    {
        // 1 to 10
        int[] values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        double p0 = UniqueIdStratigraphyEngine.ComputeQuantile(values, 0.0);
        double p10 = UniqueIdStratigraphyEngine.ComputeQuantile(values, 0.10);
        double median = UniqueIdStratigraphyEngine.ComputeQuantile(values, 0.50);
        double p90 = UniqueIdStratigraphyEngine.ComputeQuantile(values, 0.90);
        double p100 = UniqueIdStratigraphyEngine.ComputeQuantile(values, 1.0);

        Assert.Equal(1.0, p0);
        Assert.Equal(1.9, p10, 1);
        Assert.Equal(5.5, median, 1);
        Assert.Equal(9.1, p90, 1);
        Assert.Equal(10.0, p100);
    }

    [Fact]
    public void ComputeQuantile_EmptyOrSingle_HandlesGracefully()
    {
        Assert.Equal(0.0, UniqueIdStratigraphyEngine.ComputeQuantile([], 0.5));
        Assert.Equal(42.0, UniqueIdStratigraphyEngine.ComputeQuantile([42], 0.5));
    }

    [Fact]
    public void DetectLayers_GapsExceedingThreshold_CreatesDistinctLayers()
    {
        var records = new List<UniqueIdRecord>
        {
            // Layer 1: 100 .. 150
            new() { UniqueId = 100, ModelPath = "World/wmo/Dungeon.wmo", Position = Vector3.Zero, Kind = PlacementOriginKind.Wmo },
            new() { UniqueId = 120, ModelPath = "World/wmo/Dungeon.wmo", Position = Vector3.Zero, Kind = PlacementOriginKind.Wmo },
            new() { UniqueId = 150, ModelPath = "World/Generic/Tree.m2", Position = Vector3.Zero, Kind = PlacementOriginKind.M2 },

            // Gap = 850 (exceeds gapThreshold 100) -> Layer 2: 1000 .. 1050
            new() { UniqueId = 1000, ModelPath = "World/Generic/Tree.m2", Position = Vector3.Zero, Kind = PlacementOriginKind.M2 },
            new() { UniqueId = 1020, ModelPath = "Creature/Peon/Peon.m2", Position = Vector3.Zero, Kind = PlacementOriginKind.Creature, DisplayName = "Orc Peon" },
            new() { UniqueId = 1050, ModelPath = "World/Generic/Chest.m2", Position = Vector3.Zero, Kind = PlacementOriginKind.GameObject, DisplayName = "Treasure Chest" },

            // Gap = 4950 -> Layer 3: 6000
            new() { UniqueId = 6000, ModelPath = "World/Generic/Lantern.m2", Position = Vector3.Zero, Kind = PlacementOriginKind.M2 }
        };

        var layers = UniqueIdStratigraphyEngine.DetectLayers(records, gapThreshold: 100);

        Assert.Equal(3, layers.Count);

        Assert.Equal(1, layers[0].LayerNumber);
        Assert.Equal(100, layers[0].StartId);
        Assert.Equal(150, layers[0].EndId);
        Assert.Equal(3, layers[0].TotalCount);
        Assert.Equal(2, layers[0].WmoCount);
        Assert.Equal(1, layers[0].M2Count);
        Assert.Equal(0, layers[0].CreatureCount);

        Assert.Equal(2, layers[1].LayerNumber);
        Assert.Equal(1000, layers[1].StartId);
        Assert.Equal(1050, layers[1].EndId);
        Assert.Equal(3, layers[1].TotalCount);
        Assert.Equal(1, layers[1].M2Count);
        Assert.Equal(1, layers[1].CreatureCount);
        Assert.Equal(1, layers[1].GameObjectCount);

        Assert.Equal(3, layers[2].LayerNumber);
        Assert.Equal(6000, layers[2].StartId);
        Assert.Equal(6000, layers[2].EndId);
        Assert.Equal(1, layers[2].TotalCount);
    }

    [Fact]
    public void ComputeProfile_AggregatesTileSummariesAndMetrics()
    {
        var records = new List<UniqueIdRecord>
        {
            new() { UniqueId = 10, ModelPath = "Tree.m2", Position = Vector3.Zero, Kind = PlacementOriginKind.M2, TileX = 32, TileY = 32, HasTileCoordinates = true },
            new() { UniqueId = 20, ModelPath = "Tree.m2", Position = Vector3.Zero, Kind = PlacementOriginKind.M2, TileX = 32, TileY = 32, HasTileCoordinates = true },
            new() { UniqueId = 30, ModelPath = "House.wmo", Position = Vector3.Zero, Kind = PlacementOriginKind.Wmo, TileX = 32, TileY = 33, HasTileCoordinates = true },
            new() { UniqueId = 40, ModelPath = "Peon.m2", Position = Vector3.Zero, Kind = PlacementOriginKind.Creature, TileX = 32, TileY = 33, HasTileCoordinates = true }
        };

        var profile = UniqueIdStratigraphyEngine.ComputeProfile(451, "development", records);

        Assert.Equal(451, profile.MapId);
        Assert.Equal("development", profile.MapName);
        Assert.Equal(4, profile.TotalPlacements);
        Assert.Equal(4, profile.DistinctUniqueIds);
        Assert.Equal(10, profile.MinUniqueId);
        Assert.Equal(40, profile.MaxUniqueId);
        Assert.Equal(2, profile.Tiles.Count);

        var tile1 = profile.Tiles.First(t => t.TileX == 32 && t.TileY == 32);
        Assert.Equal(2, tile1.TotalCount);
        Assert.Equal(2, tile1.M2Count);
        Assert.Equal(0, tile1.WmoCount);

        var tile2 = profile.Tiles.First(t => t.TileX == 32 && t.TileY == 33);
        Assert.Equal(2, tile2.TotalCount);
        Assert.Equal(1, tile2.WmoCount);
        Assert.Equal(1, tile2.SpawnCount);
    }

    [Fact]
    public void CrossMapDivergenceEngine_IdentifiesTwinsAndBranchingThreshold()
    {
        // Reference (Overworld Map 0): Placements up to UID 2500
        var overworld = new List<UniqueIdRecord>
        {
            new() { UniqueId = 101, ModelPath = "World/wmo/Tavern.wmo", Position = new Vector3(100f, 200f, 50f), Kind = PlacementOriginKind.Wmo },
            new() { UniqueId = 205, ModelPath = "World/Generic/Tree.m2", Position = new Vector3(105f, 202f, 50f), Kind = PlacementOriginKind.M2 },
            new() { UniqueId = 2490, ModelPath = "World/Generic/Lamp.m2", Position = new Vector3(110f, 200f, 50f), Kind = PlacementOriginKind.M2 },
            new() { UniqueId = 2500, ModelPath = "World/Generic/Crate.m2", Position = new Vector3(120f, 200f, 50f), Kind = PlacementOriginKind.M2 }
        };

        // Target (Map 451): Cloned overworld base (UIDs 101, 205, 2500) + Post-fork development (UIDs 8000, 8050)
        var targetMap = new List<UniqueIdRecord>
        {
            // Exact ID and position match
            new() { UniqueId = 101, ModelPath = "World/wmo/Tavern.wmo", Position = new Vector3(100f, 200f, 50f), Kind = PlacementOriginKind.Wmo },
            
            // Spatial proximity match with slight offset (0.2 units <= 0.5 tolerance)
            new() { UniqueId = 205, ModelPath = "World/Generic/Tree.m2", Position = new Vector3(105.1f, 202.1f, 50f), Kind = PlacementOriginKind.M2 },

            // Last shared placement in overworld
            new() { UniqueId = 2500, ModelPath = "World/Generic/Crate.m2", Position = new Vector3(120f, 200f, 50f), Kind = PlacementOriginKind.M2 },

            // Post-fork custom additions (higher UIDs, new positions)
            new() { UniqueId = 8000, ModelPath = "World/Test/NewModel.m2", Position = new Vector3(500f, 500f, 10f), Kind = PlacementOriginKind.M2 },
            new() { UniqueId = 8050, ModelPath = "Creature/TestMob/TestMob.m2", Position = new Vector3(505f, 500f, 10f), Kind = PlacementOriginKind.Creature }
        };

        var report = CrossMapDivergenceEngine.AnalyzeDivergence(
            targetMapId: 451,
            targetMapName: "development",
            targetPlacements: targetMap,
            referenceMapId: 0,
            referenceMapName: "Azeroth",
            referencePlacements: overworld,
            spatialTolerance: 0.5f);

        Assert.Equal(451, report.TargetMapId);
        Assert.Equal(0, report.ReferenceMapId);
        Assert.Equal(5, report.TargetTotalPlacements);
        Assert.Equal(3, report.MatchedPlacementsCount);
        Assert.Equal(60.0, report.SharedPercentage, 1);
        Assert.Equal(2500, report.DivergenceCutoffUniqueId);

        var inherited = report.Matches.Where(m => m.Status == DivergenceStatus.InheritedFromOverworld).ToList();
        var postFork = report.Matches.Where(m => m.Status == DivergenceStatus.PostForkAuthored).ToList();

        Assert.Equal(3, inherited.Count);
        Assert.Equal(2, postFork.Count);

        Assert.Contains(inherited, m => m.TargetPlacement.UniqueId == 101 && m.UniqueIdMatched && m.SpatialMatched);
        Assert.Contains(inherited, m => m.TargetPlacement.UniqueId == 205 && m.SpatialMatched);
        Assert.Contains(inherited, m => m.TargetPlacement.UniqueId == 2500);

        Assert.Contains(postFork, m => m.TargetPlacement.UniqueId == 8000);
        Assert.Contains(postFork, m => m.TargetPlacement.UniqueId == 8050);
    }
}
