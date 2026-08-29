using System.Numerics;
using WowViewer.Core.IO.Maps;
using Xunit;

namespace WowViewer.Core.Tests;

public sealed class RosettaDatastoreTests
{
    private static RosettaAssetEntry Model(string path, float radius = 5f) =>
        new(path, RosettaAssetKind.Model, -new Vector3(radius), new Vector3(radius));

    private static RosettaAssetEntry WorldModel(string path, float radius = 25f) =>
        new(path, RosettaAssetKind.WorldModel, -new Vector3(radius), new Vector3(radius));

    [Fact]
    public void ComputeAssetId_IsDeterministicAndNormalized()
    {
        string id1 = RosettaDatastoreWriter.ComputeAssetId(@"Doodads\Trees\Pine.mdx");
        string id2 = RosettaDatastoreWriter.ComputeAssetId("doodads/trees/pine.mdx");
        string id3 = RosettaDatastoreWriter.ComputeAssetId("doodads/trees/pine.m2");

        Assert.NotEmpty(id1);
        Assert.StartsWith("objlib_", id1);
        Assert.Equal(id1, id2);
        Assert.NotEqual(id1, id3);
    }

    [Fact]
    public void IngestMap_MultiVersionDeduplicationAndTensorVerification()
    {
        string tempDatastore = Path.Combine(Path.GetTempPath(), "rosetta_test_datastore_" + Guid.NewGuid().ToString("N"));
        try
        {
            // Build 1: 0.5.3 client with assets A and B
            var assets053 = new List<RosettaAssetEntry>
            {
                Model("doodads/chest.mdx", 5f),
                WorldModel("buildings/house.wmo", 20f),
            };
            var options053 = new RosettaGeneratorOptions("RosettaAlpha");
            RosettaGenerationResult result053 = RosettaTilesetGenerator.Generate(assets053, options053);
            RosettaMapPlan map053 = result053.Maps[0];

            // Ingest Build 1
            RosettaDatastoreIngestResult ingest1 = RosettaDatastoreWriter.IngestMap(
                tempDatastore,
                "0_5_3_3368",
                @"H:\CLIENTS\Vanilla\0.x\0_5_3_3368",
                map053,
                (tileX, tileY) =>
                {
                    var tile = map053.Tiles.First(t => t.TileX == tileX && t.TileY == tileY);
                    float[] heights = new float[145];
                    Array.Fill(heights, 10.5f);
                    byte[] alpha = RosettaTilesetGenerator.BuildTileAlphaCanvas(tile, paintCellBorders: true);
                    byte[] checkers = RosettaTilesetGenerator.BuildTileCheckersCanvas(tile, options053.PedestalBevelMeters);
                    byte[] minimap = new byte[256 * 256 * 3];
                    return (heights, alpha, checkers, minimap);
                });

            Assert.Equal(2, ingest1.NewAssetsAdded);
            Assert.Equal(0, ingest1.DeduplicatedAssetsReused);
            Assert.Equal(map053.Placements.Count, ingest1.TotalPlacements);

            // Build 2: 1.12.1 client with asset A (reused), asset B (reused), and asset C (new)
            var assets112 = new List<RosettaAssetEntry>
            {
                Model("doodads/chest.mdx", 5f), // Reused
                WorldModel("buildings/house.wmo", 20f), // Reused
                Model("creature/murloc/murloc.m2", 8f), // New asset
            };
            var options112 = new RosettaGeneratorOptions("RosettaLk");
            RosettaGenerationResult result112 = RosettaTilesetGenerator.Generate(assets112, options112);
            RosettaMapPlan map112 = result112.Maps[0];

            // Ingest Build 2 into the same datastore
            RosettaDatastoreIngestResult ingest2 = RosettaDatastoreWriter.IngestMap(
                tempDatastore,
                "1_12_1_5875",
                @"H:\CLIENTS\Vanilla\1.12.1.5875",
                map112,
                (tileX, tileY) =>
                {
                    var tile = map112.Tiles.First(t => t.TileX == tileX && t.TileY == tileY);
                    float[] heights = new float[145];
                    Array.Fill(heights, 20.0f);
                    byte[] alpha = RosettaTilesetGenerator.BuildTileAlphaCanvas(tile, paintCellBorders: true);
                    byte[] checkers = RosettaTilesetGenerator.BuildTileCheckersCanvas(tile, options112.PedestalBevelMeters);
                    byte[] minimap = new byte[256 * 256 * 3];
                    return (heights, alpha, checkers, minimap);
                });

            Assert.Equal(1, ingest2.NewAssetsAdded);
            Assert.Equal(2, ingest2.DeduplicatedAssetsReused);

            // 3. Open with RosettaObjectLibrary and query
            RosettaObjectLibrary library = RosettaObjectLibrary.Open(tempDatastore);

            Assert.Equal(2, library.Builds.Count);
            Assert.Contains("0_5_3_3368", library.Builds);
            Assert.Contains("1_12_1_5875", library.Builds);
            Assert.Equal(3, library.TotalUniqueAssets);

            // Asset lookup
            RosettaGlobalAssetRecord? chest = library.LookupAsset("doodads/chest.mdx");
            Assert.NotNull(chest);
            Assert.Equal("doodads/chest.mdx", chest.NormalizedPath);
            Assert.Equal(2, chest.RefCount); // Seen in both builds!

            RosettaGlobalAssetRecord? murloc = library.LookupAsset("creature/murloc/murloc.m2");
            Assert.NotNull(murloc);
            Assert.Equal("1_12_1_5875", murloc.FirstSeenBuild);
            Assert.Equal(1, murloc.RefCount);

            // Placement queries
            IReadOnlyList<RosettaPlacementRecord> placements053 = library.GetPlacements("0_5_3_3368", "RosettaAlpha");
            Assert.Equal(2, placements053.Count);

            IReadOnlyList<RosettaPlacementRecord> placements112 = library.GetPlacements("1_12_1_5875", "RosettaLk");
            Assert.Equal(3, placements112.Count);

            // Spatial query
            IReadOnlyList<RosettaPlacementRecord> spatialResults = library.QuerySpatial(
                "0_5_3_3368", "RosettaAlpha", new Vector3(-100000, -100000, -100000), new Vector3(100000, 100000, 100000));
            Assert.Equal(2, spatialResults.Count);

            // Tensor chunk retrieval
            float[]? h053 = library.GetTileHeights("0_5_3_3368", "RosettaAlpha", 0);
            Assert.NotNull(h053);
            Assert.Equal(145, h053.Length);
            Assert.Equal(10.5f, h053[0]);

            float[]? h112 = library.GetTileHeights("1_12_1_5875", "RosettaLk", 0);
            Assert.NotNull(h112);
            Assert.Equal(145, h112.Length);
            Assert.Equal(20.0f, h112[0]);

            byte[]? alpha053 = library.GetTileAlpha("0_5_3_3368", "RosettaAlpha", 0);
            Assert.NotNull(alpha053);
            Assert.Equal(1024 * 1024, alpha053.Length);

            // Build metadata
            RosettaBuildMetadata? meta053 = library.GetBuildMetadata("0_5_3_3368");
            Assert.NotNull(meta053);
            Assert.Equal("0_5_3_3368", meta053.BuildId);
            Assert.Equal("RosettaAlpha", meta053.MapName);
            Assert.Equal(2, meta053.TotalUniqueAssets);

            // Cross-build diff
            RosettaBuildDiff diff = library.ComputeBuildDiff("0_5_3_3368", "1_12_1_5875");
            Assert.Equal("0_5_3_3368", diff.BaseBuildId);
            Assert.Equal("1_12_1_5875", diff.TargetBuildId);
            Assert.Equal(1, diff.AddedCount); // creature/murloc/murloc.m2 added
            Assert.Equal(0, diff.RemovedCount);
            Assert.Equal(2, diff.IdenticalCount); // chest and house identical
            Assert.Contains(diff.Records, r => r.Classification == RosettaDiffClassification.Added && r.TargetAssetPath == "creature/murloc/murloc.m2");
        }
        finally
        {
            if (Directory.Exists(tempDatastore))
                Directory.Delete(tempDatastore, recursive: true);
        }
    }
}
