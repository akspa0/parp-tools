using System.Numerics;
using System.Text.Json;
using WoWViewer.Workbench.Services.Pm4Workbench;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Matching;

namespace WowViewer.Core.Tests.PM4;

public sealed class Pm4ShapeMatchBatchServiceTests
{
    [Fact]
    public void RunBatch_ReconcilesMatchedLowConfidenceAndUnmatchedCountsWithExportedSummary()
    {
        var batchService = new Pm4ShapeMatchBatchService();
        var reviewService = new Pm4ShapeMatchReviewService();
        var project = new ManualCurationProject { MapName = "development" };
        var calibration = new Pm4CentroidCalibration(new Vector3(0.773f, 0f, 0f), "measured -X correction");

        var towerHull = new Pm4HullShapeInput(
            0, 0, 0x418D9F, 0, 0,
            Pm4ShapeLibraryVisualService.BuildSyntheticDescriptor(new Vector3(24.5f, 24.4f, 46.1f)) with
            {
                PrismCentroid = new Vector3(16358.5f, 16663.1f, 17.7f)
            });

        // Moderately different dimensions so it scores as low-confidence (< 0.70 when LowConfidenceThreshold = 0.75)
        var lowConfHull = new Pm4HullShapeInput(
            0, 0, 0x41AA27, 1, 0,
            Pm4ShapeLibraryVisualService.BuildSyntheticDescriptor(new Vector3(29.5f, 20.0f, 38.0f)) with
            {
                PrismCentroid = new Vector3(16400f, 16600f, 21.3f)
            });

        // Extreme dimensions that cannot match any candidate -> Unmatched
        var unmatchedHull = new Pm4HullShapeInput(
            0, 1, 0x420001, 2, 0,
            Pm4ShapeLibraryVisualService.BuildSyntheticDescriptor(new Vector3(180f, 3f, 95f)) with
            {
                PrismCentroid = new Vector3(16500f, 16700f, 30.0f)
            });

        var candidates = new[]
        {
            new Pm4ModelShapeCandidate(
                "World\\wmo\\Northrend\\Wintergrasp\\GuardTower_intact.wmo",
                "WMO",
                Pm4ShapeLibraryVisualService.BuildSyntheticDescriptor(new Vector3(24.5f, 24.4f, 46.1f))),
        };

        var tiles = new[]
        {
            new Pm4BatchTileInput(0, 0, [towerHull, lowConfHull]),
            new Pm4BatchTileInput(0, 1, [unmatchedHull]),
        };

        var scope = new Pm4BatchTileScope("development", 0, 1, 0, 1);
        var options = new Pm4ShapeMatchOptions(LowConfidenceThreshold: 0.85f);

        Pm4BatchAnalysisSummary summary = batchService.RunBatch(
            scope,
            tiles,
            candidates,
            project,
            reviewService,
            calibration,
            options);

        Assert.False(summary.WasCancelled);
        Assert.Equal(2, summary.ProcessedTilesCount);
        Assert.Equal(3, summary.TotalHulls);
        Assert.Equal(1, summary.MatchedCount);
        Assert.Equal(1, summary.LowConfidenceCount);
        Assert.Equal(1, summary.UnmatchedCount);
        Assert.Equal(
            summary.TotalHulls,
            summary.MatchedCount + summary.LowConfidenceCount + summary.UnmatchedCount);
        Assert.Equal(summary.TotalHulls, summary.Rows.Count);
        Assert.Equal(summary.MatchedCount, summary.Rows.Count(static r => r.Classification == "Matched"));
        Assert.Equal(summary.LowConfidenceCount, summary.Rows.Count(static r => r.Classification == "LowConfidence"));
        Assert.Equal(summary.UnmatchedCount, summary.Rows.Count(static r => r.Classification == "Unmatched"));

        string outputDir = Path.Combine(AppContext.BaseDirectory, "spec295-batch-export", Guid.NewGuid().ToString("N"));
        try
        {
            var (jsonPath, mdPath) = batchService.ExportSummaryFiles(summary, outputDir, "batch-test");
            Assert.True(File.Exists(jsonPath));
            Assert.True(File.Exists(mdPath));

            using var jsonDoc = JsonDocument.Parse(File.ReadAllText(jsonPath));
            Assert.Equal(summary.TotalHulls, jsonDoc.RootElement.GetProperty("TotalHulls").GetInt32());
            Assert.Equal(summary.MatchedCount, jsonDoc.RootElement.GetProperty("MatchedCount").GetInt32());
            Assert.Equal(summary.LowConfidenceCount, jsonDoc.RootElement.GetProperty("LowConfidenceCount").GetInt32());
            Assert.Equal(summary.UnmatchedCount, jsonDoc.RootElement.GetProperty("UnmatchedCount").GetInt32());
            Assert.Equal(summary.Rows.Count, jsonDoc.RootElement.GetProperty("Rows").GetArrayLength());

            string mdText = File.ReadAllText(mdPath);
            Assert.Contains("**Matched**: 1", mdText);
            Assert.Contains("**Low-Confidence**: 1", mdText);
            Assert.Contains("**Unmatched**: 1", mdText);
        }
        finally
        {
            if (Directory.Exists(outputDir))
                Directory.Delete(outputDir, recursive: true);
        }
    }

    [Fact]
    public void RunBatch_WhenCancelled_PreservesAlreadyAcceptedDecisions()
    {
        var batchService = new Pm4ShapeMatchBatchService();
        var reviewService = new Pm4ShapeMatchReviewService();
        var project = new ManualCurationProject { MapName = "development" };

        Pm4HullIdentity acceptedOnTile0 = new(0, 0, 0x418D9F, 0, 0);
        Pm4HullIdentity acceptedOnTile1 = new(0, 1, 0x40AA0A, 1, 0);

        Assert.True(ManualCurationProjectPolicy.RecordShapeMatchDecision(
            project,
            new ManualShapeMatchDecisionRecord(
                acceptedOnTile0,
                "World\\wmo\\Northrend\\Wintergrasp\\GuardTower_intact.wmo",
                "World\\wmo\\Northrend\\Wintergrasp\\GuardTower_intact.wmo",
                "WMO",
                ManualShapeMatchDecisionKind.Accepted,
                0.98f,
                0.02f,
                new Vector3(16359.3f, 16663.1f, 17.7f),
                "Manual",
                DateTimeOffset.UnixEpoch)));

        Assert.True(ManualCurationProjectPolicy.RecordShapeMatchDecision(
            project,
            new ManualShapeMatchDecisionRecord(
                acceptedOnTile1,
                "World\\wmo\\Northrend\\Wintergrasp\\GuardTower_intact.wmo",
                "World\\wmo\\Northrend\\Wintergrasp\\GuardTower_intact.wmo",
                "WMO",
                ManualShapeMatchDecisionKind.Accepted,
                0.97f,
                0.03f,
                new Vector3(16364.3f, 16760.8f, 5.3f),
                "Manual",
                DateTimeOffset.UnixEpoch)));

        var tile0Hull = new Pm4HullShapeInput(
            0, 0, 0x418D9F, 0, 0,
            Pm4ShapeLibraryVisualService.BuildSyntheticDescriptor(new Vector3(24.5f, 24.4f, 46.1f)));
        var tile1Hull = new Pm4HullShapeInput(
            0, 1, 0x40AA0A, 1, 0,
            Pm4ShapeLibraryVisualService.BuildSyntheticDescriptor(new Vector3(24.5f, 24.4f, 46.1f)));

        var candidates = new[]
        {
            new Pm4ModelShapeCandidate(
                "World\\wmo\\Northrend\\Wintergrasp\\GuardTower_intact.wmo",
                "WMO",
                Pm4ShapeLibraryVisualService.BuildSyntheticDescriptor(new Vector3(24.5f, 24.4f, 46.1f))),
        };

        var scope = new Pm4BatchTileScope("development", 0, 1, 0, 1);

        // Cancel immediately after tile index 0 (before tile 1 is processed)
        Pm4BatchAnalysisSummary summary = batchService.RunBatch(
            scope,
            [new Pm4BatchTileInput(0, 0, [tile0Hull]), new Pm4BatchTileInput(0, 1, [tile1Hull])],
            candidates,
            project,
            reviewService,
            shouldCancelAfterTileIndex: tileIdx => tileIdx == 1);

        Assert.True(summary.WasCancelled);
        Assert.Equal(1, summary.ProcessedTilesCount);
        Assert.Equal(2, summary.PreservedAcceptedCount);

        // Both accepted decisions in the project remain intact after cancellation
        Assert.True(ManualCurationProjectPolicy.TryGetActiveShapeMatchDecision(project, acceptedOnTile0, out var dec0));
        Assert.Equal(ManualShapeMatchDecisionKind.Accepted, dec0?.Decision);
        Assert.True(ManualCurationProjectPolicy.TryGetActiveShapeMatchDecision(project, acceptedOnTile1, out var dec1));
        Assert.Equal(ManualShapeMatchDecisionKind.Accepted, dec1?.Decision);
    }

    [Fact]
    public void FormatMarkdownSummary_ContainsZeroCk24AndRendersObjectCoordinates()
    {
        var batchService = new Pm4ShapeMatchBatchService();
        var reviewService = new Pm4ShapeMatchReviewService();
        var project = new ManualCurationProject { MapName = "development" };

        var hull = new Pm4HullShapeInput(
            22, 30, 0x46CCED, 0, 0x0808049E,
            Pm4ShapeLibraryVisualService.BuildSyntheticDescriptor(new Vector3(30f, 10f, 15f)) with
            {
                PrismCentroid = new Vector3(5103.7f, 652.6f, 6.9f)
            });

        var candidates = new[]
        {
            new Pm4ModelShapeCandidate(
                "WORLD\\WMO\\DUNGEON\\D_PIRATESHIP\\PIRATESHIP.WMO",
                "WMO",
                Pm4ShapeLibraryVisualService.BuildSyntheticDescriptor(new Vector3(30f, 10f, 15f))),
        };

        var scope = new Pm4BatchTileScope("development", 22, 22, 30, 30);
        Pm4BatchAnalysisSummary summary = batchService.RunBatch(
            scope,
            [new Pm4BatchTileInput(22, 30, [hull])],
            candidates,
            project,
            reviewService);

        string md = Pm4ShapeMatchBatchService.FormatMarkdownSummary(summary);

        Assert.DoesNotContain("CK24", md);
        Assert.Contains("Object ID", md);
        Assert.Contains("Position (X, Y, Z)", md);
        Assert.Contains("Proposed Candidate", md);
        Assert.Contains("`22_30`", md);
        Assert.Contains("`Obj #0`", md);
    }

    [Fact]
    public void TaughtPairing_TelemetrySerialization_PreservesAllAuthenticFields()
    {
        var original = new TaughtPairing(
            Ck24: 0x46CCED,
            DominantGroupKey: 0x12,
            ModelName: "PIRATESHIP.WMO",
            ModelPath: "WORLD\\WMO\\DUNGEON\\D_PIRATESHIP\\PIRATESHIP.WMO",
            AssetKind: "WMO",
            SolvedScale: 1.0f,
            YawDegrees: 15.1f,
            Length: 30f,
            Width: 10f,
            Height: 15f,
            TaughtUtc: DateTime.UtcNow,
            LocalBoundsMin: new Vector3(-15f, -5f, 0f),
            LocalBoundsMax: new Vector3(15f, 5f, 15f),
            YawOffset: 0f,
            VerticalSliceRadii: [1f, 0.8f, 0.5f],
            InertiaTensor: Vector3.Zero,
            NormalDistribution: Vector3.UnitZ,
            PlacementPosition: new Vector3(5103.7f, 652.6f, 6.9f),
            PlacementRotation: new Vector3(0f, 0f, 15.1f),
            PlacementScale: 1.0f,
            UniqueId: 9315669,
            TileX: 22,
            TileY: 30,
            ObjectPartId: 0,
            LinkGroupObjectId: 0x0808049E,
            HullCentroid: new Vector3(5103.7f, 652.6f, 6.9f),
            HullBoundsMin: new Vector3(5088.7f, 647.6f, -0.6f),
            HullBoundsMax: new Vector3(5118.7f, 657.6f, 14.4f));

        var options = new JsonSerializerOptions { IncludeFields = true };
        string json = JsonSerializer.Serialize(original, options);
        var deserialized = JsonSerializer.Deserialize<TaughtPairing>(json, options);

        Assert.NotNull(deserialized);
        Assert.Equal(original.ModelName, deserialized.ModelName);
        Assert.Equal(original.UniqueId, deserialized.UniqueId);
        Assert.Equal(original.TileX, deserialized.TileX);
        Assert.Equal(original.TileY, deserialized.TileY);
        Assert.Equal(original.ObjectPartId, deserialized.ObjectPartId);
        Assert.Equal(original.LinkGroupObjectId, deserialized.LinkGroupObjectId);
        Assert.Equal(original.PlacementPosition, deserialized.PlacementPosition);
        Assert.Equal(original.PlacementRotation, deserialized.PlacementRotation);
        Assert.Equal(original.PlacementScale, deserialized.PlacementScale);
        Assert.Equal(original.HullCentroid, deserialized.HullCentroid);
        Assert.Equal(original.HullBoundsMin, deserialized.HullBoundsMin);
        Assert.Equal(original.HullBoundsMax, deserialized.HullBoundsMax);
    }
}

