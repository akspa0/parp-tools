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
}

