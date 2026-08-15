using WowViewer.Core.Runtime.World;

namespace WowViewer.Core.Tests;

public sealed class WorldRenderDiagnosticsTests
{
    [Fact]
    public void Build_ReportsEveryProductionStageAndDominantCpuStage()
    {
        WorldRenderFrameStats stats = WorldRenderFrameStats.Empty with
        {
            TotalCpuMs = 48,
            WmoSubmission = new WorldRenderStageStats(24, 4, 12),
            MdxOpaqueSubmission = new WorldRenderStageStats(6, 10, 10),
        };

        WorldRenderDiagnosticReport report = WorldRenderDiagnostics.Build(
            "headless-production-world-scene",
            warmupFrameCount: 2,
            [new WorldRenderDiagnosticFrame(0, stats), new WorldRenderDiagnosticFrame(1, stats)],
            new WorldRenderDiagnosticWorkload(12, 0, 0, 0, 3, 2, 10, 4, true, 32, 32, 9, 5, 4, 1.5));

        Assert.Equal(WorldRenderDiagnostics.Schema, report.Schema);
        Assert.Equal(18, report.Stages.Count);
        Assert.All(report.Stages, stage => Assert.Equal(2, stage.SampleCount));
        Assert.Contains(report.Findings, finding => finding.Code == "cpu-frame-budget-exceeded");
        Assert.Contains(report.Findings, finding => finding.Code == "dominant-cpu-stage" && finding.Detail.Contains("wmo_submission"));
    }

    [Fact]
    public void Build_ReportsUnsettledAndUncoveredObjectPaths()
    {
        WorldRenderDiagnosticReport report = WorldRenderDiagnostics.Build(
            "headless-production-world-scene",
            warmupFrameCount: 0,
            [new WorldRenderDiagnosticFrame(0, WorldRenderFrameStats.Empty)],
            new WorldRenderDiagnosticWorkload(0, 2, 3, 4, 0, 0, 7, 8, false, null, null, 0, 0, 0, 0));

        Assert.Contains(report.Findings, finding => finding.Code == "scene-not-settled");
        Assert.Contains(report.Findings, finding => finding.Code == "wmo-path-not-covered");
        Assert.Contains(report.Findings, finding => finding.Code == "mdx-path-not-covered");
    }

    [Fact]
    public void Build_ReportsOverlayOwnersAndReconcilesCoarseOverlayDuration()
    {
        WorldOverlayOwnerFrameStats[] owners =
        [
            new(WorldOverlayOwners.ObjectWireframe, 7, true, 12, 10, "not_cached", 0),
            WorldOverlayOwnerFrameStats.Disabled(WorldOverlayOwners.SelectionBounds),
            WorldOverlayOwnerFrameStats.Disabled(WorldOverlayOwners.Pm4Bounds),
            new(WorldOverlayOwners.Pm4GeometryPrepare, 2, true, 40, 0, "not_cached", 1),
            WorldOverlayOwnerFrameStats.Disabled(WorldOverlayOwners.Pm4GeometrySubmit),
            WorldOverlayOwnerFrameStats.Disabled(WorldOverlayOwners.Pm4Nodes),
            WorldOverlayOwnerFrameStats.Disabled(WorldOverlayOwners.PoiTaxi),
            WorldOverlayOwnerFrameStats.Disabled(WorldOverlayOwners.AreaTriggers),
            WorldOverlayOwnerFrameStats.Disabled(WorldOverlayOwners.AudioEmitters),
            WorldOverlayOwnerFrameStats.Disabled(WorldOverlayOwners.OtherOverlay),
        ];
        WorldRenderFrameStats stats = WorldRenderFrameStats.Empty with
        {
            Overlay = new WorldRenderStageStats(9),
            OverlayOwners = owners,
        };

        WorldRenderDiagnosticReport report = WorldRenderDiagnostics.Build(
            "headless-production-world-scene",
            warmupFrameCount: 0,
            [new WorldRenderDiagnosticFrame(0, stats)],
            new WorldRenderDiagnosticWorkload(0, 0, 0, 0, 0, 0, 0, 0, false, null, null, 0, 0, 0, 0));

        Assert.Equal(WorldOverlayOwners.All.Count, report.OverlayOwners.Count);
        Assert.Equal(stats.Overlay.DurationMs, stats.OverlayOwners.Sum(static owner => owner.DurationMs));
        Assert.Contains(report.OverlayOwners, owner =>
            owner.OwnerId == WorldOverlayOwners.ObjectWireframe
            && owner.P95DurationMs == 7
            && owner.MaxSubmittedPrimitiveCount == 10);
        Assert.Contains(report.Findings, finding =>
            finding.Code == "dominant-cpu-stage" && finding.Detail.Contains("overlay"));
        Assert.Contains(report.Findings, finding =>
            finding.Code == "dominant-overlay-owner" && finding.Detail.Contains(WorldOverlayOwners.ObjectWireframe));

        WorldOverlayOwnerFrameStats[] disabledOwners = stats.OverlayOwners
            .Where(static owner => !owner.Enabled)
            .ToArray();
        Assert.Equal(WorldOverlayOwners.All.Count - 2, disabledOwners.Length);
        Assert.All(disabledOwners, owner =>
        {
            Assert.Equal(0, owner.PreparedPrimitiveCount);
            Assert.Equal(0, owner.SubmittedPrimitiveCount);
            Assert.Equal(0, owner.DeferredCount);
            Assert.Equal(0, owner.DurationMs);
        });
    }
}
