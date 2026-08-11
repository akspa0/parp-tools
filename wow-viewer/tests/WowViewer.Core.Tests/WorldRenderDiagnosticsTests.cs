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
            new WorldRenderDiagnosticWorkload(12, 0, 0, 0, 3, 2, 10, 4, true, 9, 5, 4, 1.5));

        Assert.Equal(WorldRenderDiagnostics.Schema, report.Schema);
        Assert.Equal(17, report.Stages.Count);
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
            new WorldRenderDiagnosticWorkload(0, 2, 3, 4, 0, 0, 7, 8, false, 0, 0, 0, 0));

        Assert.Contains(report.Findings, finding => finding.Code == "scene-not-settled");
        Assert.Contains(report.Findings, finding => finding.Code == "wmo-path-not-covered");
        Assert.Contains(report.Findings, finding => finding.Code == "mdx-path-not-covered");
    }
}
