namespace WowViewer.Core.Runtime.World;

/// <summary>
/// A stable, serializable summary of production WorldScene frame timing samples.
/// The report deliberately carries CPU stage measurements and workload counts separately:
/// a frame can be slow because it is doing too much work, not merely because one stage is slow.
/// </summary>
public sealed record WorldRenderDiagnosticReport(
    string Schema,
    string Renderer,
    int WarmupFrameCount,
    IReadOnlyList<WorldRenderDiagnosticFrame> Frames,
    WorldRenderDiagnosticWorkload Workload,
    IReadOnlyList<WorldRenderDiagnosticStageSummary> Stages,
    IReadOnlyList<WorldRenderDiagnosticFinding> Findings);

public readonly record struct WorldRenderDiagnosticFrame(int FrameIndex, WorldRenderFrameStats Stats);

public readonly record struct WorldRenderDiagnosticWorkload(
    double SceneInitializationMs,
    int PendingAssetLoadCount,
    int PendingTerrainLoadCount,
    int PendingDeferredWmoDoodadLoadCount,
    int LoadedMdxModelCount,
    int LoadedWmoModelCount,
    int MdxInstanceCount,
    int WmoInstanceCount,
    bool HierarchicalSceneTraversalActive,
    int? TargetTileX,
    int? TargetTileY,
    long DataSourceReadRequestCount,
    long DataSourceReadCacheHitCount,
    long DataSourceReadCacheMissCount,
    double AverageUncachedReadMs);

public readonly record struct WorldRenderDiagnosticStageSummary(
    string Name,
    int SampleCount,
    double AverageDurationMs,
    double P95DurationMs,
    double MaxDurationMs,
    int MaxVisibleCount,
    int MaxSubmittedCount);

public readonly record struct WorldRenderDiagnosticFinding(
    string Code,
    string Severity,
    string Detail);

public static class WorldRenderDiagnostics
{
    public const string Schema = "world-render-diagnostic-v1";

    public static WorldRenderDiagnosticReport Build(
        string renderer,
        int warmupFrameCount,
        IReadOnlyList<WorldRenderDiagnosticFrame> frames,
        WorldRenderDiagnosticWorkload workload)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(renderer);
        ArgumentOutOfRangeException.ThrowIfNegative(warmupFrameCount);
        ArgumentNullException.ThrowIfNull(frames);

        WorldRenderDiagnosticStageSummary[] stages = StageDefinitions
            .Select(definition => Summarize(definition.Name, frames, definition.Select))
            .ToArray();
        WorldRenderDiagnosticFinding[] findings = BuildFindings(frames, workload, stages).ToArray();
        return new WorldRenderDiagnosticReport(Schema, renderer, warmupFrameCount, frames, workload, stages, findings);
    }

    private static IEnumerable<WorldRenderDiagnosticFinding> BuildFindings(
        IReadOnlyList<WorldRenderDiagnosticFrame> frames,
        WorldRenderDiagnosticWorkload workload,
        IReadOnlyList<WorldRenderDiagnosticStageSummary> stages)
    {
        if (frames.Count == 0)
        {
            yield return new("no-measured-frames", "error", "The renderer produced no measured frames.");
            yield break;
        }

        double p95FrameMs = Percentile(frames.Select(static frame => frame.Stats.TotalCpuMs).ToArray(), 0.95);
        if (p95FrameMs >= 1000.0)
            yield return new("cpu-frame-stall", "error", $"P95 production render CPU time is {p95FrameMs:F1} ms.");
        else if (p95FrameMs >= 33.0)
            yield return new("cpu-frame-budget-exceeded", "warning", $"P95 production render CPU time is {p95FrameMs:F1} ms (over the 30 FPS budget).");

        if (workload.SceneInitializationMs >= 1000.0)
            yield return new("scene-initialization-stall", "warning", $"Production WorldScene initialization took {workload.SceneInitializationMs:F1} ms before frame sampling.");

        if (workload.DataSourceReadCacheMissCount > workload.DataSourceReadCacheHitCount
            && workload.AverageUncachedReadMs >= 10.0)
        {
            yield return new(
                "client-read-pressure",
                "warning",
                $"Client reads had {workload.DataSourceReadCacheMissCount} cache misses versus {workload.DataSourceReadCacheHitCount} hits at {workload.AverageUncachedReadMs:F1} ms average uncached latency.");
        }

        if (workload.PendingTerrainLoadCount > 0 || workload.PendingAssetLoadCount > 0 || workload.PendingDeferredWmoDoodadLoadCount > 0)
        {
            yield return new(
                "scene-not-settled",
                "warning",
                $"The final sample still has terrain={workload.PendingTerrainLoadCount}, asset={workload.PendingAssetLoadCount}, and WMO-doodad={workload.PendingDeferredWmoDoodadLoadCount} pending loads.");
        }

        if (workload.WmoInstanceCount > 0 && frames.All(static frame => frame.Stats.VisibleWmoCount == 0))
            yield return new("wmo-path-not-covered", "warning", "The scene owns WMO placements, but no measured camera frame admitted a WMO.");
        if (workload.MdxInstanceCount > 0 && frames.All(static frame => frame.Stats.VisibleMdxCount == 0))
            yield return new("mdx-path-not-covered", "warning", "The scene owns MDX placements, but no measured camera frame admitted an MDX instance.");

        WorldRenderDiagnosticStageSummary? dominant = stages
            .Where(static stage => stage.Name is not "deferred_asset_loads")
            .OrderByDescending(static stage => stage.P95DurationMs)
            .FirstOrDefault();
        if (dominant is { P95DurationMs: > 1.0 })
            yield return new("dominant-cpu-stage", "info", $"P95 dominant CPU stage is {dominant.Value.Name} at {dominant.Value.P95DurationMs:F1} ms.");

        yield return new(
            "gpu-timing-not-yet-attributed",
            "info",
            "This report executes the production OpenGL path and attributes every existing CPU frame stage; GPU timer-query attribution remains a separate proof gap.");
    }

    private static WorldRenderDiagnosticStageSummary Summarize(
        string name,
        IReadOnlyList<WorldRenderDiagnosticFrame> frames,
        Func<WorldRenderFrameStats, WorldRenderStageStats> select)
    {
        if (frames.Count == 0)
            return new(name, 0, 0, 0, 0, 0, 0);

        WorldRenderStageStats[] samples = frames.Select(frame => select(frame.Stats)).ToArray();
        double[] durations = samples.Select(static sample => sample.DurationMs).ToArray();
        return new WorldRenderDiagnosticStageSummary(
            name,
            samples.Length,
            durations.Average(),
            Percentile(durations, 0.95),
            durations.Max(),
            samples.Max(static sample => sample.VisibleCount),
            samples.Max(static sample => sample.SubmittedCount));
    }

    private static double Percentile(double[] values, double percentile)
    {
        if (values.Length == 0)
            return 0;

        Array.Sort(values);
        int index = (int)Math.Ceiling((values.Length - 1) * percentile);
        return values[index];
    }

    private sealed record StageDefinition(string Name, Func<WorldRenderFrameStats, WorldRenderStageStats> Select);

    private static readonly StageDefinition[] StageDefinitions =
    [
        new("deferred_asset_loads", static stats => stats.DeferredAssetLoads),
        new("taxi_actor_update", static stats => stats.TaxiActorUpdate),
        new("lighting", static stats => stats.Lighting),
        new("sky", static stats => stats.Sky),
        new("skybox_backdrop", static stats => stats.SkyboxBackdrop),
        new("wdl", static stats => stats.Wdl),
        new("terrain", static stats => stats.Terrain),
        new("wmo_visibility", static stats => stats.WmoVisibility),
        new("wmo_submission", static stats => stats.WmoSubmission),
        new("wmo_transparent_submission", static stats => stats.WmoTransparentSubmission),
        new("mdx_animation", static stats => stats.MdxAnimation),
        new("mdx_visibility", static stats => stats.MdxVisibility),
        new("mdx_opaque_submission", static stats => stats.MdxOpaqueSubmission),
        new("liquid", static stats => stats.Liquid),
        new("mdx_transparent_sort", static stats => stats.MdxTransparentSort),
        new("mdx_transparent_submission", static stats => stats.MdxTransparentSubmission),
        new("overlay", static stats => stats.Overlay),
    ];
}
