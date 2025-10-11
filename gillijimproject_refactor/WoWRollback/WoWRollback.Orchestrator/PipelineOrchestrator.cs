using System;
using System.Linq;
using WoWRollback.Core.Logging;

namespace WoWRollback.Orchestrator;

internal sealed class PipelineOrchestrator
{
    public PipelineRunResult Run(PipelineOptions options)
    {
        if (options is null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        ConsoleLogger.Info("Creating session...");
        var session = SessionManager.CreateSession(options);
        ConsoleLogger.Success($"Session created: {session.SessionId}");
        ConsoleLogger.Info($"Output directory: {session.Root}");
        Console.WriteLine();

        ConsoleLogger.Info("Running DBC stage...");
        var dbcRunner = new DbcStageRunner();
        var dbcResult = dbcRunner.Run(session);
        if (dbcResult.Success)
        {
            ConsoleLogger.Success($"DBC stage complete: {dbcResult.Versions.Count} version(s) processed");
        }
        else
        {
            ConsoleLogger.Error("DBC stage failed");
            foreach (var v in dbcResult.Versions)
            {
                if (v.Error != null)
                {
                    ConsoleLogger.Error($"  {v.SourceVersion}: {v.Error}");
                }
            }
        }
        Console.WriteLine();

        ConsoleLogger.Info("Running ADT conversion stage...");
        var adtRunner = new AdtStageRunner();
        var adtResults = adtRunner.Run(session);
        var adtSuccessCount = adtResults.Count(r => r.Success);
        if (adtSuccessCount == adtResults.Count)
        {
            ConsoleLogger.Success($"ADT stage complete: {adtResults.Count} map(s) converted");
        }
        else
        {
            ConsoleLogger.Warn($"ADT stage partial: {adtSuccessCount}/{adtResults.Count} succeeded");
            foreach (var r in adtResults.Where(r => !r.Success))
            {
                ConsoleLogger.Error($"  {r.Map} ({r.Version}): {r.Error}");
            }
        }
        
        // Show ADT statistics
        foreach (var r in adtResults.Where(r => r.Success))
        {
            ConsoleLogger.Info($"  {r.Map} ({r.Version}): {r.TilesProcessed} tiles, {r.AreaIdsPatched} area IDs patched");
        }
        Console.WriteLine();

        NoggitProjectWriter.Write(session, adtResults);
        Console.WriteLine();

        // Stage 3: Analysis
        ConsoleLogger.Info("Running analysis stage...");
        var analysisRunner = new AnalysisStageRunner();
        var analysisResult = analysisRunner.Run(session, adtResults);
        if (analysisResult.Success)
        {
            var totalOverlays = analysisResult.PerVersion.Sum(v => v.Result.OverlayCount);
            ConsoleLogger.Success($"Analysis complete: {totalOverlays} overlays generated");
        }
        else
        {
            ConsoleLogger.Warn($"Analysis stage had issues: {analysisResult.ErrorMessage}");
        }
        Console.WriteLine();

        // Stage 4: Viewer (self-contained)
        ConsoleLogger.Info("Generating viewer...");
        var viewerRunner = new ViewerBuildStageRunner();
        var viewerResult = viewerRunner.Run(session, adtResults);
        if (viewerResult.Success)
        {
            ConsoleLogger.Success($"Viewer generated: {viewerResult.ViewerDirectory}");
        }
        else
        {
            ConsoleLogger.Error($"Viewer generation failed: {viewerResult.Notes}");
        }
        Console.WriteLine();

        ConsoleLogger.Info("Writing manifest...");
        var manifestWriter = new ManifestWriter();
        var pipelineResult = new PipelineRunResult(
            dbcResult.Success && viewerResult.Success && AllSucceeded(adtResults),
            session,
            dbcResult,
            adtResults,
            analysisResult,
            viewerResult);
        manifestWriter.Write(session, pipelineResult);
        ConsoleLogger.Success($"Manifest written: {session.ManifestPath}");
        Console.WriteLine();

        return pipelineResult;
    }

    private static bool AllSucceeded(IReadOnlyList<AdtStageResult> results)
    {
        foreach (var result in results)
        {
            if (!result.Success)
            {
                return false;
            }
        }

        return true;
    }
}

internal sealed record PipelineRunResult(
    bool Success,
    SessionContext Session,
    DbcStageResult Dbc,
    IReadOnlyList<AdtStageResult> AdtResults,
    AnalysisStageResult Analysis,
    ViewerStageResult Viewer);
