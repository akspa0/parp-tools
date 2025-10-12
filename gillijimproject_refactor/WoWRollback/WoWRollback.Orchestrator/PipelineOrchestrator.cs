using System;
using System.Linq;
using System.IO;
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

        // Analyze-only: skip DBC and ADT conversion, synthesize ADT results pointing to provided directory
        IReadOnlyList<AdtStageResult> adtResults;
        DbcStageResult dbcResult;
        if (options.AnalyzeOnly || !string.IsNullOrWhiteSpace(options.AnalysisFromDir))
        {
            ConsoleLogger.Info("Analyze-only mode: skipping DBC and ADT stages");
            dbcResult = new DbcStageResult(true, Array.Empty<DbcVersionResult>());

            var sourceDir = options.AnalysisFromDir;
            if (string.IsNullOrWhiteSpace(sourceDir))
            {
                // fallback to overlay dir if provided
                sourceDir = options.AdtOverlayRoot;
            }
            if (string.IsNullOrWhiteSpace(sourceDir))
            {
                throw new InvalidOperationException("Analyze-only requires --analysis-from-dir or --adt-overlay-root pointing to a directory containing World/Maps/<Map>.");
            }
            sourceDir = Path.GetFullPath(sourceDir);

            var versionLabel = options.AnalysisVersionLabel ?? options.Versions.FirstOrDefault() ?? "analysis";
            var synthetic = new List<AdtStageResult>();
            foreach (var map in options.Maps)
            {
                var mapDir = Path.Combine(sourceDir, "World", "Maps", map);
                if (!Directory.Exists(mapDir))
                {
                    ConsoleLogger.Warn($"[analyze-only] Map directory not found: {mapDir}");
                }
                synthetic.Add(new AdtStageResult(
                    Map: map,
                    Version: versionLabel,
                    Success: true,
                    TilesProcessed: 0,
                    AreaIdsPatched: 0,
                    AdtOutputDirectory: sourceDir,
                    Error: null));
            }
            adtResults = synthetic;
        }
        else
        {
            ConsoleLogger.Info("Running DBC stage...");
            var dbcRunner = new DbcStageRunner();
            dbcResult = dbcRunner.Run(session);
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
            if (!string.IsNullOrWhiteSpace(options.AdtOverlayRoot))
            {
                ConsoleLogger.Info($"Materializing ADT overlay: {options.AdtOverlayRoot}");
                try
                {
                    LooseAdtMaterializer.Materialize(
                        session,
                        options.AdtOverlayRoot!,
                        options.AdtRoot ?? options.AlphaRoot,
                        options.Versions,
                        options.Maps);
                    ConsoleLogger.Success("ADT overlay materialization completed");
                }
                catch (Exception ex)
                {
                    ConsoleLogger.Warn($"ADT overlay materialization encountered issues: {ex.Message}");
                }
            }
            var adtRunner = new AdtStageRunner();
            adtResults = adtRunner.Run(session);
        }
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

    private static string? ResolveCommunityListfile(PipelineOptions options)
    {
        if (!string.IsNullOrWhiteSpace(options.CommunityListfile) && File.Exists(options.CommunityListfile)) return options.CommunityListfile;
        var c1 = Path.Combine(options.AlphaRoot, "listfile.csv");
        if (File.Exists(c1)) return c1;
        var c2 = Path.Combine(options.AlphaRoot, "community-listfile-withcapitals.csv");
        if (File.Exists(c2)) return c2;
        return null;
    }

    private static string? ResolveLkListfile(PipelineOptions options)
    {
        if (!string.IsNullOrWhiteSpace(options.LkListfile) && File.Exists(options.LkListfile)) return options.LkListfile;
        var c1 = Path.Combine(options.AlphaRoot, "lk_listfile.txt");
        return File.Exists(c1) ? c1 : null;
    }
}

internal sealed record PipelineRunResult(
    bool Success,
    SessionContext Session,
    DbcStageResult Dbc,
    IReadOnlyList<AdtStageResult> AdtResults,
    AnalysisStageResult Analysis,
    ViewerStageResult Viewer);
