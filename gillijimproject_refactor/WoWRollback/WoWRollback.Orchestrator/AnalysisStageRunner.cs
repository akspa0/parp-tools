using WoWRollback.AnalysisModule;
using WoWRollback.Core.Logging;

namespace WoWRollback.Orchestrator;

/// <summary>
/// Runs the Analysis stage - generates UniqueID CSVs, terrain metadata, and viewer overlays.
/// </summary>
internal sealed class AnalysisStageRunner
{
    /// <summary>
    /// Executes analysis for all successfully converted ADT results.
    /// </summary>
    public AnalysisStageResult Run(
        SessionContext session,
        IReadOnlyList<AdtStageResult> adtResults)
    {
        ConsoleLogger.Info("=== Stage 3: Analysis ===");

        var results = new List<VersionAnalysisResult>();
        var overallSuccess = true;

        foreach (var adtResult in adtResults.Where(r => r.Success))
        {
            ConsoleLogger.Info($"Analyzing {adtResult.Map} ({adtResult.Version})...");

            var adtOutputDir = Path.Combine(session.Paths.AdtDir, adtResult.Version);
            var analysisOutputDir = Path.Combine(session.Paths.AnalysisDir, adtResult.Version);
            var viewerOutputDir = session.Paths.ViewerDir;

            Directory.CreateDirectory(analysisOutputDir);

            var orchestrator = new AnalysisOrchestrator();
            var analysisOptions = new AnalysisOptions
            {
                GenerateUniqueIdCsvs = true,
                GenerateTerrainCsvs = true,
                // Disable legacy overlays under 05_viewer/overlays to avoid confusion; viewer consumes data/overlays only.
                GenerateOverlays = false,
                GenerateManifest = true,
                UniqueIdGapThreshold = 100,
                Verbose = session.Options.Verbose
            };

            var result = orchestrator.RunAnalysis(
                adtOutputDir,
                analysisOutputDir,
                viewerOutputDir,
                adtResult.Map,
                adtResult.Version,
                analysisOptions);

            if (result.Success)
            {
                ConsoleLogger.Success($"  ✓ UniqueID CSVs: {result.UniqueIdCsvs.Count}");
                ConsoleLogger.Success($"  ✓ Terrain CSVs: {result.TerrainCsvs.Count}");
                ConsoleLogger.Success($"  ✓ Overlays: {result.OverlayCount} tiles");
                if (result.ManifestPath != null)
                {
                    ConsoleLogger.Success($"  ✓ Manifest: {Path.GetFileName(result.ManifestPath)}");
                }
            }
            else
            {
                ConsoleLogger.Error($"  ✗ Analysis failed: {result.ErrorMessage}");
                overallSuccess = false;
            }

            results.Add(new VersionAnalysisResult(
                adtResult.Version,
                adtResult.Map,
                result));
        }

        if (results.Count == 0)
        {
            ConsoleLogger.Warn("No ADT results to analyze");
            return new AnalysisStageResult(
                Array.Empty<VersionAnalysisResult>(),
                Success: false,
                ErrorMessage: "No ADT results available");
        }

        return new AnalysisStageResult(
            results,
            Success: overallSuccess);
    }
}

/// <summary>
/// Result from analysis stage.
/// </summary>
internal sealed record AnalysisStageResult(
    IReadOnlyList<VersionAnalysisResult> PerVersion,
    bool Success,
    string? ErrorMessage = null);

/// <summary>
/// Analysis result for a specific version/map.
/// </summary>
internal sealed record VersionAnalysisResult(
    string Version,
    string Map,
    AnalysisResult Result);
