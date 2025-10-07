namespace WoWRollback.AnalysisModule;

/// <summary>
/// High-level orchestrator for analysis operations.
/// Coordinates UniqueID analysis, terrain CSV generation, overlay creation, and manifest building.
/// </summary>
public sealed class AnalysisOrchestrator
{
    /// <summary>
    /// Runs complete analysis pipeline on converted ADT files.
    /// </summary>
    /// <param name="adtOutputDir">Directory containing converted LK ADTs</param>
    /// <param name="analysisOutputDir">Output directory for analysis CSVs</param>
    /// <param name="viewerOutputDir">Output directory for viewer overlays</param>
    /// <param name="mapName">Map name</param>
    /// <param name="version">Version string</param>
    /// <param name="opts">Analysis options</param>
    /// <returns>Result containing paths to generated files</returns>
    public AnalysisResult RunAnalysis(
        string adtOutputDir,
        string analysisOutputDir,
        string viewerOutputDir,
        string mapName,
        string version,
        AnalysisOptions opts)
    {
        try
        {
            var uniqueIdCsvs = new List<string>();
            var terrainCsvs = new List<string>();
            var overlayCount = 0;
            string? manifestPath = null;

            // Find ADT directory for this map
            var adtMapDir = Path.Combine(adtOutputDir, "World", "Maps", mapName);
            if (!Directory.Exists(adtMapDir) || !Directory.EnumerateFiles(adtMapDir, "*.adt").Any())
            {
                return new AnalysisResult(
                    UniqueIdCsvs: Array.Empty<string>(),
                    TerrainCsvs: Array.Empty<string>(),
                    OverlayCount: 0,
                    ManifestPath: null,
                    Success: false,
                    ErrorMessage: $"No ADT files found in {adtMapDir}");
            }

            // Stage 1: UniqueID Analysis
            if (opts.GenerateUniqueIdCsvs)
            {
                var uniqueIdDir = Path.Combine(analysisOutputDir, "uniqueids");
                Directory.CreateDirectory(uniqueIdDir);

                var analyzer = new UniqueIdAnalyzer(opts.UniqueIdGapThreshold);
                var uniqueIdResult = analyzer.Analyze(adtMapDir, mapName, uniqueIdDir);

                if (uniqueIdResult.Success)
                {
                    uniqueIdCsvs.Add(uniqueIdResult.CsvPath);
                    uniqueIdCsvs.Add(uniqueIdResult.LayersJsonPath);
                }
            }

            // Stage 2: Terrain CSV Generation
            if (opts.GenerateTerrainCsvs)
            {
                var terrainDir = Path.Combine(analysisOutputDir, "terrain");
                Directory.CreateDirectory(terrainDir);

                var generator = new TerrainCsvGenerator();
                var terrainResult = generator.Generate(adtMapDir, mapName, terrainDir);

                if (terrainResult.Success)
                {
                    terrainCsvs.Add(terrainResult.TerrainCsvPath);
                    terrainCsvs.Add(terrainResult.PropertiesCsvPath);
                }
            }

            // Stage 3: Overlay Generation
            if (opts.GenerateOverlays)
            {
                var overlayGenerator = new OverlayGenerator();
                var overlayResult = overlayGenerator.Generate(
                    adtMapDir,
                    viewerOutputDir,
                    mapName,
                    version);

                if (overlayResult.Success)
                {
                    overlayCount = overlayResult.TilesProcessed;
                }
            }

            // Stage 4: Manifest Building
            if (opts.GenerateManifest)
            {
                var manifestBuilder = new OverlayManifestBuilder();
                var manifestResult = manifestBuilder.Build(
                    viewerOutputDir,
                    mapName,
                    version);

                if (manifestResult.Success)
                {
                    manifestPath = manifestResult.ManifestPath;
                }
            }

            return new AnalysisResult(
                UniqueIdCsvs: uniqueIdCsvs,
                TerrainCsvs: terrainCsvs,
                OverlayCount: overlayCount,
                ManifestPath: manifestPath,
                Success: true);
        }
        catch (Exception ex)
        {
            return new AnalysisResult(
                UniqueIdCsvs: Array.Empty<string>(),
                TerrainCsvs: Array.Empty<string>(),
                OverlayCount: 0,
                ManifestPath: null,
                Success: false,
                ErrorMessage: $"Analysis failed: {ex.Message}");
        }
    }
}
