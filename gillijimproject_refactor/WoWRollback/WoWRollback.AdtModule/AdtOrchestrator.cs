using AlphaWdtAnalyzer.Core.Export;

namespace WoWRollback.AdtModule;

/// <summary>
/// High-level orchestrator for ADT conversion operations, wrapping AlphaWdtAnalyzer.Core as a library API.
/// Provides clean interface for Alpha to Lich King ADT conversion.
/// </summary>
public sealed class AdtOrchestrator
{
    /// <summary>
    /// Converts Alpha WDT/ADT files to Lich King format with AreaID patching.
    /// </summary>
    /// <param name="wdtPath">Path to source Alpha WDT file</param>
    /// <param name="exportDir">Output directory for converted ADTs</param>
    /// <param name="mapName">Map name (derived from WDT filename)</param>
    /// <param name="srcAlias">Source version alias (e.g., "0.5.3")</param>
    /// <param name="opts">Conversion options</param>
    /// <returns>Result containing paths and statistics</returns>
    public AdtConversionResult ConvertAlphaToLk(
        string wdtPath,
        string exportDir,
        string mapName,
        string srcAlias,
        ConversionOptions opts)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(wdtPath))
                throw new ArgumentException("WDT path is required", nameof(wdtPath));
            if (string.IsNullOrWhiteSpace(exportDir))
                throw new ArgumentException("Export directory is required", nameof(exportDir));
            if (string.IsNullOrWhiteSpace(mapName))
                throw new ArgumentException("Map name is required", nameof(mapName));

            if (!File.Exists(wdtPath))
                return new AdtConversionResult(
                    AdtOutputDirectory: exportDir,
                    TerrainCsvPath: null,
                    ShadowCsvPath: null,
                    TilesProcessed: 0,
                    AreaIdsPatched: 0,
                    Success: false,
                    ErrorMessage: $"WDT file not found: {wdtPath}");

            // Create output directory
            Directory.CreateDirectory(exportDir);

            // Build pipeline options from our cleaner API
            var pipelineOptions = new AdtExportPipeline.Options
            {
                SingleWdtPath = wdtPath,
                CommunityListfilePath = opts.CommunityListfilePath,
                LkListfilePath = opts.LkListfilePath,
                ExportDir = exportDir,
                FallbackTileset = opts.FallbackTileset,
                FallbackNonTilesetBlp = opts.FallbackNonTilesetBlp,
                FallbackWmo = opts.FallbackWmo,
                FallbackM2 = opts.FallbackM2,
                ConvertToMh2o = opts.ConvertToMh2o,
                AssetFuzzy = opts.AssetFuzzy,
                UseFallbacks = opts.UseFallbacks,
                EnableFixups = opts.EnableFixups,
                RemapPath = null,
                Verbose = opts.Verbose,
                TrackAssets = false,
                DbdDir = opts.DbdDir,
                DbctoolOutRoot = opts.CrosswalkDir,
                DbctoolSrcAlias = srcAlias,
                DbctoolSrcDir = null, // Will be resolved by pipeline if needed
                DbctoolLkDir = opts.LkDbcDir,
                DbctoolPatchDir = opts.CrosswalkDir,
                DbctoolPatchFile = null,
                VizSvg = false,
                VizHtml = false,
                VizDir = null,
                PatchOnly = false,
                NoZoneFallback = false,
            };

            // Execute conversion
            AdtExportPipeline.ExportSingle(pipelineOptions);

            // Count results
            var tilesProcessed = CountGeneratedTiles(exportDir, mapName);
            var areaIdsPatched = CountPatchedAreaIds(exportDir, mapName);

            // Compute CSV paths
            var csvDir = Path.Combine(exportDir, "csv", "maps", mapName);
            var terrainCsvPath = Path.Combine(csvDir, "terrain.csv");
            var shadowCsvPath = Path.Combine(csvDir, "shadow.csv");

            return new AdtConversionResult(
                AdtOutputDirectory: exportDir,
                TerrainCsvPath: File.Exists(terrainCsvPath) ? terrainCsvPath : null,
                ShadowCsvPath: File.Exists(shadowCsvPath) ? shadowCsvPath : null,
                TilesProcessed: tilesProcessed,
                AreaIdsPatched: areaIdsPatched,
                Success: true);
        }
        catch (Exception ex)
        {
            return new AdtConversionResult(
                AdtOutputDirectory: exportDir,
                TerrainCsvPath: null,
                ShadowCsvPath: null,
                TilesProcessed: 0,
                AreaIdsPatched: 0,
                Success: false,
                ErrorMessage: $"ADT conversion failed: {ex.Message}");
        }
    }

    /// <summary>
    /// Extracts terrain CSV data from already-converted Lich King ADTs.
    /// Note: This is a placeholder - actual implementation depends on available WoWRollback.Cli logic.
    /// </summary>
    /// <param name="lkAdtDir">Directory containing Lich King ADT files</param>
    /// <param name="mapName">Map name</param>
    /// <param name="csvOutDir">Output directory for CSV files</param>
    /// <returns>Result containing paths to generated CSVs</returns>
    public TerrainCsvResult ExtractTerrainFromLkAdts(
        string lkAdtDir,
        string mapName,
        string csvOutDir)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(lkAdtDir))
                throw new ArgumentException("LK ADT directory is required", nameof(lkAdtDir));
            if (string.IsNullOrWhiteSpace(mapName))
                throw new ArgumentException("Map name is required", nameof(mapName));
            if (string.IsNullOrWhiteSpace(csvOutDir))
                throw new ArgumentException("CSV output directory is required", nameof(csvOutDir));

            if (!Directory.Exists(lkAdtDir))
                return new TerrainCsvResult(
                    string.Empty,
                    string.Empty,
                    Success: false,
                    ErrorMessage: $"LK ADT directory not found: {lkAdtDir}");

            Directory.CreateDirectory(csvOutDir);

            // TODO: Implement terrain extraction from existing LK ADTs
            // This would use WoWRollback.Cli logic if available
            // For now, return not implemented

            return new TerrainCsvResult(
                string.Empty,
                string.Empty,
                Success: false,
                ErrorMessage: "Terrain extraction from LK ADTs not yet implemented");
        }
        catch (Exception ex)
        {
            return new TerrainCsvResult(
                string.Empty,
                string.Empty,
                Success: false,
                ErrorMessage: $"Terrain extraction failed: {ex.Message}");
        }
    }

    private static int CountGeneratedTiles(string exportDir, string mapName)
    {
        var mapDir = Path.Combine(exportDir, "World", "Maps", mapName);
        if (!Directory.Exists(mapDir))
            return 0;

        return Directory.EnumerateFiles(mapDir, "*.adt", SearchOption.TopDirectoryOnly).Count();
    }

    private static int CountPatchedAreaIds(string exportDir, string mapName)
    {
        var csvDir = Path.Combine(exportDir, "csv", "maps", mapName);
        if (!Directory.Exists(csvDir))
            return 0;

        var areaPatchPath = Path.Combine(csvDir, "area_patch_crosswalk.csv");
        if (!File.Exists(areaPatchPath))
            return 0;

        var lines = File.ReadAllLines(areaPatchPath);
        return Math.Max(0, lines.Length - 1); // Subtract header line
    }
}
