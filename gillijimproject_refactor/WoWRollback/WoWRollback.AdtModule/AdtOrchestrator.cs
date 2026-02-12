using System;
using System.IO;
using System.Linq;
using AlphaWdtAnalyzer.Core;
using AlphaWdtAnalyzer.Core.Export;
using WoWRollback.AdtModule.Convert;

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
        if (opts is null)
            throw new ArgumentNullException(nameof(opts));
        if (string.IsNullOrWhiteSpace(wdtPath))
            throw new ArgumentException("WDT path is required", nameof(wdtPath));
        if (string.IsNullOrWhiteSpace(exportDir))
            throw new ArgumentException("Export directory is required", nameof(exportDir));
        if (string.IsNullOrWhiteSpace(mapName))
            throw new ArgumentException("Map name is required", nameof(mapName));

        try
        {
            if (!File.Exists(wdtPath))
            {
                return new AdtConversionResult(
                    AdtOutputDirectory: exportDir,
                    TerrainCsvPath: null,
                    ShadowCsvPath: null,
                    TilesProcessed: 0,
                    AreaIdsPatched: 0,
                    Success: false,
                    ErrorMessage: $"WDT file not found: {wdtPath}");
            }

            // Create output directory
            Directory.CreateDirectory(exportDir);

            // Execute native multithreaded conversion pipeline (faster per-tile with safe fixup logging)
            ConvertPipelineMT.Run(new ConvertPipelineMT.Options
            {
                WdtPath = wdtPath,
                ExportDir = exportDir,
                CommunityListfilePath = opts.CommunityListfilePath,
                LkListfilePath = opts.LkListfilePath,
                DbdDir = opts.DbdDir,
                DbctoolOutRoot = opts.CrosswalkDir,
                DbctoolSrcAlias = srcAlias,
                DbctoolSrcDir = null,
                DbctoolLkDir = opts.LkDbcDir,
                DbctoolPatchDir = opts.CrosswalkDir,
                DbctoolPatchFile = null,
                ConvertToMh2o = opts.ConvertToMh2o,
                AssetFuzzy = opts.AssetFuzzy,
                UseFallbacks = opts.UseFallbacks,
                EnableFixups = opts.EnableFixups,
                TrackAssets = opts.TrackAssets,
                Verbose = opts.Verbose,
                FallbackTileset = opts.FallbackTileset,
                FallbackNonTilesetBlp = opts.FallbackNonTilesetBlp,
                FallbackWmo = opts.FallbackWmo,
                FallbackM2 = opts.FallbackM2,
                MaxDegreeOfParallelism = 0,
                VersionAlias = opts.VersionAlias ?? srcAlias,
                AreaOverrides = opts.AreaOverrides
            });

            // Count results
            var tilesProcessed = CountGeneratedTiles(exportDir, mapName);
            var areaIdsPatched = CountPatchedAreaIds(exportDir, mapName);

            // Run analysis pipeline to generate terrain/shadow CSVs
            var lkAdtDir = Path.Combine(exportDir, "World", "Maps", mapName);
            var analysisDir = Path.Combine(exportDir, "analysis");

            // Prefer LK listfile; fall back to community listfile if needed
            string? listfileForAnalysis = null;
            if (!string.IsNullOrWhiteSpace(opts.LkListfilePath) && File.Exists(opts.LkListfilePath))
            {
                listfileForAnalysis = opts.LkListfilePath;
            }
            else if (!string.IsNullOrWhiteSpace(opts.CommunityListfilePath) && File.Exists(opts.CommunityListfilePath))
            {
                listfileForAnalysis = opts.CommunityListfilePath;
            }

            if (!string.IsNullOrWhiteSpace(listfileForAnalysis))
            {
                var analysisOptions = new AnalysisPipeline.Options
                {
                    WdtPath = wdtPath,
                    ListfilePath = listfileForAnalysis!,
                    OutDir = analysisDir,
                    ExtractMcnkTerrain = true,
                    ExtractMcnkShadows = true,
                    LkAdtDirectory = lkAdtDir,
                    ClusterThreshold = 10,
                    ClusterGap = 1000
                };

                try
                {
                    AnalysisPipeline.Run(analysisOptions);
                }
                catch (Exception ex)
                {
                    if (opts.Verbose)
                    {
                        Console.Error.WriteLine($"[WARN] Analysis pipeline failed: {ex.Message}");
                    }
                }
            }

            // Compute CSV paths (AnalysisPipeline generates them at analysis/csv/{mapName}/)
            var csvDir = Path.Combine(analysisDir, "csv", mapName);
            var terrainCsvPath = Path.Combine(csvDir, $"{mapName}_mcnk_terrain.csv");
            var shadowCsvPath = Path.Combine(csvDir, $"{mapName}_mcnk_shadows.csv");

            // Copy CSVs to expected location for backward compat
            var targetCsvDir = Path.Combine(exportDir, "csv", "maps", mapName);
            Directory.CreateDirectory(targetCsvDir);

            if (File.Exists(terrainCsvPath))
            {
                File.Copy(terrainCsvPath, Path.Combine(targetCsvDir, "terrain.csv"), overwrite: true);
            }
            if (File.Exists(shadowCsvPath))
            {
                File.Copy(shadowCsvPath, Path.Combine(targetCsvDir, "shadow.csv"), overwrite: true);
            }

            return new AdtConversionResult(
                AdtOutputDirectory: exportDir,
                TerrainCsvPath: File.Exists(terrainCsvPath) ? Path.Combine(targetCsvDir, "terrain.csv") : null,
                ShadowCsvPath: File.Exists(shadowCsvPath) ? Path.Combine(targetCsvDir, "shadow.csv") : null,
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
