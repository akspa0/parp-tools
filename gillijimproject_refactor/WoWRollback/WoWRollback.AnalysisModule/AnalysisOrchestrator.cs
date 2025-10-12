using System.Text.Json;
using AlphaWdtAnalyzer.Core;
using AlphaWdtAnalyzer.Core.Terrain;

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
            var placementCsvs = new List<string>();
            string? manifestPath = null;

            // Try load analysis directory with index.json. If missing, we can still generate terrain/shadow overlays from CSVs.
            // Path includes map name: adtOutputDir/analysis/{mapName}/index.json
            var analysisIndexPath = Path.Combine(adtOutputDir, "analysis", mapName, "index.json");
            AnalysisIndex? analysisIndex = null;
            if (File.Exists(analysisIndexPath))
            {
                var indexJson = File.ReadAllText(analysisIndexPath);
                analysisIndex = JsonSerializer.Deserialize<AnalysisIndex>(indexJson);
            }

            // Stage 1: UniqueID Analysis (from index.json placements)
            if (opts.GenerateUniqueIdCsvs && analysisIndex != null)
            {
                var uniqueIdDir = Path.Combine(analysisOutputDir, "uniqueids");
                Directory.CreateDirectory(uniqueIdDir);

                var analyzer = new UniqueIdAnalyzer(opts.UniqueIdGapThreshold);
                var uniqueIdResult = analyzer.AnalyzeFromIndex(analysisIndex, mapName, uniqueIdDir);

                if (uniqueIdResult.Success)
                {
                    uniqueIdCsvs.Add(uniqueIdResult.CsvPath);
                    uniqueIdCsvs.Add(uniqueIdResult.LayersJsonPath);
                }
            }

            // Stage 2c: Ensure <mapName>_placements.csv exists; if missing and split ADTs present, generate from _obj*.adt
            string? copiedPlacementsCsv = null;
            var objectsDirOut = Path.Combine(analysisOutputDir, "objects");
            Directory.CreateDirectory(objectsDirOut);
            var targetPlacementsCsv = Path.Combine(objectsDirOut, $"{mapName}_placements.csv");

            string[] candidates = new[]
            {
                Path.Combine(objectsDirOut, $"{mapName}_placements.csv"),
                Path.Combine(adtOutputDir, "analysis", "csv", $"{mapName}_placements.csv"),
                Path.Combine(adtOutputDir, "analysis", "csv", "placements.csv")
            };

            string? foundSource = null;
            foreach (var c in candidates)
            {
                if (File.Exists(c)) { foundSource = c; break; }
            }

            if (foundSource == null)
            {
                var adtMapDir = Path.Combine(adtOutputDir, "World", "Maps", mapName);
                if (Directory.Exists(adtMapDir))
                {
                    try
                    {
                        var splitExtractor = new SplitAdtPlacementsExtractor();
                        splitExtractor.GeneratePlacementsCsv(adtOutputDir, mapName, version);
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"[analysis] placements generation failed: {ex.Message}");
                    }
                }
                foreach (var c in candidates)
                {
                    if (File.Exists(c)) { foundSource = c; break; }
                }
            }

            if (foundSource != null)
            {
                if (string.Equals(Path.GetFullPath(foundSource), Path.GetFullPath(targetPlacementsCsv), StringComparison.OrdinalIgnoreCase))
                {
                    copiedPlacementsCsv = foundSource;
                }
                else
                {
                    copiedPlacementsCsv = targetPlacementsCsv;
                    File.Copy(foundSource, copiedPlacementsCsv, overwrite: true);
                }
                placementCsvs.Add(copiedPlacementsCsv);
            }

            // Stage 2b: Placements CSV (from analysis index)
            if (analysisIndex != null)
            {
                var objectsDir = Path.Combine(analysisOutputDir, "objects");
                Directory.CreateDirectory(objectsDir);
                var placements = new PlacementsCsvGenerator().Generate(analysisIndex, mapName, objectsDir);
                placementCsvs.Add(placements);

                if (File.Exists(placements) && string.IsNullOrEmpty(copiedPlacementsCsv))
                {
                    copiedPlacementsCsv = placements;
                }

                var masterWriter = new MapMasterIndexWriter();
                var masterDir = Path.Combine(analysisOutputDir, "master");
                Directory.CreateDirectory(masterDir);
                var masterResult = masterWriter.Write(analysisIndex, mapName, version, masterDir);
                placementCsvs.Add(masterResult.MasterIndexPath);
                placementCsvs.Add(masterResult.TileRangePath);
            }

            // Stage 2: Ensure Terrain/Shadow CSVs exist; if missing, extract directly from LK ADTs (analyze-only support)
            if (opts.GenerateTerrainCsvs)
            {
                var adtMapDir = Path.Combine(adtOutputDir, "World", "Maps", mapName);
                var csvMapDir = Path.Combine(adtOutputDir, "csv", "maps", mapName);
                Directory.CreateDirectory(csvMapDir);

                var sourceTerrainCsv = Path.Combine(csvMapDir, "terrain.csv");
                var sourceShadowCsv = Path.Combine(csvMapDir, "shadow.csv");

                // Generate terrain.csv if missing
                if (!File.Exists(sourceTerrainCsv) && Directory.Exists(adtMapDir))
                {
                    var terrainEntries = LkAdtTerrainExtractor.ExtractFromLkAdts(adtMapDir, mapName);
                    McnkTerrainCsvWriter.WriteCsv(terrainEntries, sourceTerrainCsv);
                }
                // Generate shadow.csv if missing
                if (!File.Exists(sourceShadowCsv) && Directory.Exists(adtMapDir))
                {
                    var shadowEntries = LkAdtShadowExtractor.ExtractFromLkAdts(adtMapDir, mapName);
                    McnkShadowCsvWriter.WriteCsv(shadowEntries, sourceShadowCsv);
                }

                // Copy to analysis output folder for convenience
                var terrainDir = Path.Combine(analysisOutputDir, "terrain");
                Directory.CreateDirectory(terrainDir);
                if (File.Exists(sourceTerrainCsv))
                {
                    var targetPath = Path.Combine(terrainDir, $"{mapName}_terrain.csv");
                    File.Copy(sourceTerrainCsv, targetPath, overwrite: true);
                    terrainCsvs.Add(targetPath);
                }
                if (File.Exists(sourceShadowCsv))
                {
                    var targetPath = Path.Combine(terrainDir, $"{mapName}_shadow.csv");
                    File.Copy(sourceShadowCsv, targetPath, overwrite: true);
                    terrainCsvs.Add(targetPath);
                }
            }

            // Stage 3: Overlay Generation
            if (opts.GenerateOverlays)
            {
                var overlayGenerator = new OverlayGenerator();

                // 3a. Objects overlays (require analysis index)
                if (analysisIndex != null)
                {
                    var objResult = overlayGenerator.GenerateFromIndex(
                        analysisIndex,
                        analysisOutputDir,
                        viewerOutputDir,
                        mapName,
                        version);
                    if (objResult.Success)
                    {
                        overlayCount += objResult.ObjectOverlays;
                    }
                }

                // 3a-alt. Also generate objects overlays from placements.csv when available
                if (!string.IsNullOrEmpty(copiedPlacementsCsv) && File.Exists(copiedPlacementsCsv))
                {
                    var objFromCsv = overlayGenerator.GenerateObjectsFromPlacementsCsv(
                        copiedPlacementsCsv,
                        analysisOutputDir,
                        viewerOutputDir,
                        mapName,
                        version);
                    if (objFromCsv.Success)
                    {
                        overlayCount += objFromCsv.ObjectOverlays;
                    }
                }

                // 3b. Terrain overlays from terrain.csv
                var terrResult = overlayGenerator.GenerateTerrainOverlaysFromCsv(
                    adtOutputDir,
                    viewerOutputDir,
                    mapName,
                    version);
                if (terrResult.Success)
                {
                    overlayCount += terrResult.TerrainOverlays;
                }

                // 3c. Shadow overlays from shadow.csv
                var shResult = overlayGenerator.GenerateShadowOverlaysFromCsv(
                    adtOutputDir,
                    viewerOutputDir,
                    mapName,
                    version);
                if (shResult.Success)
                {
                    overlayCount += shResult.ShadowOverlays;
                }
            }

            // Stage 3b: UniqueID analysis from placements.csv if index missing
            if (analysisIndex == null && !string.IsNullOrEmpty(copiedPlacementsCsv) && File.Exists(copiedPlacementsCsv))
            {
                var uniqueIdDir = Path.Combine(analysisOutputDir, "uniqueids");
                Directory.CreateDirectory(uniqueIdDir);
                var analyzerFromCsv = new UniqueIdAnalyzer(opts.UniqueIdGapThreshold);
                var uidRes = analyzerFromCsv.AnalyzeFromPlacementsCsv(copiedPlacementsCsv, mapName, uniqueIdDir);
                if (uidRes.Success)
                {
                    uniqueIdCsvs.Add(uidRes.CsvPath);
                    uniqueIdCsvs.Add(uidRes.LayersJsonPath);
                }
            }

            // Stage 4: Build minimal viewer index.json from generated overlays
            try
            {
                var combinedDir = Path.Combine(viewerOutputDir, "overlays", version, mapName, "combined");
                var tiles = new List<object>();
                if (Directory.Exists(combinedDir))
                {
                    foreach (var file in Directory.GetFiles(combinedDir, "tile_r*_c*.json"))
                    {
                        var name = Path.GetFileNameWithoutExtension(file); // tile_r{row}_c{col}
                        var parts = name.Split('_');
                        if (parts.Length >= 3 && parts[0] == "tile" && parts[1].StartsWith("r") && parts[2].StartsWith("c"))
                        {
                            if (int.TryParse(parts[1].Substring(1), out var row) && int.TryParse(parts[2].Substring(1), out var col))
                            {
                                tiles.Add(new { row, col, versions = new[] { version } });
                            }
                        }
                    }
                }

                var index = new
                {
                    comparisonKey = version,
                    versions = new[] { version },
                    defaultVersion = version,
                    maps = new[] { new { map = mapName, tiles = tiles } }
                };

                var indexPath = Path.Combine(viewerOutputDir, "index.json");
                File.WriteAllText(indexPath, JsonSerializer.Serialize(index, new JsonSerializerOptions { WriteIndented = true }));

                manifestPath = indexPath;
            }
            catch
            {
                // Non-fatal: viewer may still serve static assets
            }

            return new AnalysisResult(
                UniqueIdCsvs: uniqueIdCsvs,
                TerrainCsvs: terrainCsvs,
                PlacementCsvs: placementCsvs,
                OverlayCount: overlayCount,
                ManifestPath: manifestPath,
                Success: true);
        }
        catch (Exception ex)
        {
            return new AnalysisResult(
                UniqueIdCsvs: Array.Empty<string>(),
                TerrainCsvs: Array.Empty<string>(),
                PlacementCsvs: Array.Empty<string>(),
                OverlayCount: 0,
                ManifestPath: null,
                Success: false,
                ErrorMessage: $"Analysis failed: {ex.Message}");
        }
    }
}
