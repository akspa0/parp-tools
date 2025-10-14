using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using WoWRollback.Core.Models;
using WoWRollback.Core.Services.Viewer;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Webp;

namespace WoWRollback.AnalysisModule;

/// <summary>
/// Adapts single-map or multi-map analysis results to work with the existing ViewerReportWriter.
/// Creates a synthetic "version" for standalone map analysis.
/// </summary>
public sealed class AnalysisViewerAdapter
{
    private static StreamWriter? _logWriter;
    
    private static void Log(string message)
    {
        var msg = $"[{DateTime.Now:HH:mm:ss.fff}] {message}";
        Console.WriteLine(msg);
        _logWriter?.WriteLine(msg);
        _logWriter?.Flush();
    }
    
    /// <summary>
    /// Generates a unified viewer for multiple maps.
    /// </summary>
    public string GenerateUnifiedViewer(
        List<(string MapName, string PlacementsCsv, string? MinimapDir)> maps,
        string baseOutputDir,
        string versionLabel)
    {
        try
        {
            // Setup logging
            var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            var logPath = Path.Combine(baseOutputDir, $"unified_viewer_{timestamp}.log");
            _logWriter = new StreamWriter(logPath, append: false);
            Log($"=== Unified Viewer Generation Started ===");
            Log($"Processing {maps.Count} maps for version: {versionLabel}");

            var allPlacements = new List<AssetTimelineDetailedEntry>();

            // Load placements from all maps
            foreach (var (mapName, placementsCsv, minimapDir) in maps)
            {
                Log($"Loading map: {mapName}");
                
                // Setup minimaps if available
                if (!string.IsNullOrEmpty(minimapDir) && Directory.Exists(minimapDir))
                {
                    SetupMinimaps(minimapDir, mapName, baseOutputDir, versionLabel);
                }
                
                // Load placements
                var mapPlacements = LoadPlacementsFromCsv(placementsCsv, mapName, versionLabel);
                Log($"  Loaded {mapPlacements.Count} placements");
                
                // Add minimap tile markers
                if (!string.IsNullOrEmpty(minimapDir) && Directory.Exists(minimapDir))
                {
                    var minimapPngs = Directory.GetFiles(minimapDir, "*.png", SearchOption.TopDirectoryOnly);
                    foreach (var pngFile in minimapPngs)
                    {
                        var fileName = Path.GetFileNameWithoutExtension(pngFile);
                        var parts = fileName.Split('_');
                        if (parts.Length >= 3 &&
                            int.TryParse(parts[^2], out var tileX) &&
                            int.TryParse(parts[^1], out var tileY))
                        {
                            if (!mapPlacements.Any(p => p.TileRow == tileY && p.TileCol == tileX))
                            {
                                mapPlacements.Add(new AssetTimelineDetailedEntry(
                                    Version: versionLabel,
                                    Map: mapName,
                                    TileRow: tileY,
                                    TileCol: tileX,
                                    Kind: PlacementKind.M2,
                                    UniqueId: 0,
                                    AssetPath: "_dummy_tile_marker",
                                    Folder: "", Category: "", Subcategory: "", DesignKit: "",
                                    SourceRule: "", KitRoot: "", SubkitPath: "", SubkitTop: "",
                                    SubkitDepth: 0, FileName: "", FileStem: "", Extension: "",
                                    WorldX: 0, WorldY: 0, WorldZ: 0,
                                    RotationX: 0, RotationY: 0, RotationZ: 0,
                                    Scale: 0, Flags: 0, DoodadSet: 0, NameSet: 0
                                ));
                            }
                        }
                    }
                }
                
                allPlacements.AddRange(mapPlacements);
            }

            Log($"Total placements across all maps: {allPlacements.Count}");

            // Generate unified viewer
            var result = new VersionComparisonResult(
                RootDirectory: baseOutputDir,
                ComparisonKey: $"unified_analysis",
                Versions: new[] { versionLabel },
                RangeEntries: Array.Empty<VersionRangeEntry>(),
                MapSummaries: Array.Empty<MapVersionSummary>(),
                Overlaps: Array.Empty<RangeOverlapEntry>(),
                AssetFirstSeen: Array.Empty<AssetFirstSeenEntry>(),
                AssetFolderSummaries: Array.Empty<AssetFolderSummary>(),
                AssetFolderTimeline: Array.Empty<AssetFolderTimelineEntry>(),
                AssetTimeline: Array.Empty<AssetTimelineEntry>(),
                DesignKitAssets: Array.Empty<DesignKitAssetEntry>(),
                DesignKitRanges: Array.Empty<DesignKitRangeEntry>(),
                DesignKitSummaries: Array.Empty<DesignKitSummaryEntry>(),
                DesignKitTimeline: Array.Empty<DesignKitTimelineEntry>(),
                DesignKitAssetDetails: Array.Empty<DesignKitAssetDetailEntry>(),
                UniqueIdAssets: Array.Empty<UniqueIdAssetEntry>(),
                AssetTimelineDetailed: allPlacements,
                Warnings: Array.Empty<string>()
            );

            var viewerWriter = new ViewerReportWriter();
            var viewerOptions = new ViewerOptions(
                DefaultVersion: versionLabel,
                DiffPair: null,
                MinimapWidth: 256,
                MinimapHeight: 256,
                DiffDistanceThreshold: 10.0,
                MoveEpsilonRatio: 0.1
            );

            var viewerRoot = viewerWriter.Generate(
                baseOutputDir,
                result,
                viewerOptions,
                diffPair: null
            );

            // Generate UniqueID range CSVs for each map (for Sedimentary Layers feature)
            Log($"Generating UniqueID range CSVs for Sedimentary Layers...");
            foreach (var (mapName, placementsCsv, minimapDir) in maps)
            {
                if (File.Exists(placementsCsv))
                {
                    CopyUniqueIdCsvToViewer(baseOutputDir, viewerRoot, mapName, versionLabel, placementsCsv);
                }
            }

            Log($"=== Unified Viewer Generation Complete ===");
            _logWriter?.Close();
            _logWriter = null;

            return viewerRoot;
        }
        catch (Exception ex)
        {
            Log($"ERROR: {ex.Message}");
            Log($"Stack trace: {ex.StackTrace}");
            _logWriter?.Close();
            _logWriter = null;
            throw;
        }
    }
    
    public string GenerateViewer(
        string placementsCsvPath,
        string mapName,
        string outputDir,
        string? minimapDir = null,
        string? versionLabel = null)
    {
        try
        {
            // Setup logging with timestamp to avoid file locking issues
            var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            var logPath = Path.Combine(outputDir, $"analysis_debug_{timestamp}.log");
            _logWriter = new StreamWriter(logPath, append: false);
            Log($"=== Analysis started for map '{mapName}' ===");
            
            // Use provided version label or default to "analysis"
            string syntheticVersion = versionLabel ?? "analysis";
            
            // Setup minimap structure if provided
            if (!string.IsNullOrEmpty(minimapDir) && Directory.Exists(minimapDir))
            {
                SetupMinimaps(minimapDir, mapName, outputDir, syntheticVersion);
            }
            
            // Load placements from CSV
            var placements = LoadPlacementsFromCsv(placementsCsvPath, mapName, syntheticVersion);
            
            Log($"Loaded {placements.Count} placements from CSV");
            
            if (placements.Count == 0)
            {
                Log("No placements found in CSV");
                return string.Empty;
            }

            // Debug: Show tile distribution IMMEDIATELY after CSV load
            var tileGroups = placements.GroupBy(p => (p.TileRow, p.TileCol)).OrderBy(g => g.Key.TileRow).ThenBy(g => g.Key.TileCol).ToList();
            Log($"AFTER CSV LOAD: Placements span {tileGroups.Count} tiles");
            
            // Count entries with (0,0) vs others
            var tile00Count = placements.Count(p => p.TileRow == 0 && p.TileCol == 0);
            var otherTilesCount = placements.Count - tile00Count;
            Log($"AFTER CSV LOAD: Tile (0,0)={tile00Count}, Other tiles={otherTilesCount}");
            if (tileGroups.Count > 0 && tileGroups.Count <= 10)
            {
                foreach (var tg in tileGroups)
                {
                    var m2Count = tg.Count(p => p.Kind == PlacementKind.M2);
                    var wmoCount = tg.Count(p => p.Kind == PlacementKind.WMO);
                    Log($"  Tile (row={tg.Key.TileRow}, col={tg.Key.TileCol}): {tg.Count()} objects (M2={m2Count}, WMO={wmoCount})");
                }
            }
            else if (tileGroups.Count > 0)
            {
                var first = tileGroups[0];
                var last = tileGroups[^1];
                Log($"  First tile: (row={first.Key.TileRow}, col={first.Key.TileCol}) with {first.Count()} objects");
                Log($"  Last tile: (row={last.Key.TileRow}, col={last.Key.TileCol}) with {last.Count()} objects");
            }
            
            // Debug: Show sample entries
            var sampleEntry = placements.FirstOrDefault();
            if (sampleEntry != null)
            {
                Log($"Sample entry: Map={sampleEntry.Map}, TileRow={sampleEntry.TileRow}, TileCol={sampleEntry.TileCol}, Version={sampleEntry.Version}, Kind={sampleEntry.Kind}");
                Log($"Sample coords: World=({sampleEntry.WorldX:F1}, {sampleEntry.WorldY:F1}, {sampleEntry.WorldZ:F1}), Path={sampleEntry.AssetPath}");
            }
            
            // CRITICAL: Add entries for ALL minimap tiles, even if they have no placements
            // This ensures the viewer shows all map tiles
            if (!string.IsNullOrEmpty(minimapDir) && Directory.Exists(minimapDir))
            {
                var minimapPngs = Directory.GetFiles(minimapDir, "*.png", SearchOption.TopDirectoryOnly);
                Console.WriteLine($"[AnalysisViewerAdapter] Found {minimapPngs.Length} minimap tiles");
                
                foreach (var pngFile in minimapPngs)
                {
                    // Parse tile coords from filename: development_X_Y.png
                    var fileName = Path.GetFileNameWithoutExtension(pngFile);
                    var parts = fileName.Split('_');
                    if (parts.Length >= 3 &&
                        int.TryParse(parts[^2], out var tileX) &&
                        int.TryParse(parts[^1], out var tileY))
                    {
                        // Add dummy placement entry to ensure tile appears in index
                        // Use TileRow=Y, TileCol=X per our mapping
                        if (!placements.Any(p => p.TileRow == tileY && p.TileCol == tileX))
                        {
                            placements.Add(new AssetTimelineDetailedEntry(
                                Version: syntheticVersion,
                                Map: mapName,
                                TileRow: tileY,
                                TileCol: tileX,
                                Kind: PlacementKind.M2,  // Dummy
                                UniqueId: 0,  // Dummy entry - will be filtered out in overlays
                                AssetPath: "_dummy_tile_marker",
                                Folder: "", Category: "", Subcategory: "", DesignKit: "",
                                SourceRule: "", KitRoot: "", SubkitPath: "", SubkitTop: "",
                                SubkitDepth: 0, FileName: "", FileStem: "", Extension: "",
                                WorldX: 0, WorldY: 0, WorldZ: 0,
                                RotationX: 0, RotationY: 0, RotationZ: 0,
                                Scale: 0, Flags: 0, DoodadSet: 0, NameSet: 0
                            ));
                        }
                    }
                }
                
                Log($"Total placements after minimap tiles: {placements.Count}");
            }

            // FINAL CHECK: Verify data right before passing to ViewerReportWriter
            var finalTile00Count = placements.Count(p => p.TileRow == 0 && p.TileCol == 0);
            var finalOtherTilesCount = placements.Count - finalTile00Count;
            Log($"BEFORE ViewerReportWriter: Tile (0,0)={finalTile00Count}, Other tiles={finalOtherTilesCount}");
            
            // Convert to VersionComparisonResult format
            var result = new VersionComparisonResult(
                RootDirectory: outputDir,
                ComparisonKey: $"{mapName}_analysis",
                Versions: new[] { syntheticVersion },
                RangeEntries: Array.Empty<VersionRangeEntry>(),
                MapSummaries: Array.Empty<MapVersionSummary>(),
                Overlaps: Array.Empty<RangeOverlapEntry>(),
                AssetFirstSeen: Array.Empty<AssetFirstSeenEntry>(),
                AssetFolderSummaries: Array.Empty<AssetFolderSummary>(),
                AssetFolderTimeline: Array.Empty<AssetFolderTimelineEntry>(),
                AssetTimeline: Array.Empty<AssetTimelineEntry>(),
                DesignKitAssets: Array.Empty<DesignKitAssetEntry>(),
                DesignKitRanges: Array.Empty<DesignKitRangeEntry>(),
                DesignKitSummaries: Array.Empty<DesignKitSummaryEntry>(),
                DesignKitTimeline: Array.Empty<DesignKitTimelineEntry>(),
                DesignKitAssetDetails: Array.Empty<DesignKitAssetDetailEntry>(),
                UniqueIdAssets: Array.Empty<UniqueIdAssetEntry>(),
                AssetTimelineDetailed: placements,
                Warnings: Array.Empty<string>()
            );

            // Generate viewer using existing infrastructure
            var viewerWriter = new ViewerReportWriter();
            var viewerOptions = new ViewerOptions(
                DefaultVersion: syntheticVersion,
                DiffPair: null,
                MinimapWidth: 256,
                MinimapHeight: 256,
                DiffDistanceThreshold: 10.0,
                MoveEpsilonRatio: 0.1
            );

            var viewerRoot = viewerWriter.Generate(
                outputDir,
                result,
                viewerOptions,
                diffPair: null
            );

            Console.WriteLine($"[AnalysisViewerAdapter] Viewer root generated: {viewerRoot}");
            
            // Generate terrain overlays from MCNK terrain CSV
            GenerateTerrainOverlays(outputDir, viewerRoot, mapName, syntheticVersion);
            
            // Generate cluster overlays from spatial clusters JSON
            GenerateClusterOverlays(outputDir, viewerRoot, mapName, syntheticVersion);
            
            // Copy UniqueID analysis CSV to expected location for "Load UniqueID Ranges" feature
            CopyUniqueIdCsvToViewer(outputDir, viewerRoot, mapName, syntheticVersion, placementsCsvPath);

            // Copy pre-existing PNG minimaps directly to viewer output
            // This is optional - viewer can work without minimaps
            if (!string.IsNullOrEmpty(minimapDir) && !string.IsNullOrEmpty(viewerRoot) && Directory.Exists(minimapDir))
            {
                try
                {
                    CopyMinimapsToViewer(minimapDir, viewerRoot, mapName, syntheticVersion);
                }
                catch (Exception minimapEx)
                {
                    Console.WriteLine($"[AnalysisViewerAdapter] Minimap conversion failed (viewer will work without them): {minimapEx.Message}");
                }
            }

            Log("=== Analysis completed successfully ===");
            _logWriter?.Close();
            _logWriter = null;
            
            return viewerRoot;
        }
        catch (Exception ex)
        {
            Log($"CRITICAL ERROR: {ex.Message}");
            Log($"Stack trace: {ex.StackTrace}");
            _logWriter?.Close();
            _logWriter = null;
            return string.Empty;
        }
    }

    private List<AssetTimelineDetailedEntry> LoadPlacementsFromCsv(string csvPath, string mapName, string versionLabel)
    {
        var entries = new List<AssetTimelineDetailedEntry>();

        if (!File.Exists(csvPath))
        {
            Console.WriteLine($"[AnalysisViewerAdapter] CSV file not found: {csvPath}");
            return entries;
        }

        using var reader = new StreamReader(csvPath);
        string? header = reader.ReadLine(); // Skip header
        Console.WriteLine($"[AnalysisViewerAdapter] CSV header: {header}");

        int lineNumber = 1;
        int parsedCount = 0;
        int errorCount = 0;

        while (reader.ReadLine() is { } line)
        {
            lineNumber++;
            
            if (string.IsNullOrWhiteSpace(line))
                continue;

            var fields = SplitCsvLine(line);
            if (fields.Length < 9)
            {
                Console.WriteLine($"[AnalysisViewerAdapter] Line {lineNumber}: Not enough fields ({fields.Length})");
                continue;
            }

            try
            {
                // CSV format: map,tile_x,tile_y,type,asset_path,unique_id,world_x,world_y,world_z,...
                var tileX = int.Parse(fields[1]);
                var tileY = int.Parse(fields[2]);
                var type = fields[3];
                var assetPath = fields[4];
                var uniqueId = int.Parse(fields[5]);
                var worldX = float.Parse(fields[6]);
                var worldY = float.Parse(fields[7]);
                var worldZ = float.Parse(fields[8]);

                var kind = type.Equals("M2", StringComparison.OrdinalIgnoreCase) || type.Equals("MDX", StringComparison.OrdinalIgnoreCase)
                    ? PlacementKind.M2
                    : PlacementKind.WMO;

                var entry = new AssetTimelineDetailedEntry(
                    Version: versionLabel,
                    Map: mapName,
                    TileRow: tileY,  // tile_y → TileRow (Y is vertical/row)
                    TileCol: tileX,  // tile_x → TileCol (X is horizontal/col)
                    Kind: kind,
                    UniqueId: (uint)uniqueId,
                    AssetPath: assetPath,
                    Folder: "",
                    Category: "",
                    Subcategory: "",
                    DesignKit: "",
                    SourceRule: "",
                    KitRoot: "",
                    SubkitPath: "",
                    SubkitTop: "",
                    SubkitDepth: 0,
                    FileName: Path.GetFileName(assetPath),
                    FileStem: Path.GetFileNameWithoutExtension(assetPath),
                    Extension: Path.GetExtension(assetPath),
                    WorldX: worldX,
                    WorldY: worldY,
                    WorldZ: worldZ,
                    RotationX: 0f,  // TODO: Parse if available
                    RotationY: 0f,
                    RotationZ: 0f,
                    Scale: 1f,
                    Flags: 0,
                    DoodadSet: 0,
                    NameSet: 0
                );
                
                entries.Add(entry);
                
                // Debug: Log first few entries AND some from different tiles
                if (parsedCount < 5 || (parsedCount % 1000 == 0 && parsedCount < 10000))
                {
                    Log($"CSV Entry {parsedCount}: Map={entry.Map}, TileRow={entry.TileRow}, TileCol={entry.TileCol}, WorldX={entry.WorldX:F1}, WorldZ={entry.WorldZ:F1}, UID={entry.UniqueId}");
                }
                
                parsedCount++;
            }
            catch (Exception ex)
            {
                errorCount++;
                if (errorCount <= 5) // Only log first 5 errors
                {
                    Console.WriteLine($"[AnalysisViewerAdapter] Line {lineNumber} parse error: {ex.Message}");
                }
                continue;
            }
        }

        Console.WriteLine($"[AnalysisViewerAdapter] Parsed {parsedCount} entries, {errorCount} errors");
        return entries;
    }

    private string[] SplitCsvLine(string line)
    {
        var result = new List<string>();
        bool inQuotes = false;
        var current = new System.Text.StringBuilder();

        for (int i = 0; i < line.Length; i++)
        {
            char c = line[i];
            if (inQuotes)
            {
                if (c == '"')
                {
                    if (i + 1 < line.Length && line[i + 1] == '"')
                    {
                        current.Append('"');
                        i++;
                    }
                    else
                    {
                        inQuotes = false;
                    }
                }
                else
                {
                    current.Append(c);
                }
            }
            else
            {
                if (c == ',')
                {
                    result.Add(current.ToString());
                    current.Clear();
                }
                else if (c == '"')
                {
                    inQuotes = true;
                }
                else
                {
                    current.Append(c);
                }
            }
        }

        result.Add(current.ToString());
        return result.ToArray();
    }

    private void SetupMinimaps(string minimapSourceDir, string mapName, string outputDir, string version)
    {
        try
        {
            // Create minimap structure expected by MinimapLocator:
            // {outputDir}/{version}/World/Textures/Minimap/{mapName}/
            var minimapDestDir = Path.Combine(outputDir, version, "World", "Textures", "Minimap", mapName);
            Directory.CreateDirectory(minimapDestDir);

            // Convert PNG files to WebP (viewer expects WebP)
            var pngFiles = Directory.GetFiles(minimapSourceDir, "*.png", SearchOption.TopDirectoryOnly);
            Log($"Converting {pngFiles.Length} minimap PNG files to WebP...");

            int converted = 0;
            foreach (var sourceFile in pngFiles)
            {
                try
                {
                    // Load PNG
                    using var image = Image.Load(sourceFile);
                    
                    // Save as WebP with same filename (change extension)
                    var fileName = Path.GetFileNameWithoutExtension(sourceFile);
                    var destFile = Path.Combine(minimapDestDir, $"{fileName}.webp");
                    
                    using var outStream = File.Create(destFile);
                    image.Save(outStream, new WebpEncoder { Quality = 90 });
                    converted++;
                }
                catch (Exception ex)
                {
                    Log($"Failed to convert {Path.GetFileName(sourceFile)}: {ex.Message}");
                }
            }
            
            Log($"Converted {converted}/{pngFiles.Length} minimap tiles to WebP");
        }
        catch (Exception ex)
        {
            Log($"Failed to setup minimaps: {ex.Message}");
        }
    }

    private void GenerateTerrainOverlays(string outputDir, string viewerRoot, string mapName, string version)
    {
        try
        {
            // Look for terrain CSV: {outputDir}/{mapName}_terrain.csv
            var terrainCsvPath = Path.Combine(outputDir, $"{mapName}_terrain.csv");
            
            if (!File.Exists(terrainCsvPath))
            {
                Log($"Terrain CSV not found: {terrainCsvPath} - terrain overlays will not be available");
                return;
            }

            // Generate terrain overlay JSONs using TerrainOverlayBuilder
            // Target: {viewerRoot}/overlays/{version}/{mapName}/terrain_complete/tile_{col}_{row}.json
            var builder = new TerrainOverlayBuilder();
            builder.BuildTerrainOverlays(terrainCsvPath, mapName, version, viewerRoot);
            
            Log("Terrain overlays generated successfully");
        }
        catch (Exception ex)
        {
            Log($"Failed to generate terrain overlays: {ex.Message}");
        }
    }

    private void GenerateClusterOverlays(string outputDir, string viewerRoot, string mapName, string version)
    {
        try
        {
            // Look for cluster JSON: {outputDir}/{mapName}_spatial_clusters.json
            var clusterJsonPath = Path.Combine(outputDir, $"{mapName}_spatial_clusters.json");
            
            if (!File.Exists(clusterJsonPath))
            {
                Console.WriteLine($"[AnalysisViewerAdapter] Cluster JSON not found: {clusterJsonPath}");
                return;
            }

            // Generate cluster overlay JSONs
            // Target: {viewerRoot}/overlays/{version}/{mapName}/clusters/tile_{col}_{row}.json
            // ClusterOverlayBuilder adds {mapName}/clusters itself, so pass version root only
            var overlaysVersionRoot = Path.Combine(viewerRoot, "overlays", version);

            var options = new ViewerOptions(
                DefaultVersion: version,
                DiffPair: null,
                MinimapWidth: 512,
                MinimapHeight: 512,
                DiffDistanceThreshold: 20.0,
                MoveEpsilonRatio: 0.1
            );

            ClusterOverlayBuilder.BuildClusterOverlays(clusterJsonPath, mapName, overlaysVersionRoot, options);
            Console.WriteLine($"[AnalysisViewerAdapter] Generated cluster overlays");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[AnalysisViewerAdapter] Failed to generate cluster overlays: {ex.Message}");
        }
    }

    private void CopyUniqueIdCsvToViewer(string outputDir, string viewerRoot, string mapName, string version, string placementsCsvPath)
    {
        try
        {
            // Viewer expects: cached_maps/analysis/{version}/{mapName}/csv/id_ranges_by_map.csv
            // Format: map,min,max,count (per-tile ranges)
            var csvDestDir = Path.Combine(viewerRoot, "cached_maps", "analysis", version, mapName, "csv");
            Directory.CreateDirectory(csvDestDir);
            
            // Generate per-tile UniqueID ranges from placements CSV
            if (File.Exists(placementsCsvPath))
            {
                var ranges = GeneratePerTileUniqueIdRanges(placementsCsvPath, mapName);
                var destPath = Path.Combine(csvDestDir, "id_ranges_by_map.csv");
                File.WriteAllLines(destPath, ranges);
                Console.WriteLine($"[AnalysisViewerAdapter] Generated UniqueID ranges CSV: {destPath}");
            }
            else
            {
                Console.WriteLine($"[AnalysisViewerAdapter] Placements CSV not found: {placementsCsvPath}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[AnalysisViewerAdapter] Failed to generate UniqueID ranges: {ex.Message}");
        }
    }

    private List<string> GeneratePerTileUniqueIdRanges(string placementsCsvPath, string mapName)
    {
        // Read all placements and group by tile
        // CSV header: map,tile_x,tile_y,type,asset_path,unique_id,world_x,world_y,world_z,rot_x,rot_y,rot_z,scale,doodad_set,name_set
        var lines = File.ReadAllLines(placementsCsvPath).Skip(1); // Skip header
        var byTile = new Dictionary<(int row, int col), List<uint>>();

        foreach (var line in lines)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            
            var parts = line.Split(',');
            if (parts.Length >= 6)  // Need at least: map,tile_x,tile_y,type,asset_path,unique_id
            {
                // Column indices: map=0, tile_x=1, tile_y=2, unique_id=5
                if (uint.TryParse(parts[5], out var uniqueId) &&
                    int.TryParse(parts[1], out var tileX) &&
                    int.TryParse(parts[2], out var tileY))
                {
                    var key = (tileY, tileX);  // Use (row, col) convention
                    if (!byTile.ContainsKey(key))
                        byTile[key] = new List<uint>();
                    byTile[key].Add(uniqueId);
                }
            }
        }

        // Generate ranges per tile (detect gaps)
        var allRanges = new List<(string tileName, uint min, uint max, int count)>();
        
        foreach (var ((row, col), ids) in byTile.OrderBy(kvp => kvp.Key))
        {
            var sorted = ids.Distinct().OrderBy(x => x).ToList();
            if (sorted.Count == 0) continue;

            // Detect gaps > 1000 to create separate ranges
            uint rangeStart = sorted[0];
            uint rangeEnd = sorted[0];
            int rangeCount = 1;

            for (int i = 1; i < sorted.Count; i++)
            {
                if (sorted[i] - rangeEnd > 1000) // Gap detected
                {
                    allRanges.Add(($"{mapName}_({row}_{col})", rangeStart, rangeEnd, rangeCount));
                    rangeStart = sorted[i];
                    rangeEnd = sorted[i];
                    rangeCount = 1;
                }
                else
                {
                    rangeEnd = sorted[i];
                    rangeCount++;
                }
            }
            allRanges.Add(($"{mapName}_({row}_{col})", rangeStart, rangeEnd, rangeCount));
        }

        // Sort all ranges by min UniqueID (smallest to largest)
        var sortedRanges = allRanges.OrderBy(r => r.min).ToList();
        
        // Build output CSV
        var output = new List<string> { "map,min,max,count" };
        foreach (var (tileName, min, max, count) in sortedRanges)
        {
            output.Add($"{tileName},{min},{max},{count}");
        }

        return output;
    }

    private void CopyMinimapsToViewer(string minimapSourceDir, string viewerRoot, string mapName, string version)
    {
        try
        {
            // Viewer structure: {viewerRoot}/minimap/{version}/{mapName}/
            var mapNameSafe = mapName.Replace(" ", "_").Replace("/", "_").Replace("\\", "_");
            var viewerMinimapDir = Path.Combine(viewerRoot, "minimap", version, mapNameSafe);
            Directory.CreateDirectory(viewerMinimapDir);

            // Check if WebP files already exist
            var existingWebpFiles = Directory.GetFiles(viewerMinimapDir, "*.webp", SearchOption.TopDirectoryOnly);
            var pngFiles = Directory.GetFiles(minimapSourceDir, "*.png", SearchOption.TopDirectoryOnly);

            if (existingWebpFiles.Length > 0 && existingWebpFiles.Length >= pngFiles.Length)
            {
                Console.WriteLine($"[AnalysisViewerAdapter] Found {existingWebpFiles.Length} existing WebP tiles, skipping conversion");
                return;
            }

            // Convert PNGs to WebP for massive memory savings
            Console.WriteLine($"[AnalysisViewerAdapter] Converting {pngFiles.Length} PNGs to WebP: {viewerMinimapDir}");

            var webpEncoder = new WebpEncoder
            {
                Quality = 85,  // Good balance of quality vs size
                FileFormat = WebpFileFormatType.Lossy,
                Method = WebpEncodingMethod.BestQuality
            };

            int converted = 0;
            int skipped = 0;
            long totalSavedBytes = 0;

            foreach (var sourceFile in pngFiles)
            {
                try
                {
                    var sourceFileInfo = new FileInfo(sourceFile);
                    var fileNameWithoutExt = Path.GetFileNameWithoutExtension(sourceFile);
                    var destFile = Path.Combine(viewerMinimapDir, fileNameWithoutExt + ".webp");

                    // Skip if WebP already exists and is newer than source
                    if (File.Exists(destFile))
                    {
                        var destInfo = new FileInfo(destFile);
                        if (destInfo.LastWriteTimeUtc >= sourceFileInfo.LastWriteTimeUtc)
                        {
                            skipped++;
                            continue;
                        }
                    }

                    using (var image = Image.Load(sourceFile))
                    {
                        image.SaveAsWebp(destFile, webpEncoder);
                    }

                    var destFileInfo = new FileInfo(destFile);
                    totalSavedBytes += sourceFileInfo.Length - destFileInfo.Length;
                    converted++;

                    if ((converted + skipped) % 10 == 0)
                    {
                        Console.WriteLine($"[AnalysisViewerAdapter] Progress: {converted} converted, {skipped} skipped, {converted + skipped}/{pngFiles.Length}");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[AnalysisViewerAdapter] Failed to convert {Path.GetFileName(sourceFile)}: {ex.Message}");
                }
            }

            var savedMB = totalSavedBytes / (1024.0 * 1024.0);
            if (converted > 0)
            {
                Console.WriteLine($"[AnalysisViewerAdapter] Converted {converted} tiles to WebP, skipped {skipped} existing, saved {savedMB:F1} MB");
            }
            else
            {
                Console.WriteLine($"[AnalysisViewerAdapter] All {skipped} tiles already converted to WebP");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[AnalysisViewerAdapter] Failed to process minimaps: {ex.Message}");
        }
    }
}
