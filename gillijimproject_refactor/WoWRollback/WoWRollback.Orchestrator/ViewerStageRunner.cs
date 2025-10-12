using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AlphaWdtAnalyzer.Core;
using WoWRollback.Core.IO;
using WoWRollback.Core.Logging;
using WoWRollback.Core.Models;
using WoWRollback.Core.Services.Viewer;

namespace WoWRollback.Orchestrator;

internal sealed class ViewerStageRunner
{
    private const string ViewerAssetsSourcePath = "WoWRollback.Viewer/assets";

    public ViewerStageResult Run(SessionContext session, IReadOnlyList<AdtStageResult> adtResults)
    {
        if (session is null)
        {
            throw new ArgumentNullException(nameof(session));
        }

        try
        {
            Directory.CreateDirectory(session.Paths.ViewerDir);

            // Copy existing viewer assets (index.html, JS, CSS, etc.)
            CopyViewerAssets(session);

            // Generate minimap PNG tiles from ADT files
            var minimapCount = GenerateMinimapTiles(session, adtResults);

            // Generate viewer data files (index.json, config.json)
            GenerateViewerDataFiles(session, adtResults);

            // Generate overlay metadata
            var overlayCount = GenerateOverlayMetadata(session, adtResults);

            return new ViewerStageResult(
                Success: true,
                ViewerDirectory: session.Paths.ViewerDir,
                OverlayCount: overlayCount,
                Notes: $"Generated {minimapCount} minimap tiles and {overlayCount} overlay(s)");
        }
        catch (Exception ex)
        {
            return new ViewerStageResult(
                Success: false,
                ViewerDirectory: session.Paths.ViewerDir,
                OverlayCount: 0,
                Notes: $"Viewer generation failed: {ex.Message}");
        }
    }

    private static void CopyViewerAssets(SessionContext session)
    {
        var sourceDir = Path.GetFullPath(ViewerAssetsSourcePath);
        
        if (!Directory.Exists(sourceDir))
        {
            throw new DirectoryNotFoundException($"Viewer assets not found at: {sourceDir}");
        }

        // Copy all viewer assets to session viewer directory
        FileHelpers.CopyDirectory(sourceDir, session.Paths.ViewerDir, overwrite: true);
    }

    private static void GenerateViewerDataFiles(SessionContext session, IReadOnlyList<AdtStageResult> adtResults)
    {
        // Load actual tile data from analysis indices
        var mapTiles = new Dictionary<string, List<TileInfo>>();
        
        foreach (var result in adtResults.Where(r => r.Success))
        {
            var analysisIndexPath = Path.Combine(
                session.Paths.AdtDir, 
                result.Version, 
                "analysis", 
                result.Map,
                "index.json");
                
            if (File.Exists(analysisIndexPath))
            {
                try
                {
                    var indexJson = File.ReadAllText(analysisIndexPath);
                    var analysisIndex = JsonSerializer.Deserialize<AnalysisIndex>(indexJson);
                    
                    if (analysisIndex != null)
                    {
                        if (!mapTiles.ContainsKey(result.Map))
                        {
                            mapTiles[result.Map] = new List<TileInfo>();
                        }
                        
                        foreach (var tile in analysisIndex.Tiles)
                        {
                            mapTiles[result.Map].Add(new TileInfo
                            {
                                Row = tile.X,
                                Col = tile.Y,
                                Versions = new[] { result.Version }
                            });
                        }
                    }
                }
                catch (Exception ex)
                {
                    ConsoleLogger.Warn($"  ⚠ Failed to load analysis index for {result.Map}: {ex.Message}");
                }
            }
        }
        
        // Generate index.json in viewer-expected format
        var indexData = new
        {
            comparisonKey = session.Options.Versions.FirstOrDefault() ?? "0.5.3",
            versions = session.Options.Versions.ToArray(),
            maps = session.Options.Maps.Select(mapName => new
            {
                map = mapName,  // Viewer expects "map" property, not "name"
                tiles = mapTiles.ContainsKey(mapName) ? mapTiles[mapName].ToArray() : Array.Empty<TileInfo>()
            }).ToArray()
        };

        var indexPath = Path.Combine(session.Paths.ViewerDir, "index.json");
        File.WriteAllText(indexPath, JsonSerializer.Serialize(indexData, new JsonSerializerOptions { WriteIndented = true }));

        // Generate config.json - viewer configuration
        var configData = new
        {
            default_version = session.Options.Versions.FirstOrDefault() ?? "0.5.3",
            default_map = session.Options.Maps.FirstOrDefault() ?? "Kalimdor",
            coordMode = "wowtools",  // CRITICAL: Enable Y-axis inversion for proper tile layout
            tile_size = 512,
            minimap = new
            {
                width = 512,
                height = 512
            },
            debugOverlayCorners = false,
            diff_thresholds = new
            {
                proximity = 10.0,
                moved_epsilon = 0.005
            }
        };

        var configPath = Path.Combine(session.Paths.ViewerDir, "config.json");
        File.WriteAllText(configPath, JsonSerializer.Serialize(configData, new JsonSerializerOptions { WriteIndented = true }));
    }

    private static int GenerateMinimapTiles(SessionContext session, IReadOnlyList<AdtStageResult> adtResults)
    {
        // Build MinimapLocator - use MPQ provider if mpq-path is specified, otherwise use loose files
        MinimapLocator locator;
        if (session.Options.HasMpqPath)
        {
            // Build version -> MPQ path mapping
            var versionMpqPaths = session.Options.Versions.ToDictionary(
                v => v,
                v => Path.Combine(session.Options.MpqPath!, v));
            locator = MinimapLocator.BuildFromMpq(versionMpqPaths);
        }
        else
        {
            locator = MinimapLocator.Build(
                session.Options.AlphaRoot,
                session.Options.Versions.ToList());
        }

        var composer = new MinimapComposer();
        var options = ViewerOptions.CreateDefault();
        int totalTiles = 0;

        foreach (var result in adtResults.Where(r => r.Success))
        {
            var minimapOutDir = Path.Combine(session.Paths.ViewerDir, "minimap", result.Version, result.Map);
            Directory.CreateDirectory(minimapOutDir);

            // Enumerate actual minimap tiles from source data
            var tiles = locator.EnumerateTiles(result.Version, result.Map).ToList();
            
            foreach (var (row, col) in tiles)
            {
                if (locator.TryGetTile(result.Version, result.Map, row, col, out var tile))
                {
                    var fileName = tile.BuildFileName(result.Map);
                    var pngPath = Path.Combine(minimapOutDir, fileName);

                    try
                    {
                        using var stream = tile.Open();
                        Task.Run(async () => await composer.ComposeAsync(stream, pngPath, options)).Wait();
                        totalTiles++;
                    }
                    catch (Exception ex)
                    {
                        ConsoleLogger.Warn($"  ⚠ Failed to generate minimap for {result.Map} tile [{row},{col}]: {ex.Message}");
                    }
                }
            }

            if (tiles.Count > 0)
            {
                ConsoleLogger.Success($"  ✓ Generated {tiles.Count} minimap tiles for {result.Map}");
            }
        }

        return totalTiles;
    }

    private static int GenerateOverlayMetadata(SessionContext session, IReadOnlyList<AdtStageResult> adtResults)
    {
        var overlaysDir = Path.Combine(session.Paths.ViewerDir, "overlays");
        Directory.CreateDirectory(overlaysDir);

        var metadata = new
        {
            session_id = session.SessionId,
            maps = session.Options.Maps,
            versions = session.Options.Versions,
            adt_results = adtResults.Select(r => new
            {
                map = r.Map,
                version = r.Version,
                success = r.Success,
                tiles_processed = r.TilesProcessed,
                area_ids_patched = r.AreaIdsPatched
            }).ToList()
        };

        var metadataPath = Path.Combine(overlaysDir, "metadata.json");
        var options = new JsonSerializerOptions { WriteIndented = true };
        File.WriteAllText(metadataPath, JsonSerializer.Serialize(metadata, options));

        return 1; // metadata.json counts as 1 overlay
    }
}

internal sealed record ViewerStageResult(
    bool Success,
    string ViewerDirectory,
    int OverlayCount,
    string? Notes);

internal sealed class TileInfo
{
    [System.Text.Json.Serialization.JsonPropertyName("row")]
    public int Row { get; set; }
    
    [System.Text.Json.Serialization.JsonPropertyName("col")]
    public int Col { get; set; }
    
    [System.Text.Json.Serialization.JsonPropertyName("versions")]
    public string[] Versions { get; set; } = Array.Empty<string>();
}
