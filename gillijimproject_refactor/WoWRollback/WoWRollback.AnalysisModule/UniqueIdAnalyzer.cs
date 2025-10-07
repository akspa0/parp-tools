using System.Text;
using System.Text.Json;

namespace WoWRollback.AnalysisModule;

/// <summary>
/// Analyzes UniqueID distributions in converted LK ADTs for time-travel visualization.
/// Detects work layers based on ID gaps.
/// </summary>
public sealed class UniqueIdAnalyzer
{
    private readonly int _gapThreshold;

    public UniqueIdAnalyzer(int gapThreshold = 100)
    {
        _gapThreshold = gapThreshold;
    }

    /// <summary>
    /// Analyzes UniqueID distributions for all tiles in a map.
    /// </summary>
    /// <param name="adtMapDir">Directory containing ADT files</param>
    /// <param name="mapName">Map name</param>
    /// <param name="outputDir">Output directory for CSVs</param>
    /// <returns>Analysis result with paths to generated files</returns>
    public UniqueIdAnalysisResult Analyze(string adtMapDir, string mapName, string outputDir)
    {
        try
        {
            var adtFiles = Directory.GetFiles(adtMapDir, "*.adt", SearchOption.TopDirectoryOnly);
            if (adtFiles.Length == 0)
            {
                return new UniqueIdAnalysisResult(
                    string.Empty,
                    string.Empty,
                    0,
                    Success: false,
                    ErrorMessage: $"No ADT files found in {adtMapDir}");
            }

            var distributions = new List<TileIdDistribution>();

            // Analyze each tile
            foreach (var adtPath in adtFiles)
            {
                var tileDistribution = AnalyzeTile(adtPath, mapName);
                if (tileDistribution != null)
                {
                    distributions.Add(tileDistribution);
                }
            }

            // Export CSV
            var csvPath = Path.Combine(outputDir, $"{mapName}_uniqueID_analysis.csv");
            ExportCsv(distributions, csvPath);

            // Detect global layers
            var globalLayers = DetectGlobalLayers(distributions, mapName);
            var layersJsonPath = Path.Combine(outputDir, $"{mapName}_layers.json");
            ExportLayersJson(globalLayers, layersJsonPath);

            return new UniqueIdAnalysisResult(
                CsvPath: csvPath,
                LayersJsonPath: layersJsonPath,
                TileCount: distributions.Count,
                Success: true);
        }
        catch (Exception ex)
        {
            return new UniqueIdAnalysisResult(
                string.Empty,
                string.Empty,
                0,
                Success: false,
                ErrorMessage: $"UniqueID analysis failed: {ex.Message}");
        }
    }

    private TileIdDistribution? AnalyzeTile(string adtPath, string mapName)
    {
        try
        {
            // TODO: Read LK ADT using WoWRollback.Core or AlphaWdtAnalyzer.Core
            // For now, return null as placeholder
            // Need to implement LK ADT reading to extract MDDF/MODF chunks
            
            // Parse tile coordinates from filename (e.g., "Shadowfang_25_30.adt")
            var fileName = Path.GetFileNameWithoutExtension(adtPath);
            var parts = fileName.Split('_');
            if (parts.Length < 3 || !int.TryParse(parts[^2], out var tileX) || !int.TryParse(parts[^1], out var tileY))
            {
                return null;
            }

            // Placeholder: Will implement actual ADT reading
            return new TileIdDistribution
            {
                MapName = mapName,
                TileX = tileX,
                TileY = tileY,
                M2Distribution = null,
                WmoDistribution = null
            };
        }
        catch
        {
            return null;
        }
    }

    private List<LayerInfo> DetectLayers(List<uint> sortedIds)
    {
        if (sortedIds.Count == 0)
            return new List<LayerInfo>();

        var layers = new List<LayerInfo>();
        int currentLayer = 1;
        uint layerStart = sortedIds[0];
        uint lastId = sortedIds[0];
        var currentLayerIds = new List<uint> { sortedIds[0] };

        for (int i = 1; i < sortedIds.Count; i++)
        {
            var currentId = sortedIds[i];
            var gap = currentId - lastId;

            if (gap > _gapThreshold)
            {
                // End current layer
                layers.Add(new LayerInfo
                {
                    LayerNumber = currentLayer,
                    IdRangeStart = layerStart,
                    IdRangeEnd = lastId,
                    ObjectCount = currentLayerIds.Count
                });

                // Start new layer
                currentLayer++;
                layerStart = currentId;
                currentLayerIds = new List<uint>();
            }

            currentLayerIds.Add(currentId);
            lastId = currentId;
        }

        // Add final layer
        layers.Add(new LayerInfo
        {
            LayerNumber = currentLayer,
            IdRangeStart = layerStart,
            IdRangeEnd = lastId,
            ObjectCount = currentLayerIds.Count
        });

        return layers;
    }

    private GlobalLayerInfo DetectGlobalLayers(List<TileIdDistribution> tiles, string mapName)
    {
        // Collect all IDs across all tiles
        var allIds = new List<uint>();

        foreach (var tile in tiles)
        {
            if (tile.M2Distribution != null)
            {
                for (uint id = tile.M2Distribution.MinId; id <= tile.M2Distribution.MaxId; id++)
                {
                    allIds.Add(id);
                }
            }

            if (tile.WmoDistribution != null)
            {
                for (uint id = tile.WmoDistribution.MinId; id <= tile.WmoDistribution.MaxId; id++)
                {
                    allIds.Add(id);
                }
            }
        }

        var sortedIds = allIds.Distinct().OrderBy(id => id).ToList();
        var globalLayers = DetectLayers(sortedIds);

        return new GlobalLayerInfo
        {
            MapName = mapName,
            AnalyzedTiles = tiles.Count,
            GlobalLayers = globalLayers
        };
    }

    private void ExportCsv(List<TileIdDistribution> distributions, string csvPath)
    {
        var csv = new StringBuilder();
        csv.AppendLine("MapName,TileX,TileY,AssetType,MinId,MaxId,Count,Layers");

        foreach (var tile in distributions.OrderBy(t => t.TileY).ThenBy(t => t.TileX))
        {
            if (tile.M2Distribution != null)
            {
                var layerCount = tile.M2Distribution.Layers.Count;
                csv.AppendLine($"{tile.MapName},{tile.TileX},{tile.TileY}," +
                    $"M2,{tile.M2Distribution.MinId},{tile.M2Distribution.MaxId}," +
                    $"{tile.M2Distribution.Count},{layerCount}");
            }

            if (tile.WmoDistribution != null)
            {
                var layerCount = tile.WmoDistribution.Layers.Count;
                csv.AppendLine($"{tile.MapName},{tile.TileX},{tile.TileY}," +
                    $"WMO,{tile.WmoDistribution.MinId},{tile.WmoDistribution.MaxId}," +
                    $"{tile.WmoDistribution.Count},{layerCount}");
            }
        }

        File.WriteAllText(csvPath, csv.ToString());
    }

    private void ExportLayersJson(GlobalLayerInfo globalLayers, string jsonPath)
    {
        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };

        var json = JsonSerializer.Serialize(globalLayers, options);
        File.WriteAllText(jsonPath, json);
    }
}
