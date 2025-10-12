using System.Text;
using System.Text.Json;
using System.Linq;
using System.Globalization;
using System.IO;
using AlphaWdtAnalyzer.Core;

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
    /// Analyze UniqueIDs from placements.csv when analysis index is not available.
    /// Emits:
    /// - {map}_uniqueID_by_tile.csv (one row per unique id with assigned layer per tile)
    /// - {map}_tile_layers.csv (one row per tile-layer with range and count)
    /// - {map}_layers.json (global summary) reusing existing JSON format
    /// Also writes the legacy {map}_uniqueID_analysis.csv for compatibility.
    /// </summary>
    public UniqueIdAnalysisResult AnalyzeFromPlacementsCsv(string placementsCsvPath, string mapName, string outputDir)
    {
        try
        {
            if (!File.Exists(placementsCsvPath))
            {
                return new UniqueIdAnalysisResult(
                    string.Empty,
                    string.Empty,
                    0,
                    Success: false,
                    ErrorMessage: $"Placements CSV not found: {placementsCsvPath}");
            }

            Directory.CreateDirectory(outputDir);

            // Expected columns:
            // map,tile_x,tile_y,type,asset_path,unique_id,world_x,world_y,world_z,rot_x,rot_y,rot_z,scale,doodad_set,name_set
            var rows = new List<string[]>();
            using (var reader = new StreamReader(placementsCsvPath))
            {
                string? line = reader.ReadLine(); // header
                while ((line = reader.ReadLine()) != null)
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    rows.Add(SplitCsv(line));
                }
            }

            // Group by tile and asset type (CSV order: map,tile_x,tile_y,type,asset_path,unique_id,...)
            var groups = rows
                .Where(r => r.Length >= 15)
                .GroupBy(r => (tileX: ParseInt(r[1]), tileY: ParseInt(r[2]), type: (r[3] ?? string.Empty).Trim()))
                .OrderBy(g => g.Key.tileY).ThenBy(g => g.Key.tileX).ThenBy(g => g.Key.type)
                .ToList();

            var perIdCsv = new StringBuilder();
            perIdCsv.AppendLine("map,tile_x,tile_y,type,unique_id,layer,layer_start,layer_end,asset_path");

            var perTileLayersCsv = new StringBuilder();
            perTileLayersCsv.AppendLine("map,tile_x,tile_y,type,layer,range_start,range_end,count");

            var tileDistributions = new List<TileIdDistribution>();

            foreach (var g in groups)
            {
                var (tileX, tileY, typeStr) = g.Key;
                // Collect IDs and asset paths
                var idsWithPath = g
                    .Select(r => (id: ParseUInt(r[5]), asset: r[4] ?? string.Empty))
                    .Where(t => t.id.HasValue)
                    .Select(t => (id: t.id!.Value, asset: t.asset))
                    .OrderBy(t => t.id)
                    .ToList();

                var idsOnly = idsWithPath.Select(t => t.id).ToList();
                var layers = DetectLayers(idsOnly);

                foreach (var (id, asset) in idsWithPath)
                {
                    var layerNum = FindLayer(layers, id);
                    var l = layers.FirstOrDefault(x => x.LayerNumber == layerNum);
                    var start = l is null ? id : l.IdRangeStart;
                    var end = l is null ? id : l.IdRangeEnd;
                    perIdCsv.AppendLine(string.Join(",",
                        Csv(mapName),
                        tileX.ToString(CultureInfo.InvariantCulture),
                        tileY.ToString(CultureInfo.InvariantCulture),
                        Csv(typeStr),
                        id.ToString(CultureInfo.InvariantCulture),
                        layerNum.ToString(CultureInfo.InvariantCulture),
                        start.ToString(CultureInfo.InvariantCulture),
                        end.ToString(CultureInfo.InvariantCulture),
                        Csv(asset)));
                }

                foreach (var l in layers)
                {
                    perTileLayersCsv.AppendLine(string.Join(",",
                        Csv(mapName),
                        tileX.ToString(CultureInfo.InvariantCulture),
                        tileY.ToString(CultureInfo.InvariantCulture),
                        Csv(typeStr),
                        l.LayerNumber.ToString(CultureInfo.InvariantCulture),
                        l.IdRangeStart.ToString(CultureInfo.InvariantCulture),
                        l.IdRangeEnd.ToString(CultureInfo.InvariantCulture),
                        l.ObjectCount.ToString(CultureInfo.InvariantCulture)));
                }

                // Build distribution for global aggregation
                var isM2 = string.Equals(typeStr, "M2", StringComparison.OrdinalIgnoreCase);
                var isWmo = string.Equals(typeStr, "WMO", StringComparison.OrdinalIgnoreCase);
                tileDistributions.Add(new TileIdDistribution
                {
                    MapName = mapName,
                    TileX = tileX,
                    TileY = tileY,
                    M2Distribution = isM2 ? new IdDistribution
                    {
                        MinId = idsOnly.Count > 0 ? idsOnly.Min() : 0u,
                        MaxId = idsOnly.Count > 0 ? idsOnly.Max() : 0u,
                        Count = idsOnly.Count,
                        Layers = layers
                    } : null,
                    WmoDistribution = isWmo ? new IdDistribution
                    {
                        MinId = idsOnly.Count > 0 ? idsOnly.Min() : 0u,
                        MaxId = idsOnly.Count > 0 ? idsOnly.Max() : 0u,
                        Count = idsOnly.Count,
                        Layers = layers
                    } : null
                });
            }

            var byTileCsvPath = Path.Combine(outputDir, $"{mapName}_uniqueID_by_tile.csv");
            File.WriteAllText(byTileCsvPath, perIdCsv.ToString());

            var tileLayersCsvPath = Path.Combine(outputDir, $"{mapName}_tile_layers.csv");
            File.WriteAllText(tileLayersCsvPath, perTileLayersCsv.ToString());

            // Global layers JSON
            var global = DetectGlobalLayers(tileDistributions, mapName);
            var layersJsonPath = Path.Combine(outputDir, $"{mapName}_layers.json");
            ExportLayersJson(global, layersJsonPath);

            // Legacy compatibility CSV
            var compatCsvPath = Path.Combine(outputDir, $"{mapName}_uniqueID_analysis.csv");
            ExportCsv(tileDistributions, compatCsvPath);

            return new UniqueIdAnalysisResult(
                CsvPath: compatCsvPath,
                LayersJsonPath: layersJsonPath,
                TileCount: tileDistributions.Select(t => (t.TileX, t.TileY)).Distinct().Count(),
                Success: true);
        }
        catch (Exception ex)
        {
            return new UniqueIdAnalysisResult(
                string.Empty,
                string.Empty,
                0,
                Success: false,
                ErrorMessage: $"UniqueID analysis (placements.csv) failed: {ex.Message}");
        }

        static int FindLayer(List<LayerInfo> layers, uint id)
        {
            foreach (var l in layers)
            {
                if (id >= l.IdRangeStart && id <= l.IdRangeEnd) return l.LayerNumber;
            }
            return 0;
        }

        static string[] SplitCsv(string line)
        {
            var result = new List<string>();
            bool inQuotes = false;
            var current = new StringBuilder();
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

        static int ParseInt(string s) => int.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out var v) ? v : 0;
        static uint? ParseUInt(string s)
        {
            if (string.IsNullOrWhiteSpace(s)) return null;
            if (uint.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out var v)) return v;
            if (int.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out var si) && si >= 0) return (uint)si;
            return null;
        }
        static string Csv(string s)
        {
            if (string.IsNullOrEmpty(s)) return s;
            if (s.Contains('"') || s.Contains(',')) return '"' + s.Replace("\"", "\"\"") + '"';
            return s;
        }
    }

    /// <summary>
    /// Analyzes UniqueID distributions from AnalysisIndex (generated by AnalysisPipeline).
    /// </summary>
    /// <param name="analysisIndex">Analysis index with placement data</param>
    /// <param name="mapName">Map name</param>
    /// <param name="outputDir">Output directory for CSVs</param>
    /// <returns>Analysis result with paths to generated files</returns>
    public UniqueIdAnalysisResult AnalyzeFromIndex(AnalysisIndex analysisIndex, string mapName, string outputDir)
    {
        try
        {
            if (analysisIndex.Placements.Count == 0)
            {
                return new UniqueIdAnalysisResult(
                    string.Empty,
                    string.Empty,
                    0,
                    Success: false,
                    ErrorMessage: $"No placements found in analysis index for {mapName}");
            }

            var distributions = new List<TileIdDistribution>();

            // Group placements by tile
            var placementsByTile = analysisIndex.Placements
                .GroupBy(p => (p.TileX, p.TileY))
                .OrderBy(g => g.Key.TileY)
                .ThenBy(g => g.Key.TileX);

            foreach (var tileGroup in placementsByTile)
            {
                var (tileX, tileY) = tileGroup.Key;
                var tilePlacements = tileGroup.ToList();

                // Separate M2 and WMO placements
                var m2Placements = tilePlacements.Where(p => p.Type == AssetType.MdxOrM2 && p.UniqueId.HasValue).ToList();
                var wmoPlacements = tilePlacements.Where(p => p.Type == AssetType.Wmo && p.UniqueId.HasValue).ToList();

                IdDistribution? m2Distribution = null;
                IdDistribution? wmoDistribution = null;

                if (m2Placements.Count > 0)
                {
                    var m2Ids = m2Placements
                        .Select(p => p.UniqueId!.Value)
                        .Where(id => id >= 0)
                        .Select(id => (uint)id)
                        .OrderBy(id => id)
                        .ToList();
                    var m2Layers = DetectLayers(m2Ids);
                    m2Distribution = new IdDistribution
                    {
                        MinId = m2Ids.Count > 0 ? m2Ids.Min() : 0u,
                        MaxId = m2Ids.Count > 0 ? m2Ids.Max() : 0u,
                        Count = m2Ids.Count,
                        Layers = m2Layers
                    };
                }

                if (wmoPlacements.Count > 0)
                {
                    var wmoIds = wmoPlacements
                        .Select(p => p.UniqueId!.Value)
                        .Where(id => id >= 0)
                        .Select(id => (uint)id)
                        .OrderBy(id => id)
                        .ToList();
                    var wmoLayers = DetectLayers(wmoIds);
                    wmoDistribution = new IdDistribution
                    {
                        MinId = wmoIds.Count > 0 ? wmoIds.Min() : 0u,
                        MaxId = wmoIds.Count > 0 ? wmoIds.Max() : 0u,
                        Count = wmoIds.Count,
                        Layers = wmoLayers
                    };
                }

                distributions.Add(new TileIdDistribution
                {
                    MapName = mapName,
                    TileX = tileX,
                    TileY = tileY,
                    M2Distribution = m2Distribution,
                    WmoDistribution = wmoDistribution
                });
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

    /// <summary>
    /// Analyzes UniqueID distributions for all tiles in a map (legacy method - reads ADTs directly).
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
