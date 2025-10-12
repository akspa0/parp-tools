using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;

namespace WoWRollback.AnalysisModule;

/// <summary>
/// Analyzes spatial clusters and patterns in object placements.
/// Reveals hidden prefabs, object brushes, and linked object groups.
/// </summary>
public sealed class ClusterAnalyzer
{
    private readonly SpatialClusterDetector _detector;
    private readonly float _proximityThreshold;
    private readonly int _minClusterSize;

    public ClusterAnalyzer(float proximityThreshold = 50.0f, int minClusterSize = 3)
    {
        _proximityThreshold = proximityThreshold;
        _minClusterSize = minClusterSize;
        _detector = new SpatialClusterDetector(proximityThreshold);
    }

    /// <summary>
    /// Analyzes placements CSV and outputs spatial cluster data.
    /// </summary>
    public ClusterAnalysisResult Analyze(string placementsCsvPath, string mapName, string outputDir)
    {
        try
        {
            if (!File.Exists(placementsCsvPath))
            {
                return new ClusterAnalysisResult(
                    Success: false,
                    TotalClusters: 0,
                    TotalPatterns: 0,
                    ErrorMessage: $"Placements CSV not found: {placementsCsvPath}");
            }

            Directory.CreateDirectory(outputDir);

            // Load placements
            var placements = LoadPlacements(placementsCsvPath);
            if (placements.Count == 0)
            {
                return new ClusterAnalysisResult(
                    Success: false,
                    TotalClusters: 0,
                    TotalPatterns: 0,
                    ErrorMessage: "No placements found in CSV");
            }

            // Detect clusters per-tile
            var tileResults = new List<TileClusterResult>();
            var allClusters = new List<SpatialCluster>();

            var byTile = placements.GroupBy(p => (p.TileX, p.TileY));
            foreach (var tileGroup in byTile)
            {
                var (tileX, tileY) = tileGroup.Key;
                var tilePlacements = tileGroup.Select(p => new PlacementPoint(
                    p.UniqueId,
                    p.Type,
                    p.AssetPath,
                    p.WorldX,
                    p.WorldY,
                    p.WorldZ)).ToList();

                // Detect tile-level clusters
                var tileClusters = _detector.DetectClusters(tilePlacements, _minClusterSize);
                
                // Detect chunk-level clusters (more granular)
                var chunkClusters = _detector.DetectChunkClusters(tilePlacements, _minClusterSize);

                tileResults.Add(new TileClusterResult
                {
                    TileX = tileX,
                    TileY = tileY,
                    TotalPlacements = tilePlacements.Count,
                    TileLevelClusters = tileClusters.Count,
                    ChunkLevelClusters = chunkClusters.Count,
                    Clusters = chunkClusters // Use chunk-level for more granularity
                });

                allClusters.AddRange(chunkClusters);
            }

            // Identify recurring patterns (potential prefabs)
            var patterns = _detector.IdentifyPatterns(allClusters, similarityThreshold: 0.75f);

            // Export results
            var clustersJsonPath = Path.Combine(outputDir, $"{mapName}_spatial_clusters.json");
            var patternsJsonPath = Path.Combine(outputDir, $"{mapName}_patterns.json");
            var summaryPath = Path.Combine(outputDir, $"{mapName}_cluster_summary.csv");

            ExportClustersJson(tileResults, clustersJsonPath);
            ExportPatternsJson(patterns, patternsJsonPath);
            ExportSummaryCsv(tileResults, patterns, mapName, summaryPath);

            return new ClusterAnalysisResult(
                Success: true,
                TotalClusters: allClusters.Count,
                TotalPatterns: patterns.Count,
                ClustersJsonPath: clustersJsonPath,
                PatternsJsonPath: patternsJsonPath,
                SummaryCsvPath: summaryPath,
                ErrorMessage: null);
        }
        catch (Exception ex)
        {
            return new ClusterAnalysisResult(
                Success: false,
                TotalClusters: 0,
                TotalPatterns: 0,
                ErrorMessage: $"Cluster analysis failed: {ex.Message}");
        }
    }

    private List<CsvPlacement> LoadPlacements(string csvPath)
    {
        var placements = new List<CsvPlacement>();

        using var reader = new StreamReader(csvPath);
        string? header = reader.ReadLine(); // Skip header

        while (reader.ReadLine() is { } line)
        {
            if (string.IsNullOrWhiteSpace(line))
                continue;

            var fields = line.Split(',');
            if (fields.Length < 10)
                continue;

            if (!int.TryParse(fields[1], out var tileX) ||
                !int.TryParse(fields[2], out var tileY) ||
                !int.TryParse(fields[5], out var uniqueId) ||
                !float.TryParse(fields[6], NumberStyles.Float, CultureInfo.InvariantCulture, out var worldX) ||
                !float.TryParse(fields[7], NumberStyles.Float, CultureInfo.InvariantCulture, out var worldY) ||
                !float.TryParse(fields[8], NumberStyles.Float, CultureInfo.InvariantCulture, out var worldZ))
                continue;

            placements.Add(new CsvPlacement
            {
                TileX = tileX,
                TileY = tileY,
                Type = fields[3],
                AssetPath = fields[4],
                UniqueId = uniqueId,
                WorldX = worldX,
                WorldY = worldY,
                WorldZ = worldZ
            });
        }

        return placements;
    }

    private void ExportClustersJson(List<TileClusterResult> tileResults, string jsonPath)
    {
        var output = new
        {
            tiles = tileResults.Select(t => new
            {
                tileX = t.TileX,
                tileY = t.TileY,
                totalPlacements = t.TotalPlacements,
                tileLevelClusters = t.TileLevelClusters,
                chunkLevelClusters = t.ChunkLevelClusters,
                clusters = t.Clusters.Select(c => new
                {
                    clusterId = c.ClusterId,
                    objectCount = c.ObjectCount,
                    centroid = new { x = c.CentroidX, y = c.CentroidY, z = c.CentroidZ },
                    boundingRadius = c.BoundingRadius,
                    uniqueIdRange = new { min = c.UniqueIdMin, max = c.UniqueIdMax },
                    uniqueIds = c.UniqueIds,
                    assetTypes = c.AssetTypes,
                    assetPaths = c.Placements.Select(p => p.AssetPath).Distinct().ToList(),
                    isPlacementStamp = c.IsPlacementStamp,
                    uniqueIdGaps = DetectUniqueIdGaps(c.UniqueIds)
                }).ToList()
            }).ToList()
        };

        var options = new JsonSerializerOptions { WriteIndented = true };
        File.WriteAllText(jsonPath, JsonSerializer.Serialize(output, options));
    }

    private void ExportPatternsJson(List<ClusterPattern> patterns, string jsonPath)
    {
        var output = new
        {
            totalPatterns = patterns.Count,
            patterns = patterns.Select(p => new
            {
                patternId = p.PatternId,
                instanceCount = p.Instances.Count,
                instances = p.Instances.Select(i => new
                {
                    clusterId = i.ClusterId,
                    objectCount = i.ObjectCount,
                    centroid = new { x = i.CentroidX, y = i.CentroidY, z = i.CentroidZ },
                    boundingRadius = i.BoundingRadius,
                    uniqueIdRange = new { min = i.UniqueIdMin, max = i.UniqueIdMax }
                }).ToList()
            }).ToList()
        };

        var options = new JsonSerializerOptions { WriteIndented = true };
        File.WriteAllText(jsonPath, JsonSerializer.Serialize(output, options));
    }

    private void ExportSummaryCsv(List<TileClusterResult> tileResults, List<ClusterPattern> patterns, string mapName, string csvPath)
    {
        var csv = new StringBuilder();
        csv.AppendLine("map,tile_x,tile_y,total_placements,tile_clusters,chunk_clusters");

        foreach (var tile in tileResults)
        {
            csv.AppendLine($"{mapName},{tile.TileX},{tile.TileY},{tile.TotalPlacements},{tile.TileLevelClusters},{tile.ChunkLevelClusters}");
        }

        File.WriteAllText(csvPath, csv.ToString());
    }

    private record CsvPlacement
    {
        public required int TileX { get; init; }
        public required int TileY { get; init; }
        public required string Type { get; init; }
        public required string AssetPath { get; init; }
        public required int UniqueId { get; init; }
        public required float WorldX { get; init; }
        public required float WorldY { get; init; }
        public required float WorldZ { get; init; }
    }

    private record TileClusterResult
    {
        public required int TileX { get; init; }
        public required int TileY { get; init; }
        public required int TotalPlacements { get; init; }
        public required int TileLevelClusters { get; init; }
        public required int ChunkLevelClusters { get; init; }
        public required List<SpatialCluster> Clusters { get; init; }
    }

    /// <summary>
    /// Detects gaps in UniqueID sequences (potential missing objects or edits).
    /// </summary>
    private static List<(int Start, int End, int Gap)> DetectUniqueIdGaps(List<int> uniqueIds)
    {
        var gaps = new List<(int Start, int End, int Gap)>();
        
        if (uniqueIds.Count < 2)
            return gaps;

        for (int i = 1; i < uniqueIds.Count; i++)
        {
            int gap = uniqueIds[i] - uniqueIds[i - 1];
            if (gap > 10) // Significant gap
            {
                gaps.Add((uniqueIds[i - 1], uniqueIds[i], gap));
            }
        }

        return gaps;
    }
}

/// <summary>
/// Result of cluster analysis.
/// </summary>
public record ClusterAnalysisResult(
    bool Success,
    int TotalClusters,
    int TotalPatterns,
    string? ClustersJsonPath = null,
    string? PatternsJsonPath = null,
    string? SummaryCsvPath = null,
    string? ErrorMessage = null);
