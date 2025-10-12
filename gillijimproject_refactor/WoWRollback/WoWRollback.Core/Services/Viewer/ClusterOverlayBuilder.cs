using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace WoWRollback.Core.Services.Viewer;

/// <summary>
/// Builds cluster overlay JSON files for viewer visualization.
/// Converts spatial cluster data into per-tile overlay format.
/// </summary>
public static class ClusterOverlayBuilder
{
    public static void BuildClusterOverlays(
        string clusterJsonPath,
        string mapName,
        string outputDir,
        ViewerOptions options)
    {
        if (!File.Exists(clusterJsonPath))
        {
            Console.WriteLine($"[ClusterOverlayBuilder] Cluster JSON not found: {clusterJsonPath}");
            return;
        }

        try
        {
            // Read cluster JSON
            var json = File.ReadAllText(clusterJsonPath);
            var clusterData = JsonSerializer.Deserialize<ClusterDataRoot>(json);
            
            if (clusterData?.Tiles == null)
            {
                Console.WriteLine($"[ClusterOverlayBuilder] No cluster data found in JSON");
                return;
            }

            // Group clusters by tile
            var clustersByTile = new Dictionary<(int Row, int Col), List<ClusterInfo>>();
            
            foreach (var tile in clusterData.Tiles)
            {
                var key = (tile.TileY, tile.TileX);
                if (!clustersByTile.ContainsKey(key))
                {
                    clustersByTile[key] = new List<ClusterInfo>();
                }
                
                foreach (var cluster in tile.Clusters)
                {
                    clustersByTile[key].Add(cluster);
                }
            }

            Console.WriteLine($"[ClusterOverlayBuilder] Processing {clustersByTile.Count} tiles with clusters");

            // Generate per-tile overlay JSON
            var overlayDir = Path.Combine(outputDir, mapName, "clusters");
            Directory.CreateDirectory(overlayDir);

            foreach (var ((row, col), clusters) in clustersByTile)
            {
                var overlayData = BuildTileClusterOverlay(clusters, row, col, mapName, options);
                var outputPath = Path.Combine(overlayDir, $"tile_{col}_{row}.json");
                File.WriteAllText(outputPath, overlayData);
            }

            Console.WriteLine($"[ClusterOverlayBuilder] Generated cluster overlays for {clustersByTile.Count} tiles");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ClusterOverlayBuilder] Error: {ex.Message}");
        }
    }

    private static string BuildTileClusterOverlay(
        List<ClusterInfo> clusters,
        int tileRow,
        int tileCol,
        string mapName,
        ViewerOptions options)
    {
        var overlayItems = new List<object>();

        foreach (var cluster in clusters)
        {
            // Convert cluster centroid to tile-local coordinates
            var (worldX, worldY) = (cluster.CentroidX, cluster.CentroidZ);
            
            var (localX, localY) = CoordinateTransformer.ComputeLocalCoordinates(worldX, worldY, tileRow, tileCol);
            var (pixelX, pixelY) = CoordinateTransformer.ToPixels(localX, localY, options.MinimapWidth, options.MinimapHeight);

            // Skip clusters outside tile bounds
            if (localX < -0.1 || localX > 1.1 || localY < -0.1 || localY > 1.1)
                continue;

            overlayItems.Add(new
            {
                clusterId = cluster.ClusterId,
                objectCount = cluster.ObjectCount,
                centroid = new
                {
                    x = cluster.CentroidX,
                    y = cluster.CentroidY,
                    z = cluster.CentroidZ
                },
                bounds = new
                {
                    minX = cluster.MinX,
                    maxX = cluster.MaxX,
                    minY = cluster.MinY,
                    maxY = cluster.MaxY,
                    minZ = cluster.MinZ,
                    maxZ = cluster.MaxZ
                },
                position = new
                {
                    x = pixelX,
                    y = pixelY,
                    localX,
                    localY
                },
                radius = CalculateClusterRadius(cluster),
                isStamp = cluster.IsStamp,
                hasConsecutiveIds = cluster.HasConsecutiveIds,
                tileX = cluster.TileX,
                tileY = cluster.TileY
            });
        }

        var payload = new
        {
            map = mapName,
            tile = new { row = tileRow, col = tileCol },
            clusterCount = overlayItems.Count,
            clusters = overlayItems
        };

        return JsonSerializer.Serialize(payload, new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            WriteIndented = true
        });
    }

    private static double CalculateClusterRadius(ClusterInfo cluster)
    {
        // Calculate bounding box diagonal as cluster radius
        double dx = cluster.MaxX - cluster.MinX;
        double dz = cluster.MaxZ - cluster.MinZ;
        return Math.Sqrt(dx * dx + dz * dz) / 2.0;
    }

    // JSON deserialization models
    private class ClusterDataRoot
    {
        [JsonPropertyName("tiles")]
        public List<TileClusterData>? Tiles { get; set; }
    }

    private class TileClusterData
    {
        [JsonPropertyName("tileX")]
        public int TileX { get; set; }
        
        [JsonPropertyName("tileY")]
        public int TileY { get; set; }
        
        [JsonPropertyName("clusters")]
        public List<ClusterInfo> Clusters { get; set; } = new();
    }

    private class ClusterInfo
    {
        [JsonPropertyName("clusterId")]
        public int ClusterId { get; set; }
        
        [JsonPropertyName("objectCount")]
        public int ObjectCount { get; set; }
        
        [JsonPropertyName("centroidX")]
        public float CentroidX { get; set; }
        
        [JsonPropertyName("centroidY")]
        public float CentroidY { get; set; }
        
        [JsonPropertyName("centroidZ")]
        public float CentroidZ { get; set; }
        
        [JsonPropertyName("minX")]
        public float MinX { get; set; }
        
        [JsonPropertyName("maxX")]
        public float MaxX { get; set; }
        
        [JsonPropertyName("minY")]
        public float MinY { get; set; }
        
        [JsonPropertyName("maxY")]
        public float MaxY { get; set; }
        
        [JsonPropertyName("minZ")]
        public float MinZ { get; set; }
        
        [JsonPropertyName("maxZ")]
        public float MaxZ { get; set; }
        
        [JsonPropertyName("isStamp")]
        public bool IsStamp { get; set; }
        
        [JsonPropertyName("hasConsecutiveIds")]
        public bool HasConsecutiveIds { get; set; }
        
        [JsonPropertyName("tileX")]
        public int TileX { get; set; }
        
        [JsonPropertyName("tileY")]
        public int TileY { get; set; }
    }
}
