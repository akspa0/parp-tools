using System;
using System.Collections.Generic;
using System.Linq;

namespace WoWRollback.AnalysisModule;

/// <summary>
/// Detects spatial clusters of objects that are placed near each other.
/// These clusters likely represent prefabs, object brushes, or linked object groups
/// used by the WoW world editor.
/// </summary>
public sealed class SpatialClusterDetector
{
    private readonly float _proximityThreshold;

    /// <summary>
    /// Creates a new spatial cluster detector.
    /// </summary>
    /// <param name="proximityThreshold">Distance threshold for clustering (game units).</param>
    public SpatialClusterDetector(float proximityThreshold = 50.0f)
    {
        _proximityThreshold = proximityThreshold;
    }

    /// <summary>
    /// Detects asset-homogeneous clusters using refined spatial+asset grouping.
    /// Only groups objects with identical asset paths within quantized grid cells.
    /// </summary>
    /// <param name="placements">Placements to cluster.</param>
    /// <param name="minClusterSize">Minimum objects to form a cluster.</param>
    /// <param name="gridSize">Grid cell size for quantization (default: 32 units).</param>
    /// <returns>List of detected clusters.</returns>
    public List<SpatialCluster> DetectClusters(
        IEnumerable<PlacementPoint> placements,
        int minClusterSize = 3,
        float gridSize = 32.0f)
    {
        var points = placements.ToList();
        
        // First pass: Group by asset path + grid cell
        var assetGridGroups = points
            .GroupBy(p => (
                AssetPath: p.AssetPath,
                GridX: (int)(p.WorldX / gridSize),
                GridY: (int)(p.WorldY / gridSize),
                GridZ: (int)(p.WorldZ / gridSize)
            ))
            .Where(g => g.Count() >= minClusterSize)
            .ToList();

        var clusters = new List<SpatialCluster>();
        int clusterIndex = 0;

        foreach (var group in assetGridGroups)
        {
            var groupPoints = group.ToList();
            
            // Check for consecutive UniqueID ranges (placement stamps)
            var orderedByUid = groupPoints.OrderBy(p => p.UniqueId).ToList();
            var stamps = DetectPlacementStamps(orderedByUid);

            // Create clusters for each stamp
            foreach (var stamp in stamps)
            {
                if (stamp.Count >= minClusterSize)
                {
                    clusters.Add(CreateCluster(clusterIndex++, stamp, isStamp: true));
                }
            }

            // If no clear stamps, treat whole group as cluster
            if (stamps.Count == 0 || stamps.All(s => s.Count < minClusterSize))
            {
                clusters.Add(CreateCluster(clusterIndex++, groupPoints, isStamp: false));
            }
        }

        return clusters;
    }

    /// <summary>
    /// Detects "placement stamps" - consecutive UniqueID ranges indicating bulk placement.
    /// </summary>
    private List<List<PlacementPoint>> DetectPlacementStamps(List<PlacementPoint> orderedPoints)
    {
        var stamps = new List<List<PlacementPoint>>();
        var currentStamp = new List<PlacementPoint>();
        
        const int maxGap = 10; // Allow small gaps in UniqueID sequence

        for (int i = 0; i < orderedPoints.Count; i++)
        {
            if (currentStamp.Count == 0)
            {
                currentStamp.Add(orderedPoints[i]);
                continue;
            }

            var lastUid = currentStamp[^1].UniqueId;
            var currentUid = orderedPoints[i].UniqueId;
            
            if (currentUid - lastUid <= maxGap)
            {
                currentStamp.Add(orderedPoints[i]);
            }
            else
            {
                if (currentStamp.Count >= 3) // Minimum stamp size
                    stamps.Add(new List<PlacementPoint>(currentStamp));
                
                currentStamp.Clear();
                currentStamp.Add(orderedPoints[i]);
            }
        }

        if (currentStamp.Count >= 3)
            stamps.Add(currentStamp);

        return stamps;
    }

    /// <summary>
    /// Detects clusters constrained to a single MCNK chunk (16x16 chunk grid per ADT).
    /// </summary>
    public List<SpatialCluster> DetectChunkClusters(
        IEnumerable<PlacementPoint> placements,
        int minClusterSize = 3)
    {
        var allClusters = new List<SpatialCluster>();
        int globalClusterId = 0;

        // Group by chunk coordinates (each ADT is 533.33 units, divided into 16x16 chunks)
        var byChunk = placements.GroupBy(p => GetChunkCoord(p.WorldX, p.WorldY));

        foreach (var chunkGroup in byChunk)
        {
            var chunkPlacements = chunkGroup.ToList();
            if (chunkPlacements.Count < minClusterSize)
                continue;

            // Detect clusters within this chunk
            var chunkClusters = DetectClusters(chunkPlacements, minClusterSize);
            
            foreach (var cluster in chunkClusters)
            {
                // Re-assign cluster IDs to be globally unique
                allClusters.Add(cluster with { ClusterId = globalClusterId++ });
            }
        }

        return allClusters;
    }

    /// <summary>
    /// Identifies recurring cluster patterns across multiple locations.
    /// </summary>
    public List<ClusterPattern> IdentifyPatterns(
        List<SpatialCluster> clusters,
        float similarityThreshold = 0.8f)
    {
        var patterns = new List<ClusterPattern>();
        var usedClusters = new HashSet<int>();

        for (int i = 0; i < clusters.Count; i++)
        {
            if (usedClusters.Contains(i))
                continue;

            var pattern = new ClusterPattern
            {
                PatternId = patterns.Count,
                Instances = new List<SpatialCluster> { clusters[i] }
            };

            usedClusters.Add(i);

            // Find similar clusters
            for (int j = i + 1; j < clusters.Count; j++)
            {
                if (usedClusters.Contains(j))
                    continue;

                if (AreClustersSimil(clusters[i], clusters[j], similarityThreshold))
                {
                    pattern.Instances.Add(clusters[j]);
                    usedClusters.Add(j);
                }
            }

            // Only keep patterns that repeat
            if (pattern.Instances.Count > 1)
            {
                patterns.Add(pattern);
            }
        }

        return patterns;
    }

    private static (int ChunkX, int ChunkY) GetChunkCoord(float worldX, float worldY)
    {
        // WoW coordinate system: each ADT tile is 533.33 units, 16x16 chunks per tile
        const float chunkSize = 533.33333f / 16.0f; // ~33.33 units per chunk
        
        int chunkX = (int)(worldX / chunkSize);
        int chunkY = (int)(worldY / chunkSize);
        
        return (chunkX, chunkY);
    }

    private List<int> GetNeighbors(List<PlacementPoint> points, int index, float threshold)
    {
        var neighbors = new List<int>();
        var point = points[index];
        float thresholdSq = threshold * threshold;

        for (int i = 0; i < points.Count; i++)
        {
            if (i == index)
                continue;

            float distSq = DistanceSquared(point, points[i]);
            if (distSq <= thresholdSq)
            {
                neighbors.Add(i);
            }
        }

        return neighbors;
    }

    private static float DistanceSquared(PlacementPoint a, PlacementPoint b)
    {
        float dx = a.WorldX - b.WorldX;
        float dy = a.WorldY - b.WorldY;
        float dz = a.WorldZ - b.WorldZ;
        return dx * dx + dy * dy + dz * dz;
    }

    private static SpatialCluster CreateCluster(int clusterId, List<PlacementPoint> points, bool isStamp = false)
    {
        // Calculate centroid
        float sumX = 0, sumY = 0, sumZ = 0;
        foreach (var p in points)
        {
            sumX += p.WorldX;
            sumY += p.WorldY;
            sumZ += p.WorldZ;
        }

        int count = points.Count;
        var centroid = (X: sumX / count, Y: sumY / count, Z: sumZ / count);

        // Calculate bounding radius
        float maxDistSq = 0;
        foreach (var p in points)
        {
            float dx = p.WorldX - centroid.X;
            float dy = p.WorldY - centroid.Y;
            float dz = p.WorldZ - centroid.Z;
            float distSq = dx * dx + dy * dy + dz * dz;
            if (distSq > maxDistSq)
                maxDistSq = distSq;
        }

        // Extract UniqueID ranges
        var uniqueIds = points.Select(p => p.UniqueId).OrderBy(id => id).ToList();
        var assetTypes = points.Select(p => p.AssetType).Distinct().ToList();

        // Check if UniqueIDs are consecutive (placement stamp indicator)
        bool hasConsecutiveIds = false;
        if (uniqueIds.Count > 1)
        {
            int consecutiveCount = 0;
            for (int i = 1; i < uniqueIds.Count; i++)
            {
                if (uniqueIds[i] - uniqueIds[i - 1] <= 10) // Allow small gaps
                    consecutiveCount++;
            }
            hasConsecutiveIds = (float)consecutiveCount / (uniqueIds.Count - 1) > 0.8f; // 80% consecutive
        }

        return new SpatialCluster
        {
            ClusterId = clusterId,
            ObjectCount = count,
            CentroidX = centroid.X,
            CentroidY = centroid.Y,
            CentroidZ = centroid.Z,
            BoundingRadius = (float)Math.Sqrt(maxDistSq),
            UniqueIdMin = uniqueIds.First(),
            UniqueIdMax = uniqueIds.Last(),
            UniqueIds = uniqueIds,
            AssetTypes = assetTypes,
            Placements = points,
            IsPlacementStamp = isStamp || hasConsecutiveIds
        };
    }

    private static bool AreClustersSimil(SpatialCluster a, SpatialCluster b, float threshold)
    {
        // Check if clusters are similar based on:
        // 1. Similar object count
        // 2. Similar asset type composition
        // 3. Similar spatial structure

        if (Math.Abs(a.ObjectCount - b.ObjectCount) > 2)
            return false;

        // Check asset type similarity
        var commonTypes = a.AssetTypes.Intersect(b.AssetTypes).Count();
        var totalTypes = a.AssetTypes.Union(b.AssetTypes).Count();
        if ((float)commonTypes / totalTypes < threshold)
            return false;

        // Similar bounding size
        float sizeDiff = Math.Abs(a.BoundingRadius - b.BoundingRadius);
        if (sizeDiff > a.BoundingRadius * 0.3f) // 30% tolerance
            return false;

        return true;
    }
}

/// <summary>
/// Placement point for clustering analysis.
/// </summary>
public record PlacementPoint(
    int UniqueId,
    string AssetType,
    string AssetPath,
    float WorldX,
    float WorldY,
    float WorldZ);

/// <summary>
/// Detected spatial cluster.
/// </summary>
public record SpatialCluster
{
    public required int ClusterId { get; init; }
    public required int ObjectCount { get; init; }
    public required float CentroidX { get; init; }
    public required float CentroidY { get; init; }
    public required float CentroidZ { get; init; }
    public required float BoundingRadius { get; init; }
    public required int UniqueIdMin { get; init; }
    public required int UniqueIdMax { get; init; }
    public required List<int> UniqueIds { get; init; }
    public required List<string> AssetTypes { get; init; }
    public required List<PlacementPoint> Placements { get; init; }
    public bool IsPlacementStamp { get; init; } = false;
}

/// <summary>
/// Recurring cluster pattern (potential prefab/brush).
/// </summary>
public record ClusterPattern
{
    public required int PatternId { get; init; }
    public required List<SpatialCluster> Instances { get; init; }
}
