using System.Numerics;
using WoWRollback.PM4Module.Pipeline;

namespace WoWRollback.PM4Module.Decoding;

/// <summary>
/// Builds matchable objects from MSCN (SceneNodes) data only.
/// MSCN contains collision wall vertices that better represent object shapes
/// than the floor-only CK24/MSUR geometry.
/// </summary>
public static class MscnObjectBuilder
{
    /// <summary>
    /// Represents an MSCN-based object candidate for matching.
    /// </summary>
    public record MscnObject(
        int ObjectIndex,
        int TileX,
        int TileY,
        List<Vector3> Vertices,
        Vector3 BoundsMin,
        Vector3 BoundsMax,
        Vector3 Centroid,
        Vector3 Dimensions
    );

    /// <summary>
    /// Extract MSCN objects from PM4 file.
    /// Groups MSCN points by their spatial clustering to identify separate objects.
    /// </summary>
    public static List<MscnObject> BuildMscnObjects(Pm4FileStructure pm4, int tileX, int tileY)
    {
        var objects = new List<MscnObject>();
        
        if (pm4.SceneNodes == null || pm4.SceneNodes.Count == 0)
            return objects;

        // Group MSCN points by spatial proximity
        // Use a simple clustering approach: find connected components within distance threshold
        var clusters = ClusterMscnPoints(pm4.SceneNodes, clusterDistance: 50f);
        
        int objectIndex = 0;
        foreach (var cluster in clusters)
        {
            if (cluster.Count < 4) // Skip tiny clusters
                continue;

            var (min, max) = CalculateBounds(cluster);
            var dimensions = max - min;
            
            // Skip clusters that are too small (noise) or too large (terrain)
            if (dimensions.X < 1f || dimensions.Y < 1f) continue;
            if (dimensions.X > 533f || dimensions.Y > 533f) continue;
            
            var centroid = (min + max) / 2f;
            
            objects.Add(new MscnObject(
                ObjectIndex: objectIndex++,
                TileX: tileX,
                TileY: tileY,
                Vertices: cluster,
                BoundsMin: min,
                BoundsMax: max,
                Centroid: centroid,
                Dimensions: dimensions
            ));
        }
        
        return objects;
    }
    
    /// <summary>
    /// Cluster MSCN points by spatial proximity using Union-Find.
    /// </summary>
    private static List<List<Vector3>> ClusterMscnPoints(List<Vector3> points, float clusterDistance)
    {
        int n = points.Count;
        var parent = new int[n];
        for (int i = 0; i < n; i++) parent[i] = i;
        
        int Find(int x)
        {
            if (parent[x] != x) parent[x] = Find(parent[x]);
            return parent[x];
        }
        
        void Union(int a, int b)
        {
            int pa = Find(a), pb = Find(b);
            if (pa != pb) parent[pa] = pb;
        }
        
        // Union nearby points (O(n^2) but acceptable for typical MSCN sizes)
        float distSq = clusterDistance * clusterDistance;
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                if (Vector3.DistanceSquared(points[i], points[j]) < distSq)
                {
                    Union(i, j);
                }
            }
        }
        
        // Group by cluster
        var clusters = new Dictionary<int, List<Vector3>>();
        for (int i = 0; i < n; i++)
        {
            int root = Find(i);
            if (!clusters.ContainsKey(root))
                clusters[root] = new List<Vector3>();
            clusters[root].Add(points[i]);
        }
        
        return clusters.Values.ToList();
    }
    
    private static (Vector3 min, Vector3 max) CalculateBounds(List<Vector3> vertices)
    {
        var min = new Vector3(float.MaxValue);
        var max = new Vector3(float.MinValue);
        
        foreach (var v in vertices)
        {
            min = Vector3.Min(min, v);
            max = Vector3.Max(max, v);
        }
        
        return (min, max);
    }
}
