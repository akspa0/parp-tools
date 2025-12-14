using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using WoWRollback.PM4Module;

namespace WoWRollback.PM4Module
{
    /// <summary>
    /// Clusters raw PM4 geometry (surfaces/triangles) into coherent object candidates.
    /// Used for extracting M2 and WMO candidates from "GroupKey=0" buckets.
    /// </summary>
    public static class Pm4GeometryClusterer
    {
        public class Cluster
        {
            public List<Vector3> Vertices { get; } = new();
            public List<uint> Indices { get; } = new();
            public Vector3 Centroid { get; set; }
            public Vector3 BoundsMin { get; set; }
            public Vector3 BoundsMax { get; set; }

            public int TriangleCount => Indices.Count / 3;
        }

        /// <summary>
        /// Clusters surfaces into connected components based on shared vertex indices.
        /// </summary>
        /// <param name="surfaces">List of surfaces to cluster.</param>
        /// <param name="meshIndices">Global MSVI indices from PM4.</param>
        /// <param name="meshVertices">Global MSVT vertices from PM4.</param>
        /// <returns>List of clusters (independent objects).</returns>
        public static List<Cluster> ClusterSurfaces(
            List<MsurEntry> surfaces,
            List<uint> meshIndices,
            List<Vector3> meshVertices)
        {
            var clusters = new List<Cluster>();
            if (surfaces.Count == 0) return clusters;

            // 1. Flatten all surfaces into a list of Triangles with Global Vertex Indices
            // We use global vertex indices for connectivity.
            var triangles = new List<(uint V1, uint V2, uint V3, int SurfaceIdx)>();
            
            // Map: VertexIndex -> List of TriangleIndices (in 'triangles' list)
            var vertToTris = new Dictionary<uint, List<int>>();

            for (int sIdx = 0; sIdx < surfaces.Count; sIdx++)
            {
                var surf = surfaces[sIdx];
                int idxStart = (int)surf.MsviFirstIndex;
                int idxCount = surf.IndexCount;

                if (idxStart < 0 || idxStart + idxCount > meshIndices.Count) continue;

                for (int i = 0; i < idxCount; i += 3)
                {
                    if (i + 2 >= idxCount) break;

                    uint i1 = meshIndices[idxStart + i];
                    uint i2 = meshIndices[idxStart + i + 1];
                    uint i3 = meshIndices[idxStart + i + 2];

                    int triIdx = triangles.Count;
                    triangles.Add((i1, i2, i3, sIdx));

                    AddToMap(vertToTris, i1, triIdx);
                    AddToMap(vertToTris, i2, triIdx);
                    AddToMap(vertToTris, i3, triIdx);
                }
            }

            // 2. BFS to find Connected Components
            var visitedTriangles = new HashSet<int>();
            
            for (int t = 0; t < triangles.Count; t++)
            {
                if (visitedTriangles.Contains(t)) continue;

                // Start new cluster
                var clusterTris = new List<int>();
                var queue = new Queue<int>();
                queue.Enqueue(t);
                visitedTriangles.Add(t);

                while (queue.Count > 0)
                {
                    int current = queue.Dequeue();
                    clusterTris.Add(current);
                    var (v1, v2, v3, _) = triangles[current];

                    // Check neighbors via all 3 vertices
                    CheckNeighbors(v1, vertToTris, visitedTriangles, queue);
                    CheckNeighbors(v2, vertToTris, visitedTriangles, queue);
                    CheckNeighbors(v3, vertToTris, visitedTriangles, queue);
                }

                // 3. Build Cluster Geometry
                if (clusterTris.Count > 0)
                {
                    var cluster = BuildCluster(clusterTris, triangles, meshVertices);
                    // Filter noise (e.g. single triangles?)
                    if (cluster.TriangleCount >= 4) // Min 4 tris for a 3D object heuristic?
                    {
                        clusters.Add(cluster);
                    }
                }
            }

            return clusters;
        }

        private static void AddToMap(Dictionary<uint, List<int>> map, uint v, int t)
        {
            if (!map.TryGetValue(v, out var list))
            {
                list = new List<int>();
                map[v] = list;
            }
            list.Add(t);
        }

        private static void CheckNeighbors(uint v, Dictionary<uint, List<int>> map, HashSet<int> visited, Queue<int> queue)
        {
            if (map.TryGetValue(v, out var neighbors))
            {
                foreach (var n in neighbors)
                {
                    if (visited.Add(n))
                    {
                        queue.Enqueue(n);
                    }
                }
            }
        }

        private static Cluster BuildCluster(List<int> triIndices, List<(uint V1, uint V2, uint V3, int SurfaceIdx)> allTriangles, List<Vector3> meshVertices)
        {
            var cluster = new Cluster();
            var uniqueVerts = new Dictionary<uint, int>(); // GlobalIdx -> LocalIdx

            foreach (var tIdx in triIndices)
            {
                var (g1, g2, g3, _) = allTriangles[tIdx];

                cluster.Indices.Add(GetOrAddVert(g1, uniqueVerts, meshVertices, cluster.Vertices));
                cluster.Indices.Add(GetOrAddVert(g2, uniqueVerts, meshVertices, cluster.Vertices));
                cluster.Indices.Add(GetOrAddVert(g3, uniqueVerts, meshVertices, cluster.Vertices));
            }

            // Compute Stats
            var boundsMin = new Vector3(float.MaxValue);
            var boundsMax = new Vector3(float.MinValue);
            var centroidSum = Vector3.Zero;

            foreach (var v in cluster.Vertices)
            {
                boundsMin = Vector3.Min(boundsMin, v);
                boundsMax = Vector3.Max(boundsMax, v);
                centroidSum += v;
            }

            cluster.BoundsMin = boundsMin;
            cluster.BoundsMax = boundsMax;
            cluster.Centroid = cluster.Vertices.Count > 0 ? centroidSum / cluster.Vertices.Count : Vector3.Zero;

            return cluster;
        }

        private static uint GetOrAddVert(uint globalIdx, Dictionary<uint, int> map, List<Vector3> sourceVerts, List<Vector3> targetVerts)
        {
            if (!map.TryGetValue(globalIdx, out int localIdx))
            {
                localIdx = targetVerts.Count;
                if (globalIdx < sourceVerts.Count)
                    targetVerts.Add(sourceVerts[(int)globalIdx]);
                else
                    targetVerts.Add(Vector3.Zero); // Error fallback

                map[globalIdx] = localIdx;
            }
            return (uint)localIdx;
        }
    }
}
