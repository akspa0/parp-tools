using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using WowViewer.Core.PM4.Models;

namespace WowViewer.Core.PM4.Services;

/// <summary>
/// Spatial clustering algorithms and Union-Find disjoint sets for assembling fragmented
/// PM4 collision surfaces and components into cohesive whole physical structures (Spec 300, Spec 301).
/// </summary>
public static class Pm4SpatialClustering
{
    public const float DefaultAdjacencyToleranceYards = 2.5f;
    public const float DefaultMscnCoincidenceToleranceYards = 0.5f;

    /// <summary>
    /// Checks if two axis-aligned bounding boxes overlap or lie within the specified tolerance distance.
    /// </summary>
    public static bool OverlapsWithTolerance(
        Vector3 aMin,
        Vector3 aMax,
        Vector3 bMin,
        Vector3 bMax,
        float tolerance = DefaultAdjacencyToleranceYards)
    {
        return aMin.X - tolerance <= bMax.X && aMax.X + tolerance >= bMin.X
            && aMin.Y - tolerance <= bMax.Y && aMax.Y + tolerance >= bMin.Y
            && aMin.Z - tolerance <= bMax.Z && aMax.Z + tolerance >= bMin.Z;
    }

    /// <summary>
    /// Clusters items using spatial AABB proximity tolerance, shared non-zero MSLK LinkGroupObjectId,
    /// and semantic surface role rules (Spec 300, Spec 301, Spec 302).
    /// Prevents transitive runaway clustering where doodads, docks, and unrelated terrain collapse into giant blobs.
    /// </summary>
    public static List<List<int>> ClusterItems<T>(
        IReadOnlyList<T> items,
        Func<T, int, (Vector3 Min, Vector3 Max)> boundsSelector,
        Func<T, int, uint> linkGroupSelector,
        Func<T, int, byte>? surfaceClassSelector = null,
        float tolerance = DefaultAdjacencyToleranceYards)
    {
        if (items == null || items.Count == 0)
            return new List<List<int>>();

        int count = items.Count;
        var dsu = new DisjointSet(count);

        var bounds = new (Vector3 Min, Vector3 Max)[count];
        var linkGroups = new uint[count];
        byte[]? surfaceClasses = surfaceClassSelector != null ? new byte[count] : null;

        for (int i = 0; i < count; i++)
        {
            bounds[i] = boundsSelector(items[i], i);
            linkGroups[i] = linkGroupSelector(items[i], i);
            if (surfaceClasses != null)
                surfaceClasses[i] = surfaceClassSelector!(items[i], i);
        }

        for (int i = 0; i < count; i++)
        {
            var (minA, maxA) = bounds[i];
            uint linkA = linkGroups[i];
            byte clsA = surfaceClasses != null ? surfaceClasses[i] : (byte)0;

            for (int j = i + 1; j < count; j++)
            {
                var (minB, maxB) = bounds[j];
                uint linkB = linkGroups[j];
                byte clsB = surfaceClasses != null ? surfaceClasses[j] : (byte)0;

                bool shouldCluster = false;

                // 1. Shared non-zero LinkGroupObjectId always clusters
                if (linkA != 0 && linkA == linkB)
                {
                    shouldCluster = true;
                }
                else if (surfaceClasses != null)
                {
                    // Doodads (0x03) represent separate props (M2 models: cannons, crates, barrels, etc.).
                    // A doodad must NEVER cluster with a WMO floor/structure (0x10..0x16),
                    // and separate doodads must not cluster with each other unless they share LinkGroupObjectId!
                    bool isDoodadA = clsA == 0x03;
                    bool isDoodadB = clsB == 0x03;

                    if (isDoodadA || isDoodadB)
                    {
                        // Doodads only cluster if they share non-zero LinkGroupId (handled in rule 1 above)
                        // or if their bounding boxes are virtually identical/nested (subparts of same physical prop)
                        if (isDoodadA && isDoodadB)
                        {
                            float centerDistSq = Vector3.DistanceSquared((minA + maxA) * 0.5f, (minB + maxB) * 0.5f);
                            if (centerDistSq <= 0.25f * 0.25f)
                            {
                                shouldCluster = true;
                            }
                        }
                    }
                    else
                    {
                        // Both are WMO surfaces / structures.
                        // Slices of the SAME building represent vertically stacked floors with substantial (X, Y) footprint overlap.
                        // Compute 2D horizontal overlap in the (X, Y) plane:
                        float overlapX = Math.Min(maxA.X, maxB.X) - Math.Max(minA.X, minB.X);
                        float overlapY = Math.Min(maxA.Y, maxB.Y) - Math.Max(minA.Y, minB.Y);

                        if (overlapX > 0.5f && overlapY > 0.5f)
                        {
                            // Substantial 2D footprint overlap. Check vertical elevation gap:
                            float gapZ = Math.Max(0f, Math.Max(minA.Z, minB.Z) - Math.Min(maxA.Z, maxB.Z));
                            if (gapZ <= tolerance)
                            {
                                shouldCluster = true;
                            }
                        }
                    }
                }
                else
                {
                    // Legacy fallback when surfaceClassSelector is not supplied (e.g. geometric AABB tests)
                    if (OverlapsWithTolerance(minA, maxA, minB, maxB, tolerance))
                    {
                        shouldCluster = true;
                    }
                }

                if (shouldCluster)
                {
                    dsu.Union(i, j);
                }
            }
        }

        var groups = new Dictionary<int, List<int>>();
        for (int i = 0; i < count; i++)
        {
            int root = dsu.Find(i);
            if (!groups.TryGetValue(root, out var list))
            {
                list = new List<int>();
                groups[root] = list;
            }
            list.Add(i);
        }

        return new List<List<int>>(groups.Values);
    }

    /// <summary>
    /// Clusters compound objects across adjacent map tiles using MSCN connector anchor point coincidence
    /// and boundary AABB proximity tolerance (Spec 301).
    /// Returns clusters of indices into the input <paramref name="compounds"/> list.
    /// </summary>
    public static List<List<int>> ClusterCrossTileCompounds(
        IReadOnlyList<Pm4CrossTileCompoundDescriptor> compounds,
        float mscnCoincidenceTolerance = DefaultMscnCoincidenceToleranceYards,
        float boundaryAabbTolerance = DefaultAdjacencyToleranceYards)
    {
        if (compounds == null || compounds.Count == 0)
            return new List<List<int>>();

        int count = compounds.Count;
        var dsu = new DisjointSet(count);

        float mscnDistSqThreshold = mscnCoincidenceTolerance * mscnCoincidenceTolerance;

        for (int i = 0; i < count; i++)
        {
            var a = compounds[i];

            for (int j = i + 1; j < count; j++)
            {
                var b = compounds[j];

                // Single-tile objects are already clustered internally; skip intra-tile comparisons
                if (a.TileX == b.TileX && a.TileY == b.TileY)
                    continue;

                // Only consider adjacent tiles (Chebyshev distance <= 1)
                if (Math.Abs(a.TileX - b.TileX) > 1 || Math.Abs(a.TileY - b.TileY) > 1)
                    continue;

                // Bounding boxes must overlap or be adjacent within boundary tolerance
                if (!OverlapsWithTolerance(a.BoundsMin, a.BoundsMax, b.BoundsMin, b.BoundsMax, boundaryAabbTolerance))
                    continue;

                bool shouldCluster = false;

                // 1. Shared quantized connector keys
                if (a.ConnectorKeys != null && b.ConnectorKeys != null && a.ConnectorKeys.Count > 0 && b.ConnectorKeys.Count > 0)
                {
                    foreach (var key in a.ConnectorKeys)
                    {
                        if (b.ConnectorKeys.Contains(key))
                        {
                            shouldCluster = true;
                            break;
                        }
                    }
                }

                // 2. Coincident MSCN connector points (distance <= mscnCoincidenceTolerance)
                if (!shouldCluster && a.MscnWorldPoints.Count > 0 && b.MscnWorldPoints.Count > 0)
                {
                    for (int ai = 0; ai < a.MscnWorldPoints.Count; ai++)
                    {
                        var ptA = a.MscnWorldPoints[ai];
                        for (int bi = 0; bi < b.MscnWorldPoints.Count; bi++)
                        {
                            var ptB = b.MscnWorldPoints[bi];
                            if (Vector3.DistanceSquared(ptA, ptB) <= mscnDistSqThreshold)
                            {
                                shouldCluster = true;
                                break;
                            }
                        }
                        if (shouldCluster)
                            break;
                    }
                }

                if (shouldCluster)
                {
                    dsu.Union(i, j);
                }
            }
        }

        var groups = new Dictionary<int, List<int>>();
        for (int i = 0; i < count; i++)
        {
            int root = dsu.Find(i);
            if (!groups.TryGetValue(root, out var list))
            {
                list = new List<int>();
                groups[root] = list;
            }
            list.Add(i);
        }

        return new List<List<int>>(groups.Values);
    }

    /// <summary>
    /// Computes aggregate bounding volume, centroid, and dimensions from a collection of world bounds.
    /// </summary>
    public static void AggregateBounds(
        IReadOnlyList<(Vector3 Min, Vector3 Max)> bounds,
        out Vector3 totalMin,
        out Vector3 totalMax,
        out Vector3 centroid,
        out Vector3 dimensions)
    {
        if (bounds == null || bounds.Count == 0)
        {
            totalMin = Vector3.Zero;
            totalMax = Vector3.Zero;
            centroid = Vector3.Zero;
            dimensions = Vector3.Zero;
            return;
        }

        Vector3 min = new Vector3(float.MaxValue);
        Vector3 max = new Vector3(float.MinValue);
        Vector3 centerAccum = Vector3.Zero;

        for (int i = 0; i < bounds.Count; i++)
        {
            var b = bounds[i];
            min = Vector3.Min(min, b.Min);
            max = Vector3.Max(max, b.Max);
            centerAccum += (b.Min + b.Max) * 0.5f;
        }

        totalMin = min;
        totalMax = max;
        centroid = centerAccum / bounds.Count;
        dimensions = Vector3.Max(Vector3.Zero, max - min);
    }

    /// <summary>
    /// Disjoint Set Union (DSU) data structure with path compression and union by rank.
    /// </summary>
    public sealed class DisjointSet
    {
        private readonly int[] _parent;
        private readonly int[] _rank;

        public DisjointSet(int size)
        {
            _parent = new int[size];
            _rank = new int[size];
            for (int i = 0; i < size; i++)
                _parent[i] = i;
        }

        public int Find(int i)
        {
            if (_parent[i] != i)
                _parent[i] = Find(_parent[i]);
            return _parent[i];
        }

        public void Union(int x, int y)
        {
            int rootX = Find(x);
            int rootY = Find(y);
            if (rootX == rootY) return;

            if (_rank[rootX] < _rank[rootY])
            {
                _parent[rootX] = rootY;
            }
            else if (_rank[rootX] > _rank[rootY])
            {
                _parent[rootY] = rootX;
            }
            else
            {
                _parent[rootY] = rootX;
                _rank[rootX]++;
            }
        }
    }
}
