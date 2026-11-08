using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using WowViewer.Core.PM4.Services;

namespace WoWViewer.Terrain.Pm4;

/// <summary>
/// Service responsible for clustering and assembling fragmented PM4 collision hulls into
/// cohesive compound whole objects on each map tile (Spec 300).
/// </summary>
internal static class Pm4CompoundObjectService
{
    public const float DefaultAdjacencyToleranceYards = Pm4SpatialClustering.DefaultAdjacencyToleranceYards;

    /// <summary>
    /// Assembles raw per-CK24 PM4 hulls on a tile into compound whole objects.
    /// Returns the list of compound objects and a reverse lookup from hull key to owning compound object.
    /// </summary>
    public static (List<Pm4CompoundObject> CompoundObjects, Dictionary<(int tileX, int tileY, uint ck24, int objectPart), Pm4CompoundObject> HullLookup)
        AssembleTileCompoundObjects(
            int tileX,
            int tileY,
            IReadOnlyList<Pm4OverlayObject> rawHulls,
            float adjacencyTolerance = DefaultAdjacencyToleranceYards)
    {
        var compoundObjects = new List<Pm4CompoundObject>();
        var lookup = new Dictionary<(int tileX, int tileY, uint ck24, int objectPart), Pm4CompoundObject>();

        if (rawHulls == null || rawHulls.Count == 0)
            return (compoundObjects, lookup);

        // Precompute world-space bounds for all hulls
        var worldBounds = new (Vector3 Min, Vector3 Max)[rawHulls.Count];
        for (int i = 0; i < rawHulls.Count; i++)
        {
            var h = rawHulls[i];
            WorldScene.TransformBounds(h.BoundsMin, h.BoundsMax, h.BaseTransform, out Vector3 hMin, out Vector3 hMax);
            worldBounds[i] = (hMin, hMax);
        }

        // Cluster using core spatial & link group algorithm
        var clusters = Pm4SpatialClustering.ClusterItems(
            rawHulls,
            boundsSelector: (h, idx) => worldBounds[idx],
            linkGroupSelector: (h, idx) => h.LinkGroupObjectId,
            tolerance: adjacencyTolerance);

        // Create compound objects
        int nextCompoundId = 0;
        foreach (var clusterIndices in clusters.OrderByDescending(c => c.Count))
        {
            var clusterHulls = new List<Pm4OverlayObject>(clusterIndices.Count);
            for (int i = 0; i < clusterIndices.Count; i++)
            {
                clusterHulls.Add(rawHulls[clusterIndices[i]]);
            }

            var compound = new Pm4CompoundObject(nextCompoundId++, tileX, tileY, clusterHulls);
            compoundObjects.Add(compound);

            for (int i = 0; i < clusterHulls.Count; i++)
            {
                var h = clusterHulls[i];
                lookup[(tileX, tileY, h.Ck24, h.ObjectPartId)] = compound;
            }
        }

        return (compoundObjects, lookup);
    }
}
