using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;
using CorePm4ConnectorKey = WowViewer.Core.PM4.Models.Pm4ConnectorKey;

namespace WoWViewer.Terrain.Pm4;

/// <summary>
/// Service responsible for clustering and assembling fragmented PM4 collision hulls into
/// cohesive compound whole objects on each map tile (Spec 300) and across adjacent tile seams (Spec 301).
/// </summary>
internal static class Pm4CompoundObjectService
{
    public const float DefaultAdjacencyToleranceYards = Pm4SpatialClustering.DefaultAdjacencyToleranceYards;
    public const float DefaultMscnCoincidenceToleranceYards = Pm4SpatialClustering.DefaultMscnCoincidenceToleranceYards;

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

        // Cluster using core spatial, link group & semantic surface role algorithm
        var clusters = Pm4SpatialClustering.ClusterItems(
            rawHulls,
            boundsSelector: (h, idx) => worldBounds[idx],
            linkGroupSelector: (h, idx) => h.LinkGroupObjectId,
            surfaceClassSelector: (h, idx) => h.DominantGroupKey,
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

    /// <summary>
    /// Stitches per-tile compound objects across adjacent loaded tile seams using MSCN connector anchor points
    /// and boundary spatial proximity (Spec 301).
    /// Returns unified compound objects per tile and an updated reverse lookup mapping each hull to its owning unified compound object.
    /// </summary>
    public static (Dictionary<(int tileX, int tileY), List<Pm4CompoundObject>> TileCompounds, Dictionary<(int tileX, int tileY, uint ck24, int objectPart), Pm4CompoundObject> HullLookup)
        StitchCrossTileCompoundObjects(
            Dictionary<(int tileX, int tileY), List<Pm4CompoundObject>> perTileCompounds,
            Dictionary<(int tileX, int tileY), List<Vector3>> tileMscnPoints,
            float mscnCoincidenceTolerance = DefaultMscnCoincidenceToleranceYards,
            float boundaryAabbTolerance = DefaultAdjacencyToleranceYards)
    {
        var updatedTileCompounds = new Dictionary<(int tileX, int tileY), List<Pm4CompoundObject>>();
        var updatedLookup = new Dictionary<(int tileX, int tileY, uint ck24, int objectPart), Pm4CompoundObject>();

        if (perTileCompounds == null || perTileCompounds.Count == 0)
            return (updatedTileCompounds, updatedLookup);

        // Flatten all initial compounds and remember original tile for each hull
        var allInitial = new List<Pm4CompoundObject>();
        var hullToTile = new Dictionary<Pm4OverlayObject, (int TileX, int TileY)>();

        foreach (var kv in perTileCompounds)
        {
            (int tileX, int tileY) = kv.Key;
            foreach (var comp in kv.Value)
            {
                allInitial.Add(comp);
                for (int i = 0; i < comp.ConstituentHulls.Count; i++)
                {
                    hullToTile[comp.ConstituentHulls[i]] = (tileX, tileY);
                }
            }
        }

        if (allInitial.Count == 0)
            return (updatedTileCompounds, updatedLookup);

        // If only 1 tile exists, no cross-tile stitching is possible
        if (perTileCompounds.Count <= 1)
        {
            foreach (var kv in perTileCompounds)
                updatedTileCompounds[kv.Key] = new List<Pm4CompoundObject>(kv.Value);

            foreach (var comp in allInitial)
            {
                for (int i = 0; i < comp.ConstituentHulls.Count; i++)
                {
                    var h = comp.ConstituentHulls[i];
                    if (hullToTile.TryGetValue(h, out var tile))
                        updatedLookup[(tile.TileX, tile.TileY, h.Ck24, h.ObjectPartId)] = comp;
                }
            }
            return (updatedTileCompounds, updatedLookup);
        }

        // Build cross-tile descriptors with MSCN world points and connector keys
        var descriptors = new List<Pm4CrossTileCompoundDescriptor>(allInitial.Count);
        for (int i = 0; i < allInitial.Count; i++)
        {
            var comp = allInitial[i];
            var mscnPoints = new List<Vector3>();
            var connectorKeys = new HashSet<CorePm4ConnectorKey>();

            if (tileMscnPoints != null && tileMscnPoints.TryGetValue((comp.TileX, comp.TileY), out var tileMscn) && tileMscn.Count > 0)
            {
                for (int hIdx = 0; hIdx < comp.ConstituentHulls.Count; hIdx++)
                {
                    var h = comp.ConstituentHulls[hIdx];
                    if (h.DominantMscnRefIndex < tileMscn.Count)
                    {
                        mscnPoints.Add(tileMscn[(int)h.DominantMscnRefIndex]);
                    }
                    if (h.ConnectorKeys != null)
                    {
                        for (int k = 0; k < h.ConnectorKeys.Count; k++)
                        {
                            var ck = h.ConnectorKeys[k];
                            connectorKeys.Add(new CorePm4ConnectorKey(ck.X, ck.Y, ck.Z));
                        }
                    }
                }
            }

            descriptors.Add(new Pm4CrossTileCompoundDescriptor(
                CompoundId: comp.CompoundId,
                TileX: comp.TileX,
                TileY: comp.TileY,
                BoundsMin: comp.WorldBoundsMin,
                BoundsMax: comp.WorldBoundsMax,
                MscnWorldPoints: mscnPoints,
                ConnectorKeys: connectorKeys));
        }

        // Cluster using core cross-tile MSCN and boundary algorithm
        var clusters = Pm4SpatialClustering.ClusterCrossTileCompounds(descriptors, mscnCoincidenceTolerance, boundaryAabbTolerance);

        // Build unified compound objects
        int nextUnifiedId = 0;
        foreach (var clusterIndices in clusters.OrderByDescending(c => c.Count))
        {
            Pm4CompoundObject unified;
            if (clusterIndices.Count == 1)
            {
                var original = allInitial[clusterIndices[0]];
                unified = new Pm4CompoundObject(nextUnifiedId++, original.SpanningTiles, original.ConstituentHulls);
            }
            else
            {
                var mergedHulls = new List<Pm4OverlayObject>();
                var spanningTiles = new HashSet<(int TileX, int TileY)>();
                foreach (int idx in clusterIndices)
                {
                    var c = allInitial[idx];
                    mergedHulls.AddRange(c.ConstituentHulls);
                    foreach (var t in c.SpanningTiles)
                        spanningTiles.Add(t);
                }
                unified = new Pm4CompoundObject(nextUnifiedId++, spanningTiles, mergedHulls);
            }

            // Register in updatedTileCompounds for each spanning tile
            foreach (var tile in unified.SpanningTiles)
            {
                if (!updatedTileCompounds.TryGetValue(tile, out var list))
                {
                    list = new List<Pm4CompoundObject>();
                    updatedTileCompounds[tile] = list;
                }
                if (!list.Contains(unified))
                    list.Add(unified);
            }

            // Register each hull in reverse lookup
            foreach (var h in unified.ConstituentHulls)
            {
                if (hullToTile.TryGetValue(h, out var tile))
                {
                    updatedLookup[(tile.TileX, tile.TileY, h.Ck24, h.ObjectPartId)] = unified;
                }
            }
        }

        return (updatedTileCompounds, updatedLookup);
    }
}
