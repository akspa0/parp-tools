using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace WoWViewer.Terrain.Pm4;

/// <summary>
/// Represents a cohesive, complete 3D physical object composed of one or more
/// constituent PM4 collision hulls on one or more adjacent map tiles (Spec 300, Spec 301).
/// Unifies elevation-fragmented CK24 slices and multi-tile split structures into a single selectable compound object.
/// </summary>
internal sealed class Pm4CompoundObject
{
    public int CompoundId { get; }
    public int TileX { get; }
    public int TileY { get; }
    public IReadOnlySet<(int TileX, int TileY)> SpanningTiles { get; }
    public bool IsCrossTile => SpanningTiles.Count > 1;
    public IReadOnlyList<Pm4OverlayObject> ConstituentHulls { get; }
    public Vector3 WorldBoundsMin { get; }
    public Vector3 WorldBoundsMax { get; }
    public Vector3 WorldCentroid { get; }
    public Vector3 Dimensions { get; }
    public int TotalTriangles { get; }
    public int TotalLines { get; }
    public int TotalSurfaces { get; }
    public byte DominantGroupKey { get; }
    public uint DominantLinkGroupObjectId { get; }
    public float PlacementZ { get; }
    public string DisplayName { get; }

    public Pm4CompoundObject(
        int compoundId,
        int tileX,
        int tileY,
        IReadOnlyList<Pm4OverlayObject> constituentHulls)
        : this(compoundId, new HashSet<(int TileX, int TileY)> { (tileX, tileY) }, constituentHulls)
    {
    }

    public Pm4CompoundObject(
        int compoundId,
        IReadOnlySet<(int TileX, int TileY)> spanningTiles,
        IReadOnlyList<Pm4OverlayObject> constituentHulls)
    {
        CompoundId = compoundId;
        SpanningTiles = spanningTiles ?? new HashSet<(int TileX, int TileY)>();
        var firstTile = SpanningTiles.FirstOrDefault();
        TileX = firstTile.TileX;
        TileY = firstTile.TileY;
        ConstituentHulls = constituentHulls ?? throw new ArgumentNullException(nameof(constituentHulls));

        if (constituentHulls.Count == 0)
        {
            WorldBoundsMin = Vector3.Zero;
            WorldBoundsMax = Vector3.Zero;
            WorldCentroid = Vector3.Zero;
            Dimensions = Vector3.Zero;
            DisplayName = $"[{TileX:D2}_{TileY:D2}] Empty Compound Obj #{compoundId}";
            return;
        }

        Vector3 min = new Vector3(float.MaxValue);
        Vector3 max = new Vector3(float.MinValue);
        Vector3 centroidAccum = Vector3.Zero;
        int tris = 0;
        int lines = 0;
        int surfs = 0;
        var groupKeyCounts = new Dictionary<byte, int>();
        var linkGroupCounts = new Dictionary<uint, int>();
        float minZ = float.MaxValue;

        for (int i = 0; i < constituentHulls.Count; i++)
        {
            Pm4OverlayObject h = constituentHulls[i];
            WorldScene.TransformBounds(h.BoundsMin, h.BoundsMax, h.BaseTransform, out Vector3 hMin, out Vector3 hMax);
            min = Vector3.Min(min, hMin);
            max = Vector3.Max(max, hMax);

            Vector3 worldCenter = Vector3.Transform(h.Center, h.BaseTransform);
            centroidAccum += worldCenter;

            tris += h.Triangles.Count;
            lines += h.Lines.Count;
            surfs += h.SurfaceCount;

            groupKeyCounts[h.DominantGroupKey] = groupKeyCounts.GetValueOrDefault(h.DominantGroupKey) + 1;
            if (h.LinkGroupObjectId != 0)
                linkGroupCounts[h.LinkGroupObjectId] = linkGroupCounts.GetValueOrDefault(h.LinkGroupObjectId) + 1;

            if (hMin.Z < minZ)
                minZ = hMin.Z;
        }

        WorldBoundsMin = min;
        WorldBoundsMax = max;
        WorldCentroid = centroidAccum / constituentHulls.Count;
        Dimensions = Vector3.Max(Vector3.Zero, max - min);
        TotalTriangles = tris;
        TotalLines = lines;
        TotalSurfaces = surfs;

        DominantGroupKey = groupKeyCounts.OrderByDescending(kv => kv.Value).FirstOrDefault().Key;
        DominantLinkGroupObjectId = linkGroupCounts.OrderByDescending(kv => kv.Value).FirstOrDefault().Key;
        PlacementZ = minZ;

        if (IsCrossTile)
        {
            string tilesStr = string.Join(" + ", SpanningTiles.OrderBy(t => t.TileX).ThenBy(t => t.TileY).Select(t => $"{t.TileX:D2}_{t.TileY:D2}"));
            DisplayName = $"[{tilesStr}] Cross-Tile Whole Obj #{CompoundId} ({ConstituentHulls.Count} hulls, {TotalTriangles} tris, {Dimensions.X:F1}x{Dimensions.Y:F1}x{Dimensions.Z:F1} yd)";
        }
        else
        {
            DisplayName = $"[{TileX:D2}_{TileY:D2}] Whole Obj #{CompoundId} ({ConstituentHulls.Count} hulls, {TotalTriangles} tris, {Dimensions.X:F1}x{Dimensions.Y:F1}x{Dimensions.Z:F1} yd)";
        }
    }
}
