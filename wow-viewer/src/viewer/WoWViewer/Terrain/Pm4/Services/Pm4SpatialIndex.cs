using System.Collections.Generic;
using System.Numerics;
using WowViewer.Core.PM4.Services;

namespace WoWViewer.Terrain;

/// <summary>
/// Spatial indexing and bounding hierarchy for PM4 overlay scenes (Spec 288).
/// Conforms to AGENTS.md §10 god-class decomposition.
/// </summary>
internal sealed class Pm4SpatialIndex
{
    public static readonly Pm4SpatialIndex Instance = new();

    public float RayAABBIntersect(Vector3 origin, Vector3 dir, Vector3 bmin, Vector3 bmax)
    {
        return Pm4SpatialMath.RayAABBIntersect(origin, dir, bmin, bmax);
    }

    public void ComputeTileBounds(int tileX, int tileY, float minZ, float maxZ, out Vector3 tileMin, out Vector3 tileMax)
    {
        Pm4SpatialMath.ComputeTileWorldBounds(tileX, tileY, minZ, maxZ, out tileMin, out tileMax);
    }

    public IEnumerable<(int tileX, int tileY)> Get3x3Neighborhood(int centerTileX, int centerTileY)
    {
        return Pm4SpatialMath.Get3x3Neighborhood(centerTileX, centerTileY);
    }
}
