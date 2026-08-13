using System.Numerics;

namespace WowViewer.Core.Runtime.World;

/// <summary>
/// Conservative WMO group admission for the camera-inside case. A placement
/// must not portal-cull its own interior when the root bounds are incomplete
/// or differ from the group bounds.
/// </summary>
public static class WmoCameraVisibility
{
    public static bool IsInsideRootOrGroup(
        Vector3 localCameraPosition,
        Vector3 rootMin,
        Vector3 rootMax,
        IReadOnlyList<(Vector3 Min, Vector3 Max)> groupBounds,
        float padding)
    {
        if (ContainsExpanded(localCameraPosition, rootMin, rootMax, padding))
            return true;

        for (int index = 0; index < groupBounds.Count; index++)
        {
            (Vector3 min, Vector3 max) = groupBounds[index];
            if (ContainsExpanded(localCameraPosition, min, max, padding))
                return true;
        }

        return false;
    }

    private static bool ContainsExpanded(Vector3 point, Vector3 min, Vector3 max, float padding)
    {
        float minX = MathF.Min(min.X, max.X) - padding;
        float minY = MathF.Min(min.Y, max.Y) - padding;
        float minZ = MathF.Min(min.Z, max.Z) - padding;
        float maxX = MathF.Max(min.X, max.X) + padding;
        float maxY = MathF.Max(min.Y, max.Y) + padding;
        float maxZ = MathF.Max(min.Z, max.Z) + padding;

        return point.X >= minX && point.X <= maxX
            && point.Y >= minY && point.Y <= maxY
            && point.Z >= minZ && point.Z <= maxZ;
    }
}
