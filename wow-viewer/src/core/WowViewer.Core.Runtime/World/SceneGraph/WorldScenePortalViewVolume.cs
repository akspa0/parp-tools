using System.Numerics;

namespace WowViewer.Core.Runtime.World.SceneGraph;

public readonly record struct WorldScenePortalPlane(Vector3 Normal, float D)
{
    public float SignedDistance(Vector3 point)
        => Vector3.Dot(Normal, point) + D;
}

public sealed class WorldScenePortalViewVolume
{
    private readonly WorldScenePortalPlane[] _planes;

    private WorldScenePortalViewVolume(
        string? sourceNodeId,
        string? destinationNodeId,
        int? portalIndex,
        int depth,
        IReadOnlyList<WorldScenePortalPlane> planes)
    {
        SourceNodeId = sourceNodeId;
        DestinationNodeId = destinationNodeId;
        PortalIndex = portalIndex;
        Depth = depth;
        _planes = planes.ToArray();
    }

    public string? SourceNodeId { get; }

    public string? DestinationNodeId { get; }

    public int? PortalIndex { get; }

    public int Depth { get; }

    public IReadOnlyList<WorldScenePortalPlane> Planes => _planes;

    public static WorldScenePortalViewVolume CreateRoot(
        IEnumerable<WorldScenePortalPlane>? planes = null)
    {
        WorldScenePortalPlane[] validatedPlanes = (planes ?? []).ToArray();
        if (validatedPlanes.Any(static plane => !IsValidPlane(plane)))
            throw new ArgumentException("Root portal volume planes must be finite and normalized.", nameof(planes));

        return new WorldScenePortalViewVolume(null, null, null, 0, validatedPlanes);
    }

    public bool ContainsPoint(Vector3 point)
    {
        if (!IsFinite(point))
            return false;

        return _planes.All(plane => plane.SignedDistance(point) >= -0.0001f);
    }

    public bool IntersectsBounds(Vector3 min, Vector3 max)
    {
        if (!IsFinite(min) || !IsFinite(max) || min.X > max.X || min.Y > max.Y || min.Z > max.Z)
            return false;

        Span<Vector3> corners = stackalloc Vector3[8];
        corners[0] = new Vector3(min.X, min.Y, min.Z);
        corners[1] = new Vector3(max.X, min.Y, min.Z);
        corners[2] = new Vector3(min.X, max.Y, min.Z);
        corners[3] = new Vector3(max.X, max.Y, min.Z);
        corners[4] = new Vector3(min.X, min.Y, max.Z);
        corners[5] = new Vector3(max.X, min.Y, max.Z);
        corners[6] = new Vector3(min.X, max.Y, max.Z);
        corners[7] = new Vector3(max.X, max.Y, max.Z);

        foreach (WorldScenePortalPlane plane in _planes)
        {
            bool anyInside = false;
            foreach (Vector3 corner in corners)
            {
                if (plane.SignedDistance(corner) >= -0.0001f)
                {
                    anyInside = true;
                    break;
                }
            }

            if (!anyInside)
                return false;
        }

        return true;
    }

    internal static WorldScenePortalViewVolume CreateChild(
        WorldScenePortalViewVolume parent,
        string sourceNodeId,
        string destinationNodeId,
        int portalIndex,
        IReadOnlyList<WorldScenePortalPlane> childPlanes)
        => new(sourceNodeId, destinationNodeId, portalIndex, parent.Depth + 1, childPlanes);

    private static bool IsValidPlane(WorldScenePortalPlane plane)
        => IsFinite(plane.Normal)
            && float.IsFinite(plane.D)
            && MathF.Abs(plane.Normal.Length() - 1f) <= 0.001f;

    private static bool IsFinite(Vector3 value)
        => float.IsFinite(value.X) && float.IsFinite(value.Y) && float.IsFinite(value.Z);
}

public sealed record WorldScenePortalViewVolumeBuildResult(
    WorldScenePortalViewVolume? Volume,
    bool FallbackRequired,
    string? FallbackReason);

/// <summary>
/// Builds a bounded child volume from existing portal geometry. It intentionally exposes the
/// plane contract only; renderer frustum types and WMO runtime state remain separate owners.
/// </summary>
public static class WorldScenePortalViewVolumeBuilder
{
    public static WorldScenePortalViewVolumeBuildResult BuildChild(
        WorldScenePortalViewVolume parent,
        WorldScenePortalGeometry portal,
        string sourceNodeId,
        string destinationNodeId,
        Vector3 cameraPosition,
        int maximumDepth)
    {
        ArgumentNullException.ThrowIfNull(parent);
        ArgumentNullException.ThrowIfNull(portal);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourceNodeId);
        ArgumentException.ThrowIfNullOrWhiteSpace(destinationNodeId);
        if (maximumDepth < 0)
            throw new ArgumentOutOfRangeException(nameof(maximumDepth));

        if (parent.Depth >= maximumDepth)
            return Fallback(parent, "maximum_depth_reached");
        if (!IsFinite(cameraPosition))
            return Fallback(parent, "camera_position_invalid");
        int destinationGroupIndex = FindGroupIndex(portal, destinationNodeId);
        if (!portal.GroupIndices.Contains(destinationGroupIndex))
            return Fallback(parent, "portal_group_missing");
        if (!TryGetGroupSide(portal, destinationGroupIndex, out short destinationSide)
            || destinationSide == 0)
            return Fallback(parent, "portal_side_unknown");
        if (!TryNormalize(portal.Normal, out Vector3 portalNormal)
            || !float.IsFinite(portal.PlaneDistance)
            || portal.Vertices.Count < 3
            || portal.Vertices.Any(static vertex => !IsFinite(vertex)))
            return Fallback(parent, "portal_geometry_invalid");

        Vector3 portalCenter = Vector3.Zero;
        foreach (Vector3 vertex in portal.Vertices)
            portalCenter += vertex;
        portalCenter /= portal.Vertices.Count;

        float portalSignedCameraDistance = Vector3.Dot(portalNormal, cameraPosition)
            + portal.PlaneDistance;
        if (MathF.Abs(portalSignedCameraDistance) <= 0.0001f)
            return Fallback(parent, "camera_on_portal_plane");

        List<WorldScenePortalPlane> childPlanes = [.. parent.Planes];
        float sideSign = destinationSide < 0 ? -1f : 1f;
        childPlanes.Add(new WorldScenePortalPlane(
            portalNormal * sideSign,
            portal.PlaneDistance * sideSign));

        for (int index = 0; index < portal.Vertices.Count; index++)
        {
            Vector3 first = portal.Vertices[index];
            Vector3 second = portal.Vertices[(index + 1) % portal.Vertices.Count];
            Vector3 edge = second - first;
            Vector3 edgeNormal = Vector3.Cross(edge, cameraPosition - first);
            if (!TryNormalize(edgeNormal, out edgeNormal))
                return Fallback(parent, "portal_edge_degenerate");

            float edgeD = -Vector3.Dot(edgeNormal, cameraPosition);
            if (Vector3.Dot(edgeNormal, portalCenter) + edgeD < 0f)
            {
                edgeNormal = -edgeNormal;
                edgeD = -edgeD;
            }

            childPlanes.Add(new WorldScenePortalPlane(edgeNormal, edgeD));
        }

        return new WorldScenePortalViewVolumeBuildResult(
            WorldScenePortalViewVolume.CreateChild(
                parent,
                sourceNodeId,
                destinationNodeId,
                portal.PortalIndex,
                childPlanes),
            false,
            null);
    }

    private static int FindGroupIndex(WorldScenePortalGeometry portal, string nodeId)
    {
        const string groupMarker = "/group/";
        int markerIndex = nodeId.LastIndexOf(groupMarker, StringComparison.Ordinal);
        if (markerIndex < 0 || !int.TryParse(nodeId[(markerIndex + groupMarker.Length)..], out int groupIndex))
            return int.MinValue;
        return groupIndex;
    }

    private static bool TryGetGroupSide(
        WorldScenePortalGeometry portal,
        int groupIndex,
        out short side)
    {
        foreach (WorldScenePortalGroupSide groupSide in portal.GroupSides)
        {
            if (groupSide.GroupIndex == groupIndex)
            {
                side = groupSide.Side;
                return true;
            }
        }

        side = 0;
        return false;
    }

    private static WorldScenePortalViewVolumeBuildResult Fallback(
        WorldScenePortalViewVolume parent,
        string reason)
        => new(parent, true, reason);

    private static bool TryNormalize(Vector3 value, out Vector3 normalized)
    {
        float length = value.Length();
        if (!float.IsFinite(length) || length <= 0.000001f)
        {
            normalized = default;
            return false;
        }

        normalized = value / length;
        return IsFinite(normalized);
    }

    private static bool IsFinite(Vector3 value)
        => float.IsFinite(value.X) && float.IsFinite(value.Y) && float.IsFinite(value.Z);
}
