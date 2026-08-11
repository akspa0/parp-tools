namespace WowViewer.Core.Runtime.World;

/// <summary>
/// Per-frame evidence for one visual overlay owner. Counts are intentionally
/// primitive-oriented where the owner batches geometry, and instance-oriented
/// for object wireframe work.
/// </summary>
public readonly record struct WorldOverlayOwnerFrameStats(
    string OwnerId,
    double DurationMs,
    bool Enabled,
    int PreparedPrimitiveCount,
    int SubmittedPrimitiveCount,
    string CacheStatus,
    int DeferredCount)
{
    public static WorldOverlayOwnerFrameStats Disabled(string ownerId) =>
        new(ownerId, 0, false, 0, 0, "disabled", 0);
}

/// <summary>
/// Stable owner IDs used by production WorldScene overlay diagnostics.
/// </summary>
public static class WorldOverlayOwners
{
    public const string ObjectWireframe = "object_wireframe";
    public const string SelectionBounds = "selection_bounds";
    public const string Pm4Bounds = "pm4_bounds";
    public const string Pm4GeometryPrepare = "pm4_geometry_prepare";
    public const string Pm4GeometrySubmit = "pm4_geometry_submit";
    public const string Pm4Nodes = "pm4_nodes";
    public const string PoiTaxi = "poi_taxi";
    public const string AreaTriggers = "area_triggers";
    public const string OtherOverlay = "other_overlay";

    public static IReadOnlyList<string> All { get; } =
    [
        ObjectWireframe,
        SelectionBounds,
        Pm4Bounds,
        Pm4GeometryPrepare,
        Pm4GeometrySubmit,
        Pm4Nodes,
        PoiTaxi,
        AreaTriggers,
        OtherOverlay,
    ];
}
