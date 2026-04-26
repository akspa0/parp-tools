using System.Numerics;

namespace WowViewer.App;

internal sealed class WowViewerWorldSpatialSnapshot
{
    public static readonly WowViewerWorldSpatialSnapshot Empty = new(Vector2.Zero, Vector2.Zero, hasPlanarBounds: false);

    public WowViewerWorldSpatialSnapshot(Vector2 planarMin, Vector2 planarMax, bool hasPlanarBounds = true)
    {
        PlanarMin = planarMin;
        PlanarMax = planarMax;
        HasPlanarBounds = hasPlanarBounds;
    }

    public Vector2 PlanarMin { get; }

    public Vector2 PlanarMax { get; }

    public bool HasPlanarBounds { get; }

    public static WowViewerWorldSpatialSnapshot FromRuntimeFrame(WowViewerWorldRuntimeFrameResult runtimeFrame)
    {
        ArgumentNullException.ThrowIfNull(runtimeFrame);
        return new WowViewerWorldSpatialSnapshot(runtimeFrame.PlanarMin, runtimeFrame.PlanarMax);
    }
}