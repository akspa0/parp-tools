using System.Numerics;

namespace WowViewer.Core.Runtime.World.Validation;

/// <summary>
/// Frozen renderer-side contract for terrain-shade guidance captures.
/// These are explicit viewer constants, not decoded client light-table values.
/// </summary>
public static class ValidationTerrainShadeContract
{
    public const string Revision = "terrain-shade-v1-fixed-viewer-light";

    public static Vector3 LightDirection { get; } = new(-0.45f, -0.55f, 0.70f);

    public static Vector3 LightColor { get; } = new(0.80f, 0.82f, 0.78f);

    public static Vector3 AmbientColor { get; } = new(0.28f, 0.30f, 0.34f);
}
