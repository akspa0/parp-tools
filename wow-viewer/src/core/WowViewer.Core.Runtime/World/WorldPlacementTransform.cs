using System.Numerics;

namespace WowViewer.Core.Runtime.World;

/// <summary>
/// Builds the renderer-space transform for legacy MDDF and MODF placements.
/// The ADT placement reader has already normalized file coordinates before this
/// method is called, so both placement families use the same axis-swapped
/// rotation convention.
/// </summary>
public static class WorldPlacementTransform
{
    public static Matrix4x4 Build(Vector3 position, Vector3 rotationDegrees, float scale = 1f)
    {
        float rotationX = -rotationDegrees.Y * (MathF.PI / 180f);
        float rotationY = -rotationDegrees.X * (MathF.PI / 180f);
        float rotationZ = rotationDegrees.Z * (MathF.PI / 180f);

        // Placement meshes are uploaded with reversed winding. The half-turn
        // is part of the renderer-space contract, not a WMO-specific fix.
        return Matrix4x4.CreateRotationZ(MathF.PI)
            * Matrix4x4.CreateScale(scale)
            * Matrix4x4.CreateRotationX(rotationX)
            * Matrix4x4.CreateRotationY(rotationY)
            * Matrix4x4.CreateRotationZ(rotationZ)
            * Matrix4x4.CreateTranslation(position);
    }
}
