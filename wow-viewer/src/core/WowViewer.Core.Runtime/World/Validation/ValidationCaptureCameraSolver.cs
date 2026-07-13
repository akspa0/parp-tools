using System.Numerics;

namespace WowViewer.Core.Runtime.World.Validation;

public static class ValidationCaptureCameraSolver
{
    public static Vector2 ComputeTileCenter(
        int tileX,
        int tileY,
        float mapOrigin,
        float tileWorldSize)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(tileX);
        ArgumentOutOfRangeException.ThrowIfNegative(tileY);
        if (float.IsNaN(mapOrigin) || float.IsInfinity(mapOrigin))
            throw new ArgumentOutOfRangeException(nameof(mapOrigin), mapOrigin, "Map origin must be finite.");
        if (float.IsNaN(tileWorldSize) || float.IsInfinity(tileWorldSize) || tileWorldSize <= 0f)
            throw new ArgumentOutOfRangeException(nameof(tileWorldSize), tileWorldSize, "Tile world size must be greater than zero.");

        return new Vector2(
            mapOrigin - ((tileY + 0.5f) * tileWorldSize),
            mapOrigin - ((tileX + 0.5f) * tileWorldSize));
    }

    public static ValidationCaptureCameraFrame SolveTopDown(
        ValidationCaptureCameraInput input)
    {
        if (float.IsNaN(input.AspectRatio) || float.IsInfinity(input.AspectRatio) || input.AspectRatio <= 0f)
            throw new ArgumentOutOfRangeException(nameof(input), input.AspectRatio, "Aspect ratio must be greater than zero.");
        if (float.IsNaN(input.GroundHeight) || float.IsInfinity(input.GroundHeight))
            throw new ArgumentOutOfRangeException(nameof(input), input.GroundHeight, "Ground height must be finite.");
        if (float.IsNaN(input.DesiredSpan) || float.IsInfinity(input.DesiredSpan) || input.DesiredSpan <= 0f)
            throw new ArgumentOutOfRangeException(nameof(input), input.DesiredSpan, "Desired span must be greater than zero.");
        if (float.IsNaN(input.EyeHeightOffset) || float.IsInfinity(input.EyeHeightOffset) || input.EyeHeightOffset <= 0f)
            throw new ArgumentOutOfRangeException(nameof(input), input.EyeHeightOffset, "Eye height offset must be greater than zero.");
        if (float.IsNaN(input.NearPlane) || float.IsInfinity(input.NearPlane) || input.NearPlane <= 0f)
            throw new ArgumentOutOfRangeException(nameof(input), input.NearPlane, "Near plane must be greater than zero.");
        if (float.IsNaN(input.FarPlane) || float.IsInfinity(input.FarPlane) || input.FarPlane <= input.NearPlane)
            throw new ArgumentOutOfRangeException(nameof(input), input.FarPlane, "Far plane must be greater than the near plane.");

        Vector2 center = ComputeTileCenter(input.TileX, input.TileY, input.MapOrigin, input.TileWorldSize);
        Vector3 up = NormalizeUp(input.Up);

        float worldSpanX;
        float worldSpanY;
        if (input.AspectRatio >= 1f)
        {
            worldSpanX = input.DesiredSpan * input.AspectRatio;
            worldSpanY = input.DesiredSpan;
        }
        else
        {
            worldSpanX = input.DesiredSpan;
            worldSpanY = input.DesiredSpan / input.AspectRatio;
        }

        Vector3 eye = new(center.X, center.Y, input.GroundHeight + input.EyeHeightOffset);
        Vector3 target = new(center.X, center.Y, input.GroundHeight);
        Matrix4x4 view = Matrix4x4.CreateLookAt(eye, target, up);
        Matrix4x4 projection = Matrix4x4.CreateOrthographic(worldSpanX, worldSpanY, input.NearPlane, input.FarPlane);
        return new ValidationCaptureCameraFrame(eye, target, up, worldSpanX, worldSpanY, view, projection);
    }

    private static Vector3 NormalizeUp(Vector3 up)
    {
        if (up.LengthSquared() <= 1e-6f)
            return Vector3.UnitX;

        return Vector3.Normalize(up);
    }
}
