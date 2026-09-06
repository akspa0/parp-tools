using System.Numerics;

namespace WowViewer.Core.Runtime.World;

/// <summary>
/// Immutable orthonormal camera frame used by camera-anchored HUD renderers.
/// Local HUD coordinates are measured in world units: +X right, +Y up, and
/// +Z forward from the camera into the scene.
/// </summary>
public readonly record struct CameraHudTransform(
    Vector3 Position,
    Vector3 Right,
    Vector3 Up,
    Vector3 Forward)
{
    /// <summary>
    /// Constructs an orthonormal camera frame from a position, forward vector,
    /// and camera-up hint. The returned frame is safe to use for HUD placement.
    /// </summary>
    public static CameraHudTransform Create(Vector3 position, Vector3 forward, Vector3 upHint)
    {
        ValidateFinite(position, nameof(position));

        Vector3 normalizedForward = NormalizeRequired(forward, nameof(forward));
        Vector3 normalizedUpHint = NormalizeRequired(upHint, nameof(upHint));
        Vector3 right = Vector3.Cross(normalizedForward, normalizedUpHint);
        if (right.LengthSquared() <= 1e-10f)
            throw new ArgumentException("Camera forward and up vectors cannot be parallel.", nameof(upHint));

        right = Vector3.Normalize(right);
        Vector3 up = Vector3.Normalize(Vector3.Cross(right, normalizedForward));
        return new CameraHudTransform(position, right, up, normalizedForward);
    }

    /// <summary>
    /// Converts a local HUD coordinate into a world-space position.
    /// </summary>
    public Vector3 ToWorldPoint(Vector3 localPoint)
    {
        ValidateFinite(localPoint, nameof(localPoint));
        return Position
            + (Right * localPoint.X)
            + (Up * localPoint.Y)
            + (Forward * localPoint.Z);
    }

    /// <summary>
    /// Builds a row-vector System.Numerics model matrix that attaches a local
    /// HUD mesh to this camera frame. Callers compose scale and local rotation
    /// before this transform, matching the viewer's existing render convention.
    /// </summary>
    public Matrix4x4 CreateModelMatrix(Vector3 localPosition, Quaternion localRotation, float uniformScale = 1f)
    {
        ValidateFinite(localPosition, nameof(localPosition));
        if (!float.IsFinite(uniformScale) || uniformScale <= 0f)
            throw new ArgumentOutOfRangeException(nameof(uniformScale));
        if (!IsFinite(localRotation) || localRotation.LengthSquared() <= 1e-10f)
            throw new ArgumentException("Local rotation must be finite and non-zero.", nameof(localRotation));

        // System.Numerics transforms row vectors. The local coordinate basis
        // occupies matrix rows: local X -> Right, local Y -> Up, local Z ->
        // Forward. The local translation must be composed before this basis so
        // it is also camera-relative instead of world-axis-relative.
        Matrix4x4 cameraBasis = new(
            Right.X, Right.Y, Right.Z, 0f,
            Up.X, Up.Y, Up.Z, 0f,
            Forward.X, Forward.Y, Forward.Z, 0f,
            Position.X, Position.Y, Position.Z, 1f);

        return Matrix4x4.CreateScale(uniformScale)
            * Matrix4x4.CreateFromQuaternion(Quaternion.Normalize(localRotation))
            * Matrix4x4.CreateTranslation(localPosition.X, localPosition.Y, localPosition.Z)
            * cameraBasis;
    }

    private static Vector3 NormalizeRequired(Vector3 value, string parameterName)
    {
        ValidateFinite(value, parameterName);
        if (value.LengthSquared() <= 1e-10f)
            throw new ArgumentOutOfRangeException(parameterName);

        return Vector3.Normalize(value);
    }

    private static void ValidateFinite(Vector3 value, string parameterName)
    {
        if (!float.IsFinite(value.X) || !float.IsFinite(value.Y) || !float.IsFinite(value.Z))
            throw new ArgumentOutOfRangeException(parameterName);
    }

    private static bool IsFinite(Quaternion value) =>
        float.IsFinite(value.X)
        && float.IsFinite(value.Y)
        && float.IsFinite(value.Z)
        && float.IsFinite(value.W);
}

/// <summary>
/// Perspective-plane dimensions for a HUD rendered in camera-local space.
/// Depth is clamped before an element is placed so it cannot cross the near
/// plane or drift beyond the rig's configured HUD volume.
/// </summary>
public readonly struct CameraSpaceProjection
{
    public CameraSpaceProjection(
        float verticalFovDegrees,
        float aspectRatio,
        float nearDepth = 0.50f,
        float farDepth = 2.50f)
    {
        if (!float.IsFinite(verticalFovDegrees) || verticalFovDegrees <= 1f || verticalFovDegrees >= 179f)
            throw new ArgumentOutOfRangeException(nameof(verticalFovDegrees));
        if (!float.IsFinite(aspectRatio) || aspectRatio <= 0f)
            throw new ArgumentOutOfRangeException(nameof(aspectRatio));
        if (!float.IsFinite(nearDepth) || !float.IsFinite(farDepth) || nearDepth <= 0f || farDepth < nearDepth)
            throw new ArgumentOutOfRangeException(nameof(nearDepth));

        VerticalFovDegrees = verticalFovDegrees;
        AspectRatio = aspectRatio;
        NearDepth = nearDepth;
        FarDepth = farDepth;
    }

    public float VerticalFovDegrees { get; }
    public float AspectRatio { get; }
    public float NearDepth { get; }
    public float FarDepth { get; }

    public float ClampDepth(float depth)
    {
        if (!float.IsFinite(depth))
            return NearDepth;

        return Math.Clamp(depth, NearDepth, FarDepth);
    }

    /// <summary>Returns half-width and half-height of the perspective plane at a clamped depth.</summary>
    public Vector2 GetHalfExtents(float depth)
    {
        float clampedDepth = ClampDepth(depth);
        float halfHeight = clampedDepth * MathF.Tan(VerticalFovDegrees * (MathF.PI / 360f));
        return new Vector2(halfHeight * AspectRatio, halfHeight);
    }

    /// <summary>
    /// Converts normalized viewport coordinates in [-1, 1] into a camera-local
    /// point on the HUD plane. Coordinates outside that range are clamped to
    /// the plane edge, which keeps a rig child in its visible attachment volume.
    /// </summary>
    public Vector3 GetLocalPoint(Vector2 normalizedViewportPosition, float depth)
    {
        Vector2 extent = GetHalfExtents(depth);
        float x = Math.Clamp(normalizedViewportPosition.X, -1f, 1f) * extent.X;
        float y = Math.Clamp(normalizedViewportPosition.Y, -1f, 1f) * extent.Y;
        return new Vector3(x, y, ClampDepth(depth));
    }
}
