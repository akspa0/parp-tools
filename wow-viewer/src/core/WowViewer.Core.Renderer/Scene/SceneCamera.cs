using System.Numerics;

namespace WowViewer.Core.Renderer.Scene;

public sealed class SceneCamera
{
    private const float TileSize = 533.33333f;
    private const float DefaultFovDegrees = 50f;

    public Vector3 Position { get; set; } = new(50f, 0f, 500f);
    public float Yaw { get; set; } = 180f;
    public float Pitch { get; set; } = -60f;
    public float FieldOfViewDegrees { get; set; } = DefaultFovDegrees;
    public float AspectRatio { get; set; } = 1f;
    public float NearPlane { get; set; } = 1f;
    public float FarPlane { get; set; } = 8000f;

    public Vector3 Forward
    {
        get
        {
            float yawRad = MathF.PI / 180f * Yaw;
            float pitchRad = MathF.PI / 180f * Pitch;
            float cosPitch = MathF.Cos(pitchRad);
            return Vector3.Normalize(new Vector3(
                cosPitch * MathF.Cos(yawRad),
                cosPitch * MathF.Sin(yawRad),
                MathF.Sin(pitchRad)));
        }
    }

    public bool UseExternalMatrix { get; set; }
    public Matrix4x4 ExternalView { get; set; }
    public Matrix4x4 ExternalProjection { get; set; }

    public Matrix4x4 GetViewMatrix()
    {
        if (UseExternalMatrix)
            return ExternalView;
        Vector3 target = Position + Forward;
        return Matrix4x4.CreateLookAt(Position, target, Vector3.UnitZ);
    }

    public Matrix4x4 GetProjectionMatrix()
    {
        if (UseExternalMatrix)
            return ExternalProjection;
        float fovRad = MathF.PI / 180f * FieldOfViewDegrees;
        return Matrix4x4.CreatePerspectiveFieldOfView(fovRad, AspectRatio, NearPlane, FarPlane);
    }

    public void SetViewProjection(Matrix4x4 view, Matrix4x4 projection, Vector3 position)
    {
        ExternalView = view;
        ExternalProjection = projection;
        Position = position;
        UseExternalMatrix = true;
    }

    public void LookAtPosition(Vector3 target, float distance, float angleDegrees, float pitchDegrees)
    {
        float yawRad = MathF.PI / 180f * angleDegrees;
        float pitchRad = MathF.PI / 180f * pitchDegrees;
        float cosPitch = MathF.Cos(pitchRad);

        Position = new Vector3(
            target.X - distance * cosPitch * MathF.Cos(yawRad),
            target.Y - distance * cosPitch * MathF.Sin(yawRad),
            target.Z + distance * MathF.Sin(pitchRad));

        Yaw = angleDegrees;
        Pitch = pitchDegrees;
    }

    public void LookAtTile(int tileX, int tileY, float distance = 800f)
    {
        float centerX = (tileX - 32f) * TileSize + TileSize / 2f;
        float centerY = (tileY - 32f) * TileSize + TileSize / 2f;

        LookAtPosition(new Vector3(-centerY, -centerX, 0f), distance, 180f, -60f);
    }
}
