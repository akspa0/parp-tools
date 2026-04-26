using System.Numerics;

namespace WowViewer.App;

internal sealed class WorldViewCamera
{
    private static readonly Vector3 IdentityPosition = new(0f, 0f, 1f);
    private const float IdentityYawDegrees = 180.0f;
    private const float IdentityPitchDegrees = -10.0f;

    public Vector3 Position { get; private set; } = IdentityPosition;

    public float YawDegrees { get; private set; } = IdentityYawDegrees;

    public float PitchDegrees { get; private set; } = IdentityPitchDegrees;

    public Vector3 Target => Position + GetForwardVector();

    private Vector3 DefaultPosition { get; set; } = IdentityPosition;

    private float DefaultYawDegrees { get; set; } = IdentityYawDegrees;

    private float DefaultPitchDegrees { get; set; } = IdentityPitchDegrees;

    public void ResetToIdentity()
    {
        Position = IdentityPosition;
        YawDegrees = IdentityYawDegrees;
        PitchDegrees = IdentityPitchDegrees;
        DefaultPosition = IdentityPosition;
        DefaultYawDegrees = IdentityYawDegrees;
        DefaultPitchDegrees = IdentityPitchDegrees;
    }

    public void SetPose(Vector3 position, Vector3 target, bool saveAsDefault)
    {
        Position = position;
        Vector3 forward = target - position;
        if (forward.LengthSquared() > 1e-6f)
        {
            forward = Vector3.Normalize(forward);
            GetCameraAngles(forward, out float yawDegrees, out float pitchDegrees);
            YawDegrees = yawDegrees;
            PitchDegrees = Math.Clamp(pitchDegrees, -89.0f, 89.0f);
        }

        if (saveAsDefault)
        {
            DefaultPosition = position;
            DefaultYawDegrees = YawDegrees;
            DefaultPitchDegrees = PitchDegrees;
        }
    }

    public void Reset()
    {
        Position = DefaultPosition;
        YawDegrees = DefaultYawDegrees;
        PitchDegrees = DefaultPitchDegrees;
    }

    public Vector3 GetForwardVector()
    {
        float yawRadians = YawDegrees * MathF.PI / 180.0f;
        float pitchRadians = PitchDegrees * MathF.PI / 180.0f;
        float cosPitch = MathF.Cos(pitchRadians);
        return Vector3.Normalize(new Vector3(
            cosPitch * MathF.Cos(yawRadians),
            cosPitch * MathF.Sin(yawRadians),
            MathF.Sin(pitchRadians)));
    }

    public Matrix4x4 GetViewMatrix(Vector3 up)
    {
        return Matrix4x4.CreateLookAt(Position, Target, up);
    }

    public void RotateLook(float yawDeltaDegrees, float pitchDeltaDegrees)
    {
        YawDegrees -= yawDeltaDegrees;
        PitchDegrees = Math.Clamp(PitchDegrees + pitchDeltaDegrees, -89.0f, 89.0f);
    }

    public void Translate(float forwardDistance, float strafeDistance, float verticalDistance)
    {
        float yawRadians = YawDegrees * MathF.PI / 180.0f;
        float cosYaw = MathF.Cos(yawRadians);
        float sinYaw = MathF.Sin(yawRadians);
        Vector3 forward = new(cosYaw, sinYaw, 0.0f);
        Vector3 right = new(sinYaw, -cosYaw, 0.0f);
        Position += forward * forwardDistance;
        Position += right * strafeDistance;
        Position += Vector3.UnitZ * verticalDistance;
    }

    private static void GetCameraAngles(Vector3 forward, out float yawDegrees, out float pitchDegrees)
    {
        forward = Vector3.Normalize(forward);
        yawDegrees = MathF.Atan2(forward.Y, forward.X) * 180.0f / MathF.PI;
        pitchDegrees = MathF.Asin(forward.Z) * 180.0f / MathF.PI;
    }
}