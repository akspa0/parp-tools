using System.Numerics;

namespace WowViewer.App;

internal sealed class WorldViewCamera
{
    private static readonly Vector3 IdentityPosition = new(0f, 0f, 1f);
    private const float IdentityYawDegrees = 180.0f;
    private const float IdentityPitchDegrees = -10.0f;

    public Vector3 Position { get; set; } = IdentityPosition;

    public float YawDegrees { get; set; } = IdentityYawDegrees;

    public float Yaw
    {
        get => YawDegrees;
        set => YawDegrees = value;
    }

    public float PitchDegrees { get; set; } = IdentityPitchDegrees;

    public float Pitch
    {
        get => PitchDegrees;
        set => PitchDegrees = value;
    }

    public Vector3 Forward
    {
        get
        {
            float yawRadians = MathF.PI / 180f * YawDegrees;
            float pitchRadians = MathF.PI / 180f * PitchDegrees;
            float cosPitch = MathF.Cos(pitchRadians);
            return Vector3.Normalize(new Vector3(
                cosPitch * MathF.Cos(yawRadians),
                cosPitch * MathF.Sin(yawRadians),
                MathF.Sin(pitchRadians)));
        }
    }

    public Vector3 Target => Position + Forward;

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
        return Forward;
    }

    public Matrix4x4 GetViewMatrix()
    {
        return Matrix4x4.CreateLookAt(Position, Target, Vector3.UnitZ);
    }

    public Matrix4x4 GetViewMatrix(Vector3 up)
    {
        return GetViewMatrix();
    }

    public void RotateLook(float yawDeltaDegrees, float pitchDeltaDegrees)
    {
        YawDegrees -= yawDeltaDegrees;
        PitchDegrees = Math.Clamp(PitchDegrees + pitchDeltaDegrees, -89.0f, 89.0f);
    }

    public void Move(float forward, float right, float up, float speed)
    {
        float yawRadians = MathF.PI / 180f * YawDegrees;
        float cosYaw = MathF.Cos(yawRadians);
        float sinYaw = MathF.Sin(yawRadians);

        Vector3 forwardVector = new(cosYaw, sinYaw, 0f);
        Vector3 rightVector = new(sinYaw, -cosYaw, 0f);

        Position += (forwardVector * forward + rightVector * right + (Vector3.UnitZ * up)) * speed;
    }

    public void Translate(float forwardDistance, float strafeDistance, float verticalDistance)
    {
        Move(forwardDistance, strafeDistance, verticalDistance, 1.0f);
    }

    private static void GetCameraAngles(Vector3 forward, out float yawDegrees, out float pitchDegrees)
    {
        forward = Vector3.Normalize(forward);
        yawDegrees = MathF.Atan2(forward.Y, forward.X) * 180.0f / MathF.PI;
        pitchDegrees = MathF.Asin(forward.Z) * 180.0f / MathF.PI;
    }
}