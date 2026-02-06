using System.Numerics;

namespace MdxViewer.Rendering;

/// <summary>
/// Free-fly camera for model viewing.
/// Camera is a point in 3D space. WASD moves the camera, mouse controls view direction.
/// The model stays centered at origin. Z-up coordinate system.
/// </summary>
public class Camera
{
    // Camera position in world space
    public Vector3 Position { get; set; } = new Vector3(50f, 0f, 20f);

    // Yaw (horizontal rotation) and Pitch (vertical rotation) in degrees
    public float Yaw { get; set; } = 180f; // Face toward origin initially
    public float Pitch { get; set; } = -10f; // Slight downward angle

    public Matrix4x4 GetViewMatrix()
    {
        float yawRad = MathF.PI / 180f * Yaw;
        float pitchRad = MathF.PI / 180f * Pitch;

        // Forward direction (where camera is looking)
        float cosPitch = MathF.Cos(pitchRad);
        float sinPitch = MathF.Sin(pitchRad);
        float cosYaw = MathF.Cos(yawRad);
        float sinYaw = MathF.Sin(yawRad);

        var forward = new Vector3(
            cosPitch * cosYaw,
            cosPitch * sinYaw,
            sinPitch
        );

        // LookAt target is camera position + forward direction
        var target = Position + forward;

        // Up vector is Z-up
        return Matrix4x4.CreateLookAt(Position, target, Vector3.UnitZ);
    }

    /// <summary>Move camera in its local space (WASD).</summary>
    public void Move(float forward, float right, float up, float speed)
    {
        float yawRad = MathF.PI / 180f * Yaw;
        float pitchRad = MathF.PI / 180f * Pitch;

        float cosPitch = MathF.Cos(pitchRad);
        float sinPitch = MathF.Sin(pitchRad);
        float cosYaw = MathF.Cos(yawRad);
        float sinYaw = MathF.Sin(yawRad);

        // Forward vector (in XY plane, ignoring pitch for WASD movement so you don't fly into ground)
        var forwardVec = new Vector3(cosYaw, sinYaw, 0);
        // Right vector (perpendicular in XY plane)
        var rightVec = new Vector3(sinYaw, -cosYaw, 0);
        // Up vector is world Z
        var upVec = Vector3.UnitZ;

        Position += (forwardVec * forward + rightVec * right + upVec * up) * speed;
    }
}
