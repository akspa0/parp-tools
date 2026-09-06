using System;
using System.Numerics;
using Silk.NET.OpenGL;
using WoWViewer.Logging;
using WowViewer.Core.Runtime.World;

namespace WoWViewer.Rendering;

/// <summary>
/// Camera-rigged 3D HUD foundation. Owns camera-space attachment, depth-clamped placement, and
/// Tab-synchronized visibility only.
/// Operator correction 2026-09-06: decorative instrumentation (reticle, compass tape, visor bezel)
/// was never requested and has been removed. The rig defaults OFF until the actual assignment
/// lands — ImGui panels composited onto 3D surfaces mounted to the camera frame (Spec 212).
/// </summary>
public sealed class CameraHudRig : IDisposable
{
    private readonly GL _gl;
    private ShaderProgram? _shader;
    private ProceduralMesh? _gimbalMesh;
    private ProceduralMesh? _bracketMesh;

    /// <summary>Defaults off: the operator rejected visible decorative HUD elements in the viewport.</summary>
    public bool Enabled { get; set; } = false;
    public bool ShowGimbal { get; set; } = true;
    public bool ShowBrackets { get; set; } = true;

    /// <summary>Requested local HUD depth; the projection volume clamps it before rendering.</summary>
    public float HudDepth { get; set; } = 1.0f;
    public float GimbalScale { get; set; } = 0.08f;
    public float BracketScale { get; set; } = 0.05f;
    public float NearHudDepth { get; set; } = 0.50f;
    public float FarHudDepth { get; set; } = 2.50f;

    public CameraHudRig(GL gl)
    {
        _gl = gl;
        Initialize();
    }

    private void Initialize()
    {
        try
        {
            const string vertexShader = @"#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;

out vec3 vNormal;
out vec3 vWorldPos;

void main()
{
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    vWorldPos = worldPos.xyz;
    vNormal = mat3(transpose(inverse(uModel))) * aNormal;
    gl_Position = uProj * uView * worldPos;
}";

            const string fragmentShader = @"#version 330 core
in vec3 vNormal;
in vec3 vWorldPos;

uniform vec4 uColor;
uniform vec3 uLightDir;

out vec4 FragColor;

void main()
{
    vec3 n = normalize(vNormal);
    float diff = max(dot(n, normalize(uLightDir)), 0.0);
    float ambient = 0.45;
    vec3 col = uColor.rgb * (diff * 0.55 + ambient);
    FragColor = vec4(col, uColor.a);
}";

            _shader = ShaderProgram.Create(_gl, vertexShader, fragmentShader);

            _gimbalMesh = OpenScadAssetResolver.TryLoadMesh(_gl, "camera_hud_gimbal.off")
                ?? ProceduralMeshLoader.CreateTargetReticle(_gl, 0.5f);

            _bracketMesh = OpenScadAssetResolver.TryLoadMesh(_gl, "hud_frame_bracket.off");
        }
        catch (Exception ex)
        {
            ViewerLog.Important(ViewerLog.Category.General, $"CameraHudRig initialization failed: {ex.Message}");
        }
    }

    /// <summary>
    /// Renders the 3D camera HUD elements transformed in camera-local space.
    /// </summary>
    public void Render(Camera camera, Matrix4x4 proj, float fovDegrees, float aspectRatio)
    {
        if (!Enabled || _shader == null) return;

        CameraHudTransform transform;
        CameraSpaceProjection hudProjection;
        try
        {
            transform = CameraHudTransform.Create(camera.Position, camera.Forward, camera.Up);
            hudProjection = new CameraSpaceProjection(fovDegrees, aspectRatio, NearHudDepth, FarHudDepth);
        }
        catch (ArgumentOutOfRangeException)
        {
            // A malformed runtime camera must not destabilize the world render pass.
            return;
        }

        float depth = hudProjection.ClampDepth(HudDepth);
        Vector2 halfExtents = hudProjection.GetHalfExtents(depth);
        float halfWidth = halfExtents.X;
        float halfHeight = halfExtents.Y;

        _gl.DepthRange(0.0, 0.04);
        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Lequal);
        _gl.Enable(EnableCap.Blend);
        _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);

        _shader.Use();
        _shader.SetMat4("uView", camera.GetViewMatrix());
        _shader.SetMat4("uProj", proj);
        _shader.SetVec3("uLightDir", new Vector3(0.4f, 0.8f, 0.5f));

        // 1. Attitude & heading gimbal, bottom-right of the camera frame.
        if (ShowGimbal && _gimbalMesh != null)
        {
            float yawAngle = MathF.Atan2(transform.Forward.X, transform.Forward.Y);
            Matrix4x4 model = transform.CreateModelMatrix(
                new Vector3(halfWidth * 0.80f, -halfHeight * 0.75f, depth),
                Quaternion.CreateFromAxisAngle(Vector3.UnitZ, -yawAngle),
                GimbalScale * depth);

            _shader.SetMat4("uModel", model);
            _shader.SetVec4("uColor", new Vector4(0.95f, 0.78f, 0.25f, 0.88f)); // Warcraft brass gold
            _gimbalMesh.Draw();
        }

        // 2. Corner framing brackets at the camera frame boundary.
        if (ShowBrackets && _bracketMesh != null)
        {
            float bracketDist = hudProjection.ClampDepth(depth * 1.02f);
            float cornerX = halfWidth * 0.94f;
            float cornerY = halfHeight * 0.92f;
            float bScale = BracketScale * bracketDist;

            (float dx, float dy, float rotDeg)[] corners = {
                (-cornerX,  cornerY, 180.0f), // Top-Left
                ( cornerX,  cornerY, 270.0f), // Top-Right
                (-cornerX, -cornerY,  90.0f), // Bottom-Left
                ( cornerX, -cornerY,   0.0f), // Bottom-Right
            };

            _shader.SetVec4("uColor", new Vector4(0.35f, 0.75f, 0.95f, 0.65f)); // Holographic cyan

            foreach (var (dx, dy, rotDeg) in corners)
            {
                Matrix4x4 model = transform.CreateModelMatrix(
                    new Vector3(dx, dy, bracketDist),
                    Quaternion.CreateFromAxisAngle(Vector3.UnitZ, rotDeg * MathF.PI / 180.0f),
                    bScale);

                _shader.SetMat4("uModel", model);
                _bracketMesh.Draw();
            }
        }

        _gl.DepthRange(0.0, 1.0);
    }

    public void Dispose()
    {
        _shader?.Dispose();
        _gimbalMesh?.Dispose();
        _bracketMesh?.Dispose();
    }
}
