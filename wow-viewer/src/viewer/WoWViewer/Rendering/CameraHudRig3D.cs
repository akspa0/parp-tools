using System;
using System.Numerics;
using Silk.NET.OpenGL;
using WoWViewer.Logging;

namespace WoWViewer.Rendering;

/// <summary>
/// Camera-rigged 3D in-scene HUD system.
/// Mounts authored 3D OpenSCAD objects (compass gimbals, attitude rings, viewport brackets)
/// directly into the camera entity's local coordinate hierarchy, rendering them as true in-world
/// 3D objects with depth, lighting, and spatial orientation.
/// </summary>
public sealed class CameraHudRig3D : IDisposable
{
    private readonly GL _gl;
    private ShaderProgram? _shader;
    private ProceduralMesh? _gimbalMesh;
    private ProceduralMesh? _bracketMesh;

    public bool Enabled { get; set; } = true;
    public bool ShowGimbal { get; set; } = true;
    public bool ShowBrackets { get; set; } = true;
    public float HudDepth { get; set; } = 1.0f; // Distance in front of camera
    public float GimbalScale { get; set; } = 0.08f;
    public float BracketScale { get; set; } = 0.05f;

    public CameraHudRig3D(GL gl)
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

            // Load authored OpenSCAD assets
            _gimbalMesh = OpenScadAssetResolver.TryLoadMesh(_gl, "camera_hud_gimbal.off")
                ?? ProceduralMeshLoader.CreateTargetReticle(_gl, 0.5f);

            _bracketMesh = OpenScadAssetResolver.TryLoadMesh(_gl, "hud_frame_bracket.off");
        }
        catch (Exception ex)
        {
            ViewerLog.Important(ViewerLog.Category.General, $"CameraHudRig3D initialization failed: {ex.Message}");
        }
    }

    /// <summary>
    /// Renders the 3D camera HUD elements transformed in camera-local space.
    /// </summary>
    public void Render(Camera camera, Matrix4x4 proj, float fovDegrees, float aspectRatio)
    {
        if (!Enabled || _shader == null) return;

        float fovRad = fovDegrees * (MathF.PI / 180.0f);
        float halfHeight = HudDepth * MathF.Tan(fovRad * 0.5f);
        float halfWidth = halfHeight * aspectRatio;

        Vector3 camPos = camera.Position;
        Vector3 camFwd = Vector3.Normalize(camera.Forward);
        Vector3 camUp = Vector3.Normalize(camera.Up);
        Vector3 camRight = camera.Right;

        // Enforce always-visible HUD depth range
        _gl.DepthRange(0.0, 0.04);
        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Lequal);
        _gl.Enable(EnableCap.Blend);
        _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);

        _shader.Use();
        _shader.SetMat4("uView", camera.GetViewMatrix());
        _shader.SetMat4("uProj", proj);
        _shader.SetVec3("uLightDir", new Vector3(0.4f, 0.8f, 0.5f));

        // 1. Render 3D Attitude & Heading Gimbal (mounted bottom-right of viewport)
        if (ShowGimbal && _gimbalMesh != null)
        {
            // Position in camera local frame
            Vector3 gimbalPos = camPos
                + camRight * (halfWidth * 0.80f)
                - camUp * (halfHeight * 0.75f)
                + camFwd * HudDepth;

            // Compute attitude orientation:
            // The gimbal body faces the camera, but rotates its inner compass towards world North (+Y)
            float yawAngle = MathF.Atan2(camFwd.X, camFwd.Y);
            Matrix4x4 gimbalRot = Matrix4x4.CreateRotationZ(-yawAngle)
                * Matrix4x4.CreateBillboard(gimbalPos, camPos, camUp, camFwd);

            Matrix4x4 model = Matrix4x4.CreateScale(GimbalScale * HudDepth)
                * gimbalRot
                * Matrix4x4.CreateTranslation(gimbalPos);

            _shader.SetMat4("uModel", model);
            _shader.SetVec4("uColor", new Vector4(0.95f, 0.78f, 0.25f, 0.88f)); // Warcraft brass gold

            _gimbalMesh.Draw();
        }

        // 2. Render 3D Corner Framing Brackets
        if (ShowBrackets && _bracketMesh != null)
        {
            float bracketDist = HudDepth * 1.02f;
            float cornerX = halfWidth * 0.94f;
            float cornerY = halfHeight * 0.92f;
            float bScale = BracketScale * HudDepth;

            // (dx, dy, rotZ in degrees)
            (float dx, float dy, float angleDeg)[] corners = {
                (-cornerX,  cornerY, 180.0f), // Top-Left
                ( cornerX,  cornerY, 270.0f), // Top-Right
                (-cornerX, -cornerY,  90.0f), // Bottom-Left
                ( cornerX, -cornerY,   0.0f), // Bottom-Right
            };

            _shader.SetVec4("uColor", new Vector4(0.35f, 0.75f, 0.95f, 0.65f)); // Holographic cyan glow

            foreach (var (dx, dy, rotDeg) in corners)
            {
                Vector3 cornerPos = camPos + camRight * dx + camUp * dy + camFwd * bracketDist;
                Matrix4x4 localRot = Matrix4x4.CreateRotationZ(rotDeg * MathF.PI / 180.0f);
                Matrix4x4 billboard = Matrix4x4.CreateBillboard(cornerPos, camPos, camUp, camFwd);
                Matrix4x4 model = Matrix4x4.CreateScale(bScale) * localRot * billboard * Matrix4x4.CreateTranslation(cornerPos);

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
