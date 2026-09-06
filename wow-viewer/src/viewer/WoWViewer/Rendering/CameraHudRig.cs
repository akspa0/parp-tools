using System;
using System.Numerics;
using Silk.NET.OpenGL;
using WoWViewer.Logging;
using WowViewer.Core.Runtime.World;

namespace WoWViewer.Rendering;

/// <summary>
/// Camera-rigged HUD foundation. It owns only camera-space attachment, fog-independent rendering,
/// and Tab-synchronized visibility; interactive panel surfaces and input routing are separate
/// phases so decorative meshes cannot be mistaken for a usable spatial workbench.
/// </summary>
public sealed class CameraHudRig : IDisposable
{
    private readonly GL _gl;
    private ShaderProgram? _shader;
    private ProceduralMesh? _gimbalMesh;
    private ProceduralMesh? _bracketMesh;
    private ProceduralMesh? _reticleMesh;
    private ProceduralMesh? _compassMesh;
    private ProceduralMesh? _bezelMesh;

    public bool Enabled { get; set; } = true;
    public bool ShowGimbal { get; set; } = true;
    public bool ShowBrackets { get; set; } = true;
    public bool ShowReticle { get; set; } = true;
    public bool ShowCompass { get; set; } = true;
    public bool ShowBezel { get; set; } = true;
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

            // Load authored OpenSCAD assets
            _gimbalMesh = OpenScadAssetResolver.TryLoadMesh(_gl, "camera_hud_gimbal.off")
                ?? ProceduralMeshLoader.CreateTargetReticle(_gl, 0.5f);

            _bracketMesh = OpenScadAssetResolver.TryLoadMesh(_gl, "hud_frame_bracket.off");
            _reticleMesh = OpenScadAssetResolver.TryLoadMesh(_gl, "camera_hud_reticle_tactical.off");
            _compassMesh = OpenScadAssetResolver.TryLoadMesh(_gl, "camera_hud_compass_tape.off");
            _bezelMesh = OpenScadAssetResolver.TryLoadMesh(_gl, "camera_hud_curved_bezel.off");
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

        // 0. Tactical reticle remains camera-locked at the focal center. The
        // pitch value produces a restrained pitch-ladder tilt without changing
        // the camera's visual aim.
        if (ShowReticle && _reticleMesh != null)
        {
            float pitchRadians = camera.Pitch * (MathF.PI / 180f);
            Matrix4x4 model = transform.CreateModelMatrix(
                new Vector3(0f, 0f, depth),
                Quaternion.CreateFromAxisAngle(Vector3.UnitX, -pitchRadians * 0.20f),
                0.12f * depth);
            _shader.SetMat4("uModel", model);
            _shader.SetVec4("uColor", new Vector4(0.40f, 0.92f, 1.00f, 0.78f));
            _reticleMesh.Draw();
        }

        // 0.5. The compass is top-center and rotates against world north as
        // the camera yaw changes. It is decorative until spatial panel input
        // ships in Phase 3.
        if (ShowCompass && _compassMesh != null)
        {
            float yawRadians = camera.Yaw * (MathF.PI / 180f);
            Matrix4x4 model = transform.CreateModelMatrix(
                new Vector3(0f, halfHeight * 0.86f, depth),
                Quaternion.CreateFromAxisAngle(Vector3.UnitZ, -yawRadians),
                0.075f * depth);
            _shader.SetMat4("uModel", model);
            _shader.SetVec4("uColor", new Vector4(0.95f, 0.78f, 0.25f, 0.78f));
            _compassMesh.Draw();
        }

        // 0.75. The authored bezel provides the optical-frame treatment. Its
        // mesh size is authored independently, so scale it from the camera
        // frustum rather than a fixed world unit.
        if (ShowBezel && _bezelMesh != null)
        {
            float bezelScale = MathF.Max(halfWidth, halfHeight) * 1.85f;
            Matrix4x4 model = transform.CreateModelMatrix(
                new Vector3(0f, 0f, hudProjection.ClampDepth(depth * 1.01f)),
                Quaternion.Identity,
                bezelScale);
            _shader.SetMat4("uModel", model);
            _shader.SetVec4("uColor", new Vector4(0.24f, 0.68f, 0.95f, 0.28f));
            _bezelMesh.Draw();
        }

        // 1. Render 3D Attitude & Heading Gimbal (mounted bottom-right of viewport)
        if (ShowGimbal && _gimbalMesh != null)
        {
            // The gimbal is mounted in camera space, with its compass body yawed toward world North.
            float yawAngle = MathF.Atan2(transform.Forward.X, transform.Forward.Y);
            Matrix4x4 model = transform.CreateModelMatrix(
                new Vector3(halfWidth * 0.80f, -halfHeight * 0.75f, depth),
                Quaternion.CreateFromAxisAngle(Vector3.UnitZ, -yawAngle),
                GimbalScale * depth);

            _shader.SetMat4("uModel", model);
            _shader.SetVec4("uColor", new Vector4(0.95f, 0.78f, 0.25f, 0.88f)); // Warcraft brass gold

            _gimbalMesh.Draw();
        }

        // 2. Render 3D Corner Framing Brackets
        if (ShowBrackets && _bracketMesh != null)
        {
            float bracketDist = hudProjection.ClampDepth(depth * 1.02f);
            float cornerX = halfWidth * 0.94f;
            float cornerY = halfHeight * 0.92f;
            float bScale = BracketScale * bracketDist;

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
        _reticleMesh?.Dispose();
        _compassMesh?.Dispose();
        _bezelMesh?.Dispose();
    }
}
