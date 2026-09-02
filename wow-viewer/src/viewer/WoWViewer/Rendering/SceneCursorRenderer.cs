using System;
using System.IO;
using System.Numerics;
using Silk.NET.OpenGL;
using WoWViewer.DataSources;
using WoWViewer.Logging;
using WowViewer.Core.Mdx;
using WowViewer.Core.IO.Mdx;

namespace WoWViewer.Rendering;

public enum CursorStyle
{
    AuthenticWoWGauntlet,
    ProceduralPointer,
    TargetReticle,
    ClassicOSArrow
}

public enum SceneCursorState
{
    Pointer,
    Interact,
    Speak,
    Attack,
    CastGlow
}

/// <summary>
/// Renders an authentic 3D in-scene cursor (MDX/M2 gauntlet or procedural OpenSCAD mesh)
/// directly in world space with guaranteed camera visibility (zero culling) and distance-proportional scaling.
/// </summary>
public sealed class SceneCursorRenderer : IDisposable
{
    private readonly GL _gl;
    private IDataSource? _dataSource;
    private ReplaceableTextureResolver? _texResolver;

    private CursorStyle _style = CursorStyle.ClassicOSArrow;
    private SceneCursorState _state = SceneCursorState.Pointer;
    private float _userScale = 1.0f;

    // Procedural meshes & shader
    private ShaderProgram? _proceduralShader;
    private ProceduralMesh? _arrowMesh;
    private ProceduralMesh? _reticleMesh;

    // Authentic MDX renderer (for Alpha 0.5.3)
    private MdxRenderer? _mdxCursorRenderer;
    private bool _mdxLoadAttempted;

    private bool _disposed;

    public CursorStyle Style
    {
        get => _style;
        set => _style = value;
    }

    public SceneCursorState State
    {
        get => _state;
        set => _state = value;
    }

    public float UserScale
    {
        get => _userScale;
        set => _userScale = Math.Clamp(value, 0.2f, 5.0f);
    }

    public SceneCursorRenderer(GL gl, IDataSource? dataSource = null, ReplaceableTextureResolver? texResolver = null)
    {
        _gl = gl;
        _dataSource = dataSource;
        _texResolver = texResolver;

        InitializeProceduralResources();
    }

    public void SetDataSource(IDataSource? dataSource, ReplaceableTextureResolver? texResolver = null)
    {
        if (_dataSource != dataSource)
        {
            _dataSource = dataSource;
            if (texResolver != null)
                _texResolver = texResolver;
            _mdxCursorRenderer?.Dispose();
            _mdxCursorRenderer = null;
            _mdxLoadAttempted = false;
        }
    }

    private void InitializeProceduralResources()
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
    vec3 norm = normalize(vNormal);
    float diff = max(dot(norm, normalize(uLightDir)), 0.25);
    vec3 col = uColor.rgb * diff + uColor.rgb * 0.35; // ambient boost
    FragColor = vec4(col, uColor.a);
}";

            _proceduralShader = ShaderProgram.Create(_gl, vertexShader, fragmentShader);
            _arrowMesh = OpenScadAssetResolver.TryLoadMesh(_gl, "cursor_pointer.off")
                ?? ProceduralMeshLoader.CreatePointerArrow(_gl);
            _reticleMesh = OpenScadAssetResolver.TryLoadMesh(_gl, "cursor_reticle.off")
                ?? ProceduralMeshLoader.CreateTargetReticle(_gl, 0.5f);
        }
        catch (Exception ex)
        {
            ViewerLog.Important(ViewerLog.Category.General, $"SceneCursorRenderer procedural shader init failed: {ex.Message}");
        }
    }

    private void EnsureMdxLoaded()
    {
        if (_mdxLoadAttempted || _dataSource == null) return;
        _mdxLoadAttempted = true;

        try
        {
            string[] candidates = {
                @"Interface\Cursor\Cursor.mdx",
                @"interface\cursor\cursor.mdx",
                @"Interface\Cursor\Point.mdx"
            };

            foreach (string candidate in candidates)
            {
                if (_dataSource.FileExists(candidate))
                {
                    byte[]? data = _dataSource.ReadFile(candidate);
                    if (data != null && data.Length > 0)
                    {
                        using var ms = new MemoryStream(data);
                        var mdx = MdxFile.Load(ms);
                        string modelDir = Path.GetDirectoryName(candidate) ?? "";
                        _mdxCursorRenderer = new MdxRenderer(_gl, mdx, modelDir, _dataSource, _texResolver, candidate);
                        ViewerLog.Info(ViewerLog.Category.Mdx, $"Loaded authentic 3D cursor model: {candidate}");
                        break;
                    }
                }
            }
        }
        catch (Exception ex)
        {
            ViewerLog.Important(ViewerLog.Category.Mdx, $"Failed to load 3D cursor MDX: {ex.Message}");
        }
    }

    /// <summary>
    /// Renders the 3D in-scene cursor with the camera culling invariant enforced.
    /// </summary>
    public void Render(
        Camera camera,
        Matrix4x4 proj,
        Vector3 rayOrigin,
        Vector3 rayDir,
        float? surfaceHitDistance,
        float fovDegrees,
        float nearPlane = 0.1f)
    {
        if (_disposed || _style == CursorStyle.ClassicOSArrow)
            return;

        // ── Camera Culling Invariant: Near-Plane Clamping ──────────────────
        // Guarantee distance is strictly >= nearPlane + margin so cursor is never clipped
        const float nearClipMargin = 0.15f;
        float minDistance = nearPlane + nearClipMargin;
        float defaultFloatingDistance = 5.0f; // Floating depth when pointing at sky

        float depth = surfaceHitDistance.HasValue ? MathF.Max(surfaceHitDistance.Value, minDistance) : defaultFloatingDistance;
        Vector3 worldPos = rayOrigin + rayDir * depth;

        // ── Distance-Proportional Scaling ─────────────────────────────────
        // Screen-relative constant pixel footprint across all camera distances
        float halfFovRad = (fovDegrees * 0.5f) * (MathF.PI / 180f);
        float baseScale = 0.045f * _userScale;
        float scale = baseScale * depth * MathF.Tan(halfFovRad);

        // ── Orientation (Face Camera with slight pitch for readability) ───
        // Build billboard orientation from camera vectors
        Vector3 forward = Vector3.Normalize(camera.Forward);
        Vector3 right = Vector3.Normalize(Vector3.Cross(forward, Vector3.UnitZ));
        if (right.LengthSquared() < 1e-4f) right = Vector3.UnitX;
        Vector3 up = Vector3.Normalize(Vector3.Cross(right, forward));

        Matrix4x4 rotation = new(
            right.X, right.Y, right.Z, 0f,
            up.X, up.Y, up.Z, 0f,
            -forward.X, -forward.Y, -forward.Z, 0f,
            0f, 0f, 0f, 1f
        );

        Matrix4x4 modelMatrix = Matrix4x4.CreateScale(scale)
                              * rotation
                              * Matrix4x4.CreateTranslation(worldPos);

        Matrix4x4 view = camera.GetViewMatrix();

        // ── Camera Invariant: Depth Range Bias (Always visible over surfaces)
        _gl.DepthRange(0.0f, 0.05f); // Maps depth to near band so geometry never occludes it
        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Lequal);

        if (_style == CursorStyle.AuthenticWoWGauntlet)
        {
            EnsureMdxLoaded();
            if (_mdxCursorRenderer != null)
            {
                RenderMdxCursor(modelMatrix, view, proj, camera.Position);
                _gl.DepthRange(0.0f, 1.0f);
                return;
            }
        }

        // Fallback to procedural pointer
        RenderProceduralCursor(modelMatrix, view, proj, camera.Forward);
        _gl.DepthRange(0.0f, 1.0f);
    }

    private void RenderMdxCursor(Matrix4x4 model, Matrix4x4 view, Matrix4x4 proj, Vector3 cameraPos)
    {
        if (_mdxCursorRenderer == null) return;

        Vector3 fogColor = new(0.5f, 0.5f, 0.5f);
        float fogStart = 1000f;
        float fogEnd = 2000f;
        Vector3 lightDir = Vector3.Normalize(new Vector3(0.5f, 0.3f, 0.8f));
        Vector3 lightColor = Vector3.One;
        Vector3 ambientColor = new Vector3(0.6f, 0.6f, 0.6f);

        _mdxCursorRenderer.BeginBatch(view, proj, fogColor, fogStart, fogEnd, cameraPos, lightDir, lightColor, ambientColor);
        _mdxCursorRenderer.RenderInstance(model, RenderPass.Opaque);
        _mdxCursorRenderer.RenderInstance(model, RenderPass.Transparent);
    }

    private void RenderProceduralCursor(Matrix4x4 model, Matrix4x4 view, Matrix4x4 proj, Vector3 camForward)
    {
        if (_proceduralShader == null) return;

        ProceduralMesh? mesh = _style == CursorStyle.TargetReticle ? _reticleMesh : _arrowMesh;
        if (mesh == null) return;

        _proceduralShader.Use();
        _proceduralShader.SetMat4("uModel", model);
        _proceduralShader.SetMat4("uView", view);
        _proceduralShader.SetMat4("uProj", proj);
        _proceduralShader.SetVec3("uLightDir", -camForward);

        Vector4 color = _state switch
        {
            SceneCursorState.Interact => new Vector4(0.2f, 0.95f, 0.35f, 1.0f),   // Bright green
            SceneCursorState.Speak => new Vector4(0.35f, 0.75f, 1.0f, 1.0f),     // Speech blue
            SceneCursorState.Attack => new Vector4(0.95f, 0.25f, 0.2f, 1.0f),    // Aggressive red
            SceneCursorState.CastGlow => new Vector4(0.9f, 0.4f, 1.0f, 1.0f),    // Magic violet
            _ => new Vector4(0.95f, 0.82f, 0.25f, 1.0f)                          // Iconic WoW gold
        };

        _proceduralShader.SetVec4("uColor", color);

        mesh.Draw();
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _arrowMesh?.Dispose();
        _arrowMesh = null;
        _reticleMesh?.Dispose();
        _reticleMesh = null;
        _proceduralShader?.Dispose();
        _proceduralShader = null;
        _mdxCursorRenderer?.Dispose();
        _mdxCursorRenderer = null;
    }
}
