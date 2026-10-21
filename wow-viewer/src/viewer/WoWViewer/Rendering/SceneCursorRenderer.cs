using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using Silk.NET.OpenGL;
using WoWViewer.DataSources;
using WoWViewer.Logging;
using WowViewer.Core.Mdx;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.Runtime.M2;

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

/// <summary>Which client asset is backing the authentic in-scene cursor.</summary>
public enum CursorModelKind
{
    None,
    Mdx,
    M2
}

/// <summary>
/// Renders an authentic 3D in-scene cursor (the client's own cursor.mdx / cursor.m2 model, or a
/// procedural OpenSCAD mesh) directly in world space with guaranteed camera visibility (zero
/// culling) and distance-proportional scaling.
/// </summary>
/// <remarks>
/// The authentic path used to call <see cref="MdxFile.Load"/> on every candidate, including the
/// <c>.m2</c> ones. An M2 is not an MDX, so the modern client's cursor never loaded and the
/// renderer silently fell back to the procedural arrow. The two formats are now dispatched by
/// extension: MDX through <see cref="MdxRenderer"/>, M2 through
/// <see cref="WowViewerM2RuntimeBridge"/> + <see cref="M2Renderer"/>.
/// </remarks>
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

    // Authentic client cursor model (MDX for Alpha-era clients, M2 for modern clients).
    private MdxRenderer? _mdxCursorRenderer;
    private M2Renderer? _m2CursorRenderer;
    private bool _cursorModelLoadAttempted;
    private CursorModelKind _cursorModelKind = CursorModelKind.None;
    private string? _cursorModelPath;
    private string? _cursorLoadError;
    private float _modelScaleNormalization = 1f;

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

    /// <summary>Format of the loaded authentic cursor model, or <see cref="CursorModelKind.None"/>.</summary>
    public CursorModelKind ModelKind => _cursorModelKind;

    /// <summary>Virtual path of the loaded authentic cursor model, if any.</summary>
    public string? LoadedCursorModelPath => _cursorModelPath;

    /// <summary>Why the authentic cursor model could not be loaded, if it could not.</summary>
    public string? LastLoadError => _cursorLoadError;

    /// <summary>True when a real client cursor model is loaded and drawable.</summary>
    public bool HasAuthenticCursorModel => _mdxCursorRenderer != null || _m2CursorRenderer != null;

    public SceneCursorRenderer(GL gl, IDataSource? dataSource = null, ReplaceableTextureResolver? texResolver = null)
    {
        _gl = gl;
        _dataSource = dataSource;
        _texResolver = texResolver;

        InitializeProceduralResources();
    }

    public void SetDataSource(IDataSource? dataSource, ReplaceableTextureResolver? texResolver = null)
    {
        if (ReferenceEquals(_dataSource, dataSource))
        {
            if (texResolver != null)
                _texResolver = texResolver;
            return;
        }

        _dataSource = dataSource;
        if (texResolver != null)
            _texResolver = texResolver;
        ReleaseCursorModel();
    }

    /// <summary>Drops the loaded cursor model so the next draw re-resolves it from the data source.</summary>
    public void ReloadCursorModel() => ReleaseCursorModel();

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

    private void EnsureCursorModelLoaded()
    {
        if (_cursorModelLoadAttempted || _dataSource == null) return;
        _cursorModelLoadAttempted = true;

        // MDX first: Alpha-era clients ship cursor.mdx. Modern clients ship cursor.m2 and have no
        // MDX at all, so the M2 candidates are the ones that actually resolve there.
        string[] candidates =
        {
            @"Interface\Cursor\Cursor.mdx",
            @"interface\cursor\cursor.mdx",
            @"Interface\Cursor\Point.mdx",
            @"interface\cursor\point.mdx",
            @"Interface\Cursor\Cursor.m2",
            @"interface\cursor\cursor.m2",
            @"Interface\Cursor\Point.m2",
            @"interface\cursor\point.m2",
        };

        foreach (string candidate in candidates)
        {
            if (!_dataSource.FileExists(candidate))
                continue;

            byte[]? data = _dataSource.ReadFile(candidate);
            if (data == null || data.Length == 0)
                continue;

            try
            {
                bool loaded = candidate.EndsWith(".m2", StringComparison.OrdinalIgnoreCase)
                    ? TryLoadM2Cursor(candidate, data)
                    : TryLoadMdxCursor(candidate, data);

                if (loaded)
                    return;
            }
            catch (Exception ex)
            {
                _cursorLoadError = $"{candidate}: {ex.Message}";
                ViewerLog.Important(ViewerLog.Category.Mdx, $"Failed to load 3D cursor model '{candidate}': {ex.Message}");
            }
        }

        _cursorLoadError ??= @"No Interface\Cursor\Cursor.mdx, Cursor.m2, Point.mdx or Point.m2 in the active data source.";
        ViewerLog.Important(ViewerLog.Category.Mdx, $"Authentic 3D cursor unavailable: {_cursorLoadError}");
    }

    private bool TryLoadMdxCursor(string candidate, byte[] data)
    {
        using var ms = new MemoryStream(data);
        var mdx = MdxFile.Load(ms);
        string modelDir = Path.GetDirectoryName(candidate) ?? "";
        _mdxCursorRenderer = new MdxRenderer(_gl, mdx, modelDir, _dataSource, _texResolver, candidate);
        _cursorModelKind = CursorModelKind.Mdx;
        _cursorModelPath = candidate;
        _modelScaleNormalization = ResolveModelScaleNormalization(_mdxCursorRenderer.BoundsMin, _mdxCursorRenderer.BoundsMax);
        ViewerLog.Info(ViewerLog.Category.Mdx, $"Loaded authentic 3D cursor MDX: {candidate}");
        return true;
    }

    private bool TryLoadM2Cursor(string candidate, byte[] data)
    {
        M2StaticRenderModel? runtimeModel = null;
        string? skinPath = null;

        foreach (string skinCandidate in EnumerateSkinCandidates(candidate))
        {
            if (!_dataSource!.FileExists(skinCandidate))
                continue;

            byte[]? skinBytes = _dataSource.ReadFile(skinCandidate);
            if (skinBytes == null || skinBytes.Length == 0)
                continue;

            try
            {
                runtimeModel = WowViewerM2RuntimeBridge.BuildStaticRenderModel(data, skinBytes, candidate, skinCandidate);
                skinPath = skinCandidate;
                break;
            }
            catch (Exception ex)
            {
                ViewerLog.Info(ViewerLog.Category.Mdx, $"Cursor skin '{skinCandidate}' did not build: {ex.Message}");
            }
        }

        // Alpha-era (0x100) M2 models carry their geometry inline and have no .skin at all.
        runtimeModel ??= WowViewerM2RuntimeBridge.BuildEra100StaticRenderModel(data, candidate);

        if (runtimeModel.Sections.Count == 0)
        {
            _cursorLoadError = $"{candidate}: model has no drawable sections.";
            return false;
        }

        _m2CursorRenderer = new M2Renderer(_gl, runtimeModel, candidate, _dataSource, _texResolver);
        _cursorModelKind = CursorModelKind.M2;
        _cursorModelPath = candidate;
        _modelScaleNormalization = ResolveModelScaleNormalization(_m2CursorRenderer.BoundsMin, _m2CursorRenderer.BoundsMax);
        ViewerLog.Info(
            ViewerLog.Category.Mdx,
            $"Loaded authentic 3D cursor M2: {candidate} (skin={skinPath ?? "inline"}, sections={runtimeModel.Sections.Count})");
        return true;
    }

    private static IEnumerable<string> EnumerateSkinCandidates(string modelPath)
    {
        string dir = Path.GetDirectoryName(modelPath) ?? string.Empty;
        string baseName = Path.GetFileNameWithoutExtension(modelPath);

        yield return Path.Combine(dir, baseName + ".skin");
        yield return Path.Combine(dir, baseName + "00.skin");
        yield return Path.Combine(dir, baseName + "01.skin");
        yield return Path.Combine(dir, baseName + "02.skin");

        // Some client builds store the cursor assets lower-case.
        string lower = baseName.ToLowerInvariant();
        yield return Path.Combine(dir, lower + ".skin");
        yield return Path.Combine(dir, lower + "00.skin");
    }

    /// <summary>
    /// Cursor models are authored around one unit. Only correct models that are far outside that
    /// range, so an authentic model keeps its authored proportions instead of being rescaled to a
    /// generic size.
    /// </summary>
    private static float ResolveModelScaleNormalization(Vector3 min, Vector3 max)
    {
        Vector3 extent = max - min;
        float largest = MathF.Max(extent.X, MathF.Max(extent.Y, extent.Z));
        if (!float.IsFinite(largest) || largest <= 1e-4f)
            return 1f;

        if (largest > 4f) return 4f / largest;
        if (largest < 0.25f) return 0.25f / largest;
        return 1f;
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

        // Load before touching GL state: constructing an M2Renderer compiles shaders and uploads
        // buffers, which would otherwise clobber the depth range and blend state set below.
        if (_style == CursorStyle.AuthenticWoWGauntlet)
            EnsureCursorModelLoaded();

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
        float baseScale = 0.045f * _userScale * _modelScaleNormalization;
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
        _gl.DepthMask(true);
        _gl.Disable(EnableCap.CullFace);
        _gl.Enable(EnableCap.Blend);
        _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);

        if (_style == CursorStyle.AuthenticWoWGauntlet && HasAuthenticCursorModel)
        {
            RenderCursorModel(modelMatrix, view, proj, camera.Position);
            _gl.DepthRange(0.0f, 1.0f);
            return;
        }

        // Fallback to procedural pointer
        RenderProceduralCursor(modelMatrix, view, proj, camera.Forward);
        _gl.DepthRange(0.0f, 1.0f);
    }

    private void RenderCursorModel(Matrix4x4 model, Matrix4x4 view, Matrix4x4 proj, Vector3 cameraPos)
    {
        Vector3 fogColor = new(0.5f, 0.5f, 0.5f);
        float fogStart = 1000f;
        float fogEnd = 2000f;
        Vector3 lightDir = Vector3.Normalize(new Vector3(0.5f, 0.3f, 0.8f));
        Vector3 lightColor = Vector3.One;
        Vector3 ambientColor = new Vector3(0.6f, 0.6f, 0.6f);

        if (_m2CursorRenderer != null)
        {
            _m2CursorRenderer.BeginBatch(view, proj, fogColor, fogStart, fogEnd, cameraPos, lightDir, lightColor, ambientColor);
            _m2CursorRenderer.RenderInstance(model, RenderPass.Opaque);
            _m2CursorRenderer.RenderInstance(model, RenderPass.Transparent);
            return;
        }

        if (_mdxCursorRenderer != null)
        {
            _mdxCursorRenderer.BeginBatch(view, proj, fogColor, fogStart, fogEnd, cameraPos, lightDir, lightColor, ambientColor);
            _mdxCursorRenderer.RenderInstance(model, RenderPass.Opaque);
            _mdxCursorRenderer.RenderInstance(model, RenderPass.Transparent);
        }
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

    private void ReleaseCursorModel()
    {
        _mdxCursorRenderer?.Dispose();
        _mdxCursorRenderer = null;
        _m2CursorRenderer?.Dispose();
        _m2CursorRenderer = null;
        _cursorModelLoadAttempted = false;
        _cursorModelKind = CursorModelKind.None;
        _cursorModelPath = null;
        _cursorLoadError = null;
        _modelScaleNormalization = 1f;
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
        ReleaseCursorModel();
    }
}
