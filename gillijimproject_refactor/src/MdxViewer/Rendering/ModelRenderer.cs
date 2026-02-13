using System.Numerics;
using MdxLTool.Formats.Mdx;
using MdxViewer.DataSources;
using MdxViewer.Logging;
using SereniaBLPLib;
using Silk.NET.OpenGL;

namespace MdxViewer.Rendering;

/// <summary>
/// Diagnostic logger for MDX texture issues
/// </summary>
internal static class MdxTextureDiagnosticLogger
{
    private static StreamWriter? _logWriter;
    private static readonly object _lock = new();

    public static void Initialize(string mdxName)
    {
        lock (_lock)
        {
            _logWriter?.Dispose();
            var logPath = Path.Combine("output", $"mdx_texture_diagnostic_{mdxName}_{DateTime.Now:yyyyMMdd_HHmmss}.log");
            Directory.CreateDirectory(Path.GetDirectoryName(logPath)!);
            _logWriter = new StreamWriter(logPath, false) { AutoFlush = true };
            _logWriter.WriteLine($"=== MDX Texture Diagnostic Log: {mdxName} ===");
            _logWriter.WriteLine($"Started: {DateTime.Now}");
            _logWriter.WriteLine();
        }
    }

    public static void Log(string message)
    {
        lock (_lock)
        {
            _logWriter?.WriteLine(message);
        }
    }

    public static void Close()
    {
        lock (_lock)
        {
            _logWriter?.Dispose();
            _logWriter = null;
        }
    }
}

/// <summary>
/// Controls which material layers are rendered in a given draw call.
/// Used for two-pass rendering: opaque first (depth write ON), then transparent (back-to-front).
/// </summary>
public enum RenderPass
{
    Both,
    Opaque,
    Transparent
}

/// <summary>
/// Renders an MDX model using OpenGL.
/// Handles per-geoset VAO/VBO setup, shader management, BLP2 textured rendering.
/// </summary>
public class MdxRenderer : ISceneRenderer
{
    private readonly GL _gl;
    private readonly MdxFile _mdx;
    private readonly string _modelDir;
    private readonly IDataSource? _dataSource;
    private readonly ReplaceableTextureResolver? _texResolver;
    private readonly string? _modelVirtualPath; // Path within MPQ for DBC lookup

    private uint _shaderProgram;
    private int _uModel, _uView, _uProj, _uHasTexture, _uColor, _uAlphaTest, _uUnshaded;
    private int _uFogColor, _uFogStart, _uFogEnd, _uCameraPos, _uAlphaThreshold;
    private int _uLightDir, _uLightColor, _uAmbientColor;
    private int _uSphereEnvMap;

    private readonly List<GeosetBuffers> _geosets = new();
    private readonly Dictionary<int, uint> _textures = new(); // textureIndex → GL texture
    private bool _wireframe;
    private MdxAnimator? _animator;
    private DateTime _lastFrameTime = DateTime.UtcNow;

    /// <summary>Model-space bounding box min corner.</summary>
    public Vector3 BoundsMin => new(_mdx.Model.Bounds.Extent.Min.X, _mdx.Model.Bounds.Extent.Min.Y, _mdx.Model.Bounds.Extent.Min.Z);
    /// <summary>Model-space bounding box max corner.</summary>
    public Vector3 BoundsMax => new(_mdx.Model.Bounds.Extent.Max.X, _mdx.Model.Bounds.Extent.Max.Y, _mdx.Model.Bounds.Extent.Max.Z);

    /// <summary>Animation controller (null if model has no bones)</summary>
    public MdxAnimator? Animator => _animator;

    public MdxRenderer(GL gl, MdxFile mdx, string modelDir, IDataSource? dataSource = null,
        ReplaceableTextureResolver? texResolver = null, string? modelVirtualPath = null)
    {
        _gl = gl;
        _mdx = mdx;
        _modelDir = modelDir;
        _dataSource = dataSource;
        
        var mdxName = Path.GetFileNameWithoutExtension(modelDir);
        MdxTextureDiagnosticLogger.Initialize(mdxName);
        _texResolver = texResolver;
        _modelVirtualPath = modelVirtualPath;

        InitShaders();
        InitBuffers();
        LoadTextures();

        // Initialize animation system
        if (mdx.Bones.Count > 0)
        {
            _animator = new MdxAnimator(mdx);
            if (_animator.HasAnimation)
                ViewerLog.Info(ViewerLog.Category.Mdx, $"Animation: {mdx.Bones.Count} bones, {mdx.Sequences.Count} sequences");
        }

        // Log material→texture mapping for debugging
        ViewerLog.Info(ViewerLog.Category.Mdx, $"Materials: {_mdx.Materials.Count}, Textures: {_mdx.Textures.Count}, Geosets: {_mdx.Geosets.Count}, GeosetAnimations: {_mdx.GeosetAnimations.Count}");
        
        // Log geoset animation info
        if (_mdx.GeosetAnimations.Count > 0)
        {
            ViewerLog.Debug(ViewerLog.Category.Mdx, "Geoset Animations:");
            foreach (var anim in _mdx.GeosetAnimations)
            {
                ViewerLog.Debug(ViewerLog.Category.Mdx, $"  Geoset {anim.GeosetId}: AlphaKeys={anim.AlphaKeys.Count}, ColorKeys={anim.ColorKeys.Count}, DefaultAlpha={anim.DefaultAlpha:F3}");
            }
        }
        
        for (int i = 0; i < _mdx.Geosets.Count; i++)
        {
            var g = _mdx.Geosets[i];
            int matId = g.MaterialId;
            string layerInfo = "no material";
            if (matId >= 0 && matId < _mdx.Materials.Count)
            {
                var mat = _mdx.Materials[matId];
                layerInfo = $"mat[{matId}] {mat.Layers.Count} layers";
                for (int l = 0; l < mat.Layers.Count; l++)
                {
                    var layer = mat.Layers[l];
                    string texInfo = layer.TextureId >= 0 && layer.TextureId < _mdx.Textures.Count
                        ? $"tex[{layer.TextureId}]={(_mdx.Textures[layer.TextureId].ReplaceableId > 0 ? $"Replaceable#{_mdx.Textures[layer.TextureId].ReplaceableId}" : _mdx.Textures[layer.TextureId].Path)}"
                        : $"texId={layer.TextureId}";
                    layerInfo += $" L{l}:[blend={layer.BlendMode} {texInfo}]";
                }
            }
            ViewerLog.Debug(ViewerLog.Category.Mdx, $"  Geoset[{i}]: {layerInfo} ({g.Vertices.Count}v)");
        }
    }

    // Geoset visibility
    public int SubObjectCount => _geosets.Count;
    public string GetSubObjectName(int index) => index < _geosets.Count ? $"Geoset {_geosets[index].GeosetIndex}" : "";
    public bool GetSubObjectVisible(int index) => index < _geosets.Count && _geosets[index].Visible;
    public void SetSubObjectVisible(int index, bool visible) { if (index < _geosets.Count) _geosets[index].Visible = visible; }

    public void ToggleWireframe()
    {
        _wireframe = !_wireframe;
    }

    /// <summary>
    /// Mirror matrix for standalone viewing: negates X to convert WoW left-handed → OpenGL right-handed.
    /// WorldScene callers use RenderWithTransform directly (no mirror needed — camera handles it).
    /// </summary>
    private static readonly Matrix4x4 MirrorX = Matrix4x4.CreateScale(-1f, 1f, 1f);

    public unsafe void Render(Matrix4x4 view, Matrix4x4 proj)
    {
        // Advance animation each frame
        if (_animator != null && _animator.HasAnimation)
        {
            var now = DateTime.UtcNow;
            float deltaMs = (float)(now - _lastFrameTime).TotalMilliseconds;
            _lastFrameTime = now;
            _animator.Update(Math.Clamp(deltaMs, 0f, 100f)); // Cap to avoid huge jumps
        }

        // Two-pass rendering: opaque first (depth write ON), then transparent (depth write OFF)
        // This prevents alpha/blended geosets from occluding opaque geometry behind them.
        RenderWithTransform(MirrorX, view, proj, RenderPass.Opaque);
        RenderWithTransform(MirrorX, view, proj, RenderPass.Transparent);
    }

    /// <summary>
    /// Render this model with a custom world transform (for doodad instancing).
    /// Pass = Opaque renders only opaque layers (depth write ON).
    /// Pass = Transparent renders only blended layers (depth write OFF).
    /// Pass = Both renders all layers (legacy behavior).
    /// fadeAlpha = 0..1 multiplier for distance-based fade-in/out (1.0 = fully opaque).
    /// </summary>
    public unsafe void RenderWithTransform(Matrix4x4 modelMatrix, Matrix4x4 view, Matrix4x4 proj, RenderPass pass = RenderPass.Both, float fadeAlpha = 1.0f,
        Vector3? fogColor = null, float fogStart = 200f, float fogEnd = 1500f, Vector3? cameraPos = null,
        Vector3? lightDir = null, Vector3? lightColor = null, Vector3? ambientColor = null)
    {
        _gl.UseProgram(_shaderProgram);

        _gl.Disable(EnableCap.CullFace);
        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthMask(true);

        var model = modelMatrix;
        _gl.UniformMatrix4(_uModel, 1, false, (float*)&model);
        _gl.UniformMatrix4(_uView, 1, false, (float*)&view);
        _gl.UniformMatrix4(_uProj, 1, false, (float*)&proj);

        // Fog uniforms (match terrain fog for seamless blending)
        var fc = fogColor ?? new Vector3(0.6f, 0.7f, 0.85f);
        var cp = cameraPos ?? Vector3.Zero;
        _gl.Uniform3(_uFogColor, fc.X, fc.Y, fc.Z);
        _gl.Uniform1(_uFogStart, fogStart);
        _gl.Uniform1(_uFogEnd, fogEnd);
        _gl.Uniform3(_uCameraPos, cp.X, cp.Y, cp.Z);

        // Lighting uniforms (match terrain lighting for consistent scene illumination)
        var ld = lightDir ?? Vector3.Normalize(new Vector3(0.5f, 0.3f, 1.0f));
        var lc = lightColor ?? new Vector3(1.0f, 0.95f, 0.85f);
        var ac = ambientColor ?? new Vector3(0.35f, 0.35f, 0.4f);
        _gl.Uniform3(_uLightDir, ld.X, ld.Y, ld.Z);
        _gl.Uniform3(_uLightColor, lc.X, lc.Y, lc.Z);
        _gl.Uniform3(_uAmbientColor, ac.X, ac.Y, ac.Z);

        if (_wireframe)
            _gl.PolygonMode(TriangleFace.FrontAndBack, PolygonMode.Line);
        else
            _gl.PolygonMode(TriangleFace.FrontAndBack, PolygonMode.Fill);

        for (int i = 0; i < _geosets.Count; i++)
        {
            var gb = _geosets[i];
            if (!gb.Visible) continue;

            // Render each material layer for this geoset
            bool anyLayerRendered = false;
            var geoset = _mdx.Geosets[gb.GeosetIndex];
            if (geoset.MaterialId >= 0 && geoset.MaterialId < _mdx.Materials.Count)
            {
                var material = _mdx.Materials[geoset.MaterialId];
                for (int l = 0; l < material.Layers.Count; l++)
                {
                    var layer = material.Layers[l];
                    int texId = layer.TextureId;

                    // Determine if this layer needs blending
                    // Layer 0 + Transparent blend = alpha-tested cutout (trees/foliage)
                    // Render in opaque pass with high alpha threshold, not as blended
                    bool isAlphaCutout = l == 0 && layer.BlendMode == MdlTexOp.Transparent;
                    bool needsBlend = !isAlphaCutout && (l > 0 || layer.BlendMode != MdlTexOp.Load);

                    // Filter by render pass — alpha cutout renders in opaque pass
                    if (pass == RenderPass.Opaque && needsBlend) continue;
                    if (pass == RenderPass.Transparent && !needsBlend) continue;

                    // ── Per-layer geometry flags (Ghidra-verified MDLGEO) ──
                    var geoFlags = layer.Flags;

                    // TwoSided (0x10): culling handled globally
                    _gl.Disable(EnableCap.CullFace);

                    // NoDepthTest (0x40): disable depth testing entirely
                    if (geoFlags.HasFlag(MdlGeoFlags.NoDepthTest))
                        _gl.Disable(EnableCap.DepthTest);
                    else
                        _gl.Enable(EnableCap.DepthTest);

                    // NoDepthSet (0x80): disable depth writing
                    bool noDepthWrite = geoFlags.HasFlag(MdlGeoFlags.NoDepthSet);

                    // Unshaded (0x1): skip lighting in shader
                    _gl.Uniform1(_uUnshaded, geoFlags.HasFlag(MdlGeoFlags.Unshaded) ? 1 : 0);

                    // SphereEnvMap (0x2): generate UVs from view-space normals for reflective surfaces
                    _gl.Uniform1(_uSphereEnvMap, geoFlags.HasFlag(MdlGeoFlags.SphereEnvMap) ? 1 : 0);

                    if (isAlphaCutout)
                    {
                        // Alpha-tested cutout: opaque pass, depth writes ON, high discard threshold
                        _gl.Disable(EnableCap.Blend);
                        _gl.DepthMask(true);
                        _gl.Uniform1(_uAlphaTest, 1);
                        _gl.Uniform1(_uAlphaThreshold, 0.75f);
                    }
                    else if (needsBlend)
                    {
                        _gl.Enable(EnableCap.Blend);
                        _gl.DepthMask(false); // Don't write depth for blended layers
                        _gl.Uniform1(_uAlphaTest, 1);
                        _gl.Uniform1(_uAlphaThreshold, 0.05f); // Low threshold for smooth blending
                        switch (layer.BlendMode)
                        {
                            case MdlTexOp.Transparent:
                                _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
                                break;
                            case MdlTexOp.Add:
                            case MdlTexOp.AddAlpha:
                                _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.One);
                                break;
                            case MdlTexOp.Modulate:
                            case MdlTexOp.Modulate2X:
                                _gl.BlendFunc(BlendingFactor.DstColor, BlendingFactor.Zero);
                                break;
                            default:
                                _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
                                break;
                        }
                    }
                    else
                    {
                        // When fading, even opaque layers need alpha blending
                        if (fadeAlpha < 1.0f)
                        {
                            _gl.Enable(EnableCap.Blend);
                            _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
                            _gl.DepthMask(!noDepthWrite);
                            _gl.Uniform1(_uAlphaTest, 1);
                            _gl.Uniform1(_uAlphaThreshold, 0.05f);
                        }
                        else
                        {
                            _gl.Disable(EnableCap.Blend);
                            _gl.DepthMask(!noDepthWrite);
                            _gl.Uniform1(_uAlphaTest, 0);
                        }
                    }

                    if (texId >= 0 && _textures.TryGetValue(texId, out uint glTex))
                    {
                        _gl.ActiveTexture(TextureUnit.Texture0);
                        _gl.BindTexture(TextureTarget.Texture2D, glTex);
                        _gl.Uniform1(_uHasTexture, 1);
                    }
                    else
                    {
                        _gl.Uniform1(_uHasTexture, l == 0 ? 0 : 1);
                        if (l > 0) continue;
                    }

                    float alpha = layer.StaticAlpha * fadeAlpha;
                    _gl.Uniform4(_uColor, 1.0f, 1.0f, 1.0f, alpha);

                    _gl.BindVertexArray(gb.Vao);
                    _gl.DrawElements(PrimitiveType.Triangles, gb.IndexCount, DrawElementsType.UnsignedShort, null);
                    _gl.BindVertexArray(0);
                    anyLayerRendered = true;

                    // Restore state after this layer
                    if (needsBlend)
                    {
                        _gl.Disable(EnableCap.Blend);
                        _gl.DepthMask(true);
                    }
                    else if (noDepthWrite)
                    {
                        _gl.DepthMask(true);
                    }
                    if (geoFlags.HasFlag(MdlGeoFlags.NoDepthTest))
                        _gl.Enable(EnableCap.DepthTest);
                }
            }

            // Fallback: no material or no layers rendered — treat as opaque
            if (!anyLayerRendered && pass != RenderPass.Transparent)
            {
                _gl.Uniform1(_uHasTexture, 0);
                _gl.Uniform4(_uColor, 1.0f, 1.0f, 1.0f, 1.0f);
                _gl.BindVertexArray(gb.Vao);
                _gl.DrawElements(PrimitiveType.Triangles, gb.IndexCount, DrawElementsType.UnsignedShort, null);
                _gl.BindVertexArray(0);
            }
        }

        _gl.PolygonMode(TriangleFace.FrontAndBack, PolygonMode.Fill);
    }

    private void InitShaders()
    {
        string vertSrc = @"
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;

out vec3 vNormal;
out vec2 vTexCoord;
out vec3 vFragPos;
out vec3 vViewNormal;

void main() {
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    vFragPos = worldPos.xyz;
    vNormal = mat3(transpose(inverse(uModel))) * aNormal;
    vViewNormal = mat3(uView) * vNormal;
    vTexCoord = aTexCoord;
    gl_Position = uProj * uView * worldPos;
}
";

        string fragSrc = @"
#version 330 core
in vec3 vNormal;
in vec2 vTexCoord;
in vec3 vFragPos;
in vec3 vViewNormal;

uniform sampler2D uSampler;
uniform int uHasTexture;
uniform int uAlphaTest;
uniform float uAlphaThreshold;
uniform int uUnshaded;
uniform int uSphereEnvMap;
uniform vec4 uColor;
uniform vec3 uFogColor;
uniform float uFogStart;
uniform float uFogEnd;
uniform vec3 uCameraPos;
uniform vec3 uLightDir;
uniform vec3 uLightColor;
uniform vec3 uAmbientColor;

out vec4 FragColor;

void main() {
    // Sphere environment map: generate UVs from view-space normals
    vec2 texCoord = vTexCoord;
    if (uSphereEnvMap == 1) {
        vec3 vn = normalize(vViewNormal);
        texCoord = vn.xy * 0.5 + 0.5;
    }

    vec4 texColor;
    if (uHasTexture == 1) {
        texColor = texture(uSampler, texCoord);
        if (uAlphaTest == 1 && texColor.a < uAlphaThreshold) discard;
    } else {
        texColor = vec4(1.0, 0.0, 1.0, 1.0);
    }

    // Lighting: skip if Unshaded flag (MDLGEO 0x1) is set
    vec3 litColor = texColor.rgb;
    if (uUnshaded == 0) {
        vec3 norm = normalize(vNormal);
        vec3 lightDir = normalize(uLightDir);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = uLightColor * diff;

        // Blinn-Phong specular
        vec3 viewDir = normalize(uCameraPos - vFragPos);
        vec3 halfDir = normalize(lightDir + viewDir);
        float spec = pow(max(dot(norm, halfDir), 0.0), 32.0);
        vec3 specular = uLightColor * spec * 0.3;

        litColor = texColor.rgb * (uAmbientColor + diffuse) + specular;
    }

    // Fog: blend to fog color based on distance from camera (matches terrain fog)
    // Skip fog for untextured (magenta fallback) fragments
    vec3 finalColor = litColor;
    if (uHasTexture == 1) {
        float dist = length(vFragPos - uCameraPos);
        float fogFactor = clamp((uFogEnd - dist) / (uFogEnd - uFogStart), 0.0, 1.0);
        finalColor = mix(uFogColor, litColor, fogFactor);
    }

    // For blended/alpha-tested layers (uAlphaTest=1), use texture alpha.
    // For opaque layers (uAlphaTest=0), force alpha=1.0 to prevent invisible geometry.
    float outAlpha = (uAlphaTest == 1) ? texColor.a : 1.0;
    FragColor = vec4(finalColor, outAlpha) * uColor;
}
";

        uint vert = CompileShader(ShaderType.VertexShader, vertSrc);
        uint frag = CompileShader(ShaderType.FragmentShader, fragSrc);

        _shaderProgram = _gl.CreateProgram();
        _gl.AttachShader(_shaderProgram, vert);
        _gl.AttachShader(_shaderProgram, frag);
        _gl.LinkProgram(_shaderProgram);

        _gl.GetProgram(_shaderProgram, ProgramPropertyARB.LinkStatus, out int status);
        if (status == 0)
        {
            string log = _gl.GetProgramInfoLog(_shaderProgram);
            throw new Exception($"Shader link error: {log}");
        }

        _gl.DeleteShader(vert);
        _gl.DeleteShader(frag);

        _gl.UseProgram(_shaderProgram);
        _uModel = _gl.GetUniformLocation(_shaderProgram, "uModel");
        _uView = _gl.GetUniformLocation(_shaderProgram, "uView");
        _uProj = _gl.GetUniformLocation(_shaderProgram, "uProj");
        _uHasTexture = _gl.GetUniformLocation(_shaderProgram, "uHasTexture");
        _uAlphaTest = _gl.GetUniformLocation(_shaderProgram, "uAlphaTest");
        _uAlphaThreshold = _gl.GetUniformLocation(_shaderProgram, "uAlphaThreshold");
        _uUnshaded = _gl.GetUniformLocation(_shaderProgram, "uUnshaded");
        _uColor = _gl.GetUniformLocation(_shaderProgram, "uColor");
        _uFogColor = _gl.GetUniformLocation(_shaderProgram, "uFogColor");
        _uFogStart = _gl.GetUniformLocation(_shaderProgram, "uFogStart");
        _uFogEnd = _gl.GetUniformLocation(_shaderProgram, "uFogEnd");
        _uCameraPos = _gl.GetUniformLocation(_shaderProgram, "uCameraPos");
        _uLightDir = _gl.GetUniformLocation(_shaderProgram, "uLightDir");
        _uLightColor = _gl.GetUniformLocation(_shaderProgram, "uLightColor");
        _uAmbientColor = _gl.GetUniformLocation(_shaderProgram, "uAmbientColor");
        _uSphereEnvMap = _gl.GetUniformLocation(_shaderProgram, "uSphereEnvMap");

        int samplerLoc = _gl.GetUniformLocation(_shaderProgram, "uSampler");
        _gl.Uniform1(samplerLoc, 0);
    }

    private uint CompileShader(ShaderType type, string source)
    {
        uint shader = _gl.CreateShader(type);
        _gl.ShaderSource(shader, source);
        _gl.CompileShader(shader);

        _gl.GetShader(shader, ShaderParameterName.CompileStatus, out int status);
        if (status == 0)
        {
            string log = _gl.GetShaderInfoLog(shader);
            throw new Exception($"Shader compile error ({type}): {log}");
        }

        return shader;
    }

    private unsafe void InitBuffers()
    {
        for (int i = 0; i < _mdx.Geosets.Count; i++)
        {
            var geoset = _mdx.Geosets[i];
            if (geoset.Vertices.Count == 0 || geoset.Indices.Count == 0)
                continue;

            var gb = new GeosetBuffers { GeosetIndex = i };

            // Interleave: pos(3) + normal(3) + uv(2) = 8 floats per vertex
            int vertCount = geoset.Vertices.Count;
            bool hasNormals = geoset.Normals.Count == vertCount;
            bool hasUVs = geoset.TexCoords.Count == vertCount;
            if (!hasUVs)
                ViewerLog.Trace($"[ModelRenderer] Geoset {i}: UV count mismatch! Verts={vertCount}, UVs={geoset.TexCoords.Count}");
            else if (geoset.TexCoords.Count > 0)
            {
                float uMin = float.MaxValue, uMax = float.MinValue, vMin = float.MaxValue, vMax = float.MinValue;
                foreach (var uv in geoset.TexCoords)
                {
                    if (uv.U < uMin) uMin = uv.U; if (uv.U > uMax) uMax = uv.U;
                    if (uv.V < vMin) vMin = uv.V; if (uv.V > vMax) vMax = uv.V;
                }
                var uvRangeMsg = $"Geoset {i}: {vertCount} verts, UV range U=[{uMin:F3},{uMax:F3}] V=[{vMin:F3},{vMax:F3}]";
                ViewerLog.Trace($"[ModelRenderer] {uvRangeMsg}");
                MdxTextureDiagnosticLogger.Log(uvRangeMsg);
            }

            float[] vertexData = new float[vertCount * 8];
            for (int v = 0; v < vertCount; v++)
            {
                // Pass through raw WoW model-local coords.
                var pos = geoset.Vertices[v];
                vertexData[v * 8 + 0] = pos.X;
                vertexData[v * 8 + 1] = pos.Y;
                vertexData[v * 8 + 2] = pos.Z;

                if (hasNormals)
                {
                    var n = geoset.Normals[v];
                    vertexData[v * 8 + 3] = n.X;
                    vertexData[v * 8 + 4] = n.Y;
                    vertexData[v * 8 + 5] = n.Z;
                }
                else
                {
                    vertexData[v * 8 + 3] = 0f;
                    vertexData[v * 8 + 4] = 1f;
                    vertexData[v * 8 + 5] = 0f;
                }

                if (hasUVs)
                {
                    var uv = geoset.TexCoords[v];
                    vertexData[v * 8 + 6] = uv.U;
                    vertexData[v * 8 + 7] = uv.V;
                }
            }

            // Create VAO/VBO/EBO
            gb.Vao = _gl.GenVertexArray();
            _gl.BindVertexArray(gb.Vao);

            gb.Vbo = _gl.GenBuffer();
            _gl.BindBuffer(BufferTargetARB.ArrayBuffer, gb.Vbo);
            fixed (float* ptr = vertexData)
                _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(vertexData.Length * sizeof(float)), ptr, BufferUsageARB.StaticDraw);

            gb.Ebo = _gl.GenBuffer();
            _gl.BindBuffer(BufferTargetARB.ElementArrayBuffer, gb.Ebo);
            var indices = geoset.Indices.ToArray();
            // Reverse triangle winding: WoW/D3D uses CW front faces, OpenGL uses CCW.
            for (int t = 0; t + 2 < indices.Length; t += 3)
                (indices[t + 1], indices[t + 2]) = (indices[t + 2], indices[t + 1]);
            fixed (ushort* ptr = indices)
                _gl.BufferData(BufferTargetARB.ElementArrayBuffer, (nuint)(indices.Length * sizeof(ushort)), ptr, BufferUsageARB.StaticDraw);

            uint stride = 8 * sizeof(float);
            // Position
            _gl.EnableVertexAttribArray(0);
            _gl.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, stride, (void*)0);
            // Normal
            _gl.EnableVertexAttribArray(1);
            _gl.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, stride, (void*)(3 * sizeof(float)));
            // TexCoord
            _gl.EnableVertexAttribArray(2);
            _gl.VertexAttribPointer(2, 2, VertexAttribPointerType.Float, false, stride, (void*)(6 * sizeof(float)));

            _gl.BindVertexArray(0);

            gb.IndexCount = (uint)indices.Length;
            _geosets.Add(gb);
        }
    }

    private void LoadTextures()
    {
        string modelName = _modelVirtualPath != null ? Path.GetFileName(_modelVirtualPath) : "?";
        ViewerLog.Info(ViewerLog.Category.Mdx, $"Loading {_mdx.Textures.Count} textures for {modelName}...");
        int loaded = 0, failed = 0, replaceableResolved = 0, replaceableFailed = 0;

        for (int i = 0; i < _mdx.Textures.Count; i++)
        {
            var tex = _mdx.Textures[i];
            string? texPath = tex.Path;

            // Handle Replaceable textures via DBC resolution
            if (string.IsNullOrEmpty(texPath) && tex.ReplaceableId > 0)
            {
                texPath = ResolveReplaceableTexture(tex.ReplaceableId);
                if (texPath != null)
                {
                    ViewerLog.Debug(ViewerLog.Category.Mdx, $"Texture[{i}]: Replaceable #{tex.ReplaceableId} -> {texPath}");
                    replaceableResolved++;
                }
                else
                {
                    ViewerLog.Debug(ViewerLog.Category.Mdx, $"Texture[{i}]: Replaceable #{tex.ReplaceableId} (unresolved)");
                    replaceableFailed++;
                }
            }

            if (string.IsNullOrEmpty(texPath))
            {
                ViewerLog.Debug(ViewerLog.Category.Mdx, $"Texture[{i}]: empty path, replaceableId={tex.ReplaceableId}");
                failed++;
                continue;
            }

            byte[]? blpData = null;
            string loadSource = "";

            // Try to find the actual path in the file set (case-insensitive)
            string? actualPath = null;
            if (_dataSource is MpqDataSource mpqDS)
            {
                actualPath = mpqDS.FindInFileSet(texPath);
            }
            
            // 1. Try data source (MPQ) first
            if (_dataSource != null)
            {
                // Try with actual path from file set if available
                if (actualPath != null)
                {
                    blpData = _dataSource.ReadFile(actualPath);
                    if (blpData != null)
                    {
                        texPath = actualPath; // Use the correctly-cased path
                        loadSource = "MPQ (file set match)";
                    }
                }
                
                // Try original path if not found yet
                if (blpData == null)
                {
                    blpData = _dataSource.ReadFile(texPath);
                    if (blpData != null)
                    {
                        loadSource = "MPQ (original path)";
                    }
                }
                
                // Try with normalized slashes
                if (blpData == null)
                {
                    var normalized = texPath.Replace('/', '\\');
                    blpData = _dataSource.ReadFile(normalized);
                    if (blpData != null)
                    {
                        texPath = normalized;
                        loadSource = "MPQ (normalized path)";
                    }
                }
                
                // Try case-insensitive: lowercase
                if (blpData == null)
                {
                    var lowerPath = texPath.ToLowerInvariant();
                    if (actualPath == null && _dataSource is MpqDataSource mpqDS2)
                    {
                        actualPath = mpqDS2.FindInFileSet(lowerPath);
                    }
                    if (actualPath != null)
                    {
                        blpData = _dataSource.ReadFile(actualPath);
                        if (blpData != null)
                        {
                            texPath = actualPath;
                            loadSource = "MPQ (lowercase match)";
                        }
                    }
                }
                
                // Try case-insensitive: uppercase
                if (blpData == null)
                {
                    var upperPath = texPath.ToUpperInvariant();
                    if (actualPath == null && _dataSource is MpqDataSource mpqDS3)
                    {
                        actualPath = mpqDS3.FindInFileSet(upperPath);
                    }
                    if (actualPath != null)
                    {
                        blpData = _dataSource.ReadFile(actualPath);
                        if (blpData != null)
                        {
                            texPath = actualPath;
                            loadSource = "MPQ (uppercase match)";
                        }
                    }
                }
                
                // Try case-insensitive search with just filename in model's directory
                if (blpData == null)
                {
                    string modelDir = _modelVirtualPath != null
                        ? Path.GetDirectoryName(_modelVirtualPath)?.Replace('/', '\\') ?? ""
                        : "";
                    if (!string.IsNullOrEmpty(modelDir))
                    {
                        string altPath = Path.Combine(modelDir, Path.GetFileName(texPath));
                        
                        // Try case-insensitive match for the alt path
                        if (_dataSource is MpqDataSource mpqDS4)
                        {
                            var foundPath = mpqDS4.FindInFileSet(altPath);
                            if (foundPath != null)
                            {
                                blpData = _dataSource.ReadFile(foundPath);
                                if (blpData != null)
                                {
                                    texPath = foundPath;
                                    loadSource = "MPQ (model dir match)";
                                }
                            }
                        }
                        
                        if (blpData == null)
                        {
                            blpData = _dataSource.ReadFile(altPath);
                            if (blpData != null)
                            {
                                texPath = altPath;
                                loadSource = "MPQ (model dir)";
                            }
                        }
                    }
                }
            }

            // 2. Try local BLP file on disk
            if (blpData == null)
            {
                string blpLocal = Path.Combine(_modelDir, Path.GetFileName(texPath));
                if (File.Exists(blpLocal))
                {
                    blpData = File.ReadAllBytes(blpLocal);
                    loadSource = "Local BLP";
                }
            }

            // 3. Try local PNG fallback
            if (blpData == null)
            {
                string pngName = Path.ChangeExtension(Path.GetFileName(texPath), ".png");
                string pngPath = Path.Combine(_modelDir, pngName);
                if (File.Exists(pngPath))
                {
                    uint glTex = LoadTextureFromPng(pngPath);
                    if (glTex != 0)
                    {
                        _textures[i] = glTex;
                        ViewerLog.Debug(ViewerLog.Category.Mdx, $"Texture[{i}]: {pngName} (PNG) - loaded");
                        loaded++;
                    }
                    else
                    {
                        ViewerLog.Debug(ViewerLog.Category.Mdx, $"Texture[{i}]: {pngName} (PNG) - failed to load");
                        failed++;
                    }
                    continue;
                }
            }

            if (blpData != null && blpData.Length > 0)
            {
                // Determine wrap mode from texture flags (per-axis)
                bool clampS = (tex.Flags & 0x1) != 0; // WrapWidth clamp
                bool clampT = (tex.Flags & 0x2) != 0; // WrapHeight clamp
                
                MdxTextureDiagnosticLogger.Log($"Texture[{i}]: {Path.GetFileName(texPath)}");
                MdxTextureDiagnosticLogger.Log($"  Flags: 0x{tex.Flags:X8} (clampS={clampS}, clampT={clampT})");
                MdxTextureDiagnosticLogger.Log($"  Source: {loadSource}, Size: {blpData.Length} bytes");
                
                uint glTex = LoadTextureFromBlp(blpData, texPath, clampS, clampT);
                if (glTex != 0)
                {
                    _textures[i] = glTex;
                    ViewerLog.Debug(ViewerLog.Category.Mdx, $"Texture[{i}]: {Path.GetFileName(texPath)} (BLP2, {blpData.Length} bytes, {loadSource})" +
                        (clampS || clampT ? $" [clamp S={clampS} T={clampT}]" : ""));
                    loaded++;
                }
                else
                {
                    ViewerLog.Debug(ViewerLog.Category.Mdx, $"Texture[{i}]: {Path.GetFileName(texPath)} (BLP2, {blpData.Length} bytes, {loadSource}) - failed to decode");
                    failed++;
                }
            }
            else
            {
                ViewerLog.Debug(ViewerLog.Category.Mdx, $"Texture[{i}]: not found ({texPath})");
                failed++;
            }
        }

        ViewerLog.Info(ViewerLog.Category.Mdx, $"Texture summary: {loaded} loaded, {failed} failed, {replaceableResolved} replaceable resolved, {replaceableFailed} replaceable failed");
        MdxTextureDiagnosticLogger.Close();
    }

    /// <summary>
    /// WoW client hardcoded default textures for each REPLACEABLE_MATERIAL_ID.
    /// These are the fallback textures used when no DBC override is found.
    /// From Ghidra analysis of the Alpha 0.5.3 client's texture resolution table.
    /// </summary>
    private static readonly Dictionary<uint, string> DefaultReplaceableTextures = new()
    {
        { 1,  @"Textures\ReplaceableTextures\CreatureSkin\CreatureSkin01.blp" },
        { 2,  @"Textures\ReplaceableTextures\ObjectSkin\ObjectSkin01.blp" },
        { 3,  @"Textures\ReplaceableTextures\WeaponBlade\WeaponBlade01.blp" },
        { 4,  @"Textures\ReplaceableTextures\WeaponHandle\WeaponHandle01.blp" },
        { 5,  @"Textures\ReplaceableTextures\Environment\Environment01.blp" },
        { 6,  @"Textures\ReplaceableTextures\CharHair\CharHair00_00.blp" },
        { 7,  @"Textures\ReplaceableTextures\CharFacialHair\CharFacialHair00_00.blp" },
        { 8,  @"Textures\ReplaceableTextures\SkinExtra\SkinExtra01.blp" },
        { 9,  @"Textures\ReplaceableTextures\UISkin\UISkin01.blp" },
        { 10, @"Textures\ReplaceableTextures\TaurenMane\TaurenMane00_00.blp" },
        { 11, @"Textures\ReplaceableTextures\Monster\Monster01_01.blp" },
        { 12, @"Textures\ReplaceableTextures\Monster\Monster01_02.blp" },
        { 13, @"Textures\ReplaceableTextures\Monster\Monster01_03.blp" },
    };

    private string? ResolveReplaceableTexture(uint replaceableId)
    {
        string modelName = _modelVirtualPath != null ? Path.GetFileName(_modelVirtualPath) : "?";

        // Strategy 1: DBCD-based resolver (creatures with DBC entries)
        if (_texResolver != null && _modelVirtualPath != null)
        {
            string? resolved = _texResolver.Resolve(_modelVirtualPath, replaceableId);
            if (resolved != null)
                return resolved;
        }

        // Strategy 2: Search model's directory for BLPs matching naming conventions
        // Environment doodads (trees, shrubs, rocks) have textures alongside the MDX
        if (_dataSource is MpqDataSource mpqDS && _modelVirtualPath != null)
        {
            string modelDir = Path.GetDirectoryName(_modelVirtualPath)?.Replace('/', '\\') ?? "";
            string modelBase = Path.GetFileNameWithoutExtension(_modelVirtualPath);

            // Build candidate list: ModelName + suffix + optional number + .blp
            var suffixes = replaceableId switch
            {
                1 => new[] { "Bark", "_Bark", "Trunk", "_Trunk", "Skin", "_Skin", "Body", "_Body", "" },
                2 => new[] { "Leaf", "_Leaf", "Leaves", "_Leaves", "Detail", "_Detail", "Foliage", "_Foliage", "" },
                _ => new[] { "" }
            };

            // Try each suffix with optional numeric variants (00, 01, etc.)
            foreach (var suffix in suffixes)
            {
                string baseName = string.IsNullOrEmpty(suffix) ? modelBase : modelBase + suffix;

                // Try exact: ModelNameSuffix.blp
                string candidate = Path.Combine(modelDir, baseName + ".blp");
                var found = mpqDS.FindInFileSet(candidate);
                if (found != null)
                {
                    ViewerLog.Debug(ViewerLog.Category.Mdx, $"  Replaceable #{replaceableId} -> {Path.GetFileName(found)} (naming convention)");
                    return found;
                }

                // Try with numbers: ModelNameSuffix00.blp, ModelNameSuffix01.blp
                for (int n = 0; n <= 3; n++)
                {
                    candidate = Path.Combine(modelDir, $"{baseName}{n:D2}.blp");
                    found = mpqDS.FindInFileSet(candidate);
                    if (found != null)
                    {
                        ViewerLog.Debug(ViewerLog.Category.Mdx, $"  Replaceable #{replaceableId} -> {Path.GetFileName(found)} (naming+num)");
                        return found;
                    }
                }
            }

            // Strategy 3: Scan all BLPs in model directory for fuzzy match
            var files = _dataSource.GetFileList(".blp");
            string modelDirLower = modelDir.ToLowerInvariant();
            string modelBaseLower = modelBase.ToLowerInvariant();
            var dirCandidates = files
                .Where(f =>
                {
                    string fLower = f.ToLowerInvariant();
                    string fDir = Path.GetDirectoryName(fLower)?.Replace('/', '\\') ?? "";
                    string fName = Path.GetFileNameWithoutExtension(fLower);
                    return fDir == modelDirLower && fName.StartsWith(modelBaseLower);
                })
                .OrderBy(f => f.Length)
                .ToList();

            if (dirCandidates.Count > 0)
            {
                // Score candidates by how well they match the expected texture type
                string? best = null;
                foreach (var c in dirCandidates)
                {
                    string fname = Path.GetFileNameWithoutExtension(c).ToLowerInvariant();
                    string extra = fname[modelBaseLower.Length..]; // part after model name

                    bool isBark = extra.Contains("bark") || extra.Contains("trunk") || extra.Contains("skin") || extra.Contains("body");
                    bool isLeaf = extra.Contains("leaf") || extra.Contains("leaves") || extra.Contains("detail") || extra.Contains("foliage");

                    if (replaceableId == 1 && isBark) { best = c; break; }
                    if (replaceableId == 2 && isLeaf) { best = c; break; }
                }
                // If no keyword match, use heuristic: replaceableId 1 = first BLP, 2 = second BLP
                if (best == null && dirCandidates.Count >= (int)replaceableId)
                    best = dirCandidates[(int)replaceableId - 1];
                else if (best == null)
                    best = dirCandidates[0];

                ViewerLog.Debug(ViewerLog.Category.Mdx, $"  Replaceable #{replaceableId} -> {Path.GetFileName(best)} (dir scan, {dirCandidates.Count} candidates)");
                return best;
            }
        }

        // Strategy 4: Hardcoded default replaceable texture paths (WoW client fallback)
        if (_dataSource != null && DefaultReplaceableTextures.TryGetValue(replaceableId, out string? defaultPath))
        {
            byte[]? data = _dataSource.ReadFile(defaultPath);
            if (data != null && data.Length > 0)
            {
                ViewerLog.Debug(ViewerLog.Category.Mdx, $"  Replaceable #{replaceableId} -> {Path.GetFileName(defaultPath)} (hardcoded default)");
                return defaultPath;
            }

            // Try case-insensitive search for the default path
            if (_dataSource is MpqDataSource mpqDS2)
            {
                var found = mpqDS2.FindInFileSet(defaultPath);
                if (found != null)
                {
                    ViewerLog.Debug(ViewerLog.Category.Mdx, $"  Replaceable #{replaceableId} -> {Path.GetFileName(found)} (hardcoded default, case-fixed)");
                    return found;
                }
            }
        }

        ViewerLog.Info(ViewerLog.Category.Mdx, $"  Replaceable #{replaceableId} UNRESOLVED for {modelName} (tried DBC, naming, dir scan, defaults)");
        return null;
    }

    private unsafe uint LoadTextureFromBlp(byte[] blpData, string name, bool clampS = false, bool clampT = false)
    {
        try
        {
            using var ms = new MemoryStream(blpData);
            using var blp = new BlpFile(ms);
            var bmp = blp.GetBitmap(0);

            // Convert System.Drawing.Bitmap to RGBA bytes
            int w = bmp.Width, h = bmp.Height;
            var pixels = new byte[w * h * 4];
            var rect = new System.Drawing.Rectangle(0, 0, w, h);
            var data = bmp.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadOnly,
                System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            try
            {
                var srcBytes = new byte[data.Stride * h];
                System.Runtime.InteropServices.Marshal.Copy(data.Scan0, srcBytes, 0, srcBytes.Length);

                // BGRA -> RGBA
                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        int srcIdx = y * data.Stride + x * 4;
                        int dstIdx = (y * w + x) * 4;
                        pixels[dstIdx + 0] = srcBytes[srcIdx + 2]; // R
                        pixels[dstIdx + 1] = srcBytes[srcIdx + 1]; // G
                        pixels[dstIdx + 2] = srcBytes[srcIdx + 0]; // B
                        pixels[dstIdx + 3] = srcBytes[srcIdx + 3]; // A
                    }
                }
            }
            finally
            {
                bmp.UnlockBits(data);
            }
            bmp.Dispose();

            return UploadTexture(pixels, (uint)w, (uint)h, clampS, clampT);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[MdxRenderer] Failed to decode BLP {name}: {ex.Message}");
            return 0;
        }
    }

    private unsafe uint LoadTextureFromPng(string path)
    {
        try
        {
            using var image = SixLabors.ImageSharp.Image.Load<SixLabors.ImageSharp.PixelFormats.Rgba32>(path);
            var pixels = new byte[image.Width * image.Height * 4];
            image.CopyPixelDataTo(pixels);
            return UploadTexture(pixels, (uint)image.Width, (uint)image.Height, clampS: false, clampT: false);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[MdxRenderer] Failed to load PNG {path}: {ex.Message}");
            return 0;
        }
    }

    private unsafe uint UploadTexture(byte[] pixels, uint width, uint height, bool clampS = false, bool clampT = false)
    {
        uint tex = _gl.GenTexture();
        _gl.BindTexture(TextureTarget.Texture2D, tex);

        fixed (byte* ptr = pixels)
            _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba,
                width, height, 0,
                PixelFormat.Rgba, PixelType.UnsignedByte, ptr);

        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.LinearMipmapLinear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);

        var wrapS = clampS ? TextureWrapMode.ClampToEdge : TextureWrapMode.Repeat;
        var wrapT = clampT ? TextureWrapMode.ClampToEdge : TextureWrapMode.Repeat;
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)wrapS);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)wrapT);
        _gl.GenerateMipmap(TextureTarget.Texture2D);

        _gl.BindTexture(TextureTarget.Texture2D, 0);
        return tex;
    }


    public void Dispose()
    {
        foreach (var gb in _geosets)
        {
            _gl.DeleteVertexArray(gb.Vao);
            _gl.DeleteBuffer(gb.Vbo);
            _gl.DeleteBuffer(gb.Ebo);
        }

        foreach (var tex in _textures.Values)
            _gl.DeleteTexture(tex);

        _gl.DeleteProgram(_shaderProgram);
    }

    private class GeosetBuffers
    {
        public int GeosetIndex;
        public uint Vao, Vbo, Ebo;
        public uint IndexCount;
        public bool Visible = true;
    }
}
