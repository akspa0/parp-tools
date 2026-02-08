using System.Numerics;
using MdxLTool.Formats.Mdx;
using MdxViewer.DataSources;
using SereniaBLPLib;
using Silk.NET.OpenGL;

namespace MdxViewer.Rendering;

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

    private readonly List<GeosetBuffers> _geosets = new();
    private readonly Dictionary<int, uint> _textures = new(); // textureIndex → GL texture
    private bool _wireframe;

    /// <summary>Model-space bounding box min corner.</summary>
    public Vector3 BoundsMin => new(_mdx.Model.Bounds.Extent.Min.X, _mdx.Model.Bounds.Extent.Min.Y, _mdx.Model.Bounds.Extent.Min.Z);
    /// <summary>Model-space bounding box max corner.</summary>
    public Vector3 BoundsMax => new(_mdx.Model.Bounds.Extent.Max.X, _mdx.Model.Bounds.Extent.Max.Y, _mdx.Model.Bounds.Extent.Max.Z);

    public MdxRenderer(GL gl, MdxFile mdx, string modelDir, IDataSource? dataSource = null,
        ReplaceableTextureResolver? texResolver = null, string? modelVirtualPath = null)
    {
        _gl = gl;
        _mdx = mdx;
        _modelDir = modelDir;
        _dataSource = dataSource;
        _texResolver = texResolver;
        _modelVirtualPath = modelVirtualPath;

        InitShaders();
        InitBuffers();
        LoadTextures();

        // Log material→texture mapping for debugging
        Console.WriteLine($"[MdxRenderer] Materials: {_mdx.Materials.Count}, Textures: {_mdx.Textures.Count}, Geosets: {_mdx.Geosets.Count}, GeosetAnimations: {_mdx.GeosetAnimations.Count}");
        
        // Log geoset animation info
        if (_mdx.GeosetAnimations.Count > 0)
        {
            Console.WriteLine("[MdxRenderer] Geoset Animations:");
            foreach (var anim in _mdx.GeosetAnimations)
            {
                Console.WriteLine($"  Geoset {anim.GeosetId}: AlphaKeys={anim.AlphaKeys.Count}, ColorKeys={anim.ColorKeys.Count}, DefaultAlpha={anim.DefaultAlpha:F3}");
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
            Console.WriteLine($"[MdxRenderer]   Geoset[{i}]: {layerInfo} ({g.Vertices.Count}v)");
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

    public unsafe void Render(Matrix4x4 view, Matrix4x4 proj)
    {
        RenderWithTransform(Matrix4x4.Identity, view, proj);
    }

    /// <summary>
    /// Render this model with a custom world transform (for doodad instancing).
    /// Pass = Opaque renders only opaque layers (depth write ON).
    /// Pass = Transparent renders only blended layers (depth write OFF).
    /// Pass = Both renders all layers (legacy behavior).
    /// </summary>
    public unsafe void RenderWithTransform(Matrix4x4 modelMatrix, Matrix4x4 view, Matrix4x4 proj, RenderPass pass = RenderPass.Both)
    {
        _gl.UseProgram(_shaderProgram);

        // MDX models often have single-sided geometry — disable culling to avoid holes
        _gl.Disable(EnableCap.CullFace);

        var model = modelMatrix;
        _gl.UniformMatrix4(_uModel, 1, false, (float*)&model);
        _gl.UniformMatrix4(_uView, 1, false, (float*)&view);
        _gl.UniformMatrix4(_uProj, 1, false, (float*)&proj);

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
                    bool needsBlend = l > 0 || layer.BlendMode != MdlTexOp.Load;

                    // Filter by render pass
                    if (pass == RenderPass.Opaque && needsBlend) continue;
                    if (pass == RenderPass.Transparent && !needsBlend) continue;

                    // ── Per-layer geometry flags (Ghidra-verified MDLGEO) ──
                    var geoFlags = layer.Flags;

                    // TwoSided (0x10): disable back-face culling for this layer
                    if (geoFlags.HasFlag(MdlGeoFlags.TwoSided))
                        _gl.Disable(EnableCap.CullFace);
                    else
                        _gl.Disable(EnableCap.CullFace); // MDX models are generally single-sided geometry but often need both sides

                    // NoDepthTest (0x40): disable depth testing entirely
                    if (geoFlags.HasFlag(MdlGeoFlags.NoDepthTest))
                        _gl.Disable(EnableCap.DepthTest);
                    else
                        _gl.Enable(EnableCap.DepthTest);

                    // NoDepthSet (0x80): disable depth writing
                    bool noDepthWrite = geoFlags.HasFlag(MdlGeoFlags.NoDepthSet);

                    // Unshaded (0x1): skip lighting in shader
                    _gl.Uniform1(_uUnshaded, geoFlags.HasFlag(MdlGeoFlags.Unshaded) ? 1 : 0);

                    if (needsBlend)
                    {
                        _gl.Enable(EnableCap.Blend);
                        _gl.DepthMask(false); // Don't write depth for blended layers
                        _gl.Uniform1(_uAlphaTest, 1);
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
                        _gl.Disable(EnableCap.Blend);
                        _gl.DepthMask(!noDepthWrite); // Respect NoDepthSet flag
                        _gl.Uniform1(_uAlphaTest, 0);
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

                    float alpha = layer.StaticAlpha;
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

void main() {
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    vFragPos = worldPos.xyz;
    vNormal = mat3(transpose(inverse(uModel))) * aNormal;
    vTexCoord = aTexCoord;
    gl_Position = uProj * uView * worldPos;
}
";

        string fragSrc = @"
#version 330 core
in vec3 vNormal;
in vec2 vTexCoord;
in vec3 vFragPos;

uniform sampler2D uSampler;
uniform int uHasTexture;
uniform int uAlphaTest;
uniform int uUnshaded;
uniform vec4 uColor;

out vec4 FragColor;

void main() {
    vec4 texColor;
    if (uHasTexture == 1) {
        texColor = texture(uSampler, vTexCoord);
        if (texColor.a < 0.3) discard;
    } else {
        texColor = vec4(1.0, 0.0, 1.0, 1.0);
    }

    // Lighting: skip if Unshaded flag (MDLGEO 0x1) is set
    float lighting = 1.0;
    if (uUnshaded == 0) {
        vec3 norm = normalize(vNormal);
        vec3 lightDir = normalize(vec3(0.5, 1.0, 0.3));
        float diff = max(dot(norm, lightDir), 0.0);
        lighting = 0.35 + diff * 0.65;
    }

    float outAlpha = (uAlphaTest == 0) ? 1.0 : texColor.a;
    FragColor = vec4(texColor.rgb * lighting, outAlpha) * uColor;
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
        _uUnshaded = _gl.GetUniformLocation(_shaderProgram, "uUnshaded");
        _uColor = _gl.GetUniformLocation(_shaderProgram, "uColor");

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

            float[] vertexData = new float[vertCount * 8];
            for (int v = 0; v < vertCount; v++)
            {
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
        Console.WriteLine($"[MdxRenderer] Loading {_mdx.Textures.Count} textures for {modelName}...");
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
                    Console.WriteLine($"[MdxRenderer] Texture[{i}]: Replaceable #{tex.ReplaceableId} → {texPath}");
                    replaceableResolved++;
                }
                else
                {
                    Console.WriteLine($"[MdxRenderer] Texture[{i}]: Replaceable #{tex.ReplaceableId} (unresolved)");
                    replaceableFailed++;
                }
            }

            if (string.IsNullOrEmpty(texPath))
            {
                Console.WriteLine($"[MdxRenderer] Texture[{i}]: empty path, replaceableId={tex.ReplaceableId}");
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
                        Console.WriteLine($"[MdxRenderer] Texture[{i}]: {pngName} (PNG) - loaded");
                        loaded++;
                    }
                    else
                    {
                        Console.WriteLine($"[MdxRenderer] Texture[{i}]: {pngName} (PNG) - failed to load");
                        failed++;
                    }
                    continue;
                }
            }

            if (blpData != null && blpData.Length > 0)
            {
                // Determine wrap mode from texture flags
                bool clamp = (tex.Flags & 0x1) != 0; // WrapWidth clamp
                bool clampV = (tex.Flags & 0x2) != 0; // WrapHeight clamp
                uint glTex = LoadTextureFromBlp(blpData, texPath, clamp || clampV);
                if (glTex != 0)
                {
                    _textures[i] = glTex;
                    Console.WriteLine($"[MdxRenderer] Texture[{i}]: {Path.GetFileName(texPath)} (BLP2, {blpData.Length} bytes, {loadSource}){(clamp ? " [clamped]" : "")}");
                    loaded++;
                }
                else
                {
                    Console.WriteLine($"[MdxRenderer] Texture[{i}]: {Path.GetFileName(texPath)} (BLP2, {blpData.Length} bytes, {loadSource}) - failed to decode");
                    failed++;
                }
            }
            else
            {
                Console.WriteLine($"[MdxRenderer] Texture[{i}]: not found ({texPath})");
                failed++;
            }
        }

        Console.WriteLine($"[MdxRenderer] Texture loading summary: {loaded} loaded, {failed} failed, {replaceableResolved} replaceable resolved, {replaceableFailed} replaceable failed");
    }

    private string? ResolveReplaceableTexture(uint replaceableId)
    {
        // Try DBCD-based resolver first
        if (_texResolver != null && _modelVirtualPath != null)
        {
            string? resolved = _texResolver.Resolve(_modelVirtualPath, replaceableId);
            if (resolved != null) return resolved;
        }

        // For environment models (trees, shrubs), replaceable textures use conventions:
        // ReplaceableId 1 = bark/skin, 2 = leaves/detail
        // Try to find BLPs in the model's directory that match common naming patterns
        if (_dataSource != null && _modelVirtualPath != null)
        {
            string modelDir = Path.GetDirectoryName(_modelVirtualPath)?.Replace('/', '\\') ?? "";
            string modelBase = Path.GetFileNameWithoutExtension(_modelVirtualPath);

            // Common suffixes for replaceable texture IDs in environment models
            string[] suffixes = replaceableId switch
            {
                1 => new[] { "Bark", "Trunk", "Skin", "Body", "" },
                2 => new[] { "Leaf", "Leaves", "Detail", "Foliage", "" },
                _ => new[] { "" }
            };

            // Search for BLPs in model directory matching model name + suffix
            if (_dataSource is MpqDataSource mpqDS)
            {
                foreach (var suffix in suffixes)
                {
                    string candidate = string.IsNullOrEmpty(suffix)
                        ? Path.Combine(modelDir, modelBase + ".blp")
                        : Path.Combine(modelDir, modelBase + suffix + ".blp");
                    var found = mpqDS.FindInFileSet(candidate);
                    if (found != null)
                    {
                        Console.WriteLine($"[MdxRenderer] Replaceable #{replaceableId} resolved via naming convention: {found}");
                        return found;
                    }
                }
            }

            // Last resort: scan all BLPs in model directory for any match
            var files = _dataSource.GetFileList(".blp");
            var candidates = files
                .Where(f => f.StartsWith(modelDir, StringComparison.OrdinalIgnoreCase) &&
                            Path.GetFileNameWithoutExtension(f).StartsWith(modelBase, StringComparison.OrdinalIgnoreCase))
                .OrderBy(f => f.Length) // Prefer shorter names (more likely to be the base texture)
                .ToList();

            if (candidates.Count > 0)
            {
                // For replaceableId 1, prefer bark/trunk textures; for 2, prefer leaf textures
                string? best = null;
                foreach (var c in candidates)
                {
                    string fname = Path.GetFileNameWithoutExtension(c).ToLowerInvariant();
                    bool isBark = fname.Contains("bark") || fname.Contains("trunk");
                    bool isLeaf = fname.Contains("leaf") || fname.Contains("leaves");

                    if (replaceableId == 1 && isBark) { best = c; break; }
                    if (replaceableId == 2 && isLeaf) { best = c; break; }
                }
                if (best == null) best = candidates[0]; // Fallback to first match
                Console.WriteLine($"[MdxRenderer] Replaceable #{replaceableId} resolved via directory scan: {best}");
                return best;
            }
        }

        return null;
    }

    private unsafe uint LoadTextureFromBlp(byte[] blpData, string name, bool clamp = false)
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

            return UploadTexture(pixels, (uint)w, (uint)h, clamp);
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
            return UploadTexture(pixels, (uint)image.Width, (uint)image.Height);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[MdxRenderer] Failed to load PNG {path}: {ex.Message}");
            return 0;
        }
    }

    private unsafe uint UploadTexture(byte[] pixels, uint width, uint height, bool clamp = false)
    {
        uint tex = _gl.GenTexture();
        _gl.BindTexture(TextureTarget.Texture2D, tex);

        fixed (byte* ptr = pixels)
            _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba,
                width, height, 0,
                PixelFormat.Rgba, PixelType.UnsignedByte, ptr);

        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.LinearMipmapLinear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);

        var wrapMode = clamp ? TextureWrapMode.ClampToEdge : TextureWrapMode.Repeat;
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)wrapMode);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)wrapMode);
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
