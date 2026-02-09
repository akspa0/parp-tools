using System.Numerics;
using System.Text;
using MdxLTool.Formats.Mdx;
using MdxViewer.DataSources;
using Silk.NET.OpenGL;
using WoWMapConverter.Core.Converters;

namespace MdxViewer.Rendering;

/// <summary>
/// Renders a WMO (World Map Object) using OpenGL.
/// Uses WoWMapConverter.Core's WmoV14Data model for geometry.
/// Supports loading and rendering MDX doodads from DoodadSets.
/// </summary>
public class WmoRenderer : ISceneRenderer
{
    private readonly GL _gl;
    private readonly WmoV14ToV17Converter.WmoV14Data _wmo;
    private readonly string _modelDir;
    private readonly IDataSource? _dataSource;
    private readonly ReplaceableTextureResolver? _texResolver;

    private uint _shaderProgram;
    private int _uModel, _uView, _uProj, _uHasTexture, _uColor;

    private readonly List<GroupBuffers> _groups = new();
    private bool _wireframe;

    // Material textures: materialIndex → GL texture handle
    private readonly Dictionary<int, uint> _materialTextures = new();

    // Doodad support
    private readonly Dictionary<string, MdxRenderer?> _doodadModelCache = new(StringComparer.OrdinalIgnoreCase);
    private readonly List<DoodadInstance> _doodadInstances = new();
    private readonly List<string> _doodadNames = new(); // resolved from MODN
    private int _activeDoodadSet = 0;
    private bool _doodadsVisible = true;
    private readonly string _cacheDir;

    // WMO liquid meshes (from MLIQ chunks in groups)
    private readonly List<LiquidMeshData> _liquidMeshes = new();
    private uint _liquidShader;
    private int _uLiqModel, _uLiqView, _uLiqProj, _uLiqColor;

    public WmoRenderer(GL gl, WmoV14ToV17Converter.WmoV14Data wmo, string modelDir,
        IDataSource? dataSource = null, ReplaceableTextureResolver? texResolver = null)
    {
        _gl = gl;
        _wmo = wmo;
        _modelDir = modelDir;
        _dataSource = dataSource;
        _texResolver = texResolver;
        _cacheDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "output", "cache");

        InitShaders();
        InitLiquidShader();
        InitBuffers();
        BuildLiquidMeshes();
        LoadMaterialTextures();
        ResolveDoodadNames();
        LoadActiveDoodadSet();
    }

    /// <summary>MOHD bounding box min in WMO local space.</summary>
    public Vector3 BoundsMin => _wmo.BoundsMin;
    /// <summary>MOHD bounding box max in WMO local space.</summary>
    public Vector3 BoundsMax => _wmo.BoundsMax;

    // Sub-object visibility: WMO groups + doodad toggle
    // Layout: [0..N-1] = WMO groups, [N] = "Doodads" toggle, [N+1..] = individual doodad models
    public int SubObjectCount => _groups.Count + 1 + _doodadInstances.Count;

    public string GetSubObjectName(int index)
    {
        if (index < _groups.Count)
            return $"Group {_groups[index].GroupIndex}";
        if (index == _groups.Count)
            return $"--- Doodads ({_doodadInstances.Count}) ---";
        int di = index - _groups.Count - 1;
        if (di < _doodadInstances.Count)
        {
            var inst = _doodadInstances[di];
            return $"  Doodad: {Path.GetFileNameWithoutExtension(inst.ModelPath)}";
        }
        return "";
    }

    public bool GetSubObjectVisible(int index)
    {
        if (index < _groups.Count)
            return _groups[index].Visible;
        if (index == _groups.Count)
            return _doodadsVisible;
        int di = index - _groups.Count - 1;
        if (di < _doodadInstances.Count)
            return _doodadInstances[di].Visible;
        return false;
    }

    public void SetSubObjectVisible(int index, bool visible)
    {
        if (index < _groups.Count)
            _groups[index].Visible = visible;
        else if (index == _groups.Count)
            _doodadsVisible = visible;
        else
        {
            int di = index - _groups.Count - 1;
            if (di < _doodadInstances.Count)
                _doodadInstances[di].Visible = visible;
        }
    }

    // DoodadSet management
    public int DoodadSetCount => _wmo.DoodadSets.Count;
    public int ActiveDoodadSet => _activeDoodadSet;
    public string GetDoodadSetName(int index) =>
        index < _wmo.DoodadSets.Count ? (_wmo.DoodadSets[index].Name ?? $"Set {index}") : "";

    public void SetActiveDoodadSet(int index)
    {
        if (index == _activeDoodadSet || index < 0 || index >= _wmo.DoodadSets.Count) return;
        _activeDoodadSet = index;
        LoadActiveDoodadSet();
    }

    public void ToggleWireframe()
    {
        _wireframe = !_wireframe;
    }

    public unsafe void Render(Matrix4x4 view, Matrix4x4 proj)
    {
        RenderWithTransform(Matrix4x4.Identity, view, proj);
    }

    /// <summary>
    /// Render this WMO with a custom world transform (for placed WMO instances in WorldScene).
    /// </summary>
    public unsafe void RenderWithTransform(Matrix4x4 modelMatrix, Matrix4x4 view, Matrix4x4 proj)
    {
        // 1. Render WMO geometry
        _gl.UseProgram(_shaderProgram);

        // Disable face culling — WMO interiors need both sides visible
        _gl.Disable(EnableCap.CullFace);

        var model = modelMatrix;
        _gl.UniformMatrix4(_uModel, 1, false, (float*)&model);
        _gl.UniformMatrix4(_uView, 1, false, (float*)&view);
        _gl.UniformMatrix4(_uProj, 1, false, (float*)&proj);

        if (_wireframe)
            _gl.PolygonMode(TriangleFace.FrontAndBack, PolygonMode.Line);
        else
            _gl.PolygonMode(TriangleFace.FrontAndBack, PolygonMode.Fill);

        foreach (var gb in _groups)
        {
            if (!gb.Visible) continue;
            var group = _wmo.Groups[gb.GroupIndex];

            _gl.BindVertexArray(gb.Vao);

            // Render per-batch with material textures
            if (group.Batches.Count > 0)
            {
                foreach (var batch in group.Batches)
                {
                    int matId = batch.MaterialId;
                    if (_materialTextures.TryGetValue(matId, out uint glTex))
                    {
                        _gl.ActiveTexture(TextureUnit.Texture0);
                        _gl.BindTexture(TextureTarget.Texture2D, glTex);
                        _gl.Uniform1(_uHasTexture, 1);
                        _gl.Uniform4(_uColor, 1.0f, 1.0f, 1.0f, 1.0f);
                    }
                    else
                    {
                        // Fallback: per-group color
                        _gl.Uniform1(_uHasTexture, 0);
                        float r = ((gb.GroupIndex * 67 + 13) % 255) / 255f;
                        float g = ((gb.GroupIndex * 131 + 7) % 255) / 255f;
                        float b = ((gb.GroupIndex * 43 + 29) % 255) / 255f;
                        _gl.Uniform4(_uColor, r, g, b, 1.0f);
                    }
                    _gl.DrawElements(PrimitiveType.Triangles, batch.IndexCount,
                        DrawElementsType.UnsignedShort, (void*)(batch.FirstIndex * sizeof(ushort)));
                }
            }
            else
            {
                // No batches — draw whole group with fallback color
                _gl.Uniform1(_uHasTexture, 0);
                float r = ((gb.GroupIndex * 67 + 13) % 255) / 255f;
                float g = ((gb.GroupIndex * 131 + 7) % 255) / 255f;
                float b = ((gb.GroupIndex * 43 + 29) % 255) / 255f;
                _gl.Uniform4(_uColor, r, g, b, 1.0f);
                _gl.DrawElements(PrimitiveType.Triangles, gb.IndexCount, DrawElementsType.UnsignedShort, null);
            }

            _gl.BindVertexArray(0);
        }

        _gl.PolygonMode(TriangleFace.FrontAndBack, PolygonMode.Fill);

        // 2. Render doodad instances (their transforms are in WMO-local space,
        //    so multiply by the WMO's modelMatrix to place them in world space)
        if (_doodadsVisible)
        {
            foreach (var inst in _doodadInstances)
            {
                if (!inst.Visible || inst.Renderer == null) continue;
                var doodadWorld = inst.Transform * modelMatrix;
                inst.Renderer.RenderWithTransform(doodadWorld, view, proj);
            }
        }

        // 3. Render WMO liquid surfaces (semi-transparent water)
        if (_liquidMeshes.Count > 0)
        {
            _gl.UseProgram(_liquidShader);
            _gl.UniformMatrix4(_uLiqModel, 1, false, (float*)&model);
            _gl.UniformMatrix4(_uLiqView, 1, false, (float*)&view);
            _gl.UniformMatrix4(_uLiqProj, 1, false, (float*)&proj);

            _gl.Enable(EnableCap.Blend);
            _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
            _gl.DepthMask(false);

            foreach (var liq in _liquidMeshes)
            {
                _gl.Uniform4(_uLiqColor, liq.ColorR, liq.ColorG, liq.ColorB, liq.ColorA);
                _gl.BindVertexArray(liq.Vao);
                _gl.DrawElements(PrimitiveType.Triangles, liq.IndexCount, DrawElementsType.UnsignedShort, null);
            }

            _gl.BindVertexArray(0);
            _gl.DepthMask(true);
            _gl.Disable(EnableCap.Blend);
        }

        _gl.Enable(EnableCap.CullFace);
    }

    private void InitShaders()
    {
        // Same shader as MdxRenderer — shared basic lit shader
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
uniform vec4 uColor;

out vec4 FragColor;

void main() {
    vec3 norm = normalize(vNormal);
    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.3));
    float diff = abs(dot(norm, lightDir));
    float ambient = 0.4;
    float lighting = ambient + diff * 0.6;

    vec4 texColor;
    if (uHasTexture == 1) {
        texColor = texture(uSampler, vTexCoord);
    } else {
        texColor = uColor;
    }

    FragColor = vec4(texColor.rgb * lighting, 1.0);
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
            throw new Exception($"Shader link error: {_gl.GetProgramInfoLog(_shaderProgram)}");

        _gl.DeleteShader(vert);
        _gl.DeleteShader(frag);

        _gl.UseProgram(_shaderProgram);
        _uModel = _gl.GetUniformLocation(_shaderProgram, "uModel");
        _uView = _gl.GetUniformLocation(_shaderProgram, "uView");
        _uProj = _gl.GetUniformLocation(_shaderProgram, "uProj");
        _uHasTexture = _gl.GetUniformLocation(_shaderProgram, "uHasTexture");
        _uColor = _gl.GetUniformLocation(_shaderProgram, "uColor");
    }

    private uint CompileShader(ShaderType type, string source)
    {
        uint shader = _gl.CreateShader(type);
        _gl.ShaderSource(shader, source);
        _gl.CompileShader(shader);

        _gl.GetShader(shader, ShaderParameterName.CompileStatus, out int status);
        if (status == 0)
            throw new Exception($"Shader compile error ({type}): {_gl.GetShaderInfoLog(shader)}");

        return shader;
    }

    private unsafe void InitBuffers()
    {
        for (int gi = 0; gi < _wmo.Groups.Count; gi++)
        {
            var group = _wmo.Groups[gi];
            if (group.Vertices.Count == 0 || group.Indices.Count == 0)
                continue;

            var gb = new GroupBuffers { GroupIndex = gi };

            // Generate normals from geometry
            var normals = GenerateNormals(group);

            int vertCount = group.Vertices.Count;
            bool hasUVs = group.UVs.Count == vertCount;

            // Interleave: pos(3) + normal(3) + uv(2) = 8 floats
            float[] vertexData = new float[vertCount * 8];
            for (int v = 0; v < vertCount; v++)
            {
                // Pass through raw WoW model-local coords.
                // Coordinate conversion is handled by the placement transform.
                var pos = group.Vertices[v];
                vertexData[v * 8 + 0] = pos.X;
                vertexData[v * 8 + 1] = pos.Y;
                vertexData[v * 8 + 2] = pos.Z;

                var n = v < normals.Count ? normals[v] : Vector3.UnitY;
                vertexData[v * 8 + 3] = n.X;
                vertexData[v * 8 + 4] = n.Y;
                vertexData[v * 8 + 5] = n.Z;

                if (hasUVs)
                {
                    var uv = group.UVs[v];
                    vertexData[v * 8 + 6] = uv.X;
                    vertexData[v * 8 + 7] = uv.Y;
                }
            }

            gb.Vao = _gl.GenVertexArray();
            _gl.BindVertexArray(gb.Vao);

            gb.Vbo = _gl.GenBuffer();
            _gl.BindBuffer(BufferTargetARB.ArrayBuffer, gb.Vbo);
            fixed (float* ptr = vertexData)
                _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(vertexData.Length * sizeof(float)), ptr, BufferUsageARB.StaticDraw);

            gb.Ebo = _gl.GenBuffer();
            _gl.BindBuffer(BufferTargetARB.ElementArrayBuffer, gb.Ebo);
            var indices = group.Indices.ToArray();
            // Reverse triangle winding: WoW/D3D uses CW front faces, OpenGL uses CCW.
            // Swap v1↔v2 in each triangle to convert CW→CCW.
            for (int t = 0; t + 2 < indices.Length; t += 3)
                (indices[t + 1], indices[t + 2]) = (indices[t + 2], indices[t + 1]);
            fixed (ushort* ptr = indices)
                _gl.BufferData(BufferTargetARB.ElementArrayBuffer, (nuint)(indices.Length * sizeof(ushort)), ptr, BufferUsageARB.StaticDraw);

            uint stride = 8 * sizeof(float);
            _gl.EnableVertexAttribArray(0);
            _gl.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, stride, (void*)0);
            _gl.EnableVertexAttribArray(1);
            _gl.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, stride, (void*)(3 * sizeof(float)));
            _gl.EnableVertexAttribArray(2);
            _gl.VertexAttribPointer(2, 2, VertexAttribPointerType.Float, false, stride, (void*)(6 * sizeof(float)));

            _gl.BindVertexArray(0);

            gb.IndexCount = (uint)indices.Length;
            _groups.Add(gb);
        }
    }

    private void LoadMaterialTextures()
    {
        if (_dataSource == null) return;

        int loaded = 0, failed = 0;
        for (int i = 0; i < _wmo.Materials.Count; i++)
        {
            var mat = _wmo.Materials[i];
            string? texName = mat.Texture1Name;
            if (string.IsNullOrEmpty(texName)) continue;

            // Ensure .blp extension
            if (!texName.EndsWith(".blp", StringComparison.OrdinalIgnoreCase))
                texName += ".blp";

            byte[]? blpData = null;

            // Try data source (MPQ)
            blpData = _dataSource.ReadFile(texName);

            // Try normalized slashes
            if (blpData == null)
                blpData = _dataSource.ReadFile(texName.Replace('/', '\\'));

            // Try case-insensitive via FindInFileSet
            if (blpData == null && _dataSource is MpqDataSource mpqDs)
            {
                var found = mpqDs.FindInFileSet(texName);
                if (found != null)
                    blpData = _dataSource.ReadFile(found);
            }

            if (blpData != null && blpData.Length > 0)
            {
                uint glTex = LoadWmoTexture(blpData, texName);
                if (glTex != 0)
                {
                    _materialTextures[i] = glTex;
                    loaded++;
                }
                else
                {
                    Console.WriteLine($"[WmoRenderer] Mat {i}: BLP decode failed for '{texName}'");
                    failed++;
                }
            }
            else
            {
                Console.WriteLine($"[WmoRenderer] Mat {i}: texture not found '{texName}'");
                failed++;
            }
        }
        Console.WriteLine($"[WmoRenderer] Textures: {loaded} loaded, {failed} failed out of {_wmo.Materials.Count} materials");
    }

    private unsafe uint LoadWmoTexture(byte[] blpData, string name)
    {
        try
        {
            using var ms = new MemoryStream(blpData);
            using var blp = new SereniaBLPLib.BlpFile(ms);
            var bmp = blp.GetBitmap(0);

            int w = bmp.Width, h = bmp.Height;
            var pixels = new byte[w * h * 4];
            var rect = new System.Drawing.Rectangle(0, 0, w, h);
            var data = bmp.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadOnly,
                System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            try
            {
                var srcBytes = new byte[data.Stride * h];
                System.Runtime.InteropServices.Marshal.Copy(data.Scan0, srcBytes, 0, srcBytes.Length);
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
            finally { bmp.UnlockBits(data); }
            bmp.Dispose();

            uint tex = _gl.GenTexture();
            _gl.BindTexture(TextureTarget.Texture2D, tex);
            fixed (byte* ptr = pixels)
                _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba,
                    (uint)w, (uint)h, 0, PixelFormat.Rgba, PixelType.UnsignedByte, ptr);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.LinearMipmapLinear);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.Repeat);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.Repeat);
            _gl.GenerateMipmap(TextureTarget.Texture2D);
            return tex;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[WmoRenderer] Failed to decode BLP {name}: {ex.Message}");
            return 0;
        }
    }

    private List<Vector3> GenerateNormals(WmoV14ToV17Converter.WmoGroupData group)
    {
        var normals = new Vector3[group.Vertices.Count];
        for (int i = 0; i + 2 < group.Indices.Count; i += 3)
        {
            int i0 = group.Indices[i], i1 = group.Indices[i + 1], i2 = group.Indices[i + 2];
            if (i0 >= group.Vertices.Count || i1 >= group.Vertices.Count || i2 >= group.Vertices.Count)
                continue;
            var e1 = group.Vertices[i1] - group.Vertices[i0];
            var e2 = group.Vertices[i2] - group.Vertices[i0];
            var n = Vector3.Normalize(Vector3.Cross(e1, e2));
            if (float.IsNaN(n.X)) continue;
            normals[i0] += n;
            normals[i1] += n;
            normals[i2] += n;
        }
        return normals.Select(n => n.Length() > 0.001f ? Vector3.Normalize(n) : Vector3.UnitY).ToList();
    }

    // --- Doodad loading ---

    private void ResolveDoodadNames()
    {
        // Parse null-terminated string table from DoodadNamesRaw
        // DoodadDef.NameIndex is a byte offset into this table
        _doodadNames.Clear();
        if (_wmo.DoodadNamesRaw.Length == 0) return;

        // Build offset→name map for quick lookup
        var raw = _wmo.DoodadNamesRaw;
        int start = 0;
        for (int i = 0; i <= raw.Length; i++)
        {
            if (i == raw.Length || raw[i] == 0)
            {
                if (i > start)
                {
                    // We don't store by index here — we'll resolve by offset in GetDoodadName
                }
                start = i + 1;
            }
        }
    }

    private string GetDoodadName(uint nameOffset)
    {
        if (nameOffset >= _wmo.DoodadNamesRaw.Length) return "";
        int end = (int)nameOffset;
        while (end < _wmo.DoodadNamesRaw.Length && _wmo.DoodadNamesRaw[end] != 0)
            end++;
        if (end == (int)nameOffset) return "";
        return Encoding.UTF8.GetString(_wmo.DoodadNamesRaw, (int)nameOffset, end - (int)nameOffset);
    }

    private void LoadActiveDoodadSet()
    {
        _doodadInstances.Clear();

        if (_wmo.DoodadSets.Count == 0 || _wmo.DoodadDefs.Count == 0)
            return;

        if (_activeDoodadSet >= _wmo.DoodadSets.Count)
            _activeDoodadSet = 0;

        var set = _wmo.DoodadSets[_activeDoodadSet];
        Console.WriteLine($"[WmoRenderer] Loading DoodadSet [{_activeDoodadSet}] \"{set.Name}\": {set.Count} doodads (start={set.StartIndex})");

        int loaded = 0, failed = 0;
        for (uint i = set.StartIndex; i < set.StartIndex + set.Count && i < (uint)_wmo.DoodadDefs.Count; i++)
        {
            var def = _wmo.DoodadDefs[(int)i];
            string modelPath = GetDoodadName(def.NameIndex);

            if (string.IsNullOrEmpty(modelPath))
            {
                failed++;
                continue;
            }

            // Build transform matrix: Scale * Rotation * Translation
            var transform = Matrix4x4.CreateScale(def.Scale)
                          * Matrix4x4.CreateFromQuaternion(def.Orientation)
                          * Matrix4x4.CreateTranslation(def.Position);

            // Get or load the MDX renderer for this model
            var renderer = GetOrLoadDoodadModel(modelPath);

            _doodadInstances.Add(new DoodadInstance
            {
                ModelPath = modelPath,
                Renderer = renderer,
                Transform = transform,
                Visible = true,
                DoodadDefIndex = (int)i
            });

            if (renderer != null) loaded++;
            else failed++;
        }

        Console.WriteLine($"[WmoRenderer] Doodads: {loaded} loaded, {failed} failed/missing, {_doodadModelCache.Count} unique models cached");
    }

    private MdxRenderer? GetOrLoadDoodadModel(string modelPath)
    {
        string normalized = modelPath.Replace('/', '\\').ToLowerInvariant();

        if (_doodadModelCache.TryGetValue(normalized, out var cached))
            return cached;

        MdxRenderer? renderer = null;
        try
        {
            byte[]? mdxData = null;

            // Try loading from data source (MPQ)
            if (_dataSource != null)
            {
                mdxData = _dataSource.ReadFile(modelPath);
                if (mdxData == null)
                    mdxData = _dataSource.ReadFile(normalized);
            }

            // Try loading from local disk
            if (mdxData == null)
            {
                string localPath = Path.Combine(_modelDir, Path.GetFileName(modelPath));
                if (File.Exists(localPath))
                    mdxData = File.ReadAllBytes(localPath);
            }

            if (mdxData != null && mdxData.Length > 0)
            {
                // Write to cache dir for MdxFile.Load (expects file path)
                Directory.CreateDirectory(_cacheDir);
                string cachePath = Path.Combine(_cacheDir, Path.GetFileName(modelPath));
                File.WriteAllBytes(cachePath, mdxData);

                var mdx = MdxFile.Load(cachePath);
                renderer = new MdxRenderer(_gl, mdx, _cacheDir, _dataSource, _texResolver, modelPath);
                Console.WriteLine($"  Doodad loaded: {Path.GetFileName(modelPath)} ({mdx.Geosets.Count} geosets)");
            }
            else
            {
                Console.WriteLine($"  Doodad not found: {modelPath}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  Doodad load failed: {modelPath} — {ex.Message}");
        }

        _doodadModelCache[normalized] = renderer;
        return renderer;
    }

    private void InitLiquidShader()
    {
        string vertSrc = @"
#version 330 core
layout(location = 0) in vec3 aPos;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;

out vec3 vWorldPos;

void main() {
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    vWorldPos = worldPos.xyz;
    gl_Position = uProj * uView * worldPos;
}
";
        string fragSrc = @"
#version 330 core
in vec3 vWorldPos;

uniform vec4 uColor;

out vec4 FragColor;

void main() {
    // Simple semi-transparent liquid with slight depth variation
    float depthShade = 0.85 + 0.15 * sin(vWorldPos.x * 0.5 + vWorldPos.y * 0.5);
    FragColor = vec4(uColor.rgb * depthShade, uColor.a);
}
";
        uint vert = CompileShader(ShaderType.VertexShader, vertSrc);
        uint frag = CompileShader(ShaderType.FragmentShader, fragSrc);

        _liquidShader = _gl.CreateProgram();
        _gl.AttachShader(_liquidShader, vert);
        _gl.AttachShader(_liquidShader, frag);
        _gl.LinkProgram(_liquidShader);

        _gl.GetProgram(_liquidShader, ProgramPropertyARB.LinkStatus, out int status);
        if (status == 0)
            Console.WriteLine($"[WmoRenderer] Liquid shader link error: {_gl.GetProgramInfoLog(_liquidShader)}");

        _gl.DeleteShader(vert);
        _gl.DeleteShader(frag);

        _gl.UseProgram(_liquidShader);
        _uLiqModel = _gl.GetUniformLocation(_liquidShader, "uModel");
        _uLiqView = _gl.GetUniformLocation(_liquidShader, "uView");
        _uLiqProj = _gl.GetUniformLocation(_liquidShader, "uProj");
        _uLiqColor = _gl.GetUniformLocation(_liquidShader, "uColor");
    }

    private unsafe void BuildLiquidMeshes()
    {
        for (int gi = 0; gi < _wmo.Groups.Count; gi++)
        {
            var group = _wmo.Groups[gi];
            if (group.LiquidData == null || group.LiquidData.Length < 30)
                continue;

            try
            {
                using var ms = new MemoryStream(group.LiquidData);
                using var reader = new BinaryReader(ms);

                // MLIQ header: C2iVector verts(8), C2iVector tiles(8), C3Vector corner(12), uint16 matId(2) = 30 bytes
                int xverts = reader.ReadInt32();
                int yverts = reader.ReadInt32();
                int xtiles = reader.ReadInt32();
                int ytiles = reader.ReadInt32();
                float cornerX = reader.ReadSingle();
                float cornerY = reader.ReadSingle();
                float cornerZ = reader.ReadSingle();
                ushort matId = reader.ReadUInt16();

                if (xverts <= 0 || yverts <= 0 || xverts > 256 || yverts > 256)
                {
                    Console.WriteLine($"[WmoRenderer] MLIQ group {gi}: invalid dimensions {xverts}x{yverts}, skipping");
                    continue;
                }

                int expectedVertBytes = xverts * yverts * 8;
                int expectedTileBytes = xtiles * ytiles;
                if (ms.Length - ms.Position < expectedVertBytes)
                {
                    Console.WriteLine($"[WmoRenderer] MLIQ group {gi}: not enough data for {xverts}x{yverts} verts");
                    continue;
                }

                // Read vertex heights (8 bytes per vertex: 4 bytes flow data + 4 bytes float height)
                float[] heights = new float[xverts * yverts];
                for (int v = 0; v < xverts * yverts; v++)
                {
                    reader.ReadInt32(); // flow/filler data (skip)
                    heights[v] = reader.ReadSingle();
                }

                // Read tile flags (1 byte per tile) — check for visible tiles
                byte[] tileFlags = new byte[xtiles * ytiles];
                if (ms.Length - ms.Position >= expectedTileBytes)
                {
                    for (int t = 0; t < xtiles * ytiles; t++)
                        tileFlags[t] = reader.ReadByte();
                }

                // Build vertex positions in WMO-local space
                // WMO MLIQ tiles are the same size as map chunk tiles (1/8th of chunk)
                // But for WMO liquids, the tile size is derived from the bounding area
                float tileW = (xtiles > 0) ? (group.BoundsMax.X - group.BoundsMin.X) / xtiles : 1.0f;
                float tileH = (ytiles > 0) ? (group.BoundsMax.Y - group.BoundsMin.Y) / ytiles : 1.0f;

                // Use corner position from MLIQ header for liquid placement
                // The wiki says liquid tile size = 1/8th of map chunk = ~4.1666 units
                // For WMO local space, use the corner + standard tile spacing
                float liquidTileSize = 4.16666f; // standard WoW liquid tile size

                var vertices = new float[xverts * yverts * 3];
                for (int vy = 0; vy < yverts; vy++)
                {
                    for (int vx = 0; vx < xverts; vx++)
                    {
                        int idx = vy * xverts + vx;
                        vertices[idx * 3 + 0] = cornerX + vx * liquidTileSize;
                        vertices[idx * 3 + 1] = cornerY + vy * liquidTileSize;
                        vertices[idx * 3 + 2] = heights[idx];
                    }
                }

                // Build indices: one quad per tile (2 triangles)
                var indices = new List<ushort>();
                for (int ty = 0; ty < ytiles; ty++)
                {
                    for (int tx = 0; tx < xtiles; tx++)
                    {
                        int tileIdx = ty * xtiles + tx;
                        // Skip tiles marked as hidden (bit 0x0F = legacyLiquidType, 0x40 = fishable, etc.)
                        // A tile with flags 0x0F (all type bits set) is typically "no liquid"
                        if ((tileFlags[tileIdx] & 0x0F) == 0x0F)
                            continue;

                        ushort tl = (ushort)(ty * xverts + tx);
                        ushort tr = (ushort)(ty * xverts + tx + 1);
                        ushort bl = (ushort)((ty + 1) * xverts + tx);
                        ushort br = (ushort)((ty + 1) * xverts + tx + 1);

                        indices.Add(tl); indices.Add(bl); indices.Add(tr);
                        indices.Add(tr); indices.Add(bl); indices.Add(br);
                    }
                }

                if (indices.Count == 0)
                {
                    Console.WriteLine($"[WmoRenderer] MLIQ group {gi}: no visible tiles");
                    continue;
                }

                // Determine liquid color from group flags
                float cr = 0.15f, cg = 0.35f, cb = 0.65f, ca = 0.55f; // water default
                bool isOcean = (group.Flags & 0x80000) != 0;
                if (isOcean) { cr = 0.10f; cg = 0.25f; cb = 0.55f; ca = 0.60f; }

                // Upload to GPU
                uint vao = _gl.GenVertexArray();
                uint vbo = _gl.GenBuffer();
                uint ebo = _gl.GenBuffer();

                _gl.BindVertexArray(vao);

                _gl.BindBuffer(BufferTargetARB.ArrayBuffer, vbo);
                fixed (float* ptr = vertices)
                    _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(vertices.Length * sizeof(float)), ptr, BufferUsageARB.StaticDraw);

                _gl.BindBuffer(BufferTargetARB.ElementArrayBuffer, ebo);
                var indexArr = indices.ToArray();
                fixed (ushort* ptr = indexArr)
                    _gl.BufferData(BufferTargetARB.ElementArrayBuffer, (nuint)(indexArr.Length * sizeof(ushort)), ptr, BufferUsageARB.StaticDraw);

                _gl.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 3 * sizeof(float), (void*)0);
                _gl.EnableVertexAttribArray(0);
                _gl.BindVertexArray(0);

                _liquidMeshes.Add(new LiquidMeshData
                {
                    Vao = vao, Vbo = vbo, Ebo = ebo,
                    IndexCount = (uint)indexArr.Length,
                    ColorR = cr, ColorG = cg, ColorB = cb, ColorA = ca
                });

                Console.WriteLine($"[WmoRenderer] MLIQ group {gi}: {xverts}x{yverts} verts, {xtiles}x{ytiles} tiles, {indices.Count / 3} tris, corner=({cornerX:F1},{cornerY:F1},{cornerZ:F1})");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[WmoRenderer] MLIQ group {gi}: parse error — {ex.Message}");
            }
        }

        if (_liquidMeshes.Count > 0)
            Console.WriteLine($"[WmoRenderer] Built {_liquidMeshes.Count} liquid meshes");
    }

    public void Dispose()
    {
        foreach (var gb in _groups)
        {
            _gl.DeleteVertexArray(gb.Vao);
            _gl.DeleteBuffer(gb.Vbo);
            _gl.DeleteBuffer(gb.Ebo);
        }

        // Delete material textures
        foreach (var tex in _materialTextures.Values)
            _gl.DeleteTexture(tex);
        _materialTextures.Clear();

        // Dispose liquid meshes
        foreach (var liq in _liquidMeshes)
        {
            _gl.DeleteVertexArray(liq.Vao);
            _gl.DeleteBuffer(liq.Vbo);
            _gl.DeleteBuffer(liq.Ebo);
        }
        _liquidMeshes.Clear();

        // Dispose cached doodad renderers
        foreach (var renderer in _doodadModelCache.Values)
            renderer?.Dispose();
        _doodadModelCache.Clear();
        _doodadInstances.Clear();

        _gl.DeleteProgram(_shaderProgram);
        _gl.DeleteProgram(_liquidShader);
    }

    private class GroupBuffers
    {
        public int GroupIndex;
        public uint Vao, Vbo, Ebo;
        public uint IndexCount;
        public bool Visible = true;
    }

    private class DoodadInstance
    {
        public string ModelPath = "";
        public MdxRenderer? Renderer;
        public Matrix4x4 Transform;
        public bool Visible = true;
        public int DoodadDefIndex;
    }

    private class LiquidMeshData
    {
        public uint Vao, Vbo, Ebo;
        public uint IndexCount;
        public float ColorR, ColorG, ColorB, ColorA;
    }
}
