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

    // Doodad support
    private readonly Dictionary<string, MdxRenderer?> _doodadModelCache = new(StringComparer.OrdinalIgnoreCase);
    private readonly List<DoodadInstance> _doodadInstances = new();
    private readonly List<string> _doodadNames = new(); // resolved from MODN
    private int _activeDoodadSet = 0;
    private bool _doodadsVisible = true;
    private readonly string _cacheDir;

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
        InitBuffers();
        ResolveDoodadNames();
        LoadActiveDoodadSet();
    }

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

        _gl.Uniform1(_uHasTexture, 0); // No textures yet for WMO viewer

        foreach (var gb in _groups)
        {
            if (!gb.Visible) continue;
            float r = ((gb.GroupIndex * 67 + 13) % 255) / 255f;
            float g = ((gb.GroupIndex * 131 + 7) % 255) / 255f;
            float b = ((gb.GroupIndex * 43 + 29) % 255) / 255f;
            _gl.Uniform4(_uColor, r, g, b, 1.0f);

            _gl.BindVertexArray(gb.Vao);
            _gl.DrawElements(PrimitiveType.Triangles, gb.IndexCount, DrawElementsType.UnsignedShort, null);
            _gl.BindVertexArray(0);
        }

        _gl.PolygonMode(TriangleFace.FrontAndBack, PolygonMode.Fill);

        // 2. Render doodad instances
        if (_doodadsVisible)
        {
            foreach (var inst in _doodadInstances)
            {
                if (!inst.Visible || inst.Renderer == null) continue;
                inst.Renderer.RenderWithTransform(inst.Transform, view, proj);
            }
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
        if (texColor.a < 0.01) discard;
    } else {
        texColor = uColor;
    }

    FragColor = vec4(texColor.rgb * lighting, texColor.a);
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

    public void Dispose()
    {
        foreach (var gb in _groups)
        {
            _gl.DeleteVertexArray(gb.Vao);
            _gl.DeleteBuffer(gb.Vbo);
            _gl.DeleteBuffer(gb.Ebo);
        }

        // Dispose cached doodad renderers
        foreach (var renderer in _doodadModelCache.Values)
            renderer?.Dispose();
        _doodadModelCache.Clear();
        _doodadInstances.Clear();

        _gl.DeleteProgram(_shaderProgram);
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
}
