using System.Numerics;
using MdxViewer.Logging;
using Silk.NET.OpenGL;
using WowViewer.Core.Runtime.M2;

namespace MdxViewer.Rendering;

public sealed class M2Renderer : IModelRenderer
{
    private readonly GL? _gl;
    private readonly MdxRenderer? _legacyRenderer;
    private readonly M2StaticRenderModel? _runtimeModel;
    private readonly List<SectionBuffers> _sections = new();
    private readonly List<bool> _sectionVisibility = new();
    private bool _wireframe;
    private bool _batchStateValid;
    private Matrix4x4 _batchView;
    private Matrix4x4 _batchProj;
    private Vector3 _batchFogColor;
    private float _batchFogStart;
    private float _batchFogEnd;
    private Vector3 _batchCameraPos;
    private Vector3 _batchLightDir;
    private Vector3 _batchLightColor;
    private Vector3 _batchAmbientColor;

    private static uint _shaderProgram;
    private static int _uModel;
    private static int _uView;
    private static int _uProj;
    private static int _uFogColor;
    private static int _uFogStart;
    private static int _uFogEnd;
    private static int _uCameraPos;
    private static int _uLightDir;
    private static int _uLightColor;
    private static int _uAmbientColor;
    private static int _uBaseColor;
    private static int _uUnshaded;
    private static bool _shaderInitialized;
    private static int _shaderRefCount;

    public M2Renderer(MdxRenderer innerRenderer, string sourceModelPath)
    {
        ArgumentNullException.ThrowIfNull(innerRenderer);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourceModelPath);

        _legacyRenderer = innerRenderer;
        SourceModelPath = sourceModelPath.Replace('/', '\\');
    }

    public M2Renderer(MdxRenderer innerRenderer, M2StaticRenderModel runtimeModel, string sourceModelPath)
    {
        ArgumentNullException.ThrowIfNull(innerRenderer);
        ArgumentNullException.ThrowIfNull(runtimeModel);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourceModelPath);

        _legacyRenderer = innerRenderer;
        _runtimeModel = runtimeModel;
        SourceModelPath = sourceModelPath.Replace('/', '\\');

        for (int index = 0; index < runtimeModel.Sections.Count; index++)
            _sectionVisibility.Add(true);

        ViewerLog.Info(
            ViewerLog.Category.Mdx,
            $"[M2] wow-viewer runtime metadata + legacy draw backend ready for {Path.GetFileName(SourceModelPath)}: sections={runtimeModel.Sections.Count}, compatibilityFallback={runtimeModel.UsesCompatibilityFallback}");
    }

    public M2Renderer(GL gl, M2StaticRenderModel runtimeModel, string sourceModelPath)
    {
        ArgumentNullException.ThrowIfNull(gl);
        ArgumentNullException.ThrowIfNull(runtimeModel);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourceModelPath);

        _gl = gl;
        _runtimeModel = runtimeModel;
        SourceModelPath = sourceModelPath.Replace('/', '\\');

        for (int index = 0; index < runtimeModel.Sections.Count; index++)
            _sectionVisibility.Add(true);

        InitShaders();
        InitBuffers();

        ViewerLog.Info(
            ViewerLog.Category.Mdx,
            $"[M2] wow-viewer static runtime ready for {Path.GetFileName(SourceModelPath)}: sections={_sections.Count}, compatibilityFallback={runtimeModel.UsesCompatibilityFallback}");
    }

    public string SourceModelPath { get; }

    public bool UsesCompatibilityFallback => _legacyRenderer != null || (_runtimeModel?.UsesCompatibilityFallback ?? false);

    public Vector3 BoundsMin => _legacyRenderer?.BoundsMin ?? _runtimeModel!.BoundsMin;

    public Vector3 BoundsMax => _legacyRenderer?.BoundsMax ?? _runtimeModel!.BoundsMax;

    public bool RequiresUnbatchedWorldRender => true;

    public MdxAnimator? Animator => _legacyRenderer?.Animator;

    public int SubObjectCount => _runtimeModel?.Sections.Count ?? _legacyRenderer?.SubObjectCount ?? _sections.Count;

    public void Render(Matrix4x4 view, Matrix4x4 proj)
    {
        if (_legacyRenderer != null)
        {
            _legacyRenderer.Render(view, proj);
            return;
        }

        RenderWithTransform(Matrix4x4.Identity, view, proj);
    }

    public void ToggleWireframe()
    {
        if (_legacyRenderer != null)
        {
            _legacyRenderer.ToggleWireframe();
            return;
        }

        _wireframe = !_wireframe;
    }

    public string GetSubObjectName(int index)
    {
        if (_runtimeModel != null && index >= 0 && index < _runtimeModel.Sections.Count)
            return $"Geoset {_runtimeModel.Sections[index].SkinSectionId}";

        if (_legacyRenderer != null)
            return _legacyRenderer.GetSubObjectName(index);

        return index >= 0 && index < _sections.Count
            ? $"Geoset {_sections[index].SkinSectionId}"
            : string.Empty;
    }

    public bool GetSubObjectVisible(int index)
    {
        if (_runtimeModel != null && index >= 0 && index < _sectionVisibility.Count)
            return _sectionVisibility[index];

        if (_legacyRenderer != null)
            return _legacyRenderer.GetSubObjectVisible(index);

        return index >= 0 && index < _sections.Count && _sections[index].Visible;
    }

    public void SetSubObjectVisible(int index, bool visible)
    {
        if (_runtimeModel != null && index >= 0 && index < _sectionVisibility.Count)
            _sectionVisibility[index] = visible;

        if (_legacyRenderer != null)
        {
            if (index >= 0 && index < _legacyRenderer.SubObjectCount)
                _legacyRenderer.SetSubObjectVisible(index, visible);

            return;
        }

        if (index >= 0 && index < _sections.Count)
            _sections[index].Visible = visible;
    }

    public void UpdateAnimation()
    {
        _legacyRenderer?.UpdateAnimation();
    }

    public void ApplyTextureSamplingSettings()
    {
        _legacyRenderer?.ApplyTextureSamplingSettings();
    }

    public void BeginBatch(
        Matrix4x4 view,
        Matrix4x4 proj,
        Vector3 fogColor,
        float fogStart,
        float fogEnd,
        Vector3 cameraPos,
        Vector3 lightDir,
        Vector3 lightColor,
        Vector3 ambientColor)
    {
        if (_legacyRenderer != null)
        {
            _legacyRenderer.BeginBatch(view, proj, fogColor, fogStart, fogEnd, cameraPos, lightDir, lightColor, ambientColor);
            return;
        }

        _batchView = view;
        _batchProj = proj;
        _batchFogColor = fogColor;
        _batchFogStart = fogStart;
        _batchFogEnd = fogEnd;
        _batchCameraPos = cameraPos;
        _batchLightDir = lightDir;
        _batchLightColor = lightColor;
        _batchAmbientColor = ambientColor;
        _batchStateValid = true;
    }

    public void RenderInstance(Matrix4x4 modelMatrix, RenderPass pass, float fadeAlpha = 1.0f)
    {
        if (_legacyRenderer != null)
        {
            _legacyRenderer.RenderInstance(modelMatrix, pass, fadeAlpha);
            return;
        }

        if (!_batchStateValid)
            return;

        RenderCore(modelMatrix, _batchView, _batchProj, pass, fadeAlpha, _batchFogColor, _batchFogStart, _batchFogEnd, _batchCameraPos, _batchLightDir, _batchLightColor, _batchAmbientColor, backdrop: false);
    }

    public void RenderWithTransform(
        Matrix4x4 modelMatrix,
        Matrix4x4 view,
        Matrix4x4 proj,
        RenderPass pass = RenderPass.Both,
        float fadeAlpha = 1.0f,
        Vector3? fogColor = null,
        float fogStart = 200f,
        float fogEnd = 1500f,
        Vector3? cameraPos = null,
        Vector3? lightDir = null,
        Vector3? lightColor = null,
        Vector3? ambientColor = null)
    {
        if (_legacyRenderer != null)
        {
            _legacyRenderer.RenderWithTransform(modelMatrix, view, proj, pass, fadeAlpha, fogColor, fogStart, fogEnd, cameraPos, lightDir, lightColor, ambientColor);
            return;
        }

        RenderCore(
            modelMatrix,
            view,
            proj,
            pass,
            fadeAlpha,
            fogColor ?? new Vector3(0.6f, 0.7f, 0.85f),
            fogStart,
            fogEnd,
            cameraPos ?? Vector3.Zero,
            lightDir ?? Vector3.Normalize(new Vector3(0.5f, 0.3f, 1.0f)),
            lightColor ?? new Vector3(1.0f, 0.95f, 0.85f),
            ambientColor ?? new Vector3(0.35f, 0.35f, 0.4f),
            backdrop: false);
    }

    public void RenderBackdrop(
        Matrix4x4 modelMatrix,
        Matrix4x4 view,
        Matrix4x4 proj,
        Vector3 fogColor,
        float fogStart,
        float fogEnd,
        Vector3 cameraPos,
        Vector3 lightDir,
        Vector3 lightColor,
        Vector3 ambientColor)
    {
        if (_legacyRenderer != null)
        {
            _legacyRenderer.RenderBackdrop(modelMatrix, view, proj, fogColor, fogStart, fogEnd, cameraPos, lightDir, lightColor, ambientColor);
            return;
        }

        RenderCore(modelMatrix, view, proj, RenderPass.Both, 1.0f, fogColor, fogStart, fogEnd, cameraPos, lightDir, lightColor, ambientColor, backdrop: true);
    }

    public void RenderWireframeOverlay(
        Matrix4x4 modelMatrix,
        Matrix4x4 view,
        Matrix4x4 proj,
        Vector3? fogColor = null,
        float fogStart = 200f,
        float fogEnd = 1500f,
        Vector3? cameraPos = null,
        Vector3? lightDir = null,
        Vector3? lightColor = null,
        Vector3? ambientColor = null)
    {
        if (_legacyRenderer != null)
        {
            _legacyRenderer.RenderWireframeOverlay(modelMatrix, view, proj, fogColor, fogStart, fogEnd, cameraPos, lightDir, lightColor, ambientColor);
            return;
        }

        bool previousWireframe = _wireframe;
        _wireframe = true;
        try
        {
            RenderWithTransform(modelMatrix, view, proj, RenderPass.Both, 1.0f, fogColor, fogStart, fogEnd, cameraPos, lightDir, lightColor, ambientColor);
        }
        finally
        {
            _wireframe = previousWireframe;
        }
    }

    public void Dispose()
    {
        if (_legacyRenderer != null)
        {
            _legacyRenderer.Dispose();
            return;
        }

        if (_gl == null)
            return;

        foreach (SectionBuffers section in _sections)
        {
            _gl.DeleteVertexArray(section.Vao);
            _gl.DeleteBuffer(section.Vbo);
            _gl.DeleteBuffer(section.Ebo);
        }

        _sections.Clear();

        _shaderRefCount--;
        if (_shaderRefCount <= 0 && _shaderProgram != 0)
        {
            _gl.DeleteProgram(_shaderProgram);
            _shaderProgram = 0;
            _shaderInitialized = false;
            _shaderRefCount = 0;
        }
    }

    private void InitBuffers()
    {
        if (_gl == null || _runtimeModel == null)
            return;

        foreach (M2StaticRenderSection section in _runtimeModel.Sections)
        {
            float[] vertexData = new float[section.Vertices.Count * 6];
            for (int index = 0; index < section.Vertices.Count; index++)
            {
                M2StaticRenderVertex vertex = section.Vertices[index];
                int offset = index * 6;
                vertexData[offset + 0] = vertex.Position.X;
                vertexData[offset + 1] = vertex.Position.Y;
                vertexData[offset + 2] = vertex.Position.Z;
                vertexData[offset + 3] = vertex.Normal.X;
                vertexData[offset + 4] = vertex.Normal.Y;
                vertexData[offset + 5] = vertex.Normal.Z;
            }

            uint[] indices = section.Indices.ToArray();
            uint vao = _gl.GenVertexArray();
            uint vbo = _gl.GenBuffer();
            uint ebo = _gl.GenBuffer();

            _gl.BindVertexArray(vao);
            _gl.BindBuffer(BufferTargetARB.ArrayBuffer, vbo);
            unsafe
            {
                fixed (float* vertexPtr = vertexData)
                {
                    _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(vertexData.Length * sizeof(float)), vertexPtr, BufferUsageARB.StaticDraw);
                }

                _gl.BindBuffer(BufferTargetARB.ElementArrayBuffer, ebo);
                fixed (uint* indexPtr = indices)
                {
                    _gl.BufferData(BufferTargetARB.ElementArrayBuffer, (nuint)(indices.Length * sizeof(uint)), indexPtr, BufferUsageARB.StaticDraw);
                }

                _gl.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 6u * sizeof(float), (void*)0);
                _gl.EnableVertexAttribArray(0);
                _gl.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, 6u * sizeof(float), (void*)(3 * sizeof(float)));
                _gl.EnableVertexAttribArray(1);
            }

            _gl.BindVertexArray(0);

            _sections.Add(new SectionBuffers(section.SectionIndex, section.SkinSectionId, vao, vbo, ebo, (uint)indices.Length, section.Material));
        }
    }

    private unsafe void RenderCore(
        Matrix4x4 modelMatrix,
        Matrix4x4 view,
        Matrix4x4 proj,
        RenderPass pass,
        float fadeAlpha,
        Vector3 fogColor,
        float fogStart,
        float fogEnd,
        Vector3 cameraPos,
        Vector3 lightDir,
        Vector3 lightColor,
        Vector3 ambientColor,
        bool backdrop)
    {
        if (_gl == null)
            return;

        _gl.UseProgram(_shaderProgram);
        _gl.UniformMatrix4(_uModel, 1, false, (float*)&modelMatrix);
        _gl.UniformMatrix4(_uView, 1, false, (float*)&view);
        _gl.UniformMatrix4(_uProj, 1, false, (float*)&proj);
        _gl.Uniform3(_uFogColor, fogColor.X, fogColor.Y, fogColor.Z);
        _gl.Uniform1(_uFogStart, fogStart);
        _gl.Uniform1(_uFogEnd, fogEnd);
        _gl.Uniform3(_uCameraPos, cameraPos.X, cameraPos.Y, cameraPos.Z);
        _gl.Uniform3(_uLightDir, lightDir.X, lightDir.Y, lightDir.Z);
        _gl.Uniform3(_uLightColor, lightColor.X, lightColor.Y, lightColor.Z);
        _gl.Uniform3(_uAmbientColor, ambientColor.X, ambientColor.Y, ambientColor.Z);

        if (backdrop)
        {
            _gl.Disable(EnableCap.DepthTest);
            _gl.DepthMask(false);
        }
        else
        {
            _gl.Enable(EnableCap.DepthTest);
            _gl.DepthFunc(DepthFunction.Lequal);
        }

        _gl.PolygonMode(TriangleFace.FrontAndBack, _wireframe ? PolygonMode.Line : PolygonMode.Fill);

        foreach (SectionBuffers section in _sections)
        {
            if (!section.Visible)
                continue;

            bool transparent = section.Material.IsTransparent;
            if (pass == RenderPass.Opaque && transparent)
                continue;
            if (pass == RenderPass.Transparent && !transparent)
                continue;

            if (section.Material.IsTwoSided || backdrop)
            {
                _gl.Disable(EnableCap.CullFace);
            }
            else
            {
                _gl.Enable(EnableCap.CullFace);
                _gl.CullFace(TriangleFace.Back);
            }

            if (!backdrop && transparent)
            {
                _gl.Enable(EnableCap.Blend);
                ConfigureBlendMode(section.Material.BlendMode);
                _gl.DepthMask(false);
            }
            else
            {
                _gl.Disable(EnableCap.Blend);
                _gl.DepthMask(!backdrop);
            }

            Vector3 baseColor = ComputeSectionColor(section.SectionIndex, section.Material, fadeAlpha);
            _gl.Uniform3(_uBaseColor, baseColor.X, baseColor.Y, baseColor.Z);
            _gl.Uniform1(_uUnshaded, section.Material.IsUnshaded ? 1 : 0);

            _gl.BindVertexArray(section.Vao);
            _gl.DrawElements(PrimitiveType.Triangles, section.IndexCount, DrawElementsType.UnsignedInt, null);
        }

        _gl.BindVertexArray(0);
        _gl.Disable(EnableCap.Blend);
        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Lequal);
        _gl.DepthMask(true);
        _gl.PolygonMode(TriangleFace.FrontAndBack, PolygonMode.Fill);
    }

    private void ConfigureBlendMode(WowViewer.Core.M2.M2BlendMode blendMode)
    {
        if (_gl == null)
            return;

        switch (blendMode)
        {
            case WowViewer.Core.M2.M2BlendMode.Add:
            case WowViewer.Core.M2.M2BlendMode.NoAlphaAdd:
            case WowViewer.Core.M2.M2BlendMode.BlendAdd:
                _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.One);
                break;

            case WowViewer.Core.M2.M2BlendMode.Mod:
            case WowViewer.Core.M2.M2BlendMode.Mod2X:
                _gl.BlendFunc(BlendingFactor.DstColor, BlendingFactor.Zero);
                break;

            default:
                _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
                break;
        }
    }

    private static Vector3 ComputeSectionColor(int sectionIndex, M2StaticRenderMaterial material, float fadeAlpha)
    {
        float tintSeed = ((sectionIndex * 37) % 100) / 100f;
        Vector3 baseColor = material.IsTransparent
            ? new Vector3(0.72f, 0.83f, 0.98f)
            : new Vector3(0.84f, 0.84f, 0.82f);
        Vector3 tint = new(0.08f * tintSeed, 0.04f * (1f - tintSeed), 0.06f * (0.5f - tintSeed));
        return Vector3.Clamp((baseColor + tint) * Math.Clamp(fadeAlpha, 0.1f, 1.0f), Vector3.Zero, Vector3.One);
    }

    private void InitShaders()
    {
        if (_gl == null)
            return;

        _shaderRefCount++;
        if (_shaderInitialized)
            return;

        const string vertexSource = """
            #version 330 core
            layout (location = 0) in vec3 aPos;
            layout (location = 1) in vec3 aNormal;

            uniform mat4 uModel;
            uniform mat4 uView;
            uniform mat4 uProj;

            out vec3 vWorldPos;
            out vec3 vNormal;

            void main()
            {
                vec4 worldPos = uModel * vec4(aPos, 1.0);
                vWorldPos = worldPos.xyz;
                vNormal = normalize(mat3(uModel) * aNormal);
                gl_Position = uProj * uView * worldPos;
            }
            """;

        const string fragmentSource = """
            #version 330 core
            in vec3 vWorldPos;
            in vec3 vNormal;

            uniform vec3 uFogColor;
            uniform float uFogStart;
            uniform float uFogEnd;
            uniform vec3 uCameraPos;
            uniform vec3 uLightDir;
            uniform vec3 uLightColor;
            uniform vec3 uAmbientColor;
            uniform vec3 uBaseColor;
            uniform int uUnshaded;

            out vec4 FragColor;

            void main()
            {
                float diffuseStrength = uUnshaded == 1 ? 1.0 : max(dot(normalize(vNormal), normalize(uLightDir)), 0.0);
                vec3 litColor = uBaseColor * (uAmbientColor + (uLightColor * diffuseStrength));
                float distanceToCamera = distance(vWorldPos, uCameraPos);
                float fogRange = max(uFogEnd - uFogStart, 0.001);
                float fogFactor = clamp((uFogEnd - distanceToCamera) / fogRange, 0.0, 1.0);
                vec3 finalColor = mix(uFogColor, litColor, fogFactor);
                FragColor = vec4(finalColor, 1.0);
            }
            """;

        uint vertexShader = CompileShader(ShaderType.VertexShader, vertexSource);
        uint fragmentShader = CompileShader(ShaderType.FragmentShader, fragmentSource);

        _shaderProgram = _gl.CreateProgram();
        _gl.AttachShader(_shaderProgram, vertexShader);
        _gl.AttachShader(_shaderProgram, fragmentShader);
        _gl.LinkProgram(_shaderProgram);
        _gl.GetProgram(_shaderProgram, ProgramPropertyARB.LinkStatus, out int status);
        if (status == 0)
            throw new InvalidOperationException($"Failed to link M2 runtime shader: {_gl.GetProgramInfoLog(_shaderProgram)}");

        _gl.DeleteShader(vertexShader);
        _gl.DeleteShader(fragmentShader);

        _uModel = _gl.GetUniformLocation(_shaderProgram, "uModel");
        _uView = _gl.GetUniformLocation(_shaderProgram, "uView");
        _uProj = _gl.GetUniformLocation(_shaderProgram, "uProj");
        _uFogColor = _gl.GetUniformLocation(_shaderProgram, "uFogColor");
        _uFogStart = _gl.GetUniformLocation(_shaderProgram, "uFogStart");
        _uFogEnd = _gl.GetUniformLocation(_shaderProgram, "uFogEnd");
        _uCameraPos = _gl.GetUniformLocation(_shaderProgram, "uCameraPos");
        _uLightDir = _gl.GetUniformLocation(_shaderProgram, "uLightDir");
        _uLightColor = _gl.GetUniformLocation(_shaderProgram, "uLightColor");
        _uAmbientColor = _gl.GetUniformLocation(_shaderProgram, "uAmbientColor");
        _uBaseColor = _gl.GetUniformLocation(_shaderProgram, "uBaseColor");
        _uUnshaded = _gl.GetUniformLocation(_shaderProgram, "uUnshaded");
        _shaderInitialized = true;
    }

    private uint CompileShader(ShaderType shaderType, string source)
    {
        if (_gl == null)
            return 0;

        uint shader = _gl.CreateShader(shaderType);
        _gl.ShaderSource(shader, source);
        _gl.CompileShader(shader);
        _gl.GetShader(shader, ShaderParameterName.CompileStatus, out int status);
        if (status == 0)
            throw new InvalidOperationException($"Failed to compile M2 runtime shader ({shaderType}): {_gl.GetShaderInfoLog(shader)}");

        return shader;
    }

    private sealed class SectionBuffers
    {
        public SectionBuffers(int sectionIndex, ushort skinSectionId, uint vao, uint vbo, uint ebo, uint indexCount, M2StaticRenderMaterial material)
        {
            SectionIndex = sectionIndex;
            SkinSectionId = skinSectionId;
            Vao = vao;
            Vbo = vbo;
            Ebo = ebo;
            IndexCount = indexCount;
            Material = material;
        }

        public int SectionIndex { get; }

        public ushort SkinSectionId { get; }

        public uint Vao { get; }

        public uint Vbo { get; }

        public uint Ebo { get; }

        public uint IndexCount { get; }

        public M2StaticRenderMaterial Material { get; }

        public bool Visible { get; set; } = true;
    }
}