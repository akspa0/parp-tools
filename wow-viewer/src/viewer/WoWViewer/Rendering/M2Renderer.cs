using System.Drawing;
using System.Drawing.Imaging;
using System.Numerics;
using WoWViewer.DataSources;
using WoWViewer.Logging;
using SereniaBLPLib;
using Silk.NET.OpenGL;
using WowViewer.Core.M2;
using WowViewer.Core.IO.M2;
using WowViewer.Core.Runtime.M2;

namespace WoWViewer.Rendering;

public sealed class M2Renderer : IModelRenderer
{
    private readonly GL? _gl;
    private readonly IDataSource? _dataSource;
    private readonly ReplaceableTextureResolver? _texResolver;
    private readonly MdxRenderer? _legacyRenderer;
    private readonly M2StaticRenderModel? _runtimeModel;
    private readonly M2RuntimeAnimator? _runtimeAnimator;
    private readonly List<SectionBuffers> _sections = new();
    private readonly Dictionary<int, SectionBuffers> _sectionsByIndex = new();
    private readonly Dictionary<int, M2StaticRenderSection> _staticSectionsByIndex = new();
    private readonly List<bool> _sectionVisibility = new();
    private readonly Dictionary<string, uint> _loadedTextureCache = new(StringComparer.OrdinalIgnoreCase);
    private readonly HashSet<uint> _ownedTextureIds = new();
    private readonly string _modelDir = string.Empty;
    private readonly int? _selectedReplaceableDisplayIndex;
    private int _characterHairVariationId;
    private int _characterFacialHairVariationId;
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
    private DateTime _lastAnimationUpdateTime = DateTime.UtcNow;

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
    private static int _uHasTexture;
    private static int _uUvSet;
    private static int _uGeneratedTexCoord;
    private static int _uTexture0;
    private static int _uAlphaCutout;
    private static int _uAlpha;
    private static int _uHasUvTransform;
    private static int _uUvTranslation;
    private static int _uUvScale;
    private static int _uUvRotation;
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

    public M2Renderer(GL gl, M2StaticRenderModel runtimeModel, string sourceModelPath, IDataSource? dataSource = null, ReplaceableTextureResolver? texResolver = null)
    {
        ArgumentNullException.ThrowIfNull(gl);
        ArgumentNullException.ThrowIfNull(runtimeModel);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourceModelPath);

        _gl = gl;
        _dataSource = dataSource;
        _texResolver = texResolver;
        _runtimeModel = runtimeModel;
        _runtimeAnimator = runtimeModel.Model.SequenceCount > 0 ? new M2RuntimeAnimator(runtimeModel.Model, dataSource) : null;
        SourceModelPath = sourceModelPath.Replace('/', '\\');
        _modelDir = Path.GetDirectoryName(SourceModelPath)?.Replace('/', '\\') ?? string.Empty;
        _selectedReplaceableDisplayIndex = SelectBestReplaceableDisplayIndex(runtimeModel, SourceModelPath, texResolver);

        foreach (M2StaticRenderSection section in runtimeModel.Sections)
            _staticSectionsByIndex[section.SectionIndex] = section;

        for (int index = 0; index < runtimeModel.Sections.Count; index++)
            _sectionVisibility.Add(true);

        InitShaders();
        InitBuffers();
        LoadSectionTextures();

        ViewerLog.Info(
            ViewerLog.Category.Mdx,
            $"[M2] wow-viewer static runtime ready for {Path.GetFileName(SourceModelPath)}: sections={_sections.Count}, compatibilityFallback={runtimeModel.UsesCompatibilityFallback}");

        if (_runtimeAnimator?.HasAnimation == true)
        {
            ViewerLog.Info(
                ViewerLog.Category.Mdx,
                $"[M2] Runtime animation enabled for {Path.GetFileName(SourceModelPath)}: sequences={_runtimeAnimator.Sequences.Count}, bones={runtimeModel.Model.BoneCount}");
        }
    }

    public string SourceModelPath { get; }

    public bool UsesCompatibilityFallback => _legacyRenderer != null || (_runtimeModel?.UsesCompatibilityFallback ?? false);

    public Vector3 BoundsMin => _runtimeModel?.BoundsMin ?? _legacyRenderer?.BoundsMin ?? Vector3.Zero;

    public Vector3 BoundsMax => _runtimeModel?.BoundsMax ?? _legacyRenderer?.BoundsMax ?? Vector3.Zero;

    public bool HasTransparentWorldPass
    {
        get
        {
            if (_legacyRenderer != null)
                return _legacyRenderer.HasTransparentWorldPass;

            for (int index = 0; index < _sections.Count; index++)
            {
                SectionBuffers section = _sections[index];
                if (section.Visible && section.Material.IsTransparent)
                    return true;
            }

            return false;
        }
    }

    public bool RequiresUnbatchedWorldRender => true;

    public IAnimationController? Animator => _legacyRenderer?.Animator ?? _runtimeAnimator;

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

    public bool TryApplyCharacterSelectionGroups(IReadOnlyCollection<uint>? wantedGroups, string? reasonLabel = null)
    {
        if (_legacyRenderer != null)
            return _legacyRenderer.TryApplyCharacterSelectionGroups(wantedGroups, reasonLabel);

        if (_runtimeModel == null || wantedGroups == null || wantedGroups.Count == 0 || _sections.Count == 0)
            return false;

        ApplyCharacterSelectionGroups(wantedGroups, reasonLabel);
        return true;
    }

    public bool TryApplyCharacterCustomization(IReadOnlyCollection<uint>? wantedGroups, int? hairVariationId = null, int? facialHairVariationId = null, string? reasonLabel = null)
    {
        if (_legacyRenderer != null)
            return _legacyRenderer.TryApplyCharacterCustomization(wantedGroups, hairVariationId, facialHairVariationId, reasonLabel);

        if (!TryApplyCharacterSelectionGroups(wantedGroups, reasonLabel))
            return false;

        if (!UpdateCharacterTextureVariationState(hairVariationId, facialHairVariationId))
            return true;

        ReloadCharacterTextures();
        return true;
    }

    public void UpdateAnimation()
    {
        if (_legacyRenderer != null)
        {
            _legacyRenderer.UpdateAnimation();
            return;
        }

        if (_runtimeAnimator == null || _runtimeModel == null || _gl == null)
            return;

        DateTime now = DateTime.UtcNow;
        float deltaMs = (float)(now - _lastAnimationUpdateTime).TotalMilliseconds;
        _lastAnimationUpdateTime = now;
        _runtimeAnimator.Update(Math.Clamp(deltaMs, 0.0f, 100.0f));

        int sequenceIndex = _runtimeAnimator.CurrentSequence;
        int timeMs = _runtimeAnimator.GetCurrentTimeMs();
        M2ExternalAnimationRuntimeState? externalAnimationState = _runtimeAnimator.ResolveExternalAnimationState();
        M2AnimatedRenderState animatedState = M2AnimatedRenderStateEvaluator.Evaluate(_runtimeModel.Model, _runtimeModel, sequenceIndex, timeMs, externalAnimationState);
        M2BonePoseState bonePoseState = M2BonePoseEvaluator.Evaluate(_runtimeModel.Model, sequenceIndex, timeMs, externalAnimationState);
        M2SkinnedRenderModel skinnedRenderModel = M2SkinnedRenderModelBuilder.ApplyPose(_runtimeModel, bonePoseState);
        M2RenderConsumerFrameState consumerState = M2RenderConsumerFrameStateBuilder.Build(_runtimeModel, animatedState);
        ApplyAnimatedFrame(skinnedRenderModel, consumerState);
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

        ReleaseOwnedTextures();

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
            float[] vertexData = new float[section.Vertices.Count * 10];
            for (int index = 0; index < section.Vertices.Count; index++)
            {
                M2StaticRenderVertex vertex = section.Vertices[index];
                int offset = index * 10;
                vertexData[offset + 0] = vertex.Position.X;
                vertexData[offset + 1] = vertex.Position.Y;
                vertexData[offset + 2] = vertex.Position.Z;
                vertexData[offset + 3] = vertex.Normal.X;
                vertexData[offset + 4] = vertex.Normal.Y;
                vertexData[offset + 5] = vertex.Normal.Z;
                vertexData[offset + 6] = vertex.TextureCoords0.X;
                vertexData[offset + 7] = vertex.TextureCoords0.Y;
                vertexData[offset + 8] = vertex.TextureCoords1.X;
                vertexData[offset + 9] = vertex.TextureCoords1.Y;
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
                    _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(vertexData.Length * sizeof(float)), vertexPtr, _runtimeAnimator != null ? BufferUsageARB.DynamicDraw : BufferUsageARB.StaticDraw);
                }

                _gl.BindBuffer(BufferTargetARB.ElementArrayBuffer, ebo);
                fixed (uint* indexPtr = indices)
                {
                    _gl.BufferData(BufferTargetARB.ElementArrayBuffer, (nuint)(indices.Length * sizeof(uint)), indexPtr, BufferUsageARB.StaticDraw);
                }

                _gl.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 10u * sizeof(float), (void*)0);
                _gl.EnableVertexAttribArray(0);
                _gl.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, 10u * sizeof(float), (void*)(3 * sizeof(float)));
                _gl.EnableVertexAttribArray(1);
                _gl.VertexAttribPointer(2, 2, VertexAttribPointerType.Float, false, 10u * sizeof(float), (void*)(6 * sizeof(float)));
                _gl.EnableVertexAttribArray(2);
                _gl.VertexAttribPointer(3, 2, VertexAttribPointerType.Float, false, 10u * sizeof(float), (void*)(8 * sizeof(float)));
                _gl.EnableVertexAttribArray(3);
            }

            _gl.BindVertexArray(0);

            var buffers = new SectionBuffers(section.SectionIndex, section.SkinSectionId, vao, vbo, ebo, section.Vertices.Count, (uint)indices.Length, section.Material);
            _sections.Add(buffers);
            _sectionsByIndex[section.SectionIndex] = buffers;
        }
    }

    private void ApplyAnimatedFrame(M2SkinnedRenderModel skinnedRenderModel, M2RenderConsumerFrameState consumerState)
    {
        Dictionary<int, M2RenderConsumerPassState> firstPassBySection = consumerState.Passes
            .GroupBy(static pass => pass.AnimatedPass.SectionIndex)
            .ToDictionary(static group => group.Key, static group => group.First());

        for (int index = 0; index < _sections.Count; index++)
        {
            SectionBuffers section = _sections[index];
            bool baseVisible = index < _sectionVisibility.Count ? _sectionVisibility[index] : true;
            if (firstPassBySection.TryGetValue(section.SectionIndex, out M2RenderConsumerPassState? passState))
            {
                section.Visible = baseVisible && passState.Visible;
                section.AnimatedColor = passState.DiffuseColor;
                section.AnimatedAlpha = passState.Alpha;
                ApplyAnimatedTextureState(section, passState);
            }
            else
            {
                section.Visible = baseVisible;
                section.AnimatedColor = Vector3.One;
                section.AnimatedAlpha = 1.0f;
                ResetAnimatedTextureState(section);
            }
        }

        foreach (M2SkinnedRenderSection section in skinnedRenderModel.Sections)
            UploadAnimatedVertices(section);
    }

    private void ApplyAnimatedTextureState(SectionBuffers section, M2RenderConsumerPassState passState)
    {
        M2RenderConsumerTextureState? textureState = passState.Textures
            .OrderBy(static texture => texture.StageIndex)
            .FirstOrDefault();

        if (textureState == null)
        {
            ResetAnimatedTextureState(section);
            return;
        }

        Matrix4x4 rotationMatrix = Matrix4x4.CreateFromQuaternion(textureState.Rotation);
        section.AnimatedUvTranslation = new Vector2(textureState.Translation.X, textureState.Translation.Y);
        section.AnimatedUvScale = new Vector2(
            Math.Abs(textureState.Scaling.X) <= 0.0001f ? 1.0f : textureState.Scaling.X,
            Math.Abs(textureState.Scaling.Y) <= 0.0001f ? 1.0f : textureState.Scaling.Y);
        section.AnimatedUvRotation = new Vector2(rotationMatrix.M11, rotationMatrix.M21);
        section.HasAnimatedUvTransform = section.AnimatedUvTranslation.LengthSquared() > 0.000001f
            || Vector2.DistanceSquared(section.AnimatedUvScale, Vector2.One) > 0.000001f
            || Vector2.DistanceSquared(section.AnimatedUvRotation, new Vector2(1.0f, 0.0f)) > 0.000001f;
    }

    private static void ResetAnimatedTextureState(SectionBuffers section)
    {
        section.HasAnimatedUvTransform = false;
        section.AnimatedUvTranslation = Vector2.Zero;
        section.AnimatedUvScale = Vector2.One;
        section.AnimatedUvRotation = new Vector2(1.0f, 0.0f);
    }

    private unsafe void UploadAnimatedVertices(M2SkinnedRenderSection section)
    {
        if (_gl == null)
            return;

        if (!_sectionsByIndex.TryGetValue(section.Source.SectionIndex, out SectionBuffers? buffers)
            || !_staticSectionsByIndex.TryGetValue(section.Source.SectionIndex, out M2StaticRenderSection? sourceSection)
            || buffers.VertexCount != section.Vertices.Count
            || sourceSection.Vertices.Count != section.Vertices.Count)
        {
            return;
        }

        float[] vertexData = new float[section.Vertices.Count * 10];
        for (int index = 0; index < section.Vertices.Count; index++)
        {
            M2SkinnedRenderVertex animatedVertex = section.Vertices[index];
            M2StaticRenderVertex sourceVertex = sourceSection.Vertices[index];
            int offset = index * 10;
            vertexData[offset + 0] = animatedVertex.Position.X;
            vertexData[offset + 1] = animatedVertex.Position.Y;
            vertexData[offset + 2] = animatedVertex.Position.Z;
            vertexData[offset + 3] = animatedVertex.Normal.X;
            vertexData[offset + 4] = animatedVertex.Normal.Y;
            vertexData[offset + 5] = animatedVertex.Normal.Z;
            vertexData[offset + 6] = sourceVertex.TextureCoords0.X;
            vertexData[offset + 7] = sourceVertex.TextureCoords0.Y;
            vertexData[offset + 8] = sourceVertex.TextureCoords1.X;
            vertexData[offset + 9] = sourceVertex.TextureCoords1.Y;
        }

        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, buffers.Vbo);
        fixed (float* vertexPtr = vertexData)
        {
            _gl.BufferSubData(BufferTargetARB.ArrayBuffer, 0, (nuint)(vertexData.Length * sizeof(float)), vertexPtr);
        }
        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, 0);
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

            // Keep parity with the established M2 compatibility path until the
            // pure runtime renderer has proven stable winding or projected-pass rules.
            _gl.Disable(EnableCap.CullFace);

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

            Vector3 baseColor = ComputeSectionColor(section, fadeAlpha);
            _gl.Uniform3(_uBaseColor, baseColor.X, baseColor.Y, baseColor.Z);
            _gl.Uniform1(_uUnshaded, section.Material.IsUnshaded ? 1 : 0);
            _gl.Uniform1(_uHasTexture, section.HasTexture ? 1 : 0);
            _gl.Uniform1(_uUvSet, section.UvSet);
            _gl.Uniform1(_uGeneratedTexCoord, section.GeneratedTexCoord ? 1 : 0);
            _gl.Uniform1(_uAlphaCutout, section.AlphaCutout ? 1 : 0);
            _gl.Uniform1(_uAlpha, Math.Clamp(fadeAlpha * section.AnimatedAlpha, 0.0f, 1.0f));
            _gl.Uniform1(_uHasUvTransform, section.HasAnimatedUvTransform ? 1 : 0);
            _gl.Uniform2(_uUvTranslation, section.AnimatedUvTranslation.X, section.AnimatedUvTranslation.Y);
            _gl.Uniform2(_uUvScale, section.AnimatedUvScale.X, section.AnimatedUvScale.Y);
            _gl.Uniform2(_uUvRotation, section.AnimatedUvRotation.X, section.AnimatedUvRotation.Y);

            _gl.ActiveTexture(TextureUnit.Texture0);
            _gl.BindTexture(TextureTarget.Texture2D, section.HasTexture ? section.TextureId : 0u);
            _gl.Uniform1(_uTexture0, 0);

            _gl.BindVertexArray(section.Vao);
            _gl.DrawElements(PrimitiveType.Triangles, section.IndexCount, DrawElementsType.UnsignedInt, null);
        }

        _gl.BindTexture(TextureTarget.Texture2D, 0);
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

    private static Vector3 ComputeSectionColor(SectionBuffers section, float fadeAlpha)
    {
        float brightness = Math.Clamp(fadeAlpha, 0.1f, 1.0f);
        Vector3 animatedColor = new(
            Math.Clamp(section.AnimatedColor.X, 0.0f, 1.0f),
            Math.Clamp(section.AnimatedColor.Y, 0.0f, 1.0f),
            Math.Clamp(section.AnimatedColor.Z, 0.0f, 1.0f));
        return animatedColor * brightness;
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
layout (location = 2) in vec2 aTexCoord0;
layout (location = 3) in vec2 aTexCoord1;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;

out vec3 vWorldPos;
out vec3 vNormal;
out vec3 vViewNormal;
out vec2 vTexCoord0;
out vec2 vTexCoord1;

void main()
{
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    vWorldPos = worldPos.xyz;
    vNormal = normalize(mat3(uModel) * aNormal);
    vViewNormal = mat3(uView) * vNormal;
    vTexCoord0 = aTexCoord0;
    vTexCoord1 = aTexCoord1;
    gl_Position = uProj * uView * worldPos;
}
""";

const string fragmentSource = """
#version 330 core
in vec3 vWorldPos;
in vec3 vNormal;
in vec3 vViewNormal;

uniform vec3 uFogColor;
uniform float uFogStart;
uniform float uFogEnd;
uniform vec3 uCameraPos;
uniform vec3 uLightDir;
uniform vec3 uLightColor;
uniform vec3 uAmbientColor;
uniform vec3 uBaseColor;
uniform int uUnshaded;
uniform int uHasTexture;
uniform int uUvSet;
uniform int uGeneratedTexCoord;
uniform sampler2D uTexture0;
uniform int uAlphaCutout;
uniform float uAlpha;
uniform int uHasUvTransform;
uniform vec2 uUvTranslation;
uniform vec2 uUvScale;
uniform vec2 uUvRotation;

in vec2 vTexCoord0;
in vec2 vTexCoord1;

out vec4 FragColor;

void main()
{
    vec2 texCoord = uUvSet == 1 ? vTexCoord1 : vTexCoord0;
    if (uGeneratedTexCoord == 1)
    {
        vec3 viewNormal = normalize(vViewNormal);
        if (!gl_FrontFacing)
            viewNormal = -viewNormal;

        texCoord = viewNormal.xy * 0.5 + 0.5;
    }

    if (uHasUvTransform == 1)
    {
        mat2 uvRotationScale = mat2(
            uUvRotation.x * uUvScale.x, -uUvRotation.y * uUvScale.y,
            uUvRotation.y * uUvScale.x,  uUvRotation.x * uUvScale.y);
        texCoord = (uvRotationScale * texCoord) + uUvTranslation;
    }

    vec4 textureSample = uHasTexture == 1
        ? texture(uTexture0, texCoord)
        : vec4(1.0, 1.0, 1.0, 1.0);
    if (uAlphaCutout == 1 && textureSample.a < 0.5)
        discard;

    float diffuseStrength = uUnshaded == 1 ? 1.0 : max(dot(normalize(vNormal), normalize(uLightDir)), 0.0);
    vec3 litColor = (uBaseColor * textureSample.rgb) * (uAmbientColor + (uLightColor * diffuseStrength));
    float distanceToCamera = distance(vWorldPos, uCameraPos);
    float fogRange = max(uFogEnd - uFogStart, 0.001);
    float fogFactor = clamp((uFogEnd - distanceToCamera) / fogRange, 0.0, 1.0);
    vec3 finalColor = mix(uFogColor, litColor, fogFactor);
    FragColor = vec4(finalColor, textureSample.a * uAlpha);
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
        _uHasTexture = _gl.GetUniformLocation(_shaderProgram, "uHasTexture");
        _uUvSet = _gl.GetUniformLocation(_shaderProgram, "uUvSet");
        _uGeneratedTexCoord = _gl.GetUniformLocation(_shaderProgram, "uGeneratedTexCoord");
        _uTexture0 = _gl.GetUniformLocation(_shaderProgram, "uTexture0");
        _uAlphaCutout = _gl.GetUniformLocation(_shaderProgram, "uAlphaCutout");
        _uAlpha = _gl.GetUniformLocation(_shaderProgram, "uAlpha");
        _uHasUvTransform = _gl.GetUniformLocation(_shaderProgram, "uHasUvTransform");
        _uUvTranslation = _gl.GetUniformLocation(_shaderProgram, "uUvTranslation");
        _uUvScale = _gl.GetUniformLocation(_shaderProgram, "uUvScale");
        _uUvRotation = _gl.GetUniformLocation(_shaderProgram, "uUvRotation");
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

    private static int? SelectBestReplaceableDisplayIndex(M2StaticRenderModel runtimeModel, string modelPath, ReplaceableTextureResolver? texResolver)
    {
        if (texResolver == null)
            return null;

        uint[] replaceableIds = runtimeModel.Sections
            .SelectMany(static section => section.Material.TextureBindings)
            .Select(static binding => binding.ReplaceableId)
            .Where(static replaceableId => replaceableId != 0)
            .Distinct()
            .ToArray();

        return replaceableIds.Length == 0
            ? null
            : texResolver.SelectBestDisplayIndex(modelPath, replaceableIds);
    }

    private void LoadSectionTextures()
    {
        if (_gl == null)
            return;

        foreach (SectionBuffers section in _sections)
        {
            section.AlphaCutout = section.Material.BlendMode == WowViewer.Core.M2.M2BlendMode.AlphaKey;
            if (TryLoadMaterialTexture(section.Material, out uint textureId, out int uvSet, out bool generatedTexCoord))
            {
                section.TextureId = textureId;
                section.HasTexture = true;
                section.UvSet = uvSet;
                section.GeneratedTexCoord = generatedTexCoord;
            }
        }
    }

    private void ApplyCharacterSelectionGroups(IReadOnlyCollection<uint> wantedGroups, string? reasonLabel)
    {
        HashSet<uint> wantedGroupSet = wantedGroups as HashSet<uint> ?? new HashSet<uint>(wantedGroups);
        int hiddenCount = 0;

        foreach (SectionBuffers section in _sections)
        {
            bool visible = wantedGroupSet.Contains(section.SkinSectionId);
            section.Visible = visible;
            if (section.SectionIndex >= 0 && section.SectionIndex < _sectionVisibility.Count)
                _sectionVisibility[section.SectionIndex] = visible;
            if (!visible)
                hiddenCount++;
        }

        if (hiddenCount > 0)
        {
            ViewerLog.Info(
                ViewerLog.Category.Mdx,
                $"[M2] Applied {reasonLabel ?? "character geosets"} for {SourceModelPath}: visible={_sections.Count - hiddenCount}/{_sections.Count}, groups={string.Join(",", wantedGroupSet.OrderBy(static value => value))}");
        }
    }

    private bool UpdateCharacterTextureVariationState(int? hairVariationId, int? facialHairVariationId)
    {
        int resolvedHairVariationId = hairVariationId ?? 0;
        int resolvedFacialHairVariationId = facialHairVariationId ?? 0;
        if (_characterHairVariationId == resolvedHairVariationId && _characterFacialHairVariationId == resolvedFacialHairVariationId)
            return false;

        _characterHairVariationId = resolvedHairVariationId;
        _characterFacialHairVariationId = resolvedFacialHairVariationId;
        return true;
    }

    private void ReloadCharacterTextures()
    {
        if (_gl == null)
            return;

        ReleaseOwnedTextures();
        LoadSectionTextures();
    }

    private void ReleaseOwnedTextures()
    {
        if (_gl == null)
            return;

        foreach (uint textureId in _ownedTextureIds)
            _gl.DeleteTexture(textureId);

        _ownedTextureIds.Clear();
        _loadedTextureCache.Clear();

        foreach (SectionBuffers section in _sections)
        {
            section.TextureId = 0;
            section.HasTexture = false;
            section.UvSet = 0;
            section.GeneratedTexCoord = false;
        }
    }

    private bool TryLoadMaterialTexture(M2StaticRenderMaterial material, out uint textureId, out int uvSet, out bool generatedTexCoord)
    {
        textureId = 0;
        uvSet = 0;
        generatedTexCoord = false;

        foreach ((string? TexturePath, uint ReplaceableId, uint TextureFlags, int UvSet, bool GeneratedTexCoord) candidate in EnumerateTextureCandidates(material))
        {
            string? resolvedPath = candidate.TexturePath;
            if (string.IsNullOrWhiteSpace(resolvedPath) && candidate.ReplaceableId != 0)
                resolvedPath = ResolveReplaceableTexture(candidate.ReplaceableId);

            if (string.IsNullOrWhiteSpace(resolvedPath))
                continue;

            bool clampS = (candidate.TextureFlags & 0x1u) == 0;
            bool clampT = (candidate.TextureFlags & 0x2u) == 0;
            if (TryGetOrLoadTexture(resolvedPath, clampS, clampT, out textureId))
            {
                uvSet = candidate.UvSet;
                generatedTexCoord = candidate.GeneratedTexCoord;
                return true;
            }
        }

        return false;
    }

    private static IEnumerable<(string? TexturePath, uint ReplaceableId, uint TextureFlags, int UvSet, bool GeneratedTexCoord)> EnumerateTextureCandidates(M2StaticRenderMaterial material)
    {
        if (material.TextureBindings.Count > 0)
        {
            foreach (M2StaticRenderTextureBinding binding in material.TextureBindings.OrderBy(static binding => binding.StageIndex))
                yield return (
                    binding.TexturePath,
                    binding.ReplaceableId,
                    binding.TextureFlags,
                    NormalizeUvSet(binding.TextureCoordLookupValue),
                    UsesGeneratedTexCoord(binding.TextureCoordLookupValue));

            yield break;
        }

        yield return (material.TexturePath, material.ReplaceableId, material.TextureFlags, 0, false);
    }

    private static int NormalizeUvSet(ushort? textureCoordLookupValue)
    {
        return textureCoordLookupValue == 1 ? 1 : 0;
    }

    private static bool UsesGeneratedTexCoord(ushort? textureCoordLookupValue)
    {
        return textureCoordLookupValue == ushort.MaxValue;
    }

    private string? ResolveReplaceableTexture(uint replaceableId)
    {
        if (_texResolver == null)
            return null;

        return _texResolver.Resolve(
            SourceModelPath,
            replaceableId,
            _selectedReplaceableDisplayIndex ?? 0,
            _characterHairVariationId,
            _characterFacialHairVariationId);
    }

    private bool TryGetOrLoadTexture(string texturePath, bool clampS, bool clampT, out uint textureId)
    {
        textureId = 0;
        string cacheKey = BuildTextureCacheKey(texturePath, clampS, clampT);
        if (_loadedTextureCache.TryGetValue(cacheKey, out textureId))
            return textureId != 0;

        if (TryLoadTexture(texturePath, clampS, clampT, out textureId, out string resolvedPath))
        {
            _loadedTextureCache[cacheKey] = textureId;
            _loadedTextureCache[BuildTextureCacheKey(resolvedPath, clampS, clampT)] = textureId;
            _ownedTextureIds.Add(textureId);
            return true;
        }

        return false;
    }

    private bool TryLoadTexture(string texturePath, bool clampS, bool clampT, out uint textureId, out string resolvedPath)
    {
        textureId = 0;
        resolvedPath = texturePath.Replace('/', '\\');

        if (TryResolveImagePath(texturePath, ".png", out string pngPath))
        {
            textureId = LoadTextureFromImage(pngPath, clampS, clampT);
            if (textureId != 0)
            {
                resolvedPath = pngPath;
                return true;
            }
        }

        if (!TryReadTextureBytes(texturePath, out byte[]? blpData, out resolvedPath) || blpData == null || blpData.Length == 0)
            return false;

        textureId = LoadTextureFromBlp(blpData, resolvedPath, clampS, clampT);
        return textureId != 0;
    }

    private bool TryReadTextureBytes(string texturePath, out byte[]? bytes, out string resolvedPath)
    {
        bytes = null;
        resolvedPath = texturePath.Replace('/', '\\');

        IEnumerable<string> candidates = EnumerateTexturePathCandidates(texturePath);
        foreach (string candidate in candidates)
        {
            if (_dataSource is MpqDataSource mpqDataSource)
            {
                string? actualPath = mpqDataSource.FindInFileSet(candidate)
                    ?? mpqDataSource.FindInFileSet(candidate.Replace('\\', '/'));
                if (!string.IsNullOrWhiteSpace(actualPath))
                {
                    bytes = _dataSource.ReadFile(actualPath);
                    if (bytes != null && bytes.Length > 0)
                    {
                        resolvedPath = actualPath.Replace('/', '\\');
                        return true;
                    }
                }
            }

            if (_dataSource != null)
            {
                bytes = _dataSource.ReadFile(candidate)
                    ?? _dataSource.ReadFile(candidate.Replace('\\', '/'));
                if (bytes != null && bytes.Length > 0)
                {
                    resolvedPath = candidate.Replace('/', '\\');
                    return true;
                }
            }

            if (File.Exists(candidate))
            {
                bytes = File.ReadAllBytes(candidate);
                resolvedPath = Path.GetFullPath(candidate);
                return true;
            }
        }

        return false;
    }

    private IEnumerable<string> EnumerateTexturePathCandidates(string texturePath)
    {
        string normalized = texturePath.Replace('/', '\\').TrimStart('\\');
        yield return normalized;

        string fileName = Path.GetFileName(normalized);
        if (!string.IsNullOrWhiteSpace(_modelDir) && !string.IsNullOrWhiteSpace(fileName))
        {
            yield return Path.Combine(_modelDir, fileName);
            yield return Path.Combine(_modelDir, normalized);
        }
    }

    private bool TryResolveImagePath(string texturePath, string extension, out string resolvedPath)
    {
        resolvedPath = string.Empty;
        foreach (string candidate in EnumerateTexturePathCandidates(texturePath))
        {
            string imagePath = Path.ChangeExtension(candidate, extension);
            if (!File.Exists(imagePath))
                continue;

            resolvedPath = Path.GetFullPath(imagePath);
            return true;
        }

        return false;
    }

    private unsafe uint LoadTextureFromBlp(byte[] blpData, string name, bool clampS, bool clampT)
    {
        try
        {
            using MemoryStream memoryStream = new(blpData, writable: false);
            using BlpFile blp = new(memoryStream);
            using Bitmap bitmap = blp.GetBitmap(0);
            return UploadBitmap(bitmap, clampS, clampT);
        }
        catch (Exception ex)
        {
            ViewerLog.Debug(ViewerLog.Category.Mdx, $"[M2] Failed to decode BLP '{name}': {ex.Message}");
            return 0;
        }
    }

    private uint LoadTextureFromImage(string imagePath, bool clampS, bool clampT)
    {
        try
        {
            using Bitmap bitmap = new(imagePath);
            using Bitmap converted = bitmap.Clone(new Rectangle(0, 0, bitmap.Width, bitmap.Height), System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            return UploadBitmap(converted, clampS, clampT);
        }
        catch (Exception ex)
        {
            ViewerLog.Debug(ViewerLog.Category.Mdx, $"[M2] Failed to load image '{imagePath}': {ex.Message}");
            return 0;
        }
    }

    private unsafe uint UploadBitmap(Bitmap bitmap, bool clampS, bool clampT)
    {
        if (_gl == null)
            return 0;

        Rectangle rect = new(0, 0, bitmap.Width, bitmap.Height);
        BitmapData bitmapData = bitmap.LockBits(rect, ImageLockMode.ReadOnly, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
        try
        {
            byte[] sourceBytes = new byte[bitmapData.Stride * bitmapData.Height];
            System.Runtime.InteropServices.Marshal.Copy(bitmapData.Scan0, sourceBytes, 0, sourceBytes.Length);

            byte[] pixels = new byte[bitmap.Width * bitmap.Height * 4];
            for (int y = 0; y < bitmap.Height; y++)
            {
                for (int x = 0; x < bitmap.Width; x++)
                {
                    int sourceOffset = (y * bitmapData.Stride) + (x * 4);
                    int destinationOffset = ((y * bitmap.Width) + x) * 4;
                    pixels[destinationOffset + 0] = sourceBytes[sourceOffset + 2];
                    pixels[destinationOffset + 1] = sourceBytes[sourceOffset + 1];
                    pixels[destinationOffset + 2] = sourceBytes[sourceOffset + 0];
                    pixels[destinationOffset + 3] = sourceBytes[sourceOffset + 3];
                }
            }

            return UploadTexture(pixels, (uint)bitmap.Width, (uint)bitmap.Height, clampS, clampT);
        }
        finally
        {
            bitmap.UnlockBits(bitmapData);
        }
    }

    private unsafe uint UploadTexture(byte[] pixels, uint width, uint height, bool clampS, bool clampT)
    {
        if (_gl == null)
            return 0;

        uint textureId = _gl.GenTexture();
        _gl.BindTexture(TextureTarget.Texture2D, textureId);
        fixed (byte* pixelPtr = pixels)
        {
            _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba, width, height, 0, Silk.NET.OpenGL.PixelFormat.Rgba, PixelType.UnsignedByte, pixelPtr);
        }

        TextureWrapMode wrapS = clampS ? TextureWrapMode.ClampToEdge : TextureWrapMode.Repeat;
        TextureWrapMode wrapT = clampT ? TextureWrapMode.ClampToEdge : TextureWrapMode.Repeat;
        RenderQualitySettings.ApplySampling(_gl, TextureTarget.Texture2D, hasMipmaps: true, wrapS, wrapT);
        _gl.GenerateMipmap(TextureTarget.Texture2D);
        _gl.BindTexture(TextureTarget.Texture2D, 0);
        return textureId;
    }

    private static string BuildTextureCacheKey(string texturePath, bool clampS, bool clampT)
    {
        string normalizedPath = texturePath.Replace('/', '\\').ToLowerInvariant();
        return $"{normalizedPath}|s={(clampS ? 1 : 0)}|t={(clampT ? 1 : 0)}";
    }

    private sealed class SectionBuffers
    {
        public SectionBuffers(int sectionIndex, ushort skinSectionId, uint vao, uint vbo, uint ebo, int vertexCount, uint indexCount, M2StaticRenderMaterial material)
        {
            SectionIndex = sectionIndex;
            SkinSectionId = skinSectionId;
            Vao = vao;
            Vbo = vbo;
            Ebo = ebo;
            VertexCount = vertexCount;
            IndexCount = indexCount;
            Material = material;
        }

        public int SectionIndex { get; }

        public ushort SkinSectionId { get; }

        public uint Vao { get; }

        public uint Vbo { get; }

        public uint Ebo { get; }

        public int VertexCount { get; }

        public uint IndexCount { get; }

        public M2StaticRenderMaterial Material { get; }

        public uint TextureId { get; set; }

        public bool HasTexture { get; set; }

        public int UvSet { get; set; }

        public bool GeneratedTexCoord { get; set; }

        public bool AlphaCutout { get; set; }

        public bool Visible { get; set; } = true;

        public Vector3 AnimatedColor { get; set; } = Vector3.One;

        public float AnimatedAlpha { get; set; } = 1.0f;

        public bool HasAnimatedUvTransform { get; set; }

        public Vector2 AnimatedUvTranslation { get; set; } = Vector2.Zero;

        public Vector2 AnimatedUvScale { get; set; } = Vector2.One;

        public Vector2 AnimatedUvRotation { get; set; } = new(1.0f, 0.0f);
    }
}
