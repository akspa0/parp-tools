using System.Numerics;
using SereniaBLPLib;
using Silk.NET.OpenGL;
using WowViewer.Core.IO.Files;
using WowViewer.Core.Mdx;
using WowViewer.Core.Runtime.Mdx;

namespace WowViewer.App;

internal sealed class MdxGpuPreviewRenderer : IDisposable
{
    private const uint MdxBlendModeTransparentKey = 1;
    private const uint MdxBlendModeBlend = 2;
    private const uint MdxBlendModeAdditive = 3;
    private const uint MdxBlendModeAddAlpha = 4;
    private const uint MdxBlendModeModulate = 5;
    private const uint MdxBlendModeModulate2X = 6;
    private const float PreviewFieldOfViewDegrees = 25.0f;
    private const float PreviewPaddingScale = 1.04f;
    private const float PreviewZoomFactor = 0.72f;
    private const int MaxRenderedParticlesPerEmitter = 48;
    private const int MaxRenderedRibbonEdgesPerEmitter = 16;
    private const double FixedEffectStepSeconds = 1.0 / 60.0;
    private const double MaxEffectCatchUpSeconds = 0.25;

    private readonly GL _gl;
    private static readonly IReadOnlyDictionary<uint, string> DefaultReplaceableTextures = new Dictionary<uint, string>
    {
        [1] = @"Textures\ReplaceableTextures\CreatureSkin\CreatureSkin01.blp",
        [2] = @"Textures\ReplaceableTextures\ObjectSkin\ObjectSkin01.blp",
        [3] = @"Textures\ReplaceableTextures\WeaponBlade\WeaponBlade01.blp",
        [4] = @"Textures\ReplaceableTextures\WeaponHandle\WeaponHandle01.blp",
        [5] = @"Textures\ReplaceableTextures\Environment\Environment01.blp",
        [6] = @"Textures\ReplaceableTextures\CharHair\CharHair00_00.blp",
        [7] = @"Textures\ReplaceableTextures\CharFacialHair\CharFacialHair00_00.blp",
        [8] = @"Textures\ReplaceableTextures\SkinExtra\SkinExtra01.blp",
        [9] = @"Textures\ReplaceableTextures\UISkin\UISkin01.blp",
        [10] = @"Textures\ReplaceableTextures\TaurenMane\TaurenMane00_00.blp",
        [11] = @"Textures\ReplaceableTextures\Monster\Monster01_01.blp",
        [12] = @"Textures\ReplaceableTextures\Monster\Monster01_02.blp",
        [13] = @"Textures\ReplaceableTextures\Monster\Monster01_03.blp",
    };

    private readonly Dictionary<string, uint> _loadedTextureCache = new(StringComparer.OrdinalIgnoreCase);
    private readonly HashSet<uint> _ownedTextureIds = new();
    private readonly List<CommandBuffers> _commands = new();
    private readonly List<CommandBuffers> _effectCommands = new();
    private readonly Dictionary<int, ParticleEmitterSimulationState> _particleSimulations = new();
    private readonly Dictionary<int, RibbonEmitterSimulationState> _ribbonSimulations = new();

    private uint _shaderProgram;
    private int _uView;
    private int _uProj;
    private int _uLightDir;
    private int _uLightColor;
    private int _uAmbientColor;
    private int _uBaseColor;
    private int _uEmissiveColor;
    private int _uAlpha;
    private int _uHasTexture;
    private int _uTexture0;
    private int _uAlphaCutout;
    private int _uAlphaThreshold;
    private int _uReceivesLighting;
    private int _uUseTextureAlpha;
    private int _uPremultiplyAlpha;
    private int _uSphereEnvMap;
    private int _uUseBoneSkinning;
    private int _uBoneCount;
    private int _uBones;
    private int _uUseUvTransform;
    private int _uUvTranslation;
    private int _uUvScale;
    private int _uUvRotationRow0;
    private int _uUvRotationRow1;

    private uint _framebuffer;
    private uint _colorTexture;
    private uint _depthRenderbuffer;
    private uint _fallbackWhiteTexture;
    private int _frameWidth;
    private int _frameHeight;
    private MdxSummary? _currentSummary;
    private MdxPreviewLoadResult? _currentPreview;
    private PreviewCameraSettings _cameraSettings = new();
    private Matrix4x4[] _boneMatrices = [];
    private Vector3 _boundsMin = new(-1.0f, -1.0f, -1.0f);
    private Vector3 _boundsMax = new(1.0f, 1.0f, 1.0f);
    private Vector3 _ambientColor = new(0.35f, 0.35f, 0.4f);
    private Vector3 _lightColor = new(1.0f, 0.95f, 0.85f);
    private readonly Vector3 _lightDir = Vector3.Normalize(new Vector3(0.5f, 0.3f, 1.0f));
    private MdxEffectRuntimeState? _simulatedEffectRuntime;
    private int _simulatedEffectTimeMs;
    private double _effectStepAccumulatorSeconds;
    private bool _disposed;

    public MdxGpuPreviewRenderer(GL gl)
    {
        _gl = gl ?? throw new ArgumentNullException(nameof(gl));
        InitializeShader();
        _fallbackWhiteTexture = CreateFallbackWhiteTexture();
    }

    public uint PreviewTextureHandle => _colorTexture;

    public bool HasRenderableGeometry => _commands.Count > 0 || _effectCommands.Count > 0;

    public int CommandCount => _commands.Count + _effectCommands.Count;

    public void Dispose()
    {
        if (_disposed)
            return;

        _disposed = true;
        ClearPreview();

        foreach (uint textureId in _ownedTextureIds)
            _gl.DeleteTexture(textureId);

        _ownedTextureIds.Clear();
        _loadedTextureCache.Clear();

        if (_fallbackWhiteTexture != 0)
        {
            _gl.DeleteTexture(_fallbackWhiteTexture);
            _fallbackWhiteTexture = 0;
        }

        if (_shaderProgram != 0)
        {
            _gl.DeleteProgram(_shaderProgram);
            _shaderProgram = 0;
        }

        DeleteFramebuffer();
    }

    public void ClearPreview()
    {
        ClearGeometryCommands();
        ClearEffectCommands();
        _particleSimulations.Clear();
        _ribbonSimulations.Clear();
        _currentPreview = null;
        _currentSummary = null;
        _simulatedEffectRuntime = null;
        _simulatedEffectTimeMs = 0;
        _effectStepAccumulatorSeconds = 0.0;
        _boneMatrices = [];
    }

    public void LoadPreview(MdxPreviewLoadResult preview)
    {
        ArgumentNullException.ThrowIfNull(preview);

        ClearPreview();
        _currentPreview = preview;
        _currentSummary = preview.Summary;
        _cameraSettings = preview.Request.Camera;
        _simulatedEffectRuntime = preview.EffectRuntimeState;
        _simulatedEffectTimeMs = preview.Request.TimeMs;
        InitializeEffectSimulation(preview, preview.EffectRuntimeState);
        RebuildGeometryCommands(preview, preview.Request.TimeMs);
    }

    private void ClearGeometryCommands()
    {
        foreach (CommandBuffers command in _commands)
            command.Dispose(_gl);

        _commands.Clear();
    }

    private void RebuildGeometryCommands(MdxPreviewLoadResult preview, int animationTimeMs)
    {
        ClearGeometryCommands();

        ResolveBounds(preview.Geometry, preview.Summary, out Vector3 initialMin, out Vector3 initialMax);
        PreviewCameraPose initialPose = PreviewCameraPlanner.CreatePose(initialMin, initialMax, _cameraSettings, preview.Summary, preview.Cameras, preview.Request.SequenceIndex, animationTimeMs, preview.Request.VisualWidth, preview.Request.VisualHeight);
        _boneMatrices = preview.Bones.BoneCount > 0
            ? MdxBonePoseBuilder.Build(preview.Bones, preview.Summary, preview.Request.SequenceIndex, animationTimeMs, initialPose.CameraPosition)
            : [];

        bool hasSkinnedBounds = false;
        Vector3 skinnedBoundsMin = new(float.MaxValue);
        Vector3 skinnedBoundsMax = new(float.MinValue);

        foreach (MdxGeosetGeometry geoset in preview.Geometry.Geosets)
        {
            if (geoset.Vertices.Count == 0 || geoset.Indices.Count < 3)
                continue;



            float[] vertexData = new float[geoset.Vertices.Count * 8];
            bool usesBoneSkinning = _boneMatrices.Length > 0 && geoset.VertexGroupCount > 0 && geoset.MatrixGroupCount > 0;
            (Vector4[] boneIndices, Vector4[] boneWeights) = usesBoneSkinning
                ? MdxSkinningHelper.BuildBoneWeights(geoset, preview.Bones.Bones)
                : (Array.Empty<Vector4>(), Array.Empty<Vector4>());
            Vector3 geosetMin = new(float.MaxValue);
            Vector3 geosetMax = new(float.MinValue);
            for (int index = 0; index < geoset.Vertices.Count; index++)
            {
                Vector3 position = geoset.Vertices[index];
                Vector3 normal = index < geoset.Normals.Count ? geoset.Normals[index] : Vector3.UnitZ;
                if (usesBoneSkinning && index < boneIndices.Length && index < boneWeights.Length)
                {
                    position = MdxSkinningHelper.ApplySkinning(position, boneIndices[index], boneWeights[index], _boneMatrices);
                    normal = MdxSkinningHelper.ApplySkinningNormal(normal, boneIndices[index], boneWeights[index], _boneMatrices);
                }
                if (float.IsFinite(position.X) && float.IsFinite(position.Y) && float.IsFinite(position.Z))
                {
                    skinnedBoundsMin = Vector3.Min(skinnedBoundsMin, position);
                    skinnedBoundsMax = Vector3.Max(skinnedBoundsMax, position);
                    geosetMin = Vector3.Min(geosetMin, position);
                    geosetMax = Vector3.Max(geosetMax, position);
                    hasSkinnedBounds = true;
                }

                int offset = index * 8;
                vertexData[offset + 0] = position.X;
                vertexData[offset + 1] = position.Y;
                vertexData[offset + 2] = position.Z;
                vertexData[offset + 3] = normal.X;
                vertexData[offset + 4] = normal.Y;
                vertexData[offset + 5] = normal.Z;
                vertexData[offset + 6] = 0.0f;
                vertexData[offset + 7] = 0.0f;
            }

            float[] skinningVertexData = MdxSkinningHelper.BuildSkinningVertexData(
                boneIndices,
                boneWeights,
                geoset.Vertices.Count);
            ushort[] indices = geoset.Indices.ToArray();
            int layerCount = geoset.MaterialId >= 0 && geoset.MaterialId < preview.Summary.MaterialCount
                ? preview.Summary.Materials[geoset.MaterialId].LayerCount
                : 0;
            if (layerCount == 0)
                layerCount = 1;

            for (int layerIndex = 0; layerIndex < layerCount; layerIndex++)
            {
                    MdxResolvedMaterialState material = MdxRenderStateResolver.ResolveMaterial(preview.Summary, preview.Materials, geoset.MaterialId, layerIndex, preview.Request.SequenceIndex, animationTimeMs);
                MdxResolvedGeosetRenderState geosetState = MdxRenderStateResolver.ResolveGeosetRenderState(
                    preview.Summary,
                    preview.GeosetAnimations,
                    preview.Request.SequenceIndex,
                    animationTimeMs,
                    geoset,
                    material);
                if (geosetState.Alpha <= 0.001f)
                    continue;

                uint textureId = _fallbackWhiteTexture;
                bool hasTexture = false;
                if (TryGetOrLoadMaterialTexture(preview.Request, material, out uint loadedTextureId))
                {
                    textureId = loadedTextureId;
                    hasTexture = true;
                }

                float[] layeredVertexData = (float[])vertexData.Clone();
                IReadOnlyList<Vector2> layerUvSet = material.CoordId >= 0 && material.CoordId < geoset.UvSetCount
                    ? geoset.UvSets[material.CoordId]
                    : geoset.PrimaryUvSet;
                for (int vertexIndex = 0; vertexIndex < geoset.Vertices.Count; vertexIndex++)
                {
                    Vector2 uv = vertexIndex < layerUvSet.Count ? layerUvSet[vertexIndex] : Vector2.Zero;
                    int offset = (vertexIndex * 8) + 6;
                    layeredVertexData[offset + 0] = uv.X;
                    layeredVertexData[offset + 1] = uv.Y;
                }

                MdxResolvedTextureTransform textureTransform = MdxRenderStateResolver.ResolveTextureTransform(
                    preview.Summary,
                    preview.TextureAnimations,
                    preview.Request.SequenceIndex,
                    animationTimeMs,
                    material);

                _commands.Add(CreateCommandBuffers(
                    layeredVertexData,
                    skinningVertexData,
                    indices,
                    textureId,
                    hasTexture,
                    material.IsTransparent,
                    material.IsAdditive,
                    geosetState.DepthTest,
                    geosetState.DepthWrite,
                    material.AlphaCutout,
                    geosetState.ReceivesLighting,
                    material.UsesSphereEnvMap,
                    usesBoneSkinning,
                    textureTransform.UsesTransform,
                    textureTransform.Translation,
                    textureTransform.Scale,
                    textureTransform.RotationRow0,
                    textureTransform.RotationRow1,
                    geosetState.BaseColor,
                    Vector3.Zero,
                    geosetState.Alpha,
                    material.BlendMode,
                    geoset.MaterialId >= 0 && geoset.MaterialId < preview.Summary.MaterialCount
                        ? preview.Summary.Materials[geoset.MaterialId].PriorityPlane
                        : 0,
                    ResolveBoundsCenter(geosetMin, geosetMax)));
            }
        }

        if (hasSkinnedBounds)
        {
            _boundsMin = skinnedBoundsMin;
            _boundsMax = skinnedBoundsMax;
        }
        else
        {
            ResolveBounds(preview.Geometry, preview.Summary, out _boundsMin, out _boundsMax);
        }
    }

    public void Render(int width, int height, double deltaSeconds = 0.0)
    {
        EnsureFramebuffer(width, height);

        _gl.BindFramebuffer(GLEnum.Framebuffer, _framebuffer);
        _gl.Viewport(0, 0, (uint)_frameWidth, (uint)_frameHeight);
        _gl.ClearColor(0.08f, 0.09f, 0.11f, 1.0f);
        _gl.Clear((uint)(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit));

        if (_commands.Count > 0 || _currentPreview is not null)
        {
            int animationTimeMs = _currentPreview is not null ? _simulatedEffectTimeMs : 0;
            if (_currentPreview is not null)
                RebuildGeometryCommands(_currentPreview, animationTimeMs);

            PreviewCameraPose pose = PreviewCameraPlanner.CreatePose(_boundsMin, _boundsMax, _cameraSettings, _currentSummary, _currentPreview?.Cameras, _currentPreview?.Request.SequenceIndex ?? 0, _currentPreview is not null ? animationTimeMs : _currentPreview?.Request.TimeMs ?? 0, _frameWidth, _frameHeight);

            if (_currentPreview is not null)
            {
                MdxEffectRuntimeState effectRuntime = AdvanceEffectSimulation(_currentPreview, deltaSeconds);
                RebuildEffectCommands(_currentPreview, effectRuntime, pose);
            }

            Matrix4x4 view = pose.View;
            Matrix4x4 projection = pose.Projection;
            RenderPass(view, projection, pose.CameraPosition, transparentPass: false);
            RenderPass(view, projection, pose.CameraPosition, transparentPass: true);
        }

        _gl.BindFramebuffer(GLEnum.Framebuffer, 0);
    }

    public void CaptureBmp(string outputPath, int width, int height)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(outputPath);

        Render(width, height);

        _gl.BindFramebuffer(GLEnum.Framebuffer, _framebuffer);
        byte[] rgbaPixels = new byte[_frameWidth * _frameHeight * 4];
        unsafe
        {
            fixed (byte* pixelPtr = rgbaPixels)
            {
                _gl.ReadPixels(0, 0, (uint)_frameWidth, (uint)_frameHeight, PixelFormat.Rgba, PixelType.UnsignedByte, pixelPtr);
            }
        }

        _gl.BindFramebuffer(GLEnum.Framebuffer, 0);
        ImageOutputWriter.WriteRgbaImage(outputPath, _frameWidth, _frameHeight, rgbaPixels, sourceOriginBottomLeft: true);
    }

    private void RenderPass(Matrix4x4 view, Matrix4x4 projection, Vector3 cameraPosition, bool transparentPass)
    {
        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Lequal);
        _gl.Disable(EnableCap.CullFace);
        _gl.UseProgram(_shaderProgram);

        unsafe
        {
            _gl.UniformMatrix4(_uView, 1, false, (float*)&view);
            _gl.UniformMatrix4(_uProj, 1, false, (float*)&projection);
        }

        _gl.Uniform3(_uLightDir, _lightDir.X, _lightDir.Y, _lightDir.Z);
        _gl.Uniform3(_uLightColor, _lightColor.X, _lightColor.Y, _lightColor.Z);
        _gl.Uniform3(_uAmbientColor, _ambientColor.X, _ambientColor.Y, _ambientColor.Z);
        _gl.Uniform1(_uBoneCount, Math.Min(_boneMatrices.Length, 128));
        if (_boneMatrices.Length > 0)
        {
            unsafe
            {
                fixed (Matrix4x4* bonePtr = _boneMatrices)
                {
                    _gl.UniformMatrix4(_uBones, (uint)Math.Min(_boneMatrices.Length, 128), false, (float*)bonePtr);
                }
            }
        }

        if (transparentPass)
        {
            foreach (CommandBuffers command in _commands.Where(static command => command.IsTransparent)
                         .OrderBy(static command => command.TransparentSortPriority)
                         .ThenByDescending(command => Vector3.DistanceSquared(command.TransparentSortCenter, cameraPosition)))
            {
                RenderCommand(command, transparentPass);
            }
        }
        else
        {
            foreach (CommandBuffers command in _commands)
                RenderCommand(command, transparentPass);
        }

        foreach (CommandBuffers command in _effectCommands)
            RenderCommand(command, transparentPass);

        _gl.BindVertexArray(0);
        _gl.BindTexture(TextureTarget.Texture2D, 0);
        _gl.Disable(EnableCap.Blend);
        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthMask(true);
        _gl.UseProgram(0);
    }

    private void RenderCommand(CommandBuffers command, bool transparentPass)
    {
        if (command.IsTransparent != transparentPass)
            return;

        if (command.DepthTest)
            _gl.Enable(EnableCap.DepthTest);
        else
            _gl.Disable(EnableCap.DepthTest);

        if (command.IsTransparent)
        {
            _gl.Enable(EnableCap.Blend);
            ConfigureBlendMode(command.IsAdditive, command.BlendMode);
        }
        else
        {
            _gl.Disable(EnableCap.Blend);
        }

        ResolveAlphaHandling(command, out bool useTextureAlpha, out bool premultiplyAlpha, out float alphaThreshold);

        _gl.DepthMask(!command.IsTransparent || command.DepthWrite);

        _gl.Uniform3(_uBaseColor, command.BaseColor.X, command.BaseColor.Y, command.BaseColor.Z);
        _gl.Uniform3(_uEmissiveColor, command.EmissiveColor.X, command.EmissiveColor.Y, command.EmissiveColor.Z);
        _gl.Uniform1(_uAlpha, command.Alpha);
        _gl.Uniform1(_uHasTexture, command.HasTexture ? 1 : 0);
        _gl.Uniform1(_uAlphaCutout, command.AlphaCutout ? 1 : 0);
        _gl.Uniform1(_uAlphaThreshold, alphaThreshold);
        _gl.Uniform1(_uReceivesLighting, command.ReceivesLighting ? 1 : 0);
        _gl.Uniform1(_uUseTextureAlpha, useTextureAlpha ? 1 : 0);
        _gl.Uniform1(_uPremultiplyAlpha, premultiplyAlpha ? 1 : 0);
        _gl.Uniform1(_uSphereEnvMap, command.UsesSphereEnvMap ? 1 : 0);
        _gl.Uniform1(_uUseBoneSkinning, command.UsesBoneSkinning && _boneMatrices.Length > 0 ? 1 : 0);
        _gl.Uniform1(_uUseUvTransform, command.UsesUvTransform ? 1 : 0);
        _gl.Uniform2(_uUvTranslation, command.UvTranslation.X, command.UvTranslation.Y);
        _gl.Uniform2(_uUvScale, command.UvScale.X, command.UvScale.Y);
        _gl.Uniform2(_uUvRotationRow0, command.UvRotationRow0.X, command.UvRotationRow0.Y);
        _gl.Uniform2(_uUvRotationRow1, command.UvRotationRow1.X, command.UvRotationRow1.Y);

        _gl.ActiveTexture(TextureUnit.Texture0);
        _gl.BindTexture(TextureTarget.Texture2D, command.TextureId);
        _gl.Uniform1(_uTexture0, 0);
        _gl.BindVertexArray(command.Vao);
        unsafe
        {
            _gl.DrawElements(PrimitiveType.Triangles, command.IndexCount, DrawElementsType.UnsignedShort, (void*)0);
        }
    }

    private MdxEffectRuntimeState AdvanceEffectSimulation(MdxPreviewLoadResult preview, double deltaSeconds)
    {
        if (_simulatedEffectRuntime is null)
        {
            _simulatedEffectRuntime = MdxEffectRuntimeEvaluator.Evaluate(
                preview.Summary,
                preview.Events,
                preview.ParticleEmitters,
                preview.Ribbons,
                preview.Request.SequenceIndex,
                _simulatedEffectTimeMs);
            InitializeEffectSimulation(preview, _simulatedEffectRuntime);
        }

        _effectStepAccumulatorSeconds += Math.Clamp(deltaSeconds, 0.0, MaxEffectCatchUpSeconds);
        while (_effectStepAccumulatorSeconds >= FixedEffectStepSeconds)
        {
            StepEffectSimulation(preview, (float)FixedEffectStepSeconds);
            _effectStepAccumulatorSeconds -= FixedEffectStepSeconds;
        }

        return _simulatedEffectRuntime;
    }

    private void StepEffectSimulation(MdxPreviewLoadResult preview, float stepSeconds)
    {
        int stepMs = Math.Max(1, (int)Math.Round(stepSeconds * 1000.0f));
        _simulatedEffectTimeMs += stepMs;
        MdxEffectRuntimeState nextRuntime = MdxEffectRuntimeEvaluator.Evaluate(
            preview.Summary,
            preview.Events,
            preview.ParticleEmitters,
            preview.Ribbons,
            preview.Request.SequenceIndex,
            _simulatedEffectTimeMs);

        UpdateParticleSimulations(preview, nextRuntime, stepSeconds);
        _simulatedEffectRuntime = nextRuntime;
    }

    private void InitializeEffectSimulation(MdxPreviewLoadResult preview, MdxEffectRuntimeState runtime)
    {
        _particleSimulations.Clear();
        _ribbonSimulations.Clear();

        foreach (MdxParticleEmitter2RuntimeState particle in runtime.Particles)
        {
            MdxParticleEmitter2? definition = TryGetParticleDefinition(preview, particle.Index);
            if (definition is null)
                continue;

            ParticleEmitterSimulationState state = new(definition.Index);
            SyncParticleEmitterState(preview, particle, definition, state);
            _particleSimulations[particle.Index] = state;
        }
    }

    private void UpdateParticleSimulations(MdxPreviewLoadResult preview, MdxEffectRuntimeState runtime, float stepSeconds)
    {
        foreach (MdxParticleEmitter2RuntimeState particleRuntime in runtime.Particles)
        {
            MdxParticleEmitter2? definition = TryGetParticleDefinition(preview, particleRuntime.Index);
            if (definition is null)
                continue;

            if (!_particleSimulations.TryGetValue(particleRuntime.Index, out ParticleEmitterSimulationState? state))
            {
                state = new ParticleEmitterSimulationState(definition.Index);
                _particleSimulations.Add(particleRuntime.Index, state);
            }

            SyncParticleEmitterState(preview, particleRuntime, definition, state);
            UpdateParticleEmitter(state, particleRuntime, stepSeconds);
        }
    }

    private void UpdateRibbonSimulations(MdxEffectRuntimeState runtime, float stepSeconds)
    {
        foreach (RibbonEmitterSimulationState state in _ribbonSimulations.Values)
        {
            for (int index = state.Points.Count - 1; index >= 0; index--)
            {
                SimulatedRibbonPoint point = state.Points[index];
                point.AgeSeconds += stepSeconds;
                point.Position += Vector3.UnitZ * (-state.Gravity * stepSeconds);
                if (point.AgeSeconds >= state.EdgeLifetimeSeconds)
                {
                    state.Points.RemoveAt(index);
                    continue;
                }

                state.Points[index] = point;
            }
        }

        foreach (MdxRibbonRuntimeState ribbonRuntime in runtime.Ribbons)
        {
            if (!_ribbonSimulations.TryGetValue(ribbonRuntime.Index, out RibbonEmitterSimulationState? state))
            {
                state = new RibbonEmitterSimulationState();
                _ribbonSimulations.Add(ribbonRuntime.Index, state);
            }

            state.Gravity = ribbonRuntime.Gravity;
            state.EdgeLifetimeSeconds = Math.Max(ribbonRuntime.EdgeLifetime, 0.1f);

            if (state.Points.Count > 0)
                state.Points[0] = new SimulatedRibbonPoint(ribbonRuntime.Position, 0.0f);

            if (!ribbonRuntime.Visible || ribbonRuntime.Alpha <= 0.001f)
                continue;

            state.EdgeAccumulator += ribbonRuntime.EdgesPerSecond * stepSeconds;
            if (state.Points.Count == 0)
                state.Points.Add(new SimulatedRibbonPoint(ribbonRuntime.Position, 0.0f));

            while (state.Points.Count < MaxRenderedRibbonEdgesPerEmitter && state.EdgeAccumulator >= 1.0f)
            {
                state.Points.Insert(0, new SimulatedRibbonPoint(ribbonRuntime.Position, 0.0f));
                state.EdgeAccumulator -= 1.0f;
            }
        }
    }

    private void RebuildEffectCommands(MdxPreviewLoadResult preview, MdxEffectRuntimeState runtime, PreviewCameraPose pose)
    {
        ClearEffectCommands();

        if (runtime.Particles.Count == 0)
            return;

        Vector3 viewDirection = Vector3.Normalize(pose.FocusPoint - pose.CameraPosition);
        Vector3 upHint = MathF.Abs(Vector3.Dot(viewDirection, Vector3.UnitZ)) > 0.98f ? Vector3.UnitY : Vector3.UnitZ;
        Vector3 cameraRight = Vector3.Normalize(Vector3.Cross(viewDirection, upHint));
        Vector3 cameraUp = Vector3.Normalize(Vector3.Cross(cameraRight, viewDirection));

        foreach (MdxParticleEmitter2RuntimeState particle in runtime.Particles)
            TryAddParticleEffectCommand(preview, particle, cameraRight, cameraUp);
    }

    private void ClearEffectCommands()
    {
        foreach (CommandBuffers command in _effectCommands)
            command.Dispose(_gl);

        _effectCommands.Clear();
    }

    private void TryAddParticleEffectCommand(MdxPreviewLoadResult preview, MdxParticleEmitter2RuntimeState particle, Vector3 cameraRight, Vector3 cameraUp)
    {
        if (!particle.Enabled || particle.Visibility <= 0.001f)
            return;

        MdxParticleEmitter2? definition = TryGetParticleDefinition(preview, particle.Index);
        if (definition is null)
            return;

        if (!_particleSimulations.TryGetValue(particle.Index, out ParticleEmitterSimulationState? state) || state.Particles.Count == 0)
            return;

        int spriteCount = Math.Min(state.Particles.Count, MaxRenderedParticlesPerEmitter);
        if (spriteCount <= 0)
            return;

        uint textureId = _fallbackWhiteTexture;
        bool hasTexture = TryGetOrLoadParticleTexture(preview.Request, preview.Summary, particle, out uint loadedTextureId);
        if (hasTexture)
            textureId = loadedTextureId;

        Vector3 baseColor = Vector3.Zero;
        float alpha = 0.0f;
        if (alpha <= 0.001f)
        {
            // The averaged particle alpha is computed below from the live particle list.
        }

        float[] vertexData = new float[spriteCount * 4 * 8];
        ushort[] indices = new ushort[spriteCount * 6];
        int atlasRows = Math.Max((int)definition.Rows, 1);
        int atlasColumns = Math.Max((int)definition.Columns, 1);
        int atlasFrameCount = Math.Max(atlasRows * atlasColumns, 1);

        for (int spriteIndex = 0; spriteIndex < spriteCount; spriteIndex++)
        {
            SimulatedParticle liveParticle = state.Particles[spriteIndex];
            float agePhase = liveParticle.LifetimeSeconds <= 0.001f
                ? 1.0f
                : Math.Clamp(liveParticle.AgeSeconds / liveParticle.LifetimeSeconds, 0.0f, 1.0f);
            Vector3 particleColor = ComputeParticleColor(definition, agePhase);
            float particleAlpha = ComputeParticleAlpha(definition, agePhase);
            float scale = ComputeParticleScale(definition, agePhase);
            float halfWidth = Math.Max(0.04f, particle.Width * Math.Max(scale, 0.05f) * 0.5f);
            float halfHeight = Math.Max(0.04f, (particle.Length > 0.001f ? particle.Length : particle.Width) * Math.Max(scale, 0.05f) * 0.5f);
            int atlasFrame = Math.Clamp((int)MathF.Floor(agePhase * atlasFrameCount), 0, atlasFrameCount - 1);
            WriteBillboardQuad(
                vertexData,
                indices,
                spriteIndex,
                liveParticle.Position,
                cameraRight * halfWidth,
                cameraUp * halfHeight,
                GetAtlasUvBounds(atlasRows, atlasColumns, atlasFrame));
            baseColor += particleColor;
            alpha += particleAlpha;
        }

        baseColor /= spriteCount;
        alpha = Math.Clamp((alpha / spriteCount) * particle.Visibility, 0.0f, 1.0f);
        if (alpha <= 0.001f)
            return;

        bool isTransparent = particle.BlendMode != 0 || alpha < 0.999f;
        bool alphaCutout = particle.BlendMode == MdxBlendModeTransparentKey;
        bool isAdditive = particle.BlendMode is MdxBlendModeAdditive or MdxBlendModeAddAlpha;
        _effectCommands.Add(CreateCommandBuffers(
            vertexData,
            BuildZeroSkinningVertexData(spriteCount * 4),
            indices,
            textureId,
            hasTexture,
            isTransparent,
            isAdditive,
            depthTest: true,
            depthWrite: !isTransparent || alphaCutout,
            alphaCutout,
            receivesLighting: false,
            usesSphereEnvMap: false,
            usesBoneSkinning: false,
            usesUvTransform: false,
            Vector2.Zero,
            Vector2.One,
            new Vector2(1.0f, 0.0f),
            new Vector2(0.0f, 1.0f),
            baseColor,
            isAdditive ? baseColor * 0.3f : Vector3.Zero,
            alpha,
            particle.BlendMode));
    }

    private void TryAddRibbonEffectCommand(MdxPreviewLoadResult preview, MdxRibbonRuntimeState ribbon, Vector3 cameraPosition)
    {
        if (!_ribbonSimulations.TryGetValue(ribbon.Index, out RibbonEmitterSimulationState? state) || state.Points.Count < 2)
            return;

        if ((!ribbon.Visible && state.Points.Count < 2) || ribbon.Alpha <= 0.001f)
            return;

        MdxRibbonEmitter? definition = TryGetRibbonDefinition(preview, ribbon.Index);
        if (definition is null)
            return;

        int edgeCount = Math.Clamp(state.Points.Count, 2, MaxRenderedRibbonEdgesPerEmitter);
        int vertexCount = edgeCount * 2;
        float[] vertexData = new float[vertexCount * 8];
        ushort[] indices = new ushort[(edgeCount - 1) * 6];

        MdxResolvedMaterialState material = MdxRenderStateResolver.ResolveMaterial(preview.Summary, preview.Materials, (int)ribbon.MaterialId, 0, preview.Request.SequenceIndex, _simulatedEffectTimeMs);
        MdxResolvedTextureTransform textureTransform = MdxRenderStateResolver.ResolveTextureTransform(
            preview.Summary,
            preview.TextureAnimations,
            preview.Request.SequenceIndex,
            _simulatedEffectTimeMs,
            material);

        uint textureId = _fallbackWhiteTexture;
        bool hasTexture = false;
        if (TryGetOrLoadMaterialTexture(preview.Request, material, out uint loadedTextureId))
        {
            textureId = loadedTextureId;
            hasTexture = true;
        }

        (Vector2 uvMin, Vector2 uvMax) = GetAtlasUvBounds(Math.Max((int)definition.TextureRows, 1), Math.Max((int)definition.TextureColumns, 1), Math.Max(ribbon.TextureSlot, 0));

        for (int edgeIndex = 0; edgeIndex < edgeCount; edgeIndex++)
        {
            float t = edgeCount == 1 ? 0.0f : edgeIndex / (float)(edgeCount - 1);
            Vector3 anchor = state.Points[edgeIndex].Position;
            WriteRibbonEdge(vertexData, edgeIndex, anchor, ribbon.HeightAbove, ribbon.HeightBelow, uvMin, uvMax, t);

            if (edgeIndex >= edgeCount - 1)
                continue;

            int indexOffset = edgeIndex * 6;
            ushort baseVertex = (ushort)(edgeIndex * 2);
            indices[indexOffset + 0] = baseVertex;
            indices[indexOffset + 1] = (ushort)(baseVertex + 1);
            indices[indexOffset + 2] = (ushort)(baseVertex + 2);
            indices[indexOffset + 3] = (ushort)(baseVertex + 2);
            indices[indexOffset + 4] = (ushort)(baseVertex + 1);
            indices[indexOffset + 5] = (ushort)(baseVertex + 3);
        }

        bool isTransparent = material.IsTransparent || ribbon.Alpha < 0.999f;
        _effectCommands.Add(CreateCommandBuffers(
            vertexData,
            BuildZeroSkinningVertexData(vertexCount),
            indices,
            textureId,
            hasTexture,
            isTransparent,
            material.IsAdditive,
            depthTest: true,
            depthWrite: material.DepthWrite && !isTransparent,
            material.AlphaCutout,
            receivesLighting: false,
            usesSphereEnvMap: false,
            usesBoneSkinning: false,
            textureTransform.UsesTransform,
            textureTransform.Translation,
            textureTransform.Scale,
            textureTransform.RotationRow0,
            textureTransform.RotationRow1,
            ribbon.Color,
            material.IsAdditive ? ribbon.Color * 0.2f : Vector3.Zero,
            ribbon.Alpha,
            material.BlendMode));
    }

    private static void WriteBillboardQuad(float[] vertexData, ushort[] indices, int quadIndex, Vector3 center, Vector3 rightOffset, Vector3 upOffset, (Vector2 Min, Vector2 Max) uvBounds)
    {
        int vertexOffset = quadIndex * 32;
        ushort baseVertex = (ushort)(quadIndex * 4);
        Vector3 normal = Vector3.UnitZ;
        Vector3 bottomLeft = center - rightOffset - upOffset;
        Vector3 bottomRight = center + rightOffset - upOffset;
        Vector3 topRight = center + rightOffset + upOffset;
        Vector3 topLeft = center - rightOffset + upOffset;

        WriteVertex(vertexData, vertexOffset + 0, bottomLeft, normal, new Vector2(uvBounds.Min.X, uvBounds.Max.Y));
        WriteVertex(vertexData, vertexOffset + 8, bottomRight, normal, new Vector2(uvBounds.Max.X, uvBounds.Max.Y));
        WriteVertex(vertexData, vertexOffset + 16, topRight, normal, new Vector2(uvBounds.Max.X, uvBounds.Min.Y));
        WriteVertex(vertexData, vertexOffset + 24, topLeft, normal, new Vector2(uvBounds.Min.X, uvBounds.Min.Y));

        int indexOffset = quadIndex * 6;
        indices[indexOffset + 0] = baseVertex;
        indices[indexOffset + 1] = (ushort)(baseVertex + 1);
        indices[indexOffset + 2] = (ushort)(baseVertex + 2);
        indices[indexOffset + 3] = baseVertex;
        indices[indexOffset + 4] = (ushort)(baseVertex + 2);
        indices[indexOffset + 5] = (ushort)(baseVertex + 3);
    }

    private static void WriteRibbonEdge(float[] vertexData, int edgeIndex, Vector3 anchor, float heightAbove, float heightBelow, Vector2 uvMin, Vector2 uvMax, float u)
    {
        int vertexOffset = edgeIndex * 16;
        Vector3 normal = Vector3.UnitY;
        Vector3 bottom = anchor - new Vector3(0.0f, 0.0f, Math.Max(heightBelow, 0.02f));
        Vector3 top = anchor + new Vector3(0.0f, 0.0f, Math.Max(heightAbove, 0.02f));
        float textureU = Lerp(uvMin.X, uvMax.X, u);
        WriteVertex(vertexData, vertexOffset + 0, bottom, normal, new Vector2(textureU, uvMax.Y));
        WriteVertex(vertexData, vertexOffset + 8, top, normal, new Vector2(textureU, uvMin.Y));
    }

    private static void WriteVertex(float[] vertexData, int offset, Vector3 position, Vector3 normal, Vector2 uv)
    {
        vertexData[offset + 0] = position.X;
        vertexData[offset + 1] = position.Y;
        vertexData[offset + 2] = position.Z;
        vertexData[offset + 3] = normal.X;
        vertexData[offset + 4] = normal.Y;
        vertexData[offset + 5] = normal.Z;
        vertexData[offset + 6] = uv.X;
        vertexData[offset + 7] = uv.Y;
    }

    private static (Vector2 Min, Vector2 Max) GetAtlasUvBounds(int rows, int columns, int frame)
    {
        rows = Math.Max(rows, 1);
        columns = Math.Max(columns, 1);
        int cellCount = rows * columns;
        int clampedFrame = ((frame % cellCount) + cellCount) % cellCount;
        int row = clampedFrame / columns;
        int column = clampedFrame % columns;
        float width = 1.0f / columns;
        float height = 1.0f / rows;
        return (
            new Vector2(column * width, row * height),
            new Vector2((column + 1) * width, (row + 1) * height));
    }

    private static float[] BuildZeroSkinningVertexData(int vertexCount) => new float[Math.Max(vertexCount, 0) * 8];

    private static Vector3 ResolveBoundsCenter(Vector3 min, Vector3 max)
    {
        if (!float.IsFinite(min.X) || !float.IsFinite(min.Y) || !float.IsFinite(min.Z)
            || !float.IsFinite(max.X) || !float.IsFinite(max.Y) || !float.IsFinite(max.Z))
            return Vector3.Zero;

        return (min + max) * 0.5f;
    }

    private static Vector3 ComputeParticleColor(MdxParticleEmitter2 definition, float phase)
    {
        if (phase < 0.5f)
            return Vector3.Lerp(definition.StartColor, definition.MiddleColor, phase * 2.0f);

        return Vector3.Lerp(definition.MiddleColor, definition.EndColor, (phase - 0.5f) * 2.0f);
    }

    private static float ComputeParticleAlpha(MdxParticleEmitter2 definition, float phase)
    {
        float startAlpha = definition.StartAlpha / 255.0f;
        float middleAlpha = definition.MiddleAlpha / 255.0f;
        float endAlpha = definition.EndAlpha / 255.0f;
        if (phase < 0.5f)
            return Lerp(startAlpha, middleAlpha, phase * 2.0f);

        return Lerp(middleAlpha, endAlpha, (phase - 0.5f) * 2.0f);
    }

    private static float ComputeParticleScale(MdxParticleEmitter2 definition, float phase)
    {
        if (phase < 0.5f)
            return Lerp(definition.StartScale, definition.MiddleScale, phase * 2.0f);

        return Lerp(definition.MiddleScale, definition.EndScale, (phase - 0.5f) * 2.0f);
    }

    private static float Frac(float value) => value - MathF.Floor(value);

    private void SyncParticleEmitterState(MdxPreviewLoadResult preview, MdxParticleEmitter2RuntimeState runtime, MdxParticleEmitter2 definition, ParticleEmitterSimulationState state)
    {
        state.Gravity = runtime.Gravity;
        state.IsActive = runtime.Enabled && runtime.Visibility > 0.001f;
        state.Transform = ResolveParticleEmitterTransform(preview, definition, runtime.Position);
    }

    private void UpdateParticleEmitter(ParticleEmitterSimulationState state, MdxParticleEmitter2RuntimeState runtime, float stepSeconds)
    {
        if (!state.IsActive)
            return;

        state.TimeSinceLastEmitSeconds += stepSeconds;

        for (int index = state.Particles.Count - 1; index >= 0; index--)
        {
            SimulatedParticle particle = state.Particles[index];
            particle.AgeSeconds += stepSeconds;
            if (particle.AgeSeconds >= particle.LifetimeSeconds)
            {
                state.Particles.RemoveAt(index);
                continue;
            }

            particle.Velocity += Vector3.UnitZ * (-state.Gravity * stepSeconds);
            particle.Position += particle.Velocity * stepSeconds;
            state.Particles[index] = particle;
        }

        if (runtime.EmissionRate <= 0.0f)
            return;

        float emitInterval = 1.0f / runtime.EmissionRate;
        while (state.TimeSinceLastEmitSeconds >= emitInterval && state.Particles.Count < MaxRenderedParticlesPerEmitter)
        {
            state.Particles.Add(SpawnParticle(state, runtime));
            state.TimeSinceLastEmitSeconds -= emitInterval;
        }
    }

    private static SimulatedParticle SpawnParticle(ParticleEmitterSimulationState state, MdxParticleEmitter2RuntimeState runtime)
    {
        Vector3 emitterPosition = Vector3.Transform(Vector3.Zero, state.Transform);
        float theta = (float)(state.Random.NextDouble() * Math.PI * 2.0);
        float phi = (float)(state.Random.NextDouble() * runtime.Latitude);
        float speed = runtime.Speed + (float)(state.Random.NextDouble() - 0.5) * runtime.Variation;
        Vector3 velocity = new(
            MathF.Sin(phi) * MathF.Cos(theta),
            MathF.Sin(phi) * MathF.Sin(theta),
            MathF.Cos(phi));
        velocity *= speed;
        velocity = Vector3.TransformNormal(velocity, state.Transform);

        return new SimulatedParticle(
            emitterPosition + new Vector3(0.0f, 0.0f, runtime.ZSource),
            velocity,
            0.0f,
            Math.Max(runtime.Life, 0.001f));
    }

    private static void PrimeRibbonSimulation(RibbonEmitterSimulationState state, MdxRibbonRuntimeState runtime, MdxRibbonEmitter definition)
    {
        state.Gravity = runtime.Gravity;
        state.EdgeLifetimeSeconds = Math.Max(runtime.EdgeLifetime, 0.1f);
        int targetCount = Math.Clamp(runtime.EstimatedEdgeCount, 0, MaxRenderedRibbonEdgesPerEmitter);
        if (targetCount <= 0)
            return;

        float spacing = state.EdgeLifetimeSeconds / Math.Max(targetCount, 1);
        for (int index = 0; index < targetCount; index++)
        {
            float t = targetCount == 1 ? 0.0f : index / (float)Math.Max(targetCount - 1, 1);
            Vector3 position = runtime.Position - new Vector3(0.0f, t * Math.Max(runtime.HeightAbove + runtime.HeightBelow, 0.2f), t * runtime.Gravity * 0.05f);
            state.Points.Add(new SimulatedRibbonPoint(position, spacing * index));
        }
    }

    private Matrix4x4 ResolveParticleEmitterTransform(MdxPreviewLoadResult preview, MdxParticleEmitter2 definition, Vector3 localPosition)
    {
        Matrix4x4 transform = Matrix4x4.CreateTranslation(localPosition);
        if (!definition.HasParent)
            return transform;

        if (!TryGetParentBoneMatrix(preview.Bones, definition.ParentId, out Matrix4x4 parentBoneMatrix))
            return transform;

        return transform * parentBoneMatrix;
    }

    private bool TryGetParentBoneMatrix(MdxBoneFile boneFile, int parentObjectId, out Matrix4x4 parentBoneMatrix)
    {
        for (int index = 0; index < boneFile.Bones.Count && index < _boneMatrices.Length; index++)
        {
            if (boneFile.Bones[index].ObjectId != parentObjectId)
                continue;

            parentBoneMatrix = _boneMatrices[index];
            return true;
        }

        parentBoneMatrix = Matrix4x4.Identity;
        return false;
    }

    private MdxParticleEmitter2? TryGetParticleDefinition(MdxPreviewLoadResult preview, int index)
    {
        if (index >= 0 && index < preview.ParticleEmitters.ParticleEmitterCount)
            return preview.ParticleEmitters.ParticleEmitters[index];

        return preview.ParticleEmitters.ParticleEmitters.FirstOrDefault(candidate => candidate.Index == index);
    }

    private MdxRibbonEmitter? TryGetRibbonDefinition(MdxPreviewLoadResult preview, int index)
    {
        if (index >= 0 && index < preview.Ribbons.RibbonCount)
            return preview.Ribbons.Ribbons[index];

        return preview.Ribbons.Ribbons.FirstOrDefault(candidate => candidate.Index == index);
    }

    private static Vector3 ResolveRibbonPosition(MdxRibbonEmitter ribbon, MdxSummary summary, int sequenceIndex, int timeMs)
    {
        return ribbon.PivotPoint + MdxAnimationSampler.SampleVector3Track(ribbon.TranslationTrack, summary, sequenceIndex, timeMs, Vector3.Zero);
    }

    private bool TryGetOrLoadParticleTexture(MdxPreviewLoadRequest request, MdxSummary summary, MdxParticleEmitter2RuntimeState particle, out uint textureId)
    {
        textureId = 0;

        if (particle.TextureId >= 0 && particle.TextureId < summary.Textures.Count)
        {
            MdxTextureSummary texture = summary.Textures[particle.TextureId];
            foreach (string candidate in EnumerateTextureCandidates(request, texture.Path, texture.ReplaceableId))
            {
                if (!TryGetOrLoadTexture(request, candidate, out uint loadedTextureId))
                    continue;

                textureId = loadedTextureId;
                return true;
            }
        }

        foreach (string candidate in EnumerateTextureCandidates(request, null, particle.ReplaceableId))
        {
            if (!TryGetOrLoadTexture(request, candidate, out uint loadedTextureId))
                continue;

            textureId = loadedTextureId;
            return true;
        }

        return false;
    }

    private static IEnumerable<string> EnumerateTextureCandidates(MdxPreviewLoadRequest request, string? texturePath, uint replaceableId)
    {
        if (!string.IsNullOrWhiteSpace(texturePath))
            yield return texturePath;

        if (replaceableId == 0)
            yield break;

        foreach (string candidate in EnumerateReplaceableTextureCandidates(request, replaceableId))
            yield return candidate;
    }

    private CommandBuffers CreateCommandBuffers(
        float[] vertexData,
        float[] skinningVertexData,
        ushort[] indices,
        uint textureId,
        bool hasTexture,
        bool isTransparent,
        bool isAdditive,
        bool depthTest,
        bool depthWrite,
        bool alphaCutout,
        bool receivesLighting,
        bool usesSphereEnvMap,
        bool usesBoneSkinning,
        bool usesUvTransform,
        Vector2 uvTranslation,
        Vector2 uvScale,
        Vector2 uvRotationRow0,
        Vector2 uvRotationRow1,
        Vector3 baseColor,
        Vector3 emissiveColor,
        float alpha,
        uint blendMode,
        int transparentSortPriority = 0,
        Vector3? transparentSortCenter = null)
    {
        uint vao = _gl.GenVertexArray();
        uint vbo = _gl.GenBuffer();
        uint skinningVbo = _gl.GenBuffer();
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
            fixed (ushort* indexPtr = indices)
            {
                _gl.BufferData(BufferTargetARB.ElementArrayBuffer, (nuint)(indices.Length * sizeof(ushort)), indexPtr, BufferUsageARB.StaticDraw);
            }

            _gl.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 8u * sizeof(float), (void*)0);
            _gl.EnableVertexAttribArray(0);
            _gl.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, 8u * sizeof(float), (void*)(3 * sizeof(float)));
            _gl.EnableVertexAttribArray(1);
            _gl.VertexAttribPointer(2, 2, VertexAttribPointerType.Float, false, 8u * sizeof(float), (void*)(6 * sizeof(float)));
            _gl.EnableVertexAttribArray(2);

            _gl.BindBuffer(BufferTargetARB.ArrayBuffer, skinningVbo);
            fixed (float* skinningPtr = skinningVertexData)
            {
                _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(skinningVertexData.Length * sizeof(float)), skinningPtr, BufferUsageARB.StaticDraw);
            }

            _gl.VertexAttribPointer(3, 4, VertexAttribPointerType.Float, false, 8u * sizeof(float), (void*)0);
            _gl.EnableVertexAttribArray(3);
            _gl.VertexAttribPointer(4, 4, VertexAttribPointerType.Float, false, 8u * sizeof(float), (void*)(4 * sizeof(float)));
            _gl.EnableVertexAttribArray(4);
        }

        _gl.BindVertexArray(0);

        return new CommandBuffers(
            vao,
            vbo,
            skinningVbo,
            ebo,
            (uint)indices.Length,
            textureId,
            hasTexture,
            isTransparent,
            isAdditive,
            depthTest,
            depthWrite,
            alphaCutout,
            receivesLighting,
            usesSphereEnvMap,
            usesBoneSkinning,
            usesUvTransform,
            uvTranslation,
            uvScale,
            uvRotationRow0,
            uvRotationRow1,
            baseColor,
            emissiveColor,
            alpha,
                blendMode,
                transparentSortPriority,
                transparentSortCenter ?? Vector3.Zero);
    }

    private static float Lerp(float start, float end, float amount) => start + ((end - start) * amount);

    private void ConfigureBlendMode(bool isAdditive, uint blendMode)
    {
        if (isAdditive || blendMode is MdxBlendModeAdditive or MdxBlendModeAddAlpha)
        {
            _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.One);
            return;
        }

        if (blendMode is MdxBlendModeModulate or MdxBlendModeModulate2X)
        {
            _gl.BlendFunc(BlendingFactor.DstColor, BlendingFactor.Zero);
            return;
        }

        _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
    }

    private static void ResolveAlphaHandling(CommandBuffers command, out bool useTextureAlpha, out bool premultiplyAlpha, out float alphaThreshold)
    {
        if (command.AlphaCutout)
        {
            useTextureAlpha = true;
            premultiplyAlpha = false;
            alphaThreshold = 0.75f;
            return;
        }

        if (!command.IsTransparent)
        {
            useTextureAlpha = false;
            premultiplyAlpha = false;
            alphaThreshold = 0.0f;
            return;
        }

        switch (command.BlendMode)
        {
            case MdxBlendModeBlend:
                useTextureAlpha = true;
                premultiplyAlpha = true;
                alphaThreshold = 0.0f;
                return;
            case MdxBlendModeAdditive:
            case MdxBlendModeAddAlpha:
                useTextureAlpha = true;
                premultiplyAlpha = false;
                alphaThreshold = 0.0f;
                return;
            case MdxBlendModeModulate:
            case MdxBlendModeModulate2X:
                useTextureAlpha = false;
                premultiplyAlpha = false;
                alphaThreshold = 0.0f;
                return;
            case 0:
                useTextureAlpha = false;
                premultiplyAlpha = false;
                alphaThreshold = 0.05f;
                return;
            default:
                useTextureAlpha = true;
                premultiplyAlpha = false;
                alphaThreshold = 0.05f;
                return;
        }
    }

    private static void ResolveBounds(MdxGeometryFile geometry, MdxSummary summary, out Vector3 min, out Vector3 max)
    {
        if (TryComputeRawRenderableBounds(geometry, out Vector3 rawMin, out Vector3 rawMax))
        {
            min = rawMin;
            max = rawMax;

            if (NeedsEffectAwareFraming(summary)
                && TryGetEffectAwareDeclaredBounds(summary, rawMin, rawMax, out Vector3 declaredMin, out Vector3 declaredMax))
            {
                min = Vector3.Min(min, declaredMin);
                max = Vector3.Max(max, declaredMax);
            }

            return;
        }

        if (summary.BoundsMin is Vector3 summaryMin && summary.BoundsMax is Vector3 summaryMax)
        {
            Vector3 summaryExtent = summaryMax - summaryMin;
            if (float.IsFinite(summaryExtent.X) && float.IsFinite(summaryExtent.Y) && float.IsFinite(summaryExtent.Z)
                && summaryExtent.LengthSquared() > 0.0001f)
            {
                min = summaryMin;
                max = summaryMax;
                return;
            }
        }

        bool found = false;
        min = new Vector3(float.MaxValue);
        max = new Vector3(float.MinValue);
        foreach (MdxGeosetGeometry geoset in geometry.Geosets)
        {
            if (geoset.BoundsMin is Vector3 geosetMin && geoset.BoundsMax is Vector3 geosetMax)
            {
                min = Vector3.Min(min, geosetMin);
                max = Vector3.Max(max, geosetMax);
                found = true;
            }
        }

        if (!found)
        {
            min = new Vector3(-1.0f, -1.0f, -1.0f);
            max = new Vector3(1.0f, 1.0f, 1.0f);
        }
    }

    private static bool TryComputeRawRenderableBounds(MdxGeometryFile geometry, out Vector3 min, out Vector3 max)
    {
        min = new Vector3(float.MaxValue);
        max = new Vector3(float.MinValue);
        bool found = false;

        foreach (MdxGeosetGeometry geoset in geometry.Geosets)
        {
            foreach (Vector3 vertex in geoset.Vertices)
            {
                if (!float.IsFinite(vertex.X) || !float.IsFinite(vertex.Y) || !float.IsFinite(vertex.Z))
                    continue;

                min = Vector3.Min(min, vertex);
                max = Vector3.Max(max, vertex);
                found = true;
            }
        }

        return found;
    }

    private static bool NeedsEffectAwareFraming(MdxSummary summary) => summary.ParticleEmitter2Count > 0 || summary.RibbonCount > 0;

    private static bool TryGetEffectAwareDeclaredBounds(MdxSummary summary, Vector3 rawMin, Vector3 rawMax, out Vector3 min, out Vector3 max)
    {
        min = default;
        max = default;
        bool found = false;
        float bestDiagonal = float.MaxValue;

        if (TryConsiderDeclaredBounds(summary.BoundsMin, summary.BoundsMax, rawMin, rawMax, ref found, ref bestDiagonal, ref min, ref max))
            found = true;

        foreach (MdxSequenceSummary sequence in summary.Sequences)
        {
            if (TryConsiderDeclaredBounds(sequence.BoundsMin, sequence.BoundsMax, rawMin, rawMax, ref found, ref bestDiagonal, ref min, ref max))
                found = true;
        }

        return found;
    }

    private static bool TryConsiderDeclaredBounds(
        Vector3? candidateMin,
        Vector3? candidateMax,
        Vector3 rawMin,
        Vector3 rawMax,
        ref bool found,
        ref float bestDiagonal,
        ref Vector3 bestMin,
        ref Vector3 bestMax)
    {
        if (candidateMin is not Vector3 min || candidateMax is not Vector3 max)
            return false;

        Vector3 extent = max - min;
        if (!float.IsFinite(extent.X) || !float.IsFinite(extent.Y) || !float.IsFinite(extent.Z) || extent.LengthSquared() <= 0.0001f)
            return false;

        if (!ContainsBounds(min, max, rawMin, rawMax))
            return false;

        float candidateDiagonal = extent.Length();
        float rawDiagonal = (rawMax - rawMin).Length();
        if (candidateDiagonal <= rawDiagonal * 1.05f)
            return false;

        if (!found || candidateDiagonal < bestDiagonal)
        {
            bestDiagonal = candidateDiagonal;
            bestMin = min;
            bestMax = max;
            return true;
        }

        return false;
    }

    private static bool ContainsBounds(Vector3 outerMin, Vector3 outerMax, Vector3 innerMin, Vector3 innerMax)
    {
        const float epsilon = 0.01f;
        return innerMin.X >= outerMin.X - epsilon && innerMax.X <= outerMax.X + epsilon
            && innerMin.Y >= outerMin.Y - epsilon && innerMax.Y <= outerMax.Y + epsilon
            && innerMin.Z >= outerMin.Z - epsilon && innerMax.Z <= outerMax.Z + epsilon;
    }

    private bool TryGetOrLoadMaterialTexture(MdxPreviewLoadRequest request, MdxResolvedMaterialState material, out uint textureId)
    {
        textureId = 0;

        foreach (string candidate in EnumerateTextureCandidates(request, material))
        {
            if (!TryGetOrLoadTexture(request, candidate, out uint loadedTextureId))
                continue;

            textureId = loadedTextureId;
            return true;
        }

        return false;
    }

    private static IEnumerable<string> EnumerateTextureCandidates(MdxPreviewLoadRequest request, MdxResolvedMaterialState material)
    {
        if (!string.IsNullOrWhiteSpace(material.TexturePath))
            yield return material.TexturePath;

        if (material.ReplaceableId == 0)
            yield break;

        foreach (string candidate in EnumerateReplaceableTextureCandidates(request, material.ReplaceableId))
            yield return candidate;
    }

    private static IEnumerable<string> EnumerateReplaceableTextureCandidates(MdxPreviewLoadRequest request, uint replaceableId)
    {
        string? modelPath = request.UsesArchiveSource ? request.VirtualPath : request.InputPath;
        if (!string.IsNullOrWhiteSpace(modelPath))
        {
            string normalizedModelPath = modelPath.Replace('/', '\\');
            string? modelDirectory = Path.GetDirectoryName(normalizedModelPath);
            string? modelBaseName = Path.GetFileNameWithoutExtension(normalizedModelPath);
            if (!string.IsNullOrWhiteSpace(modelDirectory) && !string.IsNullOrWhiteSpace(modelBaseName))
            {
                foreach (string sameDirectoryCandidate in EnumerateSameDirectoryReplaceableCandidates(modelDirectory, modelBaseName, replaceableId))
                    yield return sameDirectoryCandidate;
            }
        }

        if (DefaultReplaceableTextures.TryGetValue(replaceableId, out string? fallbackTexturePath))
            yield return fallbackTexturePath;
    }

    private static IEnumerable<string> EnumerateSameDirectoryReplaceableCandidates(string modelDirectory, string modelBaseName, uint replaceableId)
    {
        int? skinIndex = replaceableId switch
        {
            1 or 11 => 1,
            2 or 12 => 2,
            3 or 13 => 3,
            _ => null,
        };

        if (!skinIndex.HasValue)
            yield break;

        yield return $"{modelDirectory}\\{modelBaseName}_Skin{skinIndex.Value:00}.blp";
        yield return $"{modelDirectory}\\{modelBaseName}Skin{skinIndex.Value:00}.blp";
        yield return $"{modelDirectory}\\Skin{skinIndex.Value:00}.blp";
    }

    private void EnsureFramebuffer(int width, int height)
    {
        int resolvedWidth = Math.Max(64, width);
        int resolvedHeight = Math.Max(64, height);
        if (_framebuffer != 0 && _frameWidth == resolvedWidth && _frameHeight == resolvedHeight)
            return;

        DeleteFramebuffer();

        _frameWidth = resolvedWidth;
        _frameHeight = resolvedHeight;
        _framebuffer = _gl.GenFramebuffer();
        _colorTexture = _gl.GenTexture();
        _depthRenderbuffer = _gl.GenRenderbuffer();

        _gl.BindFramebuffer(GLEnum.Framebuffer, _framebuffer);
        _gl.BindTexture(TextureTarget.Texture2D, _colorTexture);
        _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba8, (uint)_frameWidth, (uint)_frameHeight, 0, PixelFormat.Rgba, PixelType.UnsignedByte, ReadOnlySpan<byte>.Empty);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)GLEnum.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)GLEnum.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)GLEnum.ClampToEdge);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)GLEnum.ClampToEdge);
        _gl.FramebufferTexture2D(GLEnum.Framebuffer, GLEnum.ColorAttachment0, GLEnum.Texture2D, _colorTexture, 0);

        _gl.BindRenderbuffer(GLEnum.Renderbuffer, _depthRenderbuffer);
        _gl.RenderbufferStorage(GLEnum.Renderbuffer, GLEnum.DepthComponent24, (uint)_frameWidth, (uint)_frameHeight);
        _gl.FramebufferRenderbuffer(GLEnum.Framebuffer, GLEnum.DepthAttachment, GLEnum.Renderbuffer, _depthRenderbuffer);

        GLEnum status = _gl.CheckFramebufferStatus(GLEnum.Framebuffer);
        _gl.BindFramebuffer(GLEnum.Framebuffer, 0);
        _gl.BindTexture(TextureTarget.Texture2D, 0);
        _gl.BindRenderbuffer(GLEnum.Renderbuffer, 0);
        if (status != GLEnum.FramebufferComplete)
            throw new InvalidOperationException($"MDX GPU preview framebuffer is incomplete: {status}.");
    }

    private void DeleteFramebuffer()
    {
        if (_colorTexture != 0)
        {
            _gl.DeleteTexture(_colorTexture);
            _colorTexture = 0;
        }

        if (_depthRenderbuffer != 0)
        {
            _gl.DeleteRenderbuffer(_depthRenderbuffer);
            _depthRenderbuffer = 0;
        }

        if (_framebuffer != 0)
        {
            _gl.DeleteFramebuffer(_framebuffer);
            _framebuffer = 0;
        }

        _frameWidth = 0;
        _frameHeight = 0;
    }

    private void InitializeShader()
    {
        const string vertexSource = """
            #version 330 core
            layout (location = 0) in vec3 aPos;
            layout (location = 1) in vec3 aNormal;
            layout (location = 2) in vec2 aTexCoord;
            layout (location = 3) in vec4 aBoneIndices;
            layout (location = 4) in vec4 aBoneWeights;

            uniform mat4 uView;
            uniform mat4 uProj;
            uniform bool uUseBoneSkinning;
            uniform int uBoneCount;
            uniform mat4 uBones[128];
            uniform bool uUseUvTransform;
            uniform vec2 uUvTranslation;
            uniform vec2 uUvScale;
            uniform vec2 uUvRotationRow0;
            uniform vec2 uUvRotationRow1;

            out vec3 vNormal;
            out vec3 vViewNormal;
            out vec2 vTexCoord;

            vec4 ApplySkinning(vec4 source, vec4 boneIndices, vec4 boneWeights)
            {
                if (!uUseBoneSkinning)
                    return source;

                float totalWeight = boneWeights.x + boneWeights.y + boneWeights.z + boneWeights.w;
                if (totalWeight <= 0.0001)
                    return source;

                vec4 skinned = vec4(0.0);
                bool applied = false;

                int index0 = int(aBoneIndices.x + 0.5);
                int index1 = int(aBoneIndices.y + 0.5);
                int index2 = int(aBoneIndices.z + 0.5);
                int index3 = int(aBoneIndices.w + 0.5);

                if (boneWeights.x > 0.0 && index0 >= 0 && index0 < uBoneCount && index0 < 128)
                {
                    skinned += (uBones[index0] * source) * (boneWeights.x / totalWeight);
                    applied = true;
                }

                if (boneWeights.y > 0.0 && index1 >= 0 && index1 < uBoneCount && index1 < 128)
                {
                    skinned += (uBones[index1] * source) * (boneWeights.y / totalWeight);
                    applied = true;
                }

                if (boneWeights.z > 0.0 && index2 >= 0 && index2 < uBoneCount && index2 < 128)
                {
                    skinned += (uBones[index2] * source) * (boneWeights.z / totalWeight);
                    applied = true;
                }

                if (boneWeights.w > 0.0 && index3 >= 0 && index3 < uBoneCount && index3 < 128)
                {
                    skinned += (uBones[index3] * source) * (boneWeights.w / totalWeight);
                    applied = true;
                }

                return applied ? skinned : source;
            }

            void main()
            {
                vec4 skinnedPosition = ApplySkinning(vec4(aPos, 1.0), aBoneIndices, aBoneWeights);
                vec3 skinnedNormal = ApplySkinning(vec4(aNormal, 0.0), aBoneIndices, aBoneWeights).xyz;
                gl_Position = uProj * uView * skinnedPosition;
                vNormal = skinnedNormal;
                vViewNormal = mat3(uView) * vNormal;
                vec2 texCoord = aTexCoord;
                if (uUseUvTransform)
                {
                    vec2 centered = (texCoord - vec2(0.5, 0.5)) * uUvScale;
                    texCoord = vec2(
                        dot(centered, uUvRotationRow0),
                        dot(centered, uUvRotationRow1)) + vec2(0.5, 0.5) + uUvTranslation;
                }

                vTexCoord = texCoord;
            }
            """;

        const string fragmentSource = """
            #version 330 core
            in vec3 vNormal;
            in vec3 vViewNormal;
            in vec2 vTexCoord;

            uniform vec3 uLightDir;
            uniform vec3 uLightColor;
            uniform vec3 uAmbientColor;
            uniform vec3 uBaseColor;
            uniform vec3 uEmissiveColor;
            uniform float uAlpha;
            uniform bool uHasTexture;
            uniform sampler2D uTexture0;
            uniform bool uAlphaCutout;
            uniform float uAlphaThreshold;
            uniform bool uReceivesLighting;
            uniform bool uUseTextureAlpha;
            uniform bool uPremultiplyAlpha;
            uniform bool uSphereEnvMap;
            out vec4 FragColor;

            void main()
            {
                vec2 texCoord = vTexCoord;
                if (uSphereEnvMap)
                {
                    vec3 viewNormal = normalize(vViewNormal);
                    if (!gl_FrontFacing)
                        viewNormal = -viewNormal;

                    texCoord = viewNormal.xy * 0.5 + 0.5;
                }

                vec4 texel = uHasTexture ? texture(uTexture0, texCoord) : vec4(1.0, 1.0, 1.0, 1.0);
                vec3 texRgb = texel.rgb;
                if (uPremultiplyAlpha)
                    texRgb *= texel.a;

                float sampledAlpha = uUseTextureAlpha ? texel.a : 1.0;
                float finalAlpha = clamp(sampledAlpha * uAlpha, 0.0, 1.0);
                if ((uAlphaCutout || uAlphaThreshold > 0.0) && finalAlpha < uAlphaThreshold)
                    discard;

                vec3 shaded = texRgb * uBaseColor;
                if (uReceivesLighting)
                {
                    vec3 normal = normalize(vNormal);
                    vec3 lightDir = normalize(uLightDir);
                    float NdotL = dot(normal, lightDir);
                    float diffuse = NdotL * 0.5 + 0.5;
                    diffuse = diffuse * diffuse;
                    shaded *= clamp(uAmbientColor + (uLightColor * diffuse), vec3(0.0), vec3(1.75));
                }

                shaded += uEmissiveColor;
                FragColor = vec4(shaded, finalAlpha);
            }
            """;

        uint vertexShader = CompileShader(ShaderType.VertexShader, vertexSource);
        uint fragmentShader = CompileShader(ShaderType.FragmentShader, fragmentSource);

        _shaderProgram = _gl.CreateProgram();
        _gl.AttachShader(_shaderProgram, vertexShader);
        _gl.AttachShader(_shaderProgram, fragmentShader);
        _gl.LinkProgram(_shaderProgram);
        _gl.GetProgram(_shaderProgram, GLEnum.LinkStatus, out int linked);
        if (linked == 0)
        {
            string info = _gl.GetProgramInfoLog(_shaderProgram);
            _gl.DeleteShader(vertexShader);
            _gl.DeleteShader(fragmentShader);
            throw new InvalidOperationException($"Failed to link MDX GPU preview shader program: {info}");
        }

        _gl.DetachShader(_shaderProgram, vertexShader);
        _gl.DetachShader(_shaderProgram, fragmentShader);
        _gl.DeleteShader(vertexShader);
        _gl.DeleteShader(fragmentShader);

        _uView = _gl.GetUniformLocation(_shaderProgram, "uView");
        _uProj = _gl.GetUniformLocation(_shaderProgram, "uProj");
        _uLightDir = _gl.GetUniformLocation(_shaderProgram, "uLightDir");
        _uLightColor = _gl.GetUniformLocation(_shaderProgram, "uLightColor");
        _uAmbientColor = _gl.GetUniformLocation(_shaderProgram, "uAmbientColor");
        _uBaseColor = _gl.GetUniformLocation(_shaderProgram, "uBaseColor");
        _uEmissiveColor = _gl.GetUniformLocation(_shaderProgram, "uEmissiveColor");
        _uAlpha = _gl.GetUniformLocation(_shaderProgram, "uAlpha");
        _uHasTexture = _gl.GetUniformLocation(_shaderProgram, "uHasTexture");
        _uTexture0 = _gl.GetUniformLocation(_shaderProgram, "uTexture0");
        _uAlphaCutout = _gl.GetUniformLocation(_shaderProgram, "uAlphaCutout");
        _uAlphaThreshold = _gl.GetUniformLocation(_shaderProgram, "uAlphaThreshold");
        _uReceivesLighting = _gl.GetUniformLocation(_shaderProgram, "uReceivesLighting");
        _uUseTextureAlpha = _gl.GetUniformLocation(_shaderProgram, "uUseTextureAlpha");
        _uPremultiplyAlpha = _gl.GetUniformLocation(_shaderProgram, "uPremultiplyAlpha");
        _uSphereEnvMap = _gl.GetUniformLocation(_shaderProgram, "uSphereEnvMap");
        _uUseBoneSkinning = _gl.GetUniformLocation(_shaderProgram, "uUseBoneSkinning");
        _uBoneCount = _gl.GetUniformLocation(_shaderProgram, "uBoneCount");
        _uBones = _gl.GetUniformLocation(_shaderProgram, "uBones[0]");
        _uUseUvTransform = _gl.GetUniformLocation(_shaderProgram, "uUseUvTransform");
        _uUvTranslation = _gl.GetUniformLocation(_shaderProgram, "uUvTranslation");
        _uUvScale = _gl.GetUniformLocation(_shaderProgram, "uUvScale");
        _uUvRotationRow0 = _gl.GetUniformLocation(_shaderProgram, "uUvRotationRow0");
        _uUvRotationRow1 = _gl.GetUniformLocation(_shaderProgram, "uUvRotationRow1");
    }

    private uint CompileShader(ShaderType shaderType, string source)
    {
        uint shader = _gl.CreateShader(shaderType);
        _gl.ShaderSource(shader, source);
        _gl.CompileShader(shader);
        _gl.GetShader(shader, ShaderParameterName.CompileStatus, out int compiled);
        if (compiled == 0)
        {
            string info = _gl.GetShaderInfoLog(shader);
            _gl.DeleteShader(shader);
            throw new InvalidOperationException($"Failed to compile MDX GPU preview {shaderType}: {info}");
        }

        return shader;
    }

    private uint CreateFallbackWhiteTexture()
    {
        byte[] whitePixel = [255, 255, 255, 255];
        uint textureId = _gl.GenTexture();
        _gl.BindTexture(TextureTarget.Texture2D, textureId);
        unsafe
        {
            fixed (byte* pixelPtr = whitePixel)
            {
                _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba8, 1, 1, 0, PixelFormat.Rgba, PixelType.UnsignedByte, pixelPtr);
            }
        }

        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)GLEnum.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)GLEnum.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)GLEnum.Repeat);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)GLEnum.Repeat);
        _gl.BindTexture(TextureTarget.Texture2D, 0);
        return textureId;
    }

    private bool TryGetOrLoadTexture(MdxPreviewLoadRequest request, string texturePath, out uint textureId)
    {
        string cacheKey = texturePath.Replace('/', '\\').ToLowerInvariant();
        if (_loadedTextureCache.TryGetValue(cacheKey, out textureId))
            return textureId != 0;

        if (!TryReadTextureBytes(request, texturePath, out byte[]? bytes) || bytes == null || bytes.Length == 0)
        {
            _loadedTextureCache[cacheKey] = 0;
            textureId = 0;
            return false;
        }

        try
        {
            using MemoryStream stream = new(bytes, writable: false);
            using BlpFile blp = new(stream);
            byte[] rgbaPixels = blp.GetPixels(0, out int width, out int height, bgra: false);
            textureId = UploadTexture(rgbaPixels, width, height);
            _loadedTextureCache[cacheKey] = textureId;
            _ownedTextureIds.Add(textureId);
            return textureId != 0;
        }
        catch
        {
            _loadedTextureCache[cacheKey] = 0;
            textureId = 0;
            return false;
        }
    }

    private bool TryReadTextureBytes(MdxPreviewLoadRequest request, string texturePath, out byte[]? bytes)
    {
        bytes = null;

        if (request.UsesArchiveSource)
        {
            try
            {
                bytes = ArchiveVirtualFileReader.ReadVirtualFile(texturePath, [request.ArchiveRoot!], new ArchiveCatalogBootstrapOptions());
                return bytes.Length > 0;
            }
            catch
            {
                return false;
            }
        }

        if (string.IsNullOrWhiteSpace(request.InputPath))
            return false;

        string normalized = texturePath.Replace('/', '\\').TrimStart('\\');
        string? modelDirectory = Path.GetDirectoryName(Path.GetFullPath(request.InputPath));
        string[] candidates =
        [
            normalized,
            modelDirectory == null ? normalized : Path.Combine(modelDirectory, Path.GetFileName(normalized)),
            modelDirectory == null ? normalized : Path.Combine(modelDirectory, normalized),
        ];

        foreach (string candidate in candidates.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            if (!File.Exists(candidate))
                continue;

            bytes = File.ReadAllBytes(candidate);
            return bytes.Length > 0;
        }

        return false;
    }

    private uint UploadTexture(byte[] rgbaPixels, int width, int height)
    {
        uint textureId = _gl.GenTexture();
        _gl.BindTexture(TextureTarget.Texture2D, textureId);
        unsafe
        {
            fixed (byte* pixelPtr = rgbaPixels)
            {
                _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba8, (uint)width, (uint)height, 0, PixelFormat.Rgba, PixelType.UnsignedByte, pixelPtr);
            }
        }

        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)GLEnum.LinearMipmapLinear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)GLEnum.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)GLEnum.Repeat);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)GLEnum.Repeat);
        _gl.GenerateMipmap(TextureTarget.Texture2D);
        _gl.BindTexture(TextureTarget.Texture2D, 0);
        return textureId;
    }

    private sealed class CommandBuffers
    {
        public CommandBuffers(
            uint vao,
            uint vbo,
            uint skinningVbo,
            uint ebo,
            uint indexCount,
            uint textureId,
            bool hasTexture,
            bool isTransparent,
            bool isAdditive,
            bool depthTest,
            bool depthWrite,
            bool alphaCutout,
            bool receivesLighting,
            bool usesSphereEnvMap,
            bool usesBoneSkinning,
            bool usesUvTransform,
            Vector2 uvTranslation,
            Vector2 uvScale,
            Vector2 uvRotationRow0,
            Vector2 uvRotationRow1,
            Vector3 baseColor,
            Vector3 emissiveColor,
            float alpha,
            uint blendMode,
            int transparentSortPriority,
            Vector3 transparentSortCenter)
        {
            Vao = vao;
            Vbo = vbo;
            SkinningVbo = skinningVbo;
            Ebo = ebo;
            IndexCount = indexCount;
            TextureId = textureId;
            HasTexture = hasTexture;
            IsTransparent = isTransparent;
            IsAdditive = isAdditive;
            DepthTest = depthTest;
            DepthWrite = depthWrite;
            AlphaCutout = alphaCutout;
            ReceivesLighting = receivesLighting;
            UsesSphereEnvMap = usesSphereEnvMap;
            UsesBoneSkinning = usesBoneSkinning;
            UsesUvTransform = usesUvTransform;
            UvTranslation = uvTranslation;
            UvScale = uvScale;
            UvRotationRow0 = uvRotationRow0;
            UvRotationRow1 = uvRotationRow1;
            BaseColor = baseColor;
            EmissiveColor = emissiveColor;
            Alpha = alpha;
            BlendMode = blendMode;
            TransparentSortPriority = transparentSortPriority;
            TransparentSortCenter = transparentSortCenter;
        }

        public uint Vao { get; }

        public uint Vbo { get; }

        public uint SkinningVbo { get; }

        public uint Ebo { get; }

        public uint IndexCount { get; }

        public uint TextureId { get; }

        public bool HasTexture { get; }

        public bool IsTransparent { get; }

        public bool IsAdditive { get; }

        public bool DepthTest { get; }

        public bool DepthWrite { get; }

        public bool AlphaCutout { get; }

        public bool ReceivesLighting { get; }

        public bool UsesSphereEnvMap { get; }

        public bool UsesBoneSkinning { get; }

        public bool UsesUvTransform { get; }

        public Vector2 UvTranslation { get; }

        public Vector2 UvScale { get; }

        public Vector2 UvRotationRow0 { get; }

        public Vector2 UvRotationRow1 { get; }

        public Vector3 BaseColor { get; }

        public Vector3 EmissiveColor { get; }

        public float Alpha { get; }

        public uint BlendMode { get; }

        public int TransparentSortPriority { get; }

        public Vector3 TransparentSortCenter { get; }

        public void Dispose(GL gl)
        {
            gl.DeleteBuffer(Vbo);
            gl.DeleteBuffer(SkinningVbo);
            gl.DeleteBuffer(Ebo);
            gl.DeleteVertexArray(Vao);
        }
    }

    private sealed class ParticleEmitterSimulationState
    {
        public ParticleEmitterSimulationState(int emitterIndex)
        {
            Random = new Random(unchecked(Environment.TickCount * 31 + emitterIndex));
        }

        public List<SimulatedParticle> Particles { get; } = new();

        public Random Random { get; }

        public float TimeSinceLastEmitSeconds { get; set; }

        public float Gravity { get; set; }

        public Matrix4x4 Transform { get; set; } = Matrix4x4.Identity;

        public bool IsActive { get; set; }
    }

    private sealed class RibbonEmitterSimulationState
    {
        public List<SimulatedRibbonPoint> Points { get; } = new();

        public float EdgeAccumulator { get; set; }

        public float EdgeLifetimeSeconds { get; set; } = 0.1f;

        public float Gravity { get; set; }
    }

    private struct SimulatedParticle
    {
        public SimulatedParticle(Vector3 position, Vector3 velocity, float ageSeconds, float lifetimeSeconds)
        {
            Position = position;
            Velocity = velocity;
            AgeSeconds = ageSeconds;
            LifetimeSeconds = lifetimeSeconds;
        }

        public Vector3 Position { get; set; }

        public Vector3 Velocity { get; set; }

        public float AgeSeconds { get; set; }

        public float LifetimeSeconds { get; set; }
    }

    private struct SimulatedRibbonPoint
    {
        public SimulatedRibbonPoint(Vector3 position, float ageSeconds)
        {
            Position = position;
            AgeSeconds = ageSeconds;
        }

        public Vector3 Position { get; set; }

        public float AgeSeconds { get; set; }
    }

}
