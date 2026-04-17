using System.Numerics;
using SereniaBLPLib;
using Silk.NET.OpenGL;
using WowViewer.Core.IO.Files;
using WowViewer.Core.Mdx;

namespace WowViewer.App;

internal sealed class MdxGpuPreviewRenderer : IDisposable
{
    private const uint MdxBlendModeTransparentKey = 1;
    private const uint MdxBlendModeAdditive = 3;
    private const uint MdxBlendModeAddAlpha = 4;
    private const uint MdxBlendModeModulate = 5;
    private const uint MdxBlendModeModulate2X = 6;
    private const float PreviewFieldOfViewDegrees = 25.0f;
    private const float PreviewPaddingScale = 1.04f;
    private const float PreviewZoomFactor = 0.72f;

    private readonly GL _gl;
    private readonly Dictionary<string, uint> _loadedTextureCache = new(StringComparer.OrdinalIgnoreCase);
    private readonly HashSet<uint> _ownedTextureIds = new();
    private readonly List<CommandBuffers> _commands = new();

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
    private int _uReceivesLighting;

    private uint _framebuffer;
    private uint _colorTexture;
    private uint _depthRenderbuffer;
    private uint _fallbackWhiteTexture;
    private int _frameWidth;
    private int _frameHeight;
    private MdxSummary? _currentSummary;
    private PreviewCameraSettings _cameraSettings = new();
    private Vector3 _boundsMin = new(-1.0f, -1.0f, -1.0f);
    private Vector3 _boundsMax = new(1.0f, 1.0f, 1.0f);
    private Vector3 _ambientColor = new(0.35f, 0.35f, 0.4f);
    private Vector3 _lightColor = new(1.0f, 0.95f, 0.85f);
    private readonly Vector3 _lightDir = Vector3.Normalize(new Vector3(0.5f, 0.3f, 1.0f));
    private bool _disposed;

    public MdxGpuPreviewRenderer(GL gl)
    {
        _gl = gl ?? throw new ArgumentNullException(nameof(gl));
        InitializeShader();
        _fallbackWhiteTexture = CreateFallbackWhiteTexture();
    }

    public uint PreviewTextureHandle => _colorTexture;

    public bool HasRenderableGeometry => _commands.Count > 0;

    public int CommandCount => _commands.Count;

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
        foreach (CommandBuffers command in _commands)
            command.Dispose(_gl);

        _commands.Clear();
        _currentSummary = null;
    }

    public void LoadPreview(MdxPreviewLoadResult preview)
    {
        ArgumentNullException.ThrowIfNull(preview);

        ClearPreview();
        _currentSummary = preview.Summary;
        _cameraSettings = preview.Request.Camera;
        ResolveBounds(preview.Geometry, preview.Summary, out _boundsMin, out _boundsMax);

        foreach (MdxGeosetGeometry geoset in preview.Geometry.Geosets)
        {
            if (geoset.Vertices.Count == 0 || geoset.Indices.Count < 3)
                continue;

            MdxMaterialState material = ResolveMaterial(preview.Summary, geoset.MaterialId);

            uint textureId = _fallbackWhiteTexture;
            bool hasTexture = false;
            if (!string.IsNullOrWhiteSpace(material.TexturePath) && TryGetOrLoadTexture(preview.Request, material.TexturePath, out uint loadedTextureId))
            {
                textureId = loadedTextureId;
                hasTexture = true;
            }

            float[] vertexData = new float[geoset.Vertices.Count * 8];
            IReadOnlyList<Vector2> uvSet = geoset.PrimaryUvSet;
            for (int index = 0; index < geoset.Vertices.Count; index++)
            {
                Vector3 position = geoset.Vertices[index];
                Vector3 normal = index < geoset.Normals.Count ? geoset.Normals[index] : Vector3.UnitZ;
                Vector2 uv = index < uvSet.Count ? uvSet[index] : Vector2.Zero;
                int offset = index * 8;
                vertexData[offset + 0] = position.X;
                vertexData[offset + 1] = position.Y;
                vertexData[offset + 2] = position.Z;
                vertexData[offset + 3] = normal.X;
                vertexData[offset + 4] = normal.Y;
                vertexData[offset + 5] = normal.Z;
                vertexData[offset + 6] = uv.X;
                vertexData[offset + 7] = uv.Y;
            }

            ushort[] indices = geoset.Indices.ToArray();
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
            }

            _gl.BindVertexArray(0);

            _commands.Add(new CommandBuffers(
                vao,
                vbo,
                ebo,
                (uint)indices.Length,
                textureId,
                hasTexture,
                material.IsTransparent,
                material.IsAdditive,
                material.DepthWrite,
                material.AlphaCutout,
                receivesLighting: true,
                Vector3.One,
                Vector3.Zero,
                material.Alpha,
                material.BlendMode));
        }
    }

    public void Render(int width, int height)
    {
        EnsureFramebuffer(width, height);

        _gl.BindFramebuffer(GLEnum.Framebuffer, _framebuffer);
        _gl.Viewport(0, 0, (uint)_frameWidth, (uint)_frameHeight);
        _gl.ClearColor(0.08f, 0.09f, 0.11f, 1.0f);
        _gl.Clear((uint)(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit));

        if (_commands.Count > 0)
        {
            PreviewCameraPose pose = PreviewCameraPlanner.CreatePose(_boundsMin, _boundsMax, _cameraSettings, _currentSummary, _frameWidth, _frameHeight);
            Matrix4x4 view = pose.View;
            Matrix4x4 projection = pose.Projection;
            RenderPass(view, projection, transparentPass: false);
            RenderPass(view, projection, transparentPass: true);
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

    private void RenderPass(Matrix4x4 view, Matrix4x4 projection, bool transparentPass)
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

        foreach (CommandBuffers command in _commands)
        {
            if (command.IsTransparent != transparentPass)
                continue;

            if (command.IsTransparent)
            {
                _gl.Enable(EnableCap.Blend);
                ConfigureBlendMode(command.IsAdditive, command.BlendMode);
            }
            else
            {
                _gl.Disable(EnableCap.Blend);
            }

            _gl.DepthMask(!command.IsTransparent || command.DepthWrite);

            _gl.Uniform3(_uBaseColor, command.BaseColor.X, command.BaseColor.Y, command.BaseColor.Z);
            _gl.Uniform3(_uEmissiveColor, command.EmissiveColor.X, command.EmissiveColor.Y, command.EmissiveColor.Z);
            _gl.Uniform1(_uAlpha, command.Alpha);
            _gl.Uniform1(_uHasTexture, command.HasTexture ? 1 : 0);
            _gl.Uniform1(_uAlphaCutout, command.AlphaCutout ? 1 : 0);
            _gl.Uniform1(_uReceivesLighting, command.ReceivesLighting ? 1 : 0);

            _gl.ActiveTexture(TextureUnit.Texture0);
            _gl.BindTexture(TextureTarget.Texture2D, command.TextureId);
            _gl.Uniform1(_uTexture0, 0);
            _gl.BindVertexArray(command.Vao);
            unsafe
            {
                _gl.DrawElements(PrimitiveType.Triangles, command.IndexCount, DrawElementsType.UnsignedShort, (void*)0);
            }
        }

        _gl.BindVertexArray(0);
        _gl.BindTexture(TextureTarget.Texture2D, 0);
        _gl.Disable(EnableCap.Blend);
        _gl.DepthMask(true);
        _gl.UseProgram(0);
    }

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

    private static MdxMaterialState ResolveMaterial(MdxSummary summary, int materialId)
    {
        if (materialId < 0 || materialId >= summary.Materials.Count)
            return new MdxMaterialState(null, false, false, true, false, 1.0f, 0);

        MdxMaterialSummary material = summary.Materials[materialId];
        if (material.LayerCount == 0)
            return new MdxMaterialState(null, false, false, true, false, 1.0f, 0);

        MdxMaterialLayerSummary layer = material.Layers[0];
        string? texturePath = layer.TextureId >= 0 && layer.TextureId < summary.Textures.Count
            ? summary.Textures[layer.TextureId].Path
            : null;
        float alpha = Math.Clamp(layer.StaticAlpha <= 0.0f ? 1.0f : layer.StaticAlpha, 0.0f, 1.0f);
        bool isTransparent = layer.BlendMode != 0 || alpha < 0.999f;
        bool alphaCutout = layer.BlendMode == MdxBlendModeTransparentKey;
        bool isAdditive = layer.BlendMode is MdxBlendModeAdditive or MdxBlendModeAddAlpha;
        bool depthWrite = !isTransparent || alphaCutout;
        return new MdxMaterialState(texturePath, isTransparent, isAdditive, depthWrite, alphaCutout, alpha, layer.BlendMode);
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

            uniform mat4 uView;
            uniform mat4 uProj;

            out vec3 vNormal;
            out vec2 vTexCoord;

            void main()
            {
                gl_Position = uProj * uView * vec4(aPos, 1.0);
                vNormal = aNormal;
                vTexCoord = aTexCoord;
            }
            """;

        const string fragmentSource = """
            #version 330 core
            in vec3 vNormal;
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
            uniform bool uReceivesLighting;

            out vec4 FragColor;

            void main()
            {
                vec4 texel = uHasTexture ? texture(uTexture0, vTexCoord) : vec4(1.0, 1.0, 1.0, 1.0);
                float finalAlpha = clamp(texel.a * uAlpha, 0.0, 1.0);
                if (uAlphaCutout && finalAlpha < 0.5)
                    discard;

                vec3 shaded = texel.rgb * uBaseColor;
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
        _uReceivesLighting = _gl.GetUniformLocation(_shaderProgram, "uReceivesLighting");
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
            uint ebo,
            uint indexCount,
            uint textureId,
            bool hasTexture,
            bool isTransparent,
            bool isAdditive,
            bool depthWrite,
            bool alphaCutout,
            bool receivesLighting,
            Vector3 baseColor,
            Vector3 emissiveColor,
            float alpha,
            uint blendMode)
        {
            Vao = vao;
            Vbo = vbo;
            Ebo = ebo;
            IndexCount = indexCount;
            TextureId = textureId;
            HasTexture = hasTexture;
            IsTransparent = isTransparent;
            IsAdditive = isAdditive;
            DepthWrite = depthWrite;
            AlphaCutout = alphaCutout;
            ReceivesLighting = receivesLighting;
            BaseColor = baseColor;
            EmissiveColor = emissiveColor;
            Alpha = alpha;
            BlendMode = blendMode;
        }

        public uint Vao { get; }

        public uint Vbo { get; }

        public uint Ebo { get; }

        public uint IndexCount { get; }

        public uint TextureId { get; }

        public bool HasTexture { get; }

        public bool IsTransparent { get; }

        public bool IsAdditive { get; }

        public bool DepthWrite { get; }

        public bool AlphaCutout { get; }

        public bool ReceivesLighting { get; }

        public Vector3 BaseColor { get; }

        public Vector3 EmissiveColor { get; }

        public float Alpha { get; }

        public uint BlendMode { get; }

        public void Dispose(GL gl)
        {
            gl.DeleteBuffer(Vbo);
            gl.DeleteBuffer(Ebo);
            gl.DeleteVertexArray(Vao);
        }
    }

    private readonly record struct MdxMaterialState(
        string? TexturePath,
        bool IsTransparent,
        bool IsAdditive,
        bool DepthWrite,
        bool AlphaCutout,
        float Alpha,
        uint BlendMode);
}
