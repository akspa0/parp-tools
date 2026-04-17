using System.Numerics;
using SereniaBLPLib;
using Silk.NET.OpenGL;
using WowViewer.Core.IO.Files;
using WowViewer.Core.M2;
using WowViewer.Core.Runtime.M2;

namespace WowViewer.App;

internal sealed class M2GpuPreviewRenderer : IDisposable
{
    private readonly GL _gl;
    private readonly Dictionary<string, uint> _loadedTextureCache = new(StringComparer.OrdinalIgnoreCase);
    private readonly HashSet<uint> _ownedTextureIds = [];
    private readonly List<CommandBuffers> _commands = [];

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
    private int _uHasUvTransform;
    private int _uUvTranslation;
    private int _uUvScale;
    private int _uUvRotation;

    private uint _framebuffer;
    private uint _colorTexture;
    private uint _depthRenderbuffer;
    private uint _fallbackWhiteTexture;
    private int _frameWidth;
    private int _frameHeight;
    private Vector3 _boundsMin = new(-1.0f, -1.0f, -1.0f);
    private Vector3 _boundsMax = new(1.0f, 1.0f, 1.0f);
    private Vector3 _ambientColor = new(0.25f, 0.25f, 0.3f);
    private Vector3 _lightColor = new(0.9f, 0.9f, 0.85f);
    private readonly Vector3 _lightDir = Vector3.Normalize(new Vector3(-0.5f, 0.8f, 0.35f));
    private bool _disposed;

    public M2GpuPreviewRenderer(GL gl)
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
    }

    public void LoadPreview(M2PreviewLoadResult preview)
    {
        ArgumentNullException.ThrowIfNull(preview);

        ClearPreview();

        _boundsMin = preview.FrameResult.ConsumerState.RenderModel.BoundsMin;
        _boundsMax = preview.FrameResult.ConsumerState.RenderModel.BoundsMax;
        _ambientColor = Clamp01(preview.FrameResult.ConsumerState.ModelAmbient);
        _lightColor = Clamp01(preview.FrameResult.ConsumerState.ModelDiffuse);
        if (_ambientColor.LengthSquared() <= 0.0001f)
            _ambientColor = new Vector3(0.25f, 0.25f, 0.3f);
        if (_lightColor.LengthSquared() <= 0.0001f)
            _lightColor = new Vector3(0.9f, 0.9f, 0.85f);

        foreach (M2RenderDrawCommand command in preview.FrameResult.RenderFrame.DrawCommands)
        {
            if (command.Vertices.Count == 0 || command.Indices.Count == 0)
                continue;

            M2RenderConsumerTextureState? textureState = command.Textures
                .OrderBy(static texture => texture.StageIndex)
                .FirstOrDefault();

            uint textureId = 0;
            bool hasTexture = false;
            if (textureState != null && !string.IsNullOrWhiteSpace(textureState.TexturePath))
            {
                hasTexture = TryGetOrLoadTexture(preview.Request, textureState.TexturePath!, out textureId);
            }

            Matrix4x4 rotationMatrix = textureState != null
                ? Matrix4x4.CreateFromQuaternion(textureState.Rotation)
                : Matrix4x4.Identity;

            float[] vertexData = new float[command.Vertices.Count * 8];
            for (int index = 0; index < command.Vertices.Count; index++)
            {
                M2RenderBackendVertex vertex = command.Vertices[index];
                int offset = index * 8;
                vertexData[offset + 0] = vertex.Position.X;
                vertexData[offset + 1] = vertex.Position.Y;
                vertexData[offset + 2] = vertex.Position.Z;
                vertexData[offset + 3] = vertex.Normal.X;
                vertexData[offset + 4] = vertex.Normal.Y;
                vertexData[offset + 5] = vertex.Normal.Z;
                vertexData[offset + 6] = vertex.TextureCoords.X;
                vertexData[offset + 7] = vertex.TextureCoords.Y;
            }

            uint[] indices = command.Indices.ToArray();
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
                hasTexture ? textureId : _fallbackWhiteTexture,
                hasTexture,
                command.IsTransparent,
                command.IsAdditive,
                command.DepthWrite,
                command.AlphaTest,
                command.IsTwoSided,
                command.ReceivesLighting,
                Clamp01(command.DiffuseColor),
                Clamp01(command.EmissiveColor),
                Math.Clamp(command.Alpha * (textureState?.Alpha ?? 1.0f), 0.0f, 1.0f),
                textureState != null && HasMeaningfulUvTransform(textureState),
                new Vector2(textureState?.Translation.X ?? 0.0f, textureState?.Translation.Y ?? 0.0f),
                new Vector2(
                    Math.Abs(textureState?.Scaling.X ?? 1.0f) <= 0.0001f ? 1.0f : textureState!.Scaling.X,
                    Math.Abs(textureState?.Scaling.Y ?? 1.0f) <= 0.0001f ? 1.0f : textureState!.Scaling.Y),
                new Vector2(rotationMatrix.M11, rotationMatrix.M21),
                command.BlendMode));
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
            BuildCamera(_boundsMin, _boundsMax, _frameWidth, _frameHeight, out Matrix4x4 view, out Matrix4x4 projection);
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

        string resolvedOutputPath = Path.GetFullPath(outputPath);
        string? directory = Path.GetDirectoryName(resolvedOutputPath);
        if (!string.IsNullOrWhiteSpace(directory))
            Directory.CreateDirectory(directory);

        using FileStream stream = File.Create(resolvedOutputPath);
        BitmapWriter.WriteRgbaBitmap(stream, _frameWidth, _frameHeight, rgbaPixels);
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
            _gl.Uniform1(_uHasUvTransform, command.HasUvTransform ? 1 : 0);
            _gl.Uniform2(_uUvTranslation, command.UvTranslation.X, command.UvTranslation.Y);
            _gl.Uniform2(_uUvScale, command.UvScale.X, command.UvScale.Y);
            _gl.Uniform2(_uUvRotation, command.UvRotation.X, command.UvRotation.Y);

            _gl.ActiveTexture(TextureUnit.Texture0);
            _gl.BindTexture(TextureTarget.Texture2D, command.TextureId);
            _gl.Uniform1(_uTexture0, 0);
            _gl.BindVertexArray(command.Vao);
            unsafe
            {
                _gl.DrawElements(PrimitiveType.Triangles, command.IndexCount, DrawElementsType.UnsignedInt, (void*)0);
            }
        }

        _gl.BindVertexArray(0);
        _gl.BindTexture(TextureTarget.Texture2D, 0);
        _gl.Disable(EnableCap.Blend);
        _gl.DepthMask(true);
        _gl.UseProgram(0);
    }

    private void ConfigureBlendMode(bool isAdditive, M2BlendMode blendMode)
    {
        if (isAdditive || blendMode is M2BlendMode.Add or M2BlendMode.NoAlphaAdd or M2BlendMode.BlendAdd)
        {
            _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.One);
            return;
        }

        if (blendMode is M2BlendMode.Mod or M2BlendMode.Mod2X)
        {
            _gl.BlendFunc(BlendingFactor.DstColor, BlendingFactor.Zero);
            return;
        }

        _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
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
            throw new InvalidOperationException($"GPU preview framebuffer is incomplete: {status}.");
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
            uniform bool uHasUvTransform;
            uniform vec2 uUvTranslation;
            uniform vec2 uUvScale;
            uniform vec2 uUvRotation;

            out vec3 vNormal;
            out vec2 vTexCoord;

            void main()
            {
                gl_Position = uProj * uView * vec4(aPos, 1.0);
                vNormal = aNormal;

                vec2 uv = aTexCoord;
                if (uHasUvTransform)
                {
                    vec2 centered = uv - vec2(0.5, 0.5);
                    vec2 rotated = vec2(
                        centered.x * uUvRotation.x - centered.y * uUvRotation.y,
                        centered.x * uUvRotation.y + centered.y * uUvRotation.x);
                    uv = rotated * uUvScale + vec2(0.5, 0.5) + uUvTranslation;
                }

                vTexCoord = uv;
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
                    float diffuse = max(dot(normal, normalize(-uLightDir)), 0.0);
                    shaded *= clamp(uAmbientColor + (uLightColor * diffuse), vec3(0.0), vec3(1.5));
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
            throw new InvalidOperationException($"Failed to link GPU preview shader program: {info}");
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
        _uHasUvTransform = _gl.GetUniformLocation(_shaderProgram, "uHasUvTransform");
        _uUvTranslation = _gl.GetUniformLocation(_shaderProgram, "uUvTranslation");
        _uUvScale = _gl.GetUniformLocation(_shaderProgram, "uUvScale");
        _uUvRotation = _gl.GetUniformLocation(_shaderProgram, "uUvRotation");
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
            throw new InvalidOperationException($"Failed to compile GPU preview {shaderType}: {info}");
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

    private bool TryGetOrLoadTexture(M2PreviewLoadRequest request, string texturePath, out uint textureId)
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

    private bool TryReadTextureBytes(M2PreviewLoadRequest request, string texturePath, out byte[]? bytes)
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

    private static void BuildCamera(Vector3 min, Vector3 max, int width, int height, out Matrix4x4 view, out Matrix4x4 projection)
    {
        Vector3 center = (min + max) * 0.5f;
        Vector3 extent = max - min;
        float radius = MathF.Max(extent.Length() * 0.5f, 1.0f);
        float aspect = Math.Max(width, 1) / (float)Math.Max(height, 1);
        float fov = 45.0f * MathF.PI / 180.0f;

        float elev = 20.0f * MathF.PI / 180.0f;
        float azim = 45.0f * MathF.PI / 180.0f;
        float cosElev = MathF.Cos(elev);
        Vector3 camDir = Vector3.Normalize(new Vector3(
            cosElev * MathF.Cos(azim),
            cosElev * MathF.Sin(azim),
            -MathF.Sin(elev)));

        Vector3[] corners =
        [
            new(min.X, min.Y, min.Z),
            new(max.X, min.Y, min.Z),
            new(min.X, max.Y, min.Z),
            new(max.X, max.Y, min.Z),
            new(min.X, min.Y, max.Z),
            new(max.X, min.Y, max.Z),
            new(min.X, max.Y, max.Z),
            new(max.X, max.Y, max.Z),
        ];

        Vector3 tmpCam = center - camDir * radius;
        Vector3 up = MathF.Abs(Vector3.Dot(camDir, Vector3.UnitZ)) > 0.99f ? Vector3.UnitX : Vector3.UnitZ;
        Matrix4x4 tmpView = Matrix4x4.CreateLookAt(tmpCam, center, up);

        float halfFovV = fov * 0.5f;
        float halfFovH = MathF.Atan(MathF.Tan(halfFovV) * aspect);
        float maxDist = radius;
        foreach (Vector3 corner in corners)
        {
            Vector3 viewPos = Vector3.Transform(corner, tmpView);
            float depth = -viewPos.Z;
            float needV = MathF.Abs(viewPos.Y) / MathF.Tan(halfFovV) + depth;
            float needH = MathF.Abs(viewPos.X) / MathF.Tan(halfFovH) + depth;
            maxDist = MathF.Max(maxDist, MathF.Max(needV, needH));
        }

        float dist = maxDist * 1.15f;
        Vector3 camPos = center - camDir * dist;
        view = Matrix4x4.CreateLookAt(camPos, center, up);
        projection = Matrix4x4.CreatePerspectiveFieldOfView(fov, aspect, 0.01f, dist * 10.0f);
    }

    private static Vector3 Clamp01(Vector3 value)
    {
        return new Vector3(
            Math.Clamp(value.X, 0.0f, 1.0f),
            Math.Clamp(value.Y, 0.0f, 1.0f),
            Math.Clamp(value.Z, 0.0f, 1.0f));
    }

    private static bool HasMeaningfulUvTransform(M2RenderConsumerTextureState textureState)
    {
        Matrix4x4 rotationMatrix = Matrix4x4.CreateFromQuaternion(textureState.Rotation);
        Vector2 scale = new(
            Math.Abs(textureState.Scaling.X) <= 0.0001f ? 1.0f : textureState.Scaling.X,
            Math.Abs(textureState.Scaling.Y) <= 0.0001f ? 1.0f : textureState.Scaling.Y);
        Vector2 translation = new(textureState.Translation.X, textureState.Translation.Y);
        Vector2 rotation = new(rotationMatrix.M11, rotationMatrix.M21);
        return translation.LengthSquared() > 0.000001f
            || Vector2.DistanceSquared(scale, Vector2.One) > 0.000001f
            || Vector2.DistanceSquared(rotation, new Vector2(1.0f, 0.0f)) > 0.000001f;
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
            bool isTwoSided,
            bool receivesLighting,
            Vector3 baseColor,
            Vector3 emissiveColor,
            float alpha,
            bool hasUvTransform,
            Vector2 uvTranslation,
            Vector2 uvScale,
            Vector2 uvRotation,
            M2BlendMode blendMode = M2BlendMode.Opaque)
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
            IsTwoSided = isTwoSided;
            ReceivesLighting = receivesLighting;
            BaseColor = baseColor;
            EmissiveColor = emissiveColor;
            Alpha = alpha;
            HasUvTransform = hasUvTransform;
            UvTranslation = uvTranslation;
            UvScale = uvScale;
            UvRotation = uvRotation;
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

        public bool IsTwoSided { get; }

        public bool ReceivesLighting { get; }

        public Vector3 BaseColor { get; }

        public Vector3 EmissiveColor { get; }

        public float Alpha { get; }

        public bool HasUvTransform { get; }

        public Vector2 UvTranslation { get; }

        public Vector2 UvScale { get; }

        public Vector2 UvRotation { get; }

        public M2BlendMode BlendMode { get; }

        public void Dispose(GL gl)
        {
            gl.DeleteBuffer(Vbo);
            gl.DeleteBuffer(Ebo);
            gl.DeleteVertexArray(Vao);
        }
    }

    private static class BitmapWriter
    {
        public static void WriteRgbaBitmap(Stream stream, int width, int height, byte[] rgbaPixels)
        {
            ArgumentNullException.ThrowIfNull(stream);
            ArgumentNullException.ThrowIfNull(rgbaPixels);

            int rowStride = width * 4;
            int pixelDataLength = rowStride * height;
            int fileSize = 14 + 40 + pixelDataLength;

            using BinaryWriter writer = new(stream, System.Text.Encoding.ASCII, leaveOpen: true);
            writer.Write((byte)'B');
            writer.Write((byte)'M');
            writer.Write(fileSize);
            writer.Write(0);
            writer.Write(14 + 40);

            writer.Write(40);
            writer.Write(width);
            writer.Write(height);
            writer.Write((short)1);
            writer.Write((short)32);
            writer.Write(0);
            writer.Write(pixelDataLength);
            writer.Write(2835);
            writer.Write(2835);
            writer.Write(0);
            writer.Write(0);

            byte[] bgraRow = new byte[rowStride];
            for (int row = 0; row < height; row++)
            {
                int sourceOffset = row * rowStride;
                for (int column = 0; column < width; column++)
                {
                    int sourcePixel = sourceOffset + (column * 4);
                    int targetPixel = column * 4;
                    bgraRow[targetPixel + 0] = rgbaPixels[sourcePixel + 2];
                    bgraRow[targetPixel + 1] = rgbaPixels[sourcePixel + 1];
                    bgraRow[targetPixel + 2] = rgbaPixels[sourcePixel + 0];
                    bgraRow[targetPixel + 3] = 255;
                }

                writer.Write(bgraRow);
            }
        }
    }
}