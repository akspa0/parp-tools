using System.Numerics;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Silk.NET.OpenGL;
using Image = SixLabors.ImageSharp.Image;

namespace WowViewer.App;

internal sealed class ModelOutputGpuRenderer : IDisposable
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
    private int _uHasTexture;
    private int _uTexture0;

    private uint _framebuffer;
    private uint _colorTexture;
    private uint _depthRenderbuffer;
    private uint _fallbackWhiteTexture;
    private int _frameWidth;
    private int _frameHeight;
    private Vector3 _boundsMin = new(-1.0f, -1.0f, -1.0f);
    private Vector3 _boundsMax = new(1.0f, 1.0f, 1.0f);
    private readonly Vector3 _ambientColor = new(0.34f, 0.35f, 0.37f);
    private readonly Vector3 _lightColor = new(0.95f, 0.94f, 0.90f);
    private readonly Vector3 _lightDir = Vector3.Normalize(new Vector3(-0.45f, 0.85f, 0.25f));
    private bool _disposed;

    public ModelOutputGpuRenderer(GL gl)
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
        ClearScene();

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

    public void ClearScene()
    {
        foreach (CommandBuffers command in _commands)
            command.Dispose(_gl);

        _commands.Clear();
        _boundsMin = new Vector3(-1.0f, -1.0f, -1.0f);
        _boundsMax = new Vector3(1.0f, 1.0f, 1.0f);
    }

    public void LoadScene(ModelOutputScene scene)
    {
        ArgumentNullException.ThrowIfNull(scene);

        ClearScene();
        _boundsMin = scene.BoundsMin;
        _boundsMax = scene.BoundsMax;

        foreach (ModelOutputTileGeometry tile in scene.Tiles)
        {
            if (tile.Vertices.Length == 0 || tile.Indices.Length == 0)
                continue;

            float[] vertexData = new float[tile.Vertices.Length * 8];
            for (int index = 0; index < tile.Vertices.Length; index++)
            {
                ModelOutputVertex vertex = tile.Vertices[index];
                int offset = index * 8;
                vertexData[offset + 0] = vertex.Position.X;
                vertexData[offset + 1] = vertex.Position.Y;
                vertexData[offset + 2] = vertex.Position.Z;
                vertexData[offset + 3] = vertex.Normal.X;
                vertexData[offset + 4] = vertex.Normal.Y;
                vertexData[offset + 5] = vertex.Normal.Z;
                vertexData[offset + 6] = vertex.TexCoord.X;
                vertexData[offset + 7] = vertex.TexCoord.Y;
            }

            uint textureId = TryGetOrLoadTexture(tile.TexturePath, out uint loadedTexture)
                ? loadedTexture
                : _fallbackWhiteTexture;
            bool hasTexture = textureId != _fallbackWhiteTexture;

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
                fixed (uint* indexPtr = tile.Indices)
                {
                    _gl.BufferData(BufferTargetARB.ElementArrayBuffer, (nuint)(tile.Indices.Length * sizeof(uint)), indexPtr, BufferUsageARB.StaticDraw);
                }

                _gl.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 8u * sizeof(float), (void*)0);
                _gl.EnableVertexAttribArray(0);
                _gl.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, 8u * sizeof(float), (void*)(3 * sizeof(float)));
                _gl.EnableVertexAttribArray(1);
                _gl.VertexAttribPointer(2, 2, VertexAttribPointerType.Float, false, 8u * sizeof(float), (void*)(6 * sizeof(float)));
                _gl.EnableVertexAttribArray(2);
            }

            _gl.BindVertexArray(0);
            _commands.Add(new CommandBuffers(vao, vbo, ebo, (uint)tile.Indices.Length, textureId, hasTexture));
        }
    }

    public void Render(int width, int height, float azimuthDegrees, float elevationDegrees, float zoomFactor, Vector3 targetOffset)
    {
        EnsureFramebuffer(width, height);

        _gl.BindFramebuffer(GLEnum.Framebuffer, _framebuffer);
        _gl.Viewport(0, 0, (uint)_frameWidth, (uint)_frameHeight);
        _gl.ClearColor(0.07f, 0.09f, 0.11f, 1.0f);
        _gl.Clear((uint)(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit));

        if (_commands.Count > 0)
        {
            BuildOrbitCamera(_boundsMin, _boundsMax, _frameWidth, _frameHeight, azimuthDegrees, elevationDegrees, zoomFactor, targetOffset, out Matrix4x4 view, out Matrix4x4 projection);
            RenderPass(view, projection);
        }

        _gl.BindFramebuffer(GLEnum.Framebuffer, 0);
    }

    private void RenderPass(Matrix4x4 view, Matrix4x4 projection)
    {
        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Lequal);
        _gl.Disable(EnableCap.Blend);
        _gl.Disable(EnableCap.CullFace);
        _gl.DepthMask(true);
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
            _gl.Uniform1(_uHasTexture, command.HasTexture ? 1 : 0);
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
        _gl.UseProgram(0);
    }

    private bool TryGetOrLoadTexture(string texturePath, out uint textureId)
    {
        textureId = _fallbackWhiteTexture;
        if (string.IsNullOrWhiteSpace(texturePath) || !File.Exists(texturePath))
            return false;

        string fullPath = Path.GetFullPath(texturePath);
        if (_loadedTextureCache.TryGetValue(fullPath, out textureId))
            return true;

        using Image<Rgba32> image = Image.Load<Rgba32>(fullPath);
        Rgba32[] sourcePixels = new Rgba32[image.Width * image.Height];
        image.CopyPixelDataTo(sourcePixels);
        byte[] rgbaPixels = new byte[image.Width * image.Height * 4];
        for (int y = 0; y < image.Height; y++)
        {
            int targetRow = image.Height - 1 - y;
            int targetOffset = targetRow * image.Width * 4;
            for (int x = 0; x < image.Width; x++)
            {
                Rgba32 pixel = sourcePixels[(y * image.Width) + x];
                int pixelOffset = targetOffset + (x * 4);
                rgbaPixels[pixelOffset + 0] = pixel.R;
                rgbaPixels[pixelOffset + 1] = pixel.G;
                rgbaPixels[pixelOffset + 2] = pixel.B;
                rgbaPixels[pixelOffset + 3] = pixel.A;
            }
        }

        textureId = _gl.GenTexture();
        _gl.BindTexture(TextureTarget.Texture2D, textureId);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)GLEnum.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)GLEnum.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)GLEnum.ClampToEdge);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)GLEnum.ClampToEdge);

        unsafe
        {
            fixed (byte* pixelPtr = rgbaPixels)
            {
                _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba8, (uint)image.Width, (uint)image.Height, 0, PixelFormat.Rgba, PixelType.UnsignedByte, pixelPtr);
            }
        }

        _gl.BindTexture(TextureTarget.Texture2D, 0);
        _ownedTextureIds.Add(textureId);
        _loadedTextureCache[fullPath] = textureId;
        return true;
    }

    private void InitializeShader()
    {
        const string vertexSource = """
            #version 330 core
            layout (location = 0) in vec3 aPosition;
            layout (location = 1) in vec3 aNormal;
            layout (location = 2) in vec2 aTexCoord;

            uniform mat4 uView;
            uniform mat4 uProj;

            out vec3 vNormal;
            out vec2 vTexCoord;

            void main()
            {
                vNormal = aNormal;
                vTexCoord = aTexCoord;
                gl_Position = uProj * uView * vec4(aPosition, 1.0);
            }
            """;

        const string fragmentSource = """
            #version 330 core
            in vec3 vNormal;
            in vec2 vTexCoord;

            uniform vec3 uLightDir;
            uniform vec3 uLightColor;
            uniform vec3 uAmbientColor;
            uniform sampler2D uTexture0;
            uniform int uHasTexture;

            out vec4 FragColor;

            void main()
            {
                vec3 normal = normalize(vNormal);
                float ndotl = max(dot(normal, normalize(uLightDir)), 0.0);
                vec4 texel = uHasTexture == 1 ? texture(uTexture0, vTexCoord) : vec4(1.0, 1.0, 1.0, 1.0);
                vec3 lit = texel.rgb * (uAmbientColor + (uLightColor * ndotl));
                FragColor = vec4(lit, texel.a);
            }
            """;

        uint vertexShader = CompileShader(ShaderType.VertexShader, vertexSource);
        uint fragmentShader = CompileShader(ShaderType.FragmentShader, fragmentSource);

        _shaderProgram = _gl.CreateProgram();
        _gl.AttachShader(_shaderProgram, vertexShader);
        _gl.AttachShader(_shaderProgram, fragmentShader);
        _gl.LinkProgram(_shaderProgram);
        _gl.GetProgram(_shaderProgram, ProgramPropertyARB.LinkStatus, out int linkStatus);
        if (linkStatus == 0)
        {
            string log = _gl.GetProgramInfoLog(_shaderProgram);
            _gl.DeleteShader(vertexShader);
            _gl.DeleteShader(fragmentShader);
            throw new InvalidOperationException($"Failed to link model-output preview shader: {log}");
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
        _uHasTexture = _gl.GetUniformLocation(_shaderProgram, "uHasTexture");
        _uTexture0 = _gl.GetUniformLocation(_shaderProgram, "uTexture0");
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
            _gl.DeleteShader(shader);
            throw new InvalidOperationException($"Failed to compile model-output preview shader ({type}): {log}");
        }

        return shader;
    }

    private uint CreateFallbackWhiteTexture()
    {
        uint texture = _gl.GenTexture();
        _gl.BindTexture(TextureTarget.Texture2D, texture);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)GLEnum.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)GLEnum.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)GLEnum.ClampToEdge);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)GLEnum.ClampToEdge);

        byte[] rgba = [255, 255, 255, 255];
        unsafe
        {
            fixed (byte* pixelPtr = rgba)
            {
                _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba8, 1, 1, 0, PixelFormat.Rgba, PixelType.UnsignedByte, pixelPtr);
            }
        }

        _gl.BindTexture(TextureTarget.Texture2D, 0);
        return texture;
    }

    private unsafe void EnsureFramebuffer(int width, int height)
    {
        width = Math.Max(width, 1);
        height = Math.Max(height, 1);
        if (_framebuffer != 0 && width == _frameWidth && height == _frameHeight)
            return;

        DeleteFramebuffer();
        _frameWidth = width;
        _frameHeight = height;

        _framebuffer = _gl.GenFramebuffer();
        _gl.BindFramebuffer(FramebufferTarget.Framebuffer, _framebuffer);

        _colorTexture = _gl.GenTexture();
        _gl.BindTexture(TextureTarget.Texture2D, _colorTexture);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)GLEnum.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)GLEnum.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)GLEnum.ClampToEdge);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)GLEnum.ClampToEdge);
        _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba8, (uint)_frameWidth, (uint)_frameHeight, 0, PixelFormat.Rgba, PixelType.UnsignedByte, (void*)0);
        _gl.FramebufferTexture2D(FramebufferTarget.Framebuffer, FramebufferAttachment.ColorAttachment0, TextureTarget.Texture2D, _colorTexture, 0);

        _depthRenderbuffer = _gl.GenRenderbuffer();
        _gl.BindRenderbuffer(RenderbufferTarget.Renderbuffer, _depthRenderbuffer);
        _gl.RenderbufferStorage(RenderbufferTarget.Renderbuffer, InternalFormat.DepthComponent24, (uint)_frameWidth, (uint)_frameHeight);
        _gl.FramebufferRenderbuffer(FramebufferTarget.Framebuffer, FramebufferAttachment.DepthAttachment, RenderbufferTarget.Renderbuffer, _depthRenderbuffer);

        GLEnum status = _gl.CheckFramebufferStatus(FramebufferTarget.Framebuffer);
        if (status != GLEnum.FramebufferComplete)
        {
            DeleteFramebuffer();
            throw new InvalidOperationException($"Model-output preview framebuffer is incomplete: {status}");
        }

        _gl.BindFramebuffer(FramebufferTarget.Framebuffer, 0);
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
    }

    private static void BuildOrbitCamera(Vector3 boundsMin, Vector3 boundsMax, int width, int height, float azimuthDegrees, float elevationDegrees, float zoomFactor, Vector3 targetOffset, out Matrix4x4 view, out Matrix4x4 projection)
    {
        Vector3 center = ((boundsMin + boundsMax) * 0.5f) + targetOffset;
        Vector3 extents = boundsMax - boundsMin;
        float radius = MathF.Max(extents.Length() * 0.5f, 32.0f);
        float azimuth = MathF.PI / 180.0f * azimuthDegrees;
        float elevation = MathF.PI / 180.0f * Math.Clamp(elevationDegrees, -85.0f, 85.0f);
        float distance = MathF.Max(radius * (1.6f + zoomFactor), 96.0f);

        Vector3 cameraOffset = new(
            MathF.Cos(elevation) * MathF.Cos(azimuth) * distance,
            MathF.Sin(elevation) * distance,
            MathF.Cos(elevation) * MathF.Sin(azimuth) * distance);
        Vector3 cameraPosition = center + cameraOffset;

        view = Matrix4x4.CreateLookAt(cameraPosition, center, Vector3.UnitY);

        float aspect = Math.Max(width, 1) / (float)Math.Max(height, 1);
        float near = MathF.Max(radius * 0.01f, 0.5f);
        float far = MathF.Max(distance + (radius * 8.0f), 2048.0f);
        projection = Matrix4x4.CreatePerspectiveFieldOfView(MathF.PI / 4.0f, aspect, near, far);
    }

    private readonly record struct CommandBuffers(uint Vao, uint Vbo, uint Ebo, uint IndexCount, uint TextureId, bool HasTexture)
    {
        public void Dispose(GL gl)
        {
            gl.DeleteBuffer(Vbo);
            gl.DeleteBuffer(Ebo);
            gl.DeleteVertexArray(Vao);
        }
    }
}
