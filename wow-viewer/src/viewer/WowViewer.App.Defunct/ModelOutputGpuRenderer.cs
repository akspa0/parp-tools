using System.Numerics;
using System.Linq;
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
    private int _uTintColor;

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

    public static ModelOutputCameraFrame BuildOrbitCameraFrame(
        Vector3 boundsMin,
        Vector3 boundsMax,
        int width,
        int height,
        float azimuthDegrees,
        float elevationDegrees,
        float zoomFactor,
        Vector3 targetOffset)
    {
        BuildOrbitCamera(boundsMin, boundsMax, width, height, azimuthDegrees, elevationDegrees, zoomFactor, targetOffset, out Matrix4x4 view, out Matrix4x4 projection, out Vector3 cameraPosition);
        return new ModelOutputCameraFrame(view, projection, cameraPosition);
    }

    public static ModelOutputCameraFrame BuildFlyCameraFrame(
        Vector3 boundsMin,
        Vector3 boundsMax,
        int width,
        int height,
        Vector3 position,
        float azimuthDegrees,
        float elevationDegrees)
    {
        BuildFlyCamera(boundsMin, boundsMax, width, height, position, azimuthDegrees, elevationDegrees, out Matrix4x4 view, out Matrix4x4 projection);
        return new ModelOutputCameraFrame(view, projection, position);
    }

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

    public void LoadScene(ModelOutputScene scene, bool showObjects, bool showM2Objects, bool showWmoObjects)
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
            Vector3 boundsCenter = (scene.BoundsMin + scene.BoundsMax) * 0.5f;
            _commands.Add(CreateCommand(vertexData, tile.Indices, textureId, hasTexture, new Vector4(1.0f), transparent: false, boundsCenter));
        }

        if (showObjects)
            LoadObjectPlaceholders(scene.Objects, showM2Objects, showWmoObjects);
    }

    public void Render(int width, int height, ModelOutputCameraFrame cameraFrame)
    {
        EnsureFramebuffer(width, height);

        _gl.BindFramebuffer(GLEnum.Framebuffer, _framebuffer);
        _gl.Viewport(0, 0, (uint)_frameWidth, (uint)_frameHeight);
        _gl.ClearColor(0.07f, 0.09f, 0.11f, 1.0f);
        _gl.Clear((uint)(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit));

        if (_commands.Count > 0)
            RenderPass(cameraFrame.View, cameraFrame.Projection);

        _gl.BindFramebuffer(GLEnum.Framebuffer, 0);
    }

    private void RenderPass(Matrix4x4 view, Matrix4x4 projection)
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

        _gl.Disable(EnableCap.Blend);
        _gl.DepthMask(true);
        foreach (CommandBuffers command in _commands.Where(static command => !command.Transparent))
            DrawCommand(command);

        _gl.Enable(EnableCap.Blend);
        _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
        _gl.DepthMask(false);
        foreach (CommandBuffers command in _commands.Where(static command => command.Transparent))
            DrawCommand(command);

        _gl.BindVertexArray(0);
        _gl.BindTexture(TextureTarget.Texture2D, 0);
        _gl.DepthMask(true);
        _gl.Disable(EnableCap.Blend);
        _gl.UseProgram(0);
    }

    private void DrawCommand(CommandBuffers command)
    {
        _gl.Uniform1(_uHasTexture, command.HasTexture ? 1 : 0);
        _gl.Uniform4(_uTintColor, command.TintColor.X, command.TintColor.Y, command.TintColor.Z, command.TintColor.W);
        _gl.ActiveTexture(TextureUnit.Texture0);
        _gl.BindTexture(TextureTarget.Texture2D, command.TextureId);
        _gl.Uniform1(_uTexture0, 0);
        _gl.BindVertexArray(command.Vao);
        unsafe
        {
            _gl.DrawElements(PrimitiveType.Triangles, command.IndexCount, DrawElementsType.UnsignedInt, (void*)0);
        }
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
            uniform vec4 uTintColor;

            out vec4 FragColor;

            void main()
            {
                vec3 normal = normalize(vNormal);
                float ndotl = max(dot(normal, normalize(uLightDir)), 0.0);
                vec4 texel = uHasTexture == 1 ? texture(uTexture0, vTexCoord) : vec4(1.0, 1.0, 1.0, 1.0);
                vec3 lit = texel.rgb * uTintColor.rgb * (uAmbientColor + (uLightColor * ndotl));
                FragColor = vec4(lit, texel.a * uTintColor.a);
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
        _uTintColor = _gl.GetUniformLocation(_shaderProgram, "uTintColor");
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

    private void LoadObjectPlaceholders(IReadOnlyList<ModelOutputObjectPlacement> objects, bool showM2Objects, bool showWmoObjects)
    {
        List<ModelOutputObjectPlacement> m2Objects = [];
        List<ModelOutputObjectPlacement> wmoObjects = [];
        List<ModelOutputObjectPlacement> otherObjects = [];

        foreach (ModelOutputObjectPlacement obj in objects)
        {
            switch (obj.Category)
            {
                case "m2" when showM2Objects:
                    m2Objects.Add(obj);
                    break;
                case "wmo" when showWmoObjects:
                    wmoObjects.Add(obj);
                    break;
                case not ("m2" or "wmo"):
                    otherObjects.Add(obj);
                    break;
            }
        }

        AddObjectPlaceholderCommand(m2Objects, new Vector4(0.42f, 0.92f, 0.48f, 0.32f));
        AddObjectPlaceholderCommand(wmoObjects, new Vector4(0.95f, 0.68f, 0.30f, 0.34f));
        AddObjectPlaceholderCommand(otherObjects, new Vector4(0.48f, 0.76f, 0.98f, 0.30f));
    }

    private void AddObjectPlaceholderCommand(IReadOnlyList<ModelOutputObjectPlacement> objects, Vector4 tintColor)
    {
        if (objects.Count == 0)
            return;

        List<float> vertexData = new(objects.Count * 24 * 8);
        List<uint> indexData = new(objects.Count * 36);
        Vector3 boundsMin = new(float.MaxValue, float.MaxValue, float.MaxValue);
        Vector3 boundsMax = new(float.MinValue, float.MinValue, float.MinValue);
        foreach (ModelOutputObjectPlacement obj in objects)
        {
            AppendBoxGeometry(vertexData, indexData, obj.BoundsMin, obj.BoundsMax);
            boundsMin = Vector3.Min(boundsMin, obj.BoundsMin);
            boundsMax = Vector3.Max(boundsMax, obj.BoundsMax);
        }

        _commands.Add(CreateCommand([.. vertexData], [.. indexData], _fallbackWhiteTexture, hasTexture: false, tintColor, transparent: true, (boundsMin + boundsMax) * 0.5f));
    }

    private CommandBuffers CreateCommand(float[] vertexData, uint[] indexData, uint textureId, bool hasTexture, Vector4 tintColor, bool transparent, Vector3 boundsCenter)
    {
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
            fixed (uint* indexPtr = indexData)
            {
                _gl.BufferData(BufferTargetARB.ElementArrayBuffer, (nuint)(indexData.Length * sizeof(uint)), indexPtr, BufferUsageARB.StaticDraw);
            }

            _gl.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 8u * sizeof(float), (void*)0);
            _gl.EnableVertexAttribArray(0);
            _gl.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, 8u * sizeof(float), (void*)(3 * sizeof(float)));
            _gl.EnableVertexAttribArray(1);
            _gl.VertexAttribPointer(2, 2, VertexAttribPointerType.Float, false, 8u * sizeof(float), (void*)(6 * sizeof(float)));
            _gl.EnableVertexAttribArray(2);
        }

        _gl.BindVertexArray(0);
        return new CommandBuffers(vao, vbo, ebo, (uint)indexData.Length, textureId, hasTexture, tintColor, transparent, boundsCenter);
    }

    private static void AppendBoxGeometry(List<float> vertexData, List<uint> indexData, Vector3 boundsMin, Vector3 boundsMax)
    {
        if (boundsMax.X <= boundsMin.X)
            boundsMax.X = boundsMin.X + 8.0f;
        if (boundsMax.Y <= boundsMin.Y)
            boundsMax.Y = boundsMin.Y + 16.0f;
        if (boundsMax.Z <= boundsMin.Z)
            boundsMax.Z = boundsMin.Z + 8.0f;

        Vector3 p000 = new(boundsMin.X, boundsMin.Y, boundsMin.Z);
        Vector3 p001 = new(boundsMin.X, boundsMin.Y, boundsMax.Z);
        Vector3 p010 = new(boundsMin.X, boundsMax.Y, boundsMin.Z);
        Vector3 p011 = new(boundsMin.X, boundsMax.Y, boundsMax.Z);
        Vector3 p100 = new(boundsMax.X, boundsMin.Y, boundsMin.Z);
        Vector3 p101 = new(boundsMax.X, boundsMin.Y, boundsMax.Z);
        Vector3 p110 = new(boundsMax.X, boundsMax.Y, boundsMin.Z);
        Vector3 p111 = new(boundsMax.X, boundsMax.Y, boundsMax.Z);

        AppendFace(vertexData, indexData, p101, p001, p011, p111, Vector3.UnitZ);
        AppendFace(vertexData, indexData, p100, p110, p010, p000, -Vector3.UnitZ);
        AppendFace(vertexData, indexData, p000, p010, p011, p001, -Vector3.UnitX);
        AppendFace(vertexData, indexData, p100, p101, p111, p110, Vector3.UnitX);
        AppendFace(vertexData, indexData, p010, p110, p111, p011, Vector3.UnitY);
        AppendFace(vertexData, indexData, p000, p001, p101, p100, -Vector3.UnitY);
    }

    private static void AppendFace(List<float> vertexData, List<uint> indexData, Vector3 a, Vector3 b, Vector3 c, Vector3 d, Vector3 normal)
    {
        uint baseIndex = (uint)(vertexData.Count / 8);
        AppendVertex(vertexData, a, normal);
        AppendVertex(vertexData, b, normal);
        AppendVertex(vertexData, c, normal);
        AppendVertex(vertexData, d, normal);

        indexData.Add(baseIndex + 0);
        indexData.Add(baseIndex + 1);
        indexData.Add(baseIndex + 2);
        indexData.Add(baseIndex + 0);
        indexData.Add(baseIndex + 2);
        indexData.Add(baseIndex + 3);
    }

    private static void AppendVertex(List<float> vertexData, Vector3 position, Vector3 normal)
    {
        vertexData.Add(position.X);
        vertexData.Add(position.Y);
        vertexData.Add(position.Z);
        vertexData.Add(normal.X);
        vertexData.Add(normal.Y);
        vertexData.Add(normal.Z);
        vertexData.Add(0.0f);
        vertexData.Add(0.0f);
    }

    private static void BuildOrbitCamera(Vector3 boundsMin, Vector3 boundsMax, int width, int height, float azimuthDegrees, float elevationDegrees, float zoomFactor, Vector3 targetOffset, out Matrix4x4 view, out Matrix4x4 projection, out Vector3 cameraPosition)
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
        cameraPosition = center + cameraOffset;

        view = Matrix4x4.CreateLookAt(cameraPosition, center, Vector3.UnitY);

        float aspect = Math.Max(width, 1) / (float)Math.Max(height, 1);
        float near = MathF.Max(radius * 0.01f, 0.5f);
        float far = MathF.Max(distance + (radius * 8.0f), 2048.0f);
        projection = Matrix4x4.CreatePerspectiveFieldOfView(MathF.PI / 4.0f, aspect, near, far);
    }

    private static void BuildFlyCamera(Vector3 boundsMin, Vector3 boundsMax, int width, int height, Vector3 position, float azimuthDegrees, float elevationDegrees, out Matrix4x4 view, out Matrix4x4 projection)
    {
        float azimuth = MathF.PI / 180.0f * azimuthDegrees;
        float elevation = MathF.PI / 180.0f * Math.Clamp(elevationDegrees, -85.0f, 85.0f);
        Vector3 forward = Vector3.Normalize(new Vector3(
            MathF.Cos(elevation) * MathF.Cos(azimuth),
            MathF.Sin(elevation),
            MathF.Cos(elevation) * MathF.Sin(azimuth)));
        view = Matrix4x4.CreateLookAt(position, position + forward, Vector3.UnitY);

        Vector3 extents = boundsMax - boundsMin;
        float radius = MathF.Max(extents.Length() * 0.5f, 64.0f);
        float aspect = Math.Max(width, 1) / (float)Math.Max(height, 1);
        float near = 0.5f;
        float far = MathF.Max(radius * 12.0f, 4096.0f);
        projection = Matrix4x4.CreatePerspectiveFieldOfView(MathF.PI / 4.0f, aspect, near, far);
    }

    private readonly record struct CommandBuffers(uint Vao, uint Vbo, uint Ebo, uint IndexCount, uint TextureId, bool HasTexture, Vector4 TintColor, bool Transparent, Vector3 BoundsCenter)
    {
        public void Dispose(GL gl)
        {
            gl.DeleteBuffer(Vbo);
            gl.DeleteBuffer(Ebo);
            gl.DeleteVertexArray(Vao);
        }
    }
}

internal readonly record struct ModelOutputCameraFrame(Matrix4x4 View, Matrix4x4 Projection, Vector3 Position);
