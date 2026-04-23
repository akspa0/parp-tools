using System.Numerics;
using Silk.NET.OpenGL;
using WowViewer.Core.Wmo;

namespace WowViewer.App;

internal sealed class WmoGpuPreviewRenderer : IDisposable
{
    private readonly GL _gl;
    private readonly List<GroupBuffers> _groups = [];
    private uint _shaderProgram;
    private int _uView;
    private int _uProjection;
    private int _uColor;
    private uint _framebuffer;
    private uint _colorTexture;
    private uint _depthRenderbuffer;
    private int _frameWidth;
    private int _frameHeight;
    private Vector3 _boundsMin = new(-1f, -1f, -1f);
    private Vector3 _boundsMax = new(1f, 1f, 1f);
    private PreviewCameraSettings _cameraSettings = new()
    {
        Mode = PreviewCameraMode.Orbit,
        PresetName = null,
        ZoomFactor = 0.9f,
    };

    public WmoGpuPreviewRenderer(GL gl)
    {
        _gl = gl ?? throw new ArgumentNullException(nameof(gl));
        InitializeShader();
    }

    public uint PreviewTextureHandle => _colorTexture;

    public bool HasRenderableGeometry => _groups.Count > 0;

    public int CommandCount => _groups.Count;

    public Vector3 BoundsMin => _boundsMin;

    public Vector3 BoundsMax => _boundsMax;

    public void Dispose()
    {
        ClearPreview();
        DeleteFramebuffer();
        if (_shaderProgram != 0)
            _gl.DeleteProgram(_shaderProgram);
    }

    public void LoadPreview(WmoPreviewLoadResult preview)
    {
        ArgumentNullException.ThrowIfNull(preview);

        ClearPreview();
        _cameraSettings = preview.Request.Camera;
        BuildBuffers(preview.Document);
    }

    public void SetCameraSettings(PreviewCameraSettings settings)
    {
        _cameraSettings = settings ?? throw new ArgumentNullException(nameof(settings));
    }

    public unsafe void Render(int width, int height)
    {
        if (!HasRenderableGeometry)
            return;

        EnsureFramebuffer(width, height);
        _gl.BindFramebuffer(FramebufferTarget.Framebuffer, _framebuffer);
        _gl.Viewport(0, 0, (uint)_frameWidth, (uint)_frameHeight);
        _gl.Enable(EnableCap.DepthTest);
        _gl.Disable(EnableCap.CullFace);
        _gl.ClearColor(0.07f, 0.08f, 0.10f, 1.0f);
        _gl.Clear((uint)(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit));

        PreviewCameraPose pose = PreviewCameraPlanner.CreatePose(_boundsMin, _boundsMax, _cameraSettings, null, null, 0, 0, _frameWidth, _frameHeight);
        _gl.UseProgram(_shaderProgram);
        Matrix4x4 view = pose.View;
        Matrix4x4 projection = pose.Projection;
        _gl.UniformMatrix4(_uView, 1, false, (float*)&view.M11);
        _gl.UniformMatrix4(_uProjection, 1, false, (float*)&projection.M11);

        foreach (GroupBuffers group in _groups)
        {
            _gl.Uniform3(_uColor, group.Color.X, group.Color.Y, group.Color.Z);
            _gl.BindVertexArray(group.Vao);
            _gl.DrawElements(PrimitiveType.Triangles, group.IndexCount, DrawElementsType.UnsignedShort, null);
        }

        _gl.BindVertexArray(0);
        _gl.BindFramebuffer(FramebufferTarget.Framebuffer, 0);
    }

    public void ClearPreview()
    {
        foreach (GroupBuffers group in _groups)
        {
            _gl.DeleteBuffer(group.Vbo);
            _gl.DeleteBuffer(group.Ebo);
            _gl.DeleteVertexArray(group.Vao);
        }

        _groups.Clear();
        _boundsMin = new(-1f, -1f, -1f);
        _boundsMax = new(1f, 1f, 1f);
    }

    private unsafe void BuildBuffers(WmoRenderDocument document)
    {
        bool hasBounds = false;
        Vector3 boundsMin = new(float.MaxValue, float.MaxValue, float.MaxValue);
        Vector3 boundsMax = new(float.MinValue, float.MinValue, float.MinValue);

        foreach (WmoEmbeddedGroupMeshDetail group in document.Groups)
        {
            if (group.Mesh.Vertices.Count == 0 || group.Mesh.Indices.Count < 3)
                continue;

            float[] interleaved = new float[group.Mesh.Vertices.Count * 6];
            for (int index = 0; index < group.Mesh.Vertices.Count; index++)
            {
                Vector3 vertex = group.Mesh.Vertices[index];
                Vector3 normal = index < group.Mesh.Normals.Count ? group.Mesh.Normals[index] : Vector3.UnitZ;
                int offset = index * 6;
                interleaved[offset + 0] = vertex.X;
                interleaved[offset + 1] = vertex.Y;
                interleaved[offset + 2] = vertex.Z;
                interleaved[offset + 3] = normal.X;
                interleaved[offset + 4] = normal.Y;
                interleaved[offset + 5] = normal.Z;
                boundsMin = Vector3.Min(boundsMin, vertex);
                boundsMax = Vector3.Max(boundsMax, vertex);
                hasBounds = true;
            }

            ushort[] indices = group.Mesh.Indices.ToArray();
            uint vao = _gl.GenVertexArray();
            uint vbo = _gl.GenBuffer();
            uint ebo = _gl.GenBuffer();
            _gl.BindVertexArray(vao);
            _gl.BindBuffer(BufferTargetARB.ArrayBuffer, vbo);
            fixed (float* verticesPtr = interleaved)
                _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(interleaved.Length * sizeof(float)), verticesPtr, BufferUsageARB.StaticDraw);
            _gl.BindBuffer(BufferTargetARB.ElementArrayBuffer, ebo);
            fixed (ushort* indicesPtr = indices)
                _gl.BufferData(BufferTargetARB.ElementArrayBuffer, (nuint)(indices.Length * sizeof(ushort)), indicesPtr, BufferUsageARB.StaticDraw);
            _gl.EnableVertexAttribArray(0);
            _gl.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, (uint)(6 * sizeof(float)), (void*)0);
            _gl.EnableVertexAttribArray(1);
            _gl.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, (uint)(6 * sizeof(float)), (void*)(3 * sizeof(float)));
            _gl.BindVertexArray(0);
            _groups.Add(new GroupBuffers(vao, vbo, ebo, (uint)indices.Length, ComputeGroupColor(group.GroupIndex)));
        }

        if (hasBounds)
        {
            _boundsMin = boundsMin;
            _boundsMax = boundsMax;
        }
    }

    private unsafe void EnsureFramebuffer(int width, int height)
    {
        width = Math.Max(width, 16);
        height = Math.Max(height, 16);
        if (_framebuffer != 0 && _frameWidth == width && _frameHeight == height)
            return;

        DeleteFramebuffer();
        _frameWidth = width;
        _frameHeight = height;
        _framebuffer = _gl.GenFramebuffer();
        _gl.BindFramebuffer(FramebufferTarget.Framebuffer, _framebuffer);

        _colorTexture = _gl.GenTexture();
        _gl.BindTexture(TextureTarget.Texture2D, _colorTexture);
        _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba8, (uint)_frameWidth, (uint)_frameHeight, 0, PixelFormat.Rgba, PixelType.UnsignedByte, null);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)GLEnum.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)GLEnum.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)GLEnum.ClampToEdge);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)GLEnum.ClampToEdge);
        _gl.FramebufferTexture2D(FramebufferTarget.Framebuffer, FramebufferAttachment.ColorAttachment0, TextureTarget.Texture2D, _colorTexture, 0);

        _depthRenderbuffer = _gl.GenRenderbuffer();
        _gl.BindRenderbuffer(RenderbufferTarget.Renderbuffer, _depthRenderbuffer);
        _gl.RenderbufferStorage(RenderbufferTarget.Renderbuffer, InternalFormat.DepthComponent24, (uint)_frameWidth, (uint)_frameHeight);
        _gl.FramebufferRenderbuffer(FramebufferTarget.Framebuffer, FramebufferAttachment.DepthAttachment, RenderbufferTarget.Renderbuffer, _depthRenderbuffer);
        _gl.BindFramebuffer(FramebufferTarget.Framebuffer, 0);
    }

    private void DeleteFramebuffer()
    {
        if (_depthRenderbuffer != 0)
            _gl.DeleteRenderbuffer(_depthRenderbuffer);
        if (_colorTexture != 0)
            _gl.DeleteTexture(_colorTexture);
        if (_framebuffer != 0)
            _gl.DeleteFramebuffer(_framebuffer);

        _depthRenderbuffer = 0;
        _colorTexture = 0;
        _framebuffer = 0;
        _frameWidth = 0;
        _frameHeight = 0;
    }

    private void InitializeShader()
    {
        const string vertexShaderSource = """
            #version 330 core
            layout(location = 0) in vec3 aPosition;
            layout(location = 1) in vec3 aNormal;
            uniform mat4 uView;
            uniform mat4 uProjection;
            out vec3 vNormal;
            void main()
            {
                vNormal = aNormal;
                gl_Position = uProjection * uView * vec4(aPosition, 1.0);
            }
            """;
        const string fragmentShaderSource = """
            #version 330 core
            in vec3 vNormal;
            uniform vec3 uColor;
            out vec4 fragColor;
            void main()
            {
                vec3 lightDir = normalize(vec3(0.35, 0.45, 1.0));
                float light = max(dot(normalize(vNormal), lightDir), 0.18);
                fragColor = vec4(uColor * light, 1.0);
            }
            """;

        uint vertexShader = CompileShader(ShaderType.VertexShader, vertexShaderSource);
        uint fragmentShader = CompileShader(ShaderType.FragmentShader, fragmentShaderSource);
        _shaderProgram = _gl.CreateProgram();
        _gl.AttachShader(_shaderProgram, vertexShader);
        _gl.AttachShader(_shaderProgram, fragmentShader);
        _gl.LinkProgram(_shaderProgram);
        _gl.GetProgram(_shaderProgram, ProgramPropertyARB.LinkStatus, out int linked);
        if (linked == 0)
        {
            string info = _gl.GetProgramInfoLog(_shaderProgram);
            throw new InvalidOperationException($"WMO preview shader link failed: {info}");
        }

        _gl.DeleteShader(vertexShader);
        _gl.DeleteShader(fragmentShader);
        _uView = _gl.GetUniformLocation(_shaderProgram, "uView");
        _uProjection = _gl.GetUniformLocation(_shaderProgram, "uProjection");
        _uColor = _gl.GetUniformLocation(_shaderProgram, "uColor");
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
            throw new InvalidOperationException($"WMO preview shader compile failed: {info}");
        }

        return shader;
    }

    private static Vector3 ComputeGroupColor(int groupIndex)
    {
        float red = ((groupIndex * 67 + 13) % 255) / 255f;
        float green = ((groupIndex * 131 + 7) % 255) / 255f;
        float blue = ((groupIndex * 43 + 29) % 255) / 255f;
        return new Vector3(red, green, blue);
    }

    private sealed record GroupBuffers(uint Vao, uint Vbo, uint Ebo, uint IndexCount, Vector3 Color);
}