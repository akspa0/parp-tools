using System.Numerics;
using ImGuiNET;
using Silk.NET.OpenGL;

namespace WoWViewer.Rendering;

/// <summary>
/// Draws one decoded image (BLP/TGA/PNG/...) fit-to-viewport in the main scene viewport with a transparency
/// checkerboard. Mouse wheel zooms, drag pans, double-click resets. Used by the main File Browser for texture files.
/// </summary>
public sealed class TextureViewRenderer : ISceneRenderer
{
    private readonly GL _gl;
    private readonly int _width, _height;
    private uint _program, _vao, _vbo, _texture;
    private int _uHalf, _uCenter, _uTex;
    private bool _nearest;
    private float _zoom = 1f;
    private Vector2 _pan;
    private bool _disposed;

    public int Width => _width;
    public int Height => _height;

    public unsafe TextureViewRenderer(GL gl, byte[] rgba, int width, int height)
    {
        ArgumentNullException.ThrowIfNull(rgba);
        if (width <= 0 || height <= 0 || rgba.Length < width * height * 4)
            throw new ArgumentException("RGBA buffer does not match the image dimensions.");
        _gl = gl;
        _width = width;
        _height = height;

        const string vs = @"#version 330 core
layout(location = 0) in vec2 aPos;
uniform vec2 uHalf;
uniform vec2 uCenter;
out vec2 vUv;
void main()
{
    vUv = aPos * 0.5 + 0.5;
    vUv.y = 1.0 - vUv.y;
    gl_Position = vec4(uCenter + aPos * uHalf, 0.0, 1.0);
}";
        const string fs = @"#version 330 core
in vec2 vUv;
uniform sampler2D uTex;
out vec4 FragColor;
void main()
{
    vec4 c = texture(uTex, vUv);
    vec2 cell = floor(gl_FragCoord.xy / 8.0);
    float checker = mod(cell.x + cell.y, 2.0);
    vec3 bg = mix(vec3(0.36), vec3(0.62), checker);
    FragColor = vec4(mix(bg, c.rgb, c.a), 1.0);
}";
        uint v = Compile(ShaderType.VertexShader, vs);
        uint f = Compile(ShaderType.FragmentShader, fs);
        _program = _gl.CreateProgram();
        _gl.AttachShader(_program, v);
        _gl.AttachShader(_program, f);
        _gl.LinkProgram(_program);
        _gl.GetProgram(_program, ProgramPropertyARB.LinkStatus, out int linked);
        _gl.DeleteShader(v);
        _gl.DeleteShader(f);
        if (linked == 0) throw new InvalidOperationException("Texture view shader link failed: " + _gl.GetProgramInfoLog(_program));
        _uHalf = _gl.GetUniformLocation(_program, "uHalf");
        _uCenter = _gl.GetUniformLocation(_program, "uCenter");
        _uTex = _gl.GetUniformLocation(_program, "uTex");

        float[] quad = [-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1];
        _vao = _gl.GenVertexArray();
        _vbo = _gl.GenBuffer();
        _gl.BindVertexArray(_vao);
        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, _vbo);
        fixed (float* p = quad)
            _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(quad.Length * sizeof(float)), p, BufferUsageARB.StaticDraw);
        _gl.EnableVertexAttribArray(0);
        _gl.VertexAttribPointer(0, 2, VertexAttribPointerType.Float, false, 2 * sizeof(float), null);
        _gl.BindVertexArray(0);

        _texture = _gl.GenTexture();
        _gl.ActiveTexture(TextureUnit.Texture0);
        _gl.BindBuffer(BufferTargetARB.PixelUnpackBuffer, 0);
        _gl.BindTexture(TextureTarget.Texture2D, _texture);
        _gl.PixelStore(PixelStoreParameter.UnpackAlignment, 1);
        fixed (byte* p = rgba)
            _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba8, (uint)width, (uint)height, 0,
                PixelFormat.Rgba, PixelType.UnsignedByte, p);
        _gl.PixelStore(PixelStoreParameter.UnpackAlignment, 4);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);
        SetFilter(nearest: false, force: true);
        _gl.BindTexture(TextureTarget.Texture2D, 0);
    }

    private uint Compile(ShaderType type, string source)
    {
        uint shader = _gl.CreateShader(type);
        _gl.ShaderSource(shader, source);
        _gl.CompileShader(shader);
        _gl.GetShader(shader, ShaderParameterName.CompileStatus, out int ok);
        if (ok == 0)
        {
            string log = _gl.GetShaderInfoLog(shader);
            _gl.DeleteShader(shader);
            throw new InvalidOperationException($"Texture view {type} compile failed: {log}");
        }
        return shader;
    }

    private void SetFilter(bool nearest, bool force = false)
    {
        if (!force && nearest == _nearest) return;
        _nearest = nearest;
        int filter = (int)(nearest ? TextureMinFilter.Nearest : TextureMinFilter.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, filter);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, filter);
    }

    public unsafe void Render(Matrix4x4 view, Matrix4x4 proj)
    {
        if (_disposed) return;
        Span<int> viewport = stackalloc int[4];
        fixed (int* vp = viewport) _gl.GetInteger(GetPName.Viewport, vp);
        float vw = Math.Max(1, viewport[2]), vh = Math.Max(1, viewport[3]);

        if (ImGui.GetCurrentContext() != IntPtr.Zero)
        {
            ImGuiIOPtr io = ImGui.GetIO();
            if (!io.WantCaptureMouse)
            {
                if (io.MouseWheel != 0) _zoom = Math.Clamp(_zoom * MathF.Pow(1.15f, io.MouseWheel), 0.05f, 64f);
                if (ImGui.IsMouseDown(ImGuiMouseButton.Left) || ImGui.IsMouseDown(ImGuiMouseButton.Middle))
                    _pan += io.MouseDelta;
                if (ImGui.IsMouseDoubleClicked(ImGuiMouseButton.Left)) { _zoom = 1f; _pan = Vector2.Zero; }
            }
        }

        float fit = MathF.Min(vw / _width, vh / _height) * 0.95f;
        float scale = fit * _zoom;
        Vector2 half = new(_width * scale / vw, _height * scale / vh);
        Vector2 center = new(_pan.X * 2f / vw, -_pan.Y * 2f / vh);

        bool depth = _gl.IsEnabled(EnableCap.DepthTest);
        bool cull = _gl.IsEnabled(EnableCap.CullFace);
        bool blend = _gl.IsEnabled(EnableCap.Blend);
        _gl.Disable(EnableCap.DepthTest);
        _gl.Disable(EnableCap.CullFace);
        _gl.Disable(EnableCap.Blend);
        _gl.UseProgram(_program);
        _gl.Uniform2(_uHalf, half.X, half.Y);
        _gl.Uniform2(_uCenter, center.X, center.Y);
        _gl.Uniform1(_uTex, 0);
        _gl.ActiveTexture(TextureUnit.Texture0);
        _gl.BindTexture(TextureTarget.Texture2D, _texture);
        SetFilter(scale > 1.5f);
        _gl.BindVertexArray(_vao);
        _gl.DrawArrays(PrimitiveType.Triangles, 0, 6);
        _gl.BindVertexArray(0);
        _gl.BindTexture(TextureTarget.Texture2D, 0);
        _gl.UseProgram(0);
        if (depth) _gl.Enable(EnableCap.DepthTest);
        if (cull) _gl.Enable(EnableCap.CullFace);
        if (blend) _gl.Enable(EnableCap.Blend);
    }

    public void ToggleWireframe() { }
    public bool IsWireframe => false;
    public int SubObjectCount => 0;
    public string GetSubObjectName(int index) => string.Empty;
    public bool GetSubObjectVisible(int index) => false;
    public void SetSubObjectVisible(int index, bool visible) { }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_texture != 0) _gl.DeleteTexture(_texture);
        if (_vbo != 0) _gl.DeleteBuffer(_vbo);
        if (_vao != 0) _gl.DeleteVertexArray(_vao);
        if (_program != 0) _gl.DeleteProgram(_program);
        _texture = _vbo = _vao = _program = 0;
    }
}
