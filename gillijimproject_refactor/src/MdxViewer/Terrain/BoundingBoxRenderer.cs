using System.Numerics;
using Silk.NET.OpenGL;

namespace MdxViewer.Terrain;

/// <summary>
/// Renders wireframe bounding boxes for debug visualization of object placements.
/// Each box is drawn as 12 line segments forming a wireframe cube.
/// </summary>
public class BoundingBoxRenderer : IDisposable
{
    private readonly GL _gl;
    private uint _vao, _vbo, _ebo, _shader;
    private int _uMvp, _uColor;
    private bool _initialized;

    // 8 vertices of a unit cube (0..1), scaled/translated per box
    private static readonly float[] CubeVertices =
    {
        0, 0, 0,  1, 0, 0,  1, 1, 0,  0, 1, 0, // bottom face
        0, 0, 1,  1, 0, 1,  1, 1, 1,  0, 1, 1  // top face
    };

    // 24 indices forming 12 line segments
    private static readonly ushort[] CubeIndices =
    {
        0,1, 1,2, 2,3, 3,0, // bottom
        4,5, 5,6, 6,7, 7,4, // top
        0,4, 1,5, 2,6, 3,7  // verticals
    };

    public BoundingBoxRenderer(GL gl)
    {
        _gl = gl;
        Initialize();
    }

    private unsafe void Initialize()
    {
        // Shader
        string vertSrc = @"#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uMVP;
void main() { gl_Position = uMVP * vec4(aPos, 1.0); }";

        string fragSrc = @"#version 330 core
uniform vec3 uColor;
out vec4 FragColor;
void main() { FragColor = vec4(uColor, 1.0); }";

        uint vs = CompileShader(ShaderType.VertexShader, vertSrc);
        uint fs = CompileShader(ShaderType.FragmentShader, fragSrc);
        _shader = _gl.CreateProgram();
        _gl.AttachShader(_shader, vs);
        _gl.AttachShader(_shader, fs);
        _gl.LinkProgram(_shader);
        _gl.DeleteShader(vs);
        _gl.DeleteShader(fs);

        _uMvp = _gl.GetUniformLocation(_shader, "uMVP");
        _uColor = _gl.GetUniformLocation(_shader, "uColor");

        // VAO/VBO/EBO
        _vao = _gl.GenVertexArray();
        _vbo = _gl.GenBuffer();
        _ebo = _gl.GenBuffer();

        _gl.BindVertexArray(_vao);

        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, _vbo);
        fixed (float* p = CubeVertices)
            _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(CubeVertices.Length * sizeof(float)), p, BufferUsageARB.StaticDraw);

        _gl.BindBuffer(BufferTargetARB.ElementArrayBuffer, _ebo);
        fixed (ushort* p = CubeIndices)
            _gl.BufferData(BufferTargetARB.ElementArrayBuffer, (nuint)(CubeIndices.Length * sizeof(ushort)), p, BufferUsageARB.StaticDraw);

        _gl.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 3 * sizeof(float), (void*)0);
        _gl.EnableVertexAttribArray(0);

        _gl.BindVertexArray(0);
        _initialized = true;
    }

    /// <summary>
    /// Draw a wireframe box at the given position with the given size.
    /// </summary>
    public unsafe void DrawBox(Vector3 center, Vector3 halfExtents, Matrix4x4 view, Matrix4x4 proj, Vector3 color)
    {
        if (!_initialized) return;

        // Model matrix: scale unit cube to size, then translate to position
        var model = Matrix4x4.CreateScale(halfExtents.X * 2, halfExtents.Y * 2, halfExtents.Z * 2)
            * Matrix4x4.CreateTranslation(center - halfExtents);
        var mvp = model * view * proj;

        _gl.UseProgram(_shader);
        _gl.UniformMatrix4(_uMvp, 1, false, (float*)&mvp);
        _gl.Uniform3(_uColor, color.X, color.Y, color.Z);

        _gl.BindVertexArray(_vao);
        _gl.DrawElements(PrimitiveType.Lines, (uint)CubeIndices.Length, DrawElementsType.UnsignedShort, (void*)0);
        _gl.BindVertexArray(0);
    }

    /// <summary>
    /// Draw a wireframe box from min/max corners.
    /// </summary>
    public void DrawBoxMinMax(Vector3 min, Vector3 max, Matrix4x4 view, Matrix4x4 proj, Vector3 color)
    {
        var center = (min + max) * 0.5f;
        var halfExtents = (max - min) * 0.5f;
        // Ensure positive extents
        halfExtents = new Vector3(MathF.Abs(halfExtents.X), MathF.Abs(halfExtents.Y), MathF.Abs(halfExtents.Z));
        if (halfExtents.X < 0.1f) halfExtents = new Vector3(5f, 5f, 5f); // fallback for zero-size
        DrawBox(center, halfExtents, view, proj, color);
    }

    /// <summary>
    /// Draw a small marker box at a point position.
    /// </summary>
    public void DrawMarker(Vector3 position, float size, Matrix4x4 view, Matrix4x4 proj, Vector3 color)
    {
        var half = new Vector3(size, size, size);
        DrawBox(position, half, view, proj, color);
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
            Console.WriteLine($"[BoundingBoxRenderer] Shader compile error: {log}");
        }
        return shader;
    }

    public void Dispose()
    {
        if (_initialized)
        {
            _gl.DeleteVertexArray(_vao);
            _gl.DeleteBuffer(_vbo);
            _gl.DeleteBuffer(_ebo);
            _gl.DeleteProgram(_shader);
            _initialized = false;
        }
    }
}
