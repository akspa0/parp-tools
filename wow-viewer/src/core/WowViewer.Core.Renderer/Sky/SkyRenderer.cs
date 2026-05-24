using System.Numerics;
using Silk.NET.OpenGL;
using WowViewer.Core.Renderer.Scene;

namespace WowViewer.Core.Renderer.Sky;

public sealed unsafe class SkyRenderer : IDisposable
{
    private readonly GL _gl;
    private readonly uint _vao;
    private readonly uint _vbo;
    private readonly uint _program;
    private readonly int _uViewProj;
    private readonly int _uCameraPos;
    private readonly int _uTopColor;
    private readonly int _uBottomColor;
    private bool _disposed;

    private static readonly string VsSrc = @"#version 410 core
layout(location = 0) in vec2 aPosition;
out vec2 vUv;
uniform mat4 uViewProj;
uniform vec3 uCameraPos;
void main() {
    vec4 pos = uViewProj * vec4(aPosition.x * 5000.0 + uCameraPos.x, aPosition.y * 5000.0 + uCameraPos.y, uCameraPos.z - 2000.0, 1.0);
    gl_Position = pos;
    vUv = aPosition * 0.5 + 0.5;
}";

    private static readonly string FsSrc = @"#version 410 core
in vec2 vUv;
out vec4 FragColor;
uniform vec3 uTopColor;
uniform vec3 uBottomColor;
void main() {
    vec3 color = mix(uBottomColor, uTopColor, vUv.y);
    FragColor = vec4(color, 1.0);
}";

    public SkyRenderer(GL gl)
    {
        _gl = gl;

        _program = CreateProgram(gl, VsSrc, FsSrc);
        _uViewProj = gl.GetUniformLocation(_program, "uViewProj");
        _uCameraPos = gl.GetUniformLocation(_program, "uCameraPos");
        _uTopColor = gl.GetUniformLocation(_program, "uTopColor");
        _uBottomColor = gl.GetUniformLocation(_program, "uBottomColor");

        float[] verts = [-1f, -1f,  1f, -1f,  1f, 1f,  -1f, -1f,  1f, 1f,  -1f, 1f];
        _vao = gl.GenVertexArray();
        _vbo = gl.GenBuffer();
        gl.BindVertexArray(_vao);
        gl.BindBuffer(BufferTargetARB.ArrayBuffer, _vbo);
        fixed (float* ptr = verts)
            gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(verts.Length * sizeof(float)), ptr, BufferUsageARB.StaticDraw);
        gl.VertexAttribPointer(0, 2, VertexAttribPointerType.Float, false, 0, null);
        gl.EnableVertexAttribArray(0);
        gl.BindVertexArray(0);
    }

    public void Render(SceneCamera camera)
    {
        _gl.UseProgram(_program);
        var viewProj = camera.GetViewMatrix() * camera.GetProjectionMatrix();
        _gl.UniformMatrix4(_uViewProj, 1, false, (float*)&viewProj);
        _gl.Uniform3(_uCameraPos, camera.Position);
        _gl.Uniform3(_uTopColor, new Vector3(0.25f, 0.35f, 0.55f));
        _gl.Uniform3(_uBottomColor, new Vector3(0.72f, 0.68f, 0.62f));

        _gl.Disable(EnableCap.DepthTest);
        _gl.BindVertexArray(_vao);
        _gl.DrawArrays(PrimitiveType.Triangles, 0, 6);
        _gl.BindVertexArray(0);
        _gl.Enable(EnableCap.DepthTest);
    }

    internal static uint CreateProgram(GL gl, string vsSrc, string fsSrc)
    {
        uint vs = gl.CreateShader(ShaderType.VertexShader);
        gl.ShaderSource(vs, vsSrc);
        gl.CompileShader(vs);
        gl.GetShader(vs, ShaderParameterName.CompileStatus, out int vsOk);
        if (vsOk == 0)
            throw new InvalidOperationException($"Sky VS compile: {gl.GetShaderInfoLog(vs)}");

        uint fs = gl.CreateShader(ShaderType.FragmentShader);
        gl.ShaderSource(fs, fsSrc);
        gl.CompileShader(fs);
        gl.GetShader(fs, ShaderParameterName.CompileStatus, out int fsOk);
        if (fsOk == 0)
            throw new InvalidOperationException($"Sky FS compile: {gl.GetShaderInfoLog(fs)}");

        uint prog = gl.CreateProgram();
        gl.AttachShader(prog, vs);
        gl.AttachShader(prog, fs);
        gl.LinkProgram(prog);
        gl.GetProgram(prog, ProgramPropertyARB.LinkStatus, out int linkOk);
        if (linkOk == 0)
            throw new InvalidOperationException($"Sky link: {gl.GetProgramInfoLog(prog)}");

        gl.DeleteShader(vs);
        gl.DeleteShader(fs);
        return prog;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _gl.DeleteBuffer(_vbo);
        _gl.DeleteVertexArray(_vao);
        _gl.DeleteProgram(_program);
    }
}
