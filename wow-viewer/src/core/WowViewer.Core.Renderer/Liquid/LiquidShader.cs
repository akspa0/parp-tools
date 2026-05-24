using System.Numerics;
using Silk.NET.OpenGL;

namespace WowViewer.Core.Renderer.Liquid;

public sealed unsafe class LiquidShader : IDisposable
{
    private readonly GL _gl;
    private readonly uint _program;
    private readonly int _uViewProj;
    private readonly int _uCameraPos;
    private readonly int _uModel;
    private readonly int _uColor;
    private readonly int _uOpacity;
    private bool _disposed;

    private const string VsSrc = @"#version 410 core
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec2 aUv;
out vec2 vUv;
uniform mat4 uViewProj;
uniform mat4 uModel;
void main() {
    gl_Position = uViewProj * uModel * vec4(aPosition, 1.0);
    vUv = aUv;
}";

    private const string FsSrc = @"#version 410 core
in vec2 vUv;
out vec4 FragColor;
uniform vec3 uColor;
uniform float uOpacity;
void main() {
    FragColor = vec4(uColor, uOpacity);
}";

    public LiquidShader(GL gl)
    {
        _gl = gl;
        _program = CreateProgram(gl, VsSrc, FsSrc);
        _uViewProj = gl.GetUniformLocation(_program, "uViewProj");
        _uCameraPos = gl.GetUniformLocation(_program, "uCameraPos");
        _uModel = gl.GetUniformLocation(_program, "uModel");
        _uColor = gl.GetUniformLocation(_program, "uColor");
        _uOpacity = gl.GetUniformLocation(_program, "uOpacity");
    }

    public void Use() => _gl.UseProgram(_program);

    public void SetViewProj(Matrix4x4 viewProj)
    {
        _gl.UniformMatrix4(_uViewProj, 1, false, (float*)&viewProj);
    }

    public void SetModel(Matrix4x4 model)
    {
        _gl.UniformMatrix4(_uModel, 1, true, (float*)&model);
    }

    public void SetColor(Vector3 color) => _gl.Uniform3(_uColor, color);
    public void SetOpacity(float opacity) => _gl.Uniform1(_uOpacity, opacity);

    internal static unsafe uint CreateProgram(GL gl, string vsSrc, string fsSrc)
    {
        uint vs = gl.CreateShader(ShaderType.VertexShader);
        gl.ShaderSource(vs, vsSrc);
        gl.CompileShader(vs);
        gl.GetShader(vs, ShaderParameterName.CompileStatus, out int vsOk);
        if (vsOk == 0) throw new InvalidOperationException($"Liquid VS: {gl.GetShaderInfoLog(vs)}");

        uint fs = gl.CreateShader(ShaderType.FragmentShader);
        gl.ShaderSource(fs, fsSrc);
        gl.CompileShader(fs);
        gl.GetShader(fs, ShaderParameterName.CompileStatus, out int fsOk);
        if (fsOk == 0) throw new InvalidOperationException($"Liquid FS: {gl.GetShaderInfoLog(fs)}");

        uint prog = gl.CreateProgram();
        gl.AttachShader(prog, vs);
        gl.AttachShader(prog, fs);
        gl.LinkProgram(prog);
        gl.GetProgram(prog, ProgramPropertyARB.LinkStatus, out int linkOk);
        if (linkOk == 0) throw new InvalidOperationException($"Liquid link: {gl.GetProgramInfoLog(prog)}");

        gl.DeleteShader(vs);
        gl.DeleteShader(fs);
        return prog;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _gl.DeleteProgram(_program);
    }
}
