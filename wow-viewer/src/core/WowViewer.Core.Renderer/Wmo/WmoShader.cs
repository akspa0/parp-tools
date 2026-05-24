using System.Numerics;
using Silk.NET.OpenGL;

namespace WowViewer.Core.Renderer.Wmo;

public sealed unsafe class WmoShader : IDisposable
{
    private readonly GL _gl;
    private readonly uint _program;
    private readonly int _uViewProj;
    private readonly int _uModel;
    private readonly int _uDiffuseColor;
    private bool _disposed;

    private const string VsSrc = @"#version 410 core
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUv;
out vec2 vUv;
out vec3 vNormal;
uniform mat4 uViewProj;
uniform mat4 uModel;
void main() {
    gl_Position = uViewProj * uModel * vec4(aPosition, 1.0);
    vUv = aUv;
    vNormal = mat3(uModel) * aNormal;
}";

    private const string FsSrc = @"#version 410 core
in vec2 vUv;
in vec3 vNormal;
out vec4 FragColor;
uniform vec3 uDiffuseColor;
void main() {
    vec3 lightDir = normalize(vec3(0.5, -0.5, 0.8));
    float diff = max(dot(vNormal, lightDir), 0.15);
    FragColor = vec4(uDiffuseColor * diff, 1.0);
}";

    public WmoShader(GL gl)
    {
        _gl = gl;
        _program = CreateProgram(gl, VsSrc, FsSrc);
        _uViewProj = gl.GetUniformLocation(_program, "uViewProj");
        _uModel = gl.GetUniformLocation(_program, "uModel");
        _uDiffuseColor = gl.GetUniformLocation(_program, "uDiffuseColor");
    }

    public void Use() => _gl.UseProgram(_program);
    public void SetViewProj(Matrix4x4 m) => _gl.UniformMatrix4(_uViewProj, 1, false, (float*)&m);
    public void SetModel(Matrix4x4 m) => _gl.UniformMatrix4(_uModel, 1, true, (float*)&m);
    public void SetDiffuseColor(Vector3 c) => _gl.Uniform3(_uDiffuseColor, c);

    private static uint CreateProgram(GL gl, string vsSrc, string fsSrc)
    {
        uint vs = gl.CreateShader(ShaderType.VertexShader);
        gl.ShaderSource(vs, vsSrc);
        gl.CompileShader(vs);
        gl.GetShader(vs, ShaderParameterName.CompileStatus, out int vsOk);
        if (vsOk == 0) throw new InvalidOperationException($"WMO VS: {gl.GetShaderInfoLog(vs)}");

        uint fs = gl.CreateShader(ShaderType.FragmentShader);
        gl.ShaderSource(fs, fsSrc);
        gl.CompileShader(fs);
        gl.GetShader(fs, ShaderParameterName.CompileStatus, out int fsOk);
        if (fsOk == 0) throw new InvalidOperationException($"WMO FS: {gl.GetShaderInfoLog(fs)}");

        uint prog = gl.CreateProgram();
        gl.AttachShader(prog, vs);
        gl.AttachShader(prog, fs);
        gl.LinkProgram(prog);
        gl.GetProgram(prog, ProgramPropertyARB.LinkStatus, out int linkOk);
        if (linkOk == 0) throw new InvalidOperationException($"WMO link: {gl.GetProgramInfoLog(prog)}");
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
