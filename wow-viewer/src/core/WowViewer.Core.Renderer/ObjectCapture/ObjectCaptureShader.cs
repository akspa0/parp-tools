using System.Numerics;
using Silk.NET.OpenGL;

namespace WowViewer.Core.Renderer.ObjectCapture;

/// <summary>
/// Shared shader for single-object headless capture (WMO and MDX). Textured, lightly lit,
/// with a mask-mode uniform that switches the whole draw to a flat unlit white silhouette --
/// the harvest-side object-mask signal. One program serves both renderers so capture cost stays
/// low and the two passes (textured, mask) are guaranteed pixel-aligned (same geometry, same
/// camera, only the fragment output differs).
/// </summary>
public sealed unsafe class ObjectCaptureShader : IDisposable
{
    private readonly GL _gl;
    private readonly uint _program;
    private readonly int _uViewProj;
    private readonly int _uModel;
    private readonly int _uHasTexture;
    private readonly int _uMaskMode;
    private readonly int _uSampler;
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
uniform sampler2D uSampler;
uniform int uHasTexture;
uniform int uMaskMode;
void main() {
    if (uMaskMode != 0) {
        // Flat unlit silhouette: every drawn fragment is a solid mask pixel.
        FragColor = vec4(1.0, 1.0, 1.0, 1.0);
        return;
    }
    vec3 baseColor = uHasTexture != 0 ? texture(uSampler, vUv).rgb : vec3(0.75, 0.75, 0.75);
    // WoW geometry normals use DirectX convention (clockwise winding). OpenGL defaults to
    // counter-clockwise front faces, which inverts the Z component of the effective normals.
    // Negate lightDir.Z so the upward-pointing light correctly lights top surfaces instead of
    // bottom surfaces (the marshmallow backlighting bug).
    vec3 lightDir = normalize(vec3(0.4, -0.5, -0.85));
    float diff = max(dot(normalize(vNormal), lightDir), 0.25);
    FragColor = vec4(baseColor * diff, 1.0);
}";

    public ObjectCaptureShader(GL gl)
    {
        _gl = gl;
        _program = CreateProgram(gl, VsSrc, FsSrc);
        _uViewProj = gl.GetUniformLocation(_program, "uViewProj");
        _uModel = gl.GetUniformLocation(_program, "uModel");
        _uHasTexture = gl.GetUniformLocation(_program, "uHasTexture");
        _uMaskMode = gl.GetUniformLocation(_program, "uMaskMode");
        _uSampler = gl.GetUniformLocation(_program, "uSampler");
    }

    public void Use() => _gl.UseProgram(_program);
    public void SetViewProj(Matrix4x4 m) => _gl.UniformMatrix4(_uViewProj, 1, false, (float*)&m);
    public void SetModel(Matrix4x4 m) => _gl.UniformMatrix4(_uModel, 1, true, (float*)&m);
    public void SetHasTexture(bool has) => _gl.Uniform1(_uHasTexture, has ? 1 : 0);
    public void SetMaskMode(bool enabled) => _gl.Uniform1(_uMaskMode, enabled ? 1 : 0);
    public void SetSamplerUnit(int unit) => _gl.Uniform1(_uSampler, unit);

    private static uint CreateProgram(GL gl, string vsSrc, string fsSrc)
    {
        uint vs = gl.CreateShader(ShaderType.VertexShader);
        gl.ShaderSource(vs, vsSrc);
        gl.CompileShader(vs);
        gl.GetShader(vs, ShaderParameterName.CompileStatus, out int vsOk);
        if (vsOk == 0) throw new InvalidOperationException($"ObjectCapture VS: {gl.GetShaderInfoLog(vs)}");

        uint fs = gl.CreateShader(ShaderType.FragmentShader);
        gl.ShaderSource(fs, fsSrc);
        gl.CompileShader(fs);
        gl.GetShader(fs, ShaderParameterName.CompileStatus, out int fsOk);
        if (fsOk == 0) throw new InvalidOperationException($"ObjectCapture FS: {gl.GetShaderInfoLog(fs)}");

        uint prog = gl.CreateProgram();
        gl.AttachShader(prog, vs);
        gl.AttachShader(prog, fs);
        gl.LinkProgram(prog);
        gl.GetProgram(prog, ProgramPropertyARB.LinkStatus, out int linkOk);
        if (linkOk == 0) throw new InvalidOperationException($"ObjectCapture link: {gl.GetProgramInfoLog(prog)}");
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
