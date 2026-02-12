using System.Numerics;
using Silk.NET.OpenGL;

namespace MdxViewer.Rendering;

/// <summary>
/// Thin wrapper around an OpenGL shader program: compile, link, uniform cache.
/// Replaces duplicated shader setup in MdxRenderer / WmoRenderer.
/// </summary>
public class ShaderProgram : IDisposable
{
    private readonly GL _gl;
    private readonly uint _handle;
    private readonly Dictionary<string, int> _uniformCache = new();
    private bool _disposed;

    public uint Handle => _handle;

    private ShaderProgram(GL gl, uint handle)
    {
        _gl = gl;
        _handle = handle;
    }

    /// <summary>
    /// Compile vertex + fragment source and link into a program.
    /// Throws on compile/link failure.
    /// </summary>
    public static ShaderProgram Create(GL gl, string vertexSource, string fragmentSource)
    {
        uint vs = CompileShader(gl, ShaderType.VertexShader, vertexSource);
        uint fs = CompileShader(gl, ShaderType.FragmentShader, fragmentSource);

        uint program = gl.CreateProgram();
        gl.AttachShader(program, vs);
        gl.AttachShader(program, fs);
        gl.LinkProgram(program);

        gl.GetProgram(program, ProgramPropertyARB.LinkStatus, out int status);
        if (status == 0)
        {
            string log = gl.GetProgramInfoLog(program);
            gl.DeleteProgram(program);
            gl.DeleteShader(vs);
            gl.DeleteShader(fs);
            throw new InvalidOperationException($"Shader link failed: {log}");
        }

        gl.DetachShader(program, vs);
        gl.DetachShader(program, fs);
        gl.DeleteShader(vs);
        gl.DeleteShader(fs);

        return new ShaderProgram(gl, program);
    }

    public void Use() => _gl.UseProgram(_handle);

    // ── Uniform setters ──────────────────────────────────────────────────

    public int GetUniformLocation(string name)
    {
        if (_uniformCache.TryGetValue(name, out int loc))
            return loc;
        loc = _gl.GetUniformLocation(_handle, name);
        _uniformCache[name] = loc;
        return loc;
    }

    public void SetInt(string name, int value)
        => _gl.Uniform1(GetUniformLocation(name), value);

    public void SetFloat(string name, float value)
        => _gl.Uniform1(GetUniformLocation(name), value);

    public void SetVec3(string name, Vector3 value)
        => _gl.Uniform3(GetUniformLocation(name), value.X, value.Y, value.Z);

    public void SetVec4(string name, Vector4 value)
        => _gl.Uniform4(GetUniformLocation(name), value.X, value.Y, value.Z, value.W);

    public unsafe void SetMat4(string name, Matrix4x4 value)
        => _gl.UniformMatrix4(GetUniformLocation(name), 1, false, (float*)&value);

    // ── Helpers ──────────────────────────────────────────────────────────

    private static uint CompileShader(GL gl, ShaderType type, string source)
    {
        uint shader = gl.CreateShader(type);
        gl.ShaderSource(shader, source);
        gl.CompileShader(shader);

        gl.GetShader(shader, ShaderParameterName.CompileStatus, out int status);
        if (status == 0)
        {
            string log = gl.GetShaderInfoLog(shader);
            gl.DeleteShader(shader);
            throw new InvalidOperationException($"Shader compile ({type}) failed: {log}");
        }
        return shader;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _gl.DeleteProgram(_handle);
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }
}
