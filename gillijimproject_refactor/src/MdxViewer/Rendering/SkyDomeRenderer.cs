using System.Numerics;
using MdxViewer.Logging;
using Silk.NET.OpenGL;

namespace MdxViewer.Rendering;

/// <summary>
/// Procedural sky dome renderer based on Ghidra analysis of WoW Alpha 0.5.3 sky system.
/// Renders a hemisphere with time-of-day color gradient before terrain (depth write OFF).
/// Colors driven by TerrainLighting: zenith color at top, horizon/fog color at edges.
/// </summary>
public class SkyDomeRenderer : IDisposable
{
    private readonly GL _gl;
    private uint _vao, _vbo, _ebo;
    private uint _shaderProgram;
    private int _uView, _uProj, _uCameraPos;
    private int _uZenithColor, _uHorizonColor, _uFogColor;
    private int _indexCount;

    // Sky colors (set externally from TerrainLighting)
    public Vector3 ZenithColor { get; set; } = new(0.3f, 0.5f, 0.9f);   // deep blue
    public Vector3 HorizonColor { get; set; } = new(0.6f, 0.7f, 0.85f); // light blue
    public Vector3 SkyFogColor { get; set; } = new(0.6f, 0.7f, 0.85f);  // fog blend

    public SkyDomeRenderer(GL gl)
    {
        _gl = gl;
        BuildDome(32, 16, 5000f);
        BuildShader();
    }

    /// <summary>
    /// Update sky colors from TerrainLighting time-of-day system.
    /// </summary>
    public void UpdateFromLighting(float gameTime)
    {
        float sunAngle = gameTime * MathF.PI * 2f;
        float sunHeight = MathF.Sin(sunAngle - MathF.PI * 0.5f);
        float dayFactor = MathF.Max(0, sunHeight);

        // Zenith: dark blue at night, sky blue at day
        ZenithColor = Vector3.Lerp(
            new Vector3(0.01f, 0.01f, 0.05f),  // night
            new Vector3(0.25f, 0.45f, 0.85f),   // day
            dayFactor);

        // Horizon: dark at night, light haze at day
        HorizonColor = Vector3.Lerp(
            new Vector3(0.03f, 0.03f, 0.08f),  // night
            new Vector3(0.65f, 0.75f, 0.90f),   // day
            dayFactor);

        // Dawn/dusk warm tones
        float dawnFactor = 0f;
        if (gameTime > 0.20f && gameTime < 0.35f)
            dawnFactor = 1f - MathF.Abs(gameTime - 0.275f) / 0.075f;
        else if (gameTime > 0.65f && gameTime < 0.80f)
            dawnFactor = 1f - MathF.Abs(gameTime - 0.725f) / 0.075f;
        dawnFactor = MathF.Max(0, dawnFactor);

        if (dawnFactor > 0)
        {
            var dawnHorizon = new Vector3(0.9f, 0.5f, 0.3f); // warm orange
            HorizonColor = Vector3.Lerp(HorizonColor, dawnHorizon, dawnFactor * 0.7f);
            ZenithColor = Vector3.Lerp(ZenithColor, new Vector3(0.3f, 0.2f, 0.5f), dawnFactor * 0.3f);
        }

        SkyFogColor = HorizonColor;
    }

    public unsafe void Render(Matrix4x4 view, Matrix4x4 proj, Vector3 cameraPos)
    {
        _gl.UseProgram(_shaderProgram);

        // Sky dome follows camera (no depth write, always behind everything)
        _gl.Disable(EnableCap.DepthTest);
        _gl.DepthMask(false);
        _gl.Disable(EnableCap.CullFace);

        _gl.UniformMatrix4(_uView, 1, false, (float*)&view);
        _gl.UniformMatrix4(_uProj, 1, false, (float*)&proj);
        _gl.Uniform3(_uCameraPos, cameraPos.X, cameraPos.Y, cameraPos.Z);
        _gl.Uniform3(_uZenithColor, ZenithColor.X, ZenithColor.Y, ZenithColor.Z);
        _gl.Uniform3(_uHorizonColor, HorizonColor.X, HorizonColor.Y, HorizonColor.Z);
        _gl.Uniform3(_uFogColor, SkyFogColor.X, SkyFogColor.Y, SkyFogColor.Z);

        _gl.BindVertexArray(_vao);
        _gl.DrawElements(PrimitiveType.Triangles, (uint)_indexCount, DrawElementsType.UnsignedShort, null);
        _gl.BindVertexArray(0);

        // Restore state
        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthMask(true);
    }

    private unsafe void BuildDome(int segments, int rings, float radius)
    {
        // Generate hemisphere vertices: position (3 floats) + height factor (1 float)
        int vertCount = (rings + 1) * (segments + 1);
        float[] verts = new float[vertCount * 4]; // x, y, z, heightFactor
        int vi = 0;

        for (int r = 0; r <= rings; r++)
        {
            // phi: 0 = horizon, PI/2 = zenith
            float phi = (float)r / rings * MathF.PI * 0.5f;
            float y = MathF.Sin(phi) * radius;
            float ringRadius = MathF.Cos(phi) * radius;
            float heightFactor = (float)r / rings; // 0 at horizon, 1 at zenith

            for (int s = 0; s <= segments; s++)
            {
                float theta = (float)s / segments * MathF.PI * 2f;
                float x = MathF.Cos(theta) * ringRadius;
                float z = MathF.Sin(theta) * ringRadius;

                verts[vi++] = x;
                verts[vi++] = y;
                verts[vi++] = z;
                verts[vi++] = heightFactor;
            }
        }

        // Generate indices
        var indices = new List<ushort>();
        for (int r = 0; r < rings; r++)
        {
            for (int s = 0; s < segments; s++)
            {
                int curr = r * (segments + 1) + s;
                int next = curr + segments + 1;

                indices.Add((ushort)curr);
                indices.Add((ushort)next);
                indices.Add((ushort)(curr + 1));

                indices.Add((ushort)(curr + 1));
                indices.Add((ushort)next);
                indices.Add((ushort)(next + 1));
            }
        }
        _indexCount = indices.Count;

        // Upload to GPU
        _vao = _gl.GenVertexArray();
        _vbo = _gl.GenBuffer();
        _ebo = _gl.GenBuffer();

        _gl.BindVertexArray(_vao);

        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, _vbo);
        fixed (float* ptr = verts)
            _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(verts.Length * sizeof(float)), ptr, BufferUsageARB.StaticDraw);

        _gl.BindBuffer(BufferTargetARB.ElementArrayBuffer, _ebo);
        var idxArr = indices.ToArray();
        fixed (ushort* ptr = idxArr)
            _gl.BufferData(BufferTargetARB.ElementArrayBuffer, (nuint)(idxArr.Length * sizeof(ushort)), ptr, BufferUsageARB.StaticDraw);

        // Position: location 0, 3 floats
        _gl.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 4 * sizeof(float), (void*)0);
        _gl.EnableVertexAttribArray(0);
        // Height factor: location 1, 1 float
        _gl.VertexAttribPointer(1, 1, VertexAttribPointerType.Float, false, 4 * sizeof(float), (void*)(3 * sizeof(float)));
        _gl.EnableVertexAttribArray(1);

        _gl.BindVertexArray(0);
    }

    private void BuildShader()
    {
        string vertSrc = @"
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in float aHeight;

uniform mat4 uView;
uniform mat4 uProj;
uniform vec3 uCameraPos;

out float vHeight;

void main() {
    // Sky dome follows camera position
    vec3 worldPos = aPos + uCameraPos;
    gl_Position = uProj * uView * vec4(worldPos, 1.0);
    vHeight = aHeight;
}
";

        string fragSrc = @"
#version 330 core
in float vHeight;

uniform vec3 uZenithColor;
uniform vec3 uHorizonColor;
uniform vec3 uFogColor;

out vec4 FragColor;

void main() {
    // Gradient: fog color at very bottom, horizon at edge, zenith at top
    // Use smoothstep for natural sky gradient
    float t = smoothstep(0.0, 0.6, vHeight);
    vec3 skyColor = mix(uHorizonColor, uZenithColor, t);

    // Blend toward fog color near the very bottom (below horizon line)
    float fogBlend = smoothstep(0.15, 0.0, vHeight);
    skyColor = mix(skyColor, uFogColor, fogBlend);

    FragColor = vec4(skyColor, 1.0);
}
";

        uint vs = CompileShader(ShaderType.VertexShader, vertSrc);
        uint fs = CompileShader(ShaderType.FragmentShader, fragSrc);

        _shaderProgram = _gl.CreateProgram();
        _gl.AttachShader(_shaderProgram, vs);
        _gl.AttachShader(_shaderProgram, fs);
        _gl.LinkProgram(_shaderProgram);

        _gl.GetProgram(_shaderProgram, ProgramPropertyARB.LinkStatus, out int linkStatus);
        if (linkStatus == 0)
        {
            string log = _gl.GetProgramInfoLog(_shaderProgram);
            ViewerLog.Trace($"[SkyDome] Shader link error: {log}");
        }

        _gl.DeleteShader(vs);
        _gl.DeleteShader(fs);

        _gl.UseProgram(_shaderProgram);
        _uView = _gl.GetUniformLocation(_shaderProgram, "uView");
        _uProj = _gl.GetUniformLocation(_shaderProgram, "uProj");
        _uCameraPos = _gl.GetUniformLocation(_shaderProgram, "uCameraPos");
        _uZenithColor = _gl.GetUniformLocation(_shaderProgram, "uZenithColor");
        _uHorizonColor = _gl.GetUniformLocation(_shaderProgram, "uHorizonColor");
        _uFogColor = _gl.GetUniformLocation(_shaderProgram, "uFogColor");
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
            ViewerLog.Trace($"[SkyDome] Shader compile error ({type}): {log}");
        }
        return shader;
    }

    public void Dispose()
    {
        _gl.DeleteVertexArray(_vao);
        _gl.DeleteBuffer(_vbo);
        _gl.DeleteBuffer(_ebo);
        _gl.DeleteProgram(_shaderProgram);
    }
}
