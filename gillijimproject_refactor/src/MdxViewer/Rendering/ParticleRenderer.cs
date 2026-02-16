using System.Numerics;
using MdxLTool.Formats.Mdx;
using MdxViewer.DataSources;
using MdxViewer.Logging;
using Silk.NET.OpenGL;

namespace MdxViewer.Rendering;

/// <summary>
/// GPU-based particle renderer with billboarding and blending.
/// Uses per-particle uniforms for simplicity.
/// </summary>
public unsafe class ParticleRenderer : IDisposable
{
    private readonly GL _gl;
    private uint _shaderProgram;
    private uint _vao, _vbo, _ebo;
    private int _uView, _uProj, _uTexture;
    private int _uCameraRight, _uCameraUp;
    private int _uParticlePos, _uParticleColor, _uParticleSize;
    private int _uHasTexture;
    private int _uRows, _uColumns, _uCellIndex;

    private bool _disposed;

    public ParticleRenderer(GL gl)
    {
        _gl = gl;
        InitShader();
        InitBuffers();
    }
    
    private void InitShader()
    {
        string vertexShader = @"
#version 330 core
layout(location = 0) in vec2 aQuadPos;   // -0.5..0.5 unit quad
layout(location = 1) in vec2 aTexCoord;  // 0..1

out vec2 vTexCoord;

uniform mat4 uView;
uniform mat4 uProj;
uniform vec3 uCameraRight;
uniform vec3 uCameraUp;
uniform vec3 uParticlePos;
uniform float uParticleSize;
uniform int uRows;
uniform int uColumns;
uniform int uCellIndex;

void main()
{
    // Billboard quad facing camera
    vec3 worldPos = uParticlePos 
        + uCameraRight * aQuadPos.x * uParticleSize
        + uCameraUp    * aQuadPos.y * uParticleSize;
    
    gl_Position = uProj * uView * vec4(worldPos, 1.0);

    // Atlas UV: subdivide texture by rows/columns, pick cell
    float cellW = 1.0 / float(uColumns);
    float cellH = 1.0 / float(uRows);
    int row = uCellIndex / uColumns;
    int col = uCellIndex - row * uColumns;
    vTexCoord = vec2(
        (float(col) + aTexCoord.x) * cellW,
        (float(row) + aTexCoord.y) * cellH
    );
}
";

        string fragmentShader = @"
#version 330 core
in vec2 vTexCoord;
out vec4 FragColor;

uniform sampler2D uTexture;
uniform vec4 uParticleColor;
uniform int uHasTexture;

void main()
{
    vec4 texColor = (uHasTexture == 1) ? texture(uTexture, vTexCoord) : vec4(1.0);
    FragColor = texColor * uParticleColor;
    if (FragColor.a < 0.01) discard;
}
";

        uint vs = _gl.CreateShader(ShaderType.VertexShader);
        _gl.ShaderSource(vs, vertexShader);
        _gl.CompileShader(vs);
        CheckShaderCompile(vs, "vertex");

        uint fs = _gl.CreateShader(ShaderType.FragmentShader);
        _gl.ShaderSource(fs, fragmentShader);
        _gl.CompileShader(fs);
        CheckShaderCompile(fs, "fragment");

        _shaderProgram = _gl.CreateProgram();
        _gl.AttachShader(_shaderProgram, vs);
        _gl.AttachShader(_shaderProgram, fs);
        _gl.LinkProgram(_shaderProgram);
        CheckProgramLink(_shaderProgram);

        _gl.DeleteShader(vs);
        _gl.DeleteShader(fs);

        _uView         = _gl.GetUniformLocation(_shaderProgram, "uView");
        _uProj         = _gl.GetUniformLocation(_shaderProgram, "uProj");
        _uTexture      = _gl.GetUniformLocation(_shaderProgram, "uTexture");
        _uCameraRight  = _gl.GetUniformLocation(_shaderProgram, "uCameraRight");
        _uCameraUp     = _gl.GetUniformLocation(_shaderProgram, "uCameraUp");
        _uParticlePos  = _gl.GetUniformLocation(_shaderProgram, "uParticlePos");
        _uParticleColor= _gl.GetUniformLocation(_shaderProgram, "uParticleColor");
        _uParticleSize = _gl.GetUniformLocation(_shaderProgram, "uParticleSize");
        _uHasTexture   = _gl.GetUniformLocation(_shaderProgram, "uHasTexture");
        _uRows         = _gl.GetUniformLocation(_shaderProgram, "uRows");
        _uColumns      = _gl.GetUniformLocation(_shaderProgram, "uColumns");
        _uCellIndex    = _gl.GetUniformLocation(_shaderProgram, "uCellIndex");
    }
    
    private void InitBuffers()
    {
        // Unit quad: 2D positions + UVs
        float[] quad = {
            // pos.x  pos.y   u    v
            -0.5f, -0.5f,  0f, 1f,  // BL
             0.5f, -0.5f,  1f, 1f,  // BR
             0.5f,  0.5f,  1f, 0f,  // TR
            -0.5f,  0.5f,  0f, 0f,  // TL
        };
        ushort[] indices = { 0, 1, 2, 0, 2, 3 };

        _vao = _gl.GenVertexArray();
        _vbo = _gl.GenBuffer();
        _ebo = _gl.GenBuffer();

        _gl.BindVertexArray(_vao);

        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, _vbo);
        fixed (float* ptr = quad)
            _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(quad.Length * sizeof(float)), ptr, BufferUsageARB.StaticDraw);

        _gl.BindBuffer(BufferTargetARB.ElementArrayBuffer, _ebo);
        fixed (ushort* ptr = indices)
            _gl.BufferData(BufferTargetARB.ElementArrayBuffer, (nuint)(indices.Length * sizeof(ushort)), ptr, BufferUsageARB.StaticDraw);

        // location 0: vec2 aQuadPos
        _gl.VertexAttribPointer(0, 2, VertexAttribPointerType.Float, false, 4 * sizeof(float), (void*)0);
        _gl.EnableVertexAttribArray(0);

        // location 1: vec2 aTexCoord
        _gl.VertexAttribPointer(1, 2, VertexAttribPointerType.Float, false, 4 * sizeof(float), (void*)(2 * sizeof(float)));
        _gl.EnableVertexAttribArray(1);

        _gl.BindVertexArray(0);
    }

    /// <summary>
    /// Render all particles from the given emitters.
    /// Call after model opaque pass, before or during transparent pass.
    /// </summary>
    public void Render(
        IReadOnlyList<ParticleEmitter> emitters,
        Matrix4x4 view, Matrix4x4 proj,
        Vector3 cameraPos,
        Dictionary<int, uint> textureMap,
        IReadOnlyList<MdlTexture> textureDefs)
    {
        // Count total live particles
        int total = 0;
        foreach (var e in emitters) total += e.Particles.Count;
        if (total == 0) return;

        // Extract camera vectors for billboarding
        Matrix4x4.Invert(view, out var invView);
        Vector3 camRight = new(invView.M11, invView.M21, invView.M31);
        Vector3 camUp    = new(invView.M12, invView.M22, invView.M32);

        _gl.UseProgram(_shaderProgram);

        // Per-frame uniforms
        _gl.UniformMatrix4(_uView, 1, false, (float*)&view);
        _gl.UniformMatrix4(_uProj, 1, false, (float*)&proj);
        _gl.Uniform3(_uCameraRight, camRight.X, camRight.Y, camRight.Z);
        _gl.Uniform3(_uCameraUp,    camUp.X,    camUp.Y,    camUp.Z);
        _gl.Uniform1(_uTexture, 0);
        _gl.ActiveTexture(TextureUnit.Texture0);

        // Render state: transparent, no depth write
        _gl.Enable(EnableCap.Blend);
        _gl.DepthMask(false);
        _gl.Enable(EnableCap.DepthTest);
        _gl.Disable(EnableCap.CullFace);

        _gl.BindVertexArray(_vao);

        foreach (var emitter in emitters)
        {
            if (!emitter.IsActive || emitter.Particles.Count == 0) continue;

            var def = emitter.Definition;

            // Set blend mode based on emitter filter mode
            switch (def.FilterMode)
            {
                case ParticleFilterMode.Additive:
                    _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.One);
                    break;
                case ParticleFilterMode.Modulate:
                case ParticleFilterMode.Modulate2x:
                    _gl.BlendFunc(BlendingFactor.DstColor, BlendingFactor.Zero);
                    break;
                case ParticleFilterMode.AlphaKey:
                    _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
                    break;
                default: // Blend
                    _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
                    break;
            }

            // Bind emitter texture
            bool hasTex = false;
            if (def.TextureId >= 0 && textureMap.TryGetValue(def.TextureId, out uint glTex))
            {
                _gl.BindTexture(TextureTarget.Texture2D, glTex);
                hasTex = true;
            }
            _gl.Uniform1(_uHasTexture, hasTex ? 1 : 0);

            // Atlas subdivisions
            int rows = Math.Max(def.Rows, 1);
            int cols = Math.Max(def.Columns, 1);
            int totalCells = rows * cols;
            _gl.Uniform1(_uRows, rows);
            _gl.Uniform1(_uColumns, cols);

            foreach (var p in emitter.Particles)
            {
                var color = emitter.GetParticleColor(p);
                float size = emitter.GetParticleSize(p);

                // Texture atlas cell based on particle lifetime phase
                int cellIdx = (int)(p.LifePhase * (totalCells - 1));
                cellIdx = Math.Clamp(cellIdx, 0, totalCells - 1);

                _gl.Uniform3(_uParticlePos,   p.Position.X, p.Position.Y, p.Position.Z);
                _gl.Uniform4(_uParticleColor, color.X, color.Y, color.Z, color.W);
                _gl.Uniform1(_uParticleSize,  size);
                _gl.Uniform1(_uCellIndex,     cellIdx);

                _gl.DrawElements(PrimitiveType.Triangles, 6, DrawElementsType.UnsignedShort, null);
            }
        }

        _gl.BindVertexArray(0);

        // Restore state
        _gl.DepthMask(true);
        _gl.Disable(EnableCap.Blend);
    }

    private void CheckShaderCompile(uint shader, string type)
    {
        _gl.GetShader(shader, ShaderParameterName.CompileStatus, out int success);
        if (success == 0)
        {
            string log = _gl.GetShaderInfoLog(shader);
            ViewerLog.Error(ViewerLog.Category.Shader, $"[ParticleRenderer] {type} shader compile failed: {log}");
        }
    }
    
    private void CheckProgramLink(uint program)
    {
        _gl.GetProgram(program, ProgramPropertyARB.LinkStatus, out int success);
        if (success == 0)
        {
            string log = _gl.GetProgramInfoLog(program);
            ViewerLog.Error(ViewerLog.Category.Shader, $"[ParticleRenderer] Shader link failed: {log}");
        }
    }
    
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _gl.DeleteVertexArray(_vao);
        _gl.DeleteBuffer(_vbo);
        _gl.DeleteBuffer(_ebo);
        _gl.DeleteProgram(_shaderProgram);
    }
}

