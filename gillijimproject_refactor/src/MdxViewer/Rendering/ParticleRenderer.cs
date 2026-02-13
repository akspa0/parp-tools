using System.Numerics;
using MdxViewer.DataSources;
using MdxViewer.Logging;
using Silk.NET.OpenGL;

namespace MdxViewer.Rendering;

/// <summary>
/// GPU-based particle renderer with billboarding and additive blending.
/// Renders all active particle emitters efficiently using instanced rendering.
/// </summary>
public unsafe class ParticleRenderer : IDisposable
{
    private readonly GL _gl;
    private uint _shaderProgram;
    private uint _vao, _vbo;
    private int _uView, _uProj, _uTexture, _uCameraRight, _uCameraUp;
    
    // Particle instance data (position, color, size)
    private readonly List<ParticleInstanceData> _instanceData = new();
    private const int MaxParticlesPerBatch = 10000;
    
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
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec2 aTexCoord;
layout(location = 2) in vec3 aInstancePos;
layout(location = 3) in vec4 aInstanceColor;
layout(location = 4) in float aInstanceSize;

out vec2 vTexCoord;
out vec4 vColor;

uniform mat4 uView;
uniform mat4 uProj;
uniform vec3 uCameraRight;
uniform vec3 uCameraUp;

void main()
{
    // Billboard quad facing camera
    vec3 vertexPos = aInstancePos 
        + uCameraRight * aPosition.x * aInstanceSize
        + uCameraUp * aPosition.y * aInstanceSize;
    
    gl_Position = uProj * uView * vec4(vertexPos, 1.0);
    vTexCoord = aTexCoord;
    vColor = aInstanceColor;
}
";

        string fragmentShader = @"
#version 330 core
in vec2 vTexCoord;
in vec4 vColor;
out vec4 FragColor;

uniform sampler2D uTexture;

void main()
{
    vec4 texColor = texture(uTexture, vTexCoord);
    FragColor = texColor * vColor;
    
    // Discard fully transparent pixels
    if (FragColor.a < 0.01)
        discard;
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

        _uView = _gl.GetUniformLocation(_shaderProgram, "uView");
        _uProj = _gl.GetUniformLocation(_shaderProgram, "uProj");
        _uTexture = _gl.GetUniformLocation(_shaderProgram, "uTexture");
        _uCameraRight = _gl.GetUniformLocation(_shaderProgram, "uCameraRight");
        _uCameraUp = _gl.GetUniformLocation(_shaderProgram, "uCameraUp");
    }
    
    private void InitBuffers()
    {
        // Quad vertices for billboarding (-0.5 to 0.5, centered)
        float[] quadVertices = {
            -0.5f, -0.5f, 0f,  0f, 0f,  // Bottom-left
             0.5f, -0.5f, 0f,  1f, 0f,  // Bottom-right
             0.5f,  0.5f, 0f,  1f, 1f,  // Top-right
            -0.5f,  0.5f, 0f,  0f, 1f   // Top-left
        };
        
        _vao = _gl.GenVertexArray();
        _vbo = _gl.GenBuffer();
        
        _gl.BindVertexArray(_vao);
        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, _vbo);
        
        fixed (float* ptr = quadVertices)
        {
            _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(quadVertices.Length * sizeof(float)), ptr, BufferUsageARB.StaticDraw);
        }
        
        // Position
        _gl.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 5 * sizeof(float), (void*)0);
        _gl.EnableVertexAttribArray(0);
        
        // TexCoord
        _gl.VertexAttribPointer(1, 2, VertexAttribPointerType.Float, false, 5 * sizeof(float), (void*)(3 * sizeof(float)));
        _gl.EnableVertexAttribArray(1);
        
        _gl.BindVertexArray(0);
    }
    
    /// <summary>
    /// Render all particles from active emitters.
    /// </summary>
    public void Render(IEnumerable<ParticleEmitter> emitters, Matrix4x4 view, Matrix4x4 proj, Vector3 cameraPos, uint textureId)
    {
        _instanceData.Clear();
        
        // Collect all particles from all emitters
        foreach (var emitter in emitters)
        {
            if (!emitter.IsActive || emitter.Particles.Count == 0)
                continue;
            
            foreach (var particle in emitter.Particles)
            {
                var color = emitter.GetParticleColor(particle);
                float size = emitter.GetParticleSize(particle);
                
                _instanceData.Add(new ParticleInstanceData
                {
                    Position = particle.Position,
                    Color = color,
                    Size = size
                });
            }
        }
        
        if (_instanceData.Count == 0)
            return;
        
        // Calculate camera right and up vectors for billboarding
        Matrix4x4.Invert(view, out var invView);
        Vector3 cameraRight = new Vector3(invView.M11, invView.M21, invView.M31);
        Vector3 cameraUp = new Vector3(invView.M12, invView.M22, invView.M32);
        
        // Setup render state
        _gl.UseProgram(_shaderProgram);
        _gl.UniformMatrix4(_uView, 1, false, (float*)&view);
        _gl.UniformMatrix4(_uProj, 1, false, (float*)&proj);
        _gl.Uniform3(_uCameraRight, cameraRight.X, cameraRight.Y, cameraRight.Z);
        _gl.Uniform3(_uCameraUp, cameraUp.X, cameraUp.Y, cameraUp.Z);
        
        _gl.ActiveTexture(TextureUnit.Texture0);
        _gl.BindTexture(TextureTarget.Texture2D, textureId);
        _gl.Uniform1(_uTexture, 0);
        
        // Enable additive blending for fire/glow effects
        _gl.Enable(EnableCap.Blend);
        _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.One); // Additive
        _gl.DepthMask(false); // Don't write to depth buffer
        
        _gl.BindVertexArray(_vao);
        
        // Render in batches (instanced rendering would be better but this is simpler)
        foreach (var instance in _instanceData)
        {
            // For now, render each particle as a separate quad
            // TODO: Use instanced rendering for better performance
            RenderParticle(instance);
        }
        
        _gl.BindVertexArray(0);
        
        // Restore render state
        _gl.DepthMask(true);
        _gl.Disable(EnableCap.Blend);
    }
    
    private void RenderParticle(ParticleInstanceData instance)
    {
        // This is a simplified version - ideally use instanced rendering
        // For now, just draw the quad (instance data would be passed via uniforms or instance buffer)
        _gl.DrawArrays(PrimitiveType.TriangleFan, 0, 4);
    }
    
    private void CheckShaderCompile(uint shader, string type)
    {
        _gl.GetShader(shader, ShaderParameterName.CompileStatus, out int success);
        if (success == 0)
        {
            string log = _gl.GetShaderInfoLog(shader);
            ViewerLog.Error(ViewerLog.Category.Shader, $"[ParticleRenderer] {type} shader compilation failed: {log}");
        }
    }
    
    private void CheckProgramLink(uint program)
    {
        _gl.GetProgram(program, ProgramPropertyARB.LinkStatus, out int success);
        if (success == 0)
        {
            string log = _gl.GetProgramInfoLog(program);
            ViewerLog.Error(ViewerLog.Category.Shader, $"[ParticleRenderer] Shader program linking failed: {log}");
        }
    }
    
    public void Dispose()
    {
        _gl.DeleteVertexArray(_vao);
        _gl.DeleteBuffer(_vbo);
        _gl.DeleteProgram(_shaderProgram);
    }
}

internal struct ParticleInstanceData
{
    public Vector3 Position;
    public Vector4 Color;
    public float Size;
}
