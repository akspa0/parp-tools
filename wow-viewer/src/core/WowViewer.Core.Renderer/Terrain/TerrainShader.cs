using System.Numerics;
using Silk.NET.OpenGL;

namespace WowViewer.Core.Renderer.Terrain;

public sealed class TerrainShader : IDisposable
{
    private readonly GL _gl;
    private readonly uint _program;

    // Uniform locations
    private readonly int _uModel;
    private readonly int _uView;
    private readonly int _uProj;
    private readonly int _uDiffuseArray;
    private readonly int _uAlphaShadowArray;
    private readonly int _uDiffuseLayerCount;
    private readonly int _uUseWorldUV;
    private readonly int _uUseMccv;
    private readonly int _uLightDir;
    private readonly int _uLightColor;
    private readonly int _uAmbientColor;
    private readonly int _uFogColor;
    private readonly int _uFogStart;
    private readonly int _uFogEnd;
    private readonly int _uCameraPos;
    private readonly int _uShowLayer0;
    private readonly int _uShowLayer1;
    private readonly int _uShowLayer2;
    private readonly int _uShowLayer3;
    private readonly int _uShowShadowMap;
    private readonly int _uShadowStrength;

    private const string VertexSource = @"
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;
layout(location = 3) in uint aChunkSlice;
layout(location = 4) in uvec4 aTexIdx;
layout(location = 5) in vec4 aVertexColor;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;
uniform vec3 uLightDir;

out vec3 vWorldPos;
out float vDiffuse;
out vec2 vTexCoord;
out vec4 vVertexColor;
flat out uint vChunkSlice;
flat out uvec4 vTexIdx;

void main() {
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    vWorldPos = worldPos.xyz;
    vec3 worldNormal = normalize(mat3(uModel) * aNormal);
    vDiffuse = max(dot(worldNormal, normalize(uLightDir)), 0.0);
    vTexCoord = aTexCoord;
    vVertexColor = aVertexColor;
    vChunkSlice = aChunkSlice;
    vTexIdx = aTexIdx;
    gl_Position = uProj * uView * worldPos;
}
";

    private const string FragmentSource = @"
#version 330 core
in vec3 vWorldPos;
in float vDiffuse;
in vec2 vTexCoord;
in vec4 vVertexColor;
flat in uint vChunkSlice;
flat in uvec4 vTexIdx;

uniform sampler2DArray uDiffuseArray;
uniform sampler2DArray uAlphaShadowArray;
uniform int uDiffuseLayerCount;
uniform int uUseWorldUV;
uniform int uUseMccv;
uniform int uShowLayer0;
uniform int uShowLayer1;
uniform int uShowLayer2;
uniform int uShowLayer3;
uniform int uShowShadowMap;
uniform float uShadowStrength;
uniform vec3 uLightColor;
uniform vec3 uAmbientColor;
uniform vec3 uFogColor;
uniform float uFogStart;
uniform float uFogEnd;
uniform vec3 uCameraPos;

out vec4 FragColor;

bool HasLayer(uint idx) {
    return (idx != 65535u) && (int(idx) >= 0) && (int(idx) < uDiffuseLayerCount);
}

void main() {
    float texScale = 8.0 / 33.333;
    vec2 diffuseUV = (uUseWorldUV == 1) ? (vec2(-vWorldPos.y, -vWorldPos.x) * texScale) : (vTexCoord * 8.0);

    vec4 alphaShadow = texture(uAlphaShadowArray, vec3(vTexCoord, float(vChunkSlice)));
    bool has0 = HasLayer(vTexIdx.x);
    bool has1 = HasLayer(vTexIdx.y);
    bool has2 = HasLayer(vTexIdx.z);
    bool has3 = HasLayer(vTexIdx.w);

    float a1 = (uShowLayer1 == 1 && has1) ? alphaShadow.r : 0.0;
    float a2 = (uShowLayer2 == 1 && has2) ? alphaShadow.g : 0.0;
    float a3 = (uShowLayer3 == 1 && has3) ? alphaShadow.b : 0.0;

    float shadowVisibility = (uShowShadowMap == 1)
        ? 1.0 - clamp(alphaShadow.a, 0.0, 1.0) * clamp(uShadowStrength, 0.0, 1.0)
        : 1.0;
    vec3 lighting = uAmbientColor + uLightColor * vDiffuse * shadowVisibility;
    vec3 result = vec3(1.0);

    if (uShowLayer0 == 1 && has0) {
        vec4 c0 = texture(uDiffuseArray, vec3(diffuseUV, float(vTexIdx.x)));
        result = c0.rgb * lighting;
    }
    if (uShowLayer1 == 1 && has1) {
        vec4 c1 = texture(uDiffuseArray, vec3(diffuseUV, float(vTexIdx.y)));
        result = mix(result, c1.rgb * lighting, a1);
    }
    if (uShowLayer2 == 1 && has2) {
        vec4 c2 = texture(uDiffuseArray, vec3(diffuseUV, float(vTexIdx.z)));
        result = mix(result, c2.rgb * lighting, a2);
    }
    if (uShowLayer3 == 1 && has3) {
        vec4 c3 = texture(uDiffuseArray, vec3(diffuseUV, float(vTexIdx.w)));
        result = mix(result, c3.rgb * lighting, a3);
    }

    vec3 tintColor = clamp(vVertexColor.rgb * 2.0, 0.0, 2.0);
    float tintStrength = clamp(vVertexColor.a * 2.0 - 1.0, 0.0, 1.0);
    vec3 vertexTint = (uUseMccv == 1) ? mix(vec3(1.0), tintColor, tintStrength) : vec3(1.0);
    result *= vertexTint;

    float dist = length(vWorldPos - uCameraPos);
    float fogFactor = clamp((uFogEnd - dist) / (uFogEnd - uFogStart), 0.0, 1.0);
    vec3 finalColor = mix(uFogColor, result, fogFactor);

    FragColor = vec4(finalColor, 1.0);
}
";

    public TerrainShader(GL gl)
    {
        _gl = gl ?? throw new ArgumentNullException(nameof(gl));
        _program = CreateProgram();

        _uModel = gl.GetUniformLocation(_program, "uModel");
        _uView = gl.GetUniformLocation(_program, "uView");
        _uProj = gl.GetUniformLocation(_program, "uProj");
        _uDiffuseArray = gl.GetUniformLocation(_program, "uDiffuseArray");
        _uAlphaShadowArray = gl.GetUniformLocation(_program, "uAlphaShadowArray");
        _uDiffuseLayerCount = gl.GetUniformLocation(_program, "uDiffuseLayerCount");
        _uUseWorldUV = gl.GetUniformLocation(_program, "uUseWorldUV");
        _uUseMccv = gl.GetUniformLocation(_program, "uUseMccv");
        _uLightDir = gl.GetUniformLocation(_program, "uLightDir");
        _uLightColor = gl.GetUniformLocation(_program, "uLightColor");
        _uAmbientColor = gl.GetUniformLocation(_program, "uAmbientColor");
        _uFogColor = gl.GetUniformLocation(_program, "uFogColor");
        _uFogStart = gl.GetUniformLocation(_program, "uFogStart");
        _uFogEnd = gl.GetUniformLocation(_program, "uFogEnd");
        _uCameraPos = gl.GetUniformLocation(_program, "uCameraPos");
        _uShowLayer0 = gl.GetUniformLocation(_program, "uShowLayer0");
        _uShowLayer1 = gl.GetUniformLocation(_program, "uShowLayer1");
        _uShowLayer2 = gl.GetUniformLocation(_program, "uShowLayer2");
        _uShowLayer3 = gl.GetUniformLocation(_program, "uShowLayer3");
        _uShowShadowMap = gl.GetUniformLocation(_program, "uShowShadowMap");
        _uShadowStrength = gl.GetUniformLocation(_program, "uShadowStrength");

        Use();
        gl.Uniform1(_uDiffuseArray, 0);
        gl.Uniform1(_uAlphaShadowArray, 1);
    }

    public void Use()
    {
        _gl.UseProgram(_program);
    }

    // Matrix4x4 is row-major; GL expects column-major — transpose=true handles the conversion
    public unsafe void SetModel(Matrix4x4 mat) => _gl.UniformMatrix4(_uModel, 1, true, (float*)&mat);
    public unsafe void SetView(Matrix4x4 mat) => _gl.UniformMatrix4(_uView, 1, true, (float*)&mat);
    public unsafe void SetProjection(Matrix4x4 mat) => _gl.UniformMatrix4(_uProj, 1, true, (float*)&mat);
    public void SetDiffuseLayerCount(int count) => _gl.Uniform1(_uDiffuseLayerCount, count);
    public void SetUseWorldUV(bool enabled) => _gl.Uniform1(_uUseWorldUV, enabled ? 1 : 0);
    public void SetUseMccv(bool enabled) => _gl.Uniform1(_uUseMccv, enabled ? 1 : 0);
    public void SetLightDirection(Vector3 dir) => _gl.Uniform3(_uLightDir, dir.X, dir.Y, dir.Z);
    public void SetLightColor(Vector3 color) => _gl.Uniform3(_uLightColor, color.X, color.Y, color.Z);
    public void SetAmbientColor(Vector3 color) => _gl.Uniform3(_uAmbientColor, color.X, color.Y, color.Z);
    public void SetFogColor(Vector3 color) => _gl.Uniform3(_uFogColor, color.X, color.Y, color.Z);
    public void SetFogStart(float start) => _gl.Uniform1(_uFogStart, start);
    public void SetFogEnd(float end) => _gl.Uniform1(_uFogEnd, end);
    public void SetCameraPosition(Vector3 pos) => _gl.Uniform3(_uCameraPos, pos.X, pos.Y, pos.Z);
    public void SetShowLayer(int layer, bool visible)
    {
        int loc = layer switch
        {
            0 => _uShowLayer0,
            1 => _uShowLayer1,
            2 => _uShowLayer2,
            3 => _uShowLayer3,
            _ => -1
        };
        if (loc >= 0) _gl.Uniform1(loc, visible ? 1 : 0);
    }
    public void SetShowShadowMap(bool enabled) => _gl.Uniform1(_uShowShadowMap, enabled ? 1 : 0);
    public void SetShadowStrength(float strength) => _gl.Uniform1(_uShadowStrength, Math.Clamp(strength, 0f, 1f));

    private uint CreateProgram()
    {
        uint vert = CompileShader(ShaderType.VertexShader, VertexSource);
        uint frag = CompileShader(ShaderType.FragmentShader, FragmentSource);

        uint program = _gl.CreateProgram();
        _gl.AttachShader(program, vert);
        _gl.AttachShader(program, frag);
        _gl.LinkProgram(program);

        _gl.GetProgram(program, GLEnum.LinkStatus, out int success);
        if (success == 0)
        {
            string info = _gl.GetProgramInfoLog(program);
            _gl.DeleteProgram(program);
            throw new InvalidOperationException($"Shader program link failed: {info}");
        }

        _gl.DetachShader(program, vert);
        _gl.DetachShader(program, frag);
        _gl.DeleteShader(vert);
        _gl.DeleteShader(frag);

        return program;
    }

    private uint CompileShader(ShaderType type, string source)
    {
        uint shader = _gl.CreateShader(type);
        _gl.ShaderSource(shader, source);
        _gl.CompileShader(shader);

        _gl.GetShader(shader, ShaderParameterName.CompileStatus, out int success);
        if (success == 0)
        {
            string info = _gl.GetShaderInfoLog(shader);
            _gl.DeleteShader(shader);
            throw new InvalidOperationException($"Shader compile failed ({type}): {info}");
        }

        return shader;
    }

    public void Dispose()
    {
        if (_program != 0)
            _gl.DeleteProgram(_program);
    }
}
