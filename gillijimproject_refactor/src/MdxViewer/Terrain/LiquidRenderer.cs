using System.Numerics;
using MdxViewer.Rendering;
using Silk.NET.OpenGL;

namespace MdxViewer.Terrain;

/// <summary>
/// Renders MCLQ liquid surfaces as semi-transparent planes over terrain.
/// Each liquid chunk is a 9×9 vertex grid (8×8 quads) with per-tile visibility flags.
/// Rendered as a separate pass after terrain with alpha blending.
/// </summary>
public class LiquidRenderer : IDisposable
{
    private readonly GL _gl;
    private readonly ShaderProgram _shader;

    // Per-chunk liquid meshes
    private readonly List<LiquidMesh> _meshes = new();

    // Global toggle
    public bool ShowLiquid { get; set; } = true;

    // Animation time
    private float _time;

    public int MeshCount => _meshes.Count;

    public LiquidRenderer(GL gl)
    {
        _gl = gl;
        _shader = CreateLiquidShader();
    }

    /// <summary>
    /// Build and upload liquid meshes from terrain chunk data.
    /// Call on the render thread after terrain chunks are loaded.
    /// </summary>
    public void AddChunks(IEnumerable<TerrainChunkData> chunks)
    {
        int added = 0;
        foreach (var chunk in chunks)
        {
            if (chunk.Liquid == null) continue;
            var mesh = BuildLiquidMesh(chunk.Liquid);
            if (mesh != null)
            {
                _meshes.Add(mesh);
                added++;
            }
        }
        if (added > 0)
            Console.WriteLine($"[LiquidRenderer] Added {added} liquid meshes (total: {_meshes.Count})");
    }

    /// <summary>
    /// Remove liquid meshes associated with unloaded tiles.
    /// </summary>
    public void RemoveChunksForTile(int tileX, int tileY)
    {
        var toRemove = _meshes.Where(m => m.TileX == tileX && m.TileY == tileY).ToList();
        foreach (var mesh in toRemove)
        {
            mesh.Dispose(_gl);
            _meshes.Remove(mesh);
        }
    }

    /// <summary>
    /// Render all liquid surfaces. Call after terrain rendering.
    /// </summary>
    public unsafe void Render(Matrix4x4 view, Matrix4x4 proj, Vector3 cameraPos, TerrainLighting lighting, float deltaTime)
    {
        if (!ShowLiquid || _meshes.Count == 0) return;

        _time += deltaTime;

        _shader.Use();
        _shader.SetMat4("uView", view);
        _shader.SetMat4("uProj", proj);
        _shader.SetMat4("uModel", Matrix4x4.Identity);
        _shader.SetVec3("uCameraPos", cameraPos);
        _shader.SetFloat("uTime", _time);
        _shader.SetVec3("uLightDir", lighting.LightDirection);
        _shader.SetVec3("uLightColor", lighting.LightColor);
        _shader.SetVec3("uAmbientColor", lighting.AmbientColor);
        _shader.SetVec3("uFogColor", lighting.FogColor);
        _shader.SetFloat("uFogStart", lighting.FogStart);
        _shader.SetFloat("uFogEnd", lighting.FogEnd);

        // Alpha blending for transparent water
        _gl.Enable(EnableCap.Blend);
        _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
        _gl.DepthMask(false); // Don't write to depth buffer
        _gl.Disable(EnableCap.CullFace);

        foreach (var mesh in _meshes)
        {
            // Set liquid type color
            var (r, g, b, a) = GetLiquidColor(mesh.Type);
            _shader.SetVec4("uLiquidColor", new Vector4(r, g, b, a));

            _gl.BindVertexArray(mesh.Vao);
            _gl.DrawElements(PrimitiveType.Triangles, mesh.IndexCount, DrawElementsType.UnsignedShort, null);
        }

        _gl.BindVertexArray(0);
        _gl.DepthMask(true);
        _gl.Disable(EnableCap.Blend);
        _gl.Enable(EnableCap.CullFace);
    }

    private static (float r, float g, float b, float a) GetLiquidColor(LiquidType type)
    {
        return type switch
        {
            LiquidType.Water => (0.15f, 0.35f, 0.65f, 0.55f),
            LiquidType.Ocean => (0.10f, 0.25f, 0.55f, 0.60f),
            LiquidType.Magma => (0.85f, 0.30f, 0.05f, 0.75f),
            LiquidType.Slime => (0.20f, 0.70f, 0.15f, 0.65f),
            _ => (0.15f, 0.35f, 0.65f, 0.55f)
        };
    }

    /// <summary>
    /// Build a GPU mesh for a single liquid chunk.
    /// 9×9 vertices with 8×8 quads. Alpha 0.5.3: all quads rendered (no per-tile flags).
    /// </summary>
    private unsafe LiquidMesh? BuildLiquidMesh(LiquidChunkData liquid)
    {
        if (liquid.Heights.Length < 81)
            return null;

        float chunkSize = WoWConstants.ChunkSize / 16f; // Size of one MCNK chunk in world units
        float cellSize = chunkSize / 8f; // Size of one liquid cell

        // Build vertex data: position(3) = 3 floats per vertex, 81 vertices
        var vertices = new float[81 * 3];
        for (int vy = 0; vy < 9; vy++)
        {
            for (int vx = 0; vx < 9; vx++)
            {
                int idx = vy * 9 + vx;
                float height = liquid.Heights[idx];

                // World position: same coordinate system as terrain
                // Chunk corner is at WorldPosition, vertices extend -X and -Y
                float worldX = liquid.WorldPosition.X - vy * cellSize;
                float worldY = liquid.WorldPosition.Y - vx * cellSize;

                // Alpha 0.5.3: MCLQ heights use inverted Z convention.
                // Negate to match renderer Z-up coordinate system.
                vertices[idx * 3 + 0] = worldX;
                vertices[idx * 3 + 1] = worldY;
                vertices[idx * 3 + 2] = -height;
            }
        }

        // Build index buffer: 8×8 quads, each = 2 triangles
        // Alpha 0.5.3: No per-tile visibility flags (tiles are 4×4 floats, not 8×8 byte flags).
        // Render all 64 quads; liquid presence is determined by MCNK header flags.
        var indices = new List<ushort>(64 * 6);
        for (int ty = 0; ty < 8; ty++)
        {
            for (int tx = 0; tx < 8; tx++)
            {
                // Quad vertices:
                // TL = (ty, tx), TR = (ty, tx+1)
                // BL = (ty+1, tx), BR = (ty+1, tx+1)
                ushort tl = (ushort)(ty * 9 + tx);
                ushort tr = (ushort)(ty * 9 + tx + 1);
                ushort bl = (ushort)((ty + 1) * 9 + tx);
                ushort br = (ushort)((ty + 1) * 9 + tx + 1);

                // Two triangles per quad
                indices.Add(tl); indices.Add(bl); indices.Add(tr);
                indices.Add(tr); indices.Add(bl); indices.Add(br);
            }
        }

        if (indices.Count == 0) return null;

        // Upload to GPU
        uint vao = _gl.GenVertexArray();
        uint vbo = _gl.GenBuffer();
        uint ebo = _gl.GenBuffer();

        _gl.BindVertexArray(vao);

        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, vbo);
        fixed (float* ptr = vertices)
            _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(vertices.Length * sizeof(float)), ptr, BufferUsageARB.StaticDraw);

        _gl.BindBuffer(BufferTargetARB.ElementArrayBuffer, ebo);
        var indexArray = indices.ToArray();
        fixed (ushort* ptr = indexArray)
            _gl.BufferData(BufferTargetARB.ElementArrayBuffer, (nuint)(indexArray.Length * sizeof(ushort)), ptr, BufferUsageARB.StaticDraw);

        // Position attribute: location 0, 3 floats, stride = 12
        _gl.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 3 * sizeof(float), (void*)0);
        _gl.EnableVertexAttribArray(0);

        _gl.BindVertexArray(0);

        return new LiquidMesh
        {
            Vao = vao,
            Vbo = vbo,
            Ebo = ebo,
            IndexCount = (uint)indexArray.Length,
            Type = liquid.Type,
            TileX = liquid.TileX,
            TileY = liquid.TileY
        };
    }

    private ShaderProgram CreateLiquidShader()
    {
        string vertSrc = @"
#version 330 core
layout(location = 0) in vec3 aPos;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;
uniform float uTime;

out vec3 vWorldPos;

void main() {
    vec3 pos = aPos;
    // Gentle wave animation
    float wave = sin(pos.x * 0.3 + uTime * 1.5) * 0.15
               + cos(pos.y * 0.25 + uTime * 1.2) * 0.10;
    pos.z += wave;

    vec4 worldPos = uModel * vec4(pos, 1.0);
    vWorldPos = worldPos.xyz;
    gl_Position = uProj * uView * worldPos;
}
";

        string fragSrc = @"
#version 330 core
in vec3 vWorldPos;

uniform vec4 uLiquidColor;
uniform vec3 uCameraPos;
uniform vec3 uLightDir;
uniform vec3 uLightColor;
uniform vec3 uAmbientColor;
uniform vec3 uFogColor;
uniform float uFogStart;
uniform float uFogEnd;
uniform float uTime;

out vec4 FragColor;

void main() {
    // Base liquid color
    vec3 baseColor = uLiquidColor.rgb;
    float baseAlpha = uLiquidColor.a;

    // Simple specular highlight on water surface
    vec3 viewDir = normalize(uCameraPos - vWorldPos);
    vec3 normal = vec3(0.0, 0.0, 1.0); // Flat water surface normal
    vec3 lightDir = normalize(uLightDir);
    vec3 halfDir = normalize(viewDir + lightDir);
    float spec = pow(max(dot(normal, halfDir), 0.0), 32.0);

    // Lighting
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 lighting = uAmbientColor + uLightColor * diff;
    vec3 litColor = baseColor * lighting + uLightColor * spec * 0.3;

    // Animated caustic pattern (subtle)
    float caustic = sin(vWorldPos.x * 1.2 + uTime * 2.0)
                  * cos(vWorldPos.y * 1.1 + uTime * 1.8) * 0.5 + 0.5;
    litColor += vec3(caustic * 0.05);

    // Fog
    float dist = length(vWorldPos - uCameraPos);
    float fogFactor = clamp((uFogEnd - dist) / (uFogEnd - uFogStart), 0.0, 1.0);
    vec3 finalColor = mix(uFogColor, litColor, fogFactor);

    // Fresnel-like edge darkening: more opaque at glancing angles
    float fresnel = 1.0 - max(dot(viewDir, normal), 0.0);
    float alpha = mix(baseAlpha * 0.6, baseAlpha, fresnel);

    FragColor = vec4(finalColor, alpha);
}
";

        return ShaderProgram.Create(_gl, vertSrc, fragSrc);
    }

    public void Dispose()
    {
        foreach (var mesh in _meshes)
            mesh.Dispose(_gl);
        _meshes.Clear();
        _shader.Dispose();
    }
}

/// <summary>
/// GPU-resident mesh for a single liquid chunk surface.
/// </summary>
internal class LiquidMesh
{
    public uint Vao { get; init; }
    public uint Vbo { get; init; }
    public uint Ebo { get; init; }
    public uint IndexCount { get; init; }
    public LiquidType Type { get; init; }
    public int TileX { get; init; }
    public int TileY { get; init; }

    public void Dispose(GL gl)
    {
        gl.DeleteVertexArray(Vao);
        gl.DeleteBuffer(Vbo);
        gl.DeleteBuffer(Ebo);
    }
}
