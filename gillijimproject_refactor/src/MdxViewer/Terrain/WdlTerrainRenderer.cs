using System.Numerics;
using MdxViewer.DataSources;
using MdxViewer.Logging;
using MdxViewer.Rendering;
using Silk.NET.OpenGL;
using WoWMapConverter.Core.VLM;

namespace MdxViewer.Terrain;

/// <summary>
/// Renders a low-resolution 3D terrain mesh from WDL (World Detail Level) data.
/// Each WDL tile has a 17×17 outer + 16×16 inner height grid — same layout as MCNK
/// but at tile scale (each WDL "cell" = one ADT chunk = 533.33 world units).
/// 
/// Used as background/far terrain: rendered for all tiles at startup, then individual
/// tiles are hidden as high-detail ADT meshes stream in via TerrainManager.
/// </summary>
public class WdlTerrainRenderer : IDisposable
{
    private readonly GL _gl;
    private readonly ShaderProgram _shader;

    // Per-tile GPU mesh data
    private readonly Dictionary<int, WdlTileMesh> _tileMeshes = new(); // tileIndex → mesh
    private readonly HashSet<int> _hiddenTiles = new(); // tiles hidden by loaded ADTs

    // Stats
    public int TotalTiles => _tileMeshes.Count;
    public int VisibleTiles => _tileMeshes.Count - _hiddenTiles.Count;
    public int HiddenTiles => _hiddenTiles.Count;

    public WdlTerrainRenderer(GL gl)
    {
        _gl = gl;
        _shader = CreateShader();
    }

    /// <summary>
    /// Load WDL data and build 3D meshes for all tiles that have height data.
    /// </summary>
    public bool Load(IDataSource dataSource, string mapDirectory)
    {
        string wdlPath = $"World\\Maps\\{mapDirectory}\\{mapDirectory}.wdl";
        byte[]? wdlBytes = dataSource.ReadFile(wdlPath);

        // Alpha 0.5.3: WDL stored as .wdl.mpq
        if (wdlBytes == null || wdlBytes.Length == 0)
        {
            wdlPath += ".mpq";
            wdlBytes = dataSource.ReadFile(wdlPath);
        }

        if (wdlBytes == null || wdlBytes.Length == 0)
        {
            ViewerLog.Info(ViewerLog.Category.Terrain, $"[WDL 3D] No WDL data for {mapDirectory}");
            return false;
        }

        var wdlData = WdlParser.Parse(wdlBytes);
        if (wdlData == null)
        {
            ViewerLog.Error(ViewerLog.Category.Terrain, $"[WDL 3D] Failed to parse WDL for {mapDirectory}");
            return false;
        }

        int built = 0;
        for (int tileY = 0; tileY < 64; tileY++)
        {
            for (int tileX = 0; tileX < 64; tileX++)
            {
                int idx = tileY * 64 + tileX;
                var tile = wdlData.Tiles[idx];
                if (tile?.HasData != true) continue;

                var mesh = BuildTileMesh(tile, tileX, tileY);
                if (mesh != null)
                {
                    _tileMeshes[idx] = mesh;
                    built++;
                }
            }
        }

        ViewerLog.Important(ViewerLog.Category.Terrain, $"[WDL 3D] Built {built} low-res terrain tiles for {mapDirectory}");
        return built > 0;
    }

    /// <summary>
    /// Hide a WDL tile (called when the corresponding ADT is loaded at full detail).
    /// </summary>
    public void HideTile(int tileX, int tileY)
    {
        _hiddenTiles.Add(tileY * 64 + tileX);
    }

    /// <summary>
    /// Show a WDL tile again (called when the corresponding ADT is unloaded).
    /// </summary>
    public void ShowTile(int tileX, int tileY)
    {
        _hiddenTiles.Remove(tileY * 64 + tileX);
    }

    /// <summary>
    /// Render all visible WDL tiles.
    /// </summary>
    public unsafe void Render(Matrix4x4 view, Matrix4x4 proj, Vector3 cameraPos,
        TerrainLighting lighting, FrustumCuller? frustum = null)
    {
        if (_tileMeshes.Count == 0) return;

        _shader.Use();
        _shader.SetMat4("uView", view);
        _shader.SetMat4("uProj", proj);
        _shader.SetVec3("uLightDir", lighting.LightDirection);
        _shader.SetVec3("uLightColor", lighting.LightColor);
        _shader.SetVec3("uAmbientColor", lighting.AmbientColor);
        _shader.SetVec3("uFogColor", lighting.FogColor);
        _shader.SetFloat("uFogStart", lighting.FogStart);
        _shader.SetFloat("uFogEnd", lighting.FogEnd);
        _shader.SetVec3("uCameraPos", cameraPos);

        _gl.Disable(EnableCap.CullFace);
        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthMask(true);
        _gl.PolygonMode(TriangleFace.FrontAndBack, PolygonMode.Fill);

        // Push WDL slightly behind real terrain to prevent z-fighting at tile edges
        _gl.Enable(EnableCap.PolygonOffsetFill);
        _gl.PolygonOffset(1.0f, 1.0f);

        foreach (var (idx, mesh) in _tileMeshes)
        {
            if (_hiddenTiles.Contains(idx)) continue;

            // Frustum cull
            if (frustum != null && !frustum.TestAABB(mesh.BoundsMin, mesh.BoundsMax))
                continue;

            _gl.BindVertexArray(mesh.Vao);
            _gl.DrawElements(PrimitiveType.Triangles, mesh.IndexCount, DrawElementsType.UnsignedInt, null);
        }

        _gl.Disable(EnableCap.PolygonOffsetFill);
        _gl.BindVertexArray(0);
    }

    // ── Mesh building ────────────────────────────────────────────────────

    private unsafe WdlTileMesh? BuildTileMesh(WdlParser.WdlTile tile, int tileX, int tileY)
    {
        // WDL uses same 17×17 outer + 16×16 inner layout as MCNK but at ADT tile scale.
        // 17×17 = 289 outer vertices, 16×16 = 256 inner vertices = 545 total.
        // Each WDL cell = one ADT tile = 16 chunks × 533.33 = 8533.33 world units per edge.
        const int outerEdge = 17;
        const int innerEdge = 16;
        int totalVerts = outerEdge * outerEdge + innerEdge * innerEdge; // 545

        // Tile world origin (top-left corner in renderer space)
        // Each WDL cell spans one full ADT tile = TileSize world units
        float tileWorldX = WoWConstants.MapOrigin - tileX * WoWConstants.TileSize;
        float tileWorldY = WoWConstants.MapOrigin - tileY * WoWConstants.TileSize;
        float tileSize = WoWConstants.TileSize; // One WDL cell = one ADT tile
        float cellSize = tileSize / innerEdge; // Distance between outer vertices

        // Vertex data: position(3) + normal(3) = 6 floats per vertex
        float[] vertices = new float[totalVerts * 6];
        var boundsMin = new Vector3(float.MaxValue);
        var boundsMax = new Vector3(float.MinValue);

        // Outer vertices (17×17) — indices 0..288
        for (int r = 0; r < outerEdge; r++)
        {
            for (int c = 0; c < outerEdge; c++)
            {
                int vi = r * outerEdge + c;
                float x = tileWorldX - r * cellSize;
                float y = tileWorldY - c * cellSize;
                float z = tile.Heights[vi]; // Use flat Heights array

                int offset = vi * 6;
                vertices[offset + 0] = x;
                vertices[offset + 1] = y;
                vertices[offset + 2] = z;
                // Normal computed later
                vertices[offset + 3] = 0;
                vertices[offset + 4] = 0;
                vertices[offset + 5] = 1;

                boundsMin = Vector3.Min(boundsMin, new Vector3(x, y, z));
                boundsMax = Vector3.Max(boundsMax, new Vector3(x, y, z));
            }
        }

        // Inner vertices (16×16) — indices 289..544
        int innerBase = outerEdge * outerEdge;
        for (int r = 0; r < innerEdge; r++)
        {
            for (int c = 0; c < innerEdge; c++)
            {
                int vi = innerBase + r * innerEdge + c;
                float x = tileWorldX - (r + 0.5f) * cellSize;
                float y = tileWorldY - (c + 0.5f) * cellSize;
                float z = tile.Heights[vi]; // Use flat Heights array

                int offset = vi * 6;
                vertices[offset + 0] = x;
                vertices[offset + 1] = y;
                vertices[offset + 2] = z;
                vertices[offset + 3] = 0;
                vertices[offset + 4] = 0;
                vertices[offset + 5] = 1;

                boundsMin = Vector3.Min(boundsMin, new Vector3(x, y, z));
                boundsMax = Vector3.Max(boundsMax, new Vector3(x, y, z));
            }
        }

        // Build index buffer: 16×16 cells, each split into 4 triangles via center vertex
        // Client topology: 1024 triangles = 3072 indices per cell (0xC00)
        var indices = new List<uint>(3072);
        for (int r = 0; r < innerEdge; r++)
        {
            for (int c = 0; c < innerEdge; c++)
            {
                // Outer corner indices
                uint v00 = (uint)(r * outerEdge + c);
                uint v10 = (uint)(r * outerEdge + c + 1);
                uint v01 = (uint)((r + 1) * outerEdge + c);
                uint v11 = (uint)((r + 1) * outerEdge + c + 1);
                // Inner center vertex
                uint center = (uint)(innerBase + r * innerEdge + c);

                // 4 triangles around center (matching client CreateAreaLowDetailIndices)
                indices.Add(v00); indices.Add(v10); indices.Add(center);
                indices.Add(v10); indices.Add(v11); indices.Add(center);
                indices.Add(v11); indices.Add(v01); indices.Add(center);
                indices.Add(v01); indices.Add(v00); indices.Add(center);
            }
        }

        // Compute per-vertex normals from triangle faces
        var normals = new Vector3[totalVerts];
        for (int i = 0; i < indices.Count; i += 3)
        {
            int i0 = (int)indices[i], i1 = (int)indices[i + 1], i2 = (int)indices[i + 2];
            var v0 = new Vector3(vertices[i0 * 6], vertices[i0 * 6 + 1], vertices[i0 * 6 + 2]);
            var v1 = new Vector3(vertices[i1 * 6], vertices[i1 * 6 + 1], vertices[i1 * 6 + 2]);
            var v2 = new Vector3(vertices[i2 * 6], vertices[i2 * 6 + 1], vertices[i2 * 6 + 2]);
            var normal = Vector3.Cross(v1 - v0, v2 - v0);
            normals[i0] += normal;
            normals[i1] += normal;
            normals[i2] += normal;
        }
        for (int i = 0; i < totalVerts; i++)
        {
            var n = Vector3.Normalize(normals[i]);
            if (float.IsNaN(n.X)) n = Vector3.UnitZ;
            vertices[i * 6 + 3] = n.X;
            vertices[i * 6 + 4] = n.Y;
            vertices[i * 6 + 5] = n.Z;
        }

        // Upload to GPU
        uint vao = _gl.GenVertexArray();
        _gl.BindVertexArray(vao);

        uint vbo = _gl.GenBuffer();
        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, vbo);
        fixed (float* ptr = vertices)
            _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(vertices.Length * sizeof(float)), ptr, BufferUsageARB.StaticDraw);

        uint ebo = _gl.GenBuffer();
        _gl.BindBuffer(BufferTargetARB.ElementArrayBuffer, ebo);
        var idxArray = indices.ToArray();
        fixed (uint* ptr = idxArray)
            _gl.BufferData(BufferTargetARB.ElementArrayBuffer, (nuint)(idxArray.Length * sizeof(uint)), ptr, BufferUsageARB.StaticDraw);

        uint stride = 6 * sizeof(float);
        // Position (location 0)
        _gl.EnableVertexAttribArray(0);
        _gl.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, stride, (void*)0);
        // Normal (location 1)
        _gl.EnableVertexAttribArray(1);
        _gl.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, stride, (void*)(3 * sizeof(float)));

        _gl.BindVertexArray(0);

        return new WdlTileMesh
        {
            Vao = vao, Vbo = vbo, Ebo = ebo,
            IndexCount = (uint)idxArray.Length,
            BoundsMin = boundsMin, BoundsMax = boundsMax
        };
    }

    // ── Shader ───────────────────────────────────────────────────────────

    private ShaderProgram CreateShader()
    {
        string vertSrc = @"
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

uniform mat4 uView;
uniform mat4 uProj;

out vec3 vNormal;
out vec3 vFragPos;

void main() {
    vFragPos = aPos;
    vNormal = aNormal;
    gl_Position = uProj * uView * vec4(aPos, 1.0);
}
";

        string fragSrc = @"
#version 330 core
in vec3 vNormal;
in vec3 vFragPos;

uniform vec3 uLightDir;
uniform vec3 uLightColor;
uniform vec3 uAmbientColor;
uniform vec3 uFogColor;
uniform float uFogStart;
uniform float uFogEnd;
uniform vec3 uCameraPos;

out vec4 FragColor;

void main() {
    // Height-based coloring: green lowlands, brown hills, gray peaks
    float height = vFragPos.z;
    vec3 baseColor;
    if (height < 50.0) {
        baseColor = mix(vec3(0.2, 0.35, 0.15), vec3(0.3, 0.5, 0.2), clamp(height / 50.0, 0.0, 1.0));
    } else if (height < 200.0) {
        float t = (height - 50.0) / 150.0;
        baseColor = mix(vec3(0.3, 0.5, 0.2), vec3(0.5, 0.4, 0.25), t);
    } else {
        float t = clamp((height - 200.0) / 300.0, 0.0, 1.0);
        baseColor = mix(vec3(0.5, 0.4, 0.25), vec3(0.6, 0.6, 0.55), t);
    }

    // Lighting
    vec3 norm = normalize(vNormal);
    float diff = max(dot(norm, normalize(uLightDir)), 0.0);
    vec3 litColor = baseColor * (uAmbientColor + uLightColor * diff);

    // Fog
    float dist = length(vFragPos - uCameraPos);
    float fogFactor = clamp((uFogEnd - dist) / (uFogEnd - uFogStart), 0.0, 1.0);
    vec3 finalColor = mix(uFogColor, litColor, fogFactor);

    FragColor = vec4(finalColor, 1.0);
}
";

        return ShaderProgram.Create(_gl, vertSrc, fragSrc);
    }

    // ── Cleanup ──────────────────────────────────────────────────────────

    public void Dispose()
    {
        foreach (var mesh in _tileMeshes.Values)
        {
            _gl.DeleteVertexArray(mesh.Vao);
            _gl.DeleteBuffer(mesh.Vbo);
            _gl.DeleteBuffer(mesh.Ebo);
        }
        _tileMeshes.Clear();
        _shader.Dispose();
    }

    private class WdlTileMesh
    {
        public uint Vao, Vbo, Ebo;
        public uint IndexCount;
        public Vector3 BoundsMin, BoundsMax;
    }
}
