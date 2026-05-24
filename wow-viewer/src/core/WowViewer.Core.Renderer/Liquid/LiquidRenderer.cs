using System.Numerics;
using Silk.NET.OpenGL;
using WowViewer.Core.Maps;
using WowViewer.Core.Renderer.Scene;

namespace WowViewer.Core.Renderer.Liquid;

public sealed class LiquidRenderer : IDisposable
{
    private readonly GL _gl;
    private readonly LiquidShader _shader;
    private readonly List<LiquidMesh> _meshes = [];
    private bool _disposed;

    private sealed record LiquidMesh(uint Vao, uint Vbo, uint Ebo, int IndexCount, Vector3 Color, float Opacity);

    public LiquidRenderer(GL gl)
    {
        _gl = gl;
        _shader = new LiquidShader(gl);
    }

    public void Build(AdtLiquidFile liquidFile, int tileX, int tileY)
    {
        foreach (var chunk in liquidFile.Chunks)
        {
            foreach (var layer in chunk.Layers)
                BuildLayer(chunk.ChunkIndex, layer);
        }
    }

    private unsafe void BuildLayer(int chunkIndex, AdtLiquidLayer layer)
    {
        int cx = chunkIndex % 16;
        int cy = chunkIndex / 16;

        float chunkWorldX = cx * Terrain.TerrainConstants.ChunkSize;
        float chunkWorldY = cy * Terrain.TerrainConstants.ChunkSize;

        float layerWorldX = chunkWorldX + layer.XOffset * Terrain.TerrainConstants.HalfCellSize;
        float layerWorldY = chunkWorldY + layer.YOffset * Terrain.TerrainConstants.HalfCellSize;

        int vw = layer.Width + 1;
        int vh = layer.Height + 1;
        int vc = vw * vh;

        float[] verts = new float[vc * 5];
        for (int vy = 0; vy < vh; vy++)
        {
            for (int vx = 0; vx < vw; vx++)
            {
                float height = layer.Heights is { Length: > 0 }
                    ? layer.Heights[vy * vw + vx]
                    : layer.MinHeight;

                int vi = (vy * vw + vx) * 5;
                verts[vi + 0] = layerWorldX + vx * Terrain.TerrainConstants.HalfCellSize;
                verts[vi + 1] = layerWorldY + vy * Terrain.TerrainConstants.HalfCellSize;
                verts[vi + 2] = height;
                verts[vi + 3] = (float)vx / layer.Width;
                verts[vi + 4] = (float)vy / layer.Height;
            }
        }

        var inds = new List<int>(layer.Width * layer.Height * 6);
        for (int ty = 0; ty < layer.Height; ty++)
        {
            for (int tx = 0; tx < layer.Width; tx++)
            {
                if (!layer.TileExists(tx, ty))
                    continue;

                int i0 = ty * vw + tx;
                int i1 = ty * vw + tx + 1;
                int i2 = (ty + 1) * vw + tx;
                int i3 = (ty + 1) * vw + tx + 1;
                inds.Add(i0); inds.Add(i2); inds.Add(i1);
                inds.Add(i1); inds.Add(i2); inds.Add(i3);
            }
        }

        if (inds.Count == 0)
            return;

        Vector3 color = layer.BasicType switch
        {
            AdtLiquidBasicType.Magma => new Vector3(0.9f, 0.4f, 0.05f),
            AdtLiquidBasicType.Slime => new Vector3(0.2f, 0.5f, 0.1f),
            _ => new Vector3(0.1f, 0.3f, 0.6f),
        };

        float opacity = layer.BasicType == AdtLiquidBasicType.Water ? 0.45f : 0.7f;

        uint vao = _gl.GenVertexArray();
        uint vbo = _gl.GenBuffer();
        uint ebo = _gl.GenBuffer();

        _gl.BindVertexArray(vao);
        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, vbo);
        fixed (float* ptr = verts)
            _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(verts.Length * sizeof(float)), ptr, BufferUsageARB.StaticDraw);

        uint stride = 5 * (uint)sizeof(float);
        _gl.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, stride, null);
        _gl.EnableVertexAttribArray(0);
        _gl.VertexAttribPointer(1, 2, VertexAttribPointerType.Float, false, stride, (void*)(3 * sizeof(float)));
        _gl.EnableVertexAttribArray(1);

        ushort[] indices = inds.Select(i => (ushort)i).ToArray();
        _gl.BindBuffer(BufferTargetARB.ElementArrayBuffer, ebo);
        fixed (ushort* ptr = indices)
            _gl.BufferData(BufferTargetARB.ElementArrayBuffer, (nuint)(indices.Length * sizeof(ushort)), ptr, BufferUsageARB.StaticDraw);

        _gl.BindVertexArray(0);

        _meshes.Add(new LiquidMesh(vao, vbo, ebo, indices.Length, color, opacity));
    }

    public void Render(SceneCamera camera, RenderVariant variant)
    {
        if (variant.HideLiquids)
            return;

        _shader.Use();
        var viewProj = camera.GetViewMatrix() * camera.GetProjectionMatrix();
        _shader.SetViewProj(viewProj);

        _gl.Enable(EnableCap.Blend);
        _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthMask(false);

        foreach (var mesh in _meshes)
        {
            _shader.SetModel(Matrix4x4.Identity);
            _shader.SetColor(mesh.Color);
            _shader.SetOpacity(mesh.Opacity);

            _gl.BindVertexArray(mesh.Vao);
            _gl.DrawElements(PrimitiveType.Triangles, (uint)mesh.IndexCount, DrawElementsType.UnsignedShort, IntPtr.Zero);
        }

        _gl.BindVertexArray(0);
        _gl.DepthMask(true);
        _gl.Disable(EnableCap.Blend);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var mesh in _meshes)
        {
            _gl.DeleteVertexArray(mesh.Vao);
            _gl.DeleteBuffer(mesh.Vbo);
            _gl.DeleteBuffer(mesh.Ebo);
        }
        _meshes.Clear();
        _shader.Dispose();
    }
}
