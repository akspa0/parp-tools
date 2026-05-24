using System.Numerics;
using Silk.NET.OpenGL;
using WowViewer.Core.Renderer.Scene;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Renderer.Wmo;

public sealed class WmoRenderer : IDisposable
{
    private readonly GL _gl;
    private readonly List<GpuGroup> _groups = [];
    private readonly WmoShader _shader;
    private bool _disposed;

    private sealed record GpuGroup(uint Vao, uint Vbo, uint Ebo, int IndexCount, Vector3 BoundsMin, Vector3 BoundsMax, IReadOnlyList<WmoBatchCall> Batches);

    public WmoRenderer(GL gl)
    {
        _gl = gl;
        _shader = new WmoShader(gl);
    }

    public void Build(WmoRenderDocument doc)
    {
        foreach (var group in doc.Groups)
            BuildGroup(group, doc);
    }

    private unsafe void BuildGroup(WmoEmbeddedGroupMeshDetail group, WmoRenderDocument doc)
    {
        var mesh = group.Mesh;
        if (mesh.Indices.Count == 0)
            return;

        int vc = mesh.Vertices.Count;
        float[] verts = new float[vc * 8];

        Vector3 boundsMin = new(float.MaxValue);
        Vector3 boundsMax = new(float.MinValue);

        for (int i = 0; i < vc; i++)
        {
            Vector3 pos = mesh.Vertices[i];
            Vector3 norm = i < mesh.Normals.Count ? mesh.Normals[i] : Vector3.UnitZ;
            Vector2 uv = i < mesh.PrimaryUvs.Count ? mesh.PrimaryUvs[i] : Vector2.Zero;

            verts[i * 8 + 0] = pos.X;
            verts[i * 8 + 1] = pos.Y;
            verts[i * 8 + 2] = pos.Z;
            verts[i * 8 + 3] = norm.X;
            verts[i * 8 + 4] = norm.Y;
            verts[i * 8 + 5] = norm.Z;
            verts[i * 8 + 6] = uv.X;
            verts[i * 8 + 7] = uv.Y;

            boundsMin = Vector3.Min(boundsMin, pos);
            boundsMax = Vector3.Max(boundsMax, pos);
        }

        ushort[] indices = mesh.Indices.ToArray();

        uint vao = _gl.GenVertexArray();
        uint vbo = _gl.GenBuffer();
        uint ebo = _gl.GenBuffer();

        _gl.BindVertexArray(vao);
        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, vbo);
        fixed (float* ptr = verts)
            _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(verts.Length * sizeof(float)), ptr, BufferUsageARB.StaticDraw);

        uint stride = 8 * (uint)sizeof(float);
        _gl.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, stride, null);
        _gl.EnableVertexAttribArray(0);
        _gl.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, stride, (void*)(3 * sizeof(float)));
        _gl.EnableVertexAttribArray(1);
        _gl.VertexAttribPointer(2, 2, VertexAttribPointerType.Float, false, stride, (void*)(6 * sizeof(float)));
        _gl.EnableVertexAttribArray(2);

        _gl.BindBuffer(BufferTargetARB.ElementArrayBuffer, ebo);
        fixed (ushort* ptr = indices)
            _gl.BufferData(BufferTargetARB.ElementArrayBuffer, (nuint)(indices.Length * sizeof(ushort)), ptr, BufferUsageARB.StaticDraw);
        _gl.BindVertexArray(0);

        var batches = new List<WmoBatchCall>(mesh.Batches.Count);
        int transparentCount = group.GroupSummary.TransparentBatchCount;
        foreach (var batch in mesh.Batches)
        {
            if (!batch.MaterialId.HasValue)
                continue;
            batches.Add(new WmoBatchCall(
                batch.FirstIndex,
                batch.IndexCount,
                batch.MaterialId.Value,
                batch.BatchIndex < transparentCount));
        }

        _groups.Add(new GpuGroup(vao, vbo, ebo, indices.Length, boundsMin, boundsMax, batches));
    }

    public unsafe void Render(SceneCamera camera, RenderVariant variant)
    {
        if (variant.HideObjects)
            return;

        _shader.Use();
        var viewProj = camera.GetViewMatrix() * camera.GetProjectionMatrix();
        _shader.SetViewProj(viewProj);
        _shader.SetModel(Matrix4x4.Identity);

        _gl.Enable(EnableCap.DepthTest);
        _gl.Enable(EnableCap.CullFace);
        _gl.CullFace(TriangleFace.Back);

        foreach (var g in _groups)
        {
            _gl.BindVertexArray(g.Vao);
            foreach (var batch in g.Batches)
            {
                if (batch.IsTransparent)
                    continue;

                _shader.SetDiffuseColor(Vector3.One);
                _gl.DrawElements(PrimitiveType.Triangles, batch.IndexCount,
                    DrawElementsType.UnsignedShort, (void*)(batch.FirstIndex * sizeof(ushort)));
            }
        }

        _gl.Enable(EnableCap.Blend);
        _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
        _gl.DepthMask(false);

        foreach (var g in _groups)
        {
            _gl.BindVertexArray(g.Vao);
            foreach (var batch in g.Batches)
            {
                if (!batch.IsTransparent)
                    continue;

                _shader.SetDiffuseColor(Vector3.One * 0.8f);
                _gl.DrawElements(PrimitiveType.Triangles, batch.IndexCount,
                    DrawElementsType.UnsignedShort, (void*)(batch.FirstIndex * sizeof(ushort)));
            }
        }

        _gl.DepthMask(true);
        _gl.Disable(EnableCap.Blend);
        _gl.BindVertexArray(0);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var g in _groups)
        {
            _gl.DeleteVertexArray(g.Vao);
            _gl.DeleteBuffer(g.Vbo);
            _gl.DeleteBuffer(g.Ebo);
        }
        _groups.Clear();
        _shader.Dispose();
    }
}
