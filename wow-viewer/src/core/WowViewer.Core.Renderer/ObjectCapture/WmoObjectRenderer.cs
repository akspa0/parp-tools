using System.Numerics;
using Silk.NET.OpenGL;
using WowViewer.Core.IO.Converters;
using WowViewer.Core.Renderer.Texture;

namespace WowViewer.Core.Renderer.ObjectCapture;

/// <summary>
/// Headless, single-object WMO renderer for object capture. Consumes the same
/// <see cref="WmoV14ToV17Converter.WmoV14Data"/> the viewer's GlbExporter/ScreenshotRenderer
/// already parse successfully (no new WMO parser), builds one GPU buffer per group, and draws
/// per batch with the resolved material texture bound. Collision-only batches (MaterialId 0xFF)
/// are skipped, matching GlbExporter's existing convention.
/// </summary>
public sealed unsafe class WmoObjectRenderer : IDisposable
{
    private readonly GL _gl;
    private readonly ObjectCaptureShader _shader;
    private readonly TextureCache _textureCache;
    private readonly List<GpuGroup> _groups = [];
    private readonly Dictionary<byte, uint> _materialTextures = new();
    private bool _disposed;

    public Vector3 BoundsMin { get; private set; }
    public Vector3 BoundsMax { get; private set; }

    private sealed record GpuGroup(uint Vao, uint Vbo, uint Ebo, IReadOnlyList<WmoV14ToV17Converter.WmoBatch> Batches);

    public WmoObjectRenderer(GL gl, ObjectCaptureShader shader, TextureCache textureCache)
    {
        _gl = gl;
        _shader = shader;
        _textureCache = textureCache;
    }

    public void Build(WmoV14ToV17Converter.WmoV14Data wmo)
    {
        BoundsMin = wmo.BoundsMin;
        BoundsMax = wmo.BoundsMax;

        for (byte m = 0; m < wmo.Materials.Count; m++)
        {
            string texName = wmo.Materials[m].Texture1Name;
            _materialTextures[m] = string.IsNullOrEmpty(texName) ? 0 : _textureCache.GetOrCreateTexture(texName);
        }

        foreach (var group in wmo.Groups)
            BuildGroup(group);
    }

    private void BuildGroup(WmoV14ToV17Converter.WmoGroupData group)
    {
        if (group.Vertices.Count == 0 || group.Indices.Count == 0)
            return;

        List<Vector3> normals = group.Normals.Count == group.Vertices.Count
            ? group.Normals
            : GenerateFaceNormals(group);
        bool hasUVs = group.UVs.Count == group.Vertices.Count;

        int vc = group.Vertices.Count;
        float[] verts = new float[vc * 8];
        for (int i = 0; i < vc; i++)
        {
            Vector3 pos = group.Vertices[i];
            Vector3 norm = i < normals.Count ? normals[i] : Vector3.UnitZ;
            Vector2 uv = hasUVs ? group.UVs[i] : Vector2.Zero;

            verts[i * 8 + 0] = pos.X;
            verts[i * 8 + 1] = pos.Y;
            verts[i * 8 + 2] = pos.Z;
            verts[i * 8 + 3] = norm.X;
            verts[i * 8 + 4] = norm.Y;
            verts[i * 8 + 5] = norm.Z;
            verts[i * 8 + 6] = uv.X;
            verts[i * 8 + 7] = uv.Y;
        }

        ushort[] indices = group.Indices.ToArray();

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

        _groups.Add(new GpuGroup(vao, vbo, ebo, group.Batches));
    }

    /// <summary>Per-vertex normals via face-normal averaging (mirrors GlbExporter.GenerateNormals) for v14 WMOs with no MONR.</summary>
    private static List<Vector3> GenerateFaceNormals(WmoV14ToV17Converter.WmoGroupData group)
    {
        var normals = new Vector3[group.Vertices.Count];
        for (int i = 0; i + 2 < group.Indices.Count; i += 3)
        {
            int i0 = group.Indices[i], i1 = group.Indices[i + 1], i2 = group.Indices[i + 2];
            if (i0 >= group.Vertices.Count || i1 >= group.Vertices.Count || i2 >= group.Vertices.Count)
                continue;
            var e1 = group.Vertices[i1] - group.Vertices[i0];
            var e2 = group.Vertices[i2] - group.Vertices[i0];
            var n = Vector3.Cross(e1, e2);
            if (n.LengthSquared() > 0.0001f)
                n = Vector3.Normalize(n);
            else
                continue;
            normals[i0] += n;
            normals[i1] += n;
            normals[i2] += n;
        }
        return normals.Select(n => n.Length() > 0.001f ? Vector3.Normalize(n) : Vector3.UnitZ).ToList();
    }

    public unsafe void Render(Matrix4x4 viewProj, bool maskMode)
    {
        RenderWithTransform(viewProj, Matrix4x4.Identity, maskMode);
    }

    /// <summary>
    /// Renders this WMO with a per-instance world-space transform. Used by the tile-WMO
    /// compositor to render placed WMOs at their actual world positions (MODF placement).
    /// </summary>
    public unsafe void RenderWithTransform(Matrix4x4 viewProj, Matrix4x4 model, bool maskMode)
    {
        _shader.Use();
        _shader.SetViewProj(viewProj);
        _shader.SetModel(model);
        _shader.SetMaskMode(maskMode);
        _shader.SetSamplerUnit(0);

        _gl.Enable(EnableCap.DepthTest);
        _gl.Enable(EnableCap.CullFace);
        _gl.CullFace(TriangleFace.Back);
        _gl.ActiveTexture(TextureUnit.Texture0);

        foreach (var g in _groups)
        {
            _gl.BindVertexArray(g.Vao);
            foreach (var batch in g.Batches)
            {
                if (batch.MaterialId == 0xFF)
                    continue;

                uint tex = _materialTextures.GetValueOrDefault(batch.MaterialId, 0u);
                _gl.BindTexture(TextureTarget.Texture2D, tex);
                _shader.SetHasTexture(tex != 0);

                _gl.DrawElements(PrimitiveType.Triangles, batch.IndexCount,
                    DrawElementsType.UnsignedShort, (void*)(batch.FirstIndex * sizeof(ushort)));
            }
        }

        _gl.BindVertexArray(0);
        _gl.BindTexture(TextureTarget.Texture2D, 0);
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
    }
}
