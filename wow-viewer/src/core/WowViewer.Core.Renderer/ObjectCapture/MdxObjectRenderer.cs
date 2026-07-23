using System.Numerics;
using Silk.NET.OpenGL;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.Renderer.Texture;

namespace WowViewer.Core.Renderer.ObjectCapture;

/// <summary>
/// Headless, single-object MDX/M2 renderer for object capture. Consumes the same
/// <see cref="MdxFile"/> Core.IO data model the viewer's GlbExporter already parses successfully
/// (no new MDX parser). Static bind-pose only -- no bone/animation evaluation -- matching the
/// established convention for object-capture renders (the viewer's own roof/multi-angle capture
/// explicitly disables animation and uses raw stored vertex positions for the same reason).
/// </summary>
public sealed unsafe class MdxObjectRenderer : IDisposable
{
    private readonly GL _gl;
    private readonly ObjectCaptureShader _shader;
    private readonly TextureCache _textureCache;
    private readonly List<GpuGeoset> _geosets = [];
    private bool _disposed;

    public Vector3 BoundsMin { get; private set; }
    public Vector3 BoundsMax { get; private set; }

    private sealed record GpuGeoset(uint Vao, uint Vbo, uint Ebo, int IndexCount, uint Texture);

    public MdxObjectRenderer(GL gl, ObjectCaptureShader shader, TextureCache textureCache)
    {
        _gl = gl;
        _shader = shader;
        _textureCache = textureCache;
    }

    public void Build(MdxFile mdx)
    {
        Vector3 boundsMin = new(float.MaxValue);
        Vector3 boundsMax = new(float.MinValue);

        foreach (var geoset in mdx.Geosets)
        {
            if (geoset.Vertices.Count == 0 || geoset.Indices.Count == 0)
                continue;

            foreach (var v in geoset.Vertices)
            {
                boundsMin = Vector3.Min(boundsMin, new Vector3(v.X, v.Y, v.Z));
                boundsMax = Vector3.Max(boundsMax, new Vector3(v.X, v.Y, v.Z));
            }

            uint texture = ResolveGeosetTexture(mdx, geoset);
            BuildGeoset(geoset, texture);
        }

        BoundsMin = boundsMin.X <= boundsMax.X ? boundsMin : Vector3.Zero;
        BoundsMax = boundsMin.X <= boundsMax.X ? boundsMax : Vector3.Zero;
    }

    private uint ResolveGeosetTexture(MdxFile mdx, MdlGeoset geoset)
    {
        int texId = -1;
        if (geoset.MaterialId >= 0 && geoset.MaterialId < mdx.Materials.Count)
        {
            var mat = mdx.Materials[geoset.MaterialId];
            if (mat.Layers.Count > 0 && mat.Layers[0].TextureId >= 0 && mat.Layers[0].TextureId < mdx.Textures.Count)
                texId = mat.Layers[0].TextureId;
        }
        if (texId < 0 && mdx.Textures.Count > 0) texId = 0;
        if (texId < 0) return 0;

        string path = mdx.Textures[texId].Path;
        return string.IsNullOrEmpty(path) ? 0 : _textureCache.GetOrCreateTexture(path);
    }

    private void BuildGeoset(MdlGeoset geoset, uint texture)
    {
        bool hasNormals = geoset.Normals.Count == geoset.Vertices.Count;
        bool hasUVs = geoset.TexCoords.Count == geoset.Vertices.Count;

        int vc = geoset.Vertices.Count;
        float[] verts = new float[vc * 8];
        for (int i = 0; i < vc; i++)
        {
            C3Vector pos = geoset.Vertices[i];
            C3Vector norm = hasNormals ? geoset.Normals[i] : new C3Vector(0f, 0f, 1f);
            C2Vector uv = hasUVs ? geoset.TexCoords[i] : new C2Vector(0f, 0f);

            verts[i * 8 + 0] = pos.X;
            verts[i * 8 + 1] = pos.Y;
            verts[i * 8 + 2] = pos.Z;
            verts[i * 8 + 3] = norm.X;
            verts[i * 8 + 4] = norm.Y;
            verts[i * 8 + 5] = norm.Z;
            verts[i * 8 + 6] = uv.U;
            verts[i * 8 + 7] = uv.V;
        }

        ushort[] indices = geoset.Indices.ToArray();

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

        _geosets.Add(new GpuGeoset(vao, vbo, ebo, indices.Length, texture));
    }

    public unsafe void Render(Matrix4x4 viewProj, bool maskMode)
    {
        _shader.Use();
        _shader.SetViewProj(viewProj);
        _shader.SetModel(Matrix4x4.Identity);
        _shader.SetMaskMode(maskMode);
        _shader.SetSamplerUnit(0);

        _gl.Enable(EnableCap.DepthTest);
        _gl.Enable(EnableCap.CullFace);
        _gl.CullFace(TriangleFace.Back);
        _gl.ActiveTexture(TextureUnit.Texture0);

        foreach (var g in _geosets)
        {
            _gl.BindVertexArray(g.Vao);
            _gl.BindTexture(TextureTarget.Texture2D, g.Texture);
            _shader.SetHasTexture(g.Texture != 0);
            _gl.DrawElements(PrimitiveType.Triangles, (uint)g.IndexCount, DrawElementsType.UnsignedShort, (void*)0);
        }

        _gl.BindVertexArray(0);
        _gl.BindTexture(TextureTarget.Texture2D, 0);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var g in _geosets)
        {
            _gl.DeleteVertexArray(g.Vao);
            _gl.DeleteBuffer(g.Vbo);
            _gl.DeleteBuffer(g.Ebo);
        }
        _geosets.Clear();
    }
}
