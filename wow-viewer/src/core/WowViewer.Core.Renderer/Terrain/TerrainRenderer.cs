using System.Numerics;
using Silk.NET.OpenGL;
using WowViewer.Core.Renderer.Scene;

namespace WowViewer.Core.Renderer.Terrain;

public sealed class TerrainRenderer : IDisposable
{
    private readonly GL _gl;
    private readonly TerrainShader _shader;
    private readonly FrustumCuller _culler = new();
    private bool _disposed;

    public TerrainRenderer(GL gl, TerrainShader shader)
    {
        _gl = gl ?? throw new ArgumentNullException(nameof(gl));
        _shader = shader ?? throw new ArgumentNullException(nameof(shader));
    }

    public FrustumCuller Culler => _culler;

    public void Render(
        SceneCamera camera,
        IEnumerable<TerrainMesh> meshes,
        RenderVariant variant)
    {
        _shader.Use();
        _shader.SetView(camera.GetViewMatrix());
        _shader.SetProjection(camera.GetProjectionMatrix());
        _shader.SetCameraPosition(camera.Position);
        _shader.SetUseWorldUV(true);
        _shader.SetUseMccv(false);

        _shader.SetShowLayer(0, !variant.HideTerrain);
        _shader.SetShowLayer(1, !variant.HideTerrain);
        _shader.SetShowLayer(2, !variant.HideTerrain);
        _shader.SetShowLayer(3, !variant.HideTerrain);
        _shader.SetShowShadowMap(false);

        _shader.SetLightDirection(Vector3.Normalize(new Vector3(0.5f, -0.5f, 0.8f)));
        _shader.SetLightColor(new Vector3(0.9f, 0.85f, 0.8f));
        _shader.SetAmbientColor(new Vector3(0.35f, 0.35f, 0.35f));
        _shader.SetFogColor(new Vector3(0.34f, 0.38f, 0.42f));
        _shader.SetFogStart(4000f);
        _shader.SetFogEnd(7000f);

        _culler.ComputePlanes(camera.GetViewMatrix() * camera.GetProjectionMatrix());

        _gl.Enable(EnableCap.DepthTest);
        _gl.Enable(EnableCap.CullFace);
        _gl.CullFace(TriangleFace.Back);

        foreach (var mesh in meshes)
        {
            if (!_culler.TestAABB(mesh.BoundsMin, mesh.BoundsMax))
                continue;

            _shader.SetDiffuseLayerCount(mesh.DiffuseLayerCount);

            Matrix4x4 model = Matrix4x4.Identity;
            _shader.SetModel(model);

            _gl.ActiveTexture(TextureUnit.Texture0);
            _gl.BindTexture(TextureTarget.Texture2DArray, mesh.DiffuseArrayTexture);
            _gl.ActiveTexture(TextureUnit.Texture1);
            _gl.BindTexture(TextureTarget.Texture2DArray, mesh.AlphaShadowArrayTexture);

            _gl.BindVertexArray(mesh.Vao);
            _gl.DrawElements(PrimitiveType.Triangles, mesh.IndexCount, DrawElementsType.UnsignedShort, IntPtr.Zero);
            _gl.BindVertexArray(0);
        }
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        _shader.Dispose();
    }
}
