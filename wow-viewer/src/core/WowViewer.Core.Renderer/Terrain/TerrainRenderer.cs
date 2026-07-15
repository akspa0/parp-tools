using System.Numerics;
using Silk.NET.OpenGL;
using WowViewer.Core.Maps;
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

    public bool ShowShadowMap { get; set; } = true;

    public Vector3 LightDirection { get; set; } = Vector3.Normalize(new Vector3(0.5f, -0.5f, 0.8f));

    public Vector3 DirectionalLightColor { get; set; } = new(0.9f, 0.85f, 0.8f);

    public Vector3 AmbientLightColor { get; set; } = new(0.35f, 0.35f, 0.35f);

    public float ShadowStrength { get; set; } = TerrainLightingMath.DefaultAuthoredMcshShadowStrength;

    public Vector3 FogColor { get; set; } = new(0.34f, 0.38f, 0.42f);

    public float FogStart { get; set; } = 4000f;

    public float FogEnd { get; set; } = 7000f;

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
        _shader.SetShowShadowMap(ShowShadowMap);
        _shader.SetShadowStrength(ShadowStrength);

        Vector3 lightDirection = LightDirection.LengthSquared() > 1e-10f
            ? Vector3.Normalize(LightDirection)
            : Vector3.UnitZ;
        _shader.SetLightDirection(lightDirection);
        _shader.SetLightColor(DirectionalLightColor);
        _shader.SetAmbientColor(AmbientLightColor);
        _shader.SetFogColor(FogColor);
        _shader.SetFogStart(FogStart);
        _shader.SetFogEnd(FogEnd);

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
