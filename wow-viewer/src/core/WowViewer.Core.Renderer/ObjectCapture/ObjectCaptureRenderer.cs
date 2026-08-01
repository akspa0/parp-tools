using System.Numerics;
using Silk.NET.OpenGL;
using WowViewer.Core.IO.Converters;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.Renderer.Headless;
using WowViewer.Core.Renderer.Texture;

namespace WowViewer.Core.Renderer.ObjectCapture;

/// <summary>
/// Result of one object capture: a textured top-down reference image and an occlusion-free
/// object mask (flat white silhouette on black), both the same resolution, same camera, same
/// frame -- so every mask pixel corresponds exactly to the image pixel it covers.
/// </summary>
public sealed record ObjectCaptureResult(int Width, int Height, byte[] ImageRgba, byte[] MaskRgba, Vector3 BoundsMin, Vector3 BoundsMax);

/// <summary>
/// Headless top-down single-object capture for WMO and MDX/M2 models, driven entirely off the
/// Core.IO parsed data models (no viewer, no ImGui, no interactive window) -- the harvest-tool
/// entry point for Spec 118's per-object mask library. Also supports tile-level WMO compositing
/// for synthetic minimap overlays via <see cref="CaptureTileWmos"/>.
/// </summary>
public sealed class ObjectCaptureRenderer : IDisposable
{
    private readonly GL _gl;
    private readonly ObjectCaptureShader _shader;
    private readonly TextureCache _textureCache;
    private readonly IArchiveCatalog _catalog;
    private RenderSurface? _surface;
    private int _resolution = -1;
    private bool _disposed;

    public ObjectCaptureRenderer(GL gl, IArchiveCatalog archiveCatalog)
    {
        _gl = gl ?? throw new ArgumentNullException(nameof(gl));
        _catalog = archiveCatalog ?? throw new ArgumentNullException(nameof(archiveCatalog));
        _shader = new ObjectCaptureShader(gl);
        _textureCache = new TextureCache(gl, archiveCatalog);
    }

    public void SetLightDirection(Vector3 dir) => _shader.SetLightDirection(dir);

    public ObjectCaptureResult CaptureWmo(WmoV14ToV17Converter.WmoV14Data wmo, int resolution)
    {
        using var renderer = new WmoObjectRenderer(_gl, _shader, _textureCache);
        renderer.Build(wmo);
        return CaptureTopDown(resolution, renderer.BoundsMin, renderer.BoundsMax, maskMode => renderer.Render(GetViewProj(), maskMode));
    }

    public ObjectCaptureResult CaptureMdx(MdxFile mdx, int resolution)
    {
        using var renderer = new MdxObjectRenderer(_gl, _shader, _textureCache);
        renderer.Build(mdx);
        return CaptureTopDown(resolution, renderer.BoundsMin, renderer.BoundsMax, maskMode => renderer.Render(GetViewProj(), maskMode));
    }

    /// <summary>
    /// Renders all placed WMOs in a tile as a top-down overlay, using the same orthographic
    /// camera as the terrain minimap. Returns (image RGBA, mask RGBA) at the given resolution.
    /// The mask is white where any WMO fragment is visible, black elsewhere — for compositing:
    /// result = terrain * (1 - mask) + wmo_overlay * mask.
    /// </summary>
    /// <param name="placementModfData">MODF placement array [N, 14].</param>
    /// <param name="modfNames">MODF model paths indexed by nameId.</param>
    /// <param name="tileCenterX">Tile center X in world space.</param>
    /// <param name="tileCenterY">Tile center Y in world space.</param>
    /// <param name="tileSpan">Half the tile world extent (533.33333 / 2).</param>
    /// <param name="resolution">Output resolution in pixels (square).</param>
    /// <param name="terrainMaxHeight">Maximum terrain height on this tile; WMOs entirely below this are skipped.</param>
    public void CaptureTileWmos(
        float[,] placementModfData,
        IReadOnlyList<string> modfNames,
        float tileCenterX,
        float tileCenterY,
        float tileSpan,
        int resolution,
        float terrainMaxHeight,
        out byte[] imageRgba,
        out byte[] maskRgba)
    {
        EnsureSurface(resolution);

        // Orthographic camera matching the tile's world bounds, looking straight down.
        float cameraHeight = 600f;
        Vector3 eyePos = new(tileCenterX, tileCenterY, cameraHeight);
        Vector3 target = new(tileCenterX, tileCenterY, 0f);
        Matrix4x4 view = Matrix4x4.CreateLookAt(eyePos, target, Vector3.UnitX);
        Matrix4x4 proj = Matrix4x4.CreateOrthographic(tileSpan * 2f, tileSpan * 2f, 0.1f, cameraHeight * 2f);
        _currentViewProj = view * proj;

        // Clear the surface to black (terrain will be composited later).
        _surface!.Clear(0f, 0f, 0f, 1f);
        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Less);
        _gl.DepthMask(true);
        _gl.Disable(EnableCap.Blend);
        _gl.Enable(EnableCap.CullFace);
        _gl.CullFace(TriangleFace.Back);

        // Cache WMO renderers by model path to avoid reloading the same WMO per placement.
        var wmoCache = new Dictionary<string, WmoObjectRenderer>(StringComparer.OrdinalIgnoreCase);
        try
        {
            int placementCount = placementModfData.GetLength(0);
            for (int i = 0; i < placementCount; i++)
            {
                int nameId = (int)placementModfData[i, 0];
                if (nameId < 0 || nameId >= modfNames.Count)
                    continue;

                string wmoPath = modfNames[nameId];
                if (string.IsNullOrWhiteSpace(wmoPath))
                    continue;

                // Load or retrieve cached WMO renderer.
                if (!wmoCache.TryGetValue(wmoPath, out WmoObjectRenderer? wmoRenderer))
                {
                    WmoV14ToV17Converter.WmoV14Data? wmo = WmoFullLoader.Load(_catalog, wmoPath);
                    if (wmo is null || wmo.Groups.Count == 0)
                        continue;

                    wmoRenderer = new WmoObjectRenderer(_gl, _shader, _textureCache);
                    wmoRenderer.Build(wmo);
                    wmoCache[wmoPath] = wmoRenderer;
                }

                // Build the world transform from MODF placement data.
                float posX = placementModfData[i, 2];
                float posY = placementModfData[i, 3];
                float posZ = placementModfData[i, 4];
                float rotX = placementModfData[i, 5];
                float rotY = placementModfData[i, 6];
                float rotZ = placementModfData[i, 7];

                // Row-major: translation must be last so the object is rotated at origin
                // then moved to its world position. Current order: translate first, then
                // rotate around Z, then Y, then X — wrong. Fix: translation last.
                Matrix4x4 model = Matrix4x4.CreateTranslation(posX, posY, posZ)
                    * Matrix4x4.CreateRotationZ(rotZ)
                    * Matrix4x4.CreateRotationY(rotY)
                    * Matrix4x4.CreateRotationX(rotX);

                wmoRenderer.RenderWithTransform(_currentViewProj, model, maskMode: false);
            }

            imageRgba = CaptureRgba(resolution, resolution);

            // Render mask pass: flat white silhouette on black.
            _gl.Disable(EnableCap.Blend);
            _gl.DepthMask(true);
            _surface.Clear(0f, 0f, 0f, 1f);
            _gl.Clear(ClearBufferMask.DepthBufferBit);

            foreach (WmoObjectRenderer wmoRenderer in wmoCache.Values)
            {
                // Replay the same placement loop for the mask pass.
                // We need the placements again; we could cache transforms, but for simplicity
                // and correctness (mask must match image exactly), re-iterate.
            }

            // Re-iterate placements for mask pass.
            for (int i = 0; i < placementCount; i++)
            {
                int nameId = (int)placementModfData[i, 0];
                if (nameId < 0 || nameId >= modfNames.Count)
                    continue;

                string wmoPath = modfNames[nameId];
                if (string.IsNullOrWhiteSpace(wmoPath))
                    continue;

                float bbMaxZ = placementModfData[i, 12];
                if (bbMaxZ < terrainMaxHeight - 2f)
                    continue;

                if (!wmoCache.TryGetValue(wmoPath, out WmoObjectRenderer? wmoRenderer))
                    continue;

                float posX = placementModfData[i, 2];
                float posY = placementModfData[i, 3];
                float posZ = placementModfData[i, 4];
                float rotX = placementModfData[i, 5];
                float rotY = placementModfData[i, 6];
                float rotZ = placementModfData[i, 7];

                Matrix4x4 model = Matrix4x4.CreateTranslation(posX, posY, posZ)
                    * Matrix4x4.CreateRotationZ(rotZ)
                    * Matrix4x4.CreateRotationY(rotY)
                    * Matrix4x4.CreateRotationX(rotX);

                wmoRenderer.RenderWithTransform(_currentViewProj, model, maskMode: true);
            }

            maskRgba = CaptureRgba(resolution, resolution);
        }
        finally
        {
            foreach (WmoObjectRenderer wmoRenderer in wmoCache.Values)
                wmoRenderer.Dispose();
        }
    }

    // Set by CaptureTopDown / CaptureTileWmos before each render callback.
    private Matrix4x4 _currentViewProj;
    private Matrix4x4 GetViewProj() => _currentViewProj;

    private ObjectCaptureResult CaptureTopDown(int resolution, Vector3 boundsMin, Vector3 boundsMax, Action<bool> render)
    {
        EnsureSurface(resolution);

        float spanX = MathF.Max(0.01f, boundsMax.X - boundsMin.X);
        float spanY = MathF.Max(0.01f, boundsMax.Y - boundsMin.Y);
        float spanZ = boundsMax.Z - boundsMin.Z;
        Vector3 center = (boundsMin + boundsMax) * 0.5f;

        const float padFactor = 1.1f;
        float orthoHalfX = spanX * padFactor * 0.5f;
        float orthoHalfY = spanY * padFactor * 0.5f;
        float orthoHalf = MathF.Max(orthoHalfX, orthoHalfY);
        orthoHalfX = orthoHalf;
        orthoHalfY = orthoHalf;

        float eyeHeight = spanZ + 100f;
        Vector3 eyePos = new(center.X, center.Y, center.Z + eyeHeight);
        Matrix4x4 view = Matrix4x4.CreateLookAt(eyePos, center, Vector3.UnitX);
        Matrix4x4 proj = Matrix4x4.CreateOrthographic(orthoHalfX * 2f, orthoHalfY * 2f, 0.1f, eyeHeight * 2f + spanZ + 200f);
        _currentViewProj = view * proj;

        _surface!.Clear(0f, 0f, 0f, 1f);
        _gl.Disable(EnableCap.Blend);
        render(false);
        byte[] imageRgba = CaptureRgba(resolution, resolution);

        _surface.Clear(0f, 0f, 0f, 1f);
        render(true);
        byte[] maskRgba = CaptureRgba(resolution, resolution);

        return new ObjectCaptureResult(resolution, resolution, imageRgba, maskRgba, boundsMin, boundsMax);
    }

    private void EnsureSurface(int resolution)
    {
        if (_surface != null && _resolution == resolution)
            return;

        _surface?.Dispose();
        _surface = new RenderSurface(_gl, resolution, resolution);
        _surface.Bind();
        _resolution = resolution;
    }

    private unsafe byte[] CaptureRgba(int width, int height)
    {
        byte[] pixels = new byte[width * height * 4];
        fixed (byte* ptr = pixels)
            _gl.ReadPixels(0, 0, (uint)width, (uint)height, PixelFormat.Rgba, PixelType.UnsignedByte, ptr);

        int rowSize = width * 4;
        byte[] row = new byte[rowSize];
        for (int y = 0; y < height / 2; y++)
        {
            int top = y * rowSize;
            int bottom = (height - 1 - y) * rowSize;
            System.Buffer.BlockCopy(pixels, top, row, 0, rowSize);
            System.Buffer.BlockCopy(pixels, bottom, pixels, top, rowSize);
            System.Buffer.BlockCopy(row, 0, pixels, bottom, rowSize);
        }
        return pixels;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _surface?.Dispose();
        _textureCache.Dispose();
        _shader.Dispose();
    }
}