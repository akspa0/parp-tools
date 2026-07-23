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
/// entry point for Spec 118's per-object mask library. Reuses the same camera framing
/// (center + padded bounds, orthographic, top-down) already proven in the viewer's
/// ScreenshotRenderer roof capture.
/// </summary>
public sealed class ObjectCaptureRenderer : IDisposable
{
    private readonly GL _gl;
    private readonly ObjectCaptureShader _shader;
    private readonly TextureCache _textureCache;
    private RenderSurface? _surface;
    private int _resolution = -1;
    private bool _disposed;

    public ObjectCaptureRenderer(GL gl, IArchiveCatalog archiveCatalog)
    {
        _gl = gl ?? throw new ArgumentNullException(nameof(gl));
        _shader = new ObjectCaptureShader(gl);
        _textureCache = new TextureCache(gl, archiveCatalog);
    }

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

    // Set by CaptureTopDown before each render callback so the callback's Render(viewProj, mode)
    // closures above can read the current frame's matrix without a second parameter list.
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
        // Square capture (aspect 1.0): pad both axes to the larger half-extent so the object
        // is never clipped, matching ScreenshotRenderer.RenderWmoRoofTopDown's convention.
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

    /// <summary>Read back the bound FBO's color attachment as top-down-oriented RGBA (GL's
    /// bottom-up row order flipped to match image conventions).</summary>
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
