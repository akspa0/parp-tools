using System.Numerics;
using MdxLTool.Formats.Mdx;
using MdxViewer.DataSources;
using MdxViewer.Rendering;
using Silk.NET.OpenGL;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace MdxViewer.Catalog;

/// <summary>
/// Renders a single MDX model to an offscreen framebuffer, composites a nameplate,
/// and saves the result as a PNG screenshot.
/// </summary>
public class ScreenshotRenderer : IDisposable
{
    private readonly GL _gl;
    private readonly IDataSource? _dataSource;
    private readonly ReplaceableTextureResolver? _texResolver;

    private uint _fbo;
    private uint _colorTex;
    private uint _depthRbo;
    private int _width;
    private int _height;
    private bool _fboReady;

    public ScreenshotRenderer(GL gl, IDataSource? dataSource, ReplaceableTextureResolver? texResolver = null)
    {
        _gl = gl;
        _dataSource = dataSource;
        _texResolver = texResolver;
    }

    /// <summary>
    /// Capture a screenshot of an MDX model with a nameplate overlay.
    /// </summary>
    public bool CaptureScreenshot(AssetCatalogEntry entry, string outputPath, int width = 512, int height = 512)
    {
        if (_dataSource == null || string.IsNullOrEmpty(entry.ModelPath))
            return false;

        // Skip WMO for now — screenshot only supports MDX
        if (entry.IsWmo)
        {
            Console.WriteLine($"[Screenshot] WMO screenshot not yet supported: {entry.ModelPath}");
            return false;
        }

        // Load MDX
        byte[]? mdxData = _dataSource.ReadFile(entry.ModelPath);
        if (mdxData == null) return false;

        MdxFile mdx;
        try
        {
            using var ms = new MemoryStream(mdxData);
            using var br = new BinaryReader(ms);
            mdx = MdxFile.Load(br);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Screenshot] Failed to load MDX: {ex.Message}");
            return false;
        }

        // Create renderer for this model
        string modelDir = Path.GetDirectoryName(entry.ModelPath)?.Replace('/', '\\') ?? "";
        using var renderer = new MdxRenderer(_gl, mdx, modelDir, _dataSource, _texResolver);

        // Setup FBO
        EnsureFbo(width, height);
        _gl.BindFramebuffer(FramebufferTarget.Framebuffer, _fbo);
        _gl.Viewport(0, 0, (uint)width, (uint)height);

        // Clear with a neutral background
        _gl.ClearColor(0.15f, 0.15f, 0.2f, 1.0f);
        _gl.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Less);
        _gl.DepthMask(true);

        // Compute camera to frame the model
        var (boundsMin, boundsMax) = ComputeBounds(mdx);
        var center = (boundsMin + boundsMax) * 0.5f;
        float radius = (boundsMax - boundsMin).Length() * 0.5f;
        if (radius < 0.01f) radius = 1.0f;

        // Camera positioned at 45° elevation, looking at center
        float dist = radius * 2.5f;
        float elevation = MathF.PI / 6f; // 30 degrees
        float azimuth = MathF.PI / 4f;   // 45 degrees
        var camPos = center + new Vector3(
            dist * MathF.Cos(elevation) * MathF.Sin(azimuth),
            dist * MathF.Sin(elevation),
            dist * MathF.Cos(elevation) * MathF.Cos(azimuth));

        var view = Matrix4x4.CreateLookAt(camPos, center, Vector3.UnitY);
        float aspect = (float)width / height;
        var proj = Matrix4x4.CreatePerspectiveFieldOfView(MathF.PI / 4f, aspect, 0.01f, dist * 10f);

        // Apply scale
        var scale = Matrix4x4.CreateScale(entry.EffectiveScale);

        // Render opaque pass
        _gl.Disable(EnableCap.Blend);
        renderer.RenderWithTransform(scale, view, proj, RenderPass.Opaque, 1.0f,
            new Vector3(0.5f, 0.6f, 0.7f), dist * 5f, dist * 10f, camPos,
            Vector3.Normalize(new Vector3(-0.5f, 0.8f, 0.3f)),
            new Vector3(1.0f, 0.95f, 0.9f),
            new Vector3(0.3f, 0.3f, 0.35f));

        // Render transparent pass
        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Lequal);
        renderer.RenderWithTransform(scale, view, proj, RenderPass.Transparent, 1.0f,
            new Vector3(0.5f, 0.6f, 0.7f), dist * 5f, dist * 10f, camPos,
            Vector3.Normalize(new Vector3(-0.5f, 0.8f, 0.3f)),
            new Vector3(1.0f, 0.95f, 0.9f),
            new Vector3(0.3f, 0.3f, 0.35f));

        // Read pixels
        byte[] pixels = new byte[width * height * 4];
        unsafe
        {
            fixed (byte* ptr = pixels)
                _gl.ReadPixels(0, 0, (uint)width, (uint)height, PixelFormat.Rgba, PixelType.UnsignedByte, ptr);
        }

        // Unbind FBO
        _gl.BindFramebuffer(FramebufferTarget.Framebuffer, 0);

        // Convert to ImageSharp image (OpenGL reads bottom-up, need to flip)
        using var image = SixLabors.ImageSharp.Image.LoadPixelData<Rgba32>(pixels, width, height);
        image.Mutate(x => x.Flip(FlipMode.Vertical));

        // Draw nameplate overlay
        DrawNameplate(image, entry);

        // Save
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
        image.SaveAsPng(outputPath);
        Console.WriteLine($"[Screenshot] Saved: {outputPath}");
        return true;
    }

    private void DrawNameplate(SixLabors.ImageSharp.Image<Rgba32> image, AssetCatalogEntry entry)
    {
        // Draw a simple nameplate at the top of the image
        int w = image.Width;
        int plateHeight = 40;
        int plateY = 8;

        // Semi-transparent black background bar
        for (int y = plateY; y < plateY + plateHeight && y < image.Height; y++)
        {
            for (int x = 0; x < w; x++)
            {
                var existing = image[x, y];
                // Alpha blend with 60% black
                byte r = (byte)(existing.R * 0.4f);
                byte g = (byte)(existing.G * 0.4f);
                byte b = (byte)(existing.B * 0.4f);
                image[x, y] = new Rgba32(r, g, b, 255);
            }
        }

        // Nameplate text is embedded in the PNG filename and JSON metadata.
        // Full text rendering requires SixLabors.Fonts + SixLabors.ImageSharp.Drawing
        // packages which can be added later for richer overlays.
    }

    private static (Vector3 min, Vector3 max) ComputeBounds(MdxFile mdx)
    {
        var min = new Vector3(float.MaxValue);
        var max = new Vector3(float.MinValue);
        bool hasVerts = false;

        foreach (var geo in mdx.Geosets)
        {
            foreach (var v in geo.Vertices)
            {
                var sv = new Vector3(v.X, v.Y, v.Z);
                min = Vector3.Min(min, sv);
                max = Vector3.Max(max, sv);
                hasVerts = true;
            }
        }

        if (!hasVerts)
        {
            min = new Vector3(-1);
            max = new Vector3(1);
        }

        return (min, max);
    }

    private void EnsureFbo(int width, int height)
    {
        if (_fboReady && _width == width && _height == height) return;

        // Cleanup old
        if (_fboReady)
        {
            _gl.DeleteFramebuffer(_fbo);
            _gl.DeleteTexture(_colorTex);
            _gl.DeleteRenderbuffer(_depthRbo);
        }

        _width = width;
        _height = height;

        // Create color texture
        _colorTex = _gl.GenTexture();
        _gl.BindTexture(TextureTarget.Texture2D, _colorTex);
        _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba8,
            (uint)width, (uint)height, 0, PixelFormat.Rgba, PixelType.UnsignedByte, ReadOnlySpan<byte>.Empty);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);

        // Create depth renderbuffer
        _depthRbo = _gl.GenRenderbuffer();
        _gl.BindRenderbuffer(RenderbufferTarget.Renderbuffer, _depthRbo);
        _gl.RenderbufferStorage(RenderbufferTarget.Renderbuffer, InternalFormat.Depth24Stencil8, (uint)width, (uint)height);

        // Create FBO
        _fbo = _gl.GenFramebuffer();
        _gl.BindFramebuffer(FramebufferTarget.Framebuffer, _fbo);
        _gl.FramebufferTexture2D(FramebufferTarget.Framebuffer, FramebufferAttachment.ColorAttachment0,
            TextureTarget.Texture2D, _colorTex, 0);
        _gl.FramebufferRenderbuffer(FramebufferTarget.Framebuffer, FramebufferAttachment.DepthStencilAttachment,
            RenderbufferTarget.Renderbuffer, _depthRbo);

        var status = _gl.CheckFramebufferStatus(FramebufferTarget.Framebuffer);
        if (status != GLEnum.FramebufferComplete)
        {
            Console.WriteLine($"[Screenshot] FBO incomplete: {status}");
            _gl.BindFramebuffer(FramebufferTarget.Framebuffer, 0);
            return;
        }

        _gl.BindFramebuffer(FramebufferTarget.Framebuffer, 0);
        _fboReady = true;
    }

    public void Dispose()
    {
        if (_fboReady)
        {
            _gl.DeleteFramebuffer(_fbo);
            _gl.DeleteTexture(_colorTex);
            _gl.DeleteRenderbuffer(_depthRbo);
            _fboReady = false;
        }
    }
}
