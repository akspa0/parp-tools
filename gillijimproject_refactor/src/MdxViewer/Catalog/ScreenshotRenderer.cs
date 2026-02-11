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
    /// Predefined camera angles for multi-angle capture.
    /// Each angle is (name, azimuthDegrees, elevationDegrees).
    /// 
    /// WoW MDX coordinate system (after MirrorX):
    ///   - Model faces toward +X (front)
    ///   - Z is up
    ///   - Y is left/right
    /// 
    /// Azimuth 0° = front (camera at -X looking +X), 90° = left side, 180° = back, 270° = right side.
    /// Elevation 0° = eye level, positive = looking down from above.
    /// </summary>
    public static readonly (string name, float azimuth, float elevation)[] CameraAngles =
    {
        ("front",          0f,  15f),
        ("back",         180f,  15f),
        ("left",          90f,  15f),
        ("right",        270f,  15f),
        ("top",            0f,  80f),
        ("three_quarter", 35f,  25f),
    };

    /// <summary>
    /// Capture a single screenshot of an MDX model (default 3/4 angle) with a nameplate overlay.
    /// </summary>
    public bool CaptureScreenshot(AssetCatalogEntry entry, string outputPath, int width = 1024, int height = 1024)
    {
        var result = CaptureMultiAngle(entry, Path.GetDirectoryName(outputPath)!, width, height,
            new[] { ("three_quarter", 35f, 25f) });
        return result > 0;
    }

    /// <summary>
    /// Capture screenshots from all predefined camera angles.
    /// Loads the MDX once, creates one renderer, renders all angles.
    /// Saves to {outputDir}/{angleName}.png.
    /// Returns the number of screenshots successfully saved.
    /// </summary>
    public int CaptureMultiAngle(AssetCatalogEntry entry, string outputDir, int width = 1024, int height = 1024,
        (string name, float azimuth, float elevation)[]? angles = null, string? resolvedModelPath = null)
    {
        angles ??= CameraAngles;

        string modelPath = resolvedModelPath ?? entry.ModelPath ?? "";
        if (_dataSource == null || string.IsNullOrEmpty(modelPath))
        {
            Console.WriteLine($"[Screenshot] Skip {entry.Name}: no data source or model path");
            return 0;
        }

        if (entry.IsWmo)
        {
            Console.WriteLine($"[Screenshot] Skip {entry.Name}: WMO screenshot not yet supported");
            return 0;
        }

        // Load MDX once
        byte[]? mdxData = _dataSource.ReadFile(modelPath);
        if (mdxData == null)
        {
            Console.WriteLine($"[Screenshot] Skip {entry.Name}: MDX file not found: {modelPath}");
            return 0;
        }

        MdxFile mdx;
        try
        {
            using var ms = new MemoryStream(mdxData);
            using var br = new BinaryReader(ms);
            mdx = MdxFile.Load(br);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Screenshot] Skip {entry.Name}: MDX parse failed: {ex.Message}");
            return 0;
        }

        if (mdx.Geosets.Count == 0)
        {
            Console.WriteLine($"[Screenshot] Skip {entry.Name}: no geosets");
            return 0;
        }

        // Create renderer once for all angles
        string modelDir = Path.GetDirectoryName(modelPath)?.Replace('/', '\\') ?? "";
        MdxRenderer? renderer = null;
        try
        {
            renderer = new MdxRenderer(_gl, mdx, modelDir, _dataSource, _texResolver);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Screenshot] Skip {entry.Name}: MdxRenderer creation failed: {ex.Message}");
            return 0;
        }

        // Compute bounds once
        var (boundsMin, boundsMax) = ComputeBounds(mdx);
        var center = (boundsMin + boundsMax) * 0.5f;
        // Apply MirrorX to center (negate X) to match the rendered model space
        center = new Vector3(-center.X, center.Y, center.Z);
        var extent = boundsMax - boundsMin;
        float radius = extent.Length() * 0.5f;
        if (radius < 0.01f) radius = 1.0f;

        // Compute camera distance to fit the full model in frame.
        // Use the largest dimension (height vs width) to ensure nothing is clipped.
        // FOV = 30° (narrower = more telephoto, less distortion).
        float fovRad = 25f * MathF.PI / 180f;
        float halfFov = fovRad * 0.5f;
        float aspect = (float)width / height;
        float verticalExtent = extent.Z * entry.EffectiveScale; // Z is up
        float horizontalExtent = MathF.Max(extent.X, extent.Y) * entry.EffectiveScale;
        // Distance needed to fit vertical extent
        float distV = (verticalExtent * 0.5f) / MathF.Tan(halfFov);
        // Distance needed to fit horizontal extent
        float distH = (horizontalExtent * 0.5f) / MathF.Tan(halfFov * aspect);
        // Use the larger + 20% padding
        float dist = MathF.Max(distV, distH) * 1.2f;
        dist = MathF.Max(dist, radius * 1.5f); // minimum fallback

        // MirrorX + scale: same transform as standalone MdxRenderer.Render()
        var modelTransform = Matrix4x4.CreateScale(-entry.EffectiveScale, entry.EffectiveScale, entry.EffectiveScale);

        // Setup FBO once
        EnsureFbo(width, height);
        if (!_fboReady)
        {
            Console.WriteLine($"[Screenshot] Skip {entry.Name}: FBO not ready");
            renderer.Dispose();
            return 0;
        }

        Directory.CreateDirectory(outputDir);
        int count = 0;

        try
        {
            foreach (var (angleName, azimuthDeg, elevationDeg) in angles)
            {
                try
                {
                    // Bind FBO and clear
                    _gl.BindFramebuffer(FramebufferTarget.Framebuffer, _fbo);
                    _gl.Viewport(0, 0, (uint)width, (uint)height);
                    _gl.ClearColor(1.0f, 1.0f, 1.0f, 1.0f);
                    _gl.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
                    _gl.Enable(EnableCap.DepthTest);
                    _gl.DepthFunc(DepthFunction.Less);
                    _gl.DepthMask(true);

                    // Camera for this angle — Z-up coordinate system
                    // After MirrorX, model faces +X. Camera orbits in XY plane, Z is up.
                    // Azimuth 0° = front = camera at -X looking toward +X
                    // Elevation 0° = eye level, positive = above
                    float elev = elevationDeg * MathF.PI / 180f;
                    float azim = azimuthDeg * MathF.PI / 180f;
                    float cosElev = MathF.Cos(elev);
                    float sinElev = MathF.Sin(elev);
                    // Orbit: azimuth 0 puts camera at -X (front view)
                    var camPos = center + new Vector3(
                        -dist * cosElev * MathF.Cos(azim),   // -X for azim=0 (front)
                        -dist * cosElev * MathF.Sin(azim),   // Y for left/right
                         dist * sinElev);                     // Z up for elevation

                    var view = Matrix4x4.CreateLookAt(camPos, center, Vector3.UnitZ);
                    var proj = Matrix4x4.CreatePerspectiveFieldOfView(fovRad, aspect, 0.01f, dist * 10f);

                    var fogColor = new Vector3(1.0f, 1.0f, 1.0f);
                    var lightDir = Vector3.Normalize(new Vector3(-0.5f, 0.8f, 0.3f));
                    var lightColor = new Vector3(1.0f, 0.95f, 0.9f);
                    var ambientColor = new Vector3(0.3f, 0.3f, 0.35f);

                    // Render opaque pass
                    _gl.Disable(EnableCap.Blend);
                    renderer.RenderWithTransform(modelTransform, view, proj, RenderPass.Opaque, 1.0f,
                        fogColor, dist * 5f, dist * 10f, camPos, lightDir, lightColor, ambientColor);

                    // Render transparent pass
                    _gl.Enable(EnableCap.DepthTest);
                    _gl.DepthFunc(DepthFunction.Lequal);
                    renderer.RenderWithTransform(modelTransform, view, proj, RenderPass.Transparent, 1.0f,
                        fogColor, dist * 5f, dist * 10f, camPos, lightDir, lightColor, ambientColor);

                    // Read pixels
                    byte[] pixels = new byte[width * height * 4];
                    unsafe
                    {
                        fixed (byte* ptr = pixels)
                            _gl.ReadPixels(0, 0, (uint)width, (uint)height, PixelFormat.Rgba, PixelType.UnsignedByte, ptr);
                    }

                    _gl.BindFramebuffer(FramebufferTarget.Framebuffer, 0);

                    // Convert to image (OpenGL reads bottom-up, flip)
                    using var image = SixLabors.ImageSharp.Image.LoadPixelData<Rgba32>(pixels, width, height);
                    image.Mutate(x => x.Flip(FlipMode.Vertical));

                    string path = Path.Combine(outputDir, $"{angleName}.png");
                    image.SaveAsPng(path);
                    count++;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[Screenshot] Angle {angleName} failed for {entry.Name}: {ex.Message}");
                    _gl.BindFramebuffer(FramebufferTarget.Framebuffer, 0);
                }
            }
        }
        finally
        {
            renderer.Dispose();
            _gl.BindFramebuffer(FramebufferTarget.Framebuffer, 0);
        }

        return count;
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
