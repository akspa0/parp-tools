using System.Numerics;
using MdxLTool.Formats.Mdx;
using MdxViewer.DataSources;
using MdxViewer.Logging;
using MdxViewer.Rendering;
using Silk.NET.OpenGL;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using WoWMapConverter.Core.Converters;

namespace MdxViewer.Catalog;

/// <summary>
/// Renders a single MDX or WMO model to an offscreen framebuffer, composites a nameplate,
/// and saves the result as a PNG screenshot. Supports perspective multi-angle and orthographic
/// top-down (roof) capture modes.
/// </summary>
public class ScreenshotRenderer : IDisposable
{
    private readonly GL _gl;
    private readonly IDataSource? _dataSource;
    private readonly ReplaceableTextureResolver? _texResolver;
    private readonly string? _dbcBuild;

    private uint _fbo;
    private uint _colorTex;
    private uint _depthRbo;
    private int _width;
    private int _height;
    private bool _fboReady;

    public ScreenshotRenderer(GL gl, IDataSource? dataSource, ReplaceableTextureResolver? texResolver = null, string? dbcBuild = null)
    {
        _gl = gl;
        _dataSource = dataSource;
        _texResolver = texResolver;
        _dbcBuild = dbcBuild;
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
            ViewerLog.Trace($"[Screenshot] Skip {entry.Name}: no data source or model path");
            return 0;
        }

        if (entry.IsWmo)
        {
            return CaptureWmoMultiAngle(entry, outputDir, width, height, resolvedModelPath);
        }

        return CaptureMdxMultiAngle(entry, outputDir, width, height, resolvedModelPath);
    }

    /// <summary>
    /// Capture a single top-down orthographic image of an asset (MDX or WMO) with no terrain,
    /// black background, and the model filling the frame. Intended for roof exemplar capture.
    /// Saves to {outputDir}/roof_topdown.png and returns the output path, or null on failure.
    /// </summary>
    public string? CaptureRoofTopDown(AssetCatalogEntry entry, string outputDir, int width = 512, int height = 512,
        string? resolvedModelPath = null)
    {
        string modelPath = resolvedModelPath ?? entry.ModelPath ?? "";
        if (_dataSource == null || string.IsNullOrEmpty(modelPath))
        {
            ViewerLog.Trace($"[Screenshot] Skip roof capture {entry.Name}: no data source or model path");
            return null;
        }

        if (entry.IsWmo)
        {
            return CaptureWmoRoofTopDownByPath(modelPath, outputDir, width, height);
        }

        return CaptureMdxRoofTopDown(entry, outputDir, width, height, resolvedModelPath);
    }

    /// <summary>
    /// Capture a roof-top-down image for any supported model type (WMO or M2) by path.
    /// </summary>
    public string? CapturePathByExtension(string modelPath, string outputDir, int width = 512, int height = 512)
    {
        string ext = Path.GetExtension(modelPath).ToLowerInvariant();
        if (ext == ".wmo")
            return CaptureWmoRoofTopDownByPath(modelPath, outputDir, width, height);
        if (ext is ".m2" or ".mdx")
            return CaptureMdxRoofTopDownByPath(modelPath, outputDir, width, height);
        return null;
    }

    /// <summary>
    /// Capture all-angle images for any supported model type (WMO or M2) by path.
    /// </summary>
    public string? CaptureAllAnglesByPath(string modelPath, string outputDir, int width = 512, int height = 512)
    {
        string ext = Path.GetExtension(modelPath).ToLowerInvariant();
        if (ext == ".wmo")
            return CaptureWmoAllAngles(modelPath, outputDir, width, height);
        if (ext is ".m2" or ".mdx")
            return CaptureMdxAllAngles(modelPath, outputDir, width, height);
        return null;
    }

    /// <summary>
    /// Read model bytes with MDX/M2 extension fallback for 3.x+ clients
    /// where the listfile uses .mdx but files are stored as .m2.
    /// </summary>
    private byte[]? ReadModelData(string path)
    {
        string normalized = path.Replace('/', '\\');
        byte[]? data = _dataSource?.ReadFile(normalized);
        if (data != null) return data;

        // Try swapping extension: .mdx <-> .m2
        string ext = Path.GetExtension(normalized).ToLowerInvariant();
        if (ext is ".mdx" or ".m2")
        {
            string swapped = normalized[..^ext.Length] + (ext == ".mdx" ? ".m2" : ".mdx");
            ViewerLog.Trace($"[Screenshot] File not found as '{normalized}', trying '{swapped}'");
            data = _dataSource?.ReadFile(swapped);
            if (data != null) return data;
        }

        return null;
    }

    /// <summary>
    /// Capture top-down orthographic + all angles for an M2/MDX model by path (no AssetCatalogEntry).
    /// </summary>
    public string? CaptureMdxAllAngles(string modelPath, string outputDir, int width = 512, int height = 512)
    {
        byte[]? mdxData = ReadModelData(modelPath);
        if (mdxData == null) return null;

        MdxFile mdx;
        try { using var ms = new MemoryStream(mdxData); using var br = new BinaryReader(ms); mdx = MdxFile.Load(br); }
        catch { return null; }
        if (mdx.Geosets.Count == 0) return null;

        string modelDir = Path.GetDirectoryName(modelPath.Replace('/', '\\')) ?? "";
        MdxRenderer? renderer = null;
        try { renderer = new MdxRenderer(_gl, mdx, modelDir, _dataSource, _texResolver); }
        catch { return null; }

        try
        {
            // Animation disabled for static roof capture
            try { renderer.UpdateAnimation(); } catch { /* static pose */ }

            // Roof top-down
            var (boundsMin, boundsMax) = ComputeBounds(mdx);
            var modelTransform = Matrix4x4.CreateScale(-1f, 1f, 1f);
            string? roofPath = RenderMdxRoofTopDown(renderer, boundsMin, boundsMax, modelTransform, outputDir, width, height);

            // Multi-angle
            RenderMdxMultiAngle(renderer, boundsMin, boundsMax, modelTransform, outputDir, width, height);

            return roofPath;
        }
        finally { renderer.Dispose(); }
    }

    public string? CaptureMdxRoofTopDownByPath(string modelPath, string outputDir, int width = 512, int height = 512)
    {
        byte[]? mdxData = ReadModelData(modelPath);
        if (mdxData == null) return null;

        MdxFile mdx;
        try { using var ms = new MemoryStream(mdxData); using var br = new BinaryReader(ms); mdx = MdxFile.Load(br); }
        catch { return null; }
        if (mdx.Geosets.Count == 0) return null;

        string modelDir = Path.GetDirectoryName(modelPath.Replace('/', '\\')) ?? "";
        MdxRenderer? renderer = null;
        try { renderer = new MdxRenderer(_gl, mdx, modelDir, _dataSource, _texResolver); }
        catch { return null; }

        try
        {
            var (boundsMin, boundsMax) = ComputeBounds(mdx);
            var modelTransform = Matrix4x4.CreateScale(-1f, 1f, 1f);
            return RenderMdxRoofTopDown(renderer, boundsMin, boundsMax, modelTransform, outputDir, width, height);
        }
        finally { renderer.Dispose(); }
    }

    private void RenderMdxMultiAngle(MdxRenderer renderer, Vector3 boundsMin, Vector3 boundsMax,
        Matrix4x4 modelTransform, string outputDir, int width, int height)
    {
        var center = (boundsMin + boundsMax) * 0.5f;
        float radius = (boundsMax - boundsMin).Length() * 0.5f;
        if (radius < 0.01f) radius = 1.0f;

        EnsureFbo(width, height);
        if (!_fboReady) return;

        float fovRad = 25f * MathF.PI / 180f;
        float aspect = (float)width / height;
        var corners = ComputeBoundsCorners(boundsMin, boundsMax);

        foreach (var (angleName, azimuthDeg, elevationDeg) in CameraAngles)
        {
            try
            {
                float elev = elevationDeg * MathF.PI / 180f;
                float azim = azimuthDeg * MathF.PI / 180f;
                var camDir = Vector3.Normalize(new Vector3(MathF.Cos(elev) * MathF.Cos(azim), MathF.Cos(elev) * MathF.Sin(azim), -MathF.Sin(elev)));
                var up = MathF.Abs(Vector3.Dot(camDir, Vector3.UnitZ)) > 0.99f ? Vector3.UnitX : Vector3.UnitZ;
                var tmpView = Matrix4x4.CreateLookAt(center - camDir * radius, center, up);
                float halfFovV = fovRad * 0.5f;
                float halfFovH = MathF.Atan(MathF.Tan(halfFovV) * aspect);
                float maxDist = radius;
                for (int ci = 0; ci < 8; ci++)
                {
                    var viewPos = Vector3.Transform(corners[ci], tmpView);
                    float depth = -viewPos.Z;
                    float needV = MathF.Abs(viewPos.Y) / MathF.Tan(halfFovV) + depth;
                    float needH = MathF.Abs(viewPos.X) / MathF.Tan(halfFovH) + depth;
                    maxDist = MathF.Max(maxDist, MathF.Max(needV, needH));
                }
                float dist = maxDist * 1.15f;
                var camPos = center - camDir * dist;
                var view = Matrix4x4.CreateLookAt(camPos, center, up);
                var proj = Matrix4x4.CreatePerspectiveFieldOfView(fovRad, aspect, 0.01f, dist * 10f);

                _gl.BindFramebuffer(FramebufferTarget.Framebuffer, _fbo);
                _gl.Viewport(0, 0, (uint)width, (uint)height);
                _gl.ClearColor(1.0f, 1.0f, 1.0f, 1.0f);
                _gl.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
                _gl.Enable(EnableCap.DepthTest);
                _gl.DepthFunc(DepthFunction.Less);
                _gl.DepthMask(true);

                var fogColor = new Vector3(1.0f, 1.0f, 1.0f);
                var lightDir = Vector3.Normalize(new Vector3(-0.5f, 0.8f, 0.3f));
                var lightColor = new Vector3(1.0f, 0.95f, 0.9f);
                var ambientColor = new Vector3(0.3f, 0.3f, 0.35f);

                _gl.Disable(EnableCap.Blend);
                renderer.RenderWithTransform(modelTransform, view, proj, RenderPass.Opaque, 1.0f,
                    fogColor, dist * 5f, dist * 10f, camPos, lightDir, lightColor, ambientColor);
                _gl.Enable(EnableCap.DepthTest);
                _gl.DepthFunc(DepthFunction.Lequal);
                _gl.DepthMask(false);
                renderer.RenderWithTransform(modelTransform, view, proj, RenderPass.Transparent, 1.0f,
                    fogColor, dist * 5f, dist * 10f, camPos, lightDir, lightColor, ambientColor);
                _gl.DepthMask(true);

                byte[] pixels = new byte[width * height * 4];
                unsafe { fixed (byte* ptr = pixels) _gl.ReadPixels(0, 0, (uint)width, (uint)height, PixelFormat.Rgba, PixelType.UnsignedByte, ptr); }
                _gl.BindFramebuffer(FramebufferTarget.Framebuffer, 0);

                using var image = SixLabors.ImageSharp.Image.LoadPixelData<Rgba32>(pixels, width, height);
                image.Mutate(x => x.Flip(FlipMode.Vertical));
                image.SaveAsJpeg(Path.Combine(outputDir, $"{angleName}.jpg"), new SixLabors.ImageSharp.Formats.Jpeg.JpegEncoder { Quality = 99 });
            }
            catch (Exception ex)
            {
                ViewerLog.Trace($"[Screenshot] M2 angle {angleName} failed: {ex.Message}");
                _gl.BindFramebuffer(FramebufferTarget.Framebuffer, 0);
            }
        }
    }

    private string? RenderMdxRoofTopDown(MdxRenderer renderer, Vector3 boundsMin, Vector3 boundsMax,
        Matrix4x4 modelTransform, string outputDir, int width, int height)
    {
        var corners = ComputeBoundsCorners(boundsMin, boundsMax);
        for (int ci = 0; ci < 8; ci++)
            corners[ci] = Vector3.Transform(corners[ci], modelTransform);

        var wsMin = corners[0]; var wsMax = corners[0];
        for (int ci = 1; ci < 8; ci++) { wsMin = Vector3.Min(wsMin, corners[ci]); wsMax = Vector3.Max(wsMax, corners[ci]); }

        float spanX = wsMax.X - wsMin.X; if (spanX < 0.01f) spanX = 1.0f;
        float spanY = wsMax.Y - wsMin.Y; if (spanY < 0.01f) spanY = 1.0f;
        float spanZ = wsMax.Z - wsMin.Z;
        var center = (wsMin + wsMax) * 0.5f;

        float padFactor = 1.1f;
        float orthoHalfX = spanX * padFactor * 0.5f;
        float orthoHalfY = spanY * padFactor * 0.5f;
        float aspect = (float)width / height;
        if (aspect > 1.0f) orthoHalfX = MathF.Max(orthoHalfX, orthoHalfY * aspect);
        else orthoHalfY = MathF.Max(orthoHalfY, orthoHalfX / aspect);

        float eyeHeight = spanZ + 100.0f;
        var eyePos = new Vector3(center.X, center.Y, center.Z + eyeHeight);
        var view = Matrix4x4.CreateLookAt(eyePos, center, Vector3.UnitX);
        var proj = Matrix4x4.CreateOrthographic(orthoHalfX * 2, orthoHalfY * 2, 0.1f, eyeHeight * 2 + spanZ + 200f);

        EnsureFbo(width, height);
        if (!_fboReady) return null;
        Directory.CreateDirectory(outputDir);

        _gl.BindFramebuffer(FramebufferTarget.Framebuffer, _fbo);
        _gl.Viewport(0, 0, (uint)width, (uint)height);
        _gl.ClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        _gl.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Less);
        _gl.DepthMask(true);

        var fogColor = new Vector3(0.0f, 0.0f, 0.0f);
        var lightDir = Vector3.Normalize(new Vector3(0.3f, -0.2f, -1.0f));
        var lightColor = new Vector3(1.0f, 0.97f, 0.92f);
        var ambientColor = new Vector3(0.45f, 0.45f, 0.5f);

        _gl.Disable(EnableCap.Blend);
        renderer.RenderWithTransform(modelTransform, view, proj, RenderPass.Opaque, 1.0f,
            fogColor, eyeHeight * 2, eyeHeight * 2 + spanZ + 200f, eyePos, lightDir, lightColor, ambientColor);
        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Lequal);
        renderer.RenderWithTransform(modelTransform, view, proj, RenderPass.Transparent, 1.0f,
            fogColor, eyeHeight * 2, eyeHeight * 2 + spanZ + 200f, eyePos, lightDir, lightColor, ambientColor);

        return ReadPixelsAndSave(width, height, outputDir);
    }

    private int CaptureWmoMultiAngle(AssetCatalogEntry entry, string outputDir, int width, int height,
        string? resolvedModelPath)
    {
        string modelPath = resolvedModelPath ?? entry.ModelPath ?? "";
        WmoV14ToV17Converter.WmoV14Data? wmo = LoadWmoData(modelPath);
        if (wmo == null) return 0;

        string modelDir = Path.GetDirectoryName(modelPath)?.Replace('/', '\\') ?? "";
        WmoRenderer? renderer = null;
        try
        {
            renderer = new WmoRenderer(_gl, wmo, modelDir, _dataSource, _texResolver, _dbcBuild,
                deferInitialDoodadLoads: true,
                deferInitialMaterialTextureLoads: false,
                enableRuntimeGroupVisibility: false);
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[Screenshot] Skip {entry.Name}: WmoRenderer creation failed: {ex.Message}");
            return 0;
        }

        var boundsMin = wmo.BoundsMin;
        var boundsMax = wmo.BoundsMax;
        var center = (boundsMin + boundsMax) * 0.5f;
        float radius = (boundsMax - boundsMin).Length() * 0.5f;
        if (radius < 0.01f) radius = 1.0f;

        EnsureFbo(width, height);
        if (!_fboReady) { renderer.Dispose(); return 0; }

        Directory.CreateDirectory(outputDir);
        int count = 0;
        float fovRad = 25f * MathF.PI / 180f;
        float aspect = (float)width / height;

        try
        {
            foreach (var (angleName, azimuthDeg, elevationDeg) in CameraAngles)
            {
                try
                {
                    _gl.BindFramebuffer(FramebufferTarget.Framebuffer, _fbo);
                    _gl.Viewport(0, 0, (uint)width, (uint)height);
                    _gl.ClearColor(1.0f, 1.0f, 1.0f, 1.0f);
                    _gl.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
                    _gl.Enable(EnableCap.DepthTest);
                    _gl.DepthFunc(DepthFunction.Less);
                    _gl.DepthMask(true);

                    float elev = elevationDeg * MathF.PI / 180f;
                    float azim = azimuthDeg * MathF.PI / 180f;
                    float cosElev = MathF.Cos(elev);
                    float sinElev = MathF.Sin(elev);
                    var camDir = Vector3.Normalize(new Vector3(cosElev * MathF.Cos(azim), cosElev * MathF.Sin(azim), -sinElev));
                    var up = MathF.Abs(Vector3.Dot(camDir, Vector3.UnitZ)) > 0.99f ? Vector3.UnitX : Vector3.UnitZ;
                    var tmpView = Matrix4x4.CreateLookAt(center - camDir * radius, center, up);
                    float halfFovV = fovRad * 0.5f;
                    float halfFovH = MathF.Atan(MathF.Tan(halfFovV) * aspect);
                    float maxDist = radius;
                    var corners = ComputeBoundsCorners(boundsMin, boundsMax);
                    for (int ci = 0; ci < 8; ci++)
                    {
                        var viewPos = Vector3.Transform(corners[ci], tmpView);
                        float depth = -viewPos.Z;
                        float needV = MathF.Abs(viewPos.Y) / MathF.Tan(halfFovV) + depth;
                        float needH = MathF.Abs(viewPos.X) / MathF.Tan(halfFovH) + depth;
                        maxDist = MathF.Max(maxDist, MathF.Max(needV, needH));
                    }
                    float dist = maxDist * 1.15f;
                    var camPos = center - camDir * dist;
                    var view = Matrix4x4.CreateLookAt(camPos, center, up);
                    var proj = Matrix4x4.CreatePerspectiveFieldOfView(fovRad, aspect, 0.01f, dist * 10f);

                    var fogColor = new Vector3(1.0f, 1.0f, 1.0f);
                    var lightDir = Vector3.Normalize(new Vector3(-0.5f, 0.8f, 0.3f));
                    var lightColor = new Vector3(1.0f, 0.95f, 0.9f);
                    var ambientColor = new Vector3(0.3f, 0.3f, 0.35f);

                    _gl.Disable(EnableCap.Blend);
                    renderer.RenderWithTransform(Matrix4x4.Identity, view, proj, WmoRenderPass.Opaque,
                        fogColor, dist * 5f, dist * 10f, camPos, lightDir, lightColor, ambientColor);
                    _gl.Enable(EnableCap.DepthTest);
                    _gl.DepthFunc(DepthFunction.Lequal);
                    renderer.RenderWithTransform(Matrix4x4.Identity, view, proj, WmoRenderPass.Transparent,
                        fogColor, dist * 5f, dist * 10f, camPos, lightDir, lightColor, ambientColor);

                    byte[] pixels = new byte[width * height * 4];
                    unsafe { fixed (byte* ptr = pixels) _gl.ReadPixels(0, 0, (uint)width, (uint)height, PixelFormat.Rgba, PixelType.UnsignedByte, ptr); }
                    ForceOpaqueAlpha(pixels);
                    _gl.BindFramebuffer(FramebufferTarget.Framebuffer, 0);

                    using var image = SixLabors.ImageSharp.Image.LoadPixelData<Rgba32>(pixels, width, height);
                    image.Mutate(x => x.Flip(FlipMode.Vertical));
                    image.SaveAsJpeg(Path.Combine(outputDir, $"{angleName}.jpg"), new SixLabors.ImageSharp.Formats.Jpeg.JpegEncoder { Quality = 99 });
                    count++;
                }
                catch (Exception ex)
                {
                    ViewerLog.Trace($"[Screenshot] WMO angle {angleName} failed for {entry.Name}: {ex.Message}");
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

/// <summary>
    /// Capture a WMO model from all predefined camera angles + orthographic top-down (roof).
    /// Renders each DoodadSet into a separate subfolder.
    /// Saves roof_topdown.png + {angleName}.jpg per angle. Returns the roof_topdown path or null.
    /// </summary>
    public string? CaptureWmoAllAngles(string modelPath, string outputDir, int width = 512, int height = 512)
    {
        ViewerLog.Info(ViewerLog.Category.Wmo, $"[RoofCapture] Loading {modelPath}");
        WmoV14ToV17Converter.WmoV14Data? wmo = LoadWmoData(modelPath);
        if (wmo == null)
        {
            ViewerLog.Info(ViewerLog.Category.Wmo, $"[RoofCapture] FAILED to load {modelPath}");
            return null;
        }

        string modelDir = Path.GetDirectoryName(modelPath)?.Replace('/', '\\') ?? "";
        WmoRenderer? renderer = null;
        try
        {
            renderer = new WmoRenderer(_gl, wmo, modelDir, _dataSource, _texResolver, _dbcBuild,
                deferInitialDoodadLoads: false, deferInitialMaterialTextureLoads: false, enableRuntimeGroupVisibility: false);
        }
        catch (Exception ex) { ViewerLog.Trace($"[Screenshot] WmoRenderer creation failed for {modelPath}: {ex.Message}"); return null; }

        try
        {
            int doodadSetCount = renderer.DoodadSetCount;
            string? firstRoofPath = null;

            for (int ds = 0; ds < doodadSetCount; ds++)
            {
                string dsDir;
                if (doodadSetCount <= 1)
                {
                    dsDir = outputDir;
                }
                else
                {
                    renderer.SetActiveDoodadSet(ds);
                    string dsName = SanitizePathComponent(renderer.GetDoodadSetName(ds));
                    dsDir = Path.Combine(outputDir, $"doodadset_{ds}_{dsName}");
                }

                string? roofPath = RenderWmoRoofTopDown(renderer, wmo, dsDir, width, height);
                RenderWmoMultiAngle(renderer, wmo, modelPath, dsDir, width, height);
                firstRoofPath ??= roofPath;
            }

            return firstRoofPath;
        }
        finally { renderer.Dispose(); }
    }

    public string? CaptureWmoRoofTopDownByPath(string modelPath, string outputDir, int width = 512, int height = 512)
    {
        WmoV14ToV17Converter.WmoV14Data? wmo = LoadWmoData(modelPath);
        if (wmo == null) return null;

        string modelDir = Path.GetDirectoryName(modelPath)?.Replace('/', '\\') ?? "";
        WmoRenderer? renderer = null;
        try
        {
            renderer = new WmoRenderer(_gl, wmo, modelDir, _dataSource, _texResolver, _dbcBuild,
                deferInitialDoodadLoads: false, deferInitialMaterialTextureLoads: false, enableRuntimeGroupVisibility: false);
        }
        catch (Exception ex) { ViewerLog.Trace($"[Screenshot] WmoRenderer creation failed for {modelPath}: {ex.Message}"); return null; }

        try
        {
            int doodadSetCount = renderer.DoodadSetCount;
            string? firstRoofPath = null;

            for (int ds = 0; ds < doodadSetCount; ds++)
            {
                string dsDir;
                if (doodadSetCount <= 1)
                {
                    dsDir = outputDir;
                }
                else
                {
                    renderer.SetActiveDoodadSet(ds);
                    string dsName = SanitizePathComponent(renderer.GetDoodadSetName(ds));
                    dsDir = Path.Combine(outputDir, $"doodadset_{ds}_{dsName}");
                }

                string? roofPath = RenderWmoRoofTopDown(renderer, wmo, dsDir, width, height);
                firstRoofPath ??= roofPath;
            }

            return firstRoofPath;
        }
        finally { renderer.Dispose(); }
    }

        private void RenderWmoMultiAngle(WmoRenderer renderer, WmoV14ToV17Converter.WmoV14Data wmo,
        string modelPath, string outputDir, int width, int height)
    {
        var boundsMin = wmo.BoundsMin;
        var boundsMax = wmo.BoundsMax;
        var center = (boundsMin + boundsMax) * 0.5f;
        float radius = (boundsMax - boundsMin).Length() * 0.5f;
        if (radius < 0.01f) radius = 1.0f;

        EnsureFbo(width, height);
        if (!_fboReady) return;

        float fovRad = 25f * MathF.PI / 180f;
        float aspect = (float)width / height;

        var corners = ComputeBoundsCorners(boundsMin, boundsMax);
        foreach (var (angleName, azimuthDeg, elevationDeg) in CameraAngles)
        {
            try
            {
                float elev = elevationDeg * MathF.PI / 180f;
                float azim = azimuthDeg * MathF.PI / 180f;
                float cosElev = MathF.Cos(elev);
                float sinElev = MathF.Sin(elev);
                var camDir = Vector3.Normalize(new Vector3(cosElev * MathF.Cos(azim), cosElev * MathF.Sin(azim), -sinElev));
                var up = MathF.Abs(Vector3.Dot(camDir, Vector3.UnitZ)) > 0.99f ? Vector3.UnitX : Vector3.UnitZ;
                var tmpView = Matrix4x4.CreateLookAt(center - camDir * radius, center, up);
                float halfFovV = fovRad * 0.5f;
                float halfFovH = MathF.Atan(MathF.Tan(halfFovV) * aspect);
                float maxDist = radius;
                for (int ci = 0; ci < 8; ci++)
                {
                    var viewPos = Vector3.Transform(corners[ci], tmpView);
                    float depth = -viewPos.Z;
                    float needV = MathF.Abs(viewPos.Y) / MathF.Tan(halfFovV) + depth;
                    float needH = MathF.Abs(viewPos.X) / MathF.Tan(halfFovH) + depth;
                    maxDist = MathF.Max(maxDist, MathF.Max(needV, needH));
                }
                float dist = maxDist * 1.15f;
                var camPos = center - camDir * dist;
                var view = Matrix4x4.CreateLookAt(camPos, center, up);
                var proj = Matrix4x4.CreatePerspectiveFieldOfView(fovRad, aspect, 0.01f, dist * 10f);

                _gl.BindFramebuffer(FramebufferTarget.Framebuffer, _fbo);
                _gl.Viewport(0, 0, (uint)width, (uint)height);
                _gl.ClearColor(1.0f, 1.0f, 1.0f, 1.0f);
                _gl.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
                _gl.Enable(EnableCap.DepthTest);
                _gl.DepthFunc(DepthFunction.Less);
                _gl.DepthMask(true);

                var fogColor = new Vector3(1.0f, 1.0f, 1.0f);
                var lightDir = Vector3.Normalize(new Vector3(-0.5f, 0.8f, 0.3f));
                var lightColor = new Vector3(1.0f, 0.95f, 0.9f);
                var ambientColor = new Vector3(0.3f, 0.3f, 0.35f);

                _gl.Disable(EnableCap.Blend);
                renderer.RenderWithTransform(Matrix4x4.Identity, view, proj, WmoRenderPass.Opaque,
                    fogColor, dist * 5f, dist * 10f, camPos, lightDir, lightColor, ambientColor);
                _gl.Enable(EnableCap.DepthTest);
                _gl.DepthFunc(DepthFunction.Lequal);
                renderer.RenderWithTransform(Matrix4x4.Identity, view, proj, WmoRenderPass.Transparent,
                    fogColor, dist * 5f, dist * 10f, camPos, lightDir, lightColor, ambientColor);

                byte[] pixels = new byte[width * height * 4];
                unsafe { fixed (byte* ptr = pixels) _gl.ReadPixels(0, 0, (uint)width, (uint)height, PixelFormat.Rgba, PixelType.UnsignedByte, ptr); }
                _gl.BindFramebuffer(FramebufferTarget.Framebuffer, 0);

                using var image = SixLabors.ImageSharp.Image.LoadPixelData<Rgba32>(pixels, width, height);
                image.Mutate(x => x.Flip(FlipMode.Vertical));
                image.SaveAsJpeg(Path.Combine(outputDir, $"{angleName}.jpg"), new SixLabors.ImageSharp.Formats.Jpeg.JpegEncoder { Quality = 99 });
            }
            catch (Exception ex)
            {
                ViewerLog.Trace($"[Screenshot] WMO angle {angleName} failed: {ex.Message}");
                _gl.BindFramebuffer(FramebufferTarget.Framebuffer, 0);
            }
        }
    }

    /// <summary>
    /// Batch capture roof-only (top-down orthographic) renders for all WMO asset paths.
    /// Output: one subdirectory per asset with roof_topdown.png.
    /// </summary>
    public int CaptureBatchRoofTopDown(IReadOnlyList<string> modelPaths, string outputDir,
        int width = 512, int height = 512, Action<int, int, string>? onProgress = null)
    {
        Directory.CreateDirectory(outputDir);
        int success = 0;
        var metadata = new List<Dictionary<string, object>>();

        for (int i = 0; i < modelPaths.Count; i++)
        {
            string path = modelPaths[i];
            onProgress?.Invoke(i + 1, modelPaths.Count, Path.GetFileNameWithoutExtension(path));
            ViewerLog.Trace($"[RoofCapture] ({i + 1}/{modelPaths.Count}) {path}");

            string safeName = SanitizePathComponent(Path.GetFileNameWithoutExtension(path));
            string assetDir = Path.Combine(outputDir, safeName);
            Directory.CreateDirectory(assetDir);

            string? result = CapturePathByExtension(path, assetDir, width, height);
            if (result != null)
            {
                success++;
                metadata.Add(new Dictionary<string, object>
                {
                    ["asset_path"] = path,
                    ["roof_topdown"] = result,
                    ["success"] = true
                });
            }
            else
            {
                metadata.Add(new Dictionary<string, object>
                {
                    ["asset_path"] = path,
                    ["success"] = false
                });
            }

            if (i % 20 == 0)
                System.Threading.Thread.Yield();
        }

        string metaPath = Path.Combine(outputDir, "roof_capture_metadata.json");
        File.WriteAllText(metaPath,
            System.Text.Json.JsonSerializer.Serialize(new { captures = metadata }, new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
        return success;
    }

    /// <summary>
    /// Batch capture all-angle renders for all WMO asset paths.
    /// Output: one subdirectory per asset with roof_topdown.png + {angle}.png per angle.
    /// </summary>
    public int CaptureBatchAllAngles(IReadOnlyList<string> modelPaths, string outputDir,
        int width = 512, int height = 512, Action<int, int, string>? onProgress = null)
    {
        Directory.CreateDirectory(outputDir);
        int success = 0;
        var metadata = new List<Dictionary<string, object>>();

        for (int i = 0; i < modelPaths.Count; i++)
        {
            string path = modelPaths[i];
            onProgress?.Invoke(i + 1, modelPaths.Count, Path.GetFileNameWithoutExtension(path));
            ViewerLog.Trace($"[RoofCapture] ({i + 1}/{modelPaths.Count}) {path}");

            string safeName = SanitizePathComponent(Path.GetFileNameWithoutExtension(path));
            string assetDir = Path.Combine(outputDir, safeName);
            Directory.CreateDirectory(assetDir);

            string? result = CaptureAllAnglesByPath(path, assetDir, width, height);
            if (result != null)
            {
                success++;
                metadata.Add(new Dictionary<string, object>
                {
                    ["asset_path"] = path,
                    ["roof_topdown"] = result,
                    ["angles"] = CameraAngles.Select(a => a.name).ToList(),
                    ["success"] = true
                });
            }
            else
            {
                metadata.Add(new Dictionary<string, object>
                {
                    ["asset_path"] = path,
                    ["success"] = false
                });
            }

            if (i % 20 == 0)
                System.Threading.Thread.Yield();
        }

        string metaPath = Path.Combine(outputDir, "roof_capture_metadata.json");
        File.WriteAllText(metaPath,
            System.Text.Json.JsonSerializer.Serialize(new { captures = metadata }, new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
        return success;
    }

    private string? RenderWmoRoofTopDown(WmoRenderer renderer, WmoV14ToV17Converter.WmoV14Data wmo,
        string outputDir, int width, int height)
    {
        var boundsMin = wmo.BoundsMin;
        var boundsMax = wmo.BoundsMax;
        var center = (boundsMin + boundsMax) * 0.5f;
        float spanX = boundsMax.X - boundsMin.X; if (spanX < 0.01f) spanX = 1.0f;
        float spanY = boundsMax.Y - boundsMin.Y; if (spanY < 0.01f) spanY = 1.0f;
        float spanZ = boundsMax.Z - boundsMin.Z;

        float padFactor = 1.1f;
        float orthoHalfX = spanX * padFactor * 0.5f;
        float orthoHalfY = spanY * padFactor * 0.5f;
        float aspect = (float)width / height;
        if (aspect > 1.0f) orthoHalfX = MathF.Max(orthoHalfX, orthoHalfY * aspect);
        else orthoHalfY = MathF.Max(orthoHalfY, orthoHalfX / aspect);

        float eyeHeight = spanZ + 100.0f;
        var eyePos = new Vector3(center.X, center.Y, center.Z + eyeHeight);
        var view = Matrix4x4.CreateLookAt(eyePos, center, Vector3.UnitX);
        var proj = Matrix4x4.CreateOrthographic(orthoHalfX * 2, orthoHalfY * 2, 0.1f, eyeHeight * 2 + spanZ + 200f);

        EnsureFbo(width, height);
        if (!_fboReady) return null;
        Directory.CreateDirectory(outputDir);

        _gl.BindFramebuffer(FramebufferTarget.Framebuffer, _fbo);
        _gl.Viewport(0, 0, (uint)width, (uint)height);
        _gl.ClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        _gl.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Less);
        _gl.DepthMask(true);

        var fogColor = new Vector3(0.0f, 0.0f, 0.0f);
        var lightDir = Vector3.Normalize(new Vector3(0.3f, -0.2f, -1.0f));
        var lightColor = new Vector3(1.0f, 0.97f, 0.92f);
        var ambientColor = new Vector3(0.45f, 0.45f, 0.5f);

        _gl.Disable(EnableCap.Blend);
        renderer.RenderWithTransform(Matrix4x4.Identity, view, proj, WmoRenderPass.Opaque,
            fogColor, eyeHeight * 2, eyeHeight * 2 + spanZ + 200f, eyePos, lightDir, lightColor, ambientColor);
        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Lequal);
        renderer.RenderWithTransform(Matrix4x4.Identity, view, proj, WmoRenderPass.Transparent,
            fogColor, eyeHeight * 2, eyeHeight * 2 + spanZ + 200f, eyePos, lightDir, lightColor, ambientColor);

        return ReadPixelsAndSave(width, height, outputDir);
    }

    private string? CaptureMdxRoofTopDown(AssetCatalogEntry entry, string outputDir, int width, int height,
        string? resolvedModelPath)
    {
        string modelPath = resolvedModelPath ?? entry.ModelPath ?? "";
        byte[]? mdxData = ReadModelData(modelPath);
        if (mdxData == null) return null;

        MdxFile mdx;
        try { using var ms = new MemoryStream(mdxData); using var br = new BinaryReader(ms); mdx = MdxFile.Load(br); }
        catch (Exception ex) { ViewerLog.Trace($"[Screenshot] MDX parse failed for {modelPath}: {ex.Message}"); return null; }
        if (mdx.Geosets.Count == 0) return null;

        string modelDir = Path.GetDirectoryName(modelPath)?.Replace('/', '\\') ?? "";
        MdxRenderer? renderer = null;
        try
        {
            renderer = new MdxRenderer(_gl, mdx, modelDir, _dataSource, _texResolver,
                explicitTextureVariations: entry.TextureVariations);
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[Screenshot] Skip roof capture {entry.Name}: MdxRenderer creation failed: {ex.Message}");
            return null;
        }

        try
        {
            // Animation disabled for roof capture — static model only
            try { renderer.UpdateAnimation(); } catch { /* static bind-pose fallback */ }

            var (boundsMin, boundsMax) = ComputeBounds(mdx);
            float scale = entry.EffectiveScale;
            var modelTransform = Matrix4x4.CreateScale(-scale, scale, scale);

            var corners = new Vector3[8];
            for (int ci = 0; ci < 8; ci++)
                corners[ci] = Vector3.Transform(new Vector3(
                    (ci & 1) == 0 ? boundsMin.X : boundsMax.X,
                    (ci & 2) == 0 ? boundsMin.Y : boundsMax.Y,
                    (ci & 4) == 0 ? boundsMin.Z : boundsMax.Z), modelTransform);

            var wsMin = corners[0]; var wsMax = corners[0];
            for (int ci = 1; ci < 8; ci++) { wsMin = Vector3.Min(wsMin, corners[ci]); wsMax = Vector3.Max(wsMax, corners[ci]); }

            float spanX = wsMax.X - wsMin.X; if (spanX < 0.01f) spanX = 1.0f;
            float spanY = wsMax.Y - wsMin.Y; if (spanY < 0.01f) spanY = 1.0f;
            float spanZ = wsMax.Z - wsMin.Z;
            var center = (wsMin + wsMax) * 0.5f;

            float padFactor = 1.1f;
            float orthoHalfX = spanX * padFactor * 0.5f;
            float orthoHalfY = spanY * padFactor * 0.5f;
            float aspect = (float)width / height;
            if (aspect > 1.0f) orthoHalfX = MathF.Max(orthoHalfX, orthoHalfY * aspect);
            else orthoHalfY = MathF.Max(orthoHalfY, orthoHalfX / aspect);

            float eyeHeight = spanZ + 100.0f;
            var eyePos = new Vector3(center.X, center.Y, center.Z + eyeHeight);
            var view = Matrix4x4.CreateLookAt(eyePos, center, Vector3.UnitX);
            var proj = Matrix4x4.CreateOrthographic(orthoHalfX * 2, orthoHalfY * 2, 0.1f, eyeHeight * 2 + spanZ + 200f);

            EnsureFbo(width, height);
            if (!_fboReady) return null;
            Directory.CreateDirectory(outputDir);

            // FBO restore: always bind + clear before rendering
            _gl.BindFramebuffer(FramebufferTarget.Framebuffer, _fbo);
            _gl.Viewport(0, 0, (uint)width, (uint)height);
            _gl.ClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            _gl.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
            _gl.Enable(EnableCap.DepthTest);
            _gl.DepthFunc(DepthFunction.Less);
            _gl.DepthMask(true);

            var fogColor = new Vector3(0.0f, 0.0f, 0.0f);
            var lightDir = Vector3.Normalize(new Vector3(0.3f, -0.2f, -1.0f));
            var lightColor = new Vector3(1.0f, 0.97f, 0.92f);
            var ambientColor = new Vector3(0.45f, 0.45f, 0.5f);

            // Roof pass: render ALL geometry as opaque to avoid foliage-through-trunk issue
            // In top-down view, leaves above trunks render wrong with transparent pass
            _gl.Disable(EnableCap.Blend);
            renderer.RenderWithTransform(modelTransform, view, proj, RenderPass.Opaque, 1.0f,
                fogColor, eyeHeight * 2, eyeHeight * 2 + spanZ + 200f, eyePos, lightDir, lightColor, ambientColor);
            // Skip transparent pass in roof capture mode entirely

            return ReadPixelsAndSave(width, height, outputDir);
        }
        finally
        {
            renderer.Dispose();
        }
    }

private WmoV14ToV17Converter.WmoV14Data? LoadWmoData(string modelPath)
    {
        if (_dataSource == null)
        {
            Console.Error.WriteLine($"[RoofCapture] No data source");
            return null;
        }
string normalized = modelPath.Replace('/', '\\');
        Console.Error.WriteLine($"[RoofCapture] Reading {normalized}");
        byte[]? data = _dataSource.ReadFile(normalized);
        if (data == null || data.Length == 0)
        {
            Console.Error.WriteLine($"[RoofCapture] File not found: {normalized}");
            return null;
        }
        Console.Error.WriteLine($"[RoofCapture] Read {data.Length} bytes from {normalized}");

        string modelDir = Path.GetDirectoryName(normalized) ?? "";

        // Try v17 first (current retail)
        var v17Parser = new WmoV17ToV14Converter();
        var groupBytes = LoadWmoGroupBytes(normalized);
        Console.Error.WriteLine($"[RoofCapture] Loaded {groupBytes.Count} group files, trying v17 parse");
        try
        {
            var wmo = v17Parser.ParseV17ToModel(data, groupBytes);
            Console.Error.WriteLine($"[RoofCapture] v17 parse succeeded: {wmo.Groups.Count} groups, bounds=({wmo.BoundsMin},{wmo.BoundsMax})");
            if (wmo.Groups.Count > 0)
                return wmo;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[RoofCapture] v17 parse failed: {ex.Message}, trying v14");
        }

        // Fall back to v14
        string tmpPath = Path.Combine(Path.GetTempPath(), $"wmo_roof_{Guid.NewGuid():N}.tmp");
        try
        {
            File.WriteAllBytes(tmpPath, data);
            var converter = new WmoV14ToV17Converter();
            WmoV14ToV17Converter.WmoV14Data wmo = converter.ParseWmoV14(tmpPath);
            Console.Error.WriteLine($"[RoofCapture] Parsed v14 WMO: {wmo.Groups.Count} groups, {wmo.GroupCount} total, bounds=({wmo.BoundsMin},{wmo.BoundsMax})");

            if (wmo.Groups.Count == 0 && wmo.GroupCount > 0)
            {
                string wmoBase = Path.GetFileNameWithoutExtension(normalized);
                for (int gi = 0; gi < wmo.GroupCount; gi++)
                {
                    string groupName = $"{wmoBase}_{gi:D3}.wmo";
                    string groupPath = string.IsNullOrEmpty(modelDir) ? groupName : $"{modelDir}\\{groupName}";
                    byte[]? groupBytesFile = _dataSource.ReadFile(groupPath);
                    if (groupBytesFile != null && groupBytesFile.Length > 0)
                    {
                        Console.Error.WriteLine($"[RoofCapture] Loading group {gi}: {groupPath} ({groupBytesFile.Length} bytes)");
                        converter.ParseGroupFile(groupBytesFile, wmo, gi);
                    }
                    else
                    {
                        Console.Error.WriteLine($"[RoofCapture] Group {gi} not found: {groupPath}");
                    }
                }
            }
            Console.Error.WriteLine($"[RoofCapture] Final WMO: {wmo.Groups.Count} groups");
            return wmo;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[RoofCapture] v14 parse also failed: {ex.Message}");
            ViewerLog.Trace($"[Screenshot] WMO parse failed for {normalized}: {ex.Message}");
            return null;
        }
        finally
        {
            try { File.Delete(tmpPath); } catch { }
        }
    }

    private List<byte[]> LoadWmoGroupBytes(string modelPath)
    {
        string directory = Path.GetDirectoryName(modelPath)?.Replace('/', '\\') ?? "";
        string baseName = Path.GetFileNameWithoutExtension(modelPath);
        var groupBytesList = new List<byte[]>();
        for (int gi = 0; gi < 512; gi++)
        {
            string groupName = $"{baseName}_{gi:D3}.wmo";
            string groupPath = string.IsNullOrEmpty(directory) ? groupName : $"{directory}\\{groupName}";
            byte[]? groupBytes = _dataSource?.ReadFile(groupPath);
            if (groupBytes == null || groupBytes.Length == 0) break;
            groupBytesList.Add(groupBytes);
        }
        return groupBytesList;
    }

    private string? ReadPixelsAndSave(int width, int height, string outputDir)
    {
        byte[] pixels = new byte[width * height * 4];
        unsafe { fixed (byte* ptr = pixels) _gl.ReadPixels(0, 0, (uint)width, (uint)height, PixelFormat.Rgba, PixelType.UnsignedByte, ptr); }

        for (int i = 0; i < pixels.Length; i += 4)
        {
            byte r = pixels[i], g = pixels[i + 1], b = pixels[i + 2];
            pixels[i + 3] = (byte)((r + g + b > 10) ? 255 : 0);
        }

        _gl.BindFramebuffer(FramebufferTarget.Framebuffer, 0);

        using var image = SixLabors.ImageSharp.Image.LoadPixelData<Rgba32>(pixels, width, height);
        image.Mutate(x => x.Flip(FlipMode.Vertical));
        string path = Path.Combine(outputDir, "roof_topdown.png");
        image.SaveAsPng(path);
        return path;
    }

    private static int DetectWmoVersion(byte[] data)
    {
        if (data.Length < 4) return 14;
        uint magic = (uint)(data[0] | (data[1] << 8) | (data[2] << 16) | (data[3] << 24));
        if (magic == 0x4F4D5756) return 17; // 'WMOV'
        return 14;
    }

    private static Vector3[] ComputeBoundsCorners(Vector3 min, Vector3 max)
    {
        var corners = new Vector3[8];
        for (int i = 0; i < 8; i++)
            corners[i] = new Vector3(
                (i & 1) == 0 ? min.X : max.X,
                (i & 2) == 0 ? min.Y : max.Y,
                (i & 4) == 0 ? min.Z : max.Z);
        return corners;
    }

    private int CaptureMdxMultiAngle(AssetCatalogEntry entry, string outputDir, int width, int height,
        string? resolvedModelPath, (string name, float azimuth, float elevation)[]? angles = null)
    {
        angles ??= CameraAngles;
        string modelPath = resolvedModelPath ?? entry.ModelPath ?? "";

        // Load MDX once
        byte[]? mdxData = _dataSource.ReadFile(modelPath);
        if (mdxData == null)
        {
            ViewerLog.Trace($"[Screenshot] Skip {entry.Name}: MDX file not found: {modelPath}");
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
            ViewerLog.Trace($"[Screenshot] Skip {entry.Name}: MDX parse failed: {ex.Message}");
            return 0;
        }

        if (mdx.Geosets.Count == 0)
        {
            ViewerLog.Trace($"[Screenshot] Skip {entry.Name}: no geosets");
            return 0;
        }

        // Create renderer once for all angles
        string modelDir = Path.GetDirectoryName(modelPath)?.Replace('/', '\\') ?? "";
        MdxRenderer? renderer = null;
        try
        {
            renderer = new MdxRenderer(_gl, mdx, modelDir, _dataSource, _texResolver,
                explicitTextureVariations: entry.TextureVariations);
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[Screenshot] Skip {entry.Name}: MdxRenderer creation failed: {ex.Message}");
            return 0;
        }

        // Compute bounds once and build 8 corners in world space (after model transform)
        var (boundsMin, boundsMax) = ComputeBounds(mdx);
        float scale = entry.EffectiveScale;
        // MirrorX + scale: same transform as standalone MdxRenderer.Render()
        var modelTransform = Matrix4x4.CreateScale(-scale, scale, scale);

        // Transform all 8 bounding box corners into world space
        var corners = new Vector3[8];
        for (int ci = 0; ci < 8; ci++)
        {
            corners[ci] = Vector3.Transform(new Vector3(
                (ci & 1) == 0 ? boundsMin.X : boundsMax.X,
                (ci & 2) == 0 ? boundsMin.Y : boundsMax.Y,
                (ci & 4) == 0 ? boundsMin.Z : boundsMax.Z), modelTransform);
        }
        // World-space center and radius (after transform)
        var wsMin = corners[0]; var wsMax = corners[0];
        for (int ci = 1; ci < 8; ci++) { wsMin = Vector3.Min(wsMin, corners[ci]); wsMax = Vector3.Max(wsMax, corners[ci]); }
        var center = (wsMin + wsMax) * 0.5f;
        float radius = (wsMax - wsMin).Length() * 0.5f;
        if (radius < 0.01f) radius = 1.0f;

        float fovRad = 25f * MathF.PI / 180f;
        float aspect = (float)width / height;

        // Setup FBO once
        EnsureFbo(width, height);
        if (!_fboReady)
        {
            ViewerLog.Trace($"[Screenshot] Skip {entry.Name}: FBO not ready");
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

                    // Camera direction (unit vector from camera toward center)
                    var camDir = new Vector3(
                        cosElev * MathF.Cos(azim),
                        cosElev * MathF.Sin(azim),
                        -sinElev);
                    camDir = Vector3.Normalize(camDir);

                    // Build a temporary view at unit distance to find required framing
                    var tmpCam = center - camDir * radius;
                    // Use a stable up vector (avoid degenerate case when looking straight down)
                    var up = MathF.Abs(Vector3.Dot(camDir, Vector3.UnitZ)) > 0.99f
                        ? Vector3.UnitX : Vector3.UnitZ;
                    var tmpView = Matrix4x4.CreateLookAt(tmpCam, center, up);

                    // Project all 8 corners into view space and find the required distance
                    float halfFovV = fovRad * 0.5f;
                    float halfFovH = MathF.Atan(MathF.Tan(halfFovV) * aspect);
                    float maxDist = radius; // minimum
                    for (int ci = 0; ci < 8; ci++)
                    {
                        var viewPos = Vector3.Transform(corners[ci], tmpView);
                        // viewPos.Z is negative (into screen) in OpenGL view space
                        float depth = -viewPos.Z;
                        // Distance needed so this corner's vertical extent fits
                        float needV = MathF.Abs(viewPos.Y) / MathF.Tan(halfFovV) + depth;
                        // Distance needed so this corner's horizontal extent fits
                        float needH = MathF.Abs(viewPos.X) / MathF.Tan(halfFovH) + depth;
                        maxDist = MathF.Max(maxDist, MathF.Max(needV, needH));
                    }
                    float dist = maxDist * 1.15f; // 15% padding

                    var camPos = center - camDir * dist;
                    var view = Matrix4x4.CreateLookAt(camPos, center, up);
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

ForceOpaqueAlpha(pixels);

                    _gl.BindFramebuffer(FramebufferTarget.Framebuffer, 0);

                    // Convert to image (OpenGL reads bottom-up, flip)
                    using var image = SixLabors.ImageSharp.Image.LoadPixelData<Rgba32>(pixels, width, height);
                    image.Mutate(x => x.Flip(FlipMode.Vertical));
                    image.SaveAsPng(Path.Combine(outputDir, $"{angleName}.png"));
                    count++;
                }
                catch (Exception ex)
                {
                    ViewerLog.Trace($"[Screenshot] Angle {angleName} failed for {entry.Name}: {ex.Message}");
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

    private static void ForceOpaqueAlpha(byte[] rgbaPixels)
    {
        for (int index = 3; index < rgbaPixels.Length; index += 4)
            rgbaPixels[index] = 255;
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
            ViewerLog.Trace($"[Screenshot] FBO incomplete: {status}");
            _gl.BindFramebuffer(FramebufferTarget.Framebuffer, 0);
            return;
        }

        _gl.BindFramebuffer(FramebufferTarget.Framebuffer, 0);
        _fboReady = true;
    }

    private static string SanitizePathComponent(string name)
    {
        var invalid = Path.GetInvalidFileNameChars();
        var sb = new System.Text.StringBuilder(name.Length);
        foreach (char c in name)
        {
            if (c == ' ' || c == '/' || c == '\\') sb.Append('_');
            else if (Array.IndexOf(invalid, c) < 0 && c != '\0') sb.Append(c);
        }
        string result = sb.ToString();
        return result.Length > 100 ? result[..100] : result;
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
