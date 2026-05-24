using System.Numerics;
using Silk.NET.OpenGL;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WowViewer.Core.Renderer.Headless;
using WowViewer.Core.Renderer.Scene;
using WowViewer.Core.Runtime.World.Terrain;
using WowViewer.Core.Runtime.World.Validation;

namespace WowViewer.Core.Renderer.Validation;

public sealed class NativeValidationWorldSceneAdapter : IValidationWorldSceneAdapter
{
    private HeadlessValidationCaptureSession? _session;
    private ValidationCaptureVariantPolicy _variantPolicy;
    private HeadlessContext? _context;
    private RenderSurface? _surface;
    private SceneRenderer? _renderer;
    private FrameCapture? _frameCapture;
    private WorldTerrainTileData? _tileData;
    private float _lastGroundHeight;
    private int _lastTileX;
    private int _lastTileY;
    private ArchiveCatalogSession? _archiveSession;
    private bool _disposed;

    public void Initialize(HeadlessValidationCaptureSession session)
    {
        _session = session ?? throw new ArgumentNullException(nameof(session));
    }

    public void ApplyScenePolicy(ValidationCaptureScenePolicy scenePolicy)
    {
    }

    public void ApplyVariantPolicy(ValidationCaptureVariantPolicy variantPolicy)
    {
        _variantPolicy = variantPolicy;
    }

    public ValidationWorldSceneSnapshot CaptureSnapshot(
        ValidationCaptureTileRequest request,
        int framesObserved,
        int settledFrames)
    {
        ArgumentNullException.ThrowIfNull(request);

        string virtualPath = BuildAdtVirtualPath(request.TileX, request.TileY);
        string tex0VirtualPath = BuildTex0VirtualPath(request.TileX, request.TileY);

        ArchiveCatalogSession archiveSession = GetOrCreateArchiveSession();
        byte[]? adtBytes = archiveSession.ReadFile(virtualPath)
            ?? archiveSession.TryReadFileFromDisk(virtualPath);

        if (adtBytes is null)
        {
            return new ValidationWorldSceneSnapshot(
                HasSceneContent: false,
                FramebufferWidth: _session?.ScenePolicy.RequestedResolution ?? 512,
                FramebufferHeight: _session?.ScenePolicy.RequestedResolution ?? 512,
                TargetTileLoaded: false,
                TerrainStreaming: false,
                PendingWorldObjectLoadCount: 0);
        }

        using MemoryStream adtStream = new(adtBytes, writable: false);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(adtStream, virtualPath);

        AdtTextureFile? textureFile = null;
        byte[]? tex0Bytes = archiveSession.ReadFile(tex0VirtualPath)
            ?? archiveSession.TryReadFileFromDisk(tex0VirtualPath);
        if (tex0Bytes is { Length: > 0 })
        {
            using MemoryStream tex0Stream = new(tex0Bytes, writable: false);
            MapFileSummary tex0Summary = MapFileSummaryReader.Read(tex0Stream, tex0VirtualPath);
            tex0Stream.Position = 0;
            textureFile = AdtTextureReader.Read(tex0Stream, tex0Summary);
        }

        adtStream.Position = 0;
        _tileData = WorldTerrainTileBuilder.Read(adtStream, fileSummary, textureFile);
        _lastTileX = request.TileX;
        _lastTileY = request.TileY;
        _lastGroundHeight = _tileData.Heightmap?.CenterHeight ?? 0f;

        return new ValidationWorldSceneSnapshot(
            HasSceneContent: true,
            FramebufferWidth: _session?.ScenePolicy.RequestedResolution ?? 512,
            FramebufferHeight: _session?.ScenePolicy.RequestedResolution ?? 512,
            TargetTileLoaded: true,
            TerrainStreaming: false,
            PendingWorldObjectLoadCount: 0);
    }

    public float ResolveGroundHeight(int tileX, int tileY)
    {
        return _lastGroundHeight;
    }

    public void RenderFrame(ValidationCaptureCameraFrame cameraFrame)
    {
        EnsureRenderer();
        if (_renderer is null || _tileData is null || _surface is null)
            return;

        var camera = CreateCamera(cameraFrame);
        RenderVariant variant = MapVariant(_variantPolicy);

        _surface.Clear(0.34f, 0.38f, 0.42f, 1f);

        _renderer.SetTileData(_tileData);
        _renderer.RenderFull(camera, variant);
    }

    public byte[] ReadFramebufferRgba()
    {
        if (_frameCapture is null)
            throw new InvalidOperationException("RenderFrame must be called before reading the framebuffer.");
        return _frameCapture.CaptureRgba();
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        _renderer?.Dispose();
        _frameCapture = null;
        _context?.Dispose();
        _context = null;
    }

    private void EnsureRenderer()
    {
        if (_context is not null)
            return;

        int resolution = _session?.ScenePolicy.RequestedResolution ?? 512;
        _context = new HeadlessContext(resolution, resolution);
        _surface = new RenderSurface(_context.GL, resolution, resolution);
        _renderer = new SceneRenderer(_context.GL);
        _frameCapture = new FrameCapture(_context, resolution, resolution);
    }

    private SceneCamera CreateCamera(ValidationCaptureCameraFrame cameraFrame)
    {
        var camera = new SceneCamera
        {
            NearPlane = 0.1f,
            FarPlane = 20000f,
            AspectRatio = 1f,
        };
        camera.LookAtTile(_lastTileX, _lastTileY, 800f);
        return camera;
    }

    private ArchiveCatalogSession GetOrCreateArchiveSession()
    {
        if (_archiveSession is not null)
            return _archiveSession;

        string clientRoot = _session?.ClientRoot
            ?? throw new InvalidOperationException("Initialize must be called first.");

        _archiveSession = ArchiveCatalogSessionCache.GetOrCreate(
            [clientRoot],
            new ArchiveCatalogBootstrapOptions(),
            new NativeMpqServiceFactory());

        return _archiveSession;
    }

    private string BuildAdtVirtualPath(int tileX, int tileY)
    {
        string mapDir = _session?.BatchPlan.MapName
            ?? throw new InvalidOperationException("Session not initialized.");
        return $"World\\Maps\\{mapDir}\\{mapDir}_{tileY}_{tileX}.adt";
    }

    private string BuildTex0VirtualPath(int tileX, int tileY)
    {
        string mapDir = _session?.BatchPlan.MapName
            ?? throw new InvalidOperationException("Session not initialized.");
        return $"World\\Maps\\{mapDir}\\{mapDir}_{tileY}_{tileX}_tex0.adt";
    }

    private static RenderVariant MapVariant(ValidationCaptureVariantPolicy policy)
    {
        return new RenderVariant
        {
            HideTerrain = !policy.ShowTerrain,
            HideLiquids = !policy.ShowTerrainLiquids,
            HideObjects = !(policy.ShowWmos || policy.ShowDoodads),
        };
    }
}
