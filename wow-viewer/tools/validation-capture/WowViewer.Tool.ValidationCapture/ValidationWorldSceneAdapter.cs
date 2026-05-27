using System.Runtime.Versioning;
using Silk.NET.Maths;
using Silk.NET.OpenGL;
using Silk.NET.Windowing;
using WowViewer.App;
using WowViewer.Core.Runtime.World.Passes;
using WowViewer.Core.Runtime.World.Validation;

namespace WowViewer.Tools.ValidationCapture;

internal sealed class ValidationWorldSceneAdapter : IValidationWorldSceneAdapter
{
    private HeadlessValidationCaptureSession? _session;
    private ValidationCaptureScenePolicy? _scenePolicy;
    private ValidationWorldScenePolicyState? _policyState;
    private WowViewerWorldRuntimeFrameResult? _runtimeFrame;
    private float _lastGroundHeight;
    private HiddenWindowRenderHost? _renderHost;

    public void Initialize(HeadlessValidationCaptureSession session)
    {
        _session = session ?? throw new ArgumentNullException(nameof(session));
    }

    public void ApplyScenePolicy(ValidationCaptureScenePolicy scenePolicy)
    {
        ArgumentNullException.ThrowIfNull(scenePolicy);
        EnsureInitialized();

        _scenePolicy = scenePolicy;
        _policyState = new ValidationWorldScenePolicyState();
        ValidationWorldScenePolicyApplier.ApplyScenePolicy(_policyState, scenePolicy);
    }

    public void ApplyVariantPolicy(ValidationCaptureVariantPolicy variantPolicy)
    {
        EnsureScenePolicy();
        _policyState = ValidationWorldScenePolicyApplier.CreateState(_scenePolicy!, variantPolicy);
    }

    public ValidationWorldSceneSnapshot CaptureSnapshot(
        ValidationCaptureTileRequest request,
        int framesObserved,
        int settledFrames)
    {
        ArgumentNullException.ThrowIfNull(request);
        EnsureReadyToCapture();

        ValidationWorldScenePolicyState policyState = _policyState!;
        WowViewerWorldRuntimeFrameRequest runtimeRequest = BuildFrameRequest(_session!, request, policyState);
        WowViewerWorldRuntimeFrameResult runtimeFrame = WowViewerWorldRuntimeBridge.Build(runtimeRequest);
        _runtimeFrame = runtimeFrame;
        _lastGroundHeight = runtimeFrame.TerrainTileData.Heightmap?.CenterHeight ?? 0f;

        return CreateSnapshot(
            _scenePolicy!.RequestedResolution,
            request,
            runtimeFrame.SelectedTileX,
            runtimeFrame.SelectedTileY,
            runtimeFrame.ActiveTerrainTiles.Count,
            runtimeFrame.WmoInstances.Count,
            runtimeFrame.MdxInstances.Count,
            runtimeFrame.PendingAssetKeys.Count);
    }

    public float ResolveGroundHeight(int tileX, int tileY)
    {
        return _lastGroundHeight;
    }

    public void RenderFrame(ValidationCaptureCameraFrame cameraFrame)
    {
        if (!OperatingSystem.IsWindows())
            throw new PlatformNotSupportedException("ValidationWorldSceneAdapter GPU capture currently requires Windows.");

        EnsureReadyToRender();
        _renderHost ??= new HiddenWindowRenderHost(_session!, _scenePolicy!.RequestedResolution);
        _renderHost.Render(_runtimeFrame!, _scenePolicy!.IgnoreTerrainHolesGlobally, cameraFrame);
    }

    public byte[] ReadFramebufferRgba()
    {
        if (!OperatingSystem.IsWindows())
            throw new PlatformNotSupportedException("ValidationWorldSceneAdapter GPU capture currently requires Windows.");

        EnsureReadyToRender();
        return _renderHost!.ReadFramebufferRgba();
    }

    public void Dispose()
    {
        if (!OperatingSystem.IsWindows())
            return;

        _renderHost?.Dispose();
        _renderHost = null;
    }

    internal static WowViewerWorldRuntimeFrameRequest BuildFrameRequest(
        HeadlessValidationCaptureSession session,
        ValidationCaptureTileRequest request,
        ValidationWorldScenePolicyState policyState)
    {
        ArgumentNullException.ThrowIfNull(session);
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(policyState);

        return new WowViewerWorldRuntimeFrameRequest(
            session.ClientRoot,
            session.MapInput,
            session.BuildLabel ?? string.Empty,
            session.LooseOverlayRoot ?? string.Empty,
            request.TileX,
            request.TileY,
            BuildPassOptions(policyState),
            FogEndDistance: policyState.FogEndDistance,
            ObjectStreamingRangeMultiplier: policyState.ObjectStreamingRangeMultiplier,
            MaxVisibleMdxBoundsHeight: policyState.MaxVisibleMdxBoundsHeight,
            IgnoreDistanceCulling: policyState.IgnoreDistanceCulling,
            IgnoreProjectedSizeCulling: policyState.IgnoreProjectedSizeCulling,
            IgnoreVisionConeCulling: policyState.IgnoreVisionConeCulling,
            IgnoreFrustumCulling: policyState.IgnoreFrustumCulling,
            IgnoreMaxViewDistanceCulling: policyState.IgnoreMaxViewDistanceCulling);
    }

    internal static WorldFramePassOptions BuildPassOptions(ValidationWorldScenePolicyState policyState)
    {
        ArgumentNullException.ThrowIfNull(policyState);

        bool objectsVisible = policyState.ShowObjects && (policyState.ShowWmos || policyState.ShowDoodads);
        return new WorldFramePassOptions(
            objectsVisible,
            policyState.ShowWmos,
            policyState.ShowDoodads,
            policyState.ShowSky,
            policyState.ShowWdl,
            policyState.ShowTerrain,
            policyState.ShowTerrainLiquids,
            overlayVisible: false);
    }

    internal static ValidationWorldSceneSnapshot CreateSnapshot(
        int requestedResolution,
        ValidationCaptureTileRequest request,
        int selectedTileX,
        int selectedTileY,
        int activeTerrainTileCount,
        int wmoInstanceCount,
        int mdxInstanceCount,
        int pendingWorldObjectLoadCount)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(requestedResolution, 1);
        ArgumentNullException.ThrowIfNull(request);
        ArgumentOutOfRangeException.ThrowIfNegative(activeTerrainTileCount);
        ArgumentOutOfRangeException.ThrowIfNegative(wmoInstanceCount);
        ArgumentOutOfRangeException.ThrowIfNegative(mdxInstanceCount);
        ArgumentOutOfRangeException.ThrowIfNegative(pendingWorldObjectLoadCount);

        bool hasSceneContent = activeTerrainTileCount > 0 || wmoInstanceCount > 0 || mdxInstanceCount > 0;
        bool targetTileLoaded = activeTerrainTileCount > 0
            && selectedTileX == request.TileX
            && selectedTileY == request.TileY;

        return new ValidationWorldSceneSnapshot(
            HasSceneContent: hasSceneContent,
            FramebufferWidth: requestedResolution,
            FramebufferHeight: requestedResolution,
            TargetTileLoaded: targetTileLoaded,
            TerrainStreaming: false,
            PendingWorldObjectLoadCount: pendingWorldObjectLoadCount);
    }

    private void EnsureInitialized()
    {
        if (_session is null)
            throw new InvalidOperationException("Initialize must be called before applying validation scene policy.");
    }

    private void EnsureScenePolicy()
    {
        EnsureInitialized();
        if (_scenePolicy is null)
            throw new InvalidOperationException("ApplyScenePolicy must be called before applying validation variant policy.");
    }

    private void EnsureReadyToCapture()
    {
        EnsureScenePolicy();
        if (_policyState is null)
            throw new InvalidOperationException("ApplyVariantPolicy must be called before capturing a validation world scene snapshot.");
    }

    private void EnsureReadyToRender()
    {
        EnsureReadyToCapture();
        if (_runtimeFrame is null)
            throw new InvalidOperationException("CaptureSnapshot must be called before rendering a validation world scene frame.");
    }

    [SupportedOSPlatform("windows")]
    private sealed class HiddenWindowRenderHost : IDisposable
    {
        private readonly HeadlessValidationCaptureSession _session;
        private readonly int _resolution;
        private readonly IWindow _window;
        private readonly AutoResetEvent _loadCompleted = new(false);
        private readonly AutoResetEvent _renderCompleted = new(false);
        private readonly object _sync = new();
        private readonly Thread _windowThread;

        private GL? _gl;
        private ViewerIoService? _viewerIoService;
        private WorldGpuPreviewRenderer? _renderer;
        private WowViewerWorldRuntimeFrameResult? _pendingRuntimeFrame;
        private ValidationCaptureCameraFrame _pendingCameraFrame;
        private byte[]? _lastFramebuffer;
        private Exception? _loadFailure;
        private Exception? _renderFailure;
        private bool _renderRequested;
        private bool _disposed;

        public HiddenWindowRenderHost(HeadlessValidationCaptureSession session, int resolution)
        {
            if (!OperatingSystem.IsWindows())
                throw new PlatformNotSupportedException("ValidationWorldSceneAdapter hidden-window GPU capture currently requires Windows.");

            _session = session;
            _resolution = resolution;

            WindowOptions options = WindowOptions.Default;
            options.Title = "wow-viewer validation gpu capture";
            options.Size = new Vector2D<int>(resolution, resolution);
            options.IsVisible = false;
            options.VSync = false;
            options.API = new GraphicsAPI(ContextAPI.OpenGL, ContextProfile.Core, ContextFlags.ForwardCompatible, new APIVersion(3, 3));

            _window = Window.Create(options);
            _window.Load += OnLoad;
            _window.Render += OnRender;
            _window.Closing += OnClosing;

            _windowThread = new Thread(() => _window.Run())
            {
                IsBackground = true,
                Name = "ValidationWorldSceneAdapter.HiddenWindowRenderHost"
            };
            _windowThread.SetApartmentState(ApartmentState.STA);
            _windowThread.Start();

            _loadCompleted.WaitOne();
            if (_loadFailure is not null)
                throw new InvalidOperationException($"Failed to initialize validation GPU render host: {_loadFailure.Message}", _loadFailure);
        }

        public void Render(WowViewerWorldRuntimeFrameResult runtimeFrame, bool ignoreTerrainHolesGlobally, ValidationCaptureCameraFrame cameraFrame)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            ArgumentNullException.ThrowIfNull(runtimeFrame);

            lock (_sync)
            {
                _pendingRuntimeFrame = runtimeFrame;
                _pendingCameraFrame = cameraFrame;
                _renderFailure = null;
                _lastFramebuffer = null;
                _renderRequested = true;
            }

            _renderCompleted.WaitOne();
            if (_renderFailure is not null)
                throw new InvalidOperationException($"Validation GPU capture failed: {_renderFailure.Message}", _renderFailure);
        }

        public byte[] ReadFramebufferRgba()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_lastFramebuffer is null)
                throw new InvalidOperationException("Validation GPU capture did not produce framebuffer pixels.");

            return _lastFramebuffer;
        }

        public void Dispose()
        {
            if (_disposed)
                return;

            _disposed = true;
            try
            {
                _window.Close();
            }
            catch
            {
            }

            if (_windowThread.IsAlive)
                _windowThread.Join();

            _loadCompleted.Dispose();
            _renderCompleted.Dispose();
            _window.Dispose();
        }

        private void OnLoad()
        {
            try
            {
                _gl = _window.CreateOpenGL();
                _viewerIoService = new ViewerIoService();
                ViewerIoSourceKey sourceKey = ViewerIoSourceKey.Create(_session.ClientRoot, _session.BuildLabel, _session.LooseOverlayRoot);
                _renderer = new WorldGpuPreviewRenderer(_gl, _viewerIoService, sourceKey);
            }
            catch (Exception ex)
            {
                _loadFailure = ex;
            }
            finally
            {
                _loadCompleted.Set();
            }
        }

        private void OnRender(double _)
        {
            bool shouldRender;
            WowViewerWorldRuntimeFrameResult? runtimeFrame;
            ValidationCaptureCameraFrame cameraFrame;

            lock (_sync)
            {
                shouldRender = _renderRequested;
                runtimeFrame = _pendingRuntimeFrame;
                cameraFrame = _pendingCameraFrame;
                _renderRequested = false;
            }

            if (!shouldRender || runtimeFrame is null || _gl is null || _renderer is null)
                return;

            try
            {
                _renderer.LoadPreview(runtimeFrame, _session.ScenePolicy.IgnoreTerrainHolesGlobally, showHoleOverlay: false);
                _renderer.Render(_resolution, _resolution, cameraFrame);
                _lastFramebuffer = ReadPreviewTexture(_gl, _renderer.PreviewTextureHandle, _resolution, _resolution);
            }
            catch (Exception ex)
            {
                _renderFailure = ex;
            }
            finally
            {
                _renderCompleted.Set();
            }
        }

        private void OnClosing()
        {
            _renderer = null;
            _viewerIoService = null;
            _gl = null;
        }

        private static unsafe byte[] ReadPreviewTexture(GL gl, uint textureHandle, int width, int height)
        {
            if (textureHandle == 0)
                throw new InvalidOperationException("World GPU renderer did not produce a preview texture.");

            uint framebuffer = gl.GenFramebuffer();
            try
            {
                gl.BindFramebuffer(FramebufferTarget.Framebuffer, framebuffer);
                gl.FramebufferTexture2D(FramebufferTarget.Framebuffer, FramebufferAttachment.ColorAttachment0, TextureTarget.Texture2D, textureHandle, 0);
                byte[] rgbaPixels = new byte[checked(width * height * 4)];
                fixed (byte* pixelPtr = rgbaPixels)
                {
                    gl.ReadPixels(0, 0, (uint)width, (uint)height, PixelFormat.Rgba, PixelType.UnsignedByte, pixelPtr);
                }

                return rgbaPixels;
            }
            finally
            {
                gl.BindFramebuffer(FramebufferTarget.Framebuffer, 0);
                gl.DeleteFramebuffer(framebuffer);
            }
        }
    }
}
