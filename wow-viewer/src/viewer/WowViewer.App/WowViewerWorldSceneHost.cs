using Silk.NET.OpenGL;

namespace WowViewer.App;

internal sealed class WowViewerWorldSceneHost : IDisposable
{
    private readonly WowViewerWorldAssetInventory _assetInventory = new();
    private IViewerIoService? _viewerIoService;
    private ViewerIoSourceKey _sourceKey;
    private WorldGpuPreviewRenderer? _renderer;
    private string _rendererSourceSignature = string.Empty;

    public WowViewerWorldSessionBootstrapResult? CurrentSession { get; private set; }

    public WowViewerWorldRuntimeFrameResult? CurrentFrame { get; private set; }

    public WorldGpuPreviewRenderer? Renderer => _renderer;

    public WorldViewCamera Camera { get; } = new();

    public WowViewerWorldAssetState AssetState { get; private set; } = WowViewerWorldAssetState.Empty;

    public WowViewerWorldSceneSnapshot SceneSnapshot { get; private set; } = WowViewerWorldSceneSnapshot.Empty;

    public WowViewerWorldDiagnosticsSnapshot DiagnosticsSnapshot { get; private set; } = WowViewerWorldDiagnosticsSnapshot.Empty;

    public WowViewerWorldNavigatorState NavigatorState { get; private set; } = WowViewerWorldNavigatorState.Empty;

    public WowViewerWorldSpatialSnapshot SpatialSnapshot { get; private set; } = WowViewerWorldSpatialSnapshot.Empty;

    public void Dispose()
    {
        _renderer?.Dispose();
        _renderer = null;
        _rendererSourceSignature = string.Empty;
    }

    public void Clear()
    {
        CurrentSession = null;
        CurrentFrame = null;
        _viewerIoService = null;
        _sourceKey = default;
        _assetInventory.Reset();
        AssetState = WowViewerWorldAssetState.Empty;
        SceneSnapshot = WowViewerWorldSceneSnapshot.Empty;
        DiagnosticsSnapshot = WowViewerWorldDiagnosticsSnapshot.Empty;
        NavigatorState = WowViewerWorldNavigatorState.Empty;
        SpatialSnapshot = WowViewerWorldSpatialSnapshot.Empty;
        _renderer?.ClearPreview();
        Camera.ResetToIdentity();
    }

    public WorldGpuPreviewRenderer? EnsureRenderer(GL? gl, IViewerIoService viewerIoService, ViewerIoSourceKey sourceKey, string sourceSignature)
    {
        if (gl == null)
            return null;

        if (_renderer != null && !string.Equals(sourceSignature, _rendererSourceSignature, StringComparison.OrdinalIgnoreCase))
        {
            _renderer.Dispose();
            _renderer = null;
        }

        _rendererSourceSignature = sourceSignature;
        _renderer ??= new WorldGpuPreviewRenderer(gl, viewerIoService, sourceKey);
        return _renderer;
    }

    public void ApplyRuntimeFrame(
        GL? gl,
        IViewerIoService viewerIoService,
        ViewerIoSourceKey sourceKey,
        string sourceSignature,
        WowViewerWorldRuntimeFrameResult runtimeFrame,
        bool ignoreTerrainHoles,
        bool showHoleOverlay)
    {
        CurrentSession = runtimeFrame.Session;
        CurrentFrame = runtimeFrame;
        _viewerIoService = viewerIoService;
        _sourceKey = sourceKey;
        _assetInventory.ObserveRuntimeFrame(runtimeFrame);
        AssetState = _assetInventory.CreateState();
        SceneSnapshot = WowViewerWorldSceneSnapshot.FromRuntimeFrame(runtimeFrame);
        DiagnosticsSnapshot = WowViewerWorldDiagnosticsSnapshot.FromRuntimeFrame(runtimeFrame);
        NavigatorState = WowViewerWorldNavigatorState.FromRuntimeFrame(runtimeFrame);
        SpatialSnapshot = WowViewerWorldSpatialSnapshot.FromRuntimeFrame(runtimeFrame);
        EnsureRenderer(gl, viewerIoService, sourceKey, sourceSignature)?.LoadPreview(runtimeFrame, Camera, ignoreTerrainHoles, showHoleOverlay);
    }

    public int ProcessPendingAssetLoads(int maxLoads = 2, double maxBudgetMs = 4.0)
    {
        if (_viewerIoService == null || !_sourceKey.HasClientRoot)
            return 0;

        int processed = _assetInventory.ProcessPendingLoads(
            request => _viewerIoService.TryReadVirtualFile(_sourceKey, request.ModelKey, out _, out _),
            maxLoads,
            maxBudgetMs);

        if (processed > 0)
            AssetState = _assetInventory.CreateState();

        return processed;
    }
}