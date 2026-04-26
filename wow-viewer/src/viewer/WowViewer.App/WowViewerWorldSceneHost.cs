using Silk.NET.OpenGL;

namespace WowViewer.App;

internal sealed class WowViewerWorldSceneHost : IDisposable
{
    private WorldGpuPreviewRenderer? _renderer;
    private string _rendererSourceSignature = string.Empty;

    public WowViewerWorldSessionBootstrapResult? CurrentSession { get; private set; }

    public WowViewerWorldRuntimeFrameResult? CurrentFrame { get; private set; }

    public WorldGpuPreviewRenderer? Renderer => _renderer;

    public WorldViewCamera Camera { get; } = new();

    public WowViewerWorldAssetState AssetState { get; private set; } = WowViewerWorldAssetState.Empty;

    public WowViewerWorldSceneSnapshot SceneSnapshot { get; private set; } = WowViewerWorldSceneSnapshot.Empty;

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
        AssetState = WowViewerWorldAssetState.Empty;
        SceneSnapshot = WowViewerWorldSceneSnapshot.Empty;
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
        AssetState = WowViewerWorldAssetState.FromRuntimeFrame(runtimeFrame);
        SceneSnapshot = WowViewerWorldSceneSnapshot.FromRuntimeFrame(runtimeFrame);
        EnsureRenderer(gl, viewerIoService, sourceKey, sourceSignature)?.LoadPreview(runtimeFrame, Camera, ignoreTerrainHoles, showHoleOverlay);
    }
}