using System.Numerics;
using Silk.NET.OpenGL;
using WowViewer.Core.Runtime.World.Terrain;

namespace WowViewer.App;

internal readonly record struct WowViewerWorldScenePlan(Vector3 CameraPosition, Vector3 CameraTarget)
{
    public static WowViewerWorldScenePlan Identity { get; } = new(new Vector3(0f, 0f, 1f), Vector3.Zero);
}

internal static class WowViewerWorldScenePlanner
{
    public static WowViewerWorldScenePlan Build(WowViewerWorldRuntimeFrameResult runtimeFrame)
    {
        ArgumentNullException.ThrowIfNull(runtimeFrame);

        float minHeight = runtimeFrame.TerrainTileData.Heightmap?.MinHeight ?? 0f;
        float maxHeight = runtimeFrame.TerrainTileData.Heightmap?.MaxHeight ?? 0f;
        float centerHeight = runtimeFrame.TerrainTileData.Heightmap?.CenterHeight ?? ((minHeight + maxHeight) * 0.5f);
        Vector3 boundsMin = new(runtimeFrame.PlanarMin.X, runtimeFrame.PlanarMin.Y, minHeight - 32f);
        Vector3 boundsMax = new(runtimeFrame.PlanarMax.X, runtimeFrame.PlanarMax.Y, maxHeight + 32f);

        Vector3 cameraTarget = runtimeFrame.CameraTarget;
        if (cameraTarget.LengthSquared() <= 0.0001f)
        {
            Vector2 planarCenter = (runtimeFrame.PlanarMin + runtimeFrame.PlanarMax) * 0.5f;
            cameraTarget = new Vector3(planarCenter.X, planarCenter.Y, centerHeight);
        }

        Vector3 cameraPosition;
        if (runtimeFrame.CameraForward.LengthSquared() > 0.0001f)
        {
            Vector3 offset = runtimeFrame.CameraPosition - cameraTarget;
            cameraPosition = offset.LengthSquared() > 1f
                ? runtimeFrame.CameraPosition
                : cameraTarget - (runtimeFrame.CameraForward * 900f) + new Vector3(0f, 0f, 220f);
        }
        else
        {
            Vector3 extent = boundsMax - boundsMin;
            float radius = MathF.Max(extent.Length() * 0.5f, 128f);
            cameraPosition = cameraTarget + new Vector3(-radius * 1.15f, -radius * 1.15f, radius * 0.60f);
        }

        return new WowViewerWorldScenePlan(cameraPosition, cameraTarget);
    }
}

internal sealed class WowViewerWorldSceneHost : IDisposable
{
    private readonly WowViewerWorldAssetInventory _assetInventory = new();
    private IViewerIoService? _viewerIoService;
    private ViewerIoSourceKey _sourceKey;
    private WorldGpuPreviewRenderer? _renderer;
    private string _rendererSourceSignature = string.Empty;
    private WowViewerWorldScenePlan _scenePlan = WowViewerWorldScenePlan.Identity;

    public WowViewerWorldSessionBootstrapResult? CurrentSession { get; private set; }

    public WowViewerWorldRuntimeFrameResult? CurrentFrame { get; private set; }

    public WorldGpuPreviewRenderer? Renderer => _renderer;

    public WorldViewCamera Camera { get; } = new();

    public WowViewerWorldAssetState AssetState { get; private set; } = WowViewerWorldAssetState.Empty;

    public WowViewerWorldSceneSnapshot SceneSnapshot { get; private set; } = WowViewerWorldSceneSnapshot.Empty;

    public WowViewerWorldDiagnosticsSnapshot DiagnosticsSnapshot { get; private set; } = WowViewerWorldDiagnosticsSnapshot.Empty;

    public WowViewerWorldNavigatorState NavigatorState { get; private set; } = WowViewerWorldNavigatorState.Empty;

    public WowViewerWorldSpatialSnapshot SpatialSnapshot { get; private set; } = WowViewerWorldSpatialSnapshot.Empty;

    public WorldTerrainVisualSnapshot? TerrainPreviewSnapshot { get; private set; }

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
        TerrainPreviewSnapshot = null;
        _scenePlan = WowViewerWorldScenePlan.Identity;
        _renderer?.ClearPreview();
        ResetCameraToIdentity();
    }

    public void ResetCameraToIdentity()
    {
        Camera.ResetToIdentity();
    }

    public void ResetCamera()
    {
        Camera.Reset();
    }

    public void RotateCamera(float yawDeltaDegrees, float pitchDeltaDegrees)
    {
        Camera.RotateLook(yawDeltaDegrees, pitchDeltaDegrees);
    }

    public void TranslateCamera(float forwardDistance, float strafeDistance, float verticalDistance)
    {
        Camera.Translate(forwardDistance, strafeDistance, verticalDistance);
    }

    private void ApplySceneDefaultCamera()
    {
        Camera.SetPose(_scenePlan.CameraPosition, _scenePlan.CameraTarget, saveAsDefault: true);
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
        TerrainPreviewSnapshot = runtimeFrame.TerrainVisualSnapshot;
        _scenePlan = WowViewerWorldScenePlanner.Build(runtimeFrame);
        ApplySceneDefaultCamera();
        EnsureRenderer(gl, viewerIoService, sourceKey, sourceSignature)?.LoadPreview(runtimeFrame, ignoreTerrainHoles, showHoleOverlay);
    }

    public bool RefreshRendererPreview(bool ignoreTerrainHoles, bool showHoleOverlay)
    {
        if (CurrentFrame is null || _renderer is null)
            return false;

        _renderer.LoadPreview(CurrentFrame, ignoreTerrainHoles, showHoleOverlay);
        return true;
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