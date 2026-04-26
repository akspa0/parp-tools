using System.Numerics;
using System.Reflection;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using ImGuiNET;
using Silk.NET.Input;
using Silk.NET.Maths;
using Silk.NET.OpenGL;
using Silk.NET.OpenGL.Extensions.ImGui;
using Silk.NET.Windowing;
using WowViewer.Core;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WowViewer.Core.Mdx;
using WowViewer.Core.PM4;
using WowViewer.Core.Runtime;
using WowViewer.Core.Runtime.Mdx;
using WowViewer.Core.Runtime.M2;
using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.Liquid;
using WowViewer.Core.Runtime.World.Passes;
using WowViewer.Core.Runtime.World.Terrain;
using WowViewer.Core.Runtime.World.Visibility;
using WowViewer.Core.Runtime.World.Wdl;
using WowViewer.Core.Wmo;

namespace WowViewer.App;

internal sealed class WowViewerDesktopApp : IDisposable
{
    private const float WorldMinimapTileCount = 64f;

    private sealed record PendingWorldLoadResult(int Generation, WowViewerWorldRuntimeFrameResult RuntimeFrame);
    private sealed record PendingWorldMapDiscoveryResult(string Signature, IReadOnlyList<DiscoveredLooseWorldMap> Maps, string Summary, int Version);

    private sealed class WorldSpawnPickerState
    {
        public WorldSpawnPickerState(
            string signature,
            string summary,
            WowViewerWorldSessionBootstrapResult? session,
            int version)
        {
            Signature = signature;
            Summary = summary;
            Session = session;
            Version = version;
            OccupiedTileIndices = session is null
                ? []
                : session.OccupiedTiles.Select(static tile => (tile.TileX * 64) + tile.TileY).ToHashSet();
        }

        public string Signature { get; }

        public string Summary { get; }

        public WowViewerWorldSessionBootstrapResult? Session { get; }

        public int Version { get; }

        public HashSet<int> OccupiedTileIndices { get; }
    }

    private enum WorldSelectionKind
    {
        Wmo = 0,
        Mdx = 1,
    }

    private readonly record struct WorldObjectSelection(
        WorldSelectionKind Kind,
        int TileX,
        int TileY,
        int PlacementEntryIndex,
        int UniqueId,
        string ModelKey);

    private readonly record struct WorldNavigatorEntry(
        WorldSelectionKind Kind,
        WorldObjectInstance Instance,
        bool IsVisible,
        bool AssetReady,
        float? CenterDistance,
        bool IsTaxiActor,
        bool HasOpaqueRoute,
        bool HasTransparentRoute,
        bool RequiresUnbatchedRender,
        bool WasAnimated);

    private const string WindowTitle = "WowViewer.App";
    private static readonly string[] MdxCameraPresetLabels = ["Custom", "Front", "Back", "Left", "Right", "Top", "Three Quarter"];
    private static readonly string?[] MdxCameraPresetValues = [null, "front", "back", "left", "right", "top", PreviewCameraSettings.DefaultPresetName];
    private static readonly MethodInfo? ImGuiControllerWindowResizedMethod =
        typeof(ImGuiController).GetMethod("WindowResized", BindingFlags.Instance | BindingFlags.NonPublic);

    private readonly WowViewerSession _session;
    private readonly WowViewerSession? _initialSession;
    private readonly WowViewerAppSettings _settings;
    private readonly IViewerIoService _viewerIoService = new ViewerIoService();

    private IWindow? _window;
    private GL? _gl;
    private IInputContext? _input;
    private ImGuiController? _imGui;
    private Vector2D<int> _lastSyncedImGuiWindowSize;
    private Vector2D<int> _lastSyncedImGuiFramebufferSize;
    private bool _disposed;
    private bool _graphicsResourcesReleased;
    private bool _requestInitialLoad;
    private string _statusMessage = "Configure a game-client or local asset source, then load a preview.";
    private string _lastLoadSummary = "No workspace loaded.";
    private string? _lastError;
    private M2PreviewLoadResult? _currentPreview;
    private WmoPreviewLoadResult? _currentWmoPreview;
    private MdxPreviewLoadResult? _currentMdxPreview;
    private ModelOutputScene? _currentModelOutputScene;
    private uint _previewTextureHandle;
    private uint _worldTerrainPreviewTextureHandle;
    private M2GpuPreviewRenderer? _gpuPreviewRenderer;
    private WmoGpuPreviewRenderer? _wmoGpuPreviewRenderer;
    private MdxGpuPreviewRenderer? _mdxGpuPreviewRenderer;
    private ModelOutputGpuRenderer? _modelOutputGpuRenderer;
    private readonly WowViewerWorldSceneHost _worldSceneHost = new();
    private WorldMinimapRenderer? _worldMinimapRenderer;
    private bool _showAboutWindow = true;
    private bool _showWorkspaceWindow = true;
    private bool _showControlWindow = true;
    private bool _showDiagnosticsWindow = true;
    private bool _showBoundaryWindow = true;
    private bool _showWorldStatusWindow = true;
    private bool _showNavigatorWindow = true;
    private bool _showInspectorWindow = true;
    private bool _showWorldMinimapWindow = true;
    private bool _compactWorldSessionLayout = true;
    private bool _useFixedThreeLaneShell = true;
    private bool _showFileBrowserWindow = true;
    private bool _showWorldMapBrowserWindow;
    private float _fixedShellNavigatorWidth = 360.0f;
    private float _fixedShellInspectorWidth = 420.0f;
    private AssetFileBrowserState? _assetFileBrowserState;
    private string _fileBrowserClientRoot = string.Empty;
    private string _fileBrowserLooseOverlayRoot = string.Empty;
    private string _fileBrowserBuildLabel = string.Empty;
    private AssetFileBrowserFilter _fileBrowserFilter = AssetFileBrowserFilter.SupportedAssets;
    private string _fileBrowserTitle = "Asset File Browser";
    private string _worldMapBrowserClientRoot = string.Empty;
    private string _worldMapBrowserLooseOverlayRoot = string.Empty;
    private string _worldMapBrowserBuildLabel = string.Empty;
    private string _worldMapBrowserTitle = "World Map Browser";
    private string _worldMapBrowserFilter = string.Empty;
    private IReadOnlyList<DiscoveredLooseWorldMap> _worldMapBrowserMaps = Array.Empty<DiscoveredLooseWorldMap>();
    private string _worldMapBrowserSignature = string.Empty;
    private string _worldMapBrowserSummary = "No world maps discovered yet.";
    private bool _wantOpenGameFolder;
    private bool _wantAttachLooseFolder;
    private string? _pendingKnownGoodClientPath;
    private string? _pendingKnownGoodClientBuildLabel;
    private bool _pendingKnownGoodClientAttachLooseFolder;
    private bool _pendingKnownGoodClientOpenBrowser;
    private bool _pendingKnownGoodClientOpenWorldMapBrowser;
    private IReadOnlyList<DiscoveredLooseWorldMap> _discoveredWorldMaps = Array.Empty<DiscoveredLooseWorldMap>();
    private string _worldMapDiscoverySignature = string.Empty;
    private string _worldMapDiscoverySummary = "No client maps discovered yet.";
    private Task<PendingWorldMapDiscoveryResult>? _pendingWorldMapDiscoveryTask;
    private WorldSpawnPickerState? _worldSpawnPickerState;
    private Task<WorldSpawnPickerState>? _pendingWorldSpawnPickerTask;
    private Task<PendingWorldLoadResult>? _pendingWorldLoadTask;
    private Stopwatch? _pendingWorldLoadStopwatch;
    private int _pendingWorldLoadGeneration;
    private string _pendingWorldLoadMapInput = string.Empty;
    private int _worldMapDiscoveryVersion;
    private int _worldSpawnPickerVersion;
    private bool _worldNavigatorVisibleOnly = true;
    private bool _worldNavigatorShowWmo = true;
    private bool _worldNavigatorShowMdx = true;
    private string _worldNavigatorFilter = string.Empty;
    private string _worldMinimapSourceSignature = string.Empty;
    private float _worldMinimapZoom = 24.0f;
    private Vector2 _worldMinimapPanOffset = Vector2.Zero;
    private Vector2 _worldMinimapDragStart = Vector2.Zero;
    private bool _worldMinimapDragging;
    private WorldObjectSelection? _selectedWorldObject;
    private bool _worldPreviewInputCaptured;
    private string _datasetSearchRoot = "datasets";
    private string _datasetBuildFilter = string.Empty;
    private string _datasetArchiveRootsFile = string.Empty;
    private string _datasetArchiveRootFallback = string.Empty;
    private string _datasetResumeCheckpoint = string.Empty;
    private string _datasetOutputDir = string.Empty;
    private string _datasetCacheDir = string.Empty;
    private bool _datasetAllowCpu = true;
    private bool _datasetDryRun;
    private bool _datasetSkipMasks;
    private bool _datasetSkipCache;
    private bool _datasetForceRemask;
    private int _datasetNumEpochs;
    private int _datasetBatchSize;
    private string _datasetLastCommand = "No dataset command run from the shell yet.";
    private InteractiveOrbitCameraState _m2InteractiveCamera = new();
    private InteractiveOrbitCameraState _wmoInteractiveCamera = new();
    private InteractiveOrbitCameraState _mdxInteractiveCamera = new();
    private InteractiveOrbitCameraState _modelOutputInteractiveCamera = new();

    private WowViewerWorldSessionBootstrapResult? _currentWorldSession => _worldSceneHost.CurrentSession;

    private WowViewerWorldRuntimeFrameResult? _currentWorldRuntimeFrame => _worldSceneHost.CurrentFrame;

    private WorldGpuPreviewRenderer? _worldGpuPreviewRenderer => _worldSceneHost.Renderer;

    private WorldViewCamera _worldViewCamera => _worldSceneHost.Camera;

    public WowViewerDesktopApp(WowViewerSession? initialSession = null)
    {
        _settings = WowViewerAppSettingsStore.Load();
        _session = _settings.Session ?? WowViewerSession.CreateDefault();
        _session.Normalize();
        _initialSession = initialSession;
        ApplySettingsToState(_settings);
        ApplyWorldSessionAdtFirstDefaults();
        if (_initialSession != null)
            ApplySession(_initialSession);
    }

    public void Run()
    {
        WindowOptions options = WindowOptions.Default;
        options.Title = WindowTitle;
        options.Size = new Vector2D<int>(1600, 960);
        options.IsVisible = true;
        options.VSync = false;
        options.API = new GraphicsAPI(ContextAPI.OpenGL, ContextProfile.Core, ContextFlags.ForwardCompatible, new APIVersion(3, 3));

        _window = Window.Create(options);
        _window.Load += OnLoad;
        _window.Update += OnUpdate;
        _window.Render += OnRender;
        _window.Resize += OnWindowResize;
        _window.FramebufferResize += OnFramebufferResize;
        _window.Closing += OnWindowClosing;
        _window.Run();
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _disposed = true;
        SaveSettings();
        ReleaseGraphicsResources();
        _viewerIoService.Dispose();
        _input?.Dispose();
        _input = null;
        if (_window != null)
        {
            _window.Load -= OnLoad;
            _window.Update -= OnUpdate;
            _window.Render -= OnRender;
            _window.Resize -= OnWindowResize;
            _window.FramebufferResize -= OnFramebufferResize;
            _window.Closing -= OnWindowClosing;
        }
        _window?.Dispose();
        _window = null;
    }

    private void OnWindowClosing()
    {
        ReleaseGraphicsResources();
    }

    private void ReleaseGraphicsResources()
    {
        if (_graphicsResourcesReleased)
            return;

        _graphicsResourcesReleased = true;
        TryReleaseGraphicsResource(() => _gpuPreviewRenderer?.Dispose());
        TryReleaseGraphicsResource(() => _wmoGpuPreviewRenderer?.Dispose());
        TryReleaseGraphicsResource(() => _mdxGpuPreviewRenderer?.Dispose());
        TryReleaseGraphicsResource(() => _modelOutputGpuRenderer?.Dispose());
        TryReleaseGraphicsResource(_worldSceneHost.Dispose);
        TryReleaseGraphicsResource(() => _worldMinimapRenderer?.Dispose());
        TryReleaseGraphicsResource(DeletePreviewTexture);
        TryReleaseGraphicsResource(DeleteWorldTerrainPreviewTexture);
        TryReleaseGraphicsResource(() => _imGui?.Dispose());

        _gpuPreviewRenderer = null;
        _wmoGpuPreviewRenderer = null;
        _mdxGpuPreviewRenderer = null;
        _modelOutputGpuRenderer = null;
        _worldMinimapRenderer = null;
        _imGui = null;
    }

    private static void TryReleaseGraphicsResource(Action action)
    {
        try
        {
            action();
        }
        catch (Exception ex) when (IsMissingGraphicsContext(ex))
        {
        }
    }

    private static bool IsMissingGraphicsContext(Exception exception)
    {
        for (Exception? current = exception; current != null; current = current.InnerException)
        {
            if (current.GetType().FullName == "Silk.NET.GLFW.GlfwException"
                && current.Message.Contains("current OpenGL or OpenGL ES context", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }

        return false;
    }

    private void OnLoad()
    {
        if (_window == null)
            return;

        _gl = _window.CreateOpenGL();
        _input = _window.CreateInput();
        _imGui = new ImGuiController(_gl, _window, _input);
        SyncImGuiWindowMetrics(_window.Size, _window.FramebufferSize);
        ImGui.GetIO().ConfigFlags |= ImGuiConfigFlags.DockingEnable;
        ImGui.GetIO().ConfigWindowsMoveFromTitleBarOnly = true;

        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Lequal);
        _gl.Disable(EnableCap.CullFace);

        _requestInitialLoad = _initialSession?.HasBootstrapInput() == true;
    }

    private M2GpuPreviewRenderer? EnsureM2GpuPreviewRenderer()
    {
        if (_gl == null)
            return null;

        _gpuPreviewRenderer ??= new M2GpuPreviewRenderer(_gl);
        return _gpuPreviewRenderer;
    }

    private WmoGpuPreviewRenderer? EnsureWmoGpuPreviewRenderer()
    {
        if (_gl == null)
            return null;

        _wmoGpuPreviewRenderer ??= new WmoGpuPreviewRenderer(_gl);
        return _wmoGpuPreviewRenderer;
    }

    private MdxGpuPreviewRenderer? EnsureMdxGpuPreviewRenderer()
    {
        if (_gl == null)
            return null;

        _mdxGpuPreviewRenderer ??= new MdxGpuPreviewRenderer(_gl);
        return _mdxGpuPreviewRenderer;
    }

    private ModelOutputGpuRenderer? EnsureModelOutputGpuRenderer()
    {
        if (_gl == null)
            return null;

        _modelOutputGpuRenderer ??= new ModelOutputGpuRenderer(_gl);
        return _modelOutputGpuRenderer;
    }

    private WorldGpuPreviewRenderer? EnsureWorldGpuPreviewRenderer()
    {
        if (_gl == null)
            return null;

        string clientRoot = _session.World.ClientRoot?.Trim() ?? string.Empty;
        if (string.IsNullOrWhiteSpace(clientRoot) || !Directory.Exists(clientRoot))
            return null;

        ViewerIoSourceKey sourceKey = ViewerIoSourceKey.Create(_session.World.ClientRoot, _session.World.BuildLabel, _session.World.LooseOverlayRoot);
        string sourceSignature = BuildWorldMinimapSourceSignature();
        return _worldSceneHost.EnsureRenderer(_gl, _viewerIoService, sourceKey, sourceSignature);
    }

    private WorldMinimapRenderer? EnsureWorldMinimapRenderer()
    {
        if (_gl == null)
            return null;

        string clientRoot = _session.World.ClientRoot?.Trim() ?? string.Empty;
        if (string.IsNullOrWhiteSpace(clientRoot) || !Directory.Exists(clientRoot))
            return null;

        ViewerIoSourceKey sourceKey = ViewerIoSourceKey.Create(_session.World.ClientRoot, _session.World.BuildLabel, _session.World.LooseOverlayRoot);

        string sourceSignature = BuildWorldMinimapSourceSignature();
        if (_worldMinimapRenderer != null && !string.Equals(sourceSignature, _worldMinimapSourceSignature, StringComparison.OrdinalIgnoreCase))
        {
            _worldMinimapRenderer.Dispose();
            _worldMinimapRenderer = null;
            _worldMinimapPanOffset = Vector2.Zero;
        }

        _worldMinimapSourceSignature = sourceSignature;
        _worldMinimapRenderer ??= new WorldMinimapRenderer(_gl, _viewerIoService, sourceKey);
        return _worldMinimapRenderer;
    }

    private void OnUpdate(double deltaSeconds)
    {
        _imGui?.Update((float)deltaSeconds);
        ProcessPendingBackgroundMetadataLoads();
        ProcessPendingWorldLoad();

        if (_requestInitialLoad)
        {
            _requestInitialLoad = false;
            LoadActiveWorkspace();
        }

        HandleOpenGameFolderDialog();
    }

    private void ProcessPendingBackgroundMetadataLoads()
    {
        ProcessPendingWorldMapDiscovery();
        ProcessPendingWorldSpawnPicker();
    }

    private void ProcessPendingWorldMapDiscovery()
    {
        if (_pendingWorldMapDiscoveryTask is not { IsCompleted: true } task)
            return;

        _pendingWorldMapDiscoveryTask = null;

        try
        {
            PendingWorldMapDiscoveryResult completed = task.GetAwaiter().GetResult();
            if (completed.Version != _worldMapDiscoveryVersion
                || !string.Equals(completed.Signature, _worldMapDiscoverySignature, StringComparison.OrdinalIgnoreCase))
                return;

            _discoveredWorldMaps = completed.Maps;
            _worldMapDiscoverySummary = completed.Summary;
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or InvalidOperationException or ArgumentException or NotSupportedException)
        {
            _discoveredWorldMaps = Array.Empty<DiscoveredLooseWorldMap>();
            _worldMapDiscoverySummary = $"Could not discover maps for the current client root: {ex.Message}";
        }
    }

    private void ProcessPendingWorldSpawnPicker()
    {
        if (_pendingWorldSpawnPickerTask is not { IsCompleted: true } task)
            return;

        _pendingWorldSpawnPickerTask = null;

        try
        {
            WorldSpawnPickerState completed = task.GetAwaiter().GetResult();
            if (completed.Version != _worldSpawnPickerVersion)
                return;

            if (_worldSpawnPickerState is not null
                && !string.Equals(completed.Signature, _worldSpawnPickerState.Signature, StringComparison.OrdinalIgnoreCase)
                && !string.Equals(completed.Signature, BuildWorldSpawnPickerSignature(), StringComparison.OrdinalIgnoreCase))
            {
                return;
            }

            _worldSpawnPickerState = completed;
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or InvalidOperationException or ArgumentException or NotSupportedException)
        {
            string signature = BuildWorldSpawnPickerSignature();
            _worldSpawnPickerState = new WorldSpawnPickerState(signature, $"Could not load the world spawn grid: {ex.Message}", null, _worldSpawnPickerVersion);
        }
    }

    private bool IsWorldLoadPending()
    {
        return _pendingWorldLoadTask is { IsCompleted: false };
    }

    private void InvalidatePendingWorldLoadState()
    {
        _pendingWorldLoadGeneration++;
        _pendingWorldLoadStopwatch = null;
        _pendingWorldLoadMapInput = string.Empty;
    }

    private void ProcessPendingWorldLoad()
    {
        if (_pendingWorldLoadTask is not { IsCompleted: true } task)
        {
            if (IsWorldLoadPending() && _pendingWorldLoadStopwatch != null)
            {
                _statusMessage = $"Opening world session for {_pendingWorldLoadMapInput}... {_pendingWorldLoadStopwatch.Elapsed.TotalSeconds:F1}s elapsed. World data is still being assembled on the CPU; the loaded tile frame is rendered by the GPU preview once that stage completes.";
            }

            return;
        }

        _pendingWorldLoadTask = null;
        _pendingWorldLoadStopwatch = null;
        _pendingWorldLoadMapInput = string.Empty;

        try
        {
            PendingWorldLoadResult completed = task.GetAwaiter().GetResult();
            if (completed.Generation != _pendingWorldLoadGeneration)
                return;

            ApplyLoadedWorldSession(completed.RuntimeFrame);
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or InvalidOperationException or ArgumentException or NotSupportedException)
        {
            if (_pendingWorldLoadGeneration <= 0)
                return;

            _lastError = ex.Message;
            _selectedWorldObject = null;
            _worldSceneHost.Clear();
            _worldViewCamera.ResetToIdentity();
            DeleteWorldTerrainPreviewTexture();
            _statusMessage = "World runtime bridge failed.";
        }
    }

    private unsafe void OnRender(double deltaSeconds)
    {
        if (_gl == null || _imGui == null || _window == null)
            return;

        AdvanceInteractivePreviewCameras((float)deltaSeconds);

        if (_currentPreview != null && _gpuPreviewRenderer?.HasRenderableGeometry == true && _session.WorkspaceMode == WowViewerWorkspaceMode.StandaloneM2)
            _gpuPreviewRenderer.Render(
                _session.VisualSize,
                _session.VisualSize,
                _m2InteractiveCamera.CurrentAzimuthDegrees,
                _m2InteractiveCamera.CurrentElevationDegrees,
                _m2InteractiveCamera.CurrentZoomFactor,
                _m2InteractiveCamera.CurrentTargetOffset);
        if (_currentWmoPreview != null && _wmoGpuPreviewRenderer?.HasRenderableGeometry == true && _session.WorkspaceMode == WowViewerWorkspaceMode.StandaloneWmo)
        {
            _wmoGpuPreviewRenderer.SetCameraSettings(new PreviewCameraSettings
            {
                Mode = PreviewCameraMode.Orbit,
                PresetName = null,
                AzimuthDegrees = _wmoInteractiveCamera.CurrentAzimuthDegrees,
                ElevationDegrees = _wmoInteractiveCamera.CurrentElevationDegrees,
                FieldOfViewDegrees = _session.WmoCameraFieldOfViewDegrees,
                ZoomFactor = _wmoInteractiveCamera.CurrentZoomFactor,
                TargetOffset = _wmoInteractiveCamera.CurrentTargetOffset,
            });

            _wmoGpuPreviewRenderer.Render(_session.VisualSize, _session.VisualSize);
        }

        if (_currentMdxPreview != null && _mdxGpuPreviewRenderer?.HasRenderableGeometry == true && _session.WorkspaceMode == WowViewerWorkspaceMode.StandaloneMdx)
        {
            if (_session.MdxCameraMode == PreviewCameraMode.Orbit)
            {
                _mdxGpuPreviewRenderer.SetCameraSettings(new PreviewCameraSettings
                {
                    Mode = PreviewCameraMode.Orbit,
                    PresetName = null,
                    AzimuthDegrees = _mdxInteractiveCamera.CurrentAzimuthDegrees,
                    ElevationDegrees = _mdxInteractiveCamera.CurrentElevationDegrees,
                    FieldOfViewDegrees = _session.MdxCameraFieldOfViewDegrees,
                    ZoomFactor = _mdxInteractiveCamera.CurrentZoomFactor,
                    TargetOffset = _mdxInteractiveCamera.CurrentTargetOffset,
                });
            }

            _mdxGpuPreviewRenderer.Render(_session.VisualSize, _session.VisualSize, deltaSeconds);
        }
        if (_currentModelOutputScene != null && _modelOutputGpuRenderer?.HasRenderableGeometry == true && _session.WorkspaceMode == WowViewerWorkspaceMode.ModelOutputs)
            _modelOutputGpuRenderer.Render(_session.VisualSize, _session.VisualSize, BuildModelOutputCameraFrame());
        if (_currentWorldRuntimeFrame != null && _worldGpuPreviewRenderer?.HasRenderableGeometry == true && _session.WorkspaceMode == WowViewerWorkspaceMode.WorldSession)
            _worldGpuPreviewRenderer.Render(_session.VisualSize, _session.VisualSize, _worldViewCamera);

        _gl.Viewport(_window.FramebufferSize);
        _gl.ClearColor(0.08f, 0.09f, 0.11f, 1.0f);
        _gl.Clear((uint)(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit));

        DrawUi((float)deltaSeconds);
        _imGui.Render();
    }

    private void OnWindowResize(Vector2D<int> size)
    {
        SyncImGuiWindowMetrics(size, _window?.FramebufferSize ?? size);
    }

    private void OnFramebufferResize(Vector2D<int> size)
    {
        if (_gl != null)
            _gl.Viewport(size);

        SyncImGuiWindowMetrics(_window?.Size ?? size, size);
    }

    private void DrawUi(float deltaSeconds)
    {
        DrawMainMenuBar();

        if (_useFixedThreeLaneShell)
        {
            DrawFixedThreeLaneShell(deltaSeconds);
            if (_showFileBrowserWindow && IsStandaloneAssetWorkspace(_session.WorkspaceMode))
                DrawAssetFileBrowserWindow();
            if (_showWorldMapBrowserWindow)
                DrawWorldMapBrowserWindow();
            DrawGlobalStatusBar();
            return;
        }

        ImGui.DockSpaceOverViewport();

        if (_showWorkspaceWindow)
            DrawWorkspaceWindow();
        if (_showControlWindow)
            DrawControlWindow();
        if (_showFileBrowserWindow && IsStandaloneAssetWorkspace(_session.WorkspaceMode))
            DrawAssetFileBrowserWindow();
        if (_showWorldMapBrowserWindow)
            DrawWorldMapBrowserWindow();
        DrawPreviewWindow();
        if (_showDiagnosticsWindow)
            DrawDiagnosticsWindow(deltaSeconds);

        DrawGlobalStatusBar();
    }


    private void DrawGlobalStatusBar()
    {
        ImGuiViewportPtr viewport = ImGui.GetMainViewport();
        float barHeight = ImGui.GetFrameHeightWithSpacing() + 6.0f;
        Vector2 barPos = new(viewport.Pos.X, viewport.Pos.Y + viewport.Size.Y - barHeight);
        Vector2 barSize = new(viewport.Size.X, barHeight);

        ImGui.SetNextWindowPos(barPos);
        ImGui.SetNextWindowSize(barSize);

        ImGuiWindowFlags flags = ImGuiWindowFlags.NoDocking
            | ImGuiWindowFlags.NoDecoration
            | ImGuiWindowFlags.NoMove
            | ImGuiWindowFlags.NoSavedSettings
            | ImGuiWindowFlags.NoNav
            | ImGuiWindowFlags.NoBringToFrontOnFocus
            | ImGuiWindowFlags.NoFocusOnAppearing
            | ImGuiWindowFlags.NoInputs;

        ImGui.PushStyleVar(ImGuiStyleVar.WindowRounding, 0.0f);
        ImGui.PushStyleVar(ImGuiStyleVar.WindowBorderSize, 0.0f);
        if (ImGui.Begin("##GlobalStatusBar", flags))
        {
            float fps = ImGui.GetIO().Framerate;
            float frameMs = fps > 0.0f ? 1000.0f / fps : 0.0f;
            double heapMb = GC.GetTotalMemory(false) / (1024.0 * 1024.0);
            int gpuCommands = GetActiveGpuCommandCount();

            string summary = $"FPS {fps:F1} | Frame {frameMs:F2} ms | Heap {heapMb:F1} MB | GPU Cmds {gpuCommands}";
            if (_currentWorldRuntimeFrame is not null)
            {
                int visibleObjects = _currentWorldRuntimeFrame.Visibility.VisibleWmos.Count + _currentWorldRuntimeFrame.Visibility.VisibleMdx.Count;
                summary += $" | World CPU {_currentWorldRuntimeFrame.Stats.TotalCpuMs:F2} ms | Visible {visibleObjects}";
            }

            summary += $" | {_session.GetWorkspaceLabel()}";
            ImGui.TextUnformatted(summary);
        }

        ImGui.End();
        ImGui.PopStyleVar(2);
    }

    private int GetActiveGpuCommandCount()
    {
        return _session.WorkspaceMode switch
        {
            WowViewerWorkspaceMode.StandaloneM2 => _gpuPreviewRenderer?.CommandCount ?? 0,
            WowViewerWorkspaceMode.StandaloneWmo => _wmoGpuPreviewRenderer?.CommandCount ?? 0,
            WowViewerWorkspaceMode.StandaloneMdx => _mdxGpuPreviewRenderer?.CommandCount ?? 0,
            WowViewerWorkspaceMode.ModelOutputs => _modelOutputGpuRenderer?.CommandCount ?? 0,
            WowViewerWorkspaceMode.WorldSession => _worldGpuPreviewRenderer?.MarkerCount ?? 0,
            _ => 0,
        };
    }

    private void DrawMainMenuBar()
    {
        if (!ImGui.BeginMainMenuBar())
            return;

        if (ImGui.BeginMenu("File"))
        {
            if (ImGui.MenuItem("Open Game Folder (MPQ)..."))
                PromptOpenGameFolder();

            if (ImGui.MenuItem("Browse Current Client Files...", enabled: !string.IsNullOrWhiteSpace(_session.Source.ArchiveRoot)))
                OpenAssetFileBrowserForCurrentSource(AssetFileBrowserFilter.SupportedAssets);

            if (ImGui.MenuItem("Browse Current Client Maps...", enabled: CanBrowseCurrentWorldMaps()))
                OpenWorldMapBrowserForCurrentClient();

            if (ImGui.BeginMenu("Open Saved Game Folder", _settings.KnownGoodClients.Count > 0))
            {
                foreach (var client in _settings.KnownGoodClients)
                {
                    if (ImGui.MenuItem($"{client.Name}##open_saved_{client.Path}"))
                        OpenSavedGameFolder(client);
                }
                ImGui.EndMenu();
            }

            if (ImGui.BeginMenu("Browse Saved Client Files", _settings.KnownGoodClients.Count > 0))
            {
                foreach (var client in _settings.KnownGoodClients)
                {
                    if (ImGui.MenuItem($"{client.Name}##browse_saved_{client.Path}"))
                        OpenAssetFileBrowserForClient(client.Path, client.BuildLabel, string.Empty, AssetFileBrowserFilter.SupportedAssets, $"Browse {client.Name}");
                }
                ImGui.EndMenu();
            }

            if (ImGui.BeginMenu("Browse Saved Client Maps", _settings.KnownGoodClients.Count > 0))
            {
                foreach (var client in _settings.KnownGoodClients)
                {
                    if (ImGui.MenuItem($"{client.Name}##browse_saved_maps_{client.Path}"))
                        OpenWorldMapBrowserForClient(client.Path, client.BuildLabel, string.Empty, $"Browse {client.Name} Maps");
                }
                ImGui.EndMenu();
            }

            if (ImGui.MenuItem("Save Current As Known-Good Base", enabled: !string.IsNullOrWhiteSpace(_session.Source.ArchiveRoot)))
                SaveCurrentAsKnownGood();

            if (ImGui.MenuItem("Attach Loose Folder...", enabled: !string.IsNullOrWhiteSpace(_session.Source.ArchiveRoot)))
                PromptAttachLooseFolder();

            if (ImGui.BeginMenu("Load Loose Folder Against Saved Base", _settings.KnownGoodClients.Count > 0))
            {
                foreach (var client in _settings.KnownGoodClients)
                {
                    if (ImGui.MenuItem($"{client.Name}##attach_saved_{client.Path}"))
                        QueueKnownGoodClientAction(client.Path, client.BuildLabel, attachLooseFolder: true, openBrowser: false, openWorldMapBrowser: false);
                }
                ImGui.EndMenu();
            }

            if (ImGui.BeginMenu("Browse Loose Folder Against Saved Base", _settings.KnownGoodClients.Count > 0))
            {
                foreach (var client in _settings.KnownGoodClients)
                {
                    if (ImGui.MenuItem($"{client.Name}##browse_loose_saved_{client.Path}"))
                        QueueKnownGoodClientAction(client.Path, client.BuildLabel, attachLooseFolder: true, openBrowser: true, openWorldMapBrowser: false);
                }
                ImGui.EndMenu();
            }

            if (ImGui.BeginMenu("Browse Loose World Maps Against Saved Base", _settings.KnownGoodClients.Count > 0))
            {
                foreach (var client in _settings.KnownGoodClients)
                {
                    if (ImGui.MenuItem($"{client.Name}##browse_loose_world_saved_{client.Path}"))
                        QueueKnownGoodClientAction(client.Path, client.BuildLabel, attachLooseFolder: true, openBrowser: false, openWorldMapBrowser: true);
                }
                ImGui.EndMenu();
            }

            if (ImGui.BeginMenu("Forget Known-Good Base", _settings.KnownGoodClients.Count > 0))
            {
                foreach (var client in _settings.KnownGoodClients)
                {
                    if (ImGui.MenuItem($"{client.Name}##forget_{client.Path}"))
                        ForgetKnownGoodClient(client.Path);
                }
                ImGui.EndMenu();
            }

            ImGui.Separator();

            if (ImGui.MenuItem("Open Workspace", enabled: !string.IsNullOrWhiteSpace(_session.Source.ArchiveRoot) || !string.IsNullOrWhiteSpace(_session.Source.InputPath) || _session.ModelOutput.HasInput() || _session.World.HasBootstrapInput()))
                LoadActiveWorkspace();

            if (ImGui.MenuItem("Clear Workspace", enabled: _currentPreview != null || _currentWmoPreview != null || _currentMdxPreview != null || _currentModelOutputScene != null || _currentWorldSession != null))
                ClearWorkspace();

            ImGui.Separator();

            if (ImGui.MenuItem("Exit"))
                _window?.Close();

            ImGui.EndMenu();
        }

        if (ImGui.BeginMenu("View"))
        {
            ImGui.MenuItem("Fixed Three-Lane Shell", string.Empty, ref _useFixedThreeLaneShell);
            ImGui.Separator();
            ImGui.MenuItem("Workspaces", string.Empty, ref _showWorkspaceWindow);
            ImGui.MenuItem("Source Controls", string.Empty, ref _showControlWindow);
            ImGui.MenuItem("File Browser", string.Empty, ref _showFileBrowserWindow);
            ImGui.MenuItem("World Map Browser", string.Empty, ref _showWorldMapBrowserWindow);
            ImGui.MenuItem("Diagnostics", string.Empty, ref _showDiagnosticsWindow);
            ImGui.MenuItem("Compact World Session Layout", string.Empty, ref _compactWorldSessionLayout);
            ImGui.EndMenu();
        }

        ImGui.TextDisabled($"{ProjectIdentity.SolutionName} {ProjectIdentity.PlannedVersion}");
        ImGui.EndMainMenuBar();
    }

    private void DrawFixedThreeLaneShell(float deltaSeconds)
    {
        ImGuiViewportPtr viewport = ImGui.GetMainViewport();
        float menuHeight = ImGui.GetFrameHeight();
        float statusHeight = ImGui.GetFrameHeightWithSpacing() + 6.0f;
        float top = viewport.Pos.Y + menuHeight;
        float height = MathF.Max(120f, viewport.Size.Y - menuHeight - statusHeight);
        float maxSidebarWidth = MathF.Max(260f, viewport.Size.X * 0.42f);
        _fixedShellNavigatorWidth = Math.Clamp(_fixedShellNavigatorWidth, 240f, maxSidebarWidth);
        _fixedShellInspectorWidth = Math.Clamp(_fixedShellInspectorWidth, 280f, maxSidebarWidth);
        float minimumPreviewWidth = MathF.Min(640f, viewport.Size.X * 0.45f);
        float sidebarOverflow = (_fixedShellNavigatorWidth + _fixedShellInspectorWidth + minimumPreviewWidth) - viewport.Size.X;
        if (sidebarOverflow > 0f)
        {
            float trim = sidebarOverflow * 0.5f;
            _fixedShellNavigatorWidth = MathF.Max(220f, _fixedShellNavigatorWidth - trim);
            _fixedShellInspectorWidth = MathF.Max(260f, _fixedShellInspectorWidth - trim);
        }

        float navigatorWidth = _fixedShellNavigatorWidth;
        float inspectorWidth = _fixedShellInspectorWidth;
        float previewWidth = MathF.Max(320f, viewport.Size.X - navigatorWidth - inspectorWidth);

        DrawFixedShellNavigatorLane(new Vector2(viewport.Pos.X, top), new Vector2(navigatorWidth, height));
        DrawFixedShellPreviewLane(new Vector2(viewport.Pos.X + navigatorWidth, top), new Vector2(previewWidth, height));
        DrawFixedShellInspectorLane(new Vector2(viewport.Pos.X + navigatorWidth + previewWidth, top), new Vector2(inspectorWidth, height), deltaSeconds);
        DrawFixedShellVerticalSplitter("##NavigatorPreviewSplitter", new Vector2(viewport.Pos.X + navigatorWidth - 3f, top), height, delta =>
        {
            _fixedShellNavigatorWidth = Math.Clamp(_fixedShellNavigatorWidth + delta, 220f, maxSidebarWidth);
        });
        DrawFixedShellVerticalSplitter("##PreviewInspectorSplitter", new Vector2(viewport.Pos.X + navigatorWidth + previewWidth - 3f, top), height, delta =>
        {
            _fixedShellInspectorWidth = Math.Clamp(_fixedShellInspectorWidth - delta, 260f, maxSidebarWidth);
        });
    }

    private static void DrawFixedShellVerticalSplitter(string id, Vector2 position, float height, Action<float> onDragged)
    {
        ImGui.SetNextWindowPos(position, ImGuiCond.Always);
        ImGui.SetNextWindowSize(new Vector2(6f, height), ImGuiCond.Always);
        ImGuiWindowFlags flags =
            ImGuiWindowFlags.NoDecoration |
            ImGuiWindowFlags.NoMove |
            ImGuiWindowFlags.NoSavedSettings |
            ImGuiWindowFlags.NoBackground |
            ImGuiWindowFlags.NoBringToFrontOnFocus;
        if (!ImGui.Begin(id, flags))
        {
            ImGui.End();
            return;
        }

        ImGui.InvisibleButton("##splitter", new Vector2(6f, height));
        if (ImGui.IsItemHovered() || ImGui.IsItemActive())
            ImGui.SetMouseCursor(ImGuiMouseCursor.ResizeEW);
        if (ImGui.IsItemActive())
            onDragged(ImGui.GetIO().MouseDelta.X);

        ImGui.End();
    }

    private void DrawFixedShellNavigatorLane(Vector2 position, Vector2 size)
    {
        ImGui.SetNextWindowPos(position, ImGuiCond.Always);
        ImGui.SetNextWindowSize(size, ImGuiCond.Always);
        ImGuiWindowFlags flags = ImGuiWindowFlags.NoMove | ImGuiWindowFlags.NoResize | ImGuiWindowFlags.NoCollapse | ImGuiWindowFlags.NoSavedSettings;
        if (!ImGui.Begin("Navigator", flags))
        {
            ImGui.End();
            return;
        }

        DrawWorkspaceNavigatorSection();
        ImGui.Separator();
        DrawActiveNavigatorContents();
        ImGui.End();
    }

    private void DrawFixedShellPreviewLane(Vector2 position, Vector2 size)
    {
        ImGui.SetNextWindowPos(position, ImGuiCond.Always);
        ImGui.SetNextWindowSize(size, ImGuiCond.Always);
        DrawPreviewWindow(forceDockedSize: true);
    }

    private void DrawFixedShellInspectorLane(Vector2 position, Vector2 size, float deltaSeconds)
    {
        ImGui.SetNextWindowPos(position, ImGuiCond.Always);
        ImGui.SetNextWindowSize(size, ImGuiCond.Always);
        DrawDiagnosticsWindow(deltaSeconds, forceDockedSize: true);
    }

    private void DrawWorkspaceNavigatorSection()
    {
        if (!ImGui.CollapsingHeader("Workspaces", ImGuiTreeNodeFlags.DefaultOpen))
            return;

        DrawCompactWorkspaceOption(WowViewerWorkspaceMode.WorldSession, "World session");
        DrawCompactWorkspaceOption(WowViewerWorkspaceMode.StandaloneM2, "M2");
        DrawCompactWorkspaceOption(WowViewerWorkspaceMode.StandaloneWmo, "WMO");
        DrawCompactWorkspaceOption(WowViewerWorkspaceMode.StandaloneMdx, "MDX");
        DrawCompactWorkspaceOption(WowViewerWorkspaceMode.ModelOutputs, "Model outputs");
        DrawCompactWorkspaceOption(WowViewerWorkspaceMode.DatasetTooling, "Dataset tooling");
    }

    private void DrawCompactWorkspaceOption(WowViewerWorkspaceMode mode, string description)
    {
        bool selected = _session.WorkspaceMode == mode;
        if (ImGui.Selectable(GetWorkspaceLabel(mode), selected))
        {
            if (_session.WorkspaceMode != mode)
            {
                _session.WorkspaceMode = mode;
                if (mode == WowViewerWorkspaceMode.WorldSession)
                    ApplyLegacyWorldSessionWindowPreset();
                _lastError = null;
                _statusMessage = $"{GetWorkspaceLabel(mode)} active in the reset shell.";
            }
        }

        if (ImGui.IsItemHovered())
            ImGui.SetTooltip(description);
    }

    private void DrawActiveNavigatorContents()
    {
        switch (_session.WorkspaceMode)
        {
            case WowViewerWorkspaceMode.StandaloneM2:
                DrawM2ControlContents();
                break;
            case WowViewerWorkspaceMode.StandaloneWmo:
                DrawWmoControlContents();
                break;
            case WowViewerWorkspaceMode.StandaloneMdx:
                DrawMdxControlContents();
                break;
            case WowViewerWorkspaceMode.WorldSession:
                DrawWorldControlContents();
                break;
            case WowViewerWorkspaceMode.DatasetTooling:
                DrawDatasetToolingControlContents();
                break;
            case WowViewerWorkspaceMode.ModelOutputs:
                DrawModelOutputControlContents();
                break;
            default:
                DrawPlaceholderControlContents();
                break;
        }

        DrawStatusSection();
    }

    private void DrawWorkspaceWindow()
    {
        ImGui.SetNextWindowSize(new Vector2(360, 420), ImGuiCond.FirstUseEver);
        if (!ImGui.Begin("Workspaces", ref _showWorkspaceWindow))
        {
            ImGui.End();
            return;
        }

        ImGui.TextWrapped("The viewer shell now exposes explicit standalone workspaces. M2, WMO, and MDX all have bounded GPU preview consumers in this slice, while world rendering remains a separate runtime track.");
        ImGui.Separator();

        if (ImGui.BeginChild("WorkspaceList", new Vector2(0, 0), false, ImGuiWindowFlags.AlwaysVerticalScrollbar))
        {
            DrawWorkspaceOption(WowViewerWorkspaceMode.StandaloneM2, "Runtime-backed standalone model preview over the shared wow-viewer M2 pipeline.");
            DrawWorkspaceOption(WowViewerWorkspaceMode.StandaloneWmo, "Standalone WMO mesh preview over the shared wow-viewer render-document seam.");
            DrawWorkspaceOption(WowViewerWorkspaceMode.StandaloneMdx, "Static standalone MDX inspection workspace with a first GPU preview consumer.");
            DrawWorkspaceOption(WowViewerWorkspaceMode.WorldSession, "Shared-client-root world session bootstrap with a first GPU terrain preview over the bounded runtime frame.");
            DrawWorkspaceOption(WowViewerWorkspaceMode.DatasetTooling, "Dataset and training orchestration owned by wow-viewer, including mask generation pipelines.");
            DrawWorkspaceOption(WowViewerWorkspaceMode.ModelOutputs, "GPU-backed viewer for flat OBJ + texture terrain tiles exported from model inference runs.");
        }
        ImGui.EndChild();

        ImGui.End();
    }

    private void DrawWorkspaceOption(WowViewerWorkspaceMode mode, string description)
    {
        bool selected = _session.WorkspaceMode == mode;
        if (ImGui.Selectable(GetWorkspaceLabel(mode), selected))
        {
            if (_session.WorkspaceMode != mode)
            {
                _session.WorkspaceMode = mode;
                if (mode == WowViewerWorkspaceMode.WorldSession)
                    ApplyLegacyWorldSessionWindowPreset();
                _lastError = null;
                _statusMessage = mode == WowViewerWorkspaceMode.DatasetTooling
                    ? "Dataset Tooling active. Use the control panel to launch mask generation and training pipelines."
                    : IsImplementedWorkspace(mode)
                        ? $"{GetWorkspaceLabel(mode)} active. Configure a source and load a preview."
                        : $"{GetWorkspaceLabel(mode)} is not implemented yet. This placeholder exists to keep the cutover honest about future standalone consumers.";
            }
        }

        ImGui.TextDisabled(description);
        ImGui.TextDisabled(IsImplementedWorkspace(mode) ? "Status: implemented in this slice" : "Status: placeholder only");
        ImGui.Separator();
    }

    private static bool IsImplementedWorkspace(WowViewerWorkspaceMode mode)
    {
        return mode is WowViewerWorkspaceMode.StandaloneM2 or WowViewerWorkspaceMode.StandaloneWmo or WowViewerWorkspaceMode.StandaloneMdx or WowViewerWorkspaceMode.WorldSession or WowViewerWorkspaceMode.DatasetTooling or WowViewerWorkspaceMode.ModelOutputs;
    }

    private static string GetWorkspaceLabel(WowViewerWorkspaceMode mode)
    {
        return mode switch
        {
            WowViewerWorkspaceMode.StandaloneM2 => "Standalone M2",
            WowViewerWorkspaceMode.StandaloneWmo => "Standalone WMO",
            WowViewerWorkspaceMode.StandaloneMdx => "Standalone MDX",
            WowViewerWorkspaceMode.WorldSession => "World Session",
            WowViewerWorkspaceMode.DatasetTooling => "Dataset Tooling",
            WowViewerWorkspaceMode.ModelOutputs => "Model Outputs",
            _ => "Unknown",
        };
    }

    private void DrawControlWindow()
    {
        ImGui.SetNextWindowSize(new Vector2(430, 540), ImGuiCond.FirstUseEver);
        string title = _session.WorkspaceMode == WowViewerWorkspaceMode.WorldSession
            ? "World Navigator"
            : $"{_session.GetWorkspaceLabel()} Controls";
        if (!ImGui.Begin(title, ref _showControlWindow))
        {
            ImGui.End();
            return;
        }

        switch (_session.WorkspaceMode)
        {
            case WowViewerWorkspaceMode.StandaloneM2:
                DrawM2ControlContents();
                break;
            case WowViewerWorkspaceMode.StandaloneWmo:
                DrawWmoControlContents();
                break;
            case WowViewerWorkspaceMode.StandaloneMdx:
                DrawMdxControlContents();
                break;
            case WowViewerWorkspaceMode.WorldSession:
                DrawWorldControlContents();
                break;
            case WowViewerWorkspaceMode.DatasetTooling:
                DrawDatasetToolingControlContents();
                break;
            case WowViewerWorkspaceMode.ModelOutputs:
                DrawModelOutputControlContents();
                break;
            default:
                DrawPlaceholderControlContents();
                break;
        }

        DrawStatusSection();

        ImGui.End();
    }

    private void DrawM2ControlContents()
    {
        ImGui.TextWrapped("This first `wow-viewer` desktop shell stays library-first: it loads M2 assets only through the `wow-viewer` runtime pipeline and now exposes a bounded GPU preview consumer plus the existing deterministic software/runtime diagnostics.");
        ImGui.TextDisabled("Mouse: left-drag orbit, right/middle-drag pan, wheel zoom, double-click preview to reset camera.");
        ImGui.Separator();
        ImGui.TextDisabled($"Workspace: {_session.GetWorkspaceLabel()}");
        ImGui.TextDisabled($"Source: {_session.Source.Describe()}");
        ImGui.Separator();

        bool useArchive = _session.Source.UsesArchiveSource;
        if (ImGui.RadioButton("Game client input", useArchive))
            _session.Source.Kind = WowViewerAssetSourceKind.ArchiveVirtualPath;
        ImGui.SameLine();
        if (ImGui.RadioButton("Local file input", !_session.Source.UsesArchiveSource))
            _session.Source.Kind = WowViewerAssetSourceKind.LocalFile;

        string sharedClientRoot = _session.Source.ArchiveRoot;
        string sharedLooseOverlayRoot = _session.Source.LooseOverlayRoot;
        string buildLabel = _session.Source.BuildLabel;

        ImGui.Separator();
        if (_session.Source.UsesArchiveSource)
        {
            string virtualPath = _session.Source.VirtualPath;
            ImGui.InputText("Client Root", ref sharedClientRoot, 1024);
            ImGui.InputText("Virtual Path", ref virtualPath, 1024);
            ImGui.InputText("Loose Overlay Root", ref sharedLooseOverlayRoot, 1024);
            _session.Source.VirtualPath = virtualPath;

            if (ImGui.Button("Attach Loose Folder...", new Vector2(-1, 0)))
                PromptAttachLooseFolder();

            if (ImGui.Button("Browse Client Files...", new Vector2(-1, 0)))
                OpenAssetFileBrowserForCurrentSource(AssetFileBrowserFilter.M2);
        }
        else
        {
            string inputPath = _session.Source.InputPath;
            ImGui.InputText("Input File", ref inputPath, 1024);
            _session.Source.InputPath = inputPath;

            if (ImGui.Button("Browse File...", new Vector2(-1, 0)))
                PromptOpenLocalSourceFile();
        }

        if (_session.Source.UsesArchiveSource && ImGui.Button("Open Game Client...", new Vector2(-1, 0)))
            PromptOpenGameFolder();

        ImGui.InputText("Build Label", ref buildLabel, 256);
        ApplySharedClientSelection(sharedClientRoot, buildLabel, sharedLooseOverlayRoot);

        int profileIndex = _session.ProfileIndex;
        int sequenceIndex = _session.SequenceIndex;
        int timeMs = _session.TimeMs;
        int visualSize = _session.VisualSize;
        ImGui.InputInt("Profile Index", ref profileIndex);
        ImGui.InputInt("Sequence Index", ref sequenceIndex);
        ImGui.InputInt("Time (ms)", ref timeMs);
        ImGui.InputInt("Preview Size", ref visualSize);
        _session.ProfileIndex = profileIndex;
        _session.SequenceIndex = sequenceIndex;
        _session.TimeMs = timeMs;
        _session.VisualSize = visualSize;
        _session.Normalize();

        ImGui.Separator();
        ImGui.TextDisabled($"Camera: azimuth {_session.M2CameraAzimuthDegrees:F1} deg, elevation {_session.M2CameraElevationDegrees:F1} deg, zoom {_session.M2CameraZoomFactor:F2}");
        Vector3 m2TargetOffset = _session.GetM2CameraTargetOffset();
        ImGui.TextDisabled($"Target Offset: {m2TargetOffset.X:F1}, {m2TargetOffset.Y:F1}, {m2TargetOffset.Z:F1}");

        if (ImGui.Button("Load M2 Preview", new Vector2(-1, 0)))
            LoadActiveWorkspace();

        if (ImGui.Button("Reset Camera", new Vector2(-1, 0)))
            ResetM2Camera();

        if (ImGui.Button("Use Wolf Runtime Baseline", new Vector2(-1, 0)))
        {
            _session.WorkspaceMode = WowViewerWorkspaceMode.StandaloneM2;
            ApplySharedClientSelection(@"H:\CLIENTS\WoW335\3.X_Retail_Windows_enUS_3.3.5.12340\World of Warcraft", "3.3.5.12340", string.Empty, preferArchiveSource: true);
            _session.Source.VirtualPath = @"Creature/Wolf/Wolf.m2";
            _session.ProfileIndex = 0;
            _session.SequenceIndex = 0;
            _session.TimeMs = 0;
            _session.VisualSize = 384;
        }

        if (ImGui.Button("Use Camera Overlay Baseline", new Vector2(-1, 0)))
        {
            _session.WorkspaceMode = WowViewerWorkspaceMode.StandaloneM2;
            ApplySharedClientSelection(@"H:\CLIENTS\WoW335\3.X_Retail_Windows_enUS_3.3.5.12340\World of Warcraft", "3.3.5.12340", string.Empty, preferArchiveSource: true);
            _session.Source.VirtualPath = @"Cameras/Scry_cam.m2";
            _session.ProfileIndex = 0;
            _session.SequenceIndex = 0;
            _session.TimeMs = 0;
            _session.VisualSize = 384;
        }
    }

    private void DrawMdxControlContents()
    {
        ImGui.TextWrapped("This first standalone MDX slice stays narrow and GPU-first: it uses wow-viewer-owned MDX geometry and summary readers to drive a static OpenGL preview without claiming full animation or world-scene closure yet.");
        ImGui.TextDisabled("Mouse: drag the preview to switch to orbit and control the camera in GPU space. Double-click preview to reset orbit camera.");
        ImGui.Separator();
        ImGui.TextDisabled($"Workspace: {_session.GetWorkspaceLabel()}");
        ImGui.TextDisabled($"Source: {_session.Source.Describe()}");
        ImGui.Separator();

        bool useArchive = _session.Source.UsesArchiveSource;
        if (ImGui.RadioButton("Game client input", useArchive))
            _session.Source.Kind = WowViewerAssetSourceKind.ArchiveVirtualPath;
        ImGui.SameLine();
        if (ImGui.RadioButton("Local file input", !_session.Source.UsesArchiveSource))
            _session.Source.Kind = WowViewerAssetSourceKind.LocalFile;

        string sharedClientRoot = _session.Source.ArchiveRoot;
        string sharedLooseOverlayRoot = _session.Source.LooseOverlayRoot;
        string buildLabel = _session.Source.BuildLabel;

        ImGui.Separator();
        if (_session.Source.UsesArchiveSource)
        {
            ImGui.InputText("Client Root", ref sharedClientRoot, 1024);
            string virtualPath = _session.Source.VirtualPath;
            ImGui.InputText("Virtual Path", ref virtualPath, 1024);
            _session.Source.VirtualPath = virtualPath;

            ImGui.InputText("Loose Overlay Root", ref sharedLooseOverlayRoot, 1024);

            if (ImGui.Button("Attach Loose Folder...", new Vector2(-1, 0)))
                PromptAttachLooseFolder();

            if (ImGui.Button("Browse Client Files...", new Vector2(-1, 0)))
                OpenAssetFileBrowserForCurrentSource(AssetFileBrowserFilter.Mdx);
        }
        else
        {
            string inputPath = _session.Source.InputPath;
            ImGui.InputText("Input File", ref inputPath, 1024);
            _session.Source.InputPath = inputPath;

            if (ImGui.Button("Browse File...", new Vector2(-1, 0)))
                PromptOpenLocalSourceFile();
        }

        if (_session.Source.UsesArchiveSource && ImGui.Button("Open Game Client...", new Vector2(-1, 0)))
            PromptOpenGameFolder();

        ImGui.InputText("Build Label", ref buildLabel, 256);
        ApplySharedClientSelection(sharedClientRoot, buildLabel, sharedLooseOverlayRoot);

        int visualSize = _session.VisualSize;
        ImGui.InputInt("Preview Size", ref visualSize);
        _session.VisualSize = visualSize;
        _session.Normalize();

        ImGui.Separator();
        ImGui.TextDisabled("MDX Camera");
        int cameraModeIndex = (int)_session.MdxCameraMode;
        if (ImGui.Combo("Camera Mode", ref cameraModeIndex, "Frame\0Orbit\0Model\0"))
            _session.MdxCameraMode = (PreviewCameraMode)cameraModeIndex;

        float cameraFov = _session.MdxCameraFieldOfViewDegrees;
        if (ImGui.SliderFloat("Camera FOV", ref cameraFov, 20.0f, 90.0f, "%.0f deg"))
            _session.MdxCameraFieldOfViewDegrees = cameraFov;

        if (_session.MdxCameraMode == PreviewCameraMode.Orbit)
        {
            int presetIndex = GetMdxCameraPresetIndex(_session.MdxCameraPreset);
            if (ImGui.Combo("Orbit Preset", ref presetIndex, string.Join('\0', MdxCameraPresetLabels) + '\0'))
                _session.MdxCameraPreset = MdxCameraPresetValues[presetIndex] ?? string.Empty;

            ImGui.TextDisabled($"Orbit: azimuth {_session.MdxCameraAzimuthDegrees:F1} deg, elevation {_session.MdxCameraElevationDegrees:F1} deg, zoom {_session.MdxCameraZoomFactor:F2}");
            Vector3 mdxTargetOffset = _session.GetMdxCameraTargetOffset();
            ImGui.TextDisabled($"Target Offset: {mdxTargetOffset.X:F1}, {mdxTargetOffset.Y:F1}, {mdxTargetOffset.Z:F1}");
        }

        _session.Normalize();

        if (ImGui.Button(_currentMdxPreview == null ? "Load MDX Preview" : "Reload MDX Preview", new Vector2(-1, 0)))
            LoadActiveWorkspace();

        if (_session.MdxCameraMode == PreviewCameraMode.Orbit && ImGui.Button("Reset Orbit Camera", new Vector2(-1, 0)))
            ResetMdxOrbitCamera();

        ImGui.Separator();

        if (_session.Source.UsesArchiveSource && ImGui.Button("Browse Client Files...", new Vector2(-1, 0)))
            OpenAssetFileBrowserForCurrentSource(AssetFileBrowserFilter.Mdx);
    }

    private void DrawWmoControlContents()
    {
        ImGui.TextWrapped("This standalone WMO slice stays bounded and library-first: it now uses wow-viewer-owned WMO material plus mesh readers to drive a textured-when-resolved GPU batch preview while exposing root portal and doodad ownership in diagnostics.");
        ImGui.TextDisabled("Mouse: left-drag orbit, right/middle-drag pan, wheel zoom, double-click preview to reset camera.");
        ImGui.Separator();
        ImGui.TextDisabled($"Workspace: {_session.GetWorkspaceLabel()}");
        ImGui.TextDisabled($"Source: {_session.Source.Describe()}");
        ImGui.Separator();

        bool useArchive = _session.Source.UsesArchiveSource;
        if (ImGui.RadioButton("Game client input", useArchive))
            _session.Source.Kind = WowViewerAssetSourceKind.ArchiveVirtualPath;
        ImGui.SameLine();
        if (ImGui.RadioButton("Local file input", !_session.Source.UsesArchiveSource))
            _session.Source.Kind = WowViewerAssetSourceKind.LocalFile;

        string sharedClientRoot = _session.Source.ArchiveRoot;
        string sharedLooseOverlayRoot = _session.Source.LooseOverlayRoot;
        string buildLabel = _session.Source.BuildLabel;

        ImGui.Separator();
        if (_session.Source.UsesArchiveSource)
        {
            string virtualPath = _session.Source.VirtualPath;
            ImGui.InputText("Client Root", ref sharedClientRoot, 1024);
            ImGui.InputText("Virtual Path", ref virtualPath, 1024);
            ImGui.InputText("Loose Overlay Root", ref sharedLooseOverlayRoot, 1024);
            _session.Source.VirtualPath = virtualPath;

            if (ImGui.Button("Attach Loose Folder...", new Vector2(-1, 0)))
                PromptAttachLooseFolder();

            if (ImGui.Button("Browse Client Files...", new Vector2(-1, 0)))
                OpenAssetFileBrowserForCurrentSource(AssetFileBrowserFilter.Wmo);
        }
        else
        {
            string inputPath = _session.Source.InputPath;
            ImGui.InputText("Input File", ref inputPath, 1024);
            _session.Source.InputPath = inputPath;

            if (ImGui.Button("Browse File...", new Vector2(-1, 0)))
                PromptOpenLocalSourceFile();
        }

        if (_session.Source.UsesArchiveSource && ImGui.Button("Open Game Client...", new Vector2(-1, 0)))
            PromptOpenGameFolder();

        ImGui.InputText("Build Label", ref buildLabel, 256);
        ApplySharedClientSelection(sharedClientRoot, buildLabel, sharedLooseOverlayRoot);

        int visualSize = _session.VisualSize;
        ImGui.InputInt("Preview Size", ref visualSize);
        _session.VisualSize = visualSize;

        float cameraFov = _session.WmoCameraFieldOfViewDegrees;
        if (ImGui.SliderFloat("Camera FOV", ref cameraFov, 20.0f, 90.0f, "%.0f deg"))
            _session.WmoCameraFieldOfViewDegrees = cameraFov;

        _session.Normalize();
        ImGui.TextDisabled($"Orbit: azimuth {_session.WmoCameraAzimuthDegrees:F1} deg, elevation {_session.WmoCameraElevationDegrees:F1} deg, zoom {_session.WmoCameraZoomFactor:F2}");
        Vector3 targetOffset = _session.GetWmoCameraTargetOffset();
        ImGui.TextDisabled($"Target Offset: {targetOffset.X:F1}, {targetOffset.Y:F1}, {targetOffset.Z:F1}");

        if (ImGui.Button(_currentWmoPreview == null ? "Load WMO Preview" : "Reload WMO Preview", new Vector2(-1, 0)))
            LoadActiveWorkspace();

        if (ImGui.Button("Reset Orbit Camera", new Vector2(-1, 0)))
            ResetWmoOrbitCamera();
    }

    private void DrawAssetFileBrowserWindow()
    {
        _assetFileBrowserState ??= new AssetFileBrowserState();

        if (FileBrowserEx.DrawAssetFileBrowser(_fileBrowserTitle, ref _showFileBrowserWindow, _fileBrowserClientRoot, _fileBrowserLooseOverlayRoot, _assetFileBrowserState, _fileBrowserFilter, OnAssetFileBrowserFileSelected))
        {
            // Selection callback updates workspace and source state.
        }
    }

    private void OnAssetFileBrowserFileSelected(string virtualPath)
    {
        WowViewerWorkspaceMode workspace = DetectWorkspaceModeFromVirtualPath(virtualPath);
        _session.WorkspaceMode = workspace;
        ApplySharedClientSelection(_fileBrowserClientRoot, _fileBrowserBuildLabel, _fileBrowserLooseOverlayRoot, preferArchiveSource: true);
        _session.Source.VirtualPath = virtualPath;
        _settings.LastOpenedClientPath = _fileBrowserClientRoot;
        if (!string.IsNullOrWhiteSpace(_fileBrowserLooseOverlayRoot))
            _settings.LastOpenedLooseOverlayPath = _fileBrowserLooseOverlayRoot;
        SaveSettings();
        LoadActiveWorkspace();
    }

    private void DrawWorldMapBrowserWindow()
    {
        RefreshWorldMapBrowser();

        ImGui.SetNextWindowSize(new Vector2(720, 520), ImGuiCond.FirstUseEver);
        if (!ImGui.Begin(_worldMapBrowserTitle, ref _showWorldMapBrowserWindow))
        {
            ImGui.End();
            return;
        }

        ImGui.TextDisabled($"Client Root: {_worldMapBrowserClientRoot}");
        if (!string.IsNullOrWhiteSpace(_worldMapBrowserBuildLabel))
            ImGui.TextDisabled($"Build: {_worldMapBrowserBuildLabel}");
        if (!string.IsNullOrWhiteSpace(_worldMapBrowserLooseOverlayRoot))
            ImGui.TextDisabled($"Loose Overlay: {_worldMapBrowserLooseOverlayRoot}");

        string filter = _worldMapBrowserFilter;
        ImGui.InputText("Filter", ref filter, 256);
        _worldMapBrowserFilter = filter;
        ImGui.TextWrapped(_worldMapBrowserSummary);
        ImGui.Separator();

        if (ImGui.BeginChild("##WorldMapBrowserList", new Vector2(0, 0), true))
        {
            foreach (DiscoveredLooseWorldMap map in GetFilteredWorldMapBrowserMaps())
            {
                string label = BuildWorldMapLabel(map);

                if (ImGui.Selectable(label, false))
                {
                    _session.WorkspaceMode = WowViewerWorkspaceMode.WorldSession;
                    ApplySharedClientSelection(_worldMapBrowserClientRoot, _worldMapBrowserBuildLabel, _worldMapBrowserLooseOverlayRoot);
                    _session.World.MapInput = map.Directory;
                    _session.World.TileX = -1;
                    _session.World.TileY = -1;
                    _showWorldMapBrowserWindow = false;
                    _worldSpawnPickerState = null;
                    _statusMessage = $"Selected {map.Directory}. Choose a spawn tile in World Navigator, then open the world session.";
                    break;
                }

                if (ImGui.IsItemHovered())
                    ImGui.SetTooltip(BuildWorldMapTooltip(map));
            }
        }

        ImGui.EndChild();
        ImGui.End();
    }

    private void OpenAssetFileBrowserForCurrentSource(AssetFileBrowserFilter filter)
    {
        if (string.IsNullOrWhiteSpace(_session.Source.ArchiveRoot))
        {
            _statusMessage = "Load or open a game client before browsing files.";
            return;
        }

        string title = _session.WorkspaceMode switch
        {
            WowViewerWorkspaceMode.StandaloneM2 => "M2 File Browser",
            WowViewerWorkspaceMode.StandaloneWmo => "WMO File Browser",
            WowViewerWorkspaceMode.StandaloneMdx => "MDX File Browser",
            _ => "Asset File Browser",
        };

        OpenAssetFileBrowserForClient(_session.Source.ArchiveRoot, _session.Source.BuildLabel, _session.Source.LooseOverlayRoot, filter, title);
    }

    private void OpenAssetFileBrowserForClient(string clientRoot, string? buildLabel, string? looseOverlayRoot, AssetFileBrowserFilter filter, string title)
    {
        _fileBrowserClientRoot = clientRoot?.Trim() ?? string.Empty;
        _fileBrowserBuildLabel = buildLabel?.Trim() ?? string.Empty;
        _fileBrowserLooseOverlayRoot = looseOverlayRoot?.Trim() ?? string.Empty;
        _fileBrowserFilter = filter;
        _fileBrowserTitle = title;
        _showFileBrowserWindow = true;
        _assetFileBrowserState ??= new AssetFileBrowserState();
    }

    private static WowViewerWorkspaceMode DetectWorkspaceModeFromVirtualPath(string virtualPath)
    {
        if (virtualPath.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase)
            || virtualPath.EndsWith(".wmo.mpq", StringComparison.OrdinalIgnoreCase))
        {
            return WowViewerWorkspaceMode.StandaloneWmo;
        }

        if (virtualPath.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase))
            return WowViewerWorkspaceMode.StandaloneMdx;

        return WowViewerWorkspaceMode.StandaloneM2;
    }

    private static bool IsStandaloneAssetWorkspace(WowViewerWorkspaceMode mode)
    {
        return mode is WowViewerWorkspaceMode.StandaloneM2
            or WowViewerWorkspaceMode.StandaloneWmo
            or WowViewerWorkspaceMode.StandaloneMdx;
    }

    private void ApplySharedClientSelection(string? clientRoot, string? buildLabel, string? looseOverlayRoot, bool preferArchiveSource = false, bool clearArchiveVirtualPath = false)
    {
        string normalizedClientRoot = clientRoot?.Trim() ?? string.Empty;
        string normalizedBuildLabel = buildLabel?.Trim() ?? string.Empty;
        string normalizedLooseOverlayRoot = looseOverlayRoot?.Trim() ?? string.Empty;

        bool changed = !string.Equals(_session.Source.ArchiveRoot, normalizedClientRoot, StringComparison.OrdinalIgnoreCase)
            || !string.Equals(_session.Source.BuildLabel, normalizedBuildLabel, StringComparison.OrdinalIgnoreCase)
            || !string.Equals(_session.Source.LooseOverlayRoot, normalizedLooseOverlayRoot, StringComparison.OrdinalIgnoreCase)
            || !string.Equals(_session.World.ClientRoot, normalizedClientRoot, StringComparison.OrdinalIgnoreCase)
            || !string.Equals(_session.World.BuildLabel, normalizedBuildLabel, StringComparison.OrdinalIgnoreCase)
            || !string.Equals(_session.World.LooseOverlayRoot, normalizedLooseOverlayRoot, StringComparison.OrdinalIgnoreCase)
            || (preferArchiveSource && _session.Source.Kind != WowViewerAssetSourceKind.ArchiveVirtualPath)
            || (clearArchiveVirtualPath && !string.IsNullOrEmpty(_session.Source.VirtualPath));

        if (preferArchiveSource)
            _session.Source.Kind = WowViewerAssetSourceKind.ArchiveVirtualPath;

        _session.Source.ArchiveRoot = normalizedClientRoot;
        _session.Source.BuildLabel = normalizedBuildLabel;
        _session.Source.LooseOverlayRoot = normalizedLooseOverlayRoot;
        if (clearArchiveVirtualPath)
            _session.Source.VirtualPath = string.Empty;

        _session.World.ClientRoot = normalizedClientRoot;
        _session.World.BuildLabel = normalizedBuildLabel;
        _session.World.LooseOverlayRoot = normalizedLooseOverlayRoot;

        if (changed)
        {
            ApplyWorldSessionAdtFirstDefaults();
            InvalidateSharedClientCaches();
        }
    }

    private void ApplyWorldSessionAdtFirstDefaults()
    {
        _session.World.ShowTerrain = true;
        _session.World.ShowWdl = false;
    }

    private void InvalidateSharedClientCaches()
    {
        _assetFileBrowserState = null;
        _worldMapDiscoverySignature = string.Empty;
        _worldMapBrowserSignature = string.Empty;
        _worldMapDiscoveryVersion++;
        _worldSpawnPickerVersion++;
        _worldSpawnPickerState = null;
        _discoveredWorldMaps = Array.Empty<DiscoveredLooseWorldMap>();
        _worldMapBrowserMaps = Array.Empty<DiscoveredLooseWorldMap>();
        _worldMapDiscoverySummary = "No client maps discovered yet.";
        _worldMapBrowserSummary = "No world maps discovered yet.";
    }

    private void PromptOpenGameFolder()
    {
        _wantOpenGameFolder = true;
    }

    private void PromptAttachLooseFolder()
    {
        _wantAttachLooseFolder = true;
    }

    private void PromptOpenLocalSourceFile()
    {
        (string title, WindowsNativeFileDialogs.FileDialogFilter[] filters) = GetLocalSourceFileDialogSpec();
        string? initialDirectory = GetLocalSourceInitialDirectory();
        string? selectedPath = TryShowOpenFileDialog(title, initialDirectory, filters);
        if (string.IsNullOrWhiteSpace(selectedPath) || !File.Exists(selectedPath))
            return;

        _session.Source.Kind = WowViewerAssetSourceKind.LocalFile;
        _session.Source.InputPath = selectedPath;
        _statusMessage = $"Opened local file: {selectedPath}";
    }

    private void QueueKnownGoodClientAction(string gamePath, string? buildLabel, bool attachLooseFolder, bool openBrowser, bool openWorldMapBrowser)
    {
        _pendingKnownGoodClientPath = gamePath;
        _pendingKnownGoodClientBuildLabel = buildLabel;
        _pendingKnownGoodClientAttachLooseFolder = attachLooseFolder;
        _pendingKnownGoodClientOpenBrowser = openBrowser;
        _pendingKnownGoodClientOpenWorldMapBrowser = openWorldMapBrowser;
    }

    private void OpenSavedGameFolder(KnownGoodClientEntry client)
    {
        ApplySharedClientSelection(client.Path, client.BuildLabel, string.Empty, preferArchiveSource: true, clearArchiveVirtualPath: true);
        _settings.LastOpenedClientPath = client.Path;
        SaveSettings();
        _statusMessage = $"Loaded saved client: {client.Name}";
    }

    private void AttachLooseFolder(string selectedPath)
    {
        if (string.IsNullOrWhiteSpace(_session.Source.ArchiveRoot))
        {
            _statusMessage = "Load a base game folder before attaching a loose overlay folder.";
            return;
        }

        string normalizedRoot = Path.GetFullPath(selectedPath);
        if (!Directory.Exists(normalizedRoot))
        {
            _statusMessage = $"Loose overlay folder does not exist: {selectedPath}";
            return;
        }

        ApplySharedClientSelection(_session.Source.ArchiveRoot, _session.Source.BuildLabel, normalizedRoot);
        _settings.LastOpenedLooseOverlayPath = normalizedRoot;
        SaveSettings();
        _statusMessage = $"Attached loose overlay: {normalizedRoot}";
    }

    private void SaveCurrentAsKnownGood()
    {
        var path = _session.Source.ArchiveRoot;
        if (string.IsNullOrWhiteSpace(path))
            return;

        string name = Path.GetFileName(path.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
        string buildLabel = _session.Source.BuildLabel;

        var existing = _settings.KnownGoodClients.FindIndex(c =>
            string.Equals(c.Path, path, StringComparison.OrdinalIgnoreCase));

        var entry = new KnownGoodClientEntry
        {
            Path = path,
            Name = name,
            BuildLabel = buildLabel,
            BuildVersion = buildLabel,
        };

        if (existing >= 0)
            _settings.KnownGoodClients[existing] = entry;
        else
            _settings.KnownGoodClients.Add(entry);

        SaveSettings();
        _statusMessage = $"Saved '{name}' as known-good base.";
    }

    private void ForgetKnownGoodClient(string path)
    {
        _settings.KnownGoodClients.RemoveAll(c =>
            string.Equals(c.Path, path, StringComparison.OrdinalIgnoreCase));
        SaveSettings();
        _statusMessage = "Removed known-good base.";
    }

    private void HandleOpenGameFolderDialog()
    {
        if (_wantOpenGameFolder)
        {
            _wantOpenGameFolder = false;

            string? selectedPath = TryShowFolderDialog(
                "Select WoW game folder (containing Data/ with MPQs)",
                _settings.LastOpenedClientPath,
                showNewFolderButton: false);

            if (!string.IsNullOrEmpty(selectedPath) && Directory.Exists(selectedPath))
            {
                ApplySharedClientSelection(selectedPath, string.Empty, string.Empty, preferArchiveSource: true, clearArchiveVirtualPath: true);
                _settings.LastOpenedClientPath = selectedPath;
                SaveSettings();
                _statusMessage = $"Opened game folder: {selectedPath}";
            }
            else if (!string.IsNullOrWhiteSpace(selectedPath))
            {
                _statusMessage = $"Selected game folder does not exist: {selectedPath}";
            }
        }

        if (_wantAttachLooseFolder)
        {
            _wantAttachLooseFolder = false;

            string? overlayPath = TryShowFolderDialog(
                "Select loose overlay folder",
                _settings.LastOpenedLooseOverlayPath,
                showNewFolderButton: false);

            if (!string.IsNullOrWhiteSpace(overlayPath))
                AttachLooseFolder(overlayPath);
        }

        if (!string.IsNullOrWhiteSpace(_pendingKnownGoodClientPath))
        {
            string savedBasePath = _pendingKnownGoodClientPath!;
            string? savedBuildLabel = _pendingKnownGoodClientBuildLabel;
            bool attachLooseFolder = _pendingKnownGoodClientAttachLooseFolder;
            bool openBrowser = _pendingKnownGoodClientOpenBrowser;
            bool openWorldMapBrowser = _pendingKnownGoodClientOpenWorldMapBrowser;
            _pendingKnownGoodClientPath = null;
            _pendingKnownGoodClientBuildLabel = null;
            _pendingKnownGoodClientAttachLooseFolder = false;
            _pendingKnownGoodClientOpenBrowser = false;
            _pendingKnownGoodClientOpenWorldMapBrowser = false;

            if (!Directory.Exists(savedBasePath))
            {
                _statusMessage = $"Saved client path no longer exists: {savedBasePath}";
                return;
            }

            ApplySharedClientSelection(savedBasePath, savedBuildLabel, string.Empty, preferArchiveSource: true, clearArchiveVirtualPath: true);
            _settings.LastOpenedClientPath = savedBasePath;
            SaveSettings();

            if (attachLooseFolder)
            {
                string? overlayPath = TryShowFolderDialog(
                    "Select loose folder to load against the saved base client",
                    _settings.LastOpenedLooseOverlayPath,
                    showNewFolderButton: false);

                if (!string.IsNullOrWhiteSpace(overlayPath))
                {
                    AttachLooseFolder(overlayPath);
                    if (openBrowser)
                        OpenAssetFileBrowserForClient(savedBasePath, savedBuildLabel, _session.Source.LooseOverlayRoot, AssetFileBrowserFilter.SupportedAssets, $"Browse {Path.GetFileName(savedBasePath)} + Loose Files");
                    if (openWorldMapBrowser)
                        OpenWorldMapBrowserForClient(savedBasePath, savedBuildLabel, _session.Source.LooseOverlayRoot, $"Browse {Path.GetFileName(savedBasePath)} + Loose World Maps");
                }
            }
            else
            {
                _statusMessage = $"Loaded saved client: {savedBasePath}";
                if (openBrowser)
                    OpenAssetFileBrowserForClient(savedBasePath, savedBuildLabel, string.Empty, AssetFileBrowserFilter.SupportedAssets, $"Browse {Path.GetFileName(savedBasePath)}");
                if (openWorldMapBrowser)
                    OpenWorldMapBrowserForClient(savedBasePath, savedBuildLabel, string.Empty, $"Browse {Path.GetFileName(savedBasePath)} Maps");
            }
        }
    }

    private bool CanBrowseCurrentWorldMaps()
    {
        return !string.IsNullOrWhiteSpace(_session.World.ClientRoot)
            || !string.IsNullOrWhiteSpace(_session.Source.ArchiveRoot);
    }

    private void OpenWorldMapBrowserForCurrentClient()
    {
        string clientRoot = !string.IsNullOrWhiteSpace(_session.World.ClientRoot)
            ? _session.World.ClientRoot
            : _session.Source.ArchiveRoot;
        string buildLabel = !string.IsNullOrWhiteSpace(_session.World.BuildLabel)
            ? _session.World.BuildLabel
            : _session.Source.BuildLabel;
        string looseOverlayRoot = !string.IsNullOrWhiteSpace(_session.World.LooseOverlayRoot)
            ? _session.World.LooseOverlayRoot
            : _session.Source.LooseOverlayRoot;

        if (string.IsNullOrWhiteSpace(clientRoot))
        {
            _statusMessage = "Load a client root before browsing world maps.";
            return;
        }

        OpenWorldMapBrowserForClient(clientRoot, buildLabel, looseOverlayRoot, "Browse Current Client Maps");
    }

    private void OpenWorldMapBrowserForClient(string clientRoot, string? buildLabel, string? looseOverlayRoot, string title)
    {
        _worldMapBrowserClientRoot = clientRoot?.Trim() ?? string.Empty;
        _worldMapBrowserBuildLabel = buildLabel?.Trim() ?? string.Empty;
        _worldMapBrowserLooseOverlayRoot = looseOverlayRoot?.Trim() ?? string.Empty;
        _worldMapBrowserTitle = title;
        _worldMapBrowserFilter = string.Empty;
        _showWorldMapBrowserWindow = true;
        RefreshWorldMapBrowser(force: true);
    }

    private void RefreshWorldMapBrowser(bool force = false)
    {
        string clientRoot = _worldMapBrowserClientRoot?.Trim() ?? string.Empty;
        string buildLabel = _worldMapBrowserBuildLabel?.Trim() ?? string.Empty;
        string looseOverlayRoot = _worldMapBrowserLooseOverlayRoot?.Trim() ?? string.Empty;
        string signature = string.Join('|', clientRoot, buildLabel, looseOverlayRoot);

        if (!force && string.Equals(_worldMapBrowserSignature, signature, StringComparison.OrdinalIgnoreCase))
            return;

        _worldMapBrowserSignature = signature;

        if (string.IsNullOrWhiteSpace(clientRoot) || !Directory.Exists(clientRoot))
        {
            _worldMapBrowserMaps = Array.Empty<DiscoveredLooseWorldMap>();
            _worldMapBrowserSummary = "Set a valid client root to browse world maps.";
            return;
        }

        ViewerIoSourceKey sourceKey = ViewerIoSourceKey.Create(clientRoot, buildLabel, looseOverlayRoot);
        ViewerIoCatalogLease catalogLease = _viewerIoService.GetCatalog(sourceKey);
        _worldMapBrowserMaps = LooseWorldMapDiscovery.DiscoverWithArchiveReader(
            sourceKey.ClientRoot,
            sourceKey.LooseOverlayRoot,
            catalogLease.ArchiveCatalog);
        _worldMapBrowserSummary = _worldMapBrowserMaps.Count == 0
            ? "No Map.dbc-backed world maps were found for this source."
            : $"Loaded {_worldMapBrowserMaps.Count} maps from the effective Map.dbc for this source.";
    }

    private IEnumerable<DiscoveredLooseWorldMap> GetFilteredWorldMapBrowserMaps()
    {
        if (string.IsNullOrWhiteSpace(_worldMapBrowserFilter))
            return _worldMapBrowserMaps;

        string filter = _worldMapBrowserFilter.Trim();
        return _worldMapBrowserMaps.Where(map =>
            map.Directory.Contains(filter, StringComparison.OrdinalIgnoreCase)
            || map.Name.Contains(filter, StringComparison.OrdinalIgnoreCase)
            || (map.LooseSourceDirectory?.Contains(filter, StringComparison.OrdinalIgnoreCase) ?? false));
    }

    private static string? TryShowFolderDialog(string description, string? initialDir = null, bool showNewFolderButton = false)
    {
        return WindowsNativeFileDialogs.PickFolder(description, initialDir);
    }

    private static string? TryShowOpenFileDialog(string title, string? initialDir, IReadOnlyList<WindowsNativeFileDialogs.FileDialogFilter> filters)
    {
        return WindowsNativeFileDialogs.PickFile(title, initialDir, filters);
    }

    private (string Title, WindowsNativeFileDialogs.FileDialogFilter[] Filters) GetLocalSourceFileDialogSpec()
    {
        return _session.WorkspaceMode switch
        {
            WowViewerWorkspaceMode.StandaloneWmo => (
                "Open WMO File",
                [
                    new WindowsNativeFileDialogs.FileDialogFilter("WMO files", "*.wmo;*.wmo.mpq;*.mpq"),
                    new WindowsNativeFileDialogs.FileDialogFilter("All files", "*.*"),
                ]),
            WowViewerWorkspaceMode.StandaloneMdx => (
                "Open MDX File",
                [
                    new WindowsNativeFileDialogs.FileDialogFilter("MDX files", "*.mdx"),
                    new WindowsNativeFileDialogs.FileDialogFilter("All files", "*.*"),
                ]),
            _ => (
                "Open M2 File",
                [
                    new WindowsNativeFileDialogs.FileDialogFilter("M2 files", "*.m2"),
                    new WindowsNativeFileDialogs.FileDialogFilter("All files", "*.*"),
                ]),
        };
    }

    private string? GetLocalSourceInitialDirectory()
    {
        string currentInputPath = _session.Source.InputPath;
        if (!string.IsNullOrWhiteSpace(currentInputPath))
        {
            string? currentDirectory = Path.GetDirectoryName(currentInputPath);
            if (!string.IsNullOrWhiteSpace(currentDirectory) && Directory.Exists(currentDirectory))
                return currentDirectory;
        }

        if (!string.IsNullOrWhiteSpace(_settings.LastOpenedClientPath) && Directory.Exists(_settings.LastOpenedClientPath))
            return _settings.LastOpenedClientPath;

        return null;
    }

    private void DrawWorldControlContents()
    {
        if (ImGui.CollapsingHeader("World Overview", ImGuiTreeNodeFlags.DefaultOpen))
        {
            WowViewerWorldSceneSnapshot sceneSnapshot = _worldSceneHost.SceneSnapshot;
            if (_currentWorldSession != null)
            {
                ImGui.TextUnformatted(sceneSnapshot.ResolvedMapDirectory);
                ImGui.TextDisabled($"Tile: {FormatWorldTileLabel()}  Load: {sceneSnapshot.LoadDuration.TotalMilliseconds:F1} ms");
                ImGui.TextDisabled($"Occupied tiles: {sceneSnapshot.OccupiedTiles.Count}");
            }
            else
            {
                ImGui.TextDisabled(string.IsNullOrWhiteSpace(_session.World.ClientRoot)
                    ? "Attach a client root, choose a map, then open the world."
                    : $"Ready: {Path.GetFileName(_session.World.ClientRoot)}");
            }
        }

        ImGui.Separator();

        DrawWorldNavigatorSourceSection();

        if (ImGui.SmallButton("Use WoW335 Azeroth Baseline"))
        {
            _session.WorkspaceMode = WowViewerWorkspaceMode.WorldSession;
            ApplySharedClientSelection(@"H:\CLIENTS\WoW335\3.X_Retail_Windows_enUS_3.3.5.12340\World of Warcraft", "3.3.5.12340", string.Empty);
            _session.World.MapInput = "Azeroth";
            _session.World.TileX = -1;
            _session.World.TileY = -1;
        }
    }

    private void RefreshDiscoveredWorldMaps(bool force = false)
    {
        string clientRoot = _session.World.ClientRoot?.Trim() ?? string.Empty;
        string buildLabel = _session.World.BuildLabel?.Trim() ?? string.Empty;
        string looseOverlayRoot = _session.World.LooseOverlayRoot?.Trim() ?? string.Empty;
        string signature = string.Join('|',
            string.IsNullOrWhiteSpace(clientRoot) ? string.Empty : Path.GetFullPath(clientRoot),
            buildLabel,
            string.IsNullOrWhiteSpace(looseOverlayRoot) ? string.Empty : Path.GetFullPath(looseOverlayRoot));

        if (!force
            && string.Equals(_worldMapDiscoverySignature, signature, StringComparison.OrdinalIgnoreCase))
            return;

        _worldMapDiscoverySignature = signature;

        if (string.IsNullOrWhiteSpace(clientRoot) || !Directory.Exists(clientRoot))
        {
            _pendingWorldMapDiscoveryTask = null;
            _discoveredWorldMaps = Array.Empty<DiscoveredLooseWorldMap>();
            _worldMapDiscoverySummary = "Set a valid client root to discover maps from Map.dbc.";
            return;
        }

        int version = _worldMapDiscoveryVersion;
        _discoveredWorldMaps = Array.Empty<DiscoveredLooseWorldMap>();
        _worldMapDiscoverySummary = "Discovering maps for the current client root...";
        _pendingWorldMapDiscoveryTask = Task.Run(() => DiscoverWorldMaps(signature, clientRoot, buildLabel, looseOverlayRoot, version));
    }

    private void DrawWorldSpawnPickerSection()
    {
        RefreshWorldSpawnPickerState();

        ImGui.TextDisabled("Spawn Picker");
        ImGui.TextWrapped(_worldSpawnPickerState?.Summary ?? "Select a map to load the ADT/WDT tile grid.");

        if (_worldSpawnPickerState is null)
            return;

        if (_worldSpawnPickerState.Session is { } session)
        {
            ImGui.TextDisabled($"Resolved Map: {session.RequestedMapInput} -> {session.ResolvedMapDirectory}");
        }

        if (ImGui.Button("Refresh Spawn Grid", new Vector2(-1, 0)))
            RefreshWorldSpawnPickerState(force: true);

        if (_worldSpawnPickerState.Session is null)
            return;

        float pickerSize = Math.Clamp(ImGui.GetContentRegionAvail().X, 192f, 384f);
        Vector2 origin = ImGui.GetCursorScreenPos();
        Vector2 pickerExtent = new(pickerSize, pickerSize);
        ImGui.InvisibleButton("##WorldSpawnPicker", pickerExtent);
        DrawWorldSpawnPickerGrid(origin, pickerExtent, _worldSpawnPickerState);

        if (ImGui.IsItemHovered() && ImGui.IsMouseClicked(ImGuiMouseButton.Left))
        {
            Vector2 local = ImGui.GetMousePos() - origin;
            int previewTileX = Math.Clamp((int)(local.X / (pickerSize / 64f)), 0, 63);
            int previewTileY = Math.Clamp((int)(local.Y / (pickerSize / 64f)), 0, 63);
            (int selectedTileX, int selectedTileY) = PreviewSpawnTileToSourceTile(previewTileX, previewTileY);
            _session.World.TileX = selectedTileX;
            _session.World.TileY = selectedTileY;
            Vector3 tileCenter = BuildWorldTileCenter(selectedTileX, selectedTileY);
            _statusMessage = $"Selected ADT tile ({selectedTileX},{selectedTileY}) from map cell ({previewTileX},{previewTileY}) at {FormatVector3(tileCenter)}. Reload the world session to load that ADT tile.";
        }

        if (_session.World.TileX >= 0 && _session.World.TileY >= 0)
        {
            Vector3 tileCenter = BuildWorldTileCenter(_session.World.TileX, _session.World.TileY);
            ImGui.Text($"Spawn Tile: ({_session.World.TileX},{_session.World.TileY})");
            ImGui.Text($"ADT Tile Center: {FormatVector3(tileCenter)}");
        }
        else
        {
            ImGui.TextDisabled("Spawn Tile: auto");
        }

        if (ImGui.Button("Use Auto Tile", new Vector2(-1, 0)))
        {
            _session.World.TileX = -1;
            _session.World.TileY = -1;
            _statusMessage = "World spawn tile reset to auto selection.";
        }
    }

    private void RefreshWorldSpawnPickerState(bool force = false)
    {
        string clientRoot = _session.World.ClientRoot?.Trim() ?? string.Empty;
        string mapInput = _session.World.MapInput?.Trim() ?? string.Empty;
        string signature = BuildWorldSpawnPickerSignature();

        int version = _worldSpawnPickerVersion;

        if (!force
            && string.Equals(_worldSpawnPickerState?.Signature, signature, StringComparison.OrdinalIgnoreCase))
            return;

        if (string.IsNullOrWhiteSpace(clientRoot) || !Directory.Exists(clientRoot))
        {
            _pendingWorldSpawnPickerTask = null;
            _worldSpawnPickerState = new WorldSpawnPickerState(signature, "Set a valid client root to load the ADT/WDT tile grid.", null, version);
            return;
        }

        if (string.IsNullOrWhiteSpace(mapInput))
        {
            _pendingWorldSpawnPickerTask = null;
            _worldSpawnPickerState = new WorldSpawnPickerState(signature, "Choose a map above to load the ADT/WDT tile grid.", null, version);
            return;
        }

        _worldSpawnPickerState = new WorldSpawnPickerState(signature, "Loading ADT/WDT tile grid for the current map...", null, version);
        WowViewerWorldSessionOpenRequest request = _session.World.BuildRequest();
        _pendingWorldSpawnPickerTask = Task.Run(() => LoadWorldSpawnPickerState(signature, request, version));
    }

    private string BuildWorldSpawnPickerSignature()
    {
        string clientRoot = _session.World.ClientRoot?.Trim() ?? string.Empty;
        string buildLabel = _session.World.BuildLabel?.Trim() ?? string.Empty;
        string looseOverlayRoot = _session.World.LooseOverlayRoot?.Trim() ?? string.Empty;
        string mapInput = _session.World.MapInput?.Trim() ?? string.Empty;
        return string.Join('|', clientRoot, buildLabel, looseOverlayRoot, mapInput);
    }

    private PendingWorldMapDiscoveryResult DiscoverWorldMaps(string signature, string clientRoot, string buildLabel, string looseOverlayRoot, int version)
    {
        ViewerIoSourceKey sourceKey = ViewerIoSourceKey.Create(clientRoot, buildLabel, looseOverlayRoot);
        ViewerIoCatalogLease catalogLease = _viewerIoService.GetCatalog(sourceKey);
        IReadOnlyList<DiscoveredLooseWorldMap> discoveredWorldMaps = LooseWorldMapDiscovery.DiscoverWithArchiveReader(
            sourceKey.ClientRoot,
            sourceKey.LooseOverlayRoot,
            catalogLease.ArchiveCatalog);
        string summary = discoveredWorldMaps.Count == 0
            ? "No Map.dbc-backed maps were found for the current client or loose overlay source."
            : $"Loaded {discoveredWorldMaps.Count} maps from the shared viewer I/O catalog (bootstrap #{catalogLease.BootstrapCount}). Loose files, when present, patch the selected map data on top of archive data.";
        return new PendingWorldMapDiscoveryResult(signature, discoveredWorldMaps, summary, version);
    }

    private static WorldSpawnPickerState LoadWorldSpawnPickerState(string signature, WowViewerWorldSessionOpenRequest request, int version)
    {
        WowViewerWorldSessionBootstrapResult session = WowViewerWorldSessionBootstrapper.Open(request);
        string summary = $"Loaded ADT/WDT tile grid for {session.ResolvedMapDirectory}: {session.OccupiedTiles.Count} occupied ADT tiles. WDL is not read for this picker.";

        return new WorldSpawnPickerState(signature, summary, session, version);
    }

    private void DrawWorldSpawnPickerGrid(Vector2 origin, Vector2 extent, WorldSpawnPickerState state)
    {
        ImDrawListPtr drawList = ImGui.GetWindowDrawList();
        float cellSize = extent.X / 64f;
        uint noDataColor = ImGui.ColorConvertFloat4ToU32(new Vector4(0.13f, 0.14f, 0.16f, 1.0f));
        uint occupiedBorder = ImGui.ColorConvertFloat4ToU32(new Vector4(0.58f, 0.62f, 0.68f, 0.85f));
        uint selectedColor = ImGui.ColorConvertFloat4ToU32(new Vector4(1.0f, 0.34f, 0.22f, 1.0f));
        uint loadedColor = ImGui.ColorConvertFloat4ToU32(new Vector4(1.0f, 0.95f, 0.36f, 1.0f));

        drawList.AddRectFilled(origin, origin + extent, ImGui.ColorConvertFloat4ToU32(new Vector4(0.06f, 0.07f, 0.09f, 1.0f)), 4f);

        for (int tileY = 0; tileY < 64; tileY++)
        {
            for (int tileX = 0; tileX < 64; tileX++)
            {
                Vector2 cellMin = origin + new Vector2(tileX * cellSize, tileY * cellSize);
                Vector2 cellMax = cellMin + new Vector2(cellSize, cellSize);
                int tileIndex = (tileX * 64) + tileY;
                bool occupied = state.OccupiedTileIndices.Contains(tileIndex);
                uint fill = noDataColor;

                if (occupied)
                    fill = ImGui.ColorConvertFloat4ToU32(new Vector4(0.22f, 0.24f, 0.28f, 1.0f));

                drawList.AddRectFilled(cellMin, cellMax, fill);
                if (occupied)
                    drawList.AddRect(cellMin, cellMax, occupiedBorder, 0f, ImDrawFlags.None, 0.6f);
            }
        }

        if (_currentWorldRuntimeFrame != null)
        {
            (int previewTileX, int previewTileY) = SourceSpawnTileToPreviewTile(_currentWorldRuntimeFrame.SelectedTileX, _currentWorldRuntimeFrame.SelectedTileY);
            Vector2 loadMin = origin + new Vector2(previewTileX * cellSize, previewTileY * cellSize);
            Vector2 loadMax = loadMin + new Vector2(cellSize, cellSize);
            drawList.AddRect(loadMin, loadMax, loadedColor, 0f, ImDrawFlags.None, 2.0f);
        }

        if (_session.World.TileX >= 0 && _session.World.TileY >= 0)
        {
            (int previewTileX, int previewTileY) = SourceSpawnTileToPreviewTile(_session.World.TileX, _session.World.TileY);
            Vector2 selectedMin = origin + new Vector2(previewTileX * cellSize, previewTileY * cellSize);
            Vector2 selectedMax = selectedMin + new Vector2(cellSize, cellSize);
            drawList.AddRect(selectedMin, selectedMax, selectedColor, 0f, ImDrawFlags.None, 2.4f);
        }

        drawList.AddRect(origin, origin + extent, ImGui.ColorConvertFloat4ToU32(new Vector4(0.32f, 0.35f, 0.40f, 1.0f)), 4f, ImDrawFlags.None, 1.2f);
    }

    private static Vector3 BuildWorldTileCenter(int tileX, int tileY) => WowViewerWorldRuntimeBridge.ComputeTileCenter(tileX, tileY, 0f);

    private static (int sourceTileX, int sourceTileY) PreviewSpawnTileToSourceTile(int previewTileX, int previewTileY)
    {
        return (previewTileX, previewTileY);
    }

    private static (int previewTileX, int previewTileY) SourceSpawnTileToPreviewTile(int sourceTileX, int sourceTileY)
    {
        return (sourceTileX, sourceTileY);
    }

    private static string BuildWorldMapLabel(DiscoveredLooseWorldMap map)
    {
        return string.Equals(map.Name, map.Directory, StringComparison.OrdinalIgnoreCase)
            ? map.Directory
            : $"{map.Name} ({map.Directory})";
    }

    private static string BuildWorldMapTooltip(DiscoveredLooseWorldMap map)
    {
        if (!string.IsNullOrWhiteSpace(map.LooseSourceDirectory))
            return $"MapId={map.Id}\nDirectory={map.Directory}\nLoose source={map.LooseSourceDirectory}";

        return $"MapId={map.Id}\nDirectory={map.Directory}\nNo loose override detected for this map.";
    }

    private void DrawDatasetToolingControlContents()
    {
        ImGui.TextWrapped("This workspace makes dataset prep and training entrypoints first-class wow-viewer shell actions. The current execution path launches wow-viewer-owned scripts that orchestrate mask generation and model training.");
        ImGui.Separator();
        string scriptRootLabel = TryResolveDatasetScriptRoot(out string? scriptRoot)
            ? scriptRoot!
            : $"unresolved (base: {AppContext.BaseDirectory})";
        ImGui.TextDisabled("Workflow Root: wow-viewer/scripts");
        ImGui.TextDisabled($"Resolved Script Root: {scriptRootLabel}");

        string searchRoot = _datasetSearchRoot;
        string buildFilter = _datasetBuildFilter;
        string archiveRootsFile = _datasetArchiveRootsFile;
        string archiveRootFallback = _datasetArchiveRootFallback;
        string resumeCheckpoint = _datasetResumeCheckpoint;
        string outputDir = _datasetOutputDir;
        string cacheDir = _datasetCacheDir;
        int numEpochs = _datasetNumEpochs;
        int batchSize = _datasetBatchSize;

        ImGui.InputText("Search Root", ref searchRoot, 1024);
        ImGui.InputText("Build Filter", ref buildFilter, 128);
        ImGui.InputText("Archive Roots File", ref archiveRootsFile, 1024);
        ImGui.InputText("Archive Root Fallback", ref archiveRootFallback, 1024);
        ImGui.InputText("Resume Checkpoint", ref resumeCheckpoint, 1024);
        ImGui.InputText("Output Dir Override", ref outputDir, 1024);
        ImGui.InputText("Cache Dir (v7.6)", ref cacheDir, 1024);
        ImGui.InputInt("Epoch Override", ref numEpochs);
        ImGui.InputInt("Batch Override", ref batchSize);

        _datasetSearchRoot = searchRoot;
        _datasetBuildFilter = buildFilter;
        _datasetArchiveRootsFile = archiveRootsFile;
        _datasetArchiveRootFallback = archiveRootFallback;
        _datasetResumeCheckpoint = resumeCheckpoint;
        _datasetOutputDir = outputDir;
        _datasetCacheDir = cacheDir;
        _datasetNumEpochs = Math.Max(0, numEpochs);
        _datasetBatchSize = Math.Max(0, batchSize);

        ImGui.Checkbox("Allow CPU", ref _datasetAllowCpu);
        ImGui.SameLine();
        ImGui.Checkbox("Dry Run", ref _datasetDryRun);
        ImGui.Checkbox("Skip Masks", ref _datasetSkipMasks);
        ImGui.SameLine();
        ImGui.Checkbox("Force Remask", ref _datasetForceRemask);
        ImGui.Checkbox("Skip Cache (v7.6)", ref _datasetSkipCache);

        ImGui.Separator();
        if (ImGui.Button("Generate M2 Masks", new Vector2(-1, 0)))
            LaunchDatasetPowerShellJob("generate_m2_masks.py", BuildMaskGenerationCommand());

        if (ImGui.Button("Run V7.5.1 Pipeline", new Vector2(-1, 0)))
            LaunchDatasetPowerShellJob("run_v751_pipeline.ps1", BuildV751Command());

        if (ImGui.Button("Run V7.6 Pipeline", new Vector2(-1, 0)))
            LaunchDatasetPowerShellJob("run_v76_pipeline.ps1", BuildV76Command());

        ImGui.Separator();
        ImGui.TextDisabled("Last launch command:");
        ImGui.TextWrapped(_datasetLastCommand);
    }

    private void DrawModelOutputControlContents()
    {
        ImGui.TextWrapped("This workspace is the first GPU-backed wow-viewer consumer for model-output terrain tiles. It loads the flat OBJ + texture bundles from infer_v9.py and reconstructs them into tile-space for fast inspection.");
        if (_session.ModelOutput.CameraMode == WowViewerModelOutputCameraMode.Orbit)
            ImGui.TextDisabled("Orbit camera: left-drag orbit, right-drag pan, wheel zoom, double-click preview to reset.");
        else
            ImGui.TextDisabled("Fly camera: right-drag look, WASD move, Q/E move vertically, Shift accelerate, wheel changes move speed, double-click preview to reset.");
        ImGui.Separator();
        ImGui.TextDisabled($"Workspace: {_session.GetWorkspaceLabel()}");
        ImGui.TextDisabled($"Source: {_session.ModelOutput.Describe()}");
        ImGui.Separator();

        string inputPath = _session.ModelOutput.InputPath;
        ImGui.InputText("Flat Mesh Folder / Summary", ref inputPath, 1024);
        _session.ModelOutput.InputPath = inputPath;

        int variantIndex = (int)_session.ModelOutput.Variant;
        if (ImGui.Combo("Mesh Variant", ref variantIndex, "Predicted\0WDL Baseline\0"))
            _session.ModelOutput.Variant = (WowViewerModelOutputVariant)variantIndex;

        int cameraModeIndex = (int)_session.ModelOutput.CameraMode;
        if (ImGui.Combo("Camera Mode", ref cameraModeIndex, "Orbit\0Fly\0"))
        {
            _session.ModelOutput.CameraMode = (WowViewerModelOutputCameraMode)cameraModeIndex;
            ResetModelOutputCamera();
        }

        bool showObjects = _session.ModelOutput.ShowObjects;
        if (ImGui.Checkbox("Show Object Placeholders", ref showObjects))
        {
            _session.ModelOutput.ShowObjects = showObjects;
            RefreshModelOutputGpuScene();
        }

        bool showM2Objects = _session.ModelOutput.ShowM2Objects;
        if (ImGui.Checkbox("Show M2 Placeholders", ref showM2Objects))
        {
            _session.ModelOutput.ShowM2Objects = showM2Objects;
            RefreshModelOutputGpuScene();
        }

        bool showWmoObjects = _session.ModelOutput.ShowWmoObjects;
        if (ImGui.Checkbox("Show WMO Placeholders", ref showWmoObjects))
        {
            _session.ModelOutput.ShowWmoObjects = showWmoObjects;
            RefreshModelOutputGpuScene();
        }

        ImGui.TextDisabled($"Camera: azimuth {_session.ModelOutput.CameraAzimuthDegrees:F1} deg, elevation {_session.ModelOutput.CameraElevationDegrees:F1} deg");
        if (_session.ModelOutput.CameraMode == WowViewerModelOutputCameraMode.Orbit)
        {
            ImGui.TextDisabled($"Orbit Zoom: {_session.ModelOutput.CameraZoomFactor:F2}");
            Vector3 targetOffset = _session.ModelOutput.GetTargetOffset();
            ImGui.TextDisabled($"Target Offset: {targetOffset.X:F1}, {targetOffset.Y:F1}, {targetOffset.Z:F1}");
        }
        else
        {
            Vector3 flyPosition = _session.ModelOutput.GetFlyPosition();
            ImGui.TextDisabled($"Fly Position: {flyPosition.X:F1}, {flyPosition.Y:F1}, {flyPosition.Z:F1}");
            ImGui.TextDisabled($"Fly Speed: {_session.ModelOutput.FlyMoveSpeed:F2}x");
        }

        if (ImGui.Button(_currentModelOutputScene == null ? "Load Model Output Scene" : "Reload Model Output Scene", new Vector2(-1, 0)))
            LoadActiveWorkspace();

        if (ImGui.Button("Reset Camera", new Vector2(-1, 0)))
            ResetModelOutputCamera();

        if (ImGui.Button("Use V9 Dev Flat Mesh Sample", new Vector2(-1, 0)))
        {
            _session.WorkspaceMode = WowViewerWorkspaceMode.ModelOutputs;
            _session.ModelOutput.InputPath = @"i:\parp\parp-tools\output\ml-training\v9_development_manifest_obj_check_v91";
            _session.ModelOutput.Variant = WowViewerModelOutputVariant.Predicted;
            ResetModelOutputCamera();
        }

        if (_currentModelOutputScene != null)
        {
            ImGui.Separator();
            ImGui.TextDisabled("Loaded Scene");
            ImGui.Text($"Tiles: {_currentModelOutputScene.Tiles.Count}");
            ImGui.Text($"Vertices: {_currentModelOutputScene.VertexCount}");
            ImGui.Text($"Triangles: {_currentModelOutputScene.TriangleCount}");
            ImGui.Text($"Objects: {_currentModelOutputScene.ObjectCount}");
            ImGui.Text($"Tile Size: {_currentModelOutputScene.TileWorldSize:F3}");
            ImGui.Text($"Centered Meshes: {_currentModelOutputScene.CenterMesh}");
        }
    }

    private string BuildMaskGenerationCommand()
    {
        string searchRoot = NormalizeSearchRootForCommand(_datasetSearchRoot);
        List<string> args =
        [
            "python",
            "./generate_m2_masks.py",
            "--search-root", QuoteIfNeeded(searchRoot),
            "--skip-existing", _datasetForceRemask ? "false" : "true",
        ];

        if (!string.IsNullOrWhiteSpace(_datasetBuildFilter))
            args.AddRange(["--build-filter", QuoteIfNeeded(_datasetBuildFilter)]);
        if (!string.IsNullOrWhiteSpace(_datasetArchiveRootsFile))
            args.AddRange(["--archive-roots-file", QuoteIfNeeded(_datasetArchiveRootsFile)]);
        if (!string.IsNullOrWhiteSpace(_datasetArchiveRootFallback))
            args.AddRange(["--archive-root-fallback", QuoteIfNeeded(_datasetArchiveRootFallback)]);
        if (_datasetDryRun)
            args.Add("--dry-run");

        return string.Join(' ', args);
    }

    private string BuildV751Command()
    {
        string searchRoot = NormalizeSearchRootForCommand(_datasetSearchRoot);
        List<string> args =
        [
            "./run_v751_pipeline.ps1",
            "-SearchRoots", QuoteIfNeeded(searchRoot),
        ];

        if (_datasetSkipMasks)
            args.Add("-SkipMasks");
        if (_datasetForceRemask)
            args.Add("-ForceRemask");
        if (_datasetAllowCpu)
            args.Add("-AllowCpu");
        if (_datasetDryRun)
            args.Add("-DryRun");
        if (!string.IsNullOrWhiteSpace(_datasetBuildFilter))
            args.AddRange(["-BuildFilter", QuoteIfNeeded(_datasetBuildFilter)]);
        if (!string.IsNullOrWhiteSpace(_datasetArchiveRootsFile))
            args.AddRange(["-ArchiveRootsFile", QuoteIfNeeded(_datasetArchiveRootsFile)]);
        if (!string.IsNullOrWhiteSpace(_datasetArchiveRootFallback))
            args.AddRange(["-ArchiveRootFallback", QuoteIfNeeded(_datasetArchiveRootFallback)]);
        if (!string.IsNullOrWhiteSpace(_datasetResumeCheckpoint))
            args.AddRange(["-ResumeFrom", QuoteIfNeeded(_datasetResumeCheckpoint)]);
        if (!string.IsNullOrWhiteSpace(_datasetOutputDir))
            args.AddRange(["-OutputDir", QuoteIfNeeded(_datasetOutputDir)]);
        if (_datasetNumEpochs > 0)
            args.AddRange(["-NumEpochs", _datasetNumEpochs.ToString()]);
        if (_datasetBatchSize > 0)
            args.AddRange(["-BatchSize", _datasetBatchSize.ToString()]);

        return string.Join(' ', args);
    }

    private string BuildV76Command()
    {
        string searchRoot = NormalizeSearchRootForCommand(_datasetSearchRoot);
        List<string> args =
        [
            "./run_v76_pipeline.ps1",
            "-SearchRoots", QuoteIfNeeded(searchRoot),
        ];

        if (_datasetSkipMasks)
            args.Add("-SkipMasks");
        if (_datasetForceRemask)
            args.Add("-ForceRemask");
        if (_datasetSkipCache)
            args.Add("-SkipCache");
        if (_datasetAllowCpu)
            args.Add("-AllowCpu");
        if (_datasetDryRun)
            args.Add("-DryRun");
        if (!string.IsNullOrWhiteSpace(_datasetBuildFilter))
            args.AddRange(["-BuildFilter", QuoteIfNeeded(_datasetBuildFilter)]);
        if (!string.IsNullOrWhiteSpace(_datasetArchiveRootsFile))
            args.AddRange(["-ArchiveRootsFile", QuoteIfNeeded(_datasetArchiveRootsFile)]);
        if (!string.IsNullOrWhiteSpace(_datasetArchiveRootFallback))
            args.AddRange(["-ArchiveRootFallback", QuoteIfNeeded(_datasetArchiveRootFallback)]);
        if (!string.IsNullOrWhiteSpace(_datasetResumeCheckpoint))
            args.AddRange(["-ResumeFrom", QuoteIfNeeded(_datasetResumeCheckpoint)]);
        if (!string.IsNullOrWhiteSpace(_datasetOutputDir))
            args.AddRange(["-OutputDir", QuoteIfNeeded(_datasetOutputDir)]);
        if (!string.IsNullOrWhiteSpace(_datasetCacheDir))
            args.AddRange(["-CacheDir", QuoteIfNeeded(_datasetCacheDir)]);
        if (_datasetNumEpochs > 0)
            args.AddRange(["-NumEpochs", _datasetNumEpochs.ToString()]);
        if (_datasetBatchSize > 0)
            args.AddRange(["-BatchSize", _datasetBatchSize.ToString()]);

        return string.Join(' ', args);
    }

    private void LaunchDatasetPowerShellJob(string label, string command)
    {
        try
        {
            if (!TryResolveDatasetScriptRoot(out string? scriptRoot) || string.IsNullOrWhiteSpace(scriptRoot))
            {
                _statusMessage = $"Dataset script root not found (base: {AppContext.BaseDirectory}).";
                return;
            }

            string quotedScriptRoot = QuoteForPowerShellSingle(scriptRoot);
            string psCommand = $"Set-Location '{quotedScriptRoot}'; {command}";

            ProcessStartInfo info = new()
            {
                FileName = "pwsh",
                Arguments = $"-NoExit -ExecutionPolicy Bypass -Command \"{psCommand.Replace("\"", "`\"")}\"",
                UseShellExecute = true,
                WorkingDirectory = scriptRoot,
            };

            Process.Start(info);
            _datasetLastCommand = $"[{label}] {command}";
            _statusMessage = $"Launched dataset command in a new PowerShell window: {label}";
        }
        catch (Exception ex)
        {
            _lastError = ex.Message;
            _statusMessage = "Failed to launch dataset command.";
        }
    }

    private static bool TryResolveDatasetScriptRoot(out string? scriptRoot)
    {
        scriptRoot = null;
        DirectoryInfo? cursor = new(AppContext.BaseDirectory);

        for (int depth = 0; depth < 12 && cursor != null; depth++)
        {
            string candidate = Path.Combine(cursor.FullName, "scripts");
            if (Directory.Exists(candidate)
                && File.Exists(Path.Combine(candidate, "run_v751_pipeline.ps1"))
                && File.Exists(Path.Combine(candidate, "run_v76_pipeline.ps1"))
                && File.Exists(Path.Combine(candidate, "generate_m2_masks.py")))
            {
                scriptRoot = candidate;
                return true;
            }

            cursor = cursor.Parent;
        }

        return false;
    }

    private static string NormalizeSearchRootForCommand(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return value;

        if (Path.IsPathRooted(value))
            return value;

        if (!TryResolveDatasetScriptRoot(out string? scriptRoot) || string.IsNullOrWhiteSpace(scriptRoot))
            return value;

        string wowViewerRoot = Path.GetFullPath(Path.Combine(scriptRoot, ".."));
        string parpToolsRoot = Path.GetFullPath(Path.Combine(wowViewerRoot, ".."));
        return Path.GetFullPath(Path.Combine(parpToolsRoot, value));
    }

    private static string QuoteIfNeeded(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return value;

        return value.Contains(' ') ? $"\"{value}\"" : value;
    }

    private static string QuoteForPowerShellSingle(string value)
    {
        return value.Replace("'", "''");
    }

    private void DrawPlaceholderControlContents()
    {
        ImGui.TextWrapped("This workspace is intentionally not implemented yet. The controls below define the future standalone-source contract without claiming a live consumer in this slice.");
        ImGui.Separator();
        ImGui.TextDisabled($"Workspace: {_session.GetWorkspaceLabel()}");
        ImGui.TextDisabled("Status: placeholder only");
    }

    private void DrawStatusSection()
    {
        ImGui.Separator();
        ImGui.TextDisabled("Status");
        ImGui.TextWrapped(_statusMessage);
        if (!string.IsNullOrWhiteSpace(_lastError))
        {
            ImGui.Separator();
            ImGui.TextColored(new Vector4(0.95f, 0.42f, 0.32f, 1.0f), _lastError);
        }
    }

    private void DrawPreviewWindow(bool forceDockedSize = false)
    {
        if (!forceDockedSize)
            ImGui.SetNextWindowSize(new Vector2(880, 720), ImGuiCond.FirstUseEver);
        if (!ImGui.Begin($"{_session.GetWorkspaceLabel()} Preview"))
        {
            ImGui.End();
            return;
        }

        if (!IsImplementedWorkspace(_session.WorkspaceMode))
        {
            ImGui.TextWrapped($"{_session.GetWorkspaceLabel()} is a placeholder workspace in this slice. A dedicated preview consumer has not been implemented yet.");
            ImGui.TextDisabled("Current boundary: expose the workspace surface now, land the real consumer later.");
            ImGui.End();
            return;
        }

        if (_session.WorkspaceMode == WowViewerWorkspaceMode.WorldSession)
        {
            DrawWorldSessionPreview();
            ImGui.End();
            return;
        }

        if (_session.WorkspaceMode == WowViewerWorkspaceMode.DatasetTooling)
        {
            ImGui.TextWrapped("Dataset tooling uses external pipeline scripts launched from this shell. Open the controls panel to run mask generation and training jobs.");
            ImGui.Separator();
            ImGui.TextDisabled(_datasetLastCommand);
            ImGui.End();
            return;
        }

        if (_session.WorkspaceMode == WowViewerWorkspaceMode.ModelOutputs)
        {
            bool hasModelOutputPreview = _modelOutputGpuRenderer?.HasRenderableGeometry == true && _modelOutputGpuRenderer.PreviewTextureHandle != 0;
            if (_currentModelOutputScene == null || !hasModelOutputPreview)
            {
                ImGui.TextWrapped("No model-output tile scene loaded yet.");
                ImGui.End();
                return;
            }

            ImGui.TextDisabled(_lastLoadSummary);
            ImGui.Separator();

            Vector2 modelOutputAvailable = ImGui.GetContentRegionAvail();
            float modelOutputSize = MathF.Min(modelOutputAvailable.X, modelOutputAvailable.Y);
            modelOutputSize = MathF.Max(modelOutputSize, 128f);
            ImGui.Image((nint)_modelOutputGpuRenderer!.PreviewTextureHandle, new Vector2(modelOutputSize, modelOutputSize), new Vector2(0, 1), new Vector2(1, 0));
            HandleModelOutputPreviewMouseInput(new Vector2(modelOutputSize, modelOutputSize));
            ImGui.TextDisabled($"GPU Preview: {_session.VisualSize}x{_session.VisualSize} commands={_modelOutputGpuRenderer.CommandCount} tiles={_currentModelOutputScene.Tiles.Count}");
            ImGui.End();
            return;
        }

        if (_session.WorkspaceMode == WowViewerWorkspaceMode.StandaloneWmo)
        {
            bool hasWmoPreview = _wmoGpuPreviewRenderer?.HasRenderableGeometry == true && _wmoGpuPreviewRenderer.PreviewTextureHandle != 0;
            if (_currentWmoPreview == null || !hasWmoPreview)
            {
                ImGui.TextWrapped("No WMO GPU preview texture uploaded yet.");
                ImGui.End();
                return;
            }

            ImGui.TextDisabled(_lastLoadSummary);
            ImGui.Separator();

            Vector2 wmoAvailable = ImGui.GetContentRegionAvail();
            float wmoSize = MathF.Min(wmoAvailable.X, wmoAvailable.Y);
            wmoSize = MathF.Max(wmoSize, 128f);
            ImGui.Image((nint)_wmoGpuPreviewRenderer!.PreviewTextureHandle, new Vector2(wmoSize, wmoSize), new Vector2(0, 1), new Vector2(1, 0));
            HandleWmoPreviewMouseInput(new Vector2(wmoSize, wmoSize));
            ImGui.TextDisabled($"GPU Preview: {_session.VisualSize}x{_session.VisualSize} commands={_wmoGpuPreviewRenderer.CommandCount}");
            ImGui.End();
            return;
        }

        if (_session.WorkspaceMode == WowViewerWorkspaceMode.StandaloneMdx)
        {
            bool hasMdxPreview = _mdxGpuPreviewRenderer?.HasRenderableGeometry == true && _mdxGpuPreviewRenderer.PreviewTextureHandle != 0;
            if (_currentMdxPreview == null || !hasMdxPreview)
            {
                ImGui.TextWrapped("No MDX GPU preview texture uploaded yet.");
                ImGui.End();
                return;
            }

            ImGui.TextDisabled(_lastLoadSummary);
            ImGui.Separator();

            Vector2 mdxAvailable = ImGui.GetContentRegionAvail();
            float mdxSize = MathF.Min(mdxAvailable.X, mdxAvailable.Y);
            mdxSize = MathF.Max(mdxSize, 128f);
            ImGui.Image((nint)_mdxGpuPreviewRenderer!.PreviewTextureHandle, new Vector2(mdxSize, mdxSize), new Vector2(0, 1), new Vector2(1, 0));
            HandleMdxPreviewMouseInput(new Vector2(mdxSize, mdxSize));
            ImGui.TextDisabled($"GPU Preview: {_session.VisualSize}x{_session.VisualSize} commands={_mdxGpuPreviewRenderer.CommandCount}");
            ImGui.End();
            return;
        }

        bool hasGpuPreview = _gpuPreviewRenderer?.HasRenderableGeometry == true && _gpuPreviewRenderer.PreviewTextureHandle != 0;
        bool hasSoftwarePreview = _previewTextureHandle != 0;
        if (_currentPreview == null || (!hasGpuPreview && !hasSoftwarePreview))
        {
            ImGui.TextWrapped("No preview texture uploaded yet.");
            ImGui.End();
            return;
        }

        M2SoftwareVisualSnapshot snapshot = _currentPreview.FrameResult.VisualSnapshot;
        ImGui.TextDisabled(_lastLoadSummary);
        ImGui.Separator();

        Vector2 available = ImGui.GetContentRegionAvail();
        float size = MathF.Min(available.X, available.Y);
        size = MathF.Max(size, 128f);
        if (hasGpuPreview)
        {
            ImGui.Image((nint)_gpuPreviewRenderer!.PreviewTextureHandle, new Vector2(size, size), new Vector2(0, 1), new Vector2(1, 0));
            HandleM2PreviewMouseInput(new Vector2(size, size));
            ImGui.TextDisabled($"GPU Preview: {_session.VisualSize}x{_session.VisualSize} commands={_gpuPreviewRenderer.CommandCount}");
            if (hasSoftwarePreview)
                ImGui.TextDisabled($"Software Snapshot: {snapshot.Width}x{snapshot.Height} litPixels={snapshot.LitPixelCount}");
        }
        else
        {
            ImGui.Image((nint)_previewTextureHandle, new Vector2(size, size), new Vector2(0, 1), new Vector2(1, 0));
            ImGui.TextDisabled($"Software Snapshot: {snapshot.Width}x{snapshot.Height} litPixels={snapshot.LitPixelCount}");
            ImGui.TextDisabled("GPU Preview: fallback inactive for this loaded frame.");
        }

        ImGui.End();
    }

    private void DrawWorldNavigatorSourceSection()
    {
        string clientRoot = _session.World.ClientRoot;
        string buildLabel = _session.World.BuildLabel;
        string looseOverlayRoot = _session.World.LooseOverlayRoot;
        string mapInput = _session.World.MapInput;
        int tileX = _session.World.TileX;
        int tileY = _session.World.TileY;

        if (ImGui.CollapsingHeader("Source", ImGuiTreeNodeFlags.None))
        {
            ImGui.TextDisabled("Client Root");
            ImGui.SetNextItemWidth(-1);
            ImGui.InputText("##WorldClientRoot", ref clientRoot, 1024);
            ImGui.TextDisabled("Build Label");
            ImGui.SetNextItemWidth(-1);
            ImGui.InputText("##WorldBuildLabel", ref buildLabel, 256);
            ImGui.TextDisabled("Loose Overlay Root");
            ImGui.SetNextItemWidth(-1);
            ImGui.InputText("##WorldLooseOverlayRoot", ref looseOverlayRoot, 1024);
            ApplySharedClientSelection(clientRoot, buildLabel, looseOverlayRoot);

            if (ImGui.Button("Open Game Client...", new Vector2(-1, 0)))
                PromptOpenGameFolder();
            if (ImGui.Button("Attach Loose Folder...", new Vector2(-1, 0)))
                PromptAttachLooseFolder();

            bool canBrowseCurrentClientFiles = !string.IsNullOrWhiteSpace(_session.Source.ArchiveRoot);
            if (!canBrowseCurrentClientFiles)
                ImGui.BeginDisabled();
            if (ImGui.Button("Browse Current Client Files...", new Vector2(-1, 0)))
                OpenAssetFileBrowserForClient(_session.World.ClientRoot, _session.World.BuildLabel, _session.World.LooseOverlayRoot, AssetFileBrowserFilter.SupportedAssets, "World Client Files");
            if (!canBrowseCurrentClientFiles)
                ImGui.EndDisabled();
        }

        ImGui.Separator();
        if (ImGui.CollapsingHeader("World Maps", ImGuiTreeNodeFlags.DefaultOpen))
        {
            ImGui.SetNextItemWidth(-1);
            ImGui.InputText("##WorldMap", ref mapInput, 256);
            ImGui.TextDisabled("Map");
            ImGui.SetNextItemWidth(ImGui.GetContentRegionAvail().X * 0.5f);
            ImGui.InputInt("Tile X", ref tileX);
            ImGui.SetNextItemWidth(ImGui.GetContentRegionAvail().X * 0.5f);
            ImGui.InputInt("Tile Y", ref tileY);
            _session.World.MapInput = mapInput;
            _session.World.TileX = tileX;
            _session.World.TileY = tileY;

            if (ImGui.Button("Refresh Maps", new Vector2(-1, 0)))
                RefreshDiscoveredWorldMaps(force: true);

            if (ImGui.Button("Browse Maps...", new Vector2(-1, 0)))
                OpenWorldMapBrowserForCurrentClient();

            RefreshDiscoveredWorldMaps();
            ImGui.TextDisabled(_worldMapDiscoverySummary);
            if (_discoveredWorldMaps.Count > 0)
            {
                float listHeight = MathF.Min(165.0f, 24.0f * _discoveredWorldMaps.Count + 8.0f);
                if (ImGui.BeginChild("##WorldNavigatorMapDiscoveryList", new Vector2(0, listHeight), true))
                {
                    foreach (DiscoveredLooseWorldMap map in _discoveredWorldMaps)
                    {
                        string label = BuildWorldMapLabel(map);
                        bool selected = string.Equals(_session.World.MapInput, map.Directory, StringComparison.OrdinalIgnoreCase);
                        if (ImGui.Selectable(label, selected))
                            _session.World.MapInput = map.Directory;

                        if (ImGui.IsItemHovered())
                            ImGui.SetTooltip(BuildWorldMapTooltip(map));
                    }
                }

                ImGui.EndChild();
            }
        }

        ImGui.Separator();
        if (ImGui.CollapsingHeader("Spawn", ImGuiTreeNodeFlags.DefaultOpen))
            DrawWorldSpawnPickerSection();

        ImGui.Separator();
        if (ImGui.CollapsingHeader("Session", ImGuiTreeNodeFlags.DefaultOpen))
        {
            bool worldLoadPending = IsWorldLoadPending();
            if (worldLoadPending)
                ImGui.TextDisabled($"World load in progress for {_pendingWorldLoadMapInput} ({_pendingWorldLoadStopwatch?.Elapsed.TotalSeconds ?? 0:F1}s)");

            if (worldLoadPending)
                ImGui.BeginDisabled();

            if (ImGui.Button(_currentWorldRuntimeFrame == null ? "Open World Session" : "Reload World Session", new Vector2(-1, 0)))
                LoadActiveWorkspace();

            if (worldLoadPending)
                ImGui.EndDisabled();

            ImGui.Separator();

        float cameraSpeed = _session.World.CameraMoveSpeed;
        ImGui.TextDisabled("Viewer Camera");
        ImGui.SetNextItemWidth(-1);
        if (ImGui.SliderFloat("##WorldCameraMoveSpeed", ref cameraSpeed, 10.0f, 1200.0f, "%.0f units/s", ImGuiSliderFlags.Logarithmic))
            _session.World.CameraMoveSpeed = cameraSpeed;
        if (ImGui.Button("Reset Camera", new Vector2(-1, 0)))
            _worldViewCamera.Reset();

        ImGui.Separator();

        bool showWmos = _session.World.ShowWmos;
        bool showDoodads = _session.World.ShowDoodads;
        bool showSky = _session.World.ShowSky;
        bool showWdl = false;
        bool showTerrain = _session.World.ShowTerrain;
        bool showLiquid = _session.World.ShowLiquid;
        bool showOverlay = _session.World.ShowOverlay;
        bool ignoreTerrainHoles = _session.World.IgnoreTerrainHoles;
        bool showHoleOverlay = _session.World.ShowHoleOverlay;
        ImGui.TextDisabled("World Layers");
        ImGui.Checkbox("Sky", ref showSky);
        ImGui.SameLine();
        ImGui.Checkbox("Terrain", ref showTerrain);
        ImGui.SameLine();
        ImGui.TextDisabled("Far WDL off");
        ImGui.Checkbox("Liquid", ref showLiquid);
        ImGui.SameLine();
        ImGui.Checkbox("Overlay", ref showOverlay);
        ImGui.TextDisabled("Object Families");
        ImGui.Checkbox("WMO", ref showWmos);
        ImGui.SameLine();
        ImGui.Checkbox("Doodads (MDX/M2)", ref showDoodads);
        ImGui.TextDisabled("Terrain Debug");
        ImGui.Checkbox("Ignore Terrain Holes", ref ignoreTerrainHoles);
        ImGui.SameLine();
        ImGui.Checkbox("Show Hole Overlay", ref showHoleOverlay);
        _session.World.ShowWmos = showWmos;
        _session.World.ShowDoodads = showDoodads;
        _session.World.ShowSky = showSky;
        _session.World.ShowWdl = showWdl;
        _session.World.ShowTerrain = showTerrain;
        _session.World.ShowLiquid = showLiquid;
        _session.World.ShowOverlay = showOverlay;
        bool debugSettingsChanged = _session.World.IgnoreTerrainHoles != ignoreTerrainHoles
            || _session.World.ShowHoleOverlay != showHoleOverlay;
        _session.World.IgnoreTerrainHoles = ignoreTerrainHoles;
        _session.World.ShowHoleOverlay = showHoleOverlay;
        _session.Normalize();

        if (debugSettingsChanged && _currentWorldRuntimeFrame != null)
            EnsureWorldGpuPreviewRenderer()?.LoadPreview(_currentWorldRuntimeFrame, _worldViewCamera, _session.World.IgnoreTerrainHoles, _session.World.ShowHoleOverlay);
        }
    }

    private void ApplyLegacyWorldSessionWindowPreset()
    {
        _compactWorldSessionLayout = true;
        _showControlWindow = true;
        _showWorkspaceWindow = false;
        _showDiagnosticsWindow = true;
        _showWorldStatusWindow = false;
        _showNavigatorWindow = false;
        _showInspectorWindow = false;
        _showBoundaryWindow = false;
        _showAboutWindow = false;
    }

    private void DrawWorldSessionPreview()
    {
        if (IsWorldLoadPending())
        {
            string elapsed = _pendingWorldLoadStopwatch == null ? string.Empty : $" ({_pendingWorldLoadStopwatch.Elapsed.TotalSeconds:F1}s)";
            ImGui.TextDisabled($"Opening {_pendingWorldLoadMapInput}{elapsed}...");
        }

        WowViewerWorldSessionBootstrapResult? worldSession = _currentWorldSession;
        if (worldSession == null)
        {
            ImGui.TextDisabled("No world session opened yet.");
            return;
        }

        WowViewerWorldSceneSnapshot sceneSnapshot = _worldSceneHost.SceneSnapshot;

        if (_currentWorldRuntimeFrame == null)
        {
            ImGui.TextUnformatted(sceneSnapshot.ResolvedMapDirectory);
            ImGui.TextDisabled($"Tiles with data: {worldSession.WdtSummary.TilesWithData}/{worldSession.WdtSummary.TotalTiles}");
            ImGui.TextDisabled($"Occupied sample: {FormatTileSample(sceneSnapshot.OccupiedTiles, 12)}");
            return;
        }

        WowViewerWorldRuntimeFrameResult runtimeFrame = _currentWorldRuntimeFrame;
        if (_worldGpuPreviewRenderer is { HasRenderableGeometry: true, PreviewTextureHandle: not 0 } worldPreviewRenderer)
        {
            Vector2 gpuPreviewAvailable = ImGui.GetContentRegionAvail();
            float previewWidth = MathF.Max(280f, gpuPreviewAvailable.X);
            float statusHeight = ImGui.GetFrameHeightWithSpacing() * 2.0f;
            float previewHeight = MathF.Max(240f, gpuPreviewAvailable.Y - statusHeight);
            Vector2 previewSize = new(previewWidth, previewHeight);
            ImGuiWindowFlags viewportFlags = ImGuiWindowFlags.NoScrollbar | ImGuiWindowFlags.NoScrollWithMouse;
            float parentScrollY = ImGui.GetScrollY();
            bool consumedViewportWheel = false;
            ImGui.PushStyleVar(ImGuiStyleVar.WindowPadding, Vector2.Zero);
            if (ImGui.BeginChild("##WorldGpuViewport", previewSize, false, viewportFlags))
            {
                Vector2 viewportMin = ImGui.GetCursorScreenPos();
                Vector2 viewportMax = viewportMin + previewSize;
                ImGui.GetWindowDrawList().AddImage((nint)worldPreviewRenderer.PreviewTextureHandle, viewportMin, viewportMax, new Vector2(0, 1), new Vector2(1, 0));
                ImGui.InvisibleButton("##WorldGpuViewportInput", previewSize);
                consumedViewportWheel = HandleWorldPreviewInput(previewSize);
            }

            ImGui.EndChild();
            ImGui.PopStyleVar();
            if (consumedViewportWheel)
                ImGui.SetScrollY(parentScrollY);
            ImGui.TextDisabled($"{sceneSnapshot.ResolvedMapDirectory}  tile ({sceneSnapshot.SelectedTileX},{sceneSnapshot.SelectedTileY})  terrain tris {worldPreviewRenderer.TerrainTriangleCount}  markers {worldPreviewRenderer.MarkerCount}  speed {_session.World.CameraMoveSpeed:F0}");
            ImGui.TextDisabled("Click viewport, right-drag look, wheel dolly, WASD move, Q/E vertical, Shift faster, double-click reset.");
        }
        else
        {
            ImGui.TextDisabled("GPU world preview is not available for this frame.");
        }
    }

    private void DrawWorldDebugViews(WowViewerWorldRuntimeFrameResult result)
    {
        if (_worldTerrainPreviewTextureHandle != 0)
        {
            ImGui.TextDisabled("Software Terrain Preview");
            Vector2 previewAvailable = ImGui.GetContentRegionAvail();
            float previewSize = MathF.Max(180f, MathF.Min(previewAvailable.X, 320f));
            ImGui.Image((nint)_worldTerrainPreviewTextureHandle, new Vector2(previewSize, previewSize));
            ImGui.TextDisabled($"{result.TerrainVisualSnapshot.Width}x{result.TerrainVisualSnapshot.Height} samples={result.TerrainVisualSnapshot.SampledPixelCount}");
            ImGui.TextDisabled($"Range {FormatTerrainHeightRange(result.TerrainTileData)} hash={result.TerrainVisualSnapshot.VisualHash}");
        }

        ImGui.Separator();
        ImGui.TextDisabled("Object Marker Canvas");
        Vector2 available = ImGui.GetContentRegionAvail();
        float canvasSize = MathF.Max(200f, MathF.Min(available.X, MathF.Max(available.Y, 200f)));
        Vector2 canvas = new(canvasSize, canvasSize);
        Vector2 origin = ImGui.GetCursorScreenPos();
        ImGui.InvisibleButton("worldRuntimeCanvas", canvas);
        if (ImGui.IsItemHovered() && ImGui.IsMouseClicked(ImGuiMouseButton.Left))
            TrySelectWorldObjectAtCanvasPoint(result, origin, canvas, ImGui.GetIO().MousePos);
        DrawWorldRuntimeCanvas(origin, canvas, result);
    }

    private void DrawWorldMinimapWindow()
    {
        ImGui.SetNextWindowSize(new Vector2(420, 520), ImGuiCond.FirstUseEver);
        if (!ImGui.Begin("World Minimap", ref _showWorldMinimapWindow))
        {
            ImGui.End();
            return;
        }

        if (_session.WorkspaceMode != WowViewerWorkspaceMode.WorldSession)
        {
            ImGui.TextWrapped("Switch to World Session to inspect or select world tiles from the minimap surface.");
            ImGui.End();
            return;
        }

        DrawWorldMinimapContents();
        ImGui.End();
    }

    private void DrawDiagnosticsWindow(float deltaSeconds, bool forceDockedSize = false)
    {
        if (!forceDockedSize)
            ImGui.SetNextWindowSize(new Vector2(480, 720), ImGuiCond.FirstUseEver);
        string diagnosticsTitle = _session.WorkspaceMode == WowViewerWorkspaceMode.WorldSession ? "Inspector" : "Diagnostics";
        if (!ImGui.Begin(diagnosticsTitle, ref _showDiagnosticsWindow))
        {
            ImGui.End();
            return;
        }

        ImGui.TextDisabled(_session.GetWorkspaceLabel());
        if (!string.IsNullOrWhiteSpace(_lastError))
            ImGui.TextColored(new Vector4(0.95f, 0.42f, 0.32f, 1.0f), _lastError);

        if (!IsImplementedWorkspace(_session.WorkspaceMode))
        {
            ImGui.TextWrapped($"{_session.GetWorkspaceLabel()} diagnostics are not implemented yet. This workspace exists so later WMO or MDX consumers can land without reshaping the whole shell again.");
            ImGui.End();
            return;
        }

        if (_session.WorkspaceMode == WowViewerWorkspaceMode.WorldSession)
        {
            if (_currentWorldRuntimeFrame != null
                && _selectedWorldObject.HasValue
                && TryResolveWorldNavigatorEntry(_currentWorldRuntimeFrame, _selectedWorldObject.Value, out WorldNavigatorEntry selectedEntry))
            {
                if (ImGui.CollapsingHeader("Selection", ImGuiTreeNodeFlags.DefaultOpen))
                    DrawWorldInspectorContents(selectedEntry);
            }
            else
            {
                ImGui.TextDisabled("Select an object from the Object Navigator to inspect it here.");
            }

            ImGui.Separator();
            if (ImGui.CollapsingHeader("Runtime Summary", ImGuiTreeNodeFlags.DefaultOpen))
                DrawWorldRuntimeSummary();

            if (_currentWorldRuntimeFrame != null)
            {
                ImGui.Separator();
                ImGuiTreeNodeFlags objectNavigatorFlags = _selectedWorldObject.HasValue
                    ? ImGuiTreeNodeFlags.None
                    : ImGuiTreeNodeFlags.DefaultOpen;
                if (ImGui.CollapsingHeader("Object Navigator", objectNavigatorFlags))
                    DrawWorldNavigatorEntriesSection(_currentWorldRuntimeFrame);
            }

            ImGui.Separator();
            if (ImGui.CollapsingHeader("Minimap", ImGuiTreeNodeFlags.None))
                DrawWorldMinimapContents();

            if (ImGui.CollapsingHeader("Deep Diagnostics", ImGuiTreeNodeFlags.None))
                DrawWorldDiagnostics();

            if (_currentWorldRuntimeFrame != null && ImGui.CollapsingHeader("Debug Views", ImGuiTreeNodeFlags.None))
                DrawWorldDebugViews(_currentWorldRuntimeFrame);

            if (ImGui.CollapsingHeader("Runtime Boundaries", ImGuiTreeNodeFlags.None))
                DrawBoundaryContents();

            if (ImGui.CollapsingHeader("About / Commands", ImGuiTreeNodeFlags.None))
                DrawAboutContents();

            ImGui.End();
            return;
        }

        if (_session.WorkspaceMode == WowViewerWorkspaceMode.ModelOutputs)
        {
            if (_currentModelOutputScene == null)
            {
                ImGui.TextWrapped("Load a model-output scene to inspect tile counts, bounds, and GPU command totals.");
                ImGui.End();
                return;
            }

            ImGui.Text($"Load: {_currentModelOutputScene.LoadDuration.TotalMilliseconds:F1} ms");
            ImGui.Text($"Source: {_currentModelOutputScene.SourcePath}");
            ImGui.Text($"Variant: {_session.ModelOutput.Variant}");
            ImGui.Text($"Tiles: {_currentModelOutputScene.Tiles.Count}");
            ImGui.Text($"Vertices: {_currentModelOutputScene.VertexCount}");
            ImGui.Text($"Triangles: {_currentModelOutputScene.TriangleCount}");
            ImGui.Text($"Objects: {_currentModelOutputScene.ObjectCount}");
            ImGui.Text($"Camera Mode: {_session.ModelOutput.CameraMode}");
            ImGui.Text($"GPU Preview Commands: {_modelOutputGpuRenderer?.CommandCount ?? 0}");
            ImGui.Text($"Bounds Min: {_currentModelOutputScene.BoundsMin}");
            ImGui.Text($"Bounds Max: {_currentModelOutputScene.BoundsMax}");
            ImGui.End();
            return;
        }

        if (_session.WorkspaceMode == WowViewerWorkspaceMode.StandaloneWmo)
        {
            if (_currentWmoPreview == null)
            {
                ImGui.TextWrapped("Load a WMO preview to inspect geometry, portal ownership, and doodad placement data.");
                ImGui.End();
                return;
            }

            WmoRenderDocument document = _currentWmoPreview.Document;
            ImGui.TextDisabled("Frame Summary");
            ImGui.Text($"Source: {_session.Source.Describe()}");
            ImGui.Text($"Load: {_currentWmoPreview.LoadDuration.TotalMilliseconds:F1} ms");
            ImGui.Text($"Delta: {deltaSeconds * 1000f:F2} ms");
            ImGui.Text($"GPU Preview Commands: {_wmoGpuPreviewRenderer?.CommandCount ?? 0}");
            ImGui.Separator();

            ImGui.TextDisabled("WMO Summary");
            ImGui.Text($"Version: {(document.Version?.ToString() ?? "n/a")}");
            ImGui.Text($"Materials: {document.Materials.Count}");
            ImGui.Text($"Groups: {document.Groups.Count}");
            ImGui.Text($"Portals: {document.Portals.Count}");
            ImGui.Text($"Portal Refs: {document.PortalReferences.Count}");
            ImGui.Text($"Doodad Sets: {document.DoodadSets.Count}");
            ImGui.Text($"Doodad Placements: {document.DoodadPlacements.Count}");
            ImGui.Text($"Bounds Min: {document.Summary.BoundsMin}");
            ImGui.Text($"Bounds Max: {document.Summary.BoundsMax}");

            ImGui.Separator();
            ImGui.TextDisabled("Group Samples");
            foreach (WmoEmbeddedGroupMeshDetail group in document.Groups.Take(12))
                ImGui.BulletText($"#{group.GroupIndex} verts={group.Mesh.Vertices.Count} tris={group.Mesh.Indices.Count / 3} batches={group.Mesh.Batches.Count} doodads={group.DoodadRefs.Count} lights={group.LightRefs.Count}");
            if (document.Groups.Count > 12)
                ImGui.TextDisabled($"... {document.Groups.Count - 12} more groups");

            if (document.DoodadSets.Count > 0)
            {
                ImGui.Separator();
                ImGui.TextDisabled("Doodad Set Samples");
                foreach (WmoDoodadSetDetail set in document.DoodadSets.Take(6))
                    ImGui.BulletText($"#{set.SetIndex} {set.Name} start={set.StartIndex} count={set.Count} flags=0x{set.RawFlags:X}");
            }

            if (document.DoodadPlacements.Count > 0)
            {
                ImGui.Separator();
                ImGui.TextDisabled("Doodad Placement Samples");
                foreach (WmoDoodadPlacementDetail placement in document.DoodadPlacements.Take(6))
                    ImGui.BulletText($"#{placement.PlacementIndex} {placement.ModelPath} scale={placement.Scale:F2} alpha={placement.Alpha} pos=({placement.Position.X:F1},{placement.Position.Y:F1},{placement.Position.Z:F1})");
            }

            ImGui.End();
            return;
        }

        if (_session.WorkspaceMode == WowViewerWorkspaceMode.StandaloneMdx)
        {
            if (_currentMdxPreview == null)
            {
                ImGui.TextWrapped("Load an MDX preview to inspect geometry, material, and GPU-preview counts.");
                ImGui.End();
                return;
            }

            ImGui.TextDisabled("Frame Summary");
            ImGui.Text($"Source: {_session.Source.Describe()}");
            ImGui.Text($"Load: {_currentMdxPreview.LoadDuration.TotalMilliseconds:F1} ms");
            ImGui.Text($"Delta: {deltaSeconds * 1000f:F2} ms");
            ImGui.Text($"GPU Preview Commands: {_mdxGpuPreviewRenderer?.CommandCount ?? 0}");
            ImGui.Separator();

            ImGui.TextDisabled("MDX Summary");
            ImGui.Text($"Model: {_currentMdxPreview.Summary.ModelName ?? _currentMdxPreview.Geometry.SourcePath}");
            ImGui.Text($"Version: {(_currentMdxPreview.Summary.Version?.ToString() ?? "n/a")}");
            ImGui.Text($"Geosets: {_currentMdxPreview.Geometry.GeosetCount}");
            ImGui.Text($"Materials: {_currentMdxPreview.Summary.MaterialCount}");
            ImGui.Text($"Textures: {_currentMdxPreview.Summary.TextureCount}");
            ImGui.Text($"Layers: {_currentMdxPreview.Summary.MaterialLayerCount}");
            ImGui.Text($"Helpers: {_currentMdxPreview.Helpers.HelperCount}");
            ImGui.Text($"Attachments: {_currentMdxPreview.Attachments.AttachmentCount}");
            ImGui.Text($"Events: {_currentMdxPreview.Events.EventCount}");
            ImGui.Text($"Particle Emitters 2: {_currentMdxPreview.ParticleEmitters.ParticleEmitterCount}");
            ImGui.Text($"Ribbons: {_currentMdxPreview.Ribbons.RibbonCount}");
            ImGui.Text($"Cameras: {_currentMdxPreview.Cameras.CameraCount}");

            ImGui.Separator();
            ImGui.TextDisabled("MDX Runtime");
            ImGui.Text($"Triggered Events: {_currentMdxPreview.EffectRuntimeState.TriggeredEventCount}/{_currentMdxPreview.EffectRuntimeState.Events.Count}");
            ImGui.Text($"Visible Particles: {_currentMdxPreview.EffectRuntimeState.VisibleParticleEmitterCount}/{_currentMdxPreview.EffectRuntimeState.Particles.Count}");
            ImGui.Text($"Visible Ribbons: {_currentMdxPreview.EffectRuntimeState.VisibleRibbonEmitterCount}/{_currentMdxPreview.EffectRuntimeState.Ribbons.Count}");

            ImGui.Separator();
            ImGui.TextDisabled("Geoset Samples");
            foreach (MdxGeosetGeometry geoset in _currentMdxPreview.Geometry.Geosets.Take(12))
                ImGui.BulletText($"#{geoset.Index} verts={geoset.VertexCount} tris={geoset.TriangleCount} material={geoset.MaterialId} uvSets={geoset.UvSetCount}");

            if (_currentMdxPreview.Geometry.GeosetCount > 12)
                ImGui.TextDisabled($"... {_currentMdxPreview.Geometry.GeosetCount - 12} more geosets");

            if (_currentMdxPreview.ParticleEmitters.ParticleEmitterCount > 0)
            {
                ImGui.Separator();
                ImGui.TextDisabled("Particle Emitter Samples");
                foreach (MdxParticleEmitter2 particleEmitter in _currentMdxPreview.ParticleEmitters.ParticleEmitters.Take(6))
                {
                    string models = particleEmitter.HasGeometryModel || particleEmitter.HasRecursionModel
                        ? $" geo={particleEmitter.GeometryModel ?? "-"} recur={particleEmitter.RecursionModel ?? "-"}"
                        : string.Empty;
                    string tracks = $" tracks[v={(particleEmitter.VisibilityTrack is not null ? 1 : 0)} s={(particleEmitter.SpeedTrack is not null ? 1 : 0)} e={(particleEmitter.EmissionRateTrack is not null ? 1 : 0)} l={(particleEmitter.LifeTrack is not null ? 1 : 0)}]";
                    ImGui.BulletText($"#{particleEmitter.Index} {particleEmitter.Name} tex={particleEmitter.TextureId} blend={particleEmitter.BlendMode} rows={particleEmitter.Rows} cols={particleEmitter.Columns}{models}{tracks}");
                }

                if (_currentMdxPreview.ParticleEmitters.ParticleEmitterCount > 6)
                    ImGui.TextDisabled($"... {_currentMdxPreview.ParticleEmitters.ParticleEmitterCount - 6} more particle emitters");
            }

            if (_currentMdxPreview.EffectRuntimeState.Events.Count > 0)
            {
                ImGui.Separator();
                ImGui.TextDisabled("Event Runtime Samples");
                foreach (MdxEventRuntimeState effectEvent in _currentMdxPreview.EffectRuntimeState.Events.Take(6))
                {
                    string state = effectEvent.Triggered ? "triggered" : "idle";
                    string nextKey = effectEvent.NextKeyTime?.ToString() ?? "-";
                    ImGui.BulletText($"#{effectEvent.Index} {effectEvent.Name} {effectEvent.Tag} {state} frame={effectEvent.ResolvedFrameTime} next={nextKey}");
                }
            }

            if (_currentMdxPreview.EffectRuntimeState.Particles.Count > 0)
            {
                ImGui.Separator();
                ImGui.TextDisabled("Particle Runtime Samples");
                foreach (MdxParticleEmitter2RuntimeState particle in _currentMdxPreview.EffectRuntimeState.Particles.Take(6))
                {
                    string state = particle.Enabled ? "on" : "off";
                    string modelFlag = particle.UsesModelParticles ? " model" : string.Empty;
                    ImGui.BulletText($"#{particle.Index} {particle.Name} {state} count={particle.EstimatedParticleCount} emit={particle.EmissionRate:F2} life={particle.Life:F2} speed={particle.Speed:F2} fx={particle.EffectKey}{modelFlag}");
                }
            }

            if (_currentMdxPreview.EffectRuntimeState.Ribbons.Count > 0)
            {
                ImGui.Separator();
                ImGui.TextDisabled("Ribbon Runtime Samples");
                foreach (MdxRibbonRuntimeState ribbon in _currentMdxPreview.EffectRuntimeState.Ribbons.Take(6))
                {
                    string state = ribbon.Visible ? "on" : "off";
                    ImGui.BulletText($"#{ribbon.Index} {ribbon.Name} {state} edges={ribbon.EstimatedEdgeCount} alpha={ribbon.Alpha:F2} slot={ribbon.TextureSlot} fx={ribbon.EffectKey}");
                }
            }

            ImGui.End();
            return;
        }

        if (_currentPreview == null)
        {
            ImGui.TextWrapped("Load a preview to inspect runtime hashes, render-frame counts, and submission summaries.");
            ImGui.End();
            return;
        }

        M2RuntimeFrameResult frame = _currentPreview.FrameResult;
        M2RuntimeGoldenFrame golden = frame.GoldenFrame;
        M2RenderFrame render = frame.RenderFrame;

        ImGui.TextDisabled("Frame Summary");
        ImGui.Text($"Source: {_session.Source.Describe()}");
        ImGui.Text($"Load: {_currentPreview.LoadDuration.TotalMilliseconds:F1} ms");
        ImGui.Text($"Delta: {deltaSeconds * 1000f:F2} ms");
        ImGui.Text($"Runtime Hash: {golden.RuntimeHash}");
        ImGui.Text($"Visual Hash: {frame.VisualSnapshot.VisualHash}");
        ImGui.Text($"Render Hash: {render.FrameHash}");
        ImGui.Text($"GPU Preview Commands: {_gpuPreviewRenderer?.CommandCount ?? 0}");
        ImGui.Separator();

        ImGui.TextDisabled("M2 Runtime");
        ImGui.Text($"Model: {golden.CanonicalModelPath}");
        ImGui.Text($"Version: 0x{golden.ModelVersion:X}");
        ImGui.Text($"Sequence: {golden.RequestedSequenceIndex} -> {golden.ResolvedSequenceIndex}");
        ImGui.Text($"Bones: {golden.BoneCount}");
        ImGui.Text($"Skinned Vertices: {golden.SkinnedVertexCount}");
        ImGui.Text($"Visible Passes: {golden.VisiblePassCount}/{golden.RenderPassCount}");
        ImGui.Text($"Commands: {render.CommandCount}");
        ImGui.Text($"Backend Vertices: {render.BackendVertexCount}");
        ImGui.Text($"Backend Indices: {render.BackendIndexCount}");
        ImGui.Text($"Submitted Vertices: {render.SubmittedVertexCount}");
        ImGui.Text($"Submitted Indices: {render.SubmittedIndexCount}");
        ImGui.Text($"Particles: {frame.EffectRuntimeState.Particles.Count}");
        ImGui.Text($"Ribbons: {frame.EffectRuntimeState.Ribbons.Count}");

        ImGui.Separator();
        ImGui.TextDisabled("Submission Batches");
        foreach (M2RuntimeGoldenBatch batch in golden.Batches.Take(12))
            ImGui.BulletText($"#{batch.BatchIndex} {batch.Family} {batch.Handler} entries={batch.EntryCount} v={batch.VertexCount} i={batch.IndexCount}");

        if (golden.Batches.Count > 12)
            ImGui.TextDisabled($"... {golden.Batches.Count - 12} more batches");

        ImGui.Separator();
        ImGui.TextDisabled("World Runtime Staging");
        WorldRenderFrameStats worldStats = WorldRenderFrameStats.Empty;
        ImGui.Text($"Visible WMO: {worldStats.VisibleWmoCount}");
        ImGui.Text($"Visible Doodads: {worldStats.VisibleMdxCount}");
        ImGui.Text($"Taxi Doodads: {worldStats.VisibleTaxiMdxCount}");
        ImGui.Text($"Pending Loads: {worldStats.PendingAssetLoadCount}");
        bool objectPhaseEnabled = WorldFramePassCoordinator.Execute(
            new WorldFramePassOptions(true, true, true),
            new WorldFramePasses(
                static () => { },
                static () => { },
                static () => { },
                static () => { },
                static () => { },
                static () => { },
                static () => { },
                static () => { },
                static () => { },
                static () => { },
                static () => { }));
        ImGui.Text($"Object pass coordinator reachable: {objectPhaseEnabled}");

        ImGui.Separator();
        if (ImGui.CollapsingHeader("Runtime Boundaries", ImGuiTreeNodeFlags.None))
            DrawBoundaryContents();

        if (ImGui.CollapsingHeader("About / Commands", ImGuiTreeNodeFlags.None))
            DrawAboutContents();

        ImGui.End();
    }

    private void DrawWorldDiagnostics()
    {
        WowViewerWorldDiagnosticsSnapshot diagnosticsSnapshot = _worldSceneHost.DiagnosticsSnapshot;
        if (!diagnosticsSnapshot.HasSession)
        {
            ImGui.TextWrapped("Open a world session to inspect WDT summary, MAIN flags, and occupied tile samples.");
            return;
        }

        if (diagnosticsSnapshot.HasRuntime)
        {
            DrawWorldRuntimeDiagnostics(diagnosticsSnapshot);
            return;
        }

        ImGui.TextDisabled("World Session Summary");
        ImGui.Text($"Root: {diagnosticsSnapshot.ClientRoot}");
        ImGui.Text($"Map: {diagnosticsSnapshot.RequestedMapInput} -> {diagnosticsSnapshot.ResolvedMapDirectory}");
        ImGui.Text($"Load: {diagnosticsSnapshot.LoadDuration.TotalMilliseconds:F1} ms");
        ImGui.Text($"WDT Kind: {diagnosticsSnapshot.WdtKindText}");
        ImGui.Text($"WDT Version: {diagnosticsSnapshot.WdtVersionText}");
        ImGui.Text($"WDT Chunks: {diagnosticsSnapshot.WdtChunkCount}");
        ImGui.Separator();

        WdtSummary? summary = diagnosticsSnapshot.WdtSummary;
        if (summary == null)
            return;

        ImGui.TextDisabled("WDT Semantics");
        ImGui.Text($"Tiles With Data: {summary.TilesWithData}/{summary.TotalTiles}");
        ImGui.Text($"Main Cell Bytes: {summary.MainCellSizeBytes}");
        ImGui.Text($"Doodad Names: {summary.DoodadNameCount}");
        ImGui.Text($"WMO Names: {summary.WorldModelNameCount}");
        ImGui.Text($"Doodad Placements: {summary.DoodadPlacementCount}");
        ImGui.Text($"WMO Placements: {summary.WorldModelPlacementCount}");
        ImGui.Text($"WMO-based: {summary.IsWmoBased}");

        if (summary.MainFlags is not null)
        {
            ImGui.Separator();
            ImGui.TextDisabled("MAIN Flags");
            ImGui.Text($"Any Flags: {summary.MainFlags.CellsWithAnyFlags}");
            ImGui.Text($"Has ADT: {summary.MainFlags.CellsWithHasAdt}");
            ImGui.Text($"All Water: {summary.MainFlags.CellsWithAllWater}");
            ImGui.Text($"Loaded: {summary.MainFlags.CellsWithLoaded}");
            ImGui.Text($"Unknown: {summary.MainFlags.CellsWithUnknownFlags}");
            ImGui.Text($"Async Ids: {summary.MainFlags.CellsWithAsyncId}");
            ImGui.TextWrapped($"Distinct: {FormatWdtMainFlags(summary.MainFlags)}");
        }

        ImGui.Separator();
        ImGui.TextDisabled("Occupied Tile Sample");
        ImGui.TextWrapped(FormatTileSample(diagnosticsSnapshot.OccupiedTiles, 24));
    }

    private void DrawWorldRuntimeSummary()
    {
        if (_currentWorldSession == null)
        {
            ImGui.TextDisabled("No world session opened.");
            return;
        }

        WowViewerWorldSceneSnapshot sceneSnapshot = _worldSceneHost.SceneSnapshot;
        ImGui.TextUnformatted(sceneSnapshot.ResolvedMapDirectory);
        ImGui.TextDisabled($"Root: {Path.GetFileName(sceneSnapshot.ClientRoot)}");
        ImGui.TextDisabled($"Load: {sceneSnapshot.LoadDuration.TotalMilliseconds:F1} ms");
        ImGui.TextDisabled($"Occupied tiles: {sceneSnapshot.OccupiedTiles.Count}");

        WowViewerWorldDiagnosticsSnapshot diagnosticsSnapshot = _worldSceneHost.DiagnosticsSnapshot;
        if (!diagnosticsSnapshot.HasRuntime)
            return;

        WowViewerWorldAssetState assetState = _worldSceneHost.AssetState;
        string terrainRange = diagnosticsSnapshot.TerrainTileData is null
            ? "n/a"
            : FormatTerrainHeightRange(diagnosticsSnapshot.TerrainTileData);
        int visibleObjects = assetState.VisibleObjectCount;
        ImGui.Separator();
        ImGui.Text($"Tile: ({sceneSnapshot.SelectedTileX},{sceneSnapshot.SelectedTileY})");
        ImGui.Text($"Active ADT Tiles: {sceneSnapshot.ActiveTerrainTileCount}");
        ImGui.Text($"Placements: WMO {assetState.WmoInstanceCount} / Doodads {assetState.MdxInstanceCount}");
        if (assetState.SkyboxBackdropCount > 0)
            ImGui.Text($"Backdrop: {assetState.SkyboxBackdropCount} classified placement(s)");
        ImGui.Text($"Visible Objects: {visibleObjects}");
        ImGui.Text($"Terrain: {terrainRange}");
        ImGui.Text($"Liquid Layers: {diagnosticsSnapshot.TileStageSummary?.LiquidLayerCount ?? 0}");
        ImGui.Text($"CPU: {diagnosticsSnapshot.TotalCpuMs:F2} ms");
        ImGui.Text($"GPU: {_worldGpuPreviewRenderer?.TerrainTriangleCount ?? 0} terrain tris / {_worldGpuPreviewRenderer?.MarkerCount ?? 0} markers");
        ImGui.Separator();
        ImGui.TextDisabled("Composition");
        foreach (WowViewerWorldCompositionLayerSnapshot layer in diagnosticsSnapshot.CompositionLayers)
        {
            Vector4 color = !layer.Enabled
                ? new Vector4(0.45f, 0.45f, 0.45f, 1.0f)
                : layer.Ready
                    ? new Vector4(0.78f, 0.88f, 0.72f, 1.0f)
                    : new Vector4(0.88f, 0.74f, 0.46f, 1.0f);
            string state = !layer.Enabled
                ? "off"
                : layer.Ready
                    ? $"{layer.SubmittedCount}/{layer.SourceCount}"
                    : "pending";
            ImGui.TextColored(color, $"{layer.DisplayName}: {state}");
            if (ImGui.IsItemHovered() && !string.IsNullOrWhiteSpace(layer.Note))
                ImGui.SetTooltip(layer.Note);
        }
    }

    private void DrawWorldRuntimeDiagnostics(WowViewerWorldDiagnosticsSnapshot diagnosticsSnapshot)
    {
        if (!diagnosticsSnapshot.HasRuntime)
            return;

        WowViewerWorldAssetState assetState = _worldSceneHost.AssetState;
        ImGui.TextDisabled("World Runtime Bridge");
        ImGui.Text($"Tile: ({diagnosticsSnapshot.SelectedTileX},{diagnosticsSnapshot.SelectedTileY})");
        ImGui.Text($"Active ADT Tiles: {diagnosticsSnapshot.ActiveTerrainTileCount}");
        if (diagnosticsSnapshot.ActiveTerrainTileSample.Count > 0)
            ImGui.TextWrapped($"Active Tile Sample: {string.Join(", ", diagnosticsSnapshot.ActiveTerrainTileSample)}");
        ImGui.Text($"Placement Source: {diagnosticsSnapshot.PlacementSourcePath}");
        ImGui.Text($"Camera: {FormatVector3(diagnosticsSnapshot.CameraPosition)} -> {FormatVector3(diagnosticsSnapshot.CameraForward)}");
        ImGui.Text($"Object Phase Executed: {diagnosticsSnapshot.ObjectPhaseExecuted}");
        ImGui.Text($"Pass Options: sky={diagnosticsSnapshot.PassOptions.SkyVisible} wdl={diagnosticsSnapshot.PassOptions.WdlVisible} terrain={diagnosticsSnapshot.PassOptions.TerrainVisible} liquid={diagnosticsSnapshot.PassOptions.LiquidVisible} overlay={diagnosticsSnapshot.PassOptions.OverlayVisible}");
        ImGui.Text($"Object Filters: wmo={diagnosticsSnapshot.PassOptions.WmosVisible} doodads={diagnosticsSnapshot.PassOptions.DoodadsVisible}");
        ImGui.Text($"Total Cpu Ms: {diagnosticsSnapshot.TotalCpuMs:F2}");
        ImGui.Separator();

        ImGui.TextDisabled("Composition Layers");
        foreach (WowViewerWorldCompositionLayerSnapshot layer in diagnosticsSnapshot.CompositionLayers)
            ImGui.TextWrapped($"{layer.DisplayName}: enabled={layer.Enabled} ready={layer.Ready} source={layer.SourceCount} submitted={layer.SubmittedCount} - {layer.Note}");
        ImGui.Separator();

        ImGui.TextDisabled("Placement Inventory");
        ImGui.Text($"WMO Assets Referenced: {assetState.ReferencedWmoAssetCount}");
        ImGui.Text($"WMO Total: {assetState.WmoInstanceCount}");
        ImGui.Text($"WMO Ready: {assetState.ReadyWmoCount}");
        ImGui.Text($"WMO Versions: {(string.IsNullOrEmpty(diagnosticsSnapshot.WmoVersionSummary) ? "n/a" : diagnosticsSnapshot.WmoVersionSummary)}");
        ImGui.Text($"Embedded WMO Doodads: MDX {diagnosticsSnapshot.EmbeddedWmoMdxCount} / M2 {diagnosticsSnapshot.EmbeddedWmoM2Count} / Unknown {diagnosticsSnapshot.EmbeddedWmoUnknownCount}");
        ImGui.Text($"Doodad Assets Referenced: {assetState.ReferencedMdxAssetCount}");
        ImGui.Text($"Doodad Total: {assetState.MdxInstanceCount}");
        ImGui.Text($"Doodad Ready: {assetState.ReadyMdxCount}");
        ImGui.Text($"Skybox Backdrop Candidates: {assetState.SkyboxBackdropCount}");
        if (diagnosticsSnapshot.SkyboxBackdropSamplePaths.Count > 0)
            ImGui.TextWrapped($"Backdrop Sample: {string.Join(", ", diagnosticsSnapshot.SkyboxBackdropSamplePaths)}");
        ImGui.Text($"Pending Assets: {assetState.PendingAssetLoadCount}");
        if (assetState.PendingAssetLoadCount > 0)
            ImGui.TextWrapped($"Pending Sample: {string.Join(", ", assetState.PendingAssetKeys.Take(8))}");

        ImGui.Separator();
        ImGui.TextDisabled("Visibility");
        ImGui.Text($"Visible WMO: {assetState.VisibleWmoCount}");
        ImGui.Text($"Culled WMO: {assetState.CulledWmoCount}");
        ImGui.Text($"Visible Doodads: {assetState.VisibleMdxCount}");
        ImGui.Text($"Culled Doodads: {assetState.CulledMdxCount}");
        ImGui.Text($"Taxi Doodads: {diagnosticsSnapshot.VisibleTaxiDoodadCount}");

        ImGui.Separator();
        ImGui.TextDisabled("Pass Coordination");
        WorldWdlTileData? wdlTileData = diagnosticsSnapshot.WdlTileData;
        WorldTileStageSummary? tileStageSummary = diagnosticsSnapshot.TileStageSummary;
        WorldTerrainTileData? terrainTileData = diagnosticsSnapshot.TerrainTileData;
        WorldLiquidTileData? liquidTileData = diagnosticsSnapshot.LiquidTileData;
        string wdlRange = wdlTileData is null ? "n/a" : FormatHeightRange(wdlTileData);
        string wdlCorners = wdlTileData is null ? "n/a" : FormatWdlCorners(wdlTileData);
        string terrainRange = terrainTileData is null ? "n/a" : FormatTerrainHeightRange(terrainTileData);
        string terrainHeights = terrainTileData is null ? "center=n/a n/a" : $"center={FormatTerrainCenter(terrainTileData)} {FormatTerrainCorners(terrainTileData)}";
        string terrainSample = terrainTileData is null ? "none" : FormatTerrainChunkSample(terrainTileData);
        ImGui.Text($"WDL Tiles: {diagnosticsSnapshot.WdlVisibleTileCount}/{tileStageSummary?.WdlVisibleTileCount ?? 0}");
        ImGui.Text($"WDL Found: {wdlTileData?.SourceFound ?? false}");
        ImGui.Text($"WDL Version: {FormatOptionalUInt(wdlTileData?.Version)}");
        ImGui.Text($"WDL Range: {wdlRange}");
        ImGui.TextWrapped($"WDL Sample: center={FormatOptionalHeight(wdlTileData?.CenterHeight)} {wdlCorners}");
        ImGui.Text($"Terrain Chunks: {diagnosticsSnapshot.TerrainChunksRendered}/{tileStageSummary?.TerrainChunkCount ?? 0}");
        ImGui.Text($"Terrain Hole Chunks: {tileStageSummary?.TerrainHoleChunkCount ?? 0}");
        ImGui.Text($"Terrain Areas: {terrainTileData?.DistinctAreaIdCount ?? 0}");
        ImGui.Text($"Terrain Range: {terrainRange}");
        ImGui.TextWrapped($"Terrain Heights: {terrainHeights}");
        ImGui.Text($"Terrain Preview: {diagnosticsSnapshot.TerrainPreviewWidth}x{diagnosticsSnapshot.TerrainPreviewHeight} samples={diagnosticsSnapshot.TerrainPreviewSampledPixelCount}");
        ImGui.TextWrapped($"Terrain Visual Hash: {diagnosticsSnapshot.TerrainVisualHash}");
        ImGui.TextWrapped($"Terrain Sample: {terrainSample}");
        ImGui.Text($"Liquid Chunks: {liquidTileData?.ActiveChunkCount ?? 0}/{tileStageSummary?.LiquidChunkCount ?? 0}");
        ImGui.Text($"Liquid Layers: {tileStageSummary?.LiquidLayerCount ?? 0}");
        ImGui.Text($"Liquid Visible Tiles: {liquidTileData?.VisibleTileCount ?? 0}/{tileStageSummary?.VisibleLiquidTileCount ?? 0}");
        if (liquidTileData?.Chunks.Count > 0)
        {
            ImGui.TextWrapped($"Liquid Types: {FormatLiquidTypeCounts(liquidTileData)}");
            ImGui.TextWrapped($"Liquid Sample: {FormatLiquidChunkSample(liquidTileData)}");
        }
        ImGui.Text($"WMO Submitted: {diagnosticsSnapshot.WmoSubmittedCount}");
        ImGui.Text($"MDX Animated: {diagnosticsSnapshot.MdxAnimatedSubmittedCount}");
        ImGui.Text($"MDX Opaque Submitted: {diagnosticsSnapshot.MdxOpaqueSubmittedCount}");
        ImGui.Text($"MDX Transparent Submitted: {diagnosticsSnapshot.MdxTransparentSubmittedCount}");
        ImGui.Text($"Opaque Routes: {diagnosticsSnapshot.OpaqueRouteCount}");
        ImGui.Text($"Transparent Routes: {diagnosticsSnapshot.TransparentRouteCount}");

        ImGui.Separator();
        ImGui.TextDisabled("Timing Hint");
        ImGui.TextWrapped(diagnosticsSnapshot.OptimizationHint);
        ImGui.TextWrapped("Boundary: these timings are for the bounded app-side runtime bridge over the software terrain preview plus visibility/pass coordination, not the final 3D world renderer.");
    }

    private static string FormatLiquidTypeCounts(WorldLiquidTileData liquidTileData)
    {
        if (liquidTileData.Chunks.Count == 0)
            return "none";

        return string.Join(", ",
            liquidTileData.Chunks
                .SelectMany(static chunk => chunk.Layers)
                .GroupBy(static layer => layer.BasicType)
                .OrderBy(static group => group.Key)
                .Select(static group => $"{group.Key}:{group.Count()}"));
    }

    private static string FormatLiquidChunkSample(WorldLiquidTileData liquidTileData)
    {
        if (liquidTileData.Chunks.Count == 0)
            return "none";

        return string.Join(", ", liquidTileData.Chunks.Take(4).Select(static chunk =>
            $"({chunk.ChunkX},{chunk.ChunkY}) layers={chunk.Layers.Count} visible={chunk.VisibleTileCount} type={chunk.Layers[0].BasicType}"));
    }

    private static string FormatTerrainChunkSample(WorldTerrainTileData terrainTileData)
    {
        if (terrainTileData.Chunks.Count == 0)
            return "none";

        return string.Join(", ", terrainTileData.Chunks.Take(4).Select(static chunk =>
            $"({chunk.IndexX},{chunk.IndexY}) area={chunk.AreaId} holes={chunk.HasHoles} liquidFlags={chunk.HasLiquidFlags}"));
    }

    private static string FormatTerrainHeightRange(WorldTerrainTileData terrainTileData)
    {
        WorldTerrainHeightmapData? heightmap = terrainTileData.Heightmap;
        if (heightmap is null)
            return "n/a";

        return $"{heightmap.MinHeight:F2}..{heightmap.MaxHeight:F2}";
    }

    private static string FormatTerrainCenter(WorldTerrainTileData terrainTileData)
    {
        WorldTerrainHeightmapData? heightmap = terrainTileData.Heightmap;
        return heightmap is null ? "n/a" : $"{heightmap.CenterHeight:F2}";
    }

    private static string FormatTerrainCorners(WorldTerrainTileData terrainTileData)
    {
        WorldTerrainHeightmapData? heightmap = terrainTileData.Heightmap;
        if (heightmap is null)
            return "n/a";

        return $"nw={heightmap.NorthWestHeight:F2} ne={heightmap.NorthEastHeight:F2} sw={heightmap.SouthWestHeight:F2} se={heightmap.SouthEastHeight:F2}";
    }

    private static string FormatHeightRange(WorldWdlTileData wdlTileData)
    {
        if (!wdlTileData.MinHeight.HasValue || !wdlTileData.MaxHeight.HasValue)
            return "n/a";

        return $"{wdlTileData.MinHeight.Value}..{wdlTileData.MaxHeight.Value}";
    }

    private static string FormatWdlCorners(WorldWdlTileData wdlTileData)
    {
        if (!wdlTileData.HasData)
            return "n/a";

        return $"nw={FormatOptionalHeight(wdlTileData.NorthWestHeight)} ne={FormatOptionalHeight(wdlTileData.NorthEastHeight)} sw={FormatOptionalHeight(wdlTileData.SouthWestHeight)} se={FormatOptionalHeight(wdlTileData.SouthEastHeight)}";
    }

    private static string FormatOptionalHeight(short? value)
    {
        return value.HasValue ? value.Value.ToString() : "n/a";
    }

    private static string FormatOptionalUInt(uint? value)
    {
        return value.HasValue ? value.Value.ToString() : "n/a";
    }

    private void DrawBoundaryContents()
    {
        ImGui.TextWrapped("This app is the first real viewer host in `wow-viewer`. It is intentionally narrow: windowing, app shell, runtime preview, and diagnostics now live here without any `MdxViewer` dependency.");
        ImGui.Separator();
        ImGui.TextDisabled("Ownership");
        ImGui.BulletText($"Core: {ProjectIdentity.SolutionName} {ProjectIdentity.PlannedVersion}");
        ImGui.BulletText($"PM4 canonical owner: {Pm4Boundary.CanonicalOwner}");
        ImGui.BulletText($"PM4 legacy reference: {Pm4Boundary.LegacyReference}");

        ImGui.Separator();
        ImGui.TextDisabled("Runtime Service Areas");
        foreach (RuntimeBoundary boundary in RuntimeBoundaries.All)
            ImGui.BulletText($"{boundary.Name}: {boundary.Responsibility}");

        ImGui.Separator();
        ImGui.TextDisabled("Current Proof Boundary");
        ImGui.BulletText("This app proves a standalone wow-viewer-owned desktop shell.");
        ImGui.BulletText("The active shell now proves bounded standalone preview consumers and a bounded world-session bridge.");
        ImGui.BulletText("The world path is still an investigation/runtime surface, not the final streamed 3D world renderer.");
    }

    private void DrawBoundaryWindow()
    {
        ImGui.SetNextWindowSize(new Vector2(420, 520), ImGuiCond.FirstUseEver);
        if (!ImGui.Begin("Runtime Boundaries", ref _showBoundaryWindow))
        {
            ImGui.End();
            return;
        }

        DrawBoundaryContents();

        ImGui.End();
    }

    private void DrawAboutContents()
    {
        ImGui.TextWrapped("`WowViewer.App` is the active shell for bounded viewer/runtime proof work in `wow-viewer`. The diagnostics panel is the main place for detailed technical state now.");
        ImGui.Separator();
        ImGui.TextDisabled("Commands");
        ImGui.BulletText("No args: open the desktop viewer");
        ImGui.BulletText("viewer [options]: open the desktop viewer with an initial M2 or world-session request");
        ImGui.BulletText("m2-frame [options]: keep the existing CLI proof flow");
        ImGui.BulletText("world-bootstrap [options]: run bounded client-root plus WDT bootstrap proof");
        ImGui.BulletText("world-frame [options]: run bounded tile placement plus runtime visibility/pass proof");
    }

    private void DrawAboutWindow()
    {
        ImGui.SetNextWindowSize(new Vector2(420, 220), ImGuiCond.FirstUseEver);
        if (!ImGui.Begin("About", ref _showAboutWindow))
        {
            ImGui.End();
            return;
        }

        DrawAboutContents();
        ImGui.End();
    }

    private void DrawWorldStatusWindow()
    {
        ImGui.SetNextWindowSize(new Vector2(380, 260), ImGuiCond.FirstUseEver);
        if (!ImGui.Begin("World Status", ref _showWorldStatusWindow))
        {
            ImGui.End();
            return;
        }

        if (_session.WorkspaceMode != WowViewerWorkspaceMode.WorldSession)
        {
            ImGui.TextWrapped("Switch to World Session to inspect current map, tile, runtime counts, and selection state.");
            ImGui.End();
            return;
        }

        WowViewerWorldDiagnosticsSnapshot diagnosticsSnapshot = _worldSceneHost.DiagnosticsSnapshot;
        if (!diagnosticsSnapshot.HasRuntime)
        {
            ImGui.TextWrapped("Open a world session to populate runtime status for the current tile.");
            ImGui.End();
            return;
        }

        WowViewerWorldSceneSnapshot sceneSnapshot = _worldSceneHost.SceneSnapshot;
        WowViewerWorldAssetState assetState = _worldSceneHost.AssetState;
        ImGui.TextDisabled("Current World Frame");
        ImGui.Text($"Map: {sceneSnapshot.RequestedMapInput} -> {sceneSnapshot.ResolvedMapDirectory}");
        ImGui.Text($"Tile: ({sceneSnapshot.SelectedTileX},{sceneSnapshot.SelectedTileY})");
        ImGui.Text($"Placement Source: {sceneSnapshot.PlacementSourcePath}");
        ImGui.Text($"Load Source: {(diagnosticsSnapshot.LoadedFromArchive ? "archive catalog" : "loose file")}");
        ImGui.Separator();
        ImGui.TextDisabled("Runtime Summary");
        ImGui.Text($"WMO Visible/Total: {assetState.VisibleWmoCount}/{assetState.WmoInstanceCount}");
        ImGui.Text($"Doodads Visible/Total: {assetState.VisibleMdxCount}/{assetState.MdxInstanceCount}");
        ImGui.Text($"Referenced Assets: WMO {assetState.ReferencedWmoAssetCount} / Doodads {assetState.ReferencedMdxAssetCount}");
        ImGui.Text($"Pending Assets: {assetState.PendingAssetLoadCount}");
        ImGui.Text($"Object Phase: {diagnosticsSnapshot.ObjectPhaseExecuted}");
        ImGui.Text($"Total Cpu Ms: {diagnosticsSnapshot.TotalCpuMs:F2}");

        if (_selectedWorldObject.HasValue
            && _currentWorldRuntimeFrame is { } currentFrame
            && TryResolveWorldNavigatorEntry(currentFrame, _selectedWorldObject.Value, out WorldNavigatorEntry entry))
        {
            ImGui.Separator();
            ImGui.TextDisabled("Selection Summary");
            ImGui.Text($"Type: {entry.Kind}");
            ImGui.Text($"Model: {entry.Instance.ModelName}");
            ImGui.Text($"Unique Id: {entry.Instance.UniqueId}");
            ImGui.Text($"Visible: {entry.IsVisible}");
        }

        ImGui.End();
    }

    private void DrawWorldMinimapContents()
    {
        string mapName = ResolveWorldMinimapMapName();
        if (string.IsNullOrWhiteSpace(_session.World.ClientRoot) || !Directory.Exists(_session.World.ClientRoot))
        {
            ImGui.TextWrapped("Set a valid client root to render the world minimap.");
            return;
        }

        if (string.IsNullOrWhiteSpace(mapName))
        {
            ImGui.TextWrapped("Choose a map to load minimap tiles for the current world source.");
            return;
        }

        WorldMinimapRenderer? minimapRenderer = EnsureWorldMinimapRenderer();
        if (minimapRenderer == null)
        {
            ImGui.TextWrapped("The world minimap renderer is unavailable until a valid client-root source is attached.");
            return;
        }

        minimapRenderer.ProcessPendingLoads();

        ImGui.TextDisabled($"{mapName}  ready={minimapRenderer.UploadedTileCount} pending={minimapRenderer.PendingTileCount} failed={minimapRenderer.FailedTileCount}");
        if (minimapRenderer.IsBusy)
            ImGui.ProgressBar(minimapRenderer.LoadingProgress, new Vector2(MathF.Min(220f, ImGui.GetContentRegionAvail().X), 0f), $"Minimap {minimapRenderer.LoadingProgress * 100f:F0}%");

        if (ImGui.Button("Center On Loaded Tile"))
            _worldMinimapPanOffset = Vector2.Zero;
        ImGui.SameLine();
        if (ImGui.Button("Reset Zoom"))
            _worldMinimapZoom = 24.0f;

        float availableWidth = ImGui.GetContentRegionAvail().X;
        float availableHeight = MathF.Max(availableWidth, 280f);
        float mapSize = Math.Clamp(MathF.Min(availableWidth, availableHeight), 240f, 640f);
        Vector2 origin = ImGui.GetCursorScreenPos();
        Vector2 extent = new(mapSize, mapSize);
        ImGui.InvisibleButton("##WorldMinimapSurface", extent);
        DrawWorldMinimapSurface(origin, extent, mapName, minimapRenderer);

        float centerTileX = GetWorldMinimapCenterTileX();
        float centerTileY = GetWorldMinimapCenterTileY();
        float viewSpan = Math.Clamp(_worldMinimapZoom, 4f, WorldMinimapTileCount);
        ClampWorldMinimapPanOffset(centerTileX, centerTileY, viewSpan);
        float viewMinTileX = Math.Clamp(centerTileX - (viewSpan * 0.5f) + _worldMinimapPanOffset.X, 0f, WorldMinimapTileCount - viewSpan);
        float viewMinTileY = Math.Clamp(centerTileY - (viewSpan * 0.5f) + _worldMinimapPanOffset.Y, 0f, WorldMinimapTileCount - viewSpan);

        if (ImGui.IsItemHovered())
        {
            HandleWorldMinimapInput(origin, extent, viewMinTileX, viewMinTileY, viewSpan);
            if (TryGetWorldMinimapHoveredTile(ImGui.GetMousePos(), origin, extent, viewMinTileX, viewMinTileY, viewSpan, out int hoveredTileX, out int hoveredTileY))
                ImGui.SetTooltip($"tile ({hoveredTileX},{hoveredTileY})");
        }

        string selectedTileText = _session.World.TileX >= 0 && _session.World.TileY >= 0
            ? $"Selected Tile: ({_session.World.TileX},{_session.World.TileY})"
            : "Selected Tile: auto";
        ImGui.TextDisabled(selectedTileText);
        if (_worldSceneHost.SceneSnapshot.HasSelectedTile)
            ImGui.TextDisabled($"Loaded Tile: ({_worldSceneHost.SceneSnapshot.SelectedTileX},{_worldSceneHost.SceneSnapshot.SelectedTileY})");
    }

    private void DrawWorldNavigatorWindow()
    {
        ImGui.SetNextWindowSize(new Vector2(460, 700), ImGuiCond.FirstUseEver);
        if (!ImGui.Begin("World Navigator", ref _showNavigatorWindow))
        {
            ImGui.End();
            return;
        }

        if (_session.WorkspaceMode != WowViewerWorkspaceMode.WorldSession)
        {
            ImGui.TextWrapped("Switch to World Session to navigate runtime objects for the current tile.");
            ImGui.End();
            return;
        }

        ImGui.TextWrapped("This optional panel mirrors the compact world-session controls. Use it when you want a separate navigator window, otherwise keep Compact World Session Layout enabled.");
        ImGui.Separator();
        DrawWorldNavigatorSourceSection();
        ImGui.Separator();

        if (_currentWorldRuntimeFrame == null)
        {
            ImGui.TextWrapped("Open a world session above to browse WMO and doodad placements admitted through the runtime bridge.");
            ImGui.End();
            return;
        }

        DrawWorldNavigatorEntriesSection(_currentWorldRuntimeFrame);

        ImGui.End();
    }

    private void DrawWorldInspectorWindow()
    {
        ImGui.SetNextWindowSize(new Vector2(420, 700), ImGuiCond.FirstUseEver);
        if (!ImGui.Begin("World Inspector", ref _showInspectorWindow))
        {
            ImGui.End();
            return;
        }

        if (_session.WorkspaceMode != WowViewerWorkspaceMode.WorldSession)
        {
            ImGui.TextWrapped("Switch to World Session to inspect the selected runtime object.");
            ImGui.End();
            return;
        }

        if (_currentWorldRuntimeFrame == null)
        {
            ImGui.TextWrapped("Open a world session to inspect runtime object details.");
            ImGui.End();
            return;
        }

        if (!_selectedWorldObject.HasValue || !TryResolveWorldNavigatorEntry(_currentWorldRuntimeFrame, _selectedWorldObject.Value, out WorldNavigatorEntry entry))
        {
            ImGui.TextWrapped("Select an object from the world canvas or navigator to inspect it.");
            ImGui.End();
            return;
        }

        DrawWorldInspectorContents(entry);

        ImGui.End();
    }

    private void DrawWorldNavigatorEntriesSection(WowViewerWorldRuntimeFrameResult result)
    {
        ImGui.TextDisabled("Filters");
        ImGui.Checkbox("Visible Only", ref _worldNavigatorVisibleOnly);
        ImGui.SameLine();
        ImGui.Checkbox("WMO", ref _worldNavigatorShowWmo);
        ImGui.SameLine();
        ImGui.Checkbox("Doodads (MDX/M2)", ref _worldNavigatorShowMdx);
        ImGui.InputText("Model Filter", ref _worldNavigatorFilter, 256);
        ImGui.Separator();

        List<WorldNavigatorEntry> entries = BuildWorldNavigatorEntries(result);
        ImGui.TextDisabled($"Entries: {entries.Count}");
        if (_selectedWorldObject.HasValue && TryResolveWorldNavigatorEntry(result, _selectedWorldObject.Value, out WorldNavigatorEntry selectedEntry))
        {
            string selectedAssetPath = GetNavigatorAssetPath(selectedEntry);
            ImGui.TextDisabled("Selected Asset Path");
            ImGui.PushTextWrapPos();
            ImGui.TextUnformatted(selectedAssetPath);
            ImGui.PopTextWrapPos();
        }

        ImGui.Separator();

        if (!_worldNavigatorShowWmo && !_worldNavigatorShowMdx)
        {
            ImGui.TextWrapped("Enable at least one object family to populate the navigator.");
            return;
        }

        if (entries.Count == 0)
        {
            ImGui.TextWrapped("No runtime objects match the current navigator filters.");
            return;
        }

        if (ImGui.BeginChild("worldNavigatorList"))
        {
            foreach (WorldNavigatorEntry entry in entries)
            {
                WorldObjectSelection selection = CreateSelection(entry, result.SelectedTileX, result.SelectedTileY);
                bool selected = _selectedWorldObject.HasValue && _selectedWorldObject.Value.Equals(selection);
                ImGui.PushID($"{entry.Kind}:{entry.Instance.PlacementEntryIndex}:{entry.Instance.UniqueId}:{entry.Instance.ModelKey}");
                if (ImGui.Selectable(BuildNavigatorLabel(entry), selected))
                    SelectWorldObject(selection, entry);

                if (ImGui.IsItemHovered())
                    ImGui.SetTooltip(GetNavigatorAssetPath(entry));

                if (selected)
                    ImGui.SetItemDefaultFocus();

                ImGui.PopID();
            }

            ImGui.EndChild();
        }
    }

    private void DrawWorldInspectorContents(WorldNavigatorEntry entry)
    {
        ImGui.TextDisabled("Selection");
        ImGui.Text($"Type: {(entry.Kind == WorldSelectionKind.Wmo ? "WMO" : $"{entry.Instance.AssetKind} Doodad")}");
        ImGui.Text($"Model: {entry.Instance.ModelName}");
        ImGui.Text($"Model Key: {entry.Instance.ModelKey}");
        ImGui.Text($"Asset Kind: {entry.Instance.AssetKind}");
        ImGui.Text($"Unique Id: {entry.Instance.UniqueId}");
        ImGui.Text($"Placement Index: {entry.Instance.PlacementEntryIndex}");
        ImGui.Text($"Tile: ({entry.Instance.TileX},{entry.Instance.TileY})");
        ImGui.Separator();

        ImGui.TextDisabled("Placement");
        ImGui.Text($"Position: {FormatVector3(entry.Instance.PlacementPosition)}");
        ImGui.Text($"Rotation: {FormatVector3(entry.Instance.PlacementRotation)}");
        ImGui.Text($"Scale: {entry.Instance.PlacementScale:F3}");
        ImGui.Separator();

        ImGui.TextDisabled("Bounds");
        ImGui.Text($"Resolved: {entry.Instance.BoundsResolved}");
        ImGui.Text($"World Min: {FormatVector3(entry.Instance.BoundsMin)}");
        ImGui.Text($"World Max: {FormatVector3(entry.Instance.BoundsMax)}");
        ImGui.Text($"Local Min: {FormatVector3(entry.Instance.LocalBoundsMin)}");
        ImGui.Text($"Local Max: {FormatVector3(entry.Instance.LocalBoundsMax)}");
        ImGui.Separator();

        ImGui.TextDisabled("Runtime State");
        ImGui.Text($"Asset Ready: {entry.AssetReady}");
        ImGui.Text($"Visible: {entry.IsVisible}");
        if (entry.CenterDistance.HasValue)
            ImGui.Text($"Distance: {MathF.Sqrt(entry.CenterDistance.Value):F1}");
        if (entry.Kind == WorldSelectionKind.Wmo)
        {
            ImGui.Text($"WMO Version: {FormatOptionalUInt(entry.Instance.WmoVersion)}");
            ImGui.Text($"WMO Groups: {entry.Instance.WmoGroupCount}");
            ImGui.Text($"WMO Portals: {entry.Instance.WmoPortalCount}");
            ImGui.Text($"WMO Doodad Sets: {entry.Instance.WmoDoodadSetCount}");
            ImGui.Text($"Embedded Doodads: MDX {entry.Instance.WmoDoodadMdxCount} / M2 {entry.Instance.WmoDoodadM2Count} / Unknown {entry.Instance.WmoDoodadUnknownCount}");
        }

        if (entry.Kind == WorldSelectionKind.Mdx)
        {
            ImGui.Text($"Taxi Actor: {entry.IsTaxiActor}");
            ImGui.Text($"Animated Model: {entry.WasAnimated}");
            ImGui.Text($"Opaque Route: {entry.HasOpaqueRoute}");
            ImGui.Text($"Transparent Route: {entry.HasTransparentRoute}");
            ImGui.Text($"Unbatched: {entry.RequiresUnbatchedRender}");
        }
    }

    private void LoadActiveWorkspace()
    {
        _lastError = null;

        if (!IsImplementedWorkspace(_session.WorkspaceMode))
        {
            _statusMessage = $"{_session.GetWorkspaceLabel()} is not implemented yet. Switch to Standalone M2, Standalone MDX, or World Session for a live workspace.";
            return;
        }

        switch (_session.WorkspaceMode)
        {
            case WowViewerWorkspaceMode.StandaloneM2:
                LoadPreview();
                break;
            case WowViewerWorkspaceMode.StandaloneWmo:
                LoadWmoPreview();
                break;
            case WowViewerWorkspaceMode.StandaloneMdx:
                LoadMdxPreview();
                break;
            case WowViewerWorkspaceMode.WorldSession:
                LoadWorldSession();
                break;
            case WowViewerWorkspaceMode.DatasetTooling:
                _statusMessage = "Dataset Tooling does not have a preview load action. Use the control panel buttons to launch jobs.";
                break;
            case WowViewerWorkspaceMode.ModelOutputs:
                LoadModelOutputScene();
                break;
        }
    }

    private void LoadPreview()
    {
        InvalidatePendingWorldLoadState();
        _lastError = null;

        try
        {
            M2PreviewLoadRequest request = _session.BuildM2PreviewRequest();
            M2PreviewLoadResult preview = M2PreviewLoader.Load(request);
            UploadPreviewTexture(preview.FrameResult.VisualSnapshot);
            EnsureM2GpuPreviewRenderer()?.LoadPreview(preview);
            _currentPreview = preview;
            _currentMdxPreview = null;
            _selectedWorldObject = null;
            _worldSceneHost.Clear();
            _mdxGpuPreviewRenderer?.ClearPreview();
            DeleteWorldTerrainPreviewTexture();
            _statusMessage = $"Loaded {preview.FrameResult.GoldenFrame.CanonicalModelPath} in {preview.LoadDuration.TotalMilliseconds:F1} ms.";
            bool hasGpuPreview = _gpuPreviewRenderer?.HasRenderableGeometry == true;
            _lastLoadSummary = hasGpuPreview
                ? $"GPU {_session.VisualSize}x{_session.VisualSize}, software {preview.FrameResult.VisualSnapshot.Width}x{preview.FrameResult.VisualSnapshot.Height}, {preview.FrameResult.RenderFrame.CommandCount} draw commands"
                : $"Software {preview.FrameResult.VisualSnapshot.Width}x{preview.FrameResult.VisualSnapshot.Height}, {preview.FrameResult.RenderFrame.CommandCount} draw commands";
            SyncM2CameraTargetsFromSession(resetCurrent: true);
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or InvalidOperationException or ArgumentException or NotSupportedException)
        {
            _lastError = ex.Message;
            _statusMessage = "Preview load failed.";
        }
    }

    private void LoadWmoPreview()
    {
        InvalidatePendingWorldLoadState();
        _lastError = null;

        try
        {
            WmoPreviewLoadRequest request = _session.BuildWmoPreviewRequest();
            WmoPreviewLoadResult preview = WmoPreviewLoader.Load(request);
            EnsureWmoGpuPreviewRenderer()?.LoadPreview(preview);
            _currentWmoPreview = preview;
            _currentPreview = null;
            _currentMdxPreview = null;
            _selectedWorldObject = null;
            _gpuPreviewRenderer?.ClearPreview();
            _worldSceneHost.Clear();
            _mdxGpuPreviewRenderer?.ClearPreview();
            DeletePreviewTexture();
            DeleteWorldTerrainPreviewTexture();
            _statusMessage = $"Loaded {preview.Document.SourcePath} in {preview.LoadDuration.TotalMilliseconds:F1} ms.";
            _lastLoadSummary = $"GPU {_session.VisualSize}x{_session.VisualSize}, materials {preview.Document.Materials.Count}, groups {preview.Document.Groups.Count}, portals {preview.Document.Portals.Count}, doodads {preview.Document.DoodadPlacements.Count}";
            SyncWmoCameraTargetsFromSession(resetCurrent: true);
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or InvalidOperationException or ArgumentException or NotSupportedException)
        {
            _lastError = ex.Message;
            _statusMessage = "WMO preview load failed.";
        }
    }

    private void LoadMdxPreview()
    {
        InvalidatePendingWorldLoadState();
        _lastError = null;

        try
        {
            MdxPreviewLoadRequest request = _session.BuildMdxPreviewRequest();
            MdxPreviewLoadResult preview = MdxPreviewLoader.Load(request);
            EnsureMdxGpuPreviewRenderer()?.LoadPreview(preview);
            _currentMdxPreview = preview;
            _currentPreview = null;
            _selectedWorldObject = null;
            _gpuPreviewRenderer?.ClearPreview();
            _worldSceneHost.Clear();
            DeletePreviewTexture();
            DeleteWorldTerrainPreviewTexture();
            _statusMessage = $"Loaded {preview.Geometry.SourcePath} in {preview.LoadDuration.TotalMilliseconds:F1} ms.";
            _lastLoadSummary = $"GPU {_session.VisualSize}x{_session.VisualSize}, geosets {preview.Geometry.GeosetCount}, materials {preview.Summary.MaterialCount}, particles {preview.EffectRuntimeState.VisibleParticleEmitterCount}/{preview.EffectRuntimeState.Particles.Count}, ribbons {preview.EffectRuntimeState.VisibleRibbonEmitterCount}/{preview.EffectRuntimeState.Ribbons.Count}";
            SyncMdxCameraTargetsFromSession(resetCurrent: true);
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or InvalidOperationException or ArgumentException or NotSupportedException)
        {
            _lastError = ex.Message;
            _statusMessage = "MDX preview load failed.";
        }
    }

    private void LoadWorldSession()
    {
        if (IsWorldLoadPending())
        {
            _statusMessage = $"World session load already running for {_pendingWorldLoadMapInput}.";
            return;
        }

        _lastError = null;
        _selectedWorldObject = null;
        _session.World.ShowTerrain = true;
        _session.World.ShowWdl = false;
        _worldSceneHost.Clear();
        DeleteWorldTerrainPreviewTexture();

        WowViewerWorldRuntimeFrameRequest request = _session.World.BuildRuntimeFrameRequest();
        ViewerIoSourceKey sourceKey = ViewerIoSourceKey.Create(request.ClientRoot, request.BuildLabel, request.LooseOverlayRoot);
        ViewerIoCatalogLease catalogLease = _viewerIoService.GetCatalog(sourceKey);
        int generation = unchecked(++_pendingWorldLoadGeneration);
        _pendingWorldLoadMapInput = request.MapInput;
        _pendingWorldLoadStopwatch = Stopwatch.StartNew();
        _pendingWorldLoadTask = Task.Run(() => new PendingWorldLoadResult(generation, WowViewerWorldRuntimeBridge.Build(request, catalogLease.ArchiveCatalog)));
        _statusMessage = $"Opening world session for {request.MapInput}... Shared world data is being assembled on the CPU; the GPU tile view will populate when the runtime frame finishes.";
        _lastLoadSummary = "World session load queued on a background worker.";
    }

    private void LoadModelOutputScene()
    {
        InvalidatePendingWorldLoadState();
        _lastError = null;

        try
        {
            _session.ModelOutput.Normalize();
            ModelOutputScene scene = ModelOutputSceneLoader.Load(_session.ModelOutput.InputPath, _session.ModelOutput.Variant);
            _currentModelOutputScene = scene;
            RefreshModelOutputGpuScene();
            _currentPreview = null;
            _currentMdxPreview = null;
            _selectedWorldObject = null;
            _gpuPreviewRenderer?.ClearPreview();
            _worldSceneHost.Clear();
            _mdxGpuPreviewRenderer?.ClearPreview();
            DeletePreviewTexture();
            DeleteWorldTerrainPreviewTexture();
            _statusMessage = $"Loaded model-output scene from {scene.SourcePath} in {scene.LoadDuration.TotalMilliseconds:F1} ms.";
            _lastLoadSummary = $"tiles {scene.Tiles.Count}, triangles {scene.TriangleCount}, objects {scene.ObjectCount}, tileSize {scene.TileWorldSize:F3}, variant {_session.ModelOutput.Variant}";
            ResetModelOutputCamera();
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or InvalidOperationException or ArgumentException or NotSupportedException)
        {
            _lastError = ex.Message;
            _statusMessage = "Model-output scene load failed.";
        }
    }

    private void ResetModelOutputCamera()
    {
        _session.ModelOutput.CameraAzimuthDegrees = 45.0f;
        _session.ModelOutput.SetTargetOffset(Vector3.Zero);
        if (_session.ModelOutput.CameraMode == WowViewerModelOutputCameraMode.Orbit)
        {
            _session.ModelOutput.CameraElevationDegrees = 50.0f;
            _session.ModelOutput.CameraZoomFactor = 1.35f;
            SyncModelOutputCameraTargetsFromSession(resetCurrent: true);
            return;
        }

        _session.ModelOutput.CameraElevationDegrees = -18.0f;
        _session.ModelOutput.FlyMoveSpeed = 1.0f;
        if (_currentModelOutputScene != null)
        {
            ModelOutputCameraFrame orbitFrame = ModelOutputGpuRenderer.BuildOrbitCameraFrame(
                _currentModelOutputScene.BoundsMin,
                _currentModelOutputScene.BoundsMax,
                _session.VisualSize,
                _session.VisualSize,
                45.0f,
                35.0f,
                1.15f,
                Vector3.Zero);
            _session.ModelOutput.SetFlyPosition(orbitFrame.Position);
        }
        else
        {
            _session.ModelOutput.SetFlyPosition(new Vector3(0.0f, 256.0f, -256.0f));
        }
    }

    private void HandleModelOutputPreviewMouseInput(Vector2 previewSize)
    {
        if (_currentModelOutputScene == null || !ImGui.IsItemHovered())
            return;

        if (_session.ModelOutput.CameraMode == WowViewerModelOutputCameraMode.Fly)
        {
            HandleModelOutputFlyInput();
            return;
        }

        HandleInteractiveOrbitInput(
            previewSize,
            _currentModelOutputScene.BoundsMin,
            _currentModelOutputScene.BoundsMax,
            ref _modelOutputInteractiveCamera,
            ResetModelOutputCamera,
            () => _session.ModelOutput.CameraAzimuthDegrees,
            value => _session.ModelOutput.CameraAzimuthDegrees = value,
            () => _session.ModelOutput.CameraElevationDegrees,
            value => _session.ModelOutput.CameraElevationDegrees = value,
            () => _session.ModelOutput.CameraZoomFactor,
            value => _session.ModelOutput.CameraZoomFactor = value,
            () => _session.ModelOutput.GetTargetOffset(),
            value => _session.ModelOutput.SetTargetOffset(value));
    }

    private void ResetM2Camera()
    {
        _session.M2CameraAzimuthDegrees = 45.0f;
        _session.M2CameraElevationDegrees = 20.0f;
        _session.M2CameraZoomFactor = 1.15f;
        _session.SetM2CameraTargetOffset(Vector3.Zero);
        SyncM2CameraTargetsFromSession(resetCurrent: true);
    }

    private void ResetWmoOrbitCamera()
    {
        _session.WmoCameraAzimuthDegrees = 35.0f;
        _session.WmoCameraElevationDegrees = 25.0f;
        _session.WmoCameraZoomFactor = 0.9f;
        _session.SetWmoCameraTargetOffset(Vector3.Zero);
        SyncWmoCameraTargetsFromSession(resetCurrent: true);
    }

    private void ResetMdxOrbitCamera()
    {
        _session.MdxCameraMode = PreviewCameraMode.Orbit;
        _session.MdxCameraPreset = string.Empty;
        _session.MdxCameraAzimuthDegrees = 35.0f;
        _session.MdxCameraElevationDegrees = 25.0f;
        _session.MdxCameraZoomFactor = 0.72f;
        _session.SetMdxCameraTargetOffset(Vector3.Zero);
        SyncMdxCameraTargetsFromSession(resetCurrent: true);
    }

    private void HandleM2PreviewMouseInput(Vector2 previewSize)
    {
        if (_currentPreview == null || _gpuPreviewRenderer == null || !ImGui.IsItemHovered())
            return;

        HandleInteractiveOrbitInput(
            previewSize,
            _gpuPreviewRenderer.BoundsMin,
            _gpuPreviewRenderer.BoundsMax,
            ref _m2InteractiveCamera,
            ResetM2Camera,
            () => _session.M2CameraAzimuthDegrees,
            value => _session.M2CameraAzimuthDegrees = value,
            () => _session.M2CameraElevationDegrees,
            value => _session.M2CameraElevationDegrees = value,
            () => _session.M2CameraZoomFactor,
            value => _session.M2CameraZoomFactor = value,
            () => _session.GetM2CameraTargetOffset(),
            value => _session.SetM2CameraTargetOffset(value));
    }

    private void HandleWmoPreviewMouseInput(Vector2 previewSize)
    {
        if (_currentWmoPreview == null || _wmoGpuPreviewRenderer == null || !ImGui.IsItemHovered())
            return;

        HandleInteractiveOrbitInput(
            previewSize,
            _wmoGpuPreviewRenderer.BoundsMin,
            _wmoGpuPreviewRenderer.BoundsMax,
            ref _wmoInteractiveCamera,
            ResetWmoOrbitCamera,
            () => _session.WmoCameraAzimuthDegrees,
            value => _session.WmoCameraAzimuthDegrees = value,
            () => _session.WmoCameraElevationDegrees,
            value => _session.WmoCameraElevationDegrees = value,
            () => _session.WmoCameraZoomFactor,
            value => _session.WmoCameraZoomFactor = value,
            () => _session.GetWmoCameraTargetOffset(),
            value => _session.SetWmoCameraTargetOffset(value));
    }

    private void HandleMdxPreviewMouseInput(Vector2 previewSize)
    {
        if (_currentMdxPreview == null || _mdxGpuPreviewRenderer == null || !ImGui.IsItemHovered())
            return;

        ImGuiIOPtr io = ImGui.GetIO();
        bool wantsInteraction = ImGui.IsMouseDragging(ImGuiMouseButton.Left)
            || ImGui.IsMouseDragging(ImGuiMouseButton.Right)
            || ImGui.IsMouseDragging(ImGuiMouseButton.Middle)
            || MathF.Abs(io.MouseWheel) > float.Epsilon;
        if (_session.MdxCameraMode != PreviewCameraMode.Orbit)
        {
            if (!wantsInteraction)
                return;

            _session.MdxCameraMode = PreviewCameraMode.Orbit;
            _session.MdxCameraPreset = string.Empty;
            SyncMdxCameraTargetsFromSession(resetCurrent: true);
        }

        HandleInteractiveOrbitInput(
            previewSize,
            _mdxGpuPreviewRenderer.BoundsMin,
            _mdxGpuPreviewRenderer.BoundsMax,
            ref _mdxInteractiveCamera,
            ResetMdxOrbitCamera,
            () => _session.MdxCameraAzimuthDegrees,
            value => _session.MdxCameraAzimuthDegrees = value,
            () => _session.MdxCameraElevationDegrees,
            value => _session.MdxCameraElevationDegrees = value,
            () => _session.MdxCameraZoomFactor,
            value => _session.MdxCameraZoomFactor = value,
            () => _session.GetMdxCameraTargetOffset(),
            value => _session.SetMdxCameraTargetOffset(value));
    }

    private bool HandleWorldPreviewInput(Vector2 previewSize)
    {
        if (_currentWorldRuntimeFrame == null || _worldGpuPreviewRenderer == null)
            return false;

        ImGuiIOPtr io = ImGui.GetIO();
        bool hovered = ImGui.IsItemHovered();
        if (hovered && (ImGui.IsMouseClicked(ImGuiMouseButton.Left) || ImGui.IsMouseClicked(ImGuiMouseButton.Right)))
            _worldPreviewInputCaptured = true;
        else if (!hovered && (ImGui.IsMouseClicked(ImGuiMouseButton.Left) || ImGui.IsMouseClicked(ImGuiMouseButton.Right) || ImGui.IsMouseClicked(ImGuiMouseButton.Middle)))
            _worldPreviewInputCaptured = false;

        if (hovered || _worldPreviewInputCaptured)
            ImGui.SetMouseCursor(ImGuiMouseCursor.Hand);

        bool acceptsKeyboard = hovered || _worldPreviewInputCaptured;
        bool acceptsMouseMotion = hovered || ImGui.IsMouseDragging(ImGuiMouseButton.Right);
        bool acceptsWheel = hovered;

        if (hovered && ImGui.IsMouseDoubleClicked(ImGuiMouseButton.Left))
        {
            _worldViewCamera.Reset();
            return false;
        }

        Vector2 mouseDelta = io.MouseDelta;
        if (acceptsMouseMotion && ImGui.IsMouseDragging(ImGuiMouseButton.Right) && mouseDelta.LengthSquared() > 0.0f)
            _worldViewCamera.RotateLook(mouseDelta.X * 0.5f, -mouseDelta.Y * 0.5f);

        float deltaSeconds = Math.Clamp(io.DeltaTime, 1.0f / 240.0f, 0.05f);
        float step = _session.World.CameraMoveSpeed * deltaSeconds;
        if (IsWorldCameraKeyDown(ImGuiKey.LeftShift, Key.ShiftLeft) || IsWorldCameraKeyDown(ImGuiKey.RightShift, Key.ShiftRight))
            step *= 4.0f;

        float forwardStep = 0.0f;
        float strafeStep = 0.0f;
        float verticalStep = 0.0f;
        if (acceptsKeyboard && IsWorldCameraKeyDown(ImGuiKey.W, Key.W))
            forwardStep += step;
        if (acceptsKeyboard && IsWorldCameraKeyDown(ImGuiKey.S, Key.S))
            forwardStep -= step;
        if (acceptsKeyboard && IsWorldCameraKeyDown(ImGuiKey.D, Key.D))
            strafeStep += step;
        if (acceptsKeyboard && IsWorldCameraKeyDown(ImGuiKey.A, Key.A))
            strafeStep -= step;
        if (acceptsKeyboard && IsWorldCameraKeyDown(ImGuiKey.Q, Key.Q))
            verticalStep += step;
        if (acceptsKeyboard && IsWorldCameraKeyDown(ImGuiKey.E, Key.E))
            verticalStep -= step;

        if (forwardStep != 0.0f || strafeStep != 0.0f || verticalStep != 0.0f)
            _worldViewCamera.Translate(forwardStep, strafeStep, verticalStep);

        bool consumedWheel = acceptsWheel && MathF.Abs(io.MouseWheel) > float.Epsilon;
        if (consumedWheel)
            _worldViewCamera.Translate(io.MouseWheel * _session.World.CameraMoveSpeed * 0.35f, 0.0f, 0.0f);

        return consumedWheel;
    }

    private bool IsWorldCameraKeyDown(ImGuiKey imguiKey, Key silkKey)
    {
        if (ImGui.IsKeyDown(imguiKey))
            return true;

        if (_input == null)
            return false;

        foreach (IKeyboard keyboard in _input.Keyboards)
        {
            if (keyboard.IsKeyPressed(silkKey))
                return true;
        }

        return false;
    }

    private void AdvanceInteractivePreviewCameras(float deltaSeconds)
    {
        SyncM2CameraTargetsFromSession(resetCurrent: false);
        SyncWmoCameraTargetsFromSession(resetCurrent: false);
        SyncMdxCameraTargetsFromSession(resetCurrent: false);
        if (_session.ModelOutput.CameraMode == WowViewerModelOutputCameraMode.Orbit)
            SyncModelOutputCameraTargetsFromSession(resetCurrent: false);

        _m2InteractiveCamera.Advance(deltaSeconds);
        _wmoInteractiveCamera.Advance(deltaSeconds);
        _mdxInteractiveCamera.Advance(deltaSeconds);
        if (_session.ModelOutput.CameraMode == WowViewerModelOutputCameraMode.Orbit)
            _modelOutputInteractiveCamera.Advance(deltaSeconds);
    }

    private void SyncM2CameraTargetsFromSession(bool resetCurrent)
    {
        _m2InteractiveCamera.SetTargets(_session.M2CameraAzimuthDegrees, _session.M2CameraElevationDegrees, _session.M2CameraZoomFactor, _session.GetM2CameraTargetOffset(), resetCurrent);
    }

    private void SyncWmoCameraTargetsFromSession(bool resetCurrent)
    {
        _wmoInteractiveCamera.SetTargets(_session.WmoCameraAzimuthDegrees, _session.WmoCameraElevationDegrees, _session.WmoCameraZoomFactor, _session.GetWmoCameraTargetOffset(), resetCurrent);
    }

    private void SyncMdxCameraTargetsFromSession(bool resetCurrent)
    {
        _mdxInteractiveCamera.SetTargets(_session.MdxCameraAzimuthDegrees, _session.MdxCameraElevationDegrees, _session.MdxCameraZoomFactor, _session.GetMdxCameraTargetOffset(), resetCurrent);
    }

    private void SyncModelOutputCameraTargetsFromSession(bool resetCurrent)
    {
        _modelOutputInteractiveCamera.SetTargets(_session.ModelOutput.CameraAzimuthDegrees, _session.ModelOutput.CameraElevationDegrees, _session.ModelOutput.CameraZoomFactor, _session.ModelOutput.GetTargetOffset(), resetCurrent);
    }

    private void RefreshModelOutputGpuScene()
    {
        if (_currentModelOutputScene == null)
            return;

        EnsureModelOutputGpuRenderer()?.LoadScene(
            _currentModelOutputScene,
            _session.ModelOutput.ShowObjects,
            _session.ModelOutput.ShowM2Objects,
            _session.ModelOutput.ShowWmoObjects);
    }

    private ModelOutputCameraFrame BuildModelOutputCameraFrame()
    {
        if (_currentModelOutputScene == null)
            return default;

        if (_session.ModelOutput.CameraMode == WowViewerModelOutputCameraMode.Fly)
        {
            return ModelOutputGpuRenderer.BuildFlyCameraFrame(
                _currentModelOutputScene.BoundsMin,
                _currentModelOutputScene.BoundsMax,
                _session.VisualSize,
                _session.VisualSize,
                _session.ModelOutput.GetFlyPosition(),
                _session.ModelOutput.CameraAzimuthDegrees,
                _session.ModelOutput.CameraElevationDegrees);
        }

        return ModelOutputGpuRenderer.BuildOrbitCameraFrame(
            _currentModelOutputScene.BoundsMin,
            _currentModelOutputScene.BoundsMax,
            _session.VisualSize,
            _session.VisualSize,
            _modelOutputInteractiveCamera.CurrentAzimuthDegrees,
            _modelOutputInteractiveCamera.CurrentElevationDegrees,
            _modelOutputInteractiveCamera.CurrentZoomFactor,
            _modelOutputInteractiveCamera.CurrentTargetOffset);
    }

    private void HandleModelOutputFlyInput()
    {
        if (_currentModelOutputScene == null)
            return;

        ImGuiIOPtr io = ImGui.GetIO();
        if (ImGui.IsMouseDoubleClicked(ImGuiMouseButton.Left))
        {
            ResetModelOutputCamera();
            return;
        }

        Vector2 mouseDelta = io.MouseDelta;
        float azimuth = _session.ModelOutput.CameraAzimuthDegrees;
        float elevation = _session.ModelOutput.CameraElevationDegrees;
        if (ImGui.IsMouseDragging(ImGuiMouseButton.Right) && mouseDelta.LengthSquared() > 0.0f)
        {
            azimuth -= mouseDelta.X * 0.25f;
            elevation = Math.Clamp(elevation - (mouseDelta.Y * 0.18f), -85.0f, 85.0f);
        }

        float speed = _session.ModelOutput.FlyMoveSpeed;
        if (MathF.Abs(io.MouseWheel) > float.Epsilon)
            speed = Math.Clamp(speed * MathF.Pow(1.12f, io.MouseWheel), 0.1f, 8.0f);

        Vector3 position = _session.ModelOutput.GetFlyPosition();
        Vector3 sceneExtents = _currentModelOutputScene.BoundsMax - _currentModelOutputScene.BoundsMin;
        float sceneScale = MathF.Max(sceneExtents.Length(), 128.0f);
        float step = sceneScale * 0.0035f * speed;
        if (ImGui.IsKeyDown(ImGuiKey.LeftShift) || ImGui.IsKeyDown(ImGuiKey.RightShift))
            step *= 3.0f;

        Vector3 forward = ComputeFlyForward(azimuth, elevation);
        Vector3 right = Vector3.Normalize(Vector3.Cross(forward, Vector3.UnitY));
        if (right.LengthSquared() < 1e-6f)
            right = Vector3.UnitX;

        if (ImGui.IsKeyDown(ImGuiKey.W))
            position += forward * step;
        if (ImGui.IsKeyDown(ImGuiKey.S))
            position -= forward * step;
        if (ImGui.IsKeyDown(ImGuiKey.A))
            position -= right * step;
        if (ImGui.IsKeyDown(ImGuiKey.D))
            position += right * step;
        if (ImGui.IsKeyDown(ImGuiKey.E))
            position += Vector3.UnitY * step;
        if (ImGui.IsKeyDown(ImGuiKey.Q))
            position -= Vector3.UnitY * step;

        _session.ModelOutput.CameraAzimuthDegrees = azimuth;
        _session.ModelOutput.CameraElevationDegrees = elevation;
        _session.ModelOutput.FlyMoveSpeed = speed;
        _session.ModelOutput.SetFlyPosition(position);
    }

    private static Vector3 ComputeFlyForward(float azimuthDegrees, float elevationDegrees)
    {
        float azimuth = azimuthDegrees * MathF.PI / 180.0f;
        float elevation = elevationDegrees * MathF.PI / 180.0f;
        return Vector3.Normalize(new Vector3(
            MathF.Cos(elevation) * MathF.Cos(azimuth),
            MathF.Sin(elevation),
            MathF.Cos(elevation) * MathF.Sin(azimuth)));
    }

    private void HandleInteractiveOrbitInput(
        Vector2 previewSize,
        Vector3 boundsMin,
        Vector3 boundsMax,
        ref InteractiveOrbitCameraState cameraState,
        Action resetAction,
        Func<float> getAzimuth,
        Action<float> setAzimuth,
        Func<float> getElevation,
        Action<float> setElevation,
        Func<float> getZoom,
        Action<float> setZoom,
        Func<Vector3> getTargetOffset,
        Action<Vector3> setTargetOffset)
    {
        ImGuiIOPtr io = ImGui.GetIO();
        if (ImGui.IsMouseDoubleClicked(ImGuiMouseButton.Left))
        {
            resetAction();
            return;
        }

        Vector2 mouseDelta = io.MouseDelta;
        float azimuth = getAzimuth();
        float elevation = getElevation();
        float zoom = getZoom();
        Vector3 targetOffset = getTargetOffset();

        if (ImGui.IsMouseDragging(ImGuiMouseButton.Left) && mouseDelta.LengthSquared() > 0.0f)
        {
            azimuth -= mouseDelta.X * 0.35f;
            elevation = Math.Clamp(elevation + (mouseDelta.Y * 0.25f), -89.0f, 89.0f);
        }

        if ((ImGui.IsMouseDragging(ImGuiMouseButton.Right) || ImGui.IsMouseDragging(ImGuiMouseButton.Middle)) && mouseDelta.LengthSquared() > 0.0f)
        {
            float sceneExtent = MathF.Max((boundsMax - boundsMin).Length(), 8.0f);
            float panScale = (sceneExtent / MathF.Max(previewSize.X, 1.0f)) * MathF.Max(zoom, 0.05f);
            targetOffset += ComputeCameraPlanePanDelta(azimuth, elevation, panScale, mouseDelta);
        }

        if (MathF.Abs(io.MouseWheel) > float.Epsilon)
            zoom = Math.Clamp(zoom * MathF.Pow(0.9f, io.MouseWheel), 0.05f, 10.0f);

        setAzimuth(azimuth);
        setElevation(elevation);
        setZoom(zoom);
        setTargetOffset(targetOffset);
        cameraState.SetTargets(azimuth, elevation, zoom, targetOffset, resetCurrent: false);
    }

    private static Vector3 ComputeCameraPlanePanDelta(float azimuthDegrees, float elevationDegrees, float panScale, Vector2 mouseDelta)
    {
        float elevationRadians = elevationDegrees * MathF.PI / 180.0f;
        float azimuthRadians = azimuthDegrees * MathF.PI / 180.0f;
        float cosElevation = MathF.Cos(elevationRadians);
        Vector3 forward = Vector3.Normalize(new Vector3(
            cosElevation * MathF.Cos(azimuthRadians),
            cosElevation * MathF.Sin(azimuthRadians),
            -MathF.Sin(elevationRadians)));
        Vector3 worldUp = MathF.Abs(Vector3.Dot(forward, Vector3.UnitZ)) > 0.99f ? Vector3.UnitY : Vector3.UnitZ;
        Vector3 right = Vector3.Normalize(Vector3.Cross(forward, worldUp));
        Vector3 cameraUp = Vector3.Normalize(Vector3.Cross(right, forward));
        return (-mouseDelta.X * panScale * right) + (mouseDelta.Y * panScale * cameraUp);
    }

    private static int GetMdxCameraPresetIndex(string? preset)
    {
        for (int index = 0; index < MdxCameraPresetValues.Length; index++)
        {
            if (string.Equals(MdxCameraPresetValues[index], preset, StringComparison.OrdinalIgnoreCase))
                return index;
        }

        return 0;
    }

    private void ClearWorkspace()
    {
        InvalidatePendingWorldLoadState();
        _currentPreview = null;
        _currentWmoPreview = null;
        _currentMdxPreview = null;
        _currentModelOutputScene = null;
        _selectedWorldObject = null;
        _lastError = null;
        _lastLoadSummary = "No workspace loaded.";
        _statusMessage = "Workspace cleared.";
        _gpuPreviewRenderer?.ClearPreview();
        _wmoGpuPreviewRenderer?.ClearPreview();
        _mdxGpuPreviewRenderer?.ClearPreview();
        _modelOutputGpuRenderer?.ClearScene();
        _worldSceneHost.Clear();
        DeletePreviewTexture();
        DeleteWorldTerrainPreviewTexture();
    }

    private void ApplyLoadedWorldSession(WowViewerWorldRuntimeFrameResult runtimeFrame)
    {
        _currentPreview = null;
        _currentMdxPreview = null;
        _currentWmoPreview = null;
        _currentModelOutputScene = null;
        _selectedWorldObject = SelectDefaultWorldObject(runtimeFrame);
        _gpuPreviewRenderer?.ClearPreview();
        _wmoGpuPreviewRenderer?.ClearPreview();
        _mdxGpuPreviewRenderer?.ClearPreview();
        _modelOutputGpuRenderer?.ClearScene();
        ViewerIoSourceKey sourceKey = ViewerIoSourceKey.Create(_session.World.ClientRoot, _session.World.BuildLabel, _session.World.LooseOverlayRoot);
        _worldSceneHost.ApplyRuntimeFrame(_gl, _viewerIoService, sourceKey, BuildWorldMinimapSourceSignature(), runtimeFrame, _session.World.IgnoreTerrainHoles, _session.World.ShowHoleOverlay);
        WowViewerWorldSceneSnapshot sceneSnapshot = _worldSceneHost.SceneSnapshot;
        WowViewerWorldAssetState assetState = _worldSceneHost.AssetState;
        DeletePreviewTexture();
        UploadWorldTerrainPreviewTexture(runtimeFrame.TerrainVisualSnapshot);
        _statusMessage = $"Opened selected tile runtime frame for {sceneSnapshot.ResolvedMapDirectory} tile ({sceneSnapshot.SelectedTileX},{sceneSnapshot.SelectedTileY}) in {runtimeFrame.Stats.TotalCpuMs:F1} ms.";
        _lastLoadSummary = $"GPU {_session.VisualSize}x{_session.VisualSize}, WMO {assetState.VisibleWmoCount}/{assetState.WmoInstanceCount}, doodads {assetState.VisibleMdxCount}/{assetState.MdxInstanceCount}, terrain {sceneSnapshot.TerrainVisualWidth}x{sceneSnapshot.TerrainVisualHeight}, pending {assetState.PendingAssetLoadCount}";
    }

    private unsafe void UploadPreviewTexture(M2SoftwareVisualSnapshot snapshot)
    {
        if (_gl == null)
            return;

        DeletePreviewTexture();
        _previewTextureHandle = _gl.GenTexture();
        _gl.BindTexture(TextureTarget.Texture2D, _previewTextureHandle);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)GLEnum.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)GLEnum.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)GLEnum.ClampToEdge);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)GLEnum.ClampToEdge);

        byte[] rgbaPixels = new byte[snapshot.Width * snapshot.Height * 4];
        for (int index = 0, target = 0; index < snapshot.RgbPixels.Length; index += 3, target += 4)
        {
            rgbaPixels[target + 0] = snapshot.RgbPixels[index + 0];
            rgbaPixels[target + 1] = snapshot.RgbPixels[index + 1];
            rgbaPixels[target + 2] = snapshot.RgbPixels[index + 2];
            rgbaPixels[target + 3] = 255;
        }

        fixed (byte* pixels = rgbaPixels)
        {
            _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba8, (uint)snapshot.Width, (uint)snapshot.Height, 0, PixelFormat.Rgba, PixelType.UnsignedByte, pixels);
        }

        _gl.BindTexture(TextureTarget.Texture2D, 0);
    }

    private void DeletePreviewTexture()
    {
        if (_gl == null || _previewTextureHandle == 0)
            return;

        _gl.DeleteTexture(_previewTextureHandle);
        _previewTextureHandle = 0;
    }

    private unsafe void UploadWorldTerrainPreviewTexture(WorldTerrainVisualSnapshot snapshot)
    {
        if (_gl == null)
            return;

        DeleteWorldTerrainPreviewTexture();
        _worldTerrainPreviewTextureHandle = _gl.GenTexture();
        _gl.BindTexture(TextureTarget.Texture2D, _worldTerrainPreviewTextureHandle);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)GLEnum.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)GLEnum.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)GLEnum.ClampToEdge);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)GLEnum.ClampToEdge);

        fixed (byte* pixels = snapshot.RgbaPixels)
        {
            _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba8, (uint)snapshot.Width, (uint)snapshot.Height, 0, PixelFormat.Rgba, PixelType.UnsignedByte, pixels);
        }

        _gl.BindTexture(TextureTarget.Texture2D, 0);
    }

    private void DeleteWorldTerrainPreviewTexture()
    {
        if (_gl == null || _worldTerrainPreviewTextureHandle == 0)
            return;

        _gl.DeleteTexture(_worldTerrainPreviewTextureHandle);
        _worldTerrainPreviewTextureHandle = 0;
    }

    private void ApplySession(WowViewerSession session)
    {
        ArgumentNullException.ThrowIfNull(session);
        _session.WorkspaceMode = session.WorkspaceMode;
        _session.Source.Kind = session.Source.Kind;
        _session.Source.ArchiveRoot = session.Source.ArchiveRoot ?? string.Empty;
        _session.Source.VirtualPath = session.Source.VirtualPath ?? string.Empty;
        _session.Source.InputPath = session.Source.InputPath ?? string.Empty;
        _session.Source.BuildLabel = session.Source.BuildLabel ?? string.Empty;
        _session.Source.LooseOverlayRoot = session.Source.LooseOverlayRoot ?? string.Empty;
        _session.World.ClientRoot = session.World.ClientRoot ?? string.Empty;
        _session.World.MapInput = session.World.MapInput ?? string.Empty;
        _session.World.BuildLabel = session.World.BuildLabel ?? string.Empty;
        _session.World.LooseOverlayRoot = session.World.LooseOverlayRoot ?? string.Empty;
        _session.World.TileX = session.World.TileX;
        _session.World.TileY = session.World.TileY;
        _session.ModelOutput.InputPath = session.ModelOutput.InputPath ?? string.Empty;
        _session.ModelOutput.Variant = session.ModelOutput.Variant;
        _session.ModelOutput.CameraMode = session.ModelOutput.CameraMode;
        _session.ModelOutput.ShowObjects = session.ModelOutput.ShowObjects;
        _session.ModelOutput.ShowM2Objects = session.ModelOutput.ShowM2Objects;
        _session.ModelOutput.ShowWmoObjects = session.ModelOutput.ShowWmoObjects;
        _session.ModelOutput.CameraAzimuthDegrees = session.ModelOutput.CameraAzimuthDegrees;
        _session.ModelOutput.CameraElevationDegrees = session.ModelOutput.CameraElevationDegrees;
        _session.ModelOutput.CameraZoomFactor = session.ModelOutput.CameraZoomFactor;
        _session.ModelOutput.SetTargetOffset(session.ModelOutput.GetTargetOffset());
        _session.ModelOutput.SetFlyPosition(session.ModelOutput.GetFlyPosition());
        _session.ModelOutput.FlyMoveSpeed = session.ModelOutput.FlyMoveSpeed;
        _session.M2CameraAzimuthDegrees = session.M2CameraAzimuthDegrees;
        _session.M2CameraElevationDegrees = session.M2CameraElevationDegrees;
        _session.M2CameraZoomFactor = session.M2CameraZoomFactor;
        _session.SetM2CameraTargetOffset(session.GetM2CameraTargetOffset());
        _session.MdxCameraMode = session.MdxCameraMode;
        _session.MdxCameraPreset = session.MdxCameraPreset;
        _session.MdxCameraAzimuthDegrees = session.MdxCameraAzimuthDegrees;
        _session.MdxCameraElevationDegrees = session.MdxCameraElevationDegrees;
        _session.MdxCameraFieldOfViewDegrees = session.MdxCameraFieldOfViewDegrees;
        _session.MdxCameraZoomFactor = session.MdxCameraZoomFactor;
        _session.SetMdxCameraTargetOffset(session.GetMdxCameraTargetOffset());
        _session.WmoCameraAzimuthDegrees = session.WmoCameraAzimuthDegrees;
        _session.WmoCameraElevationDegrees = session.WmoCameraElevationDegrees;
        _session.WmoCameraFieldOfViewDegrees = session.WmoCameraFieldOfViewDegrees;
        _session.WmoCameraZoomFactor = session.WmoCameraZoomFactor;
        _session.SetWmoCameraTargetOffset(session.GetWmoCameraTargetOffset());
        _session.ProfileIndex = session.ProfileIndex;
        _session.SequenceIndex = session.SequenceIndex;
        _session.TimeMs = session.TimeMs;
        _session.VisualSize = session.VisualSize;
        _session.Normalize();
        SyncM2CameraTargetsFromSession(resetCurrent: true);
        SyncWmoCameraTargetsFromSession(resetCurrent: true);
        SyncMdxCameraTargetsFromSession(resetCurrent: true);
        SyncModelOutputCameraTargetsFromSession(resetCurrent: true);
    }

    private sealed class InteractiveOrbitCameraState
    {
        private const float SmoothingRate = 12.0f;

        public bool IsInitialized { get; private set; }

        public float CurrentAzimuthDegrees { get; private set; }

        public float CurrentElevationDegrees { get; private set; }

        public float CurrentZoomFactor { get; private set; }

        public Vector3 CurrentTargetOffset { get; private set; }

        public float TargetAzimuthDegrees { get; private set; }

        public float TargetElevationDegrees { get; private set; }

        public float TargetZoomFactor { get; private set; }

        public Vector3 TargetTargetOffset { get; private set; }

        public void SetTargets(float azimuthDegrees, float elevationDegrees, float zoomFactor, Vector3 targetOffset, bool resetCurrent)
        {
            TargetAzimuthDegrees = azimuthDegrees;
            TargetElevationDegrees = elevationDegrees;
            TargetZoomFactor = zoomFactor;
            TargetTargetOffset = targetOffset;
            if (!IsInitialized || resetCurrent)
            {
                CurrentAzimuthDegrees = azimuthDegrees;
                CurrentElevationDegrees = elevationDegrees;
                CurrentZoomFactor = zoomFactor;
                CurrentTargetOffset = targetOffset;
                IsInitialized = true;
            }
        }

        public void Advance(float deltaSeconds)
        {
            if (!IsInitialized)
                return;

            float blend = deltaSeconds <= 0.0f ? 1.0f : 1.0f - MathF.Exp(-SmoothingRate * MathF.Min(deltaSeconds, 0.1f));
            CurrentAzimuthDegrees = LerpAngleDegrees(CurrentAzimuthDegrees, TargetAzimuthDegrees, blend);
            CurrentElevationDegrees = CurrentElevationDegrees + ((TargetElevationDegrees - CurrentElevationDegrees) * blend);
            CurrentZoomFactor = CurrentZoomFactor + ((TargetZoomFactor - CurrentZoomFactor) * blend);
            CurrentTargetOffset = Vector3.Lerp(CurrentTargetOffset, TargetTargetOffset, blend);
        }

        private static float LerpAngleDegrees(float current, float target, float blend)
        {
            float delta = ((target - current + 540.0f) % 360.0f) - 180.0f;
            return current + (delta * blend);
        }
    }

    private void ApplySettingsToState(WowViewerAppSettings settings)
    {
        ApplySession(settings.Session ?? WowViewerSession.CreateDefault());
        _showAboutWindow = settings.ShowAboutWindow;
        _showWorkspaceWindow = settings.ShowWorkspaceWindow;
        _showControlWindow = settings.ShowControlWindow;
        _showDiagnosticsWindow = settings.ShowDiagnosticsWindow;
        _showBoundaryWindow = settings.ShowBoundaryWindow;
        _showWorldStatusWindow = settings.ShowWorldStatusWindow;
        _showNavigatorWindow = settings.ShowNavigatorWindow;
        _showInspectorWindow = settings.ShowInspectorWindow;
        _showWorldMinimapWindow = settings.ShowWorldMinimapWindow;
        _compactWorldSessionLayout = settings.CompactWorldSessionLayout;

        if (_session.WorkspaceMode == WowViewerWorkspaceMode.WorldSession)
            ApplyLegacyWorldSessionWindowPreset();
    }

    private void SaveSettings()
    {
        _session.Normalize();
        _settings.Session = _session;
        _settings.ShowAboutWindow = _showAboutWindow;
        _settings.ShowWorkspaceWindow = _showWorkspaceWindow;
        _settings.ShowControlWindow = _showControlWindow;
        _settings.ShowDiagnosticsWindow = _showDiagnosticsWindow;
        _settings.ShowBoundaryWindow = _showBoundaryWindow;
        _settings.ShowWorldStatusWindow = _showWorldStatusWindow;
        _settings.ShowNavigatorWindow = _showNavigatorWindow;
        _settings.ShowInspectorWindow = _showInspectorWindow;
        _settings.ShowWorldMinimapWindow = _showWorldMinimapWindow;
        _settings.CompactWorldSessionLayout = _compactWorldSessionLayout;
        WowViewerAppSettingsStore.Save(_settings);
    }

    private string BuildWorldMinimapSourceSignature()
    {
        string clientRoot = _session.World.ClientRoot?.Trim() ?? string.Empty;
        string looseOverlayRoot = _session.World.LooseOverlayRoot?.Trim() ?? string.Empty;
        string buildLabel = _session.World.BuildLabel?.Trim() ?? string.Empty;
        return string.Join('|',
            string.IsNullOrWhiteSpace(clientRoot) ? string.Empty : Path.GetFullPath(clientRoot),
            string.IsNullOrWhiteSpace(looseOverlayRoot) ? string.Empty : Path.GetFullPath(looseOverlayRoot),
            buildLabel);
    }

    private string ResolveWorldMinimapMapName()
    {
        if (!string.IsNullOrWhiteSpace(_currentWorldSession?.ResolvedMapDirectory))
            return _currentWorldSession.ResolvedMapDirectory;

        return _session.World.MapInput?.Trim() ?? string.Empty;
    }

    private IReadOnlyList<WdtTileCoordinate> GetWorldMinimapOccupiedTiles()
    {
        if (_currentWorldSession is not null)
            return _worldSceneHost.SceneSnapshot.OccupiedTiles;

        if (_worldSpawnPickerState?.Session is not null)
            return _worldSpawnPickerState.Session.OccupiedTiles;

        return Array.Empty<WdtTileCoordinate>();
    }

    private float GetWorldMinimapCenterTileX()
    {
        if (_worldSceneHost.SceneSnapshot.HasSelectedTile)
            return _worldSceneHost.SceneSnapshot.SelectedTileX + 0.5f;
        if (_session.World.TileX >= 0)
            return _session.World.TileX + 0.5f;

        IReadOnlyList<WdtTileCoordinate> occupiedTiles = GetWorldMinimapOccupiedTiles();
        return occupiedTiles.Count > 0
            ? occupiedTiles.Average(static tile => tile.TileX + 0.5f)
            : 32f;
    }

    private float GetWorldMinimapCenterTileY()
    {
        if (_worldSceneHost.SceneSnapshot.HasSelectedTile)
            return _worldSceneHost.SceneSnapshot.SelectedTileY + 0.5f;
        if (_session.World.TileY >= 0)
            return _session.World.TileY + 0.5f;

        IReadOnlyList<WdtTileCoordinate> occupiedTiles = GetWorldMinimapOccupiedTiles();
        return occupiedTiles.Count > 0
            ? occupiedTiles.Average(static tile => tile.TileY + 0.5f)
            : 32f;
    }

    private void ClampWorldMinimapPanOffset(float centerTileX, float centerTileY, float viewSpan)
    {
        float baseViewMinTileX = centerTileX - (viewSpan * 0.5f);
        float baseViewMinTileY = centerTileY - (viewSpan * 0.5f);
        float maxViewMin = MathF.Max(0f, WorldMinimapTileCount - viewSpan);
        _worldMinimapPanOffset = new Vector2(
            Math.Clamp(_worldMinimapPanOffset.X, -baseViewMinTileX, maxViewMin - baseViewMinTileX),
            Math.Clamp(_worldMinimapPanOffset.Y, -baseViewMinTileY, maxViewMin - baseViewMinTileY));
    }

    private void HandleWorldMinimapInput(Vector2 origin, Vector2 extent, float viewMinTileX, float viewMinTileY, float viewSpan)
    {
        ImGuiIOPtr io = ImGui.GetIO();
        float cellSize = extent.X / MathF.Max(viewSpan, 1f);

        if (MathF.Abs(io.MouseWheel) > float.Epsilon)
        {
            _worldMinimapZoom = Math.Clamp(_worldMinimapZoom * MathF.Pow(0.88f, io.MouseWheel), 4f, WorldMinimapTileCount);
            return;
        }

        if (ImGui.IsMouseClicked(ImGuiMouseButton.Right))
        {
            _worldMinimapDragging = true;
            _worldMinimapDragStart = ImGui.GetMousePos();
        }

        if (_worldMinimapDragging && ImGui.IsMouseDown(ImGuiMouseButton.Right))
        {
            Vector2 mouseDelta = ImGui.GetMousePos() - _worldMinimapDragStart;
            if (mouseDelta.LengthSquared() > 0.01f)
            {
                _worldMinimapPanOffset -= new Vector2(mouseDelta.Y / cellSize, mouseDelta.X / cellSize);
                _worldMinimapDragStart = ImGui.GetMousePos();
            }
        }
        else if (_worldMinimapDragging)
        {
            _worldMinimapDragging = false;
        }

        if (!ImGui.IsMouseClicked(ImGuiMouseButton.Left)
            && !ImGui.IsMouseDoubleClicked(ImGuiMouseButton.Left))
        {
            return;
        }

        if (!TryGetWorldMinimapHoveredTile(ImGui.GetMousePos(), origin, extent, viewMinTileX, viewMinTileY, viewSpan, out int tileX, out int tileY))
            return;

        _session.World.TileX = tileX;
        _session.World.TileY = tileY;
        _statusMessage = $"Selected world minimap tile ({tileX},{tileY}).";

        if (ImGui.IsMouseDoubleClicked(ImGuiMouseButton.Left) && !IsWorldLoadPending())
            LoadActiveWorkspace();
    }

    private static bool TryGetWorldMinimapHoveredTile(
        Vector2 mousePosition,
        Vector2 origin,
        Vector2 extent,
        float viewMinTileX,
        float viewMinTileY,
        float viewSpan,
        out int tileX,
        out int tileY)
    {
        tileX = -1;
        tileY = -1;

        if (mousePosition.X < origin.X || mousePosition.Y < origin.Y || mousePosition.X > origin.X + extent.X || mousePosition.Y > origin.Y + extent.Y)
            return false;

        float relativeTileY = (mousePosition.X - origin.X) / extent.X * viewSpan + viewMinTileY;
        float relativeTileX = (mousePosition.Y - origin.Y) / extent.Y * viewSpan + viewMinTileX;
        if (relativeTileX < 0f || relativeTileY < 0f || relativeTileX >= WorldMinimapTileCount || relativeTileY >= WorldMinimapTileCount)
            return false;

        tileX = Math.Clamp((int)MathF.Floor(relativeTileX), 0, 63);
        tileY = Math.Clamp((int)MathF.Floor(relativeTileY), 0, 63);
        return true;
    }

    private void DrawWorldMinimapSurface(Vector2 origin, Vector2 extent, string mapName, WorldMinimapRenderer minimapRenderer)
    {
        ImDrawListPtr drawList = ImGui.GetWindowDrawList();
        float centerTileX = GetWorldMinimapCenterTileX();
        float centerTileY = GetWorldMinimapCenterTileY();
        float viewSpan = Math.Clamp(_worldMinimapZoom, 4f, WorldMinimapTileCount);
        ClampWorldMinimapPanOffset(centerTileX, centerTileY, viewSpan);
        float viewMinTileX = Math.Clamp(centerTileX - (viewSpan * 0.5f) + _worldMinimapPanOffset.X, 0f, WorldMinimapTileCount - viewSpan);
        float viewMinTileY = Math.Clamp(centerTileY - (viewSpan * 0.5f) + _worldMinimapPanOffset.Y, 0f, WorldMinimapTileCount - viewSpan);
        float cellSize = extent.X / viewSpan;

        uint background = ImGui.ColorConvertFloat4ToU32(new Vector4(0.05f, 0.06f, 0.08f, 1.0f));
        uint missingTile = ImGui.ColorConvertFloat4ToU32(new Vector4(0.12f, 0.13f, 0.15f, 1.0f));
        uint occupiedBorder = ImGui.ColorConvertFloat4ToU32(new Vector4(0.56f, 0.60f, 0.66f, 0.9f));
        uint gridLine = ImGui.ColorConvertFloat4ToU32(new Vector4(0.18f, 0.19f, 0.22f, 0.45f));
        uint loadedColor = ImGui.ColorConvertFloat4ToU32(new Vector4(1.0f, 0.95f, 0.35f, 1.0f));
        uint selectedColor = ImGui.ColorConvertFloat4ToU32(new Vector4(1.0f, 0.35f, 0.22f, 1.0f));

        drawList.AddRectFilled(origin, origin + extent, background, 4f);

        HashSet<int> occupiedTileIndices = GetWorldMinimapOccupiedTiles()
            .Select(static tile => (tile.TileX * 64) + tile.TileY)
            .ToHashSet();
        int minTileX = Math.Max(0, (int)MathF.Floor(viewMinTileX));
        int maxTileX = Math.Min(63, (int)MathF.Ceiling(viewMinTileX + viewSpan));
        int minTileY = Math.Max(0, (int)MathF.Floor(viewMinTileY));
        int maxTileY = Math.Min(63, (int)MathF.Ceiling(viewMinTileY + viewSpan));

        for (int tileX = minTileX; tileX <= maxTileX; tileX++)
        {
            for (int tileY = minTileY; tileY <= maxTileY; tileY++)
            {
                float imageMinX = origin.X + ((tileY - viewMinTileY) * cellSize);
                float imageMinY = origin.Y + ((tileX - viewMinTileX) * cellSize);
                Vector2 tileMin = new(imageMinX, imageMinY);
                Vector2 tileMax = tileMin + new Vector2(cellSize, cellSize);
                bool occupied = occupiedTileIndices.Contains((tileX * 64) + tileY);

                uint textureHandle = minimapRenderer.GetTileTexture(mapName, tileX, tileY);
                if (textureHandle != 0)
                    drawList.AddImage((nint)textureHandle, tileMin, tileMax, new Vector2(0, 0), new Vector2(1, 1));
                else
                    drawList.AddRectFilled(tileMin, tileMax, missingTile);

                drawList.AddRect(tileMin, tileMax, gridLine, 0f, ImDrawFlags.None, 1f);
                if (occupied)
                    drawList.AddRect(tileMin, tileMax, occupiedBorder, 0f, ImDrawFlags.None, 1.3f);
            }
        }

        if (_currentWorldRuntimeFrame != null)
        {
            Vector2 loadedMin = origin + new Vector2((_currentWorldRuntimeFrame.SelectedTileY - viewMinTileY) * cellSize, (_currentWorldRuntimeFrame.SelectedTileX - viewMinTileX) * cellSize);
            Vector2 loadedMax = loadedMin + new Vector2(cellSize, cellSize);
            drawList.AddRect(loadedMin, loadedMax, loadedColor, 0f, ImDrawFlags.None, 2.2f);
        }

        if (_session.World.TileX >= 0 && _session.World.TileY >= 0)
        {
            Vector2 selectedMin = origin + new Vector2((_session.World.TileY - viewMinTileY) * cellSize, (_session.World.TileX - viewMinTileX) * cellSize);
            Vector2 selectedMax = selectedMin + new Vector2(cellSize, cellSize);
            drawList.AddRect(selectedMin, selectedMax, selectedColor, 0f, ImDrawFlags.None, 2.4f);
        }
    }

    private void SyncImGuiWindowMetrics(Vector2D<int> windowSize, Vector2D<int> framebufferSize)
    {
        if (_imGui == null)
            return;

        if (_lastSyncedImGuiWindowSize == windowSize && _lastSyncedImGuiFramebufferSize == framebufferSize)
            return;

        _lastSyncedImGuiWindowSize = windowSize;
        _lastSyncedImGuiFramebufferSize = framebufferSize;
        ImGuiControllerWindowResizedMethod?.Invoke(_imGui, [windowSize]);
    }

    private static string FormatTileSample(IReadOnlyList<WdtTileCoordinate> tiles, int limit)
    {
        if (tiles.Count == 0)
            return "none";

        string sample = string.Join(", ", tiles.Take(limit).Select(static tile => $"({tile.TileX},{tile.TileY})"));
        return tiles.Count > limit ? $"{sample}, ... ({tiles.Count - limit} more)" : sample;
    }

    private static string FormatWdtMainFlags(WdtMainFlagsSummary summary)
    {
        return summary.DistinctNonZeroValues.Count == 0
            ? "none"
            : string.Join(",", summary.DistinctNonZeroValues.Select(static value => $"0x{value.Value:x}:{value.TileCount}"));
    }

    private void DrawWorldRuntimeCanvas(Vector2 origin, Vector2 size, WowViewerWorldRuntimeFrameResult result)
    {
        ImDrawListPtr drawList = ImGui.GetWindowDrawList();
        uint background = ImGui.ColorConvertFloat4ToU32(new Vector4(0.08f, 0.09f, 0.11f, 1.0f));
        uint border = ImGui.ColorConvertFloat4ToU32(new Vector4(0.32f, 0.35f, 0.40f, 1.0f));
        uint wmoColor = ImGui.ColorConvertFloat4ToU32(new Vector4(0.88f, 0.58f, 0.24f, 0.35f));
        uint wmoVisibleColor = ImGui.ColorConvertFloat4ToU32(new Vector4(0.98f, 0.76f, 0.36f, 1.0f));
        uint mdxColor = ImGui.ColorConvertFloat4ToU32(new Vector4(0.30f, 0.66f, 0.94f, 0.28f));
        uint mdxVisibleColor = ImGui.ColorConvertFloat4ToU32(new Vector4(0.40f, 0.82f, 1.0f, 1.0f));

        drawList.AddRectFilled(origin, origin + size, background, 6f);
        drawList.AddRect(origin, origin + size, border, 6f, ImDrawFlags.None, 1.5f);

        foreach (WorldObjectInstance instance in result.WmoInstances)
            drawList.AddCircleFilled(MapWorldPositionToCanvas(instance.PlacementPosition, origin, size, result), 2.5f, wmoColor);

        foreach (WorldObjectInstance instance in result.MdxInstances)
            drawList.AddCircleFilled(MapWorldPositionToCanvas(instance.PlacementPosition, origin, size, result), 2.0f, mdxColor);

        foreach (var visible in result.Visibility.VisibleWmos)
            drawList.AddCircleFilled(MapWorldPositionToCanvas(visible.Instance.PlacementPosition, origin, size, result), 3.5f, wmoVisibleColor);

        foreach (var visible in result.Visibility.VisibleMdx)
            drawList.AddCircleFilled(MapWorldPositionToCanvas(visible.Instance.PlacementPosition, origin, size, result), 3.0f, mdxVisibleColor);

        if (_selectedWorldObject.HasValue && TryResolveWorldNavigatorEntry(result, _selectedWorldObject.Value, out WorldNavigatorEntry selectedEntry))
        {
            Vector2 center = MapWorldPositionToCanvas(selectedEntry.Instance.PlacementPosition, origin, size, result);
            uint selectedColor = selectedEntry.Kind == WorldSelectionKind.Wmo
                ? ImGui.ColorConvertFloat4ToU32(new Vector4(1.0f, 0.96f, 0.56f, 1.0f))
                : ImGui.ColorConvertFloat4ToU32(new Vector4(0.72f, 0.96f, 1.0f, 1.0f));
            drawList.AddCircle(center, selectedEntry.Kind == WorldSelectionKind.Wmo ? 7f : 6f, selectedColor, 0, 2f);
            drawList.AddCircle(center, selectedEntry.Kind == WorldSelectionKind.Wmo ? 10f : 9f, selectedColor, 0, 1f);
        }

        drawList.AddText(origin + new Vector2(8f, 8f), border, $"tile ({result.SelectedTileX},{result.SelectedTileY})");
    }

    private static Vector2 MapWorldPositionToCanvas(Vector3 position, Vector2 origin, Vector2 size, WowViewerWorldRuntimeFrameResult result)
    {
        Vector2 planarMin = result.PlanarMin;
        Vector2 planarMax = result.PlanarMax;
        float width = MathF.Max(1f, planarMax.X - planarMin.X);
        float height = MathF.Max(1f, planarMax.Y - planarMin.Y);
        float nx = 1f - ((position.Y - planarMin.Y) / height);
        float ny = 1f - ((position.X - planarMin.X) / width);
        return new Vector2(origin.X + (nx * size.X), origin.Y + (ny * size.Y));
    }

    private void TrySelectWorldObjectAtCanvasPoint(WowViewerWorldRuntimeFrameResult result, Vector2 origin, Vector2 size, Vector2 mousePosition)
    {
        WorldNavigatorEntry? nearestVisible = null;
        float nearestVisibleDistanceSq = float.MaxValue;
        WorldNavigatorEntry? nearestAny = null;
        float nearestAnyDistanceSq = float.MaxValue;
        const float pickRadius = 14f;
        float pickRadiusSq = pickRadius * pickRadius;

        foreach (WorldNavigatorEntry entry in EnumerateWorldNavigatorEntries(result))
        {
            Vector2 center = MapWorldPositionToCanvas(entry.Instance.PlacementPosition, origin, size, result);
            float distanceSq = Vector2.DistanceSquared(center, mousePosition);
            if (distanceSq > pickRadiusSq)
                continue;

            if (distanceSq < nearestAnyDistanceSq)
            {
                nearestAny = entry;
                nearestAnyDistanceSq = distanceSq;
            }

            if (entry.IsVisible && distanceSq < nearestVisibleDistanceSq)
            {
                nearestVisible = entry;
                nearestVisibleDistanceSq = distanceSq;
            }
        }

        if (nearestVisible.HasValue)
        {
            WorldNavigatorEntry pickedVisible = nearestVisible.Value;
            SelectWorldObject(CreateSelection(pickedVisible, result.SelectedTileX, result.SelectedTileY), pickedVisible);
            return;
        }

        if (nearestAny.HasValue)
        {
            WorldNavigatorEntry picked = nearestAny.Value;
            SelectWorldObject(CreateSelection(picked, result.SelectedTileX, result.SelectedTileY), picked);
        }
    }

    private void SelectWorldObject(WorldObjectSelection selection, WorldNavigatorEntry entry)
    {
        _selectedWorldObject = selection;
        _statusMessage = $"Selected {entry.Kind} {entry.Instance.ModelName} #{entry.Instance.UniqueId} on tile ({selection.TileX},{selection.TileY}).";
    }

    private static WorldObjectSelection? SelectDefaultWorldObject(WowViewerWorldRuntimeFrameResult result)
    {
        if (result.Visibility.VisibleWmos.Count > 0)
        {
            WorldVisibleWmoEntry visibleWmo = result.Visibility.VisibleWmos.OrderBy(static entry => entry.CenterDistanceSq).First();
            WorldNavigatorEntry entry = CreateWorldNavigatorEntry(result, WorldSelectionKind.Wmo, visibleWmo.Instance, visibleWmo.CenterDistanceSq, isVisible: true, isTaxiActor: false);
            return CreateSelection(entry, result.SelectedTileX, result.SelectedTileY);
        }

        if (result.Visibility.VisibleMdx.Count > 0)
        {
            WorldVisibleMdxEntry visibleMdx = result.Visibility.VisibleMdx.OrderBy(static entry => entry.CenterDistanceSq).First();
            WorldNavigatorEntry entry = CreateWorldNavigatorEntry(result, WorldSelectionKind.Mdx, visibleMdx.Instance, visibleMdx.CenterDistanceSq, isVisible: true, visibleMdx.IsTaxiActor);
            return CreateSelection(entry, result.SelectedTileX, result.SelectedTileY);
        }

        if (result.WmoInstances.Count > 0)
        {
            WorldObjectInstance firstWmo = result.WmoInstances[0];
            WorldNavigatorEntry entry = CreateWorldNavigatorEntry(result, WorldSelectionKind.Wmo, firstWmo, centerDistanceSq: null, isVisible: false, isTaxiActor: false);
            return CreateSelection(entry, result.SelectedTileX, result.SelectedTileY);
        }

        if (result.MdxInstances.Count > 0)
        {
            WorldObjectInstance firstMdx = result.MdxInstances[0];
            WorldNavigatorEntry entry = CreateWorldNavigatorEntry(result, WorldSelectionKind.Mdx, firstMdx, centerDistanceSq: null, isVisible: false, isTaxiActor: false);
            return CreateSelection(entry, result.SelectedTileX, result.SelectedTileY);
        }

        return null;
    }

    private List<WorldNavigatorEntry> BuildWorldNavigatorEntries(WowViewerWorldRuntimeFrameResult result)
    {
        string filter = _worldNavigatorFilter.Trim();
        List<WorldNavigatorEntry> entries = new();
        foreach (WorldNavigatorEntry entry in EnumerateWorldNavigatorEntries(result))
        {
            if (entry.Kind == WorldSelectionKind.Wmo && !_worldNavigatorShowWmo)
                continue;

            if (entry.Kind == WorldSelectionKind.Mdx && !_worldNavigatorShowMdx)
                continue;

            if (_worldNavigatorVisibleOnly && !entry.IsVisible)
                continue;

            if (!string.IsNullOrWhiteSpace(filter)
                && entry.Instance.ModelName.IndexOf(filter, StringComparison.OrdinalIgnoreCase) < 0
                && entry.Instance.ModelKey.IndexOf(filter, StringComparison.OrdinalIgnoreCase) < 0)
                continue;

            entries.Add(entry);
        }

        entries.Sort(static (left, right) =>
        {
            int visibleComparison = right.IsVisible.CompareTo(left.IsVisible);
            if (visibleComparison != 0)
                return visibleComparison;

            int kindComparison = left.Kind.CompareTo(right.Kind);
            if (kindComparison != 0)
                return kindComparison;

            float leftDistance = left.CenterDistance ?? float.MaxValue;
            float rightDistance = right.CenterDistance ?? float.MaxValue;
            int distanceComparison = leftDistance.CompareTo(rightDistance);
            if (distanceComparison != 0)
                return distanceComparison;

            return string.Compare(left.Instance.ModelName, right.Instance.ModelName, StringComparison.OrdinalIgnoreCase);
        });

        return entries;
    }

    private IEnumerable<WorldNavigatorEntry> EnumerateWorldNavigatorEntries(WowViewerWorldRuntimeFrameResult result)
    {
        Dictionary<int, WorldVisibleWmoEntry> visibleWmoByIndex = result.Visibility.VisibleWmos.ToDictionary(static entry => entry.Instance.PlacementEntryIndex);
        Dictionary<int, WorldVisibleMdxEntry> visibleMdxByIndex = result.Visibility.VisibleMdx.ToDictionary(static entry => entry.Instance.PlacementEntryIndex);
        HashSet<int> opaqueRoutes = result.PassFrame.OpaqueVisibleMdxRoutes.Select(static route => route.VisibleMdxIndex).ToHashSet();
        HashSet<int> transparentRoutes = result.PassFrame.TransparentVisibleMdxRoutes.Select(static route => route.VisibleMdxIndex).ToHashSet();
        HashSet<int> unbatchedRoutes = result.PassFrame.UnbatchedVisibleMdxIndices;
        HashSet<string> animatedModels = result.PassFrame.UpdatedMdxModelKeys;

        foreach (WorldObjectInstance instance in result.WmoInstances)
        {
            if (visibleWmoByIndex.TryGetValue(instance.PlacementEntryIndex, out WorldVisibleWmoEntry visibleWmo))
                yield return CreateWorldNavigatorEntry(result, WorldSelectionKind.Wmo, visibleWmo.Instance, visibleWmo.CenterDistanceSq, isVisible: true, isTaxiActor: false);
            else
                yield return CreateWorldNavigatorEntry(result, WorldSelectionKind.Wmo, instance, centerDistanceSq: null, isVisible: false, isTaxiActor: false);
        }

        for (int index = 0; index < result.MdxInstances.Count; index++)
        {
            WorldObjectInstance instance = result.MdxInstances[index];
            if (visibleMdxByIndex.TryGetValue(instance.PlacementEntryIndex, out WorldVisibleMdxEntry visibleMdx))
            {
                yield return CreateWorldNavigatorEntry(
                    result,
                    WorldSelectionKind.Mdx,
                    visibleMdx.Instance,
                    visibleMdx.CenterDistanceSq,
                    isVisible: true,
                    visibleMdx.IsTaxiActor,
                    opaqueRoutes.Contains(index),
                    transparentRoutes.Contains(index),
                    unbatchedRoutes.Contains(index),
                    animatedModels.Contains(visibleMdx.Instance.ModelKey));
            }
            else
            {
                yield return CreateWorldNavigatorEntry(
                    result,
                    WorldSelectionKind.Mdx,
                    instance,
                    centerDistanceSq: null,
                    isVisible: false,
                    isTaxiActor: false,
                    hasOpaqueRoute: false,
                    hasTransparentRoute: false,
                    requiresUnbatchedRender: false,
                    wasAnimated: animatedModels.Contains(instance.ModelKey));
            }
        }
    }

    private static WorldNavigatorEntry CreateWorldNavigatorEntry(
        WowViewerWorldRuntimeFrameResult result,
        WorldSelectionKind kind,
        WorldObjectInstance instance,
        float? centerDistanceSq,
        bool isVisible,
        bool isTaxiActor,
        bool hasOpaqueRoute = false,
        bool hasTransparentRoute = false,
        bool requiresUnbatchedRender = false,
        bool wasAnimated = false)
    {
        bool assetReady = kind == WorldSelectionKind.Wmo
            ? result.WmoInstances.Any(candidate => candidate.PlacementEntryIndex == instance.PlacementEntryIndex && candidate.BoundsResolved)
            : result.MdxInstances.Any(candidate => candidate.PlacementEntryIndex == instance.PlacementEntryIndex && candidate.BoundsResolved);

        return new WorldNavigatorEntry(
            kind,
            instance,
            isVisible,
            assetReady,
            centerDistanceSq,
            isTaxiActor,
            hasOpaqueRoute,
            hasTransparentRoute,
            requiresUnbatchedRender,
            wasAnimated);
    }

    private static WorldObjectSelection CreateSelection(WorldNavigatorEntry entry, int tileX, int tileY)
    {
        return new WorldObjectSelection(entry.Kind, tileX, tileY, entry.Instance.PlacementEntryIndex, entry.Instance.UniqueId, entry.Instance.ModelKey);
    }

    private static string BuildNavigatorLabel(WorldNavigatorEntry entry)
    {
        string visibility = entry.IsVisible ? "visible" : "hidden";
        string ready = entry.AssetReady ? "ready" : "pending";
        string distance = entry.CenterDistance.HasValue ? $" d={MathF.Sqrt(entry.CenterDistance.Value):F1}" : string.Empty;
        return $"[{entry.Kind}] {entry.Instance.ModelName} #{entry.Instance.UniqueId} {visibility} {ready}{distance}";
    }

    private static string GetNavigatorAssetPath(WorldNavigatorEntry entry)
    {
        return string.IsNullOrWhiteSpace(entry.Instance.ModelPath)
            ? entry.Instance.ModelKey
            : entry.Instance.ModelPath;
    }

    private static bool TryResolveWorldNavigatorEntry(WowViewerWorldRuntimeFrameResult result, WorldObjectSelection selection, out WorldNavigatorEntry entry)
    {
        if (selection.TileX != result.SelectedTileX || selection.TileY != result.SelectedTileY)
        {
            entry = default;
            return false;
        }

        foreach (WorldNavigatorEntry candidate in EnumerateWorldNavigatorEntriesStatic(result))
        {
            if (candidate.Kind == selection.Kind
                && candidate.Instance.PlacementEntryIndex == selection.PlacementEntryIndex
                && candidate.Instance.UniqueId == selection.UniqueId
                && string.Equals(candidate.Instance.ModelKey, selection.ModelKey, StringComparison.OrdinalIgnoreCase))
            {
                entry = candidate;
                return true;
            }
        }

        entry = default;
        return false;
    }

    private static IEnumerable<WorldNavigatorEntry> EnumerateWorldNavigatorEntriesStatic(WowViewerWorldRuntimeFrameResult result)
    {
        Dictionary<int, WorldVisibleWmoEntry> visibleWmoByIndex = result.Visibility.VisibleWmos.ToDictionary(static entry => entry.Instance.PlacementEntryIndex);
        Dictionary<int, WorldVisibleMdxEntry> visibleMdxByIndex = result.Visibility.VisibleMdx.ToDictionary(static entry => entry.Instance.PlacementEntryIndex);
        HashSet<int> opaqueRoutes = result.PassFrame.OpaqueVisibleMdxRoutes.Select(static route => route.VisibleMdxIndex).ToHashSet();
        HashSet<int> transparentRoutes = result.PassFrame.TransparentVisibleMdxRoutes.Select(static route => route.VisibleMdxIndex).ToHashSet();
        HashSet<int> unbatchedRoutes = result.PassFrame.UnbatchedVisibleMdxIndices;
        HashSet<string> animatedModels = result.PassFrame.UpdatedMdxModelKeys;

        foreach (WorldObjectInstance instance in result.WmoInstances)
        {
            if (visibleWmoByIndex.TryGetValue(instance.PlacementEntryIndex, out WorldVisibleWmoEntry visibleWmo))
                yield return CreateWorldNavigatorEntry(result, WorldSelectionKind.Wmo, visibleWmo.Instance, visibleWmo.CenterDistanceSq, isVisible: true, isTaxiActor: false);
            else
                yield return CreateWorldNavigatorEntry(result, WorldSelectionKind.Wmo, instance, centerDistanceSq: null, isVisible: false, isTaxiActor: false);
        }

        for (int index = 0; index < result.MdxInstances.Count; index++)
        {
            WorldObjectInstance instance = result.MdxInstances[index];
            if (visibleMdxByIndex.TryGetValue(instance.PlacementEntryIndex, out WorldVisibleMdxEntry visibleMdx))
            {
                yield return CreateWorldNavigatorEntry(
                    result,
                    WorldSelectionKind.Mdx,
                    visibleMdx.Instance,
                    visibleMdx.CenterDistanceSq,
                    isVisible: true,
                    visibleMdx.IsTaxiActor,
                    opaqueRoutes.Contains(index),
                    transparentRoutes.Contains(index),
                    unbatchedRoutes.Contains(index),
                    animatedModels.Contains(visibleMdx.Instance.ModelKey));
            }
            else
            {
                yield return CreateWorldNavigatorEntry(
                    result,
                    WorldSelectionKind.Mdx,
                    instance,
                    centerDistanceSq: null,
                    isVisible: false,
                    isTaxiActor: false,
                    hasOpaqueRoute: false,
                    hasTransparentRoute: false,
                    requiresUnbatchedRender: false,
                    wasAnimated: animatedModels.Contains(instance.ModelKey));
            }
        }
    }

    private static string FormatVector3(Vector3 value)
    {
        return $"({value.X:F1}, {value.Y:F1}, {value.Z:F1})";
    }

    private string FormatWorldTileLabel()
    {
        if (_worldSceneHost.SceneSnapshot.HasSelectedTile)
            return $"({_worldSceneHost.SceneSnapshot.SelectedTileX},{_worldSceneHost.SceneSnapshot.SelectedTileY})";

        return _session.World.TileX >= 0 && _session.World.TileY >= 0
            ? $"({_session.World.TileX},{_session.World.TileY})"
            : "auto";
    }
}
