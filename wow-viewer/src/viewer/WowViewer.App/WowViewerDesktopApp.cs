using System.Numerics;
using System.Reflection;
using System.Diagnostics;
using ImGuiNET;
using Silk.NET.Input;
using Silk.NET.Maths;
using Silk.NET.OpenGL;
using Silk.NET.OpenGL.Extensions.ImGui;
using Silk.NET.Windowing;
using WowViewer.Core;
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

    private IWindow? _window;
    private GL? _gl;
    private IInputContext? _input;
    private ImGuiController? _imGui;
    private Vector2D<int> _lastSyncedImGuiWindowSize;
    private Vector2D<int> _lastSyncedImGuiFramebufferSize;
    private bool _disposed;
    private bool _requestInitialLoad;
    private string _statusMessage = "Configure an archive-backed or local asset source, then load a preview.";
    private string _lastLoadSummary = "No workspace loaded.";
    private string? _lastError;
    private M2PreviewLoadResult? _currentPreview;
    private WmoPreviewLoadResult? _currentWmoPreview;
    private MdxPreviewLoadResult? _currentMdxPreview;
    private ModelOutputScene? _currentModelOutputScene;
    private WowViewerWorldSessionBootstrapResult? _currentWorldSession;
    private WowViewerWorldRuntimeFrameResult? _currentWorldRuntimeFrame;
    private uint _previewTextureHandle;
    private uint _worldTerrainPreviewTextureHandle;
    private M2GpuPreviewRenderer? _gpuPreviewRenderer;
    private WmoGpuPreviewRenderer? _wmoGpuPreviewRenderer;
    private MdxGpuPreviewRenderer? _mdxGpuPreviewRenderer;
    private ModelOutputGpuRenderer? _modelOutputGpuRenderer;
    private bool _showAboutWindow = true;
    private bool _showWorkspaceWindow = true;
    private bool _showControlWindow = true;
    private bool _showDiagnosticsWindow = true;
    private bool _showBoundaryWindow = true;
    private bool _showWorldStatusWindow = true;
    private bool _showNavigatorWindow = true;
    private bool _showInspectorWindow = true;
    private bool _showFileBrowserWindow = true;
    private MdxFileBrowserState? _mdxFileBrowserState;
    private bool _wantOpenGameFolder;
    private bool _wantAttachLooseFolder;
    private string? _pendingKnownGoodClientPath;
    private string? _pendingKnownGoodClientBuildLabel;
    private bool _pendingKnownGoodClientAttachLooseFolder;
    private bool _worldNavigatorVisibleOnly = true;
    private bool _worldNavigatorShowWmo = true;
    private bool _worldNavigatorShowMdx = true;
    private string _worldNavigatorFilter = string.Empty;
    private WorldObjectSelection? _selectedWorldObject;
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

    public WowViewerDesktopApp(WowViewerSession? initialSession = null)
    {
        _settings = WowViewerAppSettingsStore.Load();
        _session = _settings.Session ?? WowViewerSession.CreateDefault();
        _session.Normalize();
        _initialSession = initialSession;
        ApplySettingsToState(_settings);
        if (_initialSession != null)
            ApplySession(_initialSession);
    }

    public void Run()
    {
        WindowOptions options = WindowOptions.Default;
        options.Title = WindowTitle;
        options.Size = new Vector2D<int>(1600, 960);
        options.VSync = false;
        options.API = new GraphicsAPI(ContextAPI.OpenGL, ContextProfile.Core, ContextFlags.ForwardCompatible, new APIVersion(3, 3));

        _window = Window.Create(options);
        _window.Load += OnLoad;
        _window.Update += OnUpdate;
        _window.Render += OnRender;
        _window.Resize += OnWindowResize;
        _window.FramebufferResize += OnFramebufferResize;
        _window.Run();
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _disposed = true;
        SaveSettings();
        _gpuPreviewRenderer?.Dispose();
        _wmoGpuPreviewRenderer?.Dispose();
        _mdxGpuPreviewRenderer?.Dispose();
        _modelOutputGpuRenderer?.Dispose();
        DeletePreviewTexture();
        DeleteWorldTerrainPreviewTexture();
        _imGui?.Dispose();
        _input?.Dispose();
        if (_window != null)
        {
            _window.Load -= OnLoad;
            _window.Update -= OnUpdate;
            _window.Render -= OnRender;
            _window.Resize -= OnWindowResize;
            _window.FramebufferResize -= OnFramebufferResize;
        }
        _window?.Dispose();
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
        _gpuPreviewRenderer = new M2GpuPreviewRenderer(_gl);
        _wmoGpuPreviewRenderer = new WmoGpuPreviewRenderer(_gl);
        _mdxGpuPreviewRenderer = new MdxGpuPreviewRenderer(_gl);
        _modelOutputGpuRenderer = new ModelOutputGpuRenderer(_gl);

        _requestInitialLoad = _initialSession?.HasBootstrapInput() == true;
    }

    private void OnUpdate(double deltaSeconds)
    {
        _imGui?.Update((float)deltaSeconds);

        if (_requestInitialLoad)
        {
            _requestInitialLoad = false;
            LoadActiveWorkspace();
        }

        HandleOpenGameFolderDialog();
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
        ImGui.DockSpaceOverViewport();

        if (_showWorkspaceWindow)
            DrawWorkspaceWindow();
        if (_showControlWindow)
            DrawControlWindow();
        if (_showFileBrowserWindow && _session.WorkspaceMode == WowViewerWorkspaceMode.StandaloneMdx)
            DrawMdxFileBrowserWindow();
        DrawPreviewWindow();
        if (_showDiagnosticsWindow)
            DrawDiagnosticsWindow(deltaSeconds);
        if (_showWorldStatusWindow)
            DrawWorldStatusWindow();
        if (_showNavigatorWindow)
            DrawWorldNavigatorWindow();
        if (_showInspectorWindow)
            DrawWorldInspectorWindow();
        if (_showBoundaryWindow)
            DrawBoundaryWindow();
        if (_showAboutWindow)
            DrawAboutWindow();

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

            if (ImGui.BeginMenu("Open Saved Game Folder", _settings.KnownGoodClients.Count > 0))
            {
                foreach (var client in _settings.KnownGoodClients)
                {
                    if (ImGui.MenuItem($"{client.Name}##open_saved_{client.Path}"))
                        OpenSavedGameFolder(client);
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
                        QueueKnownGoodClientAction(client.Path, client.BuildLabel, attachLooseFolder: true);
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

            if (ImGui.MenuItem("Open Workspace", enabled: !string.IsNullOrWhiteSpace(_session.Source.ArchiveRoot) || !string.IsNullOrWhiteSpace(_session.Source.InputPath) || _session.ModelOutput.HasInput()))
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
            ImGui.MenuItem("Workspaces", string.Empty, ref _showWorkspaceWindow);
            ImGui.MenuItem("Source Controls", string.Empty, ref _showControlWindow);
            ImGui.MenuItem("File Browser", string.Empty, ref _showFileBrowserWindow);
            ImGui.MenuItem("Diagnostics", string.Empty, ref _showDiagnosticsWindow);
            ImGui.MenuItem("World Status", string.Empty, ref _showWorldStatusWindow);
            ImGui.MenuItem("World Navigator", string.Empty, ref _showNavigatorWindow);
            ImGui.MenuItem("World Inspector", string.Empty, ref _showInspectorWindow);
            ImGui.MenuItem("Runtime Boundaries", string.Empty, ref _showBoundaryWindow);
            ImGui.MenuItem("About", string.Empty, ref _showAboutWindow);
            ImGui.EndMenu();
        }

        ImGui.TextDisabled($"{ProjectIdentity.SolutionName} {ProjectIdentity.PlannedVersion}");
        ImGui.EndMainMenuBar();
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
            DrawWorkspaceOption(WowViewerWorkspaceMode.WorldSession, "Bounded client-root attach and WDT-backed world session bootstrap. No world renderer yet.");
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
        if (!ImGui.Begin($"{_session.GetWorkspaceLabel()} Controls", ref _showControlWindow))
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
        if (ImGui.RadioButton("Archive-backed input", useArchive))
            _session.Source.Kind = WowViewerAssetSourceKind.ArchiveVirtualPath;
        ImGui.SameLine();
        if (ImGui.RadioButton("Local file input", !_session.Source.UsesArchiveSource))
            _session.Source.Kind = WowViewerAssetSourceKind.LocalFile;

        ImGui.Separator();
        if (_session.Source.UsesArchiveSource)
        {
            string archiveRoot = _session.Source.ArchiveRoot;
            string virtualPath = _session.Source.VirtualPath;
            string looseOverlayRoot = _session.Source.LooseOverlayRoot;
            ImGui.InputText("Archive Root", ref archiveRoot, 1024);
            ImGui.InputText("Virtual Path", ref virtualPath, 1024);
            ImGui.InputText("Loose Overlay Root", ref looseOverlayRoot, 1024);
            _session.Source.ArchiveRoot = archiveRoot;
            _session.Source.VirtualPath = virtualPath;
            _session.Source.LooseOverlayRoot = looseOverlayRoot;

            if (ImGui.Button("Attach Loose Folder...", new Vector2(-1, 0)))
                PromptAttachLooseFolder();
        }
        else
        {
            string inputPath = _session.Source.InputPath;
            ImGui.InputText("Input File", ref inputPath, 1024);
            _session.Source.InputPath = inputPath;
        }

        string buildLabel = _session.Source.BuildLabel;
        ImGui.InputText("Build Label", ref buildLabel, 256);
        _session.Source.BuildLabel = buildLabel;

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
            _session.Source.Kind = WowViewerAssetSourceKind.ArchiveVirtualPath;
            _session.Source.ArchiveRoot = @"H:\CLIENTS\WoW335\3.X_Retail_Windows_enUS_3.3.5.12340\World of Warcraft";
            _session.Source.VirtualPath = @"Creature/Wolf/Wolf.m2";
            _session.Source.BuildLabel = "3.3.5.12340";
            _session.ProfileIndex = 0;
            _session.SequenceIndex = 0;
            _session.TimeMs = 0;
            _session.VisualSize = 384;
        }

        if (ImGui.Button("Use Camera Overlay Baseline", new Vector2(-1, 0)))
        {
            _session.WorkspaceMode = WowViewerWorkspaceMode.StandaloneM2;
            _session.Source.Kind = WowViewerAssetSourceKind.ArchiveVirtualPath;
            _session.Source.ArchiveRoot = @"H:\CLIENTS\WoW335\3.X_Retail_Windows_enUS_3.3.5.12340\World of Warcraft";
            _session.Source.VirtualPath = @"Cameras/Scry_cam.m2";
            _session.Source.BuildLabel = "3.3.5.12340";
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
        if (ImGui.RadioButton("Archive-backed input", useArchive))
            _session.Source.Kind = WowViewerAssetSourceKind.ArchiveVirtualPath;
        ImGui.SameLine();
        if (ImGui.RadioButton("Local file input", !_session.Source.UsesArchiveSource))
            _session.Source.Kind = WowViewerAssetSourceKind.LocalFile;

        ImGui.Separator();
        if (_session.Source.UsesArchiveSource)
        {
            string archiveRoot = _session.Source.ArchiveRoot;
            string looseOverlayRoot = _session.Source.LooseOverlayRoot;
            ImGui.InputText("Archive Root", ref archiveRoot, 1024);
            _session.Source.ArchiveRoot = archiveRoot;

            string virtualPath = _session.Source.VirtualPath;
            ImGui.InputText("Virtual Path", ref virtualPath, 1024);
            _session.Source.VirtualPath = virtualPath;

            ImGui.InputText("Loose Overlay Root", ref looseOverlayRoot, 1024);
            _session.Source.LooseOverlayRoot = looseOverlayRoot;

            if (ImGui.Button("Attach Loose Folder...", new Vector2(-1, 0)))
                PromptAttachLooseFolder();
        }
        else
        {
            string inputPath = _session.Source.InputPath;
            ImGui.InputText("Input File", ref inputPath, 1024);
            _session.Source.InputPath = inputPath;
        }

        if (ImGui.Button("Browse..."))
            PromptOpenGameFolder();

        string buildLabel = _session.Source.BuildLabel;
        ImGui.InputText("Build Label", ref buildLabel, 256);
        _session.Source.BuildLabel = buildLabel;

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

        if (ImGui.Button("Browse Files...", new Vector2(-1, 0)))
        {
            _showFileBrowserWindow = true;
            _mdxFileBrowserState ??= new MdxFileBrowserState();
        }
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
        if (ImGui.RadioButton("Archive-backed input", useArchive))
            _session.Source.Kind = WowViewerAssetSourceKind.ArchiveVirtualPath;
        ImGui.SameLine();
        if (ImGui.RadioButton("Local file input", !_session.Source.UsesArchiveSource))
            _session.Source.Kind = WowViewerAssetSourceKind.LocalFile;

        ImGui.Separator();
        if (_session.Source.UsesArchiveSource)
        {
            string archiveRoot = _session.Source.ArchiveRoot;
            string virtualPath = _session.Source.VirtualPath;
            string looseOverlayRoot = _session.Source.LooseOverlayRoot;
            ImGui.InputText("Archive Root", ref archiveRoot, 1024);
            ImGui.InputText("Virtual Path", ref virtualPath, 1024);
            ImGui.InputText("Loose Overlay Root", ref looseOverlayRoot, 1024);
            _session.Source.ArchiveRoot = archiveRoot;
            _session.Source.VirtualPath = virtualPath;
            _session.Source.LooseOverlayRoot = looseOverlayRoot;

            if (ImGui.Button("Attach Loose Folder...", new Vector2(-1, 0)))
                PromptAttachLooseFolder();
        }
        else
        {
            string inputPath = _session.Source.InputPath;
            ImGui.InputText("Input File", ref inputPath, 1024);
            _session.Source.InputPath = inputPath;
        }

        if (ImGui.Button("Browse..."))
            PromptOpenGameFolder();

        string buildLabel = _session.Source.BuildLabel;
        ImGui.InputText("Build Label", ref buildLabel, 256);
        _session.Source.BuildLabel = buildLabel;

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

    private void DrawMdxFileBrowserWindow()
    {
        _mdxFileBrowserState ??= new MdxFileBrowserState();

        string archiveRoot = _session.Source.ArchiveRoot;
        string looseOverlayRoot = _session.Source.LooseOverlayRoot;
        if (FileBrowserEx.DrawMdxFileBrowser("MDX File Browser", ref _showFileBrowserWindow, archiveRoot, looseOverlayRoot, _mdxFileBrowserState, OnMdxFileBrowserFileSelected))
        {
            // File was selected - the control window state is already updated by OnMdxFileBrowserFileSelected
        }
    }

    private void OnMdxFileBrowserFileSelected(string virtualPath)
    {
        _session.WorkspaceMode = WowViewerWorkspaceMode.StandaloneMdx;
        _session.Source.Kind = WowViewerAssetSourceKind.ArchiveVirtualPath;
        _session.Source.ArchiveRoot = _session.Source.ArchiveRoot;
        _session.Source.VirtualPath = virtualPath;
        LoadActiveWorkspace();
    }

    private void PromptOpenGameFolder()
    {
        _wantOpenGameFolder = true;
    }

    private void PromptAttachLooseFolder()
    {
        _wantAttachLooseFolder = true;
    }

    private void QueueKnownGoodClientAction(string gamePath, string? buildLabel, bool attachLooseFolder)
    {
        _pendingKnownGoodClientPath = gamePath;
        _pendingKnownGoodClientBuildLabel = buildLabel;
        _pendingKnownGoodClientAttachLooseFolder = attachLooseFolder;
    }

    private void OpenSavedGameFolder(KnownGoodClientEntry client)
    {
        _session.Source.Kind = WowViewerAssetSourceKind.ArchiveVirtualPath;
        _session.Source.ArchiveRoot = client.Path;
        _session.Source.BuildLabel = client.BuildLabel;
        _session.Source.VirtualPath = string.Empty;
        _session.Source.LooseOverlayRoot = string.Empty;
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

        _session.Source.LooseOverlayRoot = normalizedRoot;
        _settings.LastOpenedLooseOverlayPath = normalizedRoot;
        _mdxFileBrowserState = null;
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
                _session.Source.Kind = WowViewerAssetSourceKind.ArchiveVirtualPath;
                _session.Source.ArchiveRoot = selectedPath;
                _session.Source.BuildLabel = string.Empty;
                _session.Source.VirtualPath = string.Empty;
                _session.Source.LooseOverlayRoot = string.Empty;
                _settings.LastOpenedClientPath = selectedPath;
                SaveSettings();
                _statusMessage = $"Opened game folder: {selectedPath}";
            }
            else
            {
                _statusMessage = "Folder picker unavailable on this platform/runtime. Enter the client path manually in Archive Root/Client Root.";
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
            _pendingKnownGoodClientPath = null;
            _pendingKnownGoodClientBuildLabel = null;
            _pendingKnownGoodClientAttachLooseFolder = false;

            if (!Directory.Exists(savedBasePath))
            {
                _statusMessage = $"Saved client path no longer exists: {savedBasePath}";
                return;
            }

            _session.Source.Kind = WowViewerAssetSourceKind.ArchiveVirtualPath;
            _session.Source.ArchiveRoot = savedBasePath;
            _session.Source.BuildLabel = savedBuildLabel ?? string.Empty;
            _session.Source.VirtualPath = string.Empty;
            _session.Source.LooseOverlayRoot = string.Empty;
            _settings.LastOpenedClientPath = savedBasePath;
            SaveSettings();

            if (attachLooseFolder)
            {
                string? overlayPath = TryShowFolderDialog(
                    "Select loose folder to load against the saved base client",
                    _settings.LastOpenedLooseOverlayPath,
                    showNewFolderButton: false);

                if (!string.IsNullOrWhiteSpace(overlayPath))
                    AttachLooseFolder(overlayPath);
            }
            else
            {
                _statusMessage = $"Loaded saved client: {savedBasePath}";
            }
        }
    }

    private static string? TryShowFolderDialog(string description, string? initialDir = null, bool showNewFolderButton = false)
    {
        if (!OperatingSystem.IsWindows())
            return null;

        string? result = null;

        var thread = new System.Threading.Thread(() =>
        {
            try
            {
                const string folderDialogTypeName = "System.Windows.Forms.FolderBrowserDialog, System.Windows.Forms";
                const string dialogResultTypeName = "System.Windows.Forms.DialogResult, System.Windows.Forms";

                Type? folderDialogType = Type.GetType(folderDialogTypeName, throwOnError: false);
                Type? dialogResultType = Type.GetType(dialogResultTypeName, throwOnError: false);
                if (folderDialogType == null || dialogResultType == null)
                    return;

                using IDisposable? dialog = Activator.CreateInstance(folderDialogType) as IDisposable;
                if (dialog == null)
                    return;

                folderDialogType.GetProperty("Description")?.SetValue(dialog, description);
                folderDialogType.GetProperty("ShowNewFolderButton")?.SetValue(dialog, showNewFolderButton);

                if (!string.IsNullOrEmpty(initialDir) && Directory.Exists(initialDir))
                    folderDialogType.GetProperty("SelectedPath")?.SetValue(dialog, initialDir);

                object? okResult = Enum.Parse(dialogResultType, "OK");
                object? dialogResult = folderDialogType.GetMethod("ShowDialog", Type.EmptyTypes)?.Invoke(dialog, null);

                if (dialogResult != null && okResult != null && dialogResult.Equals(okResult))
                    result = folderDialogType.GetProperty("SelectedPath")?.GetValue(dialog) as string;
            }
            catch
            {
                // Keep this bounded and non-fatal for cross-platform hosts.
            }
        });

        try
        {
            thread.SetApartmentState(System.Threading.ApartmentState.STA);
        }
        catch
        {
            return null;
        }

        thread.Start();
        thread.Join();

        return result;
    }

    private void DrawWorldControlContents()
    {
        ImGui.TextWrapped("This slice keeps the world path bounded: one selected ADT tile is opened through wow-viewer-owned bootstrap, placement, visibility, and pass-planning seams. The shell now adds navigator and inspector surfaces around that frame, but it still does not claim the final 3D world renderer.");
        ImGui.Separator();
        ImGui.TextDisabled($"Workspace: {_session.GetWorkspaceLabel()}");
        ImGui.TextDisabled($"Source: {_session.World.Describe()}");
        ImGui.Separator();

        string clientRoot = _session.World.ClientRoot;
        string mapInput = _session.World.MapInput;
        string buildLabel = _session.World.BuildLabel;
        string looseOverlayRoot = _session.World.LooseOverlayRoot;
        int tileX = _session.World.TileX;
        int tileY = _session.World.TileY;
        bool showWmos = _session.World.ShowWmos;
        bool showDoodads = _session.World.ShowDoodads;
        bool showSky = _session.World.ShowSky;
        bool showWdl = _session.World.ShowWdl;
        bool showTerrain = _session.World.ShowTerrain;
        bool showLiquid = _session.World.ShowLiquid;
        bool showOverlay = _session.World.ShowOverlay;
        ImGui.InputText("Client Root", ref clientRoot, 1024);
        ImGui.InputText("Map", ref mapInput, 256);
        ImGui.InputText("Build Label", ref buildLabel, 256);
        ImGui.InputText("Loose Overlay Root", ref looseOverlayRoot, 1024);
        ImGui.InputInt("Tile X", ref tileX);
        ImGui.InputInt("Tile Y", ref tileY);
        _session.World.ClientRoot = clientRoot;
        _session.World.MapInput = mapInput;
        _session.World.BuildLabel = buildLabel;
        _session.World.LooseOverlayRoot = looseOverlayRoot;
        _session.World.TileX = tileX;
        _session.World.TileY = tileY;
        ImGui.Separator();
        ImGui.TextDisabled("World Layers");
        ImGui.Checkbox("Sky", ref showSky);
        ImGui.SameLine();
        ImGui.Checkbox("WDL", ref showWdl);
        ImGui.SameLine();
        ImGui.Checkbox("Terrain", ref showTerrain);
        ImGui.Checkbox("Liquid", ref showLiquid);
        ImGui.SameLine();
        ImGui.Checkbox("Overlay", ref showOverlay);
        ImGui.Separator();
        ImGui.TextDisabled("Object Families");
        ImGui.Checkbox("WMO", ref showWmos);
        ImGui.SameLine();
        ImGui.Checkbox("Doodads (MDX/M2)", ref showDoodads);
        _session.World.ShowWmos = showWmos;
        _session.World.ShowDoodads = showDoodads;
        _session.World.ShowSky = showSky;
        _session.World.ShowWdl = showWdl;
        _session.World.ShowTerrain = showTerrain;
        _session.World.ShowLiquid = showLiquid;
        _session.World.ShowOverlay = showOverlay;
        _session.Normalize();

        if (ImGui.Button("Open World Session", new Vector2(-1, 0)))
            LoadActiveWorkspace();

        if (ImGui.Button("Use WoW335 Azeroth Baseline", new Vector2(-1, 0)))
        {
            _session.WorkspaceMode = WowViewerWorkspaceMode.WorldSession;
            _session.World.ClientRoot = @"H:\CLIENTS\WoW335\3.X_Retail_Windows_enUS_3.3.5.12340\World of Warcraft";
            _session.World.MapInput = "Azeroth";
            _session.World.BuildLabel = "3.3.5.12340";
            _session.World.TileX = -1;
            _session.World.TileY = -1;
        }

        if (_currentWorldRuntimeFrame != null)
        {
            ImGui.Separator();
            ImGui.TextDisabled("Current Runtime Frame");
            ImGui.Text($"Resolved Map: {_currentWorldRuntimeFrame.Session.ResolvedMapDirectory}");
            ImGui.Text($"Selected Tile: ({_currentWorldRuntimeFrame.SelectedTileX},{_currentWorldRuntimeFrame.SelectedTileY})");
            ImGui.Text($"Visible Objects: {_currentWorldRuntimeFrame.Visibility.VisibleWmos.Count + _currentWorldRuntimeFrame.Visibility.VisibleMdx.Count}");
            ImGui.Text($"Pending Assets: {_currentWorldRuntimeFrame.PendingAssetKeys.Count}");
            ImGui.Text($"WDL Range: {FormatHeightRange(_currentWorldRuntimeFrame.WdlTileData)}");
            ImGui.Text($"Terrain Chunks: {_currentWorldRuntimeFrame.Stats.TerrainChunksRendered}/{_currentWorldRuntimeFrame.TileStageSummary.TerrainChunkCount}");
            ImGui.Text($"Terrain Areas: {_currentWorldRuntimeFrame.TerrainTileData.DistinctAreaIdCount}");
            ImGui.Text($"Liquid Chunks: {_currentWorldRuntimeFrame.Stats.Liquid.VisibleCount}/{_currentWorldRuntimeFrame.TileStageSummary.LiquidChunkCount}");
            ImGui.Text($"Liquid Types: {FormatLiquidTypeCounts(_currentWorldRuntimeFrame.LiquidTileData)}");
            ImGui.Text($"Pass Options: WMO {_currentWorldRuntimeFrame.PassOptions.WmosVisible}, Doodads {_currentWorldRuntimeFrame.PassOptions.DoodadsVisible}, WDL {_currentWorldRuntimeFrame.PassOptions.WdlVisible}, Terrain {_currentWorldRuntimeFrame.PassOptions.TerrainVisible}, Liquid {_currentWorldRuntimeFrame.PassOptions.LiquidVisible}, Overlay {_currentWorldRuntimeFrame.PassOptions.OverlayVisible}");
        }
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
            ImGui.TextDisabled("Fly camera: left-drag look, WASD move, Q/E move vertically, Shift accelerate, wheel changes move speed, double-click preview to reset.");
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

    private void DrawPreviewWindow()
    {
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

    private void DrawWorldSessionPreview()
    {
        if (_currentWorldSession == null)
        {
            ImGui.TextWrapped("No world session opened yet.");
            return;
        }

        ImGui.TextWrapped(_currentWorldRuntimeFrame == null
            ? "World session bootstrap is active. This slice proves fixed-root attach plus shared WDT summary/tile discovery only; it does not render the world yet."
            : "World runtime bridge is active over one selected ADT tile. This view now includes a bounded software terrain preview plus shared visibility and pass coordinators, not the final 3D world renderer.");
        ImGui.Separator();
        ImGui.TextDisabled($"Client Root: {_currentWorldSession.ClientRoot}");
        if (!string.IsNullOrWhiteSpace(_currentWorldSession.BuildLabel))
            ImGui.TextDisabled($"Build: {_currentWorldSession.BuildLabel}");
        ImGui.Text($"Map: {_currentWorldSession.RequestedMapInput} -> {_currentWorldSession.ResolvedMapDirectory}");
        ImGui.Text($"WDT Source: {_currentWorldSession.WdtSourcePath}");
        ImGui.Text($"Load Source: {(_currentWorldSession.LoadedFromArchive ? "archive catalog" : "loose file")}");
        ImGui.Text($"Map.dbc Resolution: {(_currentWorldSession.ResolvedViaDbc ? "resolved" : (_currentWorldSession.UsedMapDirectoryLookup ? "direct directory fallback" : "lookup unavailable; direct directory fallback"))}");

        if (_currentWorldRuntimeFrame == null)
        {
            ImGui.Separator();
            ImGui.Text($"Tiles With Data: {_currentWorldSession.WdtSummary.TilesWithData}/{_currentWorldSession.WdtSummary.TotalTiles}");
            ImGui.Text($"WMO Based: {_currentWorldSession.WdtSummary.IsWmoBased}");
            ImGui.Text($"Top-level Chunks: {_currentWorldSession.FileSummary.ChunkCount}");
            ImGui.Text($"Occupancy Sample: {FormatTileSample(_currentWorldSession.OccupiedTiles, 12)}");
            return;
        }

        ImGui.Separator();
        ImGui.Text($"Selected Tile: ({_currentWorldRuntimeFrame.SelectedTileX},{_currentWorldRuntimeFrame.SelectedTileY})");
        ImGui.Text($"Placement Source: {_currentWorldRuntimeFrame.PlacementSourcePath}");
        ImGui.Text($"Placements: WMO {_currentWorldRuntimeFrame.WmoInstances.Count} / Doodads {_currentWorldRuntimeFrame.MdxInstances.Count}");
        ImGui.Text($"Visible: WMO {_currentWorldRuntimeFrame.Visibility.VisibleWmos.Count} / Doodads {_currentWorldRuntimeFrame.Visibility.VisibleMdx.Count}");
        ImGui.Text($"Pending Assets: {_currentWorldRuntimeFrame.PendingAssetKeys.Count}");
        if (_selectedWorldObject.HasValue && TryResolveWorldNavigatorEntry(_currentWorldRuntimeFrame, _selectedWorldObject.Value, out WorldNavigatorEntry selectedEntry))
            ImGui.Text($"Selection: {(selectedEntry.Kind == WorldSelectionKind.Wmo ? "WMO" : $"{selectedEntry.Instance.AssetKind} Doodad")} {selectedEntry.Instance.ModelName} #{selectedEntry.Instance.UniqueId}");

        if (_worldTerrainPreviewTextureHandle != 0)
        {
            ImGui.Separator();
            ImGui.TextDisabled("Terrain Preview");
            Vector2 previewAvailable = ImGui.GetContentRegionAvail();
            float previewSize = MathF.Max(180f, MathF.Min(previewAvailable.X, 320f));
            ImGui.Image((nint)_worldTerrainPreviewTextureHandle, new Vector2(previewSize, previewSize));
            ImGui.TextDisabled($"{_currentWorldRuntimeFrame.TerrainVisualSnapshot.Width}x{_currentWorldRuntimeFrame.TerrainVisualSnapshot.Height} samples={_currentWorldRuntimeFrame.TerrainVisualSnapshot.SampledPixelCount}");
            ImGui.TextDisabled($"Range {FormatTerrainHeightRange(_currentWorldRuntimeFrame.TerrainTileData)} hash={_currentWorldRuntimeFrame.TerrainVisualSnapshot.VisualHash}");
        }

        ImGui.Separator();
        ImGui.TextDisabled("Object Navigator");
        Vector2 available = ImGui.GetContentRegionAvail();
        float canvasSize = MathF.Max(200f, MathF.Min(available.X, available.Y));
        Vector2 canvas = new(canvasSize, canvasSize);
        Vector2 origin = ImGui.GetCursorScreenPos();
        ImGui.InvisibleButton("worldRuntimeCanvas", canvas);
        if (ImGui.IsItemHovered() && ImGui.IsMouseClicked(ImGuiMouseButton.Left))
            TrySelectWorldObjectAtCanvasPoint(_currentWorldRuntimeFrame, origin, canvas, ImGui.GetIO().MousePos);
        DrawWorldRuntimeCanvas(origin, canvas, _currentWorldRuntimeFrame);
    }

    private void DrawDiagnosticsWindow(float deltaSeconds)
    {
        ImGui.SetNextWindowSize(new Vector2(480, 720), ImGuiCond.FirstUseEver);
        if (!ImGui.Begin($"{_session.GetWorkspaceLabel()} Diagnostics", ref _showDiagnosticsWindow))
        {
            ImGui.End();
            return;
        }

        if (!IsImplementedWorkspace(_session.WorkspaceMode))
        {
            ImGui.TextWrapped($"{_session.GetWorkspaceLabel()} diagnostics are not implemented yet. This workspace exists so later WMO or MDX consumers can land without reshaping the whole shell again.");
            ImGui.End();
            return;
        }

        if (_session.WorkspaceMode == WowViewerWorkspaceMode.WorldSession)
        {
            DrawWorldDiagnostics();
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

        ImGui.End();
    }

    private void DrawWorldDiagnostics()
    {
        if (_currentWorldSession == null)
        {
            ImGui.TextWrapped("Open a world session to inspect WDT summary, MAIN flags, and occupied tile samples.");
            return;
        }

        if (_currentWorldRuntimeFrame != null)
        {
            DrawWorldRuntimeDiagnostics(_currentWorldRuntimeFrame);
            return;
        }

        ImGui.TextDisabled("World Session Summary");
        ImGui.Text($"Root: {_currentWorldSession.ClientRoot}");
        ImGui.Text($"Map: {_currentWorldSession.RequestedMapInput} -> {_currentWorldSession.ResolvedMapDirectory}");
        ImGui.Text($"Load: {_currentWorldSession.LoadDuration.TotalMilliseconds:F1} ms");
        ImGui.Text($"WDT Kind: {_currentWorldSession.FileSummary.Kind}");
        ImGui.Text($"WDT Version: {_currentWorldSession.FileSummary.Version?.ToString() ?? "n/a"}");
        ImGui.Text($"WDT Chunks: {_currentWorldSession.FileSummary.ChunkCount}");
        ImGui.Separator();

        WdtSummary summary = _currentWorldSession.WdtSummary;
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
        ImGui.TextWrapped(FormatTileSample(_currentWorldSession.OccupiedTiles, 24));
    }

    private void DrawWorldRuntimeDiagnostics(WowViewerWorldRuntimeFrameResult result)
    {
        int embeddedWmoMdxCount = result.WmoInstances.Sum(static instance => instance.WmoDoodadMdxCount);
        int embeddedWmoM2Count = result.WmoInstances.Sum(static instance => instance.WmoDoodadM2Count);
        int embeddedWmoUnknownCount = result.WmoInstances.Sum(static instance => instance.WmoDoodadUnknownCount);
        string wmoVersionSummary = string.Join(", ", result.WmoInstances
            .Where(static instance => instance.WmoVersion.HasValue)
            .Select(static instance => instance.WmoVersion!.Value)
            .Distinct()
            .OrderBy(static version => version));

        ImGui.TextDisabled("World Runtime Bridge");
        ImGui.Text($"Tile: ({result.SelectedTileX},{result.SelectedTileY})");
        ImGui.Text($"Placement Source: {result.PlacementSourcePath}");
        ImGui.Text($"Camera: {FormatVector3(result.CameraPosition)} -> {FormatVector3(result.CameraForward)}");
        ImGui.Text($"Object Phase Executed: {result.ObjectPhaseExecuted}");
        ImGui.Text($"Pass Options: sky={result.PassOptions.SkyVisible} wdl={result.PassOptions.WdlVisible} terrain={result.PassOptions.TerrainVisible} liquid={result.PassOptions.LiquidVisible} overlay={result.PassOptions.OverlayVisible}");
        ImGui.Text($"Object Filters: wmo={result.PassOptions.WmosVisible} doodads={result.PassOptions.DoodadsVisible}");
        ImGui.Text($"Total Cpu Ms: {result.Stats.TotalCpuMs:F2}");
        ImGui.Separator();

        ImGui.TextDisabled("Placement Inventory");
        ImGui.Text($"WMO Total: {result.WmoInstances.Count}");
        ImGui.Text($"WMO Ready: {result.ReadyWmoCount}");
        ImGui.Text($"WMO Versions: {(string.IsNullOrEmpty(wmoVersionSummary) ? "n/a" : wmoVersionSummary)}");
        ImGui.Text($"Embedded WMO Doodads: MDX {embeddedWmoMdxCount} / M2 {embeddedWmoM2Count} / Unknown {embeddedWmoUnknownCount}");
        ImGui.Text($"Doodad Total: {result.MdxInstances.Count}");
        ImGui.Text($"Doodad Ready: {result.ReadyMdxCount}");
        ImGui.Text($"Pending Assets: {result.PendingAssetKeys.Count}");
        if (result.PendingAssetKeys.Count > 0)
            ImGui.TextWrapped($"Pending Sample: {string.Join(", ", result.PendingAssetKeys.Take(8))}");

        ImGui.Separator();
        ImGui.TextDisabled("Visibility");
        ImGui.Text($"Visible WMO: {result.Visibility.VisibleWmos.Count}");
        ImGui.Text($"Culled WMO: {result.CulledWmoCount}");
        ImGui.Text($"Visible Doodads: {result.Visibility.VisibleMdx.Count}");
        ImGui.Text($"Culled Doodads: {result.CulledMdxCount}");
        ImGui.Text($"Taxi Doodads: {result.Visibility.VisibleTaxiMdxCount}");

        ImGui.Separator();
        ImGui.TextDisabled("Pass Coordination");
        ImGui.Text($"WDL Tiles: {result.Stats.WdlVisibleTileCount}/{result.TileStageSummary.WdlVisibleTileCount}");
        ImGui.Text($"WDL Found: {result.WdlTileData.SourceFound}");
        ImGui.Text($"WDL Version: {FormatOptionalUInt(result.WdlTileData.Version)}");
        ImGui.Text($"WDL Range: {FormatHeightRange(result.WdlTileData)}");
        ImGui.TextWrapped($"WDL Sample: center={FormatOptionalHeight(result.WdlTileData.CenterHeight)} {FormatWdlCorners(result.WdlTileData)}");
        ImGui.Text($"Terrain Chunks: {result.Stats.TerrainChunksRendered}/{result.TileStageSummary.TerrainChunkCount}");
        ImGui.Text($"Terrain Hole Chunks: {result.TileStageSummary.TerrainHoleChunkCount}");
        ImGui.Text($"Terrain Areas: {result.TerrainTileData.DistinctAreaIdCount}");
        ImGui.Text($"Terrain Range: {FormatTerrainHeightRange(result.TerrainTileData)}");
        ImGui.TextWrapped($"Terrain Heights: center={FormatTerrainCenter(result.TerrainTileData)} {FormatTerrainCorners(result.TerrainTileData)}");
        ImGui.Text($"Terrain Preview: {result.TerrainVisualSnapshot.Width}x{result.TerrainVisualSnapshot.Height} samples={result.TerrainVisualSnapshot.SampledPixelCount}");
        ImGui.TextWrapped($"Terrain Visual Hash: {result.TerrainVisualSnapshot.VisualHash}");
        ImGui.TextWrapped($"Terrain Sample: {FormatTerrainChunkSample(result.TerrainTileData)}");
        ImGui.Text($"Liquid Chunks: {result.Stats.Liquid.VisibleCount}/{result.TileStageSummary.LiquidChunkCount}");
        ImGui.Text($"Liquid Layers: {result.TileStageSummary.LiquidLayerCount}");
        ImGui.Text($"Liquid Visible Tiles: {result.Stats.Liquid.SubmittedCount}/{result.TileStageSummary.VisibleLiquidTileCount}");
        if (result.LiquidTileData.Chunks.Count > 0)
        {
            ImGui.TextWrapped($"Liquid Types: {FormatLiquidTypeCounts(result.LiquidTileData)}");
            ImGui.TextWrapped($"Liquid Sample: {FormatLiquidChunkSample(result.LiquidTileData)}");
        }
        ImGui.Text($"WMO Submitted: {result.Stats.WmoSubmission.SubmittedCount}");
        ImGui.Text($"MDX Animated: {result.Stats.MdxAnimation.SubmittedCount}");
        ImGui.Text($"MDX Opaque Submitted: {result.Stats.MdxOpaqueSubmission.SubmittedCount}");
        ImGui.Text($"MDX Transparent Submitted: {result.Stats.MdxTransparentSubmission.SubmittedCount}");
        ImGui.Text($"Opaque Routes: {result.PassFrame.OpaqueVisibleMdxRoutes.Count}");
        ImGui.Text($"Transparent Routes: {result.PassFrame.TransparentVisibleMdxRoutes.Count}");

        ImGui.Separator();
        ImGui.TextDisabled("Timing Hint");
        ImGui.TextWrapped(result.OptimizationHint);
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

    private void DrawBoundaryWindow()
    {
        ImGui.SetNextWindowSize(new Vector2(420, 520), ImGuiCond.FirstUseEver);
        if (!ImGui.Begin("Runtime Boundaries", ref _showBoundaryWindow))
        {
            ImGui.End();
            return;
        }

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
        ImGui.BulletText("The active shell now proves two bounded consumers: standalone M2 preview and a world-session bridge over one selected ADT tile.");
        ImGui.BulletText("The M2 path now includes a bounded wow-viewer-owned GPU preview consumer, but it is still not full native material parity.");
        ImGui.BulletText("The world path now consumes shared terrain height or preview plus visibility and pass coordinators for a bounded frame summary and world-session surfaces, but textured or 3D world rendering still remains a later slice.");

        ImGui.End();
    }

    private void DrawAboutWindow()
    {
        ImGui.SetNextWindowSize(new Vector2(420, 220), ImGuiCond.FirstUseEver);
        if (!ImGui.Begin("About", ref _showAboutWindow))
        {
            ImGui.End();
            return;
        }

        ImGui.TextWrapped("`WowViewer.App` now has a real desktop host, a bounded GPU M2 preview consumer, and a bounded world runtime bridge over one selected ADT tile with navigator and inspector surfaces. This keeps the new repo as the owner of the app shell and attach/open flow instead of continuing to route viewer work through `MdxViewer`.");
        ImGui.Separator();
        ImGui.TextDisabled("Commands");
        ImGui.BulletText("No args: open the desktop viewer");
        ImGui.BulletText("viewer [options]: open the desktop viewer with an initial M2 or world-session request");
        ImGui.BulletText("m2-frame [options]: keep the existing CLI proof flow");
        ImGui.BulletText("world-bootstrap [options]: run bounded client-root plus WDT bootstrap proof");
        ImGui.BulletText("world-frame [options]: run bounded tile placement plus runtime visibility/pass proof");
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

        if (_currentWorldRuntimeFrame == null)
        {
            ImGui.TextWrapped("Open a world session to populate runtime status for the current tile.");
            ImGui.End();
            return;
        }

        WowViewerWorldRuntimeFrameResult result = _currentWorldRuntimeFrame;
        ImGui.TextDisabled("Current World Frame");
        ImGui.Text($"Map: {result.Session.RequestedMapInput} -> {result.Session.ResolvedMapDirectory}");
        ImGui.Text($"Tile: ({result.SelectedTileX},{result.SelectedTileY})");
        ImGui.Text($"Placement Source: {result.PlacementSourcePath}");
        ImGui.Text($"Load Source: {(result.Session.LoadedFromArchive ? "archive catalog" : "loose file")}");
        ImGui.Separator();
        ImGui.TextDisabled("Runtime Summary");
        ImGui.Text($"WMO Visible/Total: {result.Visibility.VisibleWmos.Count}/{result.WmoInstances.Count}");
        ImGui.Text($"Doodads Visible/Total: {result.Visibility.VisibleMdx.Count}/{result.MdxInstances.Count}");
        ImGui.Text($"Pending Assets: {result.PendingAssetKeys.Count}");
        ImGui.Text($"Object Phase: {result.ObjectPhaseExecuted}");
        ImGui.Text($"Total Cpu Ms: {result.Stats.TotalCpuMs:F2}");

        if (_selectedWorldObject.HasValue && TryResolveWorldNavigatorEntry(result, _selectedWorldObject.Value, out WorldNavigatorEntry entry))
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

        if (_currentWorldRuntimeFrame == null)
        {
            ImGui.TextWrapped("Open a world session to browse WMO and doodad placements admitted through the runtime bridge.");
            ImGui.End();
            return;
        }

        WowViewerWorldRuntimeFrameResult result = _currentWorldRuntimeFrame;
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
        ImGui.Separator();

        if (!_worldNavigatorShowWmo && !_worldNavigatorShowMdx)
        {
            ImGui.TextWrapped("Enable at least one object family to populate the navigator.");
            ImGui.End();
            return;
        }

        if (entries.Count == 0)
        {
            ImGui.TextWrapped("No runtime objects match the current navigator filters.");
            ImGui.End();
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

                if (selected)
                    ImGui.SetItemDefaultFocus();

                ImGui.PopID();
            }

            ImGui.EndChild();
        }

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

        ImGui.End();
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
        _lastError = null;

        try
        {
            M2PreviewLoadRequest request = _session.BuildM2PreviewRequest();
            M2PreviewLoadResult preview = M2PreviewLoader.Load(request);
            UploadPreviewTexture(preview.FrameResult.VisualSnapshot);
            _gpuPreviewRenderer?.LoadPreview(preview);
            _currentPreview = preview;
            _currentMdxPreview = null;
            _currentWorldSession = null;
            _currentWorldRuntimeFrame = null;
            _selectedWorldObject = null;
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
        _lastError = null;

        try
        {
            WmoPreviewLoadRequest request = _session.BuildWmoPreviewRequest();
            WmoPreviewLoadResult preview = WmoPreviewLoader.Load(request);
            _wmoGpuPreviewRenderer?.LoadPreview(preview);
            _currentWmoPreview = preview;
            _currentPreview = null;
            _currentMdxPreview = null;
            _currentWorldSession = null;
            _currentWorldRuntimeFrame = null;
            _selectedWorldObject = null;
            _gpuPreviewRenderer?.ClearPreview();
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
        _lastError = null;

        try
        {
            MdxPreviewLoadRequest request = _session.BuildMdxPreviewRequest();
            MdxPreviewLoadResult preview = MdxPreviewLoader.Load(request);
            _mdxGpuPreviewRenderer?.LoadPreview(preview);
            _currentMdxPreview = preview;
            _currentPreview = null;
            _currentWorldSession = null;
            _currentWorldRuntimeFrame = null;
            _selectedWorldObject = null;
            _gpuPreviewRenderer?.ClearPreview();
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
        _lastError = null;

        try
        {
            WowViewerWorldRuntimeFrameResult runtimeFrame = WowViewerWorldRuntimeBridge.Build(_session.World.BuildRuntimeFrameRequest());
            _currentWorldRuntimeFrame = runtimeFrame;
            _currentWorldSession = runtimeFrame.Session;
            _currentPreview = null;
            _currentMdxPreview = null;
            _selectedWorldObject = SelectDefaultWorldObject(runtimeFrame);
            _gpuPreviewRenderer?.ClearPreview();
            _mdxGpuPreviewRenderer?.ClearPreview();
            DeletePreviewTexture();
            UploadWorldTerrainPreviewTexture(runtimeFrame.TerrainVisualSnapshot);
            _statusMessage = $"Opened world runtime frame for {runtimeFrame.Session.ResolvedMapDirectory} tile ({runtimeFrame.SelectedTileX},{runtimeFrame.SelectedTileY}) in {runtimeFrame.Stats.TotalCpuMs:F1} ms.";
            _lastLoadSummary = $"WMO {runtimeFrame.Visibility.VisibleWmos.Count}/{runtimeFrame.WmoInstances.Count}, doodads {runtimeFrame.Visibility.VisibleMdx.Count}/{runtimeFrame.MdxInstances.Count}, terrain {runtimeFrame.TerrainVisualSnapshot.Width}x{runtimeFrame.TerrainVisualSnapshot.Height}, pending {runtimeFrame.PendingAssetKeys.Count}";
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or InvalidOperationException or ArgumentException or NotSupportedException)
        {
            _lastError = ex.Message;
            _statusMessage = "World runtime bridge failed.";
        }
    }

    private void LoadModelOutputScene()
    {
        _lastError = null;

        try
        {
            _session.ModelOutput.Normalize();
            ModelOutputScene scene = ModelOutputSceneLoader.Load(_session.ModelOutput.InputPath, _session.ModelOutput.Variant);
            _currentModelOutputScene = scene;
            RefreshModelOutputGpuScene();
            _currentPreview = null;
            _currentMdxPreview = null;
            _currentWorldSession = null;
            _currentWorldRuntimeFrame = null;
            _selectedWorldObject = null;
            _gpuPreviewRenderer?.ClearPreview();
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

        _modelOutputGpuRenderer?.LoadScene(
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
        if (ImGui.IsMouseDragging(ImGuiMouseButton.Left) && mouseDelta.LengthSquared() > 0.0f)
        {
            azimuth += mouseDelta.X * 0.25f;
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
            azimuth += mouseDelta.X * 0.35f;
            elevation = Math.Clamp(elevation - (mouseDelta.Y * 0.25f), -89.0f, 89.0f);
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
        _currentPreview = null;
        _currentWmoPreview = null;
        _currentMdxPreview = null;
        _currentModelOutputScene = null;
        _currentWorldSession = null;
        _currentWorldRuntimeFrame = null;
        _selectedWorldObject = null;
        _lastError = null;
        _lastLoadSummary = "No workspace loaded.";
        _statusMessage = "Workspace cleared.";
        _gpuPreviewRenderer?.ClearPreview();
        _wmoGpuPreviewRenderer?.ClearPreview();
        _mdxGpuPreviewRenderer?.ClearPreview();
        _modelOutputGpuRenderer?.ClearScene();
        DeletePreviewTexture();
        DeleteWorldTerrainPreviewTexture();
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
        WowViewerAppSettingsStore.Save(_settings);
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
}
