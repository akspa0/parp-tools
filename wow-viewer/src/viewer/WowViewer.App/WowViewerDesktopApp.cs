using System.Numerics;
using System.Reflection;
using ImGuiNET;
using Silk.NET.Input;
using Silk.NET.Maths;
using Silk.NET.OpenGL;
using Silk.NET.OpenGL.Extensions.ImGui;
using Silk.NET.Windowing;
using WowViewer.Core;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4;
using WowViewer.Core.Runtime;
using WowViewer.Core.Runtime.M2;
using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.Passes;

namespace WowViewer.App;

internal sealed class WowViewerDesktopApp : IDisposable
{
    private const string WindowTitle = "WowViewer.App";
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
    private string _statusMessage = "Configure an archive-backed or local M2 source, then load a preview.";
    private string _lastLoadSummary = "No workspace loaded.";
    private string? _lastError;
    private M2PreviewLoadResult? _currentPreview;
    private WowViewerWorldSessionBootstrapResult? _currentWorldSession;
    private uint _previewTextureHandle;
    private M2GpuPreviewRenderer? _gpuPreviewRenderer;
    private bool _showAboutWindow = true;
    private bool _showWorkspaceWindow = true;
    private bool _showControlWindow = true;
    private bool _showDiagnosticsWindow = true;
    private bool _showBoundaryWindow = true;

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
        _window.Closing += OnClose;
        _window.Run();
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _disposed = true;
        SaveSettings();
        _gpuPreviewRenderer?.Dispose();
        DeletePreviewTexture();
        _imGui?.Dispose();
        _input?.Dispose();
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

        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Lequal);
        _gl.Disable(EnableCap.CullFace);
        _gpuPreviewRenderer = new M2GpuPreviewRenderer(_gl);

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
    }

    private unsafe void OnRender(double deltaSeconds)
    {
        if (_gl == null || _imGui == null || _window == null)
            return;

        if (_currentPreview != null && _gpuPreviewRenderer?.HasRenderableGeometry == true && _session.WorkspaceMode == WowViewerWorkspaceMode.StandaloneM2)
            _gpuPreviewRenderer.Render(_session.VisualSize, _session.VisualSize);

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

    private void OnClose()
    {
        Dispose();
    }

    private void DrawUi(float deltaSeconds)
    {
        DrawMainMenuBar();
        ImGui.DockSpaceOverViewport();

        if (_showWorkspaceWindow)
            DrawWorkspaceWindow();
        if (_showControlWindow)
            DrawControlWindow();
        DrawPreviewWindow();
        if (_showDiagnosticsWindow)
            DrawDiagnosticsWindow(deltaSeconds);
        if (_showBoundaryWindow)
            DrawBoundaryWindow();
        if (_showAboutWindow)
            DrawAboutWindow();
    }

    private void DrawMainMenuBar()
    {
        if (!ImGui.BeginMainMenuBar())
            return;

        if (ImGui.BeginMenu("File"))
        {
            if (ImGui.MenuItem("Open Workspace"))
                LoadActiveWorkspace();

            if (ImGui.MenuItem("Clear Workspace", enabled: _currentPreview != null || _currentWorldSession != null))
                ClearWorkspace();

            if (ImGui.MenuItem("Exit"))
                _window?.Close();

            ImGui.EndMenu();
        }

        if (ImGui.BeginMenu("View"))
        {
            ImGui.MenuItem("Workspaces", string.Empty, ref _showWorkspaceWindow);
            ImGui.MenuItem("Source Controls", string.Empty, ref _showControlWindow);
            ImGui.MenuItem("Diagnostics", string.Empty, ref _showDiagnosticsWindow);
            ImGui.MenuItem("Runtime Boundaries", string.Empty, ref _showBoundaryWindow);
            ImGui.MenuItem("About", string.Empty, ref _showAboutWindow);
            ImGui.EndMenu();
        }

        ImGui.TextDisabled($"{ProjectIdentity.SolutionName} {ProjectIdentity.PlannedVersion}");
        ImGui.EndMainMenuBar();
    }

    private void DrawWorkspaceWindow()
    {
        ImGui.SetNextWindowSize(new Vector2(320, 280), ImGuiCond.FirstUseEver);
        if (!ImGui.Begin("Workspaces", ref _showWorkspaceWindow))
        {
            ImGui.End();
            return;
        }

        ImGui.TextWrapped("The viewer shell now exposes explicit standalone workspaces. Only the M2 workspace is implemented in this slice; WMO and MDX are deliberate placeholders so later consumers land on a stable app boundary.");
        ImGui.Separator();

        DrawWorkspaceOption(WowViewerWorkspaceMode.StandaloneM2, "Runtime-backed standalone model preview over the shared wow-viewer M2 pipeline.");
        DrawWorkspaceOption(WowViewerWorkspaceMode.StandaloneWmo, "Planned standalone WMO inspection workspace. Not implemented yet.");
        DrawWorkspaceOption(WowViewerWorkspaceMode.StandaloneMdx, "Planned standalone MDX inspection workspace. Not implemented yet.");
        DrawWorkspaceOption(WowViewerWorkspaceMode.WorldSession, "Bounded client-root attach and WDT-backed world session bootstrap. No world renderer yet.");

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
                _statusMessage = IsImplementedWorkspace(mode)
                    ? "M2 workspace active. Configure a source and load a preview."
                    : $"{GetWorkspaceLabel(mode)} is not implemented yet. This placeholder exists to keep the cutover honest about future standalone consumers.";
            }
        }

        ImGui.TextDisabled(description);
        ImGui.TextDisabled(IsImplementedWorkspace(mode) ? "Status: implemented in this slice" : "Status: placeholder only");
        ImGui.Separator();
    }

    private static bool IsImplementedWorkspace(WowViewerWorkspaceMode mode)
    {
        return mode is WowViewerWorkspaceMode.StandaloneM2 or WowViewerWorkspaceMode.WorldSession;
    }

    private static string GetWorkspaceLabel(WowViewerWorkspaceMode mode)
    {
        return mode switch
        {
            WowViewerWorkspaceMode.StandaloneM2 => "Standalone M2",
            WowViewerWorkspaceMode.StandaloneWmo => "Standalone WMO",
            WowViewerWorkspaceMode.StandaloneMdx => "Standalone MDX",
            WowViewerWorkspaceMode.WorldSession => "World Session",
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
            case WowViewerWorkspaceMode.WorldSession:
                DrawWorldControlContents();
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
            ImGui.InputText("Archive Root", ref archiveRoot, 1024);
            ImGui.InputText("Virtual Path", ref virtualPath, 1024);
            _session.Source.ArchiveRoot = archiveRoot;
            _session.Source.VirtualPath = virtualPath;
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

        if (ImGui.Button("Load M2 Preview", new Vector2(-1, 0)))
            LoadActiveWorkspace();

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

    private void DrawWorldControlContents()
    {
        ImGui.TextWrapped("This slice adds a bounded wow-viewer-owned world bootstrap path over fixed client roots and shared map readers. It opens a map session, resolves a WDT, and reports tile coverage, but it does not render terrain or world objects yet.");
        ImGui.Separator();
        ImGui.TextDisabled($"Workspace: {_session.GetWorkspaceLabel()}");
        ImGui.TextDisabled($"Source: {_session.World.Describe()}");
        ImGui.Separator();

        string clientRoot = _session.World.ClientRoot;
        string mapInput = _session.World.MapInput;
        string buildLabel = _session.World.BuildLabel;
        ImGui.InputText("Client Root", ref clientRoot, 1024);
        ImGui.InputText("Map", ref mapInput, 256);
        ImGui.InputText("Build Label", ref buildLabel, 256);
        _session.World.ClientRoot = clientRoot;
        _session.World.MapInput = mapInput;
        _session.World.BuildLabel = buildLabel;
        _session.Normalize();

        if (ImGui.Button("Open World Session", new Vector2(-1, 0)))
            LoadActiveWorkspace();

        if (ImGui.Button("Use WoW335 Azeroth Baseline", new Vector2(-1, 0)))
        {
            _session.WorkspaceMode = WowViewerWorkspaceMode.WorldSession;
            _session.World.ClientRoot = @"H:\CLIENTS\WoW335\3.X_Retail_Windows_enUS_3.3.5.12340\World of Warcraft";
            _session.World.MapInput = "Azeroth";
            _session.World.BuildLabel = "3.3.5.12340";
        }
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

        ImGui.TextWrapped("World session bootstrap is active. This slice proves fixed-root attach plus shared WDT summary/tile discovery only; it does not render the world yet.");
        ImGui.Separator();
        ImGui.TextDisabled($"Client Root: {_currentWorldSession.ClientRoot}");
        if (!string.IsNullOrWhiteSpace(_currentWorldSession.BuildLabel))
            ImGui.TextDisabled($"Build: {_currentWorldSession.BuildLabel}");
        ImGui.Text($"Map: {_currentWorldSession.RequestedMapInput} -> {_currentWorldSession.ResolvedMapDirectory}");
        ImGui.Text($"WDT Source: {_currentWorldSession.WdtSourcePath}");
        ImGui.Text($"Load Source: {(_currentWorldSession.LoadedFromArchive ? "archive catalog" : "loose file")}");
        ImGui.Text($"Map.dbc Resolution: {(_currentWorldSession.ResolvedViaDbc ? "resolved" : (_currentWorldSession.UsedMapDirectoryLookup ? "direct directory fallback" : "lookup unavailable; direct directory fallback"))}");
        ImGui.Separator();
        ImGui.Text($"Tiles With Data: {_currentWorldSession.WdtSummary.TilesWithData}/{_currentWorldSession.WdtSummary.TotalTiles}");
        ImGui.Text($"WMO Based: {_currentWorldSession.WdtSummary.IsWmoBased}");
        ImGui.Text($"Top-level Chunks: {_currentWorldSession.FileSummary.ChunkCount}");
        ImGui.Text($"Occupancy Sample: {FormatTileSample(_currentWorldSession.OccupiedTiles, 12)}");
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
        ImGui.Text($"Visible MDX: {worldStats.VisibleMdxCount}");
        ImGui.Text($"Taxi MDX: {worldStats.VisibleTaxiMdxCount}");
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
        ImGui.BulletText("The active shell now proves two bounded consumers: standalone M2 preview and fixed-root world-session bootstrap.");
        ImGui.BulletText("The M2 path now includes a bounded wow-viewer-owned GPU preview consumer, but it is still not full native material parity.");
        ImGui.BulletText("World session bootstrap currently stops at attach/open plus WDT summary and occupied-tile discovery; world rendering remains a later slice.");

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

        ImGui.TextWrapped("`WowViewer.App` now has a real desktop host, a bounded GPU M2 preview consumer, and a bounded world-session bootstrap path. This keeps the new repo as the owner of the app shell and attach/open flow instead of continuing to route viewer work through `MdxViewer`.");
        ImGui.Separator();
        ImGui.TextDisabled("Commands");
        ImGui.BulletText("No args: open the desktop viewer");
        ImGui.BulletText("viewer [options]: open the desktop viewer with an initial M2 or world-session request");
        ImGui.BulletText("m2-frame [options]: keep the existing CLI proof flow");
        ImGui.BulletText("world-bootstrap [options]: run bounded client-root plus WDT bootstrap proof");
        ImGui.End();
    }

    private void LoadActiveWorkspace()
    {
        _lastError = null;

        if (!IsImplementedWorkspace(_session.WorkspaceMode))
        {
            _statusMessage = $"{_session.GetWorkspaceLabel()} is not implemented yet. Switch to Standalone M2 or World Session for a live workspace.";
            return;
        }

        switch (_session.WorkspaceMode)
        {
            case WowViewerWorkspaceMode.StandaloneM2:
                LoadPreview();
                break;
            case WowViewerWorkspaceMode.WorldSession:
                LoadWorldSession();
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
            _currentWorldSession = null;
            _statusMessage = $"Loaded {preview.FrameResult.GoldenFrame.CanonicalModelPath} in {preview.LoadDuration.TotalMilliseconds:F1} ms.";
            bool hasGpuPreview = _gpuPreviewRenderer?.HasRenderableGeometry == true;
            _lastLoadSummary = hasGpuPreview
                ? $"GPU {_session.VisualSize}x{_session.VisualSize}, software {preview.FrameResult.VisualSnapshot.Width}x{preview.FrameResult.VisualSnapshot.Height}, {preview.FrameResult.RenderFrame.CommandCount} draw commands"
                : $"Software {preview.FrameResult.VisualSnapshot.Width}x{preview.FrameResult.VisualSnapshot.Height}, {preview.FrameResult.RenderFrame.CommandCount} draw commands";
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or InvalidOperationException or ArgumentException or NotSupportedException)
        {
            _lastError = ex.Message;
            _statusMessage = "Preview load failed.";
        }
    }

    private void LoadWorldSession()
    {
        _lastError = null;

        try
        {
            WowViewerWorldSessionBootstrapResult result = WowViewerWorldSessionBootstrapper.Open(_session.World.BuildRequest());
            _currentWorldSession = result;
            _currentPreview = null;
            _gpuPreviewRenderer?.ClearPreview();
            DeletePreviewTexture();
            _statusMessage = $"Opened world session for {result.ResolvedMapDirectory} in {result.LoadDuration.TotalMilliseconds:F1} ms.";
            _lastLoadSummary = $"{result.OccupiedTiles.Count} occupied tiles, WDT {result.FileSummary.Kind}, source {(result.LoadedFromArchive ? "archive" : "loose")}";
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or InvalidOperationException or ArgumentException or NotSupportedException)
        {
            _lastError = ex.Message;
            _statusMessage = "World session bootstrap failed.";
        }
    }

    private void ClearWorkspace()
    {
        _currentPreview = null;
        _currentWorldSession = null;
        _lastError = null;
        _lastLoadSummary = "No workspace loaded.";
        _statusMessage = "Workspace cleared.";
        _gpuPreviewRenderer?.ClearPreview();
        DeletePreviewTexture();
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

    private void ApplySession(WowViewerSession session)
    {
        ArgumentNullException.ThrowIfNull(session);
        _session.WorkspaceMode = session.WorkspaceMode;
        _session.Source.Kind = session.Source.Kind;
        _session.Source.ArchiveRoot = session.Source.ArchiveRoot ?? string.Empty;
        _session.Source.VirtualPath = session.Source.VirtualPath ?? string.Empty;
        _session.Source.InputPath = session.Source.InputPath ?? string.Empty;
        _session.Source.BuildLabel = session.Source.BuildLabel ?? string.Empty;
        _session.World.ClientRoot = session.World.ClientRoot ?? string.Empty;
        _session.World.MapInput = session.World.MapInput ?? string.Empty;
        _session.World.BuildLabel = session.World.BuildLabel ?? string.Empty;
        _session.ProfileIndex = session.ProfileIndex;
        _session.SequenceIndex = session.SequenceIndex;
        _session.TimeMs = session.TimeMs;
        _session.VisualSize = session.VisualSize;
        _session.Normalize();
    }

    private void ApplySettingsToState(WowViewerAppSettings settings)
    {
        ApplySession(settings.Session ?? WowViewerSession.CreateDefault());
        _showAboutWindow = settings.ShowAboutWindow;
        _showWorkspaceWindow = settings.ShowWorkspaceWindow;
        _showControlWindow = settings.ShowControlWindow;
        _showDiagnosticsWindow = settings.ShowDiagnosticsWindow;
        _showBoundaryWindow = settings.ShowBoundaryWindow;
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
}