using System.Numerics;
using System.Reflection;
using ImGuiNET;
using Silk.NET.Input;
using Silk.NET.Maths;
using Silk.NET.OpenGL;
using Silk.NET.OpenGL.Extensions.ImGui;
using Silk.NET.Windowing;
using WowViewer.Core;
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
    private string _lastLoadSummary = "No preview loaded.";
    private string? _lastError;
    private M2PreviewLoadResult? _currentPreview;
    private uint _previewTextureHandle;
    private bool _showAboutWindow = true;
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

        _requestInitialLoad = _initialSession != null;
    }

    private void OnUpdate(double deltaSeconds)
    {
        _imGui?.Update((float)deltaSeconds);

        if (_requestInitialLoad)
        {
            _requestInitialLoad = false;
            LoadPreview();
        }
    }

    private unsafe void OnRender(double deltaSeconds)
    {
        if (_gl == null || _imGui == null || _window == null)
            return;

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
            if (ImGui.MenuItem("Load Preview"))
                LoadPreview();

            if (ImGui.MenuItem("Clear Preview", enabled: _currentPreview != null))
                ClearPreview();

            if (ImGui.MenuItem("Exit"))
                _window?.Close();

            ImGui.EndMenu();
        }

        if (ImGui.BeginMenu("View"))
        {
            ImGui.MenuItem("Source Controls", string.Empty, ref _showControlWindow);
            ImGui.MenuItem("Diagnostics", string.Empty, ref _showDiagnosticsWindow);
            ImGui.MenuItem("Runtime Boundaries", string.Empty, ref _showBoundaryWindow);
            ImGui.MenuItem("About", string.Empty, ref _showAboutWindow);
            ImGui.EndMenu();
        }

        ImGui.TextDisabled($"{ProjectIdentity.SolutionName} {ProjectIdentity.PlannedVersion}");
        ImGui.EndMainMenuBar();
    }

    private void DrawControlWindow()
    {
        ImGui.SetNextWindowSize(new Vector2(430, 540), ImGuiCond.FirstUseEver);
        if (!ImGui.Begin("M2 Source", ref _showControlWindow))
        {
            ImGui.End();
            return;
        }

        ImGui.TextWrapped("This first `wow-viewer` desktop shell stays library-first: it loads M2 assets only through the `wow-viewer` runtime pipeline and shows the software visual preview plus deterministic runtime diagnostics.");
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

        if (ImGui.Button("Load Preview", new Vector2(-1, 0)))
            LoadPreview();

        if (ImGui.Button("Use Wolf Runtime Baseline", new Vector2(-1, 0)))
        {
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
            _session.Source.Kind = WowViewerAssetSourceKind.ArchiveVirtualPath;
            _session.Source.ArchiveRoot = @"H:\CLIENTS\WoW335\3.X_Retail_Windows_enUS_3.3.5.12340\World of Warcraft";
            _session.Source.VirtualPath = @"Cameras/Scry_cam.m2";
            _session.Source.BuildLabel = "3.3.5.12340";
            _session.ProfileIndex = 0;
            _session.SequenceIndex = 0;
            _session.TimeMs = 0;
            _session.VisualSize = 384;
        }

        ImGui.Separator();
        ImGui.TextDisabled("Status");
        ImGui.TextWrapped(_statusMessage);
        if (!string.IsNullOrWhiteSpace(_lastError))
        {
            ImGui.Separator();
            ImGui.TextColored(new Vector4(0.95f, 0.42f, 0.32f, 1.0f), _lastError);
        }

        ImGui.End();
    }

    private void DrawPreviewWindow()
    {
        ImGui.SetNextWindowSize(new Vector2(880, 720), ImGuiCond.FirstUseEver);
        if (!ImGui.Begin("Preview"))
        {
            ImGui.End();
            return;
        }

        if (_currentPreview == null || _previewTextureHandle == 0)
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
        ImGui.Image((nint)_previewTextureHandle, new Vector2(size, size), new Vector2(0, 1), new Vector2(1, 0));
        ImGui.TextDisabled($"Preview: {snapshot.Width}x{snapshot.Height} litPixels={snapshot.LitPixelCount}");

        ImGui.End();
    }

    private void DrawDiagnosticsWindow(float deltaSeconds)
    {
        ImGui.SetNextWindowSize(new Vector2(480, 720), ImGuiCond.FirstUseEver);
        if (!ImGui.Begin("Diagnostics", ref _showDiagnosticsWindow))
        {
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
        ImGui.BulletText("The active render path is the deterministic software visual snapshot, not full GPU/world-scene parity.");
        ImGui.BulletText("WorldScene cutover remains a later runtime-consumer slice.");

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

        ImGui.TextWrapped("`WowViewer.App` now has a real desktop host. This slice keeps the new repo as the owner of the app shell and runtime preview path instead of continuing to route viewer work through `MdxViewer`.");
        ImGui.Separator();
        ImGui.TextDisabled("Commands");
        ImGui.BulletText("No args: open the desktop viewer");
        ImGui.BulletText("viewer [options]: open the desktop viewer with an initial M2 request");
        ImGui.BulletText("m2-frame [options]: keep the existing CLI proof flow");
        ImGui.End();
    }

    private void LoadPreview()
    {
        _lastError = null;

        try
        {
            M2PreviewLoadRequest request = _session.BuildM2PreviewRequest();
            M2PreviewLoadResult preview = M2PreviewLoader.Load(request);
            UploadPreviewTexture(preview.FrameResult.VisualSnapshot);
            _currentPreview = preview;
            _statusMessage = $"Loaded {preview.FrameResult.GoldenFrame.CanonicalModelPath} in {preview.LoadDuration.TotalMilliseconds:F1} ms.";
            _lastLoadSummary = $"{preview.FrameResult.VisualSnapshot.Width}x{preview.FrameResult.VisualSnapshot.Height} visual, {preview.FrameResult.RenderFrame.CommandCount} draw commands";
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or InvalidOperationException or ArgumentException or NotSupportedException)
        {
            _lastError = ex.Message;
            _statusMessage = "Preview load failed.";
        }
    }

    private void ClearPreview()
    {
        _currentPreview = null;
        _lastError = null;
        _lastLoadSummary = "No preview loaded.";
        _statusMessage = "Preview cleared.";
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
        _showControlWindow = settings.ShowControlWindow;
        _showDiagnosticsWindow = settings.ShowDiagnosticsWindow;
        _showBoundaryWindow = settings.ShowBoundaryWindow;
    }

    private void SaveSettings()
    {
        _session.Normalize();
        _settings.Session = _session;
        _settings.ShowAboutWindow = _showAboutWindow;
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
}