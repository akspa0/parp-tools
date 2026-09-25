using System.Numerics;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Text.Json;
using ImGuiNET;
using Silk.NET.Maths;
using Silk.NET.OpenGL;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using WoWViewer.Logging;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using WoWViewer.Capture;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Runtime.Marketing;
using WoWViewer.Terrain.Vlm;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

/// <summary>
/// Capture automation: capture window, camera shot points, capture queue and request readiness.
/// Extracted verbatim from <see cref="ViewerApp"/> (Epic 251 U-01). World and app state it
/// needs comes only through <see cref="IViewerAppHost"/>; the bridge members below keep the
/// names the moved code used inside ViewerApp, so no moved body was edited.
/// </summary>
internal sealed partial class CaptureAutomationService
{
    private readonly IViewerAppHost _host;

    internal CaptureAutomationService(IViewerAppHost host)
    {
        _host = host;
    }

    // Host bridge: see CaptureAutomationService.Host.cs.

    private static readonly string CameraShotPointsPath = Path.Combine(SettingsDir, "camera_shot_points.json");
    private const float MkHarvestViewerValidationMaxVisibleMdxBoundsHeight = 24f;
    internal const int DefaultRequiredSettledFrames = 12;
    internal const int DefaultMaxFramesBeforeCapture = 480;
    internal const int DefaultBatchSettledFrames = 2;

    internal readonly List<CameraShotPoint> _cameraShotPoints = new();
    internal readonly Queue<PendingCaptureRequest> _captureQueue = new();
    internal PendingCaptureRequest? _activeCaptureRequest;
    internal MkHarvestViewerValidationCapturePlan? _pendingMkHarvestViewerValidationCapturePlan;
    private ActiveMkHarvestViewerValidationBatch? _activeMkHarvestViewerValidationBatch;
    private int _selectedCameraShotIndex = -1;
    private string _newCameraShotName = "";
    internal string _captureOutputDir = Path.Combine(OutputDir, "captures");
    private bool _captureFilterCurrentMapAndBuild = true;
    internal string _videoEncoderExecutable = "ffmpeg";
    internal int _videoCaptureFps = 30;
    internal bool _videoCaptureIncludeUi;
    internal int _videoCaptureContainerIndex;
    internal bool _taxiRideCameraEnabled;
    internal int _taxiRideCameraRouteId = -1;
    internal WorldScene? _taxiRideCameraScene;
    internal TaxiRideCameraMode _taxiRideCameraMode = TaxiRideCameraMode.Cockpit;
    internal float _taxiRideChaseDistance = 42f;
    internal float _taxiRideChaseHeight = 16f;
    internal float _taxiRideCockpitHeight = 10f;
    internal float _taxiRideFreeLookYawOffset;
    internal float _taxiRideFreeLookPitchOffset;
    internal bool _taxiRideCameraPoseInitialized;
    private Vector3 _taxiRideCameraSmoothedPosition;
    private Vector3 _taxiRideCameraSmoothedForward;
    internal long _lastTaxiRideCameraTick;
    internal ActiveVideoRecording? _activeVideoRecording;

    private const float TaxiRideCameraSmoothingHz = 12f;

    internal enum TaxiRideCameraMode
    {
        Cockpit = 0,
        Chase = 1,
    }

    private static readonly string[] VideoContainerExtensions = { ".mp4", ".mov" };
    private static readonly string[] VideoContainerLabels = { "MP4 (H.264)", "MOV (H.264)" };

    internal sealed class CameraShotPoint
    {
        public string Name { get; set; } = "shot";
        public string MapName { get; set; } = "unknown";
        public string BuildVersion { get; set; } = "unknown";
        public float PositionX { get; set; }
        public float PositionY { get; set; }
        public float PositionZ { get; set; }
        public float Yaw { get; set; }
        public float Pitch { get; set; }
        public float Roll { get; set; }
        public float FovDegrees { get; set; }
    }

    internal sealed class PendingCaptureRequest
    {
        public CameraShotPoint Shot { get; set; } = new();
        public string OutputPath { get; set; } = string.Empty;
        public bool IncludeUi { get; set; }
        public bool Applied { get; set; }
        public bool ExitAfterCapture { get; set; }
        public bool AllowWindowCloseOnCapture { get; set; }
        public bool WaitForSceneReady { get; set; }
        public int? TargetTileX { get; set; }
        public int? TargetTileY { get; set; }
        public int RequiredSettledFrames { get; set; }
        public int MaxFramesBeforeCapture { get; set; }
        public int FramesSinceApplied { get; set; }
        public int SettledFrames { get; set; }
        public bool TimedOutWaitingForScene { get; set; }
        public string? CaptureLabel { get; set; }
        public bool IsMkHarvestViewerValidationCapture { get; set; }
        public bool HideTerrainLiquids { get; set; }
        public bool HideObjects { get; set; }
        public bool HideTerrain { get; set; }
        public bool RequiresCameraPathPreload { get; set; }
        // 069 Phase 7: archeology playback per-shot
        public bool ApplyArcheologyPlayback { get; set; }
    }

    internal sealed class CaptureQueueOptions
    {
        public string? OutputPathOverride { get; init; }
        public bool WaitForSceneReady { get; init; }
        public int? TargetTileX { get; init; }
        public int? TargetTileY { get; init; }
        public int RequiredSettledFrames { get; init; } = 1;
        public int MaxFramesBeforeCapture { get; init; } = 1;
        public string? CaptureLabel { get; init; }
        public bool IsMkHarvestViewerValidationCapture { get; init; }
        public bool HideTerrainLiquids { get; init; }
        public bool HideObjects { get; init; }
        public bool HideTerrain { get; init; }
        public bool RequiresCameraPathPreload { get; init; }
        public bool AllowWindowCloseOnCapture { get; init; }
    }

    internal sealed class MkHarvestViewerValidationCaptureTile
    {
        public string TileName { get; set; } = string.Empty;
        public int TileX { get; set; }
        public int TileY { get; set; }
        public string OutputPath { get; set; } = string.Empty;
        public bool HideTerrainLiquids { get; set; }
        public bool HideObjects { get; set; }
        public bool HideTerrain { get; set; }
    }

    private sealed class CaptureTimingRecord
    {
        public string TileName { get; set; } = string.Empty;
        public string BuildVersion { get; set; } = string.Empty;
        public string MapName { get; set; } = string.Empty;
        public int TileX { get; set; }
        public int TileY { get; set; }
        public string Variant { get; set; } = string.Empty;
        public int SettledFrames { get; set; }
        public int TotalFramesSinceApplied { get; set; }
        public bool TimedOut { get; set; }
        public int RequiredSettledFrames { get; set; }
        public int MaxFramesBeforeCapture { get; set; }
        public string OutputPath { get; set; } = string.Empty;
    }

    internal sealed class MkHarvestViewerValidationCapturePlan
    {
        public string DatasetRoot { get; set; } = string.Empty;
        public string MapName { get; set; } = string.Empty;
        public string OutputDirectory { get; set; } = string.Empty;
        public string NoLiquidsOutputDirectory { get; set; } = string.Empty;
        public string NoObjectsOutputDirectory { get; set; } = string.Empty;
        public string ObjectsOnlyOutputDirectory { get; set; } = string.Empty;
        public int RequestedResolution { get; set; }
        public bool RestoreWorldRequested { get; set; }
        public bool ExitAfterCompletion { get; set; }
        public List<MkHarvestViewerValidationCaptureTile> Tiles { get; set; } = new();
        public int RequiredSettledFrames { get; set; } = DefaultRequiredSettledFrames;
        public int MaxFramesBeforeCapture { get; set; } = DefaultMaxFramesBeforeCapture;
        public int BatchSettledFrames { get; set; } = DefaultBatchSettledFrames;
        public bool FastSettleAfterBatchReady { get; set; } = true;
    }

    private sealed class ActiveMkHarvestViewerValidationBatch
    {
        public required Vector2D<int> PreviousWindowSize { get; init; }
        public required bool PreviousHideUiChrome { get; init; }
        public required int PreviousDetailedTileCountOverride { get; init; }
        public required float PreviousFogStart { get; init; }
        public required float PreviousFogEnd { get; init; }
        public required bool PreviousTerrainLightDirectionOverride { get; init; }
        public required Vector3 PreviousTerrainLightDirection { get; init; }
        public required bool PreviousVlmLightDirectionOverride { get; init; }
        public required Vector3 PreviousVlmLightDirection { get; init; }
        public required bool PreviousTerrainLiquidsVisible { get; init; }
        public required bool PreviousVlmTerrainLiquidsVisible { get; init; }
        public required bool PreviousTerrainVisible { get; init; }
        public required bool PreviousVlmTerrainVisible { get; init; }
        public required bool PreviousObjectFogEnabled { get; init; }
        public required bool PreviousShowWdlTerrain { get; init; }
        public required bool PreviousShowSky { get; init; }
        public required bool PreviousObjectsVisible { get; init; }
        public required bool PreviousWmosVisible { get; init; }
        public required bool PreviousDoodadsVisible { get; init; }
        public required bool PreviousWlLiquidsVisible { get; init; }
        public required bool PreviousIgnoreTerrainHolesGlobally { get; init; }
        public required bool PreviousIgnoreVlmTerrainHolesGlobally { get; init; }
        public required bool PreviousObjectPathFiltersEnabled { get; init; }
        public required float PreviousObjectStreamingRangeMultiplier { get; init; }
        public required float PreviousMaxVisibleMdxBoundsHeight { get; init; }
        public required bool PreviousHideTerrainOccludedMdx { get; init; }
        public required bool PreviousEnableRuntimeWmoGroupVisibility { get; init; }
        public required bool PreviousEnableRuntimeWmoGroupLiquids { get; init; }
        public required string DatasetRoot { get; init; }
        public required string MapName { get; init; }
        public required string OutputDirectory { get; init; }
        public required string NoLiquidsOutputDirectory { get; init; }
        public required string NoObjectsOutputDirectory { get; init; }
        public required string ObjectsOnlyOutputDirectory { get; init; }
        public required int RequestedResolution { get; init; }
        public required bool ExitAfterCompletion { get; init; }
        public int RemainingCaptures { get; set; }
        public bool BatchHasSettled { get; set; }
        public int RequiredSettledFrames { get; init; }
        public int MaxFramesBeforeCapture { get; init; }
        public int BatchSettledFrames { get; init; }
        public bool FastSettleAfterBatchReady { get; init; }
    }

    internal sealed class ActiveVideoRecording
    {
        public required Process EncoderProcess { get; init; }
        public required Stream EncoderInput { get; init; }
        public required StringBuilder EncoderErrorOutput { get; init; }
        public required string OutputPath { get; init; }
        public required bool IncludeUi { get; init; }
        public required int Width { get; init; }
        public required int Height { get; init; }
        public required double FrameIntervalSeconds { get; init; }
        public double FrameAccumulatorSeconds { get; set; }
        public byte[] FrameBuffer { get; set; } = Array.Empty<byte>();
        // 069 Phase 7: archeology playback
        public bool ApplyArcheologyPlayback { get; set; }
        public bool StartedArcheologyPlayback { get; init; }
        public MarketingTourAttempt? MarketingTourAttempt { get; set; }
        public bool RestoreUiChromeAfterMarketingTour { get; set; }
        public bool PreviousHideUiChrome { get; set; }
    }

    private sealed class CameraShotPointDocument
    {
        public List<CameraShotPoint> Shots { get; set; } = new();
    }

    internal void DrawCaptureAutomationWindow()
    {
        // 069 Phase 16: wrapper keeps legacy floating-window behavior.
        // Workbench sub-tab uses DrawCaptureAutomationContent directly.
        if (!ImGui.Begin("Capture Automation", ref _showCaptureAutomationWindow))
        {
            ImGui.End();
            return;
        }
        DrawCaptureAutomationContent();
        ImGui.End();
    }

    internal void DrawCaptureAutomationContent()
    {
        ImGui.TextDisabled(BuildSceneBookmarkText(CreateCameraShotPoint("current")));

        if (ImGui.Button("Copy Current Scene Bookmark"))
            _navigatorPanel.CopyTextToClipboard(BuildSceneBookmarkText(CreateCameraShotPoint("current")), "scene bookmark");

        if (ImGui.Button("Log Current Scene Bookmark"))
            LogSceneBookmark(CreateCameraShotPoint("current"));

        ImGui.Separator();

        string outputDir = _captureOutputDir;
        if (ImGui.InputText("Output Directory", ref outputDir, 1024))
            _captureOutputDir = outputDir;

        string currentMapName = GetCurrentCaptureMapName();
        string currentBuildVersion = GetCurrentCaptureBuildVersion();

        string ffmpegExecutable = _videoEncoderExecutable;
        if (ImGui.InputText("ffmpeg Executable", ref ffmpegExecutable, 1024))
            _videoEncoderExecutable = ffmpegExecutable;

        VideoEncoderResolution encoderResolution = VideoEncoderExecutableResolver.Resolve(_videoEncoderExecutable, AppContext.BaseDirectory);
        ImGui.TextDisabled($"Encoder: {encoderResolution.DisplayName} ({encoderResolution.Executable})");
        if (ImGui.Button("Verify ffmpeg"))
        {
            VideoEncoderProbeResult probe = VideoEncoderExecutableResolver.Probe(encoderResolution);
            _statusMessage = probe.Message;
        }
        ImGui.SameLine();
        ImGui.TextDisabled("Requires libx264");

        ImGui.Checkbox("Filter list to current map+build", ref _captureFilterCurrentMapAndBuild);

        int videoFps = _videoCaptureFps;
        if (ImGui.SliderInt("Video FPS", ref videoFps, 12, 60))
            _videoCaptureFps = videoFps;

        ImGui.Combo("Video Container", ref _videoCaptureContainerIndex, VideoContainerLabels, VideoContainerLabels.Length);
        ImGui.Checkbox("Video Includes UI", ref _videoCaptureIncludeUi);
        if (!_videoCaptureIncludeUi)
            ImGui.TextDisabled("Scene-only recording captures the viewport before ImGui. Use Tab before starting only when a full-window scene is desired.");

        if (_activeVideoRecording == null)
        {
            if (ImGui.Button("Start Video Recording"))
                TryStartCurrentViewVideoRecording(_videoCaptureIncludeUi);
        }
        else
        {
            if (ImGui.Button("Stop Video Recording"))
                StopVideoRecording();
        }

        if (_activeVideoRecording != null)
            ImGui.TextDisabled($"Recording: {Path.GetFileName(_activeVideoRecording.OutputPath)}");
        else
            ImGui.TextDisabled("Direct video capture uses ffmpeg to write mp4/mov from the current framebuffer.");

        if (ImGui.Button("Capture Current (No UI)"))
            QueueCurrentCameraCapture(includeUi: false);
        ImGui.SameLine();
        if (ImGui.Button("Capture Current (With UI)"))
            QueueCurrentCameraCapture(includeUi: true);

        ImGui.Separator();

        string newName = _newCameraShotName;
        if (ImGui.InputTextWithHint("Shot Name", "e.g. deadmines_entrance_pan", ref newName, 128))
            _newCameraShotName = newName;

        if (ImGui.Button("Add Shot Point From Current Camera"))
            AddCameraShotPointFromCurrentCamera();

        ImGui.Separator();

        if (ImGui.BeginChild("##camera_shot_list", new Vector2(0f, 240f), true))
        {
            for (int i = 0; i < _cameraShotPoints.Count; i++)
            {
                CameraShotPoint shot = _cameraShotPoints[i];
                if (_captureFilterCurrentMapAndBuild
                    && !string.Equals(shot.MapName, currentMapName, StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                if (_captureFilterCurrentMapAndBuild
                    && !string.Equals(shot.BuildVersion, currentBuildVersion, StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                bool selected = i == _selectedCameraShotIndex;
                string label = $"{shot.Name}##shot_{i}";
                if (ImGui.Selectable(label, selected))
                    _selectedCameraShotIndex = i;

                if (ImGui.IsItemHovered())
                {
                    ImGui.SetTooltip(
                        $"map={shot.MapName} build={shot.BuildVersion}\npos=({shot.PositionX:F2}, {shot.PositionY:F2}, {shot.PositionZ:F2}) yaw={shot.Yaw:F2} pitch={shot.Pitch:F2} fov={shot.FovDegrees:F1}");
                }
            }
        }
        ImGui.EndChild();

        bool hasSelection = _selectedCameraShotIndex >= 0 && _selectedCameraShotIndex < _cameraShotPoints.Count;
        if (hasSelection)
        {
            CameraShotPoint selectedShot = _cameraShotPoints[_selectedCameraShotIndex];
            ImGui.Separator();
            ImGui.TextDisabled($"Selected shot: {selectedShot.Name}");
            ImGui.TextDisabled(BuildSceneBookmarkText(selectedShot));

            if (ImGui.Button("Copy Selected Scene Bookmark"))
                _navigatorPanel.CopyTextToClipboard(BuildSceneBookmarkText(selectedShot), "scene bookmark");

            ImGui.SameLine();
            if (ImGui.Button("Log Selected Scene Bookmark"))
                LogSceneBookmark(selectedShot);
        }

        if (ImGui.Button("Move Camera To Selected") && hasSelection)
            ApplyCameraShotPoint(_cameraShotPoints[_selectedCameraShotIndex]);

        ImGui.SameLine();
        if (ImGui.Button("Capture Selected (No UI)") && hasSelection)
            EnqueueShotCapture(_cameraShotPoints[_selectedCameraShotIndex], includeUi: false);

        ImGui.SameLine();
        if (ImGui.Button("Capture Selected (With UI)") && hasSelection)
            EnqueueShotCapture(_cameraShotPoints[_selectedCameraShotIndex], includeUi: true);

        if (ImGui.Button("Capture Filtered Set (No UI)"))
            EnqueueFilteredShotCaptures(includeUi: false);
        ImGui.SameLine();
        if (ImGui.Button("Capture Filtered Set (With UI)"))
            EnqueueFilteredShotCaptures(includeUi: true);

        if (ImGui.Button("Delete Selected") && hasSelection)
        {
            _cameraShotPoints.RemoveAt(_selectedCameraShotIndex);
            _selectedCameraShotIndex = Math.Clamp(_selectedCameraShotIndex, 0, _cameraShotPoints.Count - 1);
            SaveCameraShotPoints();
        }

        ImGui.SameLine();
        if (ImGui.Button("Save Shot Points"))
            SaveCameraShotPoints();

        ImGui.SameLine();
        if (ImGui.Button("Reload Shot Points"))
            LoadCameraShotPoints();

        ImGui.TextDisabled($"Queued captures: {_captureQueue.Count + (_activeCaptureRequest != null ? 1 : 0)}");
    }

    private static (float wowX, float wowY, float wowZ) GetWowCoordinates(float positionX, float positionY, float positionZ)
    {
        float wowX = WoWConstants.MapOrigin - positionY;
        float wowY = WoWConstants.MapOrigin - positionX;
        return (wowX, wowY, positionZ);
    }

    internal static float GetWorldFacingDegrees(float yawDegrees)
    {
        // Compass heading from camera yaw. True North is yaw = 0 (so N/S land correctly), but
        // the yaw increases in the opposite rotational sense to the compass, which swapped
        // East and West. Negating the angle — (360 - yaw) — fixes E/W while leaving the North
        // (0deg) and South (180deg) fixed points unchanged.
        float degrees = (360f - (yawDegrees % 360f)) % 360f;
        if (degrees < 0f)
            degrees += 360f;

        return degrees;
    }

    internal static string GetWorldFacingLabel(float degrees)
    {
        string[] labels =
        {
            "N", "NE", "E", "SE", "S", "SW", "W", "NW"
        };

        int index = (int)MathF.Round(degrees / 45f) % labels.Length;
        return labels[index];
    }

    private static string BuildSceneBookmarkText(CameraShotPoint shot)
    {
        var (wowX, wowY, wowZ) = GetWowCoordinates(shot.PositionX, shot.PositionY, shot.PositionZ);
        float facingDegrees = GetWorldFacingDegrees(shot.Yaw);
        string facingLabel = GetWorldFacingLabel(facingDegrees);

        return $"Scene: map={shot.MapName} build={shot.BuildVersion} WoW=({wowX:F1}, {wowY:F1}, {wowZ:F1}) Facing={facingDegrees:F1}° {facingLabel} Local=({shot.PositionX:F1}, {shot.PositionY:F1}, {shot.PositionZ:F1}) Yaw={shot.Yaw:F1} Pitch={shot.Pitch:F1} Roll={shot.Roll:F1} FOV={shot.FovDegrees:F1}";
    }

    private void LogSceneBookmark(CameraShotPoint shot)
    {
        _statusMessage = BuildSceneBookmarkText(shot);
    }

    private void AddCameraShotPointFromCurrentCamera()
    {
        string name = string.IsNullOrWhiteSpace(_newCameraShotName)
            ? $"shot_{DateTime.UtcNow:yyyyMMdd_HHmmss}"
            : _newCameraShotName.Trim();

        CameraShotPoint shot = CreateCameraShotPoint(name);
        _cameraShotPoints.Add(shot);
        _selectedCameraShotIndex = _cameraShotPoints.Count - 1;
        _newCameraShotName = string.Empty;
        SaveCameraShotPoints();
        _statusMessage = $"Saved shot point '{shot.Name}' for map {shot.MapName} [{shot.BuildVersion}].";
    }

    private CameraShotPoint CreateCameraShotPoint(string name)
    {
        return new CameraShotPoint
        {
            Name = name,
            MapName = GetCurrentCaptureMapName(),
            BuildVersion = GetCurrentCaptureBuildVersion(),
            PositionX = _camera.Position.X,
            PositionY = _camera.Position.Y,
            PositionZ = _camera.Position.Z,
            Yaw = _camera.Yaw,
            Pitch = _camera.Pitch,
            Roll = _camera.Roll,
            FovDegrees = _fovDegrees,
        };
    }

    private void ApplyCameraShotPoint(CameraShotPoint shot)
    {
        _camera.Position = new Vector3(shot.PositionX, shot.PositionY, shot.PositionZ);
        _camera.Yaw = shot.Yaw;
        _camera.Pitch = shot.Pitch;
        _camera.Roll = shot.Roll;
        _fovDegrees = Math.Clamp(shot.FovDegrees, 20f, 90f);
    }

    internal void QueueCurrentCameraCapture(bool includeUi, bool exitAfterCapture = false, int captureAfterFrames = 1, bool allowWindowCloseOnCapture = false)
    {
        CameraShotPoint shot = CreateCameraShotPoint($"current_{DateTime.UtcNow:yyyyMMdd_HHmmss}");
        EnqueueShotCapture(
            shot,
            includeUi,
            exitAfterCapture,
            new CaptureQueueOptions
            {
                WaitForSceneReady = captureAfterFrames > 1,
                RequiredSettledFrames = captureAfterFrames > 1 ? captureAfterFrames : 1,
                MaxFramesBeforeCapture = captureAfterFrames > 1 ? Math.Max(captureAfterFrames * 12, 120) : 1,
                AllowWindowCloseOnCapture = allowWindowCloseOnCapture,
            });
    }

    private void EnqueueFilteredShotCaptures(bool includeUi)
    {
        string currentMapName = GetCurrentCaptureMapName();
        string currentBuildVersion = GetCurrentCaptureBuildVersion();

        int queued = 0;
        foreach (CameraShotPoint shot in _cameraShotPoints)
        {
            if (_captureFilterCurrentMapAndBuild
                && !string.Equals(shot.MapName, currentMapName, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            if (_captureFilterCurrentMapAndBuild
                && !string.Equals(shot.BuildVersion, currentBuildVersion, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            EnqueueShotCapture(shot, includeUi);
            queued++;
        }

        if (queued == 0)
            _statusMessage = "No shot points matched the current filter.";
    }

    internal void EnqueueShotCapture(CameraShotPoint shot, bool includeUi, bool exitAfterCapture = false)
        => EnqueueShotCapture(shot, includeUi, exitAfterCapture, null);

    internal void EnqueueShotCapture(CameraShotPoint shot, bool includeUi, bool exitAfterCapture, CaptureQueueOptions? options)
    {
        if (string.IsNullOrWhiteSpace(_captureOutputDir))
            _captureOutputDir = Path.Combine(OutputDir, "captures");

        string outputPath;
        if (!string.IsNullOrWhiteSpace(options?.OutputPathOverride))
        {
            outputPath = Path.GetFullPath(options.OutputPathOverride);
        }
        else
        {
            string safeMap = MakeSafePathSegment(shot.MapName);
            string safeBuild = MakeSafePathSegment(shot.BuildVersion);
            string safeShotName = MakeSafePathSegment(shot.Name);
            string outputMode = includeUi ? "with_ui" : "no_ui";
            string fileName = $"{DateTime.UtcNow:yyyyMMdd_HHmmssfff}_{safeShotName}_{outputMode}.png";
            outputPath = Path.Combine(_captureOutputDir, safeMap, safeBuild, fileName);
        }

        _captureQueue.Enqueue(new PendingCaptureRequest
        {
            Shot = new CameraShotPoint
            {
                Name = shot.Name,
                MapName = shot.MapName,
                BuildVersion = shot.BuildVersion,
                PositionX = shot.PositionX,
                PositionY = shot.PositionY,
                PositionZ = shot.PositionZ,
                Yaw = shot.Yaw,
                Pitch = shot.Pitch,
                Roll = shot.Roll,
                FovDegrees = shot.FovDegrees,
            },
            OutputPath = outputPath,
            IncludeUi = includeUi,
            ExitAfterCapture = exitAfterCapture,
            AllowWindowCloseOnCapture = options?.AllowWindowCloseOnCapture == true,
            WaitForSceneReady = options?.WaitForSceneReady == true,
            TargetTileX = options?.TargetTileX,
            TargetTileY = options?.TargetTileY,
            RequiredSettledFrames = Math.Max(1, options?.RequiredSettledFrames ?? 1),
            MaxFramesBeforeCapture = Math.Max(1, options?.MaxFramesBeforeCapture ?? 1),
            CaptureLabel = options?.CaptureLabel,
            IsMkHarvestViewerValidationCapture = options?.IsMkHarvestViewerValidationCapture == true,
            HideTerrainLiquids = options?.HideTerrainLiquids == true,
            HideObjects = options?.HideObjects == true,
            HideTerrain = options?.HideTerrain == true,
            RequiresCameraPathPreload = options?.RequiresCameraPathPreload == true,
            ApplyArcheologyPlayback = _archeologyApplyToNextCapture,
        });

        // 069 Phase 7: auto-start playback on first queued capture when enabled.
        if (_archeologyApplyToNextCapture && !_archeologyPlaybackActive && _worldScene != null)
            _archaeologyPanel.StartArcheologyPlayback();

        string mode = includeUi ? "with_ui" : "no_ui";
        _statusMessage = $"Queued capture '{shot.Name}' ({mode}).";
    }

    internal void PrepareNextCaptureRequest()
    {
        if (_activeCaptureRequest != null || _captureQueue.Count == 0)
            return;

        PendingCaptureRequest request = _captureQueue.Dequeue();
        _activeCaptureRequest = request;

        if (!request.IncludeUi)
            _hideUiChrome = true;

        ApplyCameraShotPoint(request.Shot);
        ApplyCaptureRequestSceneOverrides(request);

        // 069 Phase 7: advance archeology playback one step per shot.
        if (request.ApplyArcheologyPlayback && _worldScene != null
            && _worldScene.TryGetUniqueIdFilterRange(out int minId, out int maxId, out _))
        {
            int stepSize = Math.Max(1, (maxId - minId) / 32);
            int newMax = Math.Min(maxId, _worldScene.UniqueIdFilterMax + stepSize);
            _worldScene.UniqueIdFilterMax = newMax;
            _worldScene.UniqueIdFilterEnabled = true;
        }

        request.Applied = true;
        _activeCaptureRequest = request;
    }

    private void ApplyCaptureRequestSceneOverrides(PendingCaptureRequest request)
    {
        if (!request.IsMkHarvestViewerValidationCapture)
            return;

        bool showTerrainLiquids = !request.HideTerrainLiquids;
        if (_terrainManager?.LiquidRenderer != null)
            _terrainManager.LiquidRenderer.ShowLiquid = showTerrainLiquids;

        if (_terrainManager != null)
            _terrainManager.TerrainVisible = !request.HideTerrain;

        if (_vlmTerrainManager?.LiquidRenderer != null)
            _vlmTerrainManager.LiquidRenderer.ShowLiquid = showTerrainLiquids;

        if (_vlmTerrainManager != null)
            _vlmTerrainManager.TerrainVisible = !request.HideTerrain;

        if (_worldScene != null)
        {
            _worldScene.ShowWlLiquids = false;
            _worldScene.EnableRuntimeWmoGroupLiquids = showTerrainLiquids;
            _worldScene.ShowWdlTerrain = !request.HideTerrain;
            _worldScene.ShowSky = !request.HideTerrain;

            bool showObjects = !request.HideObjects;
            _worldScene.ObjectsVisible = showObjects;
            _worldScene.WmosVisible = showObjects;
            _worldScene.DoodadsVisible = showObjects;
        }
    }

    internal void CompleteCaptureIfReady(bool includeUi)
    {
        if (_activeCaptureRequest == null)
            return;

        PendingCaptureRequest request = _activeCaptureRequest;
        if (!request.Applied || request.IncludeUi != includeUi)
            return;

        if (!IsCaptureRequestReady(request))
            return;

        bool ok = TryCaptureFramebufferToPng(request.OutputPath, includeUi);
        _activeCaptureRequest = null;

        if (!includeUi)
            _hideUiChrome = false;

        _statusMessage = ok
            ? $"Captured shot: {request.OutputPath}"
            : $"Capture failed: {request.OutputPath}";

        if (ok)
        {
            ViewerLog.Important(ViewerLog.Category.Export,
                $"[Capture] Saved {(includeUi ? "with-ui" : "scene-only")} frame: {request.OutputPath}");
        }
        else
        {
            string timeoutNote = request.TimedOutWaitingForScene ? " after scene-settle timeout" : string.Empty;
            ViewerLog.Error(ViewerLog.Category.Export,
                $"[Capture] Failed {(includeUi ? "with-ui" : "scene-only")} frame{timeoutNote}: {request.OutputPath}");
            Environment.ExitCode = 1;
        }

        if (request.RequiresCameraPathPreload
            && _captureQueue.Count == 0
            && _activeCaptureRequest == null
            && !_cameraPaths._cameraPathVideoCaptureActive)
        {
            _cameraPaths.EndCameraPathPreload();
        }

        if (request.IsMkHarvestViewerValidationCapture)
        {
            if (ok)
                _mkHarvestViewerValidationCompleted++;
            else
                _mkHarvestViewerValidationFailed++;

            string timeoutNote = request.TimedOutWaitingForScene ? " after scene-settle timeout" : string.Empty;
            _datasetExportDialogs.AppendMkHarvestLogLine(
                $"{(ok ? "Captured" : "FAILED")} WoWViewer validation minimap {request.CaptureLabel ?? request.Shot.Name}{timeoutNote}: {request.OutputPath}");

            if (_activeMkHarvestViewerValidationBatch != null)
            {
                if (ok && !request.TimedOutWaitingForScene)
                    _activeMkHarvestViewerValidationBatch.BatchHasSettled = true;

                WriteCaptureTimingMetadata(request);

                _activeMkHarvestViewerValidationBatch.RemainingCaptures = Math.Max(0, _activeMkHarvestViewerValidationBatch.RemainingCaptures - 1);
                if (_activeMkHarvestViewerValidationBatch.RemainingCaptures == 0 && _captureQueue.Count == 0)
                {
                    RestoreMkHarvestViewerValidationBatch(
                        $"WoWViewer validation capture batch complete: {_mkHarvestViewerValidationCompleted} saved, {_mkHarvestViewerValidationFailed} failed.");
                }
            }
        }

        if (request.ExitAfterCapture && request.AllowWindowCloseOnCapture)
            _window.Close();
    }

    private bool IsCaptureRequestReady(PendingCaptureRequest request)
    {
        if (!request.WaitForSceneReady)
            return true;

        request.FramesSinceApplied++;

        if (request.RequiresCameraPathPreload
            && (_cameraPaths._cameraPathPreload == null || !_cameraPaths._cameraPathPreload.Ready))
        {
            request.SettledFrames = 0;
            if (request.FramesSinceApplied < request.MaxFramesBeforeCapture)
                return false;

            request.TimedOutWaitingForScene = true;
            ViewerLog.Error(ViewerLog.Category.Export,
                $"[Capture] Camera-path preload timeout: ready={_cameraPaths._cameraPathPreload?.Ready == true} frames={request.FramesSinceApplied}/{request.MaxFramesBeforeCapture}");
            return true;
        }

        if (!HasCaptureSceneContent() || !HasCaptureFramebufferReady(request.IncludeUi))
        {
            request.SettledFrames = 0;
            if (request.FramesSinceApplied < request.MaxFramesBeforeCapture)
                return false;

            request.TimedOutWaitingForScene = true;
            ViewerLog.Error(ViewerLog.Category.Export,
                $"[Capture] Scene readiness timeout: includeUi={request.IncludeUi} contentReady={HasCaptureSceneContent()} framebufferReady={HasCaptureFramebufferReady(request.IncludeUi)} frames={request.FramesSinceApplied}/{request.MaxFramesBeforeCapture}");
            return true;
        }

        if (request.IsMkHarvestViewerValidationCapture && _activeMkHarvestViewerValidationBatch != null)
        {
            Vector2D<int> framebufferSize = _window.FramebufferSize;
            if (framebufferSize.X < _activeMkHarvestViewerValidationBatch.RequestedResolution
                || framebufferSize.Y < _activeMkHarvestViewerValidationBatch.RequestedResolution)
            {
                request.SettledFrames = 0;
                if (request.FramesSinceApplied < request.MaxFramesBeforeCapture)
                    return false;

                request.TimedOutWaitingForScene = true;
                ViewerLog.Error(ViewerLog.Category.Export,
                    $"[Capture] Viewer validation timeout waiting for framebuffer size {framebufferSize.X}x{framebufferSize.Y}; required {_activeMkHarvestViewerValidationBatch.RequestedResolution}px; frames={request.FramesSinceApplied}/{request.MaxFramesBeforeCapture}");
                return true;
            }

            if (_worldScene != null && _worldScene.PendingWorldObjectLoadCount > 0)
            {
                request.SettledFrames = 0;
                if (request.FramesSinceApplied < request.MaxFramesBeforeCapture)
                    return false;

                request.TimedOutWaitingForScene = true;
                ViewerLog.Error(ViewerLog.Category.Export,
                    $"[Capture] Viewer validation timeout waiting for world objects: pending={_worldScene.PendingWorldObjectLoadCount} frames={request.FramesSinceApplied}/{request.MaxFramesBeforeCapture}");
                return true;
            }
        }

        if (request.TargetTileX is int targetTileX && request.TargetTileY is int targetTileY)
        {
            if (_terrainManager == null || !_terrainManager.IsTileLoaded(targetTileX, targetTileY) || _terrainManager.IsStreaming)
            {
                request.SettledFrames = 0;
                if (request.FramesSinceApplied < request.MaxFramesBeforeCapture)
                    return false;

                request.TimedOutWaitingForScene = true;
                ViewerLog.Error(ViewerLog.Category.Export,
                    $"[Capture] Tile readiness timeout: tile=({targetTileX},{targetTileY}) loaded={_terrainManager?.IsTileLoaded(targetTileX, targetTileY) == true} streaming={_terrainManager?.IsStreaming == true} frames={request.FramesSinceApplied}/{request.MaxFramesBeforeCapture}");
                return true;
            }
        }

        request.SettledFrames++;
        int effectiveRequiredSettledFrames = request.RequiredSettledFrames;
        if (request.IsMkHarvestViewerValidationCapture && _activeMkHarvestViewerValidationBatch != null
            && _activeMkHarvestViewerValidationBatch.FastSettleAfterBatchReady
            && _activeMkHarvestViewerValidationBatch.BatchHasSettled)
        {
            effectiveRequiredSettledFrames = Math.Max(1, _activeMkHarvestViewerValidationBatch.BatchSettledFrames);
        }

        if (request.SettledFrames < effectiveRequiredSettledFrames)
        {
            if (request.FramesSinceApplied < request.MaxFramesBeforeCapture)
                return false;

            request.TimedOutWaitingForScene = true;
            ViewerLog.Error(ViewerLog.Category.Export,
                $"[Capture] Settle-frame timeout: settled={request.SettledFrames}/{request.RequiredSettledFrames} frames={request.FramesSinceApplied}/{request.MaxFramesBeforeCapture}");
        }

        return true;
    }

    private bool HasCaptureSceneContent()
    {
        return _renderer != null || _terrainManager != null || _worldScene != null;
    }

    private bool HasCaptureFramebufferReady(bool includeUi)
    {
        if (!includeUi && _shellLayout.TryGetSceneFramebufferViewport(out _, out _, out uint sceneWidth, out uint sceneHeight))
            return sceneWidth > 0 && sceneHeight > 0;

        Vector2D<int> framebufferSize = _window.FramebufferSize;
        return framebufferSize.X > 0 && framebufferSize.Y > 0;
    }
}
