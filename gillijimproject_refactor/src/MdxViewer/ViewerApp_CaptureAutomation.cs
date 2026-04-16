using System.Numerics;
using System.Diagnostics;
using System.Text;
using System.Text.Json;
using ImGuiNET;
using Silk.NET.Maths;
using Silk.NET.OpenGL;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using MdxViewer.Logging;
using MdxViewer.Rendering;
using MdxViewer.Terrain;
using WoWMapConverter.Core.VLM;

namespace MdxViewer;

public partial class ViewerApp
{
    private static readonly string CameraShotPointsPath = Path.Combine(SettingsDir, "camera_shot_points.json");

    private readonly List<CameraShotPoint> _cameraShotPoints = new();
    private readonly Queue<PendingCaptureRequest> _captureQueue = new();
    private PendingCaptureRequest? _activeCaptureRequest;
    private MkHarvestViewerValidationCapturePlan? _pendingMkHarvestViewerValidationCapturePlan;
    private ActiveMkHarvestViewerValidationBatch? _activeMkHarvestViewerValidationBatch;
    private int _selectedCameraShotIndex = -1;
    private string _newCameraShotName = "";
    private string _captureOutputDir = Path.Combine(OutputDir, "captures");
    private bool _captureFilterCurrentMapAndBuild = true;
    private string _videoEncoderExecutable = "ffmpeg";
    private int _videoCaptureFps = 30;
    private bool _videoCaptureIncludeUi;
    private int _videoCaptureContainerIndex;
    private bool _taxiRideCameraEnabled;
    private int _taxiRideCameraRouteId = -1;
    private TaxiRideCameraMode _taxiRideCameraMode = TaxiRideCameraMode.Cockpit;
    private float _taxiRideChaseDistance = 42f;
    private float _taxiRideChaseHeight = 16f;
    private float _taxiRideLookAhead = 28f;
    private float _taxiRideCockpitHeight = 10f;
    private float _taxiRideFreeLookYawOffset;
    private float _taxiRideFreeLookPitchOffset;
    private ActiveVideoRecording? _activeVideoRecording;

    private enum TaxiRideCameraMode
    {
        Cockpit = 0,
        Chase = 1,
    }

    private static readonly string[] VideoContainerExtensions = { ".mp4", ".mov" };
    private static readonly string[] VideoContainerLabels = { "MP4 (H.264)", "MOV (H.264)" };

    private sealed class CameraShotPoint
    {
        public string Name { get; set; } = "shot";
        public string MapName { get; set; } = "unknown";
        public string BuildVersion { get; set; } = "unknown";
        public float PositionX { get; set; }
        public float PositionY { get; set; }
        public float PositionZ { get; set; }
        public float Yaw { get; set; }
        public float Pitch { get; set; }
        public float FovDegrees { get; set; }
    }

    private sealed class PendingCaptureRequest
    {
        public CameraShotPoint Shot { get; set; } = new();
        public string OutputPath { get; set; } = string.Empty;
        public bool IncludeUi { get; set; }
        public bool Applied { get; set; }
        public bool ExitAfterCapture { get; set; }
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
    }

    private sealed class CaptureQueueOptions
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
    }

    private sealed class MkHarvestViewerValidationCaptureTile
    {
        public string TileName { get; set; } = string.Empty;
        public int TileX { get; set; }
        public int TileY { get; set; }
        public string OutputPath { get; set; } = string.Empty;
        public bool HideTerrainLiquids { get; set; }
        public bool HideObjects { get; set; }
        public bool HideTerrain { get; set; }
    }

    private sealed class MkHarvestViewerValidationCapturePlan
    {
        public string DatasetRoot { get; set; } = string.Empty;
        public string MapName { get; set; } = string.Empty;
        public string OutputDirectory { get; set; } = string.Empty;
        public string NoLiquidsOutputDirectory { get; set; } = string.Empty;
        public string NoObjectsOutputDirectory { get; set; } = string.Empty;
        public string ObjectsOnlyOutputDirectory { get; set; } = string.Empty;
        public int RequestedResolution { get; set; }
        public bool RestoreWorldRequested { get; set; }
        public List<MkHarvestViewerValidationCaptureTile> Tiles { get; set; } = new();
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
        public required bool PreviousObjectPathFiltersEnabled { get; init; }
        public required float PreviousObjectStreamingRangeMultiplier { get; init; }
        public required string DatasetRoot { get; init; }
        public required string MapName { get; init; }
        public required string OutputDirectory { get; init; }
        public required string NoLiquidsOutputDirectory { get; init; }
        public required string NoObjectsOutputDirectory { get; init; }
        public required string ObjectsOnlyOutputDirectory { get; init; }
        public required int RequestedResolution { get; init; }
        public int RemainingCaptures { get; set; }
    }

    private sealed class ActiveVideoRecording
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
    }

    private sealed class CameraShotPointDocument
    {
        public List<CameraShotPoint> Shots { get; set; } = new();
    }

    private void DrawCaptureAutomationWindow()
    {
        if (!ImGui.Begin("Capture Automation", ref _showCaptureAutomationWindow))
        {
            ImGui.End();
            return;
        }

        ImGui.TextDisabled(BuildSceneBookmarkText(CreateCameraShotPoint("current")));

        if (ImGui.Button("Copy Current Scene Bookmark"))
            CopyTextToClipboard(BuildSceneBookmarkText(CreateCameraShotPoint("current")), "scene bookmark");

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

        ImGui.Checkbox("Filter list to current map+build", ref _captureFilterCurrentMapAndBuild);

        int videoFps = _videoCaptureFps;
        if (ImGui.SliderInt("Video FPS", ref videoFps, 12, 60))
            _videoCaptureFps = videoFps;

        ImGui.Combo("Video Container", ref _videoCaptureContainerIndex, VideoContainerLabels, VideoContainerLabels.Length);
        ImGui.Checkbox("Video Includes UI", ref _videoCaptureIncludeUi);

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
                CopyTextToClipboard(BuildSceneBookmarkText(selectedShot), "scene bookmark");

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

        ImGui.End();
    }

    private static (float wowX, float wowY, float wowZ) GetWowCoordinates(float positionX, float positionY, float positionZ)
    {
        float wowX = WoWConstants.MapOrigin - positionY;
        float wowY = WoWConstants.MapOrigin - positionX;
        return (wowX, wowY, positionZ);
    }

    private static float GetWorldFacingDegrees(float yawDegrees)
    {
        float yawRad = yawDegrees * MathF.PI / 180f;
        float rendererForwardX = MathF.Cos(yawRad);
        float rendererForwardY = MathF.Sin(yawRad);
        float wowForwardX = -rendererForwardY;
        float wowForwardY = -rendererForwardX;

        float degrees = MathF.Atan2(-wowForwardY, wowForwardX) * 180f / MathF.PI;
        if (degrees < 0f)
            degrees += 360f;

        return degrees;
    }

    private static string GetWorldFacingLabel(float degrees)
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

        return $"Scene: map={shot.MapName} build={shot.BuildVersion} WoW=({wowX:F1}, {wowY:F1}, {wowZ:F1}) Facing={facingDegrees:F1}° {facingLabel} Local=({shot.PositionX:F1}, {shot.PositionY:F1}, {shot.PositionZ:F1}) Yaw={shot.Yaw:F1} Pitch={shot.Pitch:F1} FOV={shot.FovDegrees:F1}";
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
            FovDegrees = _fovDegrees,
        };
    }

    private void ApplyCameraShotPoint(CameraShotPoint shot)
    {
        _camera.Position = new Vector3(shot.PositionX, shot.PositionY, shot.PositionZ);
        _camera.Yaw = shot.Yaw;
        _camera.Pitch = shot.Pitch;
        _fovDegrees = Math.Clamp(shot.FovDegrees, 20f, 90f);
    }

    private void QueueCurrentCameraCapture(bool includeUi, bool exitAfterCapture = false, int captureAfterFrames = 1)
    {
        CameraShotPoint shot = CreateCameraShotPoint($"current_{DateTime.UtcNow:yyyyMMdd_HHmmss}");
        EnqueueShotCapture(
            shot,
            includeUi,
            exitAfterCapture,
            captureAfterFrames > 1
                ? new CaptureQueueOptions
                {
                    WaitForSceneReady = true,
                    RequiredSettledFrames = captureAfterFrames,
                    MaxFramesBeforeCapture = Math.Max(captureAfterFrames * 12, 120),
                }
                : null);
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

    private void EnqueueShotCapture(CameraShotPoint shot, bool includeUi, bool exitAfterCapture = false)
        => EnqueueShotCapture(shot, includeUi, exitAfterCapture, null);

    private void EnqueueShotCapture(CameraShotPoint shot, bool includeUi, bool exitAfterCapture, CaptureQueueOptions? options)
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
                FovDegrees = shot.FovDegrees,
            },
            OutputPath = outputPath,
            IncludeUi = includeUi,
            ExitAfterCapture = exitAfterCapture,
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
        });

        string mode = includeUi ? "with_ui" : "no_ui";
        _statusMessage = $"Queued capture '{shot.Name}' ({mode}).";
    }

    private void PrepareNextCaptureRequest()
    {
        if (_activeCaptureRequest != null || _captureQueue.Count == 0)
            return;

        PendingCaptureRequest request = _captureQueue.Dequeue();
        _activeCaptureRequest = request;

        ApplyCameraShotPoint(request.Shot);
        ApplyCaptureRequestSceneOverrides(request);
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
            _worldScene.ShowWdlTerrain = !request.HideTerrain;
            _worldScene.ShowSky = !request.HideTerrain;

            bool showObjects = !request.HideObjects;
            _worldScene.ObjectsVisible = showObjects;
            _worldScene.WmosVisible = showObjects;
            _worldScene.DoodadsVisible = false;
        }
    }

    private void CompleteCaptureIfReady(bool includeUi)
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

        if (request.IsMkHarvestViewerValidationCapture)
        {
            if (ok)
                _mkHarvestViewerValidationCompleted++;
            else
                _mkHarvestViewerValidationFailed++;

            string timeoutNote = request.TimedOutWaitingForScene ? " after scene-settle timeout" : string.Empty;
            AppendMkHarvestLogLine(
                $"{(ok ? "Captured" : "FAILED")} MdxViewer validation minimap {request.CaptureLabel ?? request.Shot.Name}{timeoutNote}: {request.OutputPath}");

            if (_activeMkHarvestViewerValidationBatch != null)
            {
                _activeMkHarvestViewerValidationBatch.RemainingCaptures = Math.Max(0, _activeMkHarvestViewerValidationBatch.RemainingCaptures - 1);
                if (_activeMkHarvestViewerValidationBatch.RemainingCaptures == 0 && _captureQueue.Count == 0)
                {
                    RestoreMkHarvestViewerValidationBatch(
                        $"MdxViewer validation capture batch complete: {_mkHarvestViewerValidationCompleted} saved, {_mkHarvestViewerValidationFailed} failed.");
                }
            }
        }

        if (request.ExitAfterCapture)
            _window.Close();
    }

    private bool IsCaptureRequestReady(PendingCaptureRequest request)
    {
        if (!request.WaitForSceneReady)
            return true;

        request.FramesSinceApplied++;

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
        if (request.SettledFrames < request.RequiredSettledFrames)
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
        if (!includeUi && TryGetSceneFramebufferViewport(out _, out _, out uint sceneWidth, out uint sceneHeight))
            return sceneWidth > 0 && sceneHeight > 0;

        Vector2D<int> framebufferSize = _window.FramebufferSize;
        return framebufferSize.X > 0 && framebufferSize.Y > 0;
    }

    private void PromotePendingMkHarvestViewerValidationCapturePlan()
    {
        if (_pendingMkHarvestViewerValidationCapturePlan == null || _activeMkHarvestViewerValidationBatch != null)
            return;

        MkHarvestViewerValidationCapturePlan plan = _pendingMkHarvestViewerValidationCapturePlan;
        if (_terrainManager == null)
        {
            if (!plan.RestoreWorldRequested && HasWorldReturnTarget())
            {
                string returnMapName = Path.GetFileNameWithoutExtension(_lastWorldSceneWdtPath!);
                if (string.Equals(returnMapName, plan.MapName, StringComparison.OrdinalIgnoreCase))
                {
                    plan.RestoreWorldRequested = true;
                    AppendMkHarvestLogLine($"Restoring world '{plan.MapName}' before running the MdxViewer validation capture batch.");
                    ReturnToLastWorldScene();
                    return;
                }
            }

            AppendMkHarvestLogLine($"Skipping MdxViewer validation captures because world map '{plan.MapName}' is not currently loaded in the viewer.");
            _pendingMkHarvestViewerValidationCapturePlan = null;
            return;
        }

        if (!string.Equals(_terrainManager.MapName, plan.MapName, StringComparison.OrdinalIgnoreCase))
        {
            AppendMkHarvestLogLine(
                $"Skipping MdxViewer validation captures because the current world '{_terrainManager.MapName}' does not match dataset map '{plan.MapName}'.");
            _pendingMkHarvestViewerValidationCapturePlan = null;
            return;
        }

        StartMkHarvestViewerValidationBatch(plan);
        _pendingMkHarvestViewerValidationCapturePlan = null;
    }

    private void StartMkHarvestViewerValidationBatch(MkHarvestViewerValidationCapturePlan plan)
    {
        if (_terrainManager == null)
            return;

        int requestedResolution = Math.Clamp(plan.RequestedResolution, 512, 4096);
        TerrainLighting terrainLighting = _terrainManager.Lighting;
        TerrainLighting? vlmLighting = _vlmTerrainManager?.Lighting;
        _activeMkHarvestViewerValidationBatch = new ActiveMkHarvestViewerValidationBatch
        {
            PreviousWindowSize = _window.Size,
            PreviousHideUiChrome = _hideUiChrome,
            PreviousDetailedTileCountOverride = _terrainManager.DetailedTileCountOverride,
            PreviousFogStart = terrainLighting.FogStart,
            PreviousFogEnd = terrainLighting.FogEnd,
            PreviousTerrainLightDirectionOverride = terrainLighting.HasExternalLightDirectionOverride,
            PreviousTerrainLightDirection = terrainLighting.ExternalLightDirection,
            PreviousVlmLightDirectionOverride = vlmLighting?.HasExternalLightDirectionOverride ?? false,
            PreviousVlmLightDirection = vlmLighting?.ExternalLightDirection ?? Vector3.Zero,
            PreviousTerrainLiquidsVisible = _terrainManager.LiquidRenderer?.ShowLiquid ?? true,
            PreviousVlmTerrainLiquidsVisible = _vlmTerrainManager?.LiquidRenderer?.ShowLiquid ?? true,
            PreviousTerrainVisible = _terrainManager.TerrainVisible,
            PreviousVlmTerrainVisible = _vlmTerrainManager?.TerrainVisible ?? true,
            PreviousObjectFogEnabled = _worldScene?.ObjectFogEnabled ?? true,
            PreviousShowWdlTerrain = _worldScene?.ShowWdlTerrain ?? true,
            PreviousShowSky = _worldScene?.ShowSky ?? true,
            PreviousObjectsVisible = _worldScene?.ObjectsVisible ?? true,
            PreviousWmosVisible = _worldScene?.WmosVisible ?? true,
            PreviousDoodadsVisible = _worldScene?.DoodadsVisible ?? true,
            PreviousWlLiquidsVisible = _worldScene?.ShowWlLiquids ?? true,
            PreviousObjectPathFiltersEnabled = _worldScene?.ObjectPathFiltersEnabled ?? true,
            PreviousObjectStreamingRangeMultiplier = _worldScene?.ObjectStreamingRangeMultiplier ?? 0.5f,
            DatasetRoot = plan.DatasetRoot,
            MapName = plan.MapName,
            OutputDirectory = plan.OutputDirectory,
            NoLiquidsOutputDirectory = plan.NoLiquidsOutputDirectory,
            NoObjectsOutputDirectory = plan.NoObjectsOutputDirectory,
            ObjectsOnlyOutputDirectory = plan.ObjectsOnlyOutputDirectory,
            RequestedResolution = requestedResolution,
            RemainingCaptures = plan.Tiles.Count,
        };

        _hideUiChrome = true;
        _window.Size = new Vector2D<int>(requestedResolution, requestedResolution);
        _terrainManager.DetailedTileCountOverride = Math.Min(25, TerrainManager.MaxManualDetailedTileCount);
        terrainLighting.FogStart = MaxTerrainFogDistance * 0.75f;
        terrainLighting.FogEnd = MaxTerrainFogDistance;
        Vector3 validationLightDirection = BuildMkHarvestViewerValidationLightDirection(terrainLighting.LightDirection);
        terrainLighting.ApplyExternalLightDirection(validationLightDirection);
        vlmLighting?.ApplyExternalLightDirection(validationLightDirection);
        if (_worldScene != null)
        {
            _worldScene.ObjectFogEnabled = false;
            _worldScene.ObjectsVisible = true;
            _worldScene.WmosVisible = true;
            _worldScene.DoodadsVisible = false;
            _worldScene.ShowWlLiquids = false;
            _worldScene.ObjectPathFiltersEnabled = false;
            _worldScene.ObjectStreamingRangeMultiplier = Math.Max(_worldScene.ObjectStreamingRangeMultiplier, 1.0f);
        }

        foreach (MkHarvestViewerValidationCaptureTile tile in plan.Tiles)
        {
            CameraShotPoint shot = BuildMkHarvestViewerValidationShot(plan.MapName, tile);
            EnqueueShotCapture(
                shot,
                includeUi: false,
                exitAfterCapture: false,
                new CaptureQueueOptions
                {
                    OutputPathOverride = tile.OutputPath,
                    WaitForSceneReady = true,
                    TargetTileX = tile.TileX,
                    TargetTileY = tile.TileY,
                    RequiredSettledFrames = 24,
                    MaxFramesBeforeCapture = 1800,
                    CaptureLabel = tile.HideTerrain
                        ? $"{tile.TileName} (objectsonly)"
                        : (tile.HideObjects
                            ? $"{tile.TileName} (noobjects)"
                            : (tile.HideTerrainLiquids ? $"{tile.TileName} (noliquids)" : tile.TileName)),
                    IsMkHarvestViewerValidationCapture = true,
                    HideTerrainLiquids = tile.HideTerrainLiquids,
                    HideObjects = tile.HideObjects,
                    HideTerrain = tile.HideTerrain,
                });
        }

        AppendMkHarvestLogLine(
            $"Started MdxViewer validation capture batch for {plan.Tiles.Count} capture(s). Viewer chrome is hidden, doodads and WL liquids are disabled for all variants, object path filters are disabled, the primary output keeps terrain liquids and visible objects, the 'noliquids' sub-folder disables terrain liquids, the 'noobjects' sub-folder hides world objects, the 'objectsonly' sub-folder hides terrain, WDL, liquids, and sky while keeping visible world objects, object streaming is widened, the validation sun direction is forced for deterministic top-down shading, the batch waits for world assets to settle, and the window was resized to {requestedResolution}x{requestedResolution} for the batch.");
    }

    private static Vector3 BuildMkHarvestViewerValidationLightDirection(Vector3 currentLightDirection)
    {
        Vector3 source = currentLightDirection.LengthSquared() > 1e-6f
            ? Vector3.Normalize(currentLightDirection)
            : Vector3.Normalize(new Vector3(0f, 0.3f, 1f));

        return Vector3.Normalize(new Vector3(source.X, -source.Y, MathF.Abs(source.Z)));
    }

    private CameraShotPoint BuildMkHarvestViewerValidationShot(string mapName, MkHarvestViewerValidationCaptureTile tile)
    {
        const float capturePitch = -89f;
        const float captureYaw = 0f;
        const float captureFovDegrees = 24f;

        float centerX = WoWConstants.MapOrigin - ((tile.TileX + 0.5f) * WoWConstants.ChunkSize);
        float centerY = WoWConstants.MapOrigin - ((tile.TileY + 0.5f) * WoWConstants.ChunkSize);
        float targetGroundHeight = 0f;
        if (_terrainManager != null
            && TrySampleTerrainHeightLoaded(_terrainManager.Renderer, centerX, centerY, out float loadedTerrainHeight, out _))
        {
            targetGroundHeight = loadedTerrainHeight;
        }
        else if (_vlmTerrainManager != null
            && TrySampleTerrainHeightLoaded(_vlmTerrainManager.Renderer, centerX, centerY, out float loadedVlmTerrainHeight, out _))
        {
            targetGroundHeight = loadedVlmTerrainHeight;
        }

        float desiredSpan = WoWConstants.ChunkSize;
        float heightAboveGround = 256f + (desiredSpan / (2f * MathF.Tan((captureFovDegrees * MathF.PI / 180f) * 0.5f)));
        float captureHeight = targetGroundHeight + heightAboveGround;

        return new CameraShotPoint
        {
            Name = $"{tile.TileName}_viewer_validation",
            MapName = mapName,
            BuildVersion = GetCurrentCaptureBuildVersion(),
            PositionX = centerX,
            PositionY = centerY,
            PositionZ = captureHeight,
            Yaw = captureYaw,
            Pitch = capturePitch,
            FovDegrees = captureFovDegrees,
        };
    }

    private bool TryGetMkHarvestViewerValidationSceneMatrices(float aspect, out Matrix4x4 view, out Matrix4x4 proj)
    {
        view = Matrix4x4.Identity;
        proj = Matrix4x4.Identity;

        if (_activeMkHarvestViewerValidationBatch == null
            || _activeCaptureRequest?.IsMkHarvestViewerValidationCapture != true
            || _activeCaptureRequest.TargetTileX is not int tileX
            || _activeCaptureRequest.TargetTileY is not int tileY)
        {
            return false;
        }

        float centerX = WoWConstants.MapOrigin - ((tileX + 0.5f) * WoWConstants.ChunkSize);
        float centerY = WoWConstants.MapOrigin - ((tileY + 0.5f) * WoWConstants.ChunkSize);
        float targetGroundHeight = 0f;
        if (_terrainManager != null
            && TrySampleTerrainHeightLoaded(_terrainManager.Renderer, centerX, centerY, out float loadedTerrainHeight, out _))
        {
            targetGroundHeight = loadedTerrainHeight;
        }
        else if (_vlmTerrainManager != null
            && TrySampleTerrainHeightLoaded(_vlmTerrainManager.Renderer, centerX, centerY, out float loadedVlmTerrainHeight, out _))
        {
            targetGroundHeight = loadedVlmTerrainHeight;
        }

        float worldSpanX = WoWConstants.ChunkSize * Math.Max(1f, aspect);
        float worldSpanY = WoWConstants.ChunkSize / Math.Max(1f, aspect <= 0f ? 1f : Math.Min(1f, aspect));
        if (aspect > 0f && aspect < 1f)
            worldSpanX = WoWConstants.ChunkSize;
        if (aspect >= 1f)
            worldSpanY = WoWConstants.ChunkSize;

        Vector3 eye = new(centerX, centerY, targetGroundHeight + 2048f);
        Vector3 target = new(centerX, centerY, targetGroundHeight);
        view = Matrix4x4.CreateLookAt(eye, target, Vector3.UnitX);
        proj = Matrix4x4.CreateOrthographic(worldSpanX, worldSpanY, 0.1f, GetSceneFarPlane());
        return true;
    }

    private void RestoreMkHarvestViewerValidationBatch(string? statusMessage = null)
    {
        if (_activeMkHarvestViewerValidationBatch == null)
            return;

        ActiveMkHarvestViewerValidationBatch batch = _activeMkHarvestViewerValidationBatch;
        _activeMkHarvestViewerValidationBatch = null;

        _hideUiChrome = batch.PreviousHideUiChrome;
        _window.Size = batch.PreviousWindowSize;

        if (_terrainManager != null)
        {
            _terrainManager.DetailedTileCountOverride = batch.PreviousDetailedTileCountOverride;
            _terrainManager.Lighting.FogStart = batch.PreviousFogStart;
            _terrainManager.Lighting.FogEnd = batch.PreviousFogEnd;
            _terrainManager.TerrainVisible = batch.PreviousTerrainVisible;
            if (_terrainManager.LiquidRenderer != null)
                _terrainManager.LiquidRenderer.ShowLiquid = batch.PreviousTerrainLiquidsVisible;
            if (batch.PreviousTerrainLightDirectionOverride)
                _terrainManager.Lighting.ApplyExternalLightDirection(batch.PreviousTerrainLightDirection);
            else
                _terrainManager.Lighting.ClearExternalLightDirection();
        }

        if (_vlmTerrainManager != null)
        {
            _vlmTerrainManager.TerrainVisible = batch.PreviousVlmTerrainVisible;
            if (_vlmTerrainManager.LiquidRenderer != null)
                _vlmTerrainManager.LiquidRenderer.ShowLiquid = batch.PreviousVlmTerrainLiquidsVisible;
            if (batch.PreviousVlmLightDirectionOverride)
                _vlmTerrainManager.Lighting.ApplyExternalLightDirection(batch.PreviousVlmLightDirection);
            else
                _vlmTerrainManager.Lighting.ClearExternalLightDirection();
        }

        if (_worldScene != null)
        {
            _worldScene.ObjectFogEnabled = batch.PreviousObjectFogEnabled;
            _worldScene.ShowWdlTerrain = batch.PreviousShowWdlTerrain;
            _worldScene.ShowSky = batch.PreviousShowSky;
            _worldScene.ObjectsVisible = batch.PreviousObjectsVisible;
            _worldScene.WmosVisible = batch.PreviousWmosVisible;
            _worldScene.DoodadsVisible = batch.PreviousDoodadsVisible;
            _worldScene.ShowWlLiquids = batch.PreviousWlLiquidsVisible;
            _worldScene.ObjectPathFiltersEnabled = batch.PreviousObjectPathFiltersEnabled;
            _worldScene.ObjectStreamingRangeMultiplier = batch.PreviousObjectStreamingRangeMultiplier;
        }

        StitchMkHarvestViewerValidationOutputs(
            batch.MapName,
            batch.OutputDirectory,
            batch.NoLiquidsOutputDirectory,
            batch.NoObjectsOutputDirectory,
            batch.ObjectsOnlyOutputDirectory,
            batch.RequestedResolution);
        GenerateMkHarvestViewerValidationObjectArtifacts(batch.DatasetRoot, batch.OutputDirectory, batch.NoObjectsOutputDirectory, batch.ObjectsOnlyOutputDirectory);

        if (!string.IsNullOrWhiteSpace(statusMessage))
        {
            _statusMessage = statusMessage;
            AppendMkHarvestLogLine(statusMessage);
        }
    }

    private void StitchMkHarvestViewerValidationOutputs(
        string mapName,
        string outputDirectory,
        string noLiquidsOutputDirectory,
        string noObjectsOutputDirectory,
        string objectsOnlyOutputDirectory,
        int requestedResolution)
    {
        TryStitchMkHarvestViewerValidationDirectory(outputDirectory, mapName, requestedResolution, "viewer_validation_minimaps");
        TryStitchMkHarvestViewerValidationDirectory(noLiquidsOutputDirectory, mapName, requestedResolution, "viewer_validation_minimaps/noliquids");
        TryStitchMkHarvestViewerValidationDirectory(noObjectsOutputDirectory, mapName, requestedResolution, "viewer_validation_minimaps/noobjects");
        TryStitchMkHarvestViewerValidationDirectory(objectsOnlyOutputDirectory, mapName, requestedResolution, "viewer_validation_minimaps/objectsonly");
    }

    private static readonly JsonSerializerOptions MkDatasetJsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals,
    };

    private void GenerateMkHarvestViewerValidationObjectArtifacts(
        string datasetRoot,
        string withObjectsOutputDirectory,
        string noObjectsOutputDirectory,
        string objectsOnlyOutputDirectory)
    {
        if (string.IsNullOrWhiteSpace(datasetRoot)
            || string.IsNullOrWhiteSpace(withObjectsOutputDirectory)
            || string.IsNullOrWhiteSpace(noObjectsOutputDirectory)
            || !Directory.Exists(withObjectsOutputDirectory)
            || !Directory.Exists(noObjectsOutputDirectory))
        {
            return;
        }

        string datasetDirectory = Path.Combine(datasetRoot, "dataset");
        if (!Directory.Exists(datasetDirectory))
            return;

        string imagesDirectory = Path.Combine(datasetRoot, "images");
        Directory.CreateDirectory(imagesDirectory);

        int updatedTiles = 0;
        int skippedTiles = 0;

        foreach (string jsonPath in Directory.GetFiles(datasetDirectory, "*.json"))
        {
            string fileName = Path.GetFileName(jsonPath);
            if (string.Equals(fileName, "texture_database.json", StringComparison.OrdinalIgnoreCase))
                continue;

            string tileName = Path.GetFileNameWithoutExtension(jsonPath);
            string withObjectsPath = Path.Combine(withObjectsOutputDirectory, $"{tileName}_viewer_validation.png");
            string noObjectsPath = Path.Combine(noObjectsOutputDirectory, $"{tileName}_viewer_validation.png");
            if (!File.Exists(withObjectsPath) || !File.Exists(noObjectsPath))
            {
                skippedTiles++;
                continue;
            }

            try
            {
                using Image<Rgba32> withObjectsImage = SixLabors.ImageSharp.Image.Load<Rgba32>(withObjectsPath);
                using Image<Rgba32> noObjectsImage = SixLabors.ImageSharp.Image.Load<Rgba32>(noObjectsPath);
                if (noObjectsImage.Width != withObjectsImage.Width || noObjectsImage.Height != withObjectsImage.Height)
                {
                    noObjectsImage.Mutate(ctx => ctx.Resize(withObjectsImage.Width, withObjectsImage.Height));
                }

                using Image<L8> maskImage = TryBuildDirectObjectVisibilityMask(tileName, withObjectsImage.Width, withObjectsImage.Height, objectsOnlyOutputDirectory)
                    ?? BuildObjectVisibilityDiffMask(withObjectsImage, noObjectsImage);

                string objectMaskFileName = $"{tileName}_object_visibility_mask.png";
                string noObjectFileName = $"{tileName}_no_objects.png";
                string objectMaskRelativePath = $"images/{objectMaskFileName}";
                string noObjectRelativePath = $"images/{noObjectFileName}";

                string objectMaskPath = Path.Combine(imagesDirectory, objectMaskFileName);
                string noObjectOutPath = Path.Combine(imagesDirectory, noObjectFileName);

                maskImage.SaveAsPng(objectMaskPath);
                noObjectsImage.SaveAsPng(noObjectOutPath);

                VlmTrainingSample? sample = JsonSerializer.Deserialize<VlmTrainingSample>(File.ReadAllText(jsonPath), MkDatasetJsonOptions);
                if (sample?.TerrainData == null)
                {
                    skippedTiles++;
                    continue;
                }

                VlmTerrainData updatedTerrain = sample.TerrainData with
                {
                    ObjectVisibilityMaskPath = objectMaskRelativePath,
                    NoObjectMinimapPath = noObjectRelativePath,
                };

                VlmTrainingSample updatedSample = sample with { TerrainData = updatedTerrain };
                File.WriteAllText(jsonPath, JsonSerializer.Serialize(updatedSample, MkDatasetJsonOptions));
                updatedTiles++;
            }
            catch
            {
                skippedTiles++;
            }
        }

        AppendMkHarvestLogLine($"Object-visibility artifacts: updated {updatedTiles} tile json(s), skipped {skippedTiles} tile(s) without matching captures. Direct object-only masks are preferred when available; with/no-object diff remains the fallback.");
    }

    private static Image<L8>? TryBuildDirectObjectVisibilityMask(string tileName, int width, int height, string objectsOnlyOutputDirectory)
    {
        if (string.IsNullOrWhiteSpace(objectsOnlyOutputDirectory) || !Directory.Exists(objectsOnlyOutputDirectory))
            return null;

        string objectsOnlyPath = Path.Combine(objectsOnlyOutputDirectory, $"{tileName}_viewer_validation.png");
        if (!File.Exists(objectsOnlyPath))
            return null;

        using Image<Rgba32> objectsOnlyImage = SixLabors.ImageSharp.Image.Load<Rgba32>(objectsOnlyPath);
        if (objectsOnlyImage.Width != width || objectsOnlyImage.Height != height)
            objectsOnlyImage.Mutate(ctx => ctx.Resize(width, height));

        return BuildObjectVisibilityMaskFromObjectsOnlyCapture(objectsOnlyImage);
    }

    private static Image<L8> BuildObjectVisibilityMaskFromObjectsOnlyCapture(Image<Rgba32> objectsOnly)
    {
        const int intensityThreshold = 4;
        var mask = new Image<L8>(objectsOnly.Width, objectsOnly.Height);

        for (int y = 0; y < objectsOnly.Height; y++)
        {
            for (int x = 0; x < objectsOnly.Width; x++)
            {
                Rgba32 pixel = objectsOnly[x, y];
                int intensity = Math.Max(pixel.R, Math.Max(pixel.G, pixel.B));
                mask[x, y] = new L8((byte)(intensity > intensityThreshold ? 255 : 0));
            }
        }

        return mask;
    }

    private static Image<L8> BuildObjectVisibilityDiffMask(Image<Rgba32> withObjects, Image<Rgba32> noObjects)
    {
        const int diffThreshold = 8;
        var mask = new Image<L8>(withObjects.Width, withObjects.Height);

        for (int y = 0; y < withObjects.Height; y++)
        {
            for (int x = 0; x < withObjects.Width; x++)
            {
                Rgba32 withPixel = withObjects[x, y];
                Rgba32 noPixel = noObjects[x, y];
                int diffR = Math.Abs(withPixel.R - noPixel.R);
                int diffG = Math.Abs(withPixel.G - noPixel.G);
                int diffB = Math.Abs(withPixel.B - noPixel.B);
                int diff = Math.Max(diffR, Math.Max(diffG, diffB));
                mask[x, y] = new L8((byte)(diff >= diffThreshold ? 255 : 0));
            }
        }

        return mask;
    }

    private void TryStitchMkHarvestViewerValidationDirectory(string imagesDirectory, string mapName, int requestedResolution, string variantLabel)
    {
        if (string.IsNullOrWhiteSpace(imagesDirectory) || string.IsNullOrWhiteSpace(mapName) || !Directory.Exists(imagesDirectory))
            return;

        string stitchedDirectory = Path.Combine(imagesDirectory, "stitched");
        Directory.CreateDirectory(stitchedDirectory);

        string outputPath = Path.Combine(stitchedDirectory, $"{mapName}_full_viewer_validation.png");
        var bounds = TileStitchingService.StitchFullMap(
            imagesDirectory,
            mapName,
            requestedResolution,
            outputPath,
            suffix: "_viewer_validation.png");

        if (bounds.HasValue)
        {
            AppendMkHarvestLogLine(
                $"Stitched {variantLabel} into {outputPath} using tile bounds {bounds.Value.minX:D2},{bounds.Value.minY:D2} -> {bounds.Value.maxX:D2},{bounds.Value.maxY:D2}.");
        }
    }

    private bool TryStartTaxiRideVideoCapture()
    {
        if (_worldScene == null || _worldScene.SelectedTaxiRouteId < 0)
        {
            _statusMessage = "Select a taxi route before starting ride capture.";
            return false;
        }

        if (!TryAttachTaxiRideCameraToSelectedRoute())
            return false;

        return TryStartCurrentViewVideoRecording(_videoCaptureIncludeUi, GetTaxiRouteDisplayLabel(_taxiRideCameraRouteId));
    }

    private bool TryStartCurrentViewVideoRecording(bool includeUi, string? label = null)
    {
        if (_activeVideoRecording != null)
        {
            _statusMessage = "A video recording is already in progress.";
            return false;
        }

        if (!TryGetCaptureRegion(includeUi, out _, out _, out int width, out int height))
        {
            _statusMessage = includeUi
                ? "Unable to resolve the current framebuffer size for video capture."
                : "Unable to resolve the scene viewport for no-UI video capture.";
            return false;
        }

        if (width <= 0 || height <= 0)
        {
            _statusMessage = "Video capture dimensions were invalid.";
            return false;
        }

        string encoderExecutable = string.IsNullOrWhiteSpace(_videoEncoderExecutable)
            ? "ffmpeg"
            : _videoEncoderExecutable.Trim();

        string extension = VideoContainerExtensions[Math.Clamp(_videoCaptureContainerIndex, 0, VideoContainerExtensions.Length - 1)];
        string safeMap = MakeSafePathSegment(GetCurrentCaptureMapName());
        string safeBuild = MakeSafePathSegment(GetCurrentCaptureBuildVersion());
        string safeLabel = MakeSafePathSegment(string.IsNullOrWhiteSpace(label) ? "current_view" : label);
        string captureMode = includeUi ? "with_ui" : "no_ui";
        string outputPath = Path.Combine(
            string.IsNullOrWhiteSpace(_captureOutputDir) ? Path.Combine(OutputDir, "captures") : _captureOutputDir,
            safeMap,
            safeBuild,
            $"{DateTime.UtcNow:yyyyMMdd_HHmmssfff}_{safeLabel}_{captureMode}{extension}");

        string? outputDirectory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrWhiteSpace(outputDirectory))
            Directory.CreateDirectory(outputDirectory);

        try
        {
            var startInfo = new ProcessStartInfo
            {
                FileName = encoderExecutable,
                UseShellExecute = false,
                RedirectStandardInput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
                WorkingDirectory = Environment.CurrentDirectory,
            };

            StringBuilder encoderErrorOutput = new();

            startInfo.ArgumentList.Add("-y");
            startInfo.ArgumentList.Add("-f");
            startInfo.ArgumentList.Add("rawvideo");
            startInfo.ArgumentList.Add("-pixel_format");
            startInfo.ArgumentList.Add("rgba");
            startInfo.ArgumentList.Add("-video_size");
            startInfo.ArgumentList.Add($"{width}x{height}");
            startInfo.ArgumentList.Add("-framerate");
            startInfo.ArgumentList.Add(_videoCaptureFps.ToString());
            startInfo.ArgumentList.Add("-i");
            startInfo.ArgumentList.Add("-");
            startInfo.ArgumentList.Add("-vf");
            startInfo.ArgumentList.Add(BuildVideoCaptureFilter(width, height));
            startInfo.ArgumentList.Add("-an");
            startInfo.ArgumentList.Add("-c:v");
            startInfo.ArgumentList.Add("libx264");
            startInfo.ArgumentList.Add("-preset");
            startInfo.ArgumentList.Add("veryfast");
            startInfo.ArgumentList.Add("-pix_fmt");
            startInfo.ArgumentList.Add("yuv420p");
            startInfo.ArgumentList.Add(outputPath);

            Process process = Process.Start(startInfo)
                ?? throw new InvalidOperationException("ffmpeg did not start.");
            process.ErrorDataReceived += (_, args) => AppendVideoEncoderError(encoderErrorOutput, args.Data);
            process.BeginErrorReadLine();

            _activeVideoRecording = new ActiveVideoRecording
            {
                EncoderProcess = process,
                EncoderInput = process.StandardInput.BaseStream,
                EncoderErrorOutput = encoderErrorOutput,
                OutputPath = outputPath,
                IncludeUi = includeUi,
                Width = width,
                Height = height,
                FrameIntervalSeconds = 1.0 / Math.Max(1, _videoCaptureFps),
                FrameAccumulatorSeconds = 0.0,
                FrameBuffer = new byte[width * height * 4],
            };

            _statusMessage = $"Started video recording: {outputPath}";
            return true;
        }
        catch (Exception ex)
        {
            _statusMessage = $"Failed to start video recording: {ex.Message}";
            return false;
        }
    }

    private void StopVideoRecording(string? statusOverride = null)
    {
        if (_activeVideoRecording == null)
            return;

        ActiveVideoRecording recording = _activeVideoRecording;
        _activeVideoRecording = null;

        bool success = false;
        string statusMessage = statusOverride ?? $"Saved video: {recording.OutputPath}";

        try
        {
            recording.EncoderInput.Flush();
        }
        catch
        {
        }

        try
        {
            recording.EncoderInput.Dispose();
        }
        catch
        {
        }

        try
        {
            if (!recording.EncoderProcess.WaitForExit(10000))
                recording.EncoderProcess.Kill(entireProcessTree: true);

            success = recording.EncoderProcess.ExitCode == 0;
            if (!success && statusOverride == null)
                statusMessage = $"Video encode failed for {recording.OutputPath} (exit {recording.EncoderProcess.ExitCode}).";

            string encoderError = GetVideoEncoderErrorSummary(recording);
            if (!string.IsNullOrWhiteSpace(encoderError) && (!success || statusOverride != null))
                statusMessage = $"{statusMessage} ffmpeg: {encoderError}";
        }
        catch (Exception ex)
        {
            statusMessage = statusOverride ?? $"Failed to finish video recording: {ex.Message}";
        }
        finally
        {
            try
            {
                recording.EncoderProcess.CancelErrorRead();
            }
            catch
            {
            }

            recording.EncoderProcess.Dispose();
        }

        if (!success && statusOverride == null && File.Exists(recording.OutputPath))
        {
            try
            {
                File.Delete(recording.OutputPath);
            }
            catch
            {
            }
        }

        _statusMessage = statusMessage;
    }

    private void CaptureVideoFrameIfNeeded(bool includeUi, double dt)
    {
        if (_activeVideoRecording == null || _activeVideoRecording.IncludeUi != includeUi)
            return;

        ActiveVideoRecording recording = _activeVideoRecording;
        recording.FrameAccumulatorSeconds += Math.Max(0.0, dt);
        if (recording.FrameAccumulatorSeconds + 1e-6 < recording.FrameIntervalSeconds)
        {
            _activeVideoRecording = recording;
            return;
        }

        if (!TryGetCaptureRegion(includeUi, out int readX, out int readY, out int width, out int height))
        {
            StopVideoRecording(includeUi
                ? "Video recording stopped because the framebuffer was unavailable."
                : "Video recording stopped because the scene viewport was unavailable.");
            return;
        }

        if (width != recording.Width || height != recording.Height)
        {
            StopVideoRecording("Video recording stopped because the capture size changed during recording.");
            return;
        }

        if (recording.EncoderProcess.HasExited)
        {
            StopVideoRecording("Video recording stopped because ffmpeg exited before the first frame was accepted.");
            return;
        }

        int framesToWrite = Math.Max(1, (int)(recording.FrameAccumulatorSeconds / recording.FrameIntervalSeconds));
        recording.FrameAccumulatorSeconds -= framesToWrite * recording.FrameIntervalSeconds;

        byte[] pixels = recording.FrameBuffer.Length == recording.Width * recording.Height * 4
            ? recording.FrameBuffer
            : new byte[recording.Width * recording.Height * 4];
        recording.FrameBuffer = pixels;

        if (!TryReadFramebufferRgba(readX, readY, recording.Width, recording.Height, pixels))
        {
            StopVideoRecording("Video recording stopped because framebuffer capture failed.");
            return;
        }

        try
        {
            for (int frameIndex = 0; frameIndex < framesToWrite; frameIndex++)
                recording.EncoderInput.Write(pixels, 0, pixels.Length);
            _activeVideoRecording = recording;
        }
        catch (Exception ex)
        {
            StopVideoRecording($"Video recording stopped because ffmpeg write failed: {ex.Message}");
        }
    }

    private bool TryAttachTaxiRideCameraToSelectedRoute()
    {
        if (_worldScene == null || _worldScene.SelectedTaxiRouteId < 0)
        {
            _statusMessage = "Select a taxi route before enabling the ride camera.";
            return false;
        }

        _worldScene.ShowTaxi = true;
        _worldScene.ShowTaxiActors = true;
        _taxiRideCameraRouteId = _worldScene.SelectedTaxiRouteId;
        _taxiRideCameraEnabled = true;
        _taxiRideFreeLookYawOffset = 0f;
        _taxiRideFreeLookPitchOffset = 0f;
        _statusMessage = $"Ride camera attached to {GetTaxiRouteDisplayLabel(_taxiRideCameraRouteId)}.";
        return true;
    }

    private void StopTaxiRideCamera(string? statusMessage = null)
    {
        _taxiRideCameraEnabled = false;
        _taxiRideCameraRouteId = -1;
        _taxiRideFreeLookYawOffset = 0f;
        _taxiRideFreeLookPitchOffset = 0f;
        if (!string.IsNullOrWhiteSpace(statusMessage))
            _statusMessage = statusMessage;
    }

    private void AdjustTaxiRideFreeLook(float deltaYawDegrees, float deltaPitchDegrees)
    {
        if (!_taxiRideCameraEnabled)
            return;

        _taxiRideFreeLookYawOffset += deltaYawDegrees;
        while (_taxiRideFreeLookYawOffset > 180f)
            _taxiRideFreeLookYawOffset -= 360f;
        while (_taxiRideFreeLookYawOffset < -180f)
            _taxiRideFreeLookYawOffset += 360f;

        _taxiRideFreeLookPitchOffset = Math.Clamp(_taxiRideFreeLookPitchOffset + deltaPitchDegrees, -75f, 75f);
    }

    private void UpdateTaxiRideCamera()
    {
        if (!_taxiRideCameraEnabled)
            return;

        if (_worldScene == null)
        {
            StopTaxiRideCamera("Ride camera detached because the world scene is no longer active.");
            return;
        }

        if (!_worldScene.TryGetTaxiActorPose(_taxiRideCameraRouteId, out TaxiActorPose pose))
            return;

        Vector3 forward = pose.Forward.LengthSquared() > 0.0001f
            ? Vector3.Normalize(pose.Forward)
            : _camera.Forward;
        Vector3 horizontalForward = new Vector3(forward.X, forward.Y, 0f);
        if (horizontalForward.LengthSquared() > 0.0001f)
            horizontalForward = Vector3.Normalize(horizontalForward);
        else
            horizontalForward = new Vector3(_camera.Forward.X, _camera.Forward.Y, 0f);

        if (horizontalForward.LengthSquared() <= 0.0001f)
            horizontalForward = Vector3.UnitY;

        float baseYawDegrees = MathF.Atan2(horizontalForward.Y, horizontalForward.X) * 180f / MathF.PI;
        float desiredYawDegrees = baseYawDegrees + _taxiRideFreeLookYawOffset;
        Vector3 orbitForward = GetDirectionFromYawPitch(desiredYawDegrees, 0f);
        Vector3 lookForward = GetDirectionFromYawPitch(desiredYawDegrees, _taxiRideFreeLookPitchOffset);

        float scale = Math.Max(0.25f, pose.Scale);
        if (_taxiRideCameraMode == TaxiRideCameraMode.Cockpit)
        {
            Vector3 eyePosition = pose.Position + Vector3.UnitZ * (_taxiRideCockpitHeight * scale);
            ApplyDirectionalRideCamera(eyePosition, lookForward);
            return;
        }

        Vector3 chaseFocus = pose.Position + Vector3.UnitZ * Math.Max(6f, _taxiRideCockpitHeight * 0.65f * scale);
        Vector3 chasePosition = chaseFocus - orbitForward * _taxiRideChaseDistance + Vector3.UnitZ * _taxiRideChaseHeight;
        ApplyDirectionalRideCamera(chasePosition, lookForward);
    }

    private void ApplyDirectionalRideCamera(Vector3 position, Vector3 forward)
    {
        if (forward.LengthSquared() <= 0.0001f)
            return;

        Vector3 direction = Vector3.Normalize(forward);

        _camera.Position = position;
        _camera.Yaw = MathF.Atan2(direction.Y, direction.X) * 180f / MathF.PI;

        float horizontalLength = MathF.Sqrt(direction.X * direction.X + direction.Y * direction.Y);
        _camera.Pitch = Math.Clamp(MathF.Atan2(direction.Z, MathF.Max(0.0001f, horizontalLength)) * 180f / MathF.PI, -89f, 89f);
    }

    private static Vector3 GetDirectionFromYawPitch(float yawDegrees, float pitchDegrees)
    {
        float yawRadians = yawDegrees * MathF.PI / 180f;
        float pitchRadians = pitchDegrees * MathF.PI / 180f;
        float cosPitch = MathF.Cos(pitchRadians);
        return Vector3.Normalize(new Vector3(
            cosPitch * MathF.Cos(yawRadians),
            cosPitch * MathF.Sin(yawRadians),
            MathF.Sin(pitchRadians)));
    }

    private unsafe bool TryCaptureFramebufferToPng(string outputPath, bool includeUi)
    {
        try
        {
            if (!TryGetCaptureRegion(includeUi, out int readX, out int readY, out int width, out int height))
            {
                ViewerLog.Error(ViewerLog.Category.Export,
                    $"[Capture] No valid capture region for {(includeUi ? "with-ui" : "scene-only")} request. Window={_window.FramebufferSize.X}x{_window.FramebufferSize.Y}");
                return false;
            }

            byte[] pixels = new byte[width * height * 4];
            if (!TryReadFramebufferRgba(readX, readY, width, height, pixels))
            {
                ViewerLog.Error(ViewerLog.Category.Export,
                    $"[Capture] Failed to read framebuffer RGBA: rect=({readX},{readY},{width},{height}) includeUi={includeUi}");
                return false;
            }

            ForceOpaqueAlpha(pixels);

            string? outputDirectory = Path.GetDirectoryName(outputPath);
            if (!string.IsNullOrWhiteSpace(outputDirectory))
                Directory.CreateDirectory(outputDirectory);

            using Image<Rgba32> image = SixLabors.ImageSharp.Image.LoadPixelData<Rgba32>(pixels, width, height);
            image.Mutate(x => x.Flip(FlipMode.Vertical));
            image.SaveAsPng(outputPath);
            return true;
        }
        catch (Exception ex)
        {
            _statusMessage = $"Capture failed: {ex.Message}";
            ViewerLog.Error(ViewerLog.Category.Export,
                $"[Capture] Exception saving PNG '{outputPath}': {ex}");
            return false;
        }
    }

    private bool TryGetCaptureRegion(bool includeUi, out int readX, out int readY, out int width, out int height)
    {
        readX = 0;
        readY = 0;

        if (!includeUi && TryGetSceneFramebufferViewport(out readX, out readY, out uint sceneWidth, out uint sceneHeight))
        {
            width = (int)sceneWidth;
            height = (int)sceneHeight;
            return width > 0 && height > 0;
        }

        Vector2D<int> framebufferSize = _window.FramebufferSize;
        width = framebufferSize.X;
        height = framebufferSize.Y;
        return width > 0 && height > 0;
    }

    private unsafe bool TryReadFramebufferRgba(int readX, int readY, int width, int height, byte[] pixels)
    {
        if (width <= 0 || height <= 0 || pixels.Length < width * height * 4)
            return false;

        fixed (byte* ptr = pixels)
        {
            _gl.ReadPixels(readX, readY, (uint)width, (uint)height, PixelFormat.Rgba, PixelType.UnsignedByte, ptr);
        }

        return true;
    }

    private static void ForceOpaqueAlpha(byte[] rgbaPixels)
    {
        for (int index = 3; index < rgbaPixels.Length; index += 4)
            rgbaPixels[index] = 255;
    }

    private static string BuildVideoCaptureFilter(int width, int height)
    {
        if ((width & 1) == 0 && (height & 1) == 0)
            return "vflip";

        return "vflip,pad=ceil(iw/2)*2:ceil(ih/2)*2";
    }

    private static void AppendVideoEncoderError(StringBuilder output, string? line)
    {
        if (string.IsNullOrWhiteSpace(line))
            return;

        lock (output)
        {
            if (output.Length >= 4096)
                return;

            if (output.Length > 0)
                output.AppendLine();

            output.Append(line.Trim());
        }
    }

    private static string GetVideoEncoderErrorSummary(ActiveVideoRecording recording)
    {
        lock (recording.EncoderErrorOutput)
        {
            if (recording.EncoderErrorOutput.Length == 0)
                return string.Empty;

            string[] lines = recording.EncoderErrorOutput
                .ToString()
                .Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

            if (lines.Length == 0)
                return string.Empty;

            return string.Join(" | ", lines.TakeLast(Math.Min(3, lines.Length)));
        }
    }

    private void LoadCameraShotPoints()
    {
        try
        {
            _cameraShotPoints.Clear();
            if (!File.Exists(CameraShotPointsPath))
                return;

            string json = File.ReadAllText(CameraShotPointsPath);
            CameraShotPointDocument? doc = JsonSerializer.Deserialize<CameraShotPointDocument>(json);
            if (doc?.Shots == null)
                return;

            _cameraShotPoints.AddRange(doc.Shots);
            _selectedCameraShotIndex = Math.Clamp(_selectedCameraShotIndex, -1, _cameraShotPoints.Count - 1);
        }
        catch (Exception ex)
        {
            _statusMessage = $"Failed to load shot points: {ex.Message}";
        }
    }

    private void SaveCameraShotPoints()
    {
        try
        {
            Directory.CreateDirectory(SettingsDir);
            CameraShotPointDocument doc = new()
            {
                Shots = _cameraShotPoints,
            };

            string json = JsonSerializer.Serialize(doc, new JsonSerializerOptions
            {
                WriteIndented = true,
            });

            File.WriteAllText(CameraShotPointsPath, json);
        }
        catch (Exception ex)
        {
            _statusMessage = $"Failed to save shot points: {ex.Message}";
        }
    }

    private string GetCurrentCaptureMapName()
    {
        if (_terrainManager != null && !string.IsNullOrWhiteSpace(_terrainManager.MapName))
            return _terrainManager.MapName;

        if (_selectedMapForPreview?.Name is string selectedMapName && !string.IsNullOrWhiteSpace(selectedMapName))
            return selectedMapName;

        if (!string.IsNullOrWhiteSpace(_lastWorldSceneWdtPath))
            return Path.GetFileNameWithoutExtension(_lastWorldSceneWdtPath);

        return "standalone";
    }

    private string GetCurrentCaptureBuildVersion()
    {
        return string.IsNullOrWhiteSpace(_dbcBuild)
            ? "unknown_build"
            : _dbcBuild;
    }

    private static string MakeSafePathSegment(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return "unnamed";

        Span<char> invalid = stackalloc char[]
        {
            '<', '>', ':', '"', '/', '\\', '|', '?', '*'
        };

        char[] chars = value.Trim().ToCharArray();
        for (int i = 0; i < chars.Length; i++)
        {
            if (char.IsControl(chars[i]))
            {
                chars[i] = '_';
                continue;
            }

            for (int j = 0; j < invalid.Length; j++)
            {
                if (chars[i] == invalid[j])
                {
                    chars[i] = '_';
                    break;
                }
            }
        }

        string cleaned = new string(chars).Trim();
        return string.IsNullOrWhiteSpace(cleaned) ? "unnamed" : cleaned;
    }
}
