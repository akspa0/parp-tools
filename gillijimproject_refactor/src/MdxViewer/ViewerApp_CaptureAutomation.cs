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
using MdxViewer.Rendering;
using MdxViewer.Terrain;

namespace MdxViewer;

public partial class ViewerApp
{
    private static readonly string CameraShotPointsPath = Path.Combine(SettingsDir, "camera_shot_points.json");

    private readonly List<CameraShotPoint> _cameraShotPoints = new();
    private readonly Queue<PendingCaptureRequest> _captureQueue = new();
    private PendingCaptureRequest? _activeCaptureRequest;
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

    private void QueueCurrentCameraCapture(bool includeUi)
    {
        CameraShotPoint shot = CreateCameraShotPoint($"current_{DateTime.UtcNow:yyyyMMdd_HHmmss}");
        EnqueueShotCapture(shot, includeUi);
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
    {
        if (string.IsNullOrWhiteSpace(_captureOutputDir))
            _captureOutputDir = Path.Combine(OutputDir, "captures");

        string safeMap = MakeSafePathSegment(shot.MapName);
        string safeBuild = MakeSafePathSegment(shot.BuildVersion);
        string safeShotName = MakeSafePathSegment(shot.Name);
        string mode = includeUi ? "with_ui" : "no_ui";
        string fileName = $"{DateTime.UtcNow:yyyyMMdd_HHmmssfff}_{safeShotName}_{mode}.png";
        string outputPath = Path.Combine(_captureOutputDir, safeMap, safeBuild, fileName);

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
        });

        _statusMessage = $"Queued capture '{shot.Name}' ({mode}).";
    }

    private void PrepareNextCaptureRequest()
    {
        if (_activeCaptureRequest != null || _captureQueue.Count == 0)
            return;

        PendingCaptureRequest request = _captureQueue.Dequeue();
        _activeCaptureRequest = request;

        ApplyCameraShotPoint(request.Shot);
        request.Applied = true;
        _activeCaptureRequest = request;
    }

    private void CompleteCaptureIfReady(bool includeUi)
    {
        if (_activeCaptureRequest == null)
            return;

        PendingCaptureRequest request = _activeCaptureRequest;
        if (!request.Applied || request.IncludeUi != includeUi)
            return;

        bool ok = TryCaptureFramebufferToPng(request.OutputPath, includeUi);
        _activeCaptureRequest = null;

        _statusMessage = ok
            ? $"Captured shot: {request.OutputPath}"
            : $"Capture failed: {request.OutputPath}";

        if (request.ExitAfterCapture)
            _window.Close();
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
                return false;

            byte[] pixels = new byte[width * height * 4];
            if (!TryReadFramebufferRgba(readX, readY, width, height, pixels))
                return false;

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
