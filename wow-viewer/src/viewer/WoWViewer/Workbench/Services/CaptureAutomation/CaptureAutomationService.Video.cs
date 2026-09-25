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

// CaptureAutomationService: video recording, taxi-ride camera, framebuffer readback/PNG capture, camera shot-point persistence and capture naming.
// CaptureAutomationService: members moved from ViewerApp_CaptureAutomation.cs; this file keeps that file's using directives so every
// name in the moved code resolves exactly as it did there.
internal sealed partial class CaptureAutomationService
{

    internal bool TryStartCurrentViewVideoRecording(bool includeUi, string? label = null)
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

        // 069 Phase 7: if archeology playback to video is enabled, start playback.
        // Track ownership so a failed recording start never stops a playback
        // session the operator started independently.
        bool startedArcheologyPlayback = false;
        if (_archeologyApplyToVideoRecording && !_archeologyPlaybackActive)
        {
            _archaeologyPanel.StartArcheologyPlayback();
            startedArcheologyPlayback = _archeologyPlaybackActive;
        }

        VideoEncoderResolution encoderResolution = VideoEncoderExecutableResolver.Resolve(_videoEncoderExecutable, AppContext.BaseDirectory);

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

        try
        {
            string? outputDirectory = Path.GetDirectoryName(outputPath);
            if (!string.IsNullOrWhiteSpace(outputDirectory))
                Directory.CreateDirectory(outputDirectory);

            var startInfo = new ProcessStartInfo
            {
                FileName = encoderResolution.Executable,
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
                ApplyArcheologyPlayback = _archeologyApplyToVideoRecording,
                StartedArcheologyPlayback = startedArcheologyPlayback,
            };

            _statusMessage = $"Started video recording with {encoderResolution.DisplayName}: {outputPath}";
            return true;
        }
        catch (Win32Exception ex)
        {
            if (startedArcheologyPlayback && _archeologyPlaybackActive)
                _archaeologyPanel.StopArcheologyPlayback(restoreRange: true);
            _statusMessage = VideoEncoderExecutableResolver.BuildUnavailableMessage(encoderResolution, ex.Message);
            return false;
        }
        catch (Exception ex)
        {
            if (startedArcheologyPlayback && _archeologyPlaybackActive)
                _archaeologyPanel.StopArcheologyPlayback(restoreRange: true);
            _statusMessage = $"Failed to start video recording: {ex.Message}";
            return false;
        }
    }

    internal void StopVideoRecording(string? statusOverride = null)
    {
        if (_activeVideoRecording == null)
            return;

        ActiveVideoRecording recording = _activeVideoRecording;
        _activeVideoRecording = null;

        // 069 Phase 7: stop archeology playback if it was started for video.
        if (recording.StartedArcheologyPlayback && _archeologyPlaybackActive)
            _archaeologyPanel.StopArcheologyPlayback(restoreRange: true);

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

        if (recording.RestoreUiChromeAfterMarketingTour)
            _hideUiChrome = recording.PreviousHideUiChrome;

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

    internal void CaptureVideoFrameIfNeeded(bool includeUi, double dt)
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

    internal void StopTaxiRideCamera(string? statusMessage = null)
    {
        _taxiRideCameraEnabled = false;
        _taxiRideCameraRouteId = -1;
        _taxiRideCameraScene?.ActiveTaxiRideRouteId = -1;
        if (_worldScene != null)
            _worldScene.ActiveTaxiRideRouteId = -1;
        _taxiRideCameraScene = null;
        _taxiRideFreeLookYawOffset = 0f;
        _taxiRideFreeLookPitchOffset = 0f;
        _taxiRideCameraPoseInitialized = false;
        _lastTaxiRideCameraTick = 0;
        if (!string.IsNullOrWhiteSpace(statusMessage))
            _statusMessage = statusMessage;
    }

    internal void AdjustTaxiRideFreeLook(float deltaYawDegrees, float deltaPitchDegrees)
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

    internal void UpdateTaxiRideCamera()
    {
        if (!_taxiRideCameraEnabled)
            return;

        if (_worldScene == null)
        {
            StopTaxiRideCamera("Ride camera detached because the world scene is no longer active.");
            return;
        }

        if (!ReferenceEquals(_taxiRideCameraScene, _worldScene))
        {
            StopTaxiRideCamera("Ride camera detached because the world scene changed.");
            return;
        }

        if (_worldScene.GetTaxiRoute(_taxiRideCameraRouteId) == null)
        {
            StopTaxiRideCamera("Ride camera detached because its taxi route is no longer loaded.");
            return;
        }

        long now = Stopwatch.GetTimestamp();
        float deltaSeconds = _lastTaxiRideCameraTick == 0
            ? 0f
            : (float)((now - _lastTaxiRideCameraTick) / (double)Stopwatch.Frequency);
        _lastTaxiRideCameraTick = now;

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
        Vector3 desiredPosition;
        if (_taxiRideCameraMode == TaxiRideCameraMode.Cockpit)
        {
            desiredPosition = pose.Position + Vector3.UnitZ * (_taxiRideCockpitHeight * scale);
        }
        else
        {
            Vector3 chaseFocus = pose.Position + Vector3.UnitZ * Math.Max(6f, _taxiRideCockpitHeight * 0.65f * scale);
            desiredPosition = chaseFocus - orbitForward * _taxiRideChaseDistance + Vector3.UnitZ * _taxiRideChaseHeight;
        }

        if (!_taxiRideCameraPoseInitialized)
        {
            // The model may become available several frames after the route
            // starts. Begin from the current camera pose so that the first
            // resolved actor does not teleport the view across the map.
            _taxiRideCameraSmoothedPosition = _camera.Position;
            _taxiRideCameraSmoothedForward = _camera.Forward;
            _taxiRideCameraPoseInitialized = true;
        }

        float blend = 1f - MathF.Exp(-TaxiRideCameraSmoothingHz * Math.Clamp(deltaSeconds, 0f, 0.25f));
        if (blend <= 0f)
            blend = 0.2f;
        _taxiRideCameraSmoothedPosition = Vector3.Lerp(_taxiRideCameraSmoothedPosition, desiredPosition, blend);
        Vector3 blendedForward = Vector3.Lerp(_taxiRideCameraSmoothedForward, lookForward, blend);
        if (blendedForward.LengthSquared() > 0.0001f)
            _taxiRideCameraSmoothedForward = Vector3.Normalize(blendedForward);

        ApplyDirectionalRideCamera(_taxiRideCameraSmoothedPosition, _taxiRideCameraSmoothedForward);
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

        if (!includeUi && _shellLayout.TryGetSceneFramebufferViewport(out readX, out readY, out uint sceneWidth, out uint sceneHeight))
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

    internal void LoadCameraShotPoints()
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

    internal string GetCurrentCaptureMapName()
    {
        if (_terrainManager != null && !string.IsNullOrWhiteSpace(_terrainManager.MapName))
            return _terrainManager.MapName;

        if (_selectedMapForPreview?.Name is string selectedMapName && !string.IsNullOrWhiteSpace(selectedMapName))
            return selectedMapName;

        if (!string.IsNullOrWhiteSpace(_lastWorldSceneWdtPath))
            return Path.GetFileNameWithoutExtension(_lastWorldSceneWdtPath);

        return "standalone";
    }

    internal string GetCurrentCaptureBuildVersion()
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
