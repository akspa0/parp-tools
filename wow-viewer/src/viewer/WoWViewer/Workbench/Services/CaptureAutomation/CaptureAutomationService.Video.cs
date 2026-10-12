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
using WowViewer.Core.Runtime.PromoVideo;
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
        var request = new RecordingRequest
        {
            SourceKind = RecordingSourceKind.Manual,
            Label = label,
            IncludeUi = includeUi,
            Fps = _videoCaptureFps,
            ContainerIndex = _videoCaptureContainerIndex,
        };

        return _host.RecordingCoordinator.TryStartRecording(request, out _);
    }

    internal void StopVideoRecording(string? statusOverride = null)
    {
        _host.RecordingCoordinator.StopRecording(statusOverride);
    }

    internal void CaptureVideoFrameIfNeeded(bool includeUi, double dt)
    {
        _host.RecordingCoordinator.CaptureVideoFrameIfNeeded(includeUi, dt);
    }


    internal void StopTaxiRideCamera(string? statusMessage = null)
    {
        _taxiRideCameraEnabled = false;
        _taxiRideCameraRouteId = -1;
        _taxiRideCameraScene?.TaxiActors.ActiveTaxiRideRouteId = -1;
        if (_worldScene != null)
            _worldScene.TaxiActors.ActiveTaxiRideRouteId = -1;
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

        if (_worldScene.TaxiActors.GetTaxiRoute(_taxiRideCameraRouteId) == null)
        {
            StopTaxiRideCamera("Ride camera detached because its taxi route is no longer loaded.");
            return;
        }

        long now = Stopwatch.GetTimestamp();
        float deltaSeconds = _lastTaxiRideCameraTick == 0
            ? 0f
            : (float)((now - _lastTaxiRideCameraTick) / (double)Stopwatch.Frequency);
        _lastTaxiRideCameraTick = now;

        if (!_worldScene.TaxiActors.TryGetTaxiActorPose(_taxiRideCameraRouteId, out TaxiActorPose pose))
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
