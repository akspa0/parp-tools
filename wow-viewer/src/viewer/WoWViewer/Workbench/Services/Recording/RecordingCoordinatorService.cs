using System.ComponentModel;
using System.Diagnostics;
using System.Text;
using Silk.NET.Maths;
using Silk.NET.OpenGL;
using WoWViewer.Capture;
using WoWViewer.Logging;
using WowViewer.Core.Runtime.PromoVideo;

namespace WoWViewer;

/// <summary>
/// Centralized coordinator for all video recording operations (manual viewports,
/// cinematic camera paths, taxi flight routes, and automated headless sessions).
/// </summary>
internal sealed class RecordingCoordinatorService
{
    private readonly IViewerAppHost _host;

    private static readonly string[] VideoContainerExtensions = { ".mp4", ".mov" };
    private static readonly string[] VideoContainerLabels = { "MP4 (H.264)", "MOV (H.264)" };

    private ActiveRecordingSession? _activeSession;
    private RecordingSessionSummary? _lastCompletedSession;

    internal RecordingCoordinatorService(IViewerAppHost host)
    {
        _host = host;
    }

    public bool IsRecording => _activeSession != null;
    public ActiveRecordingSession? ActiveSession => _activeSession;
    public RecordingSessionSummary? LastCompletedSession => _lastCompletedSession;
    public FeatureTourPresentation? ActiveTourPresentation => _activeSession?.TourAttempt?.ActivePresentation;

    public static IReadOnlyList<string> SupportedContainerExtensions => VideoContainerExtensions;
    public static IReadOnlyList<string> SupportedContainerLabels => VideoContainerLabels;

    /// <summary>
    /// Attempts to start a new video recording session.
    /// </summary>
    public bool TryStartRecording(RecordingRequest request, out string? errorMessage)
    {
        if (_activeSession != null)
        {
            errorMessage = "A video recording is already in progress.";
            _host.StatusMessage = errorMessage;
            return false;
        }

        if (!TryGetCaptureRegion(request.IncludeUi, out _, out _, out int width, out int height))
        {
            errorMessage = request.IncludeUi
                ? "Unable to resolve the current framebuffer size for video capture."
                : "Unable to resolve the scene viewport for no-UI video capture.";
            _host.StatusMessage = errorMessage;
            return false;
        }

        if (width <= 0 || height <= 0)
        {
            errorMessage = "Video capture dimensions were invalid.";
            _host.StatusMessage = errorMessage;
            return false;
        }

        bool startedArcheologyPlayback = false;
        if (_host.ArcheologyApplyToVideoRecording && !_host.ArcheologyPlaybackActive)
        {
            _host.ArchaeologyPanel.StartArcheologyPlayback();
            startedArcheologyPlayback = _host.ArcheologyPlaybackActive;
        }

        string encoderExe = string.IsNullOrWhiteSpace(_host.VideoEncoderExecutable) ? "ffmpeg" : _host.VideoEncoderExecutable;
        VideoEncoderResolution encoderResolution = VideoEncoderExecutableResolver.Resolve(encoderExe, AppContext.BaseDirectory);

        int containerIdx = Math.Clamp(request.ContainerIndex, 0, VideoContainerExtensions.Length - 1);
        string extension = VideoContainerExtensions[containerIdx];

        string outputPath;
        if (!string.IsNullOrWhiteSpace(request.OutputPathOverride))
        {
            outputPath = Path.GetFullPath(request.OutputPathOverride);
        }
        else
        {
            string safeMap = MakeSafePathSegment(_host.GetCurrentCaptureMapName());
            string safeBuild = MakeSafePathSegment(_host.GetCurrentCaptureBuildVersion());
            string safeLabel = MakeSafePathSegment(string.IsNullOrWhiteSpace(request.Label) ? "recording" : request.Label);
            string captureMode = request.IncludeUi ? "with_ui" : "no_ui";
            string baseDir = string.IsNullOrWhiteSpace(_host.CaptureOutputDir)
                ? Path.Combine(ViewerApp.OutputDir, "captures")
                : _host.CaptureOutputDir;

            outputPath = Path.Combine(
                baseDir,
                safeMap,
                safeBuild,
                $"{DateTime.UtcNow:yyyyMMdd_HHmmssfff}_{safeLabel}_{captureMode}{extension}");
        }

        try
        {
            string? outputDir = Path.GetDirectoryName(outputPath);
            if (!string.IsNullOrWhiteSpace(outputDir))
                Directory.CreateDirectory(outputDir);

            var startInfo = new ProcessStartInfo
            {
                FileName = encoderResolution.Executable,
                UseShellExecute = false,
                RedirectStandardInput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
                WorkingDirectory = Environment.CurrentDirectory,
            };

            int effectiveFps = Math.Clamp(request.Fps > 0 ? request.Fps : _host.VideoCaptureFps, 12, 60);

            startInfo.ArgumentList.Add("-y");
            startInfo.ArgumentList.Add("-f");
            startInfo.ArgumentList.Add("rawvideo");
            startInfo.ArgumentList.Add("-pixel_format");
            startInfo.ArgumentList.Add("rgba");
            startInfo.ArgumentList.Add("-video_size");
            startInfo.ArgumentList.Add($"{width}x{height}");
            startInfo.ArgumentList.Add("-framerate");
            startInfo.ArgumentList.Add(effectiveFps.ToString());
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

            StringBuilder encoderErrorOutput = new();
            Process process = Process.Start(startInfo)
                ?? throw new InvalidOperationException("ffmpeg process did not start.");

            process.ErrorDataReceived += (_, args) => AppendVideoEncoderError(encoderErrorOutput, args.Data);
            process.BeginErrorReadLine();

            PromoTourAttempt? tourAttempt = request.TourAttempt;
            if (tourAttempt == null && request.TourRecipe != null)
            {
                double durationEstimate = request.MaxDurationSeconds ?? 60.0;
                var tourStart = PromoTourAttempt.TryStart(request.TourRecipe, durationEstimate);
                if (tourStart.IsStarted)
                {
                    tourAttempt = tourStart.Attempt;
                }
            }

            if (tourAttempt != null)
            {
                tourAttempt.Advance(0);
            }

            float initialTravel = 0f;
            float initialLength = 0f;
            if (request.SourceKind == RecordingSourceKind.TaxiRoute && request.TaxiRouteId >= 0 && _host.WorldScene != null)
            {
                if (_host.WorldScene.TaxiActors.TryGetTaxiRouteProgress(request.TaxiRouteId, out float tDist, out float tLen))
                {
                    initialTravel = tDist;
                    initialLength = tLen;
                }
            }

            _activeSession = new ActiveRecordingSession
            {
                Request = request,
                EncoderProcess = process,
                EncoderInput = process.StandardInput.BaseStream,
                EncoderErrorOutput = encoderErrorOutput,
                OutputPath = outputPath,
                IncludeUi = request.IncludeUi,
                Width = width,
                Height = height,
                FrameIntervalSeconds = 1.0 / effectiveFps,
                FrameAccumulatorSeconds = 0.0,
                FrameBuffer = new byte[width * height * 4],
                ApplyArcheologyPlayback = _host.ArcheologyApplyToVideoRecording,
                StartedArcheologyPlayback = startedArcheologyPlayback,
                TourAttempt = tourAttempt,
                RestoreUiChromeOnStop = request.RestoreUiChromeOnStop,
                PreviousHideUiChrome = request.PreviousHideUiChrome,
                RouteStartTravelDistance = initialTravel,
                RouteTotalLength = initialLength,
            };

            errorMessage = null;
            _host.StatusMessage = $"Started video recording ({request.SourceKind}): {outputPath}";
            ViewerLog.Important(ViewerLog.Category.Export, $"[Recording] Started session: {outputPath}");
            return true;
        }
        catch (Win32Exception ex)
        {
            if (startedArcheologyPlayback && _host.ArcheologyPlaybackActive)
                _host.ArchaeologyPanel.StopArcheologyPlayback(restoreRange: true);

            errorMessage = VideoEncoderExecutableResolver.BuildUnavailableMessage(encoderResolution, ex.Message);
            _host.StatusMessage = errorMessage;
            ViewerLog.Error(ViewerLog.Category.Export, $"[Recording] Win32Exception starting ffmpeg: {ex.Message}");
            return false;
        }
        catch (Exception ex)
        {
            if (startedArcheologyPlayback && _host.ArcheologyPlaybackActive)
                _host.ArchaeologyPanel.StopArcheologyPlayback(restoreRange: true);

            errorMessage = $"Failed to start video recording: {ex.Message}";
            _host.StatusMessage = errorMessage;
            ViewerLog.Error(ViewerLog.Category.Export, $"[Recording] Exception starting ffmpeg: {ex}");
            return false;
        }
    }

    /// <summary>
    /// Gracefully stops the active recording session, disposes encoder resources,
    /// restores UI/archaeology state, and creates a non-null summary.
    /// </summary>
    public void StopRecording(string? statusOverride = null)
    {
        if (_activeSession == null)
            return;

        ActiveRecordingSession session = _activeSession;
        _activeSession = null;

        if (session.StartedArcheologyPlayback && _host.ArcheologyPlaybackActive)
            _host.ArchaeologyPanel.StopArcheologyPlayback(restoreRange: true);

        bool success = false;
        string statusMessage = statusOverride ?? $"Saved video: {session.OutputPath}";

        try
        {
            session.EncoderInput.Flush();
        }
        catch
        {
        }

        try
        {
            session.EncoderInput.Dispose();
        }
        catch
        {
        }

        try
        {
            if (!session.EncoderProcess.WaitForExit(10000))
                session.EncoderProcess.Kill(entireProcessTree: true);

            success = session.EncoderProcess.ExitCode == 0;
            if (!success && statusOverride == null)
                statusMessage = $"Video encode failed for {session.OutputPath} (exit {session.EncoderProcess.ExitCode}).";

            string encoderError = GetVideoEncoderErrorSummary(session);
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
                session.EncoderProcess.CancelErrorRead();
            }
            catch
            {
            }

            session.EncoderProcess.Dispose();
        }

        if (session.RestoreUiChromeOnStop)
            _host.HideUiChrome = session.PreviousHideUiChrome;

        if (session.Request.DetachCameraOnStop && session.Request.SourceKind == RecordingSourceKind.TaxiRoute)
        {
            _host.StopTaxiRideCamera("Taxi route recording ended.");
        }

        if (!success && statusOverride == null && File.Exists(session.OutputPath))
        {
            try
            {
                File.Delete(session.OutputPath);
            }
            catch
            {
            }
        }

        _lastCompletedSession = new RecordingSessionSummary(
            session.Request.SourceKind,
            session.OutputPath,
            session.ElapsedSeconds,
            session.RecordedFrames,
            success,
            statusMessage,
            DateTime.UtcNow);

        _host.StatusMessage = statusMessage;
        ViewerLog.Important(ViewerLog.Category.Export, $"[Recording] Stopped session: {statusMessage}");

        try
        {
            session.Request.OnCompleted?.Invoke(_lastCompletedSession);
        }
        catch (Exception ex)
        {
            ViewerLog.Error(ViewerLog.Category.Export, $"[Recording] Error executing OnCompleted callback: {ex.Message}");
        }

        if (session.Request.ExitAfterRecording)
        {
            ViewerLog.Important(ViewerLog.Category.Export, "[Recording] ExitAfterRecording is set. Closing viewer window.");
            _host.Window.Close();
        }
    }

    /// <summary>
    /// Called every frame from ViewerApp.OnUpdate. Handles time progression,
    /// tour presentation advancing, and auto-stop triggers.
    /// </summary>
    public void Update(double dt)
    {
        if (_activeSession == null)
            return;

        ActiveRecordingSession session = _activeSession;
        session.ElapsedSeconds += Math.Max(0.0, dt);

        if (session.TourAttempt != null)
        {
            session.TourAttempt.Advance(session.ElapsedSeconds);
        }

        if (session.Request.MaxDurationSeconds is double maxDuration && session.ElapsedSeconds >= maxDuration)
        {
            StopRecording($"Video recording reached max duration ({maxDuration:F1}s): {session.OutputPath}");
            return;
        }

        // Auto-stop detection for TaxiRoute flights
        if (session.Request.SourceKind == RecordingSourceKind.TaxiRoute && session.Request.AutoStopOnRouteArrival)
        {
            int routeId = session.Request.TaxiRouteId;
            if (routeId >= 0 && _host.WorldScene != null)
            {
                if (_host.WorldScene.TaxiActors.TryGetTaxiRouteProgress(routeId, out float currentTravel, out float routeLen))
                {
                    // Detect when actor has traveled a substantial distance and wrapped or reached destination.
                    float deltaTravel = currentTravel - session.RouteStartTravelDistance;
                    if (deltaTravel < 0)
                        deltaTravel += routeLen;

                    if (deltaTravel > routeLen * 0.25f)
                        session.HasTraveledSignificantDistance = true;

                    // If it has traveled nearly the full route (or wrapped past end back to start)
                    if (session.HasTraveledSignificantDistance && (deltaTravel >= routeLen * 0.98f || deltaTravel < 10f))
                    {
                        StopRecording($"Taxi flight arrived at destination ({session.ElapsedSeconds:F1}s): {session.OutputPath}");
                        return;
                    }
                }
            }
        }
    }

    /// <summary>
    /// Captures a frame from the OpenGL framebuffer if enough time has passed.
    /// Called during rendering (once for scene-only before ImGui, and once for with-UI after ImGui).
    /// </summary>
    public void CaptureVideoFrameIfNeeded(bool includeUi, double dt)
    {
        if (_activeSession == null || _activeSession.IncludeUi != includeUi)
            return;

        ActiveRecordingSession session = _activeSession;
        session.FrameAccumulatorSeconds += Math.Max(0.0, dt);
        if (session.FrameAccumulatorSeconds + 1e-6 < session.FrameIntervalSeconds)
            return;

        if (!TryGetCaptureRegion(includeUi, out int readX, out int readY, out int width, out int height))
        {
            StopRecording(includeUi
                ? "Video recording stopped because the framebuffer was unavailable."
                : "Video recording stopped because the scene viewport was unavailable.");
            return;
        }

        if (width != session.Width || height != session.Height)
        {
            StopRecording("Video recording stopped because the capture size changed during recording.");
            return;
        }

        if (session.EncoderProcess.HasExited)
        {
            StopRecording("Video recording stopped because ffmpeg exited unexpectedly.");
            return;
        }

        int framesToWrite = Math.Max(1, (int)(session.FrameAccumulatorSeconds / session.FrameIntervalSeconds));
        session.FrameAccumulatorSeconds -= framesToWrite * session.FrameIntervalSeconds;

        byte[] pixels = session.FrameBuffer.Length == session.Width * session.Height * 4
            ? session.FrameBuffer
            : new byte[session.Width * session.Height * 4];
        session.FrameBuffer = pixels;

        if (!TryReadFramebufferRgba(readX, readY, session.Width, session.Height, pixels))
        {
            StopRecording("Video recording stopped because framebuffer capture failed.");
            return;
        }

        try
        {
            for (int i = 0; i < framesToWrite; i++)
            {
                session.EncoderInput.Write(pixels, 0, pixels.Length);
                session.RecordedFrames++;
            }
        }
        catch (Exception ex)
        {
            StopRecording($"Video recording stopped because ffmpeg write failed: {ex.Message}");
        }
    }

    private bool TryGetCaptureRegion(bool includeUi, out int readX, out int readY, out int width, out int height)
    {
        readX = 0;
        readY = 0;

        if (!includeUi && _host.ShellLayout.TryGetSceneFramebufferViewport(out readX, out readY, out uint sceneWidth, out uint sceneHeight))
        {
            width = (int)sceneWidth;
            height = (int)sceneHeight;
            return width > 0 && height > 0;
        }

        Vector2D<int> framebufferSize = _host.Window.FramebufferSize;
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
            _host.Gl.ReadPixels(readX, readY, (uint)width, (uint)height, PixelFormat.Rgba, PixelType.UnsignedByte, ptr);
        }

        return true;
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

    private static string GetVideoEncoderErrorSummary(ActiveRecordingSession session)
    {
        lock (session.EncoderErrorOutput)
        {
            if (session.EncoderErrorOutput.Length == 0)
                return string.Empty;

            string[] lines = session.EncoderErrorOutput
                .ToString()
                .Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

            if (lines.Length == 0)
                return string.Empty;

            return string.Join(" | ", lines.TakeLast(Math.Min(3, lines.Length)));
        }
    }

    public static string MakeSafePathSegment(string value)
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
