using System.ComponentModel;
using System.Diagnostics;

namespace WoWViewer.Capture;

/// <summary>
/// Resolves the encoder used by viewer video capture without making a developer's PATH
/// an implicit release dependency. A compatible <c>ffmpeg.exe</c> beside the viewer
/// executable wins over the conventional PATH fallback.
/// </summary>
internal static class VideoEncoderExecutableResolver
{
    private const string DefaultExecutable = "ffmpeg";
    private const string AppLocalExecutableName = "ffmpeg.exe";

    internal static VideoEncoderResolution Resolve(string? configuredExecutable, string appBaseDirectory)
    {
        string configured = NormalizeConfiguredExecutable(configuredExecutable);
        string appLocalCandidate = Path.Combine(appBaseDirectory, AppLocalExecutableName);

        if (string.IsNullOrWhiteSpace(configured)
            || string.Equals(configured, DefaultExecutable, StringComparison.OrdinalIgnoreCase)
            || string.Equals(configured, AppLocalExecutableName, StringComparison.OrdinalIgnoreCase))
        {
            if (File.Exists(appLocalCandidate))
                return new VideoEncoderResolution(appLocalCandidate, VideoEncoderSource.AppLocal, appLocalCandidate);

            return new VideoEncoderResolution(DefaultExecutable, VideoEncoderSource.Path, appLocalCandidate);
        }

        return new VideoEncoderResolution(configured, VideoEncoderSource.Configured, appLocalCandidate);
    }

    internal static VideoEncoderProbeResult Probe(VideoEncoderResolution resolution)
    {
        try
        {
            var startInfo = new ProcessStartInfo
            {
                FileName = resolution.Executable,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
            };
            startInfo.ArgumentList.Add("-hide_banner");
            startInfo.ArgumentList.Add("-h");
            startInfo.ArgumentList.Add("encoder=libx264");

            using Process process = Process.Start(startInfo)
                ?? throw new InvalidOperationException("The video encoder process did not start.");
            Task<string> standardOutput = process.StandardOutput.ReadToEndAsync();
            Task<string> standardError = process.StandardError.ReadToEndAsync();

            if (!process.WaitForExit(5000))
            {
                process.Kill(entireProcessTree: true);
                process.WaitForExit();
                Task.WaitAll(standardOutput, standardError);
                return new VideoEncoderProbeResult(false, "ffmpeg did not respond within five seconds.");
            }

            Task.WaitAll(standardOutput, standardError);
            if (process.ExitCode == 0)
                return new VideoEncoderProbeResult(true, $"{resolution.DisplayName} is ready with libx264.");

            string diagnostic = SummarizeDiagnostic(standardError.Result, standardOutput.Result);
            return new VideoEncoderProbeResult(
                false,
                string.IsNullOrWhiteSpace(diagnostic)
                    ? $"ffmpeg exited with code {process.ExitCode} while checking libx264."
                    : $"ffmpeg cannot provide libx264: {diagnostic}");
        }
        catch (Win32Exception ex)
        {
            return new VideoEncoderProbeResult(false, BuildUnavailableMessage(resolution, ex.Message));
        }
        catch (Exception ex)
        {
            return new VideoEncoderProbeResult(false, $"ffmpeg verification failed: {ex.Message}");
        }
    }

    internal static string BuildUnavailableMessage(VideoEncoderResolution resolution, string detail)
    {
        if (resolution.Source == VideoEncoderSource.Path)
        {
            return $"ffmpeg was not found. Place a compatible ffmpeg.exe beside the viewer at '{resolution.AppLocalCandidate}', "
                + "or configure its full path in Capture Automation. "
                + $"System detail: {detail}";
        }

        return $"The configured ffmpeg executable '{resolution.Executable}' was unavailable: {detail}";
    }

    internal static string NormalizeConfiguredExecutable(string? configuredExecutable)
    {
        string configured = configuredExecutable?.Trim() ?? string.Empty;
        if (configured.Length >= 2
            && configured[0] == '"'
            && configured[^1] == '"')
        {
            configured = configured[1..^1].Trim();
        }

        return configured;
    }

    private static string SummarizeDiagnostic(string standardError, string standardOutput)
    {
        string text = string.IsNullOrWhiteSpace(standardError) ? standardOutput : standardError;
        string[] lines = text.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        return lines.Length == 0
            ? string.Empty
            : string.Join(" | ", lines.TakeLast(Math.Min(2, lines.Length)));
    }
}

internal enum VideoEncoderSource
{
    AppLocal,
    Configured,
    Path,
}

internal sealed record VideoEncoderResolution(string Executable, VideoEncoderSource Source, string AppLocalCandidate)
{
    internal string DisplayName => Source switch
    {
        VideoEncoderSource.AppLocal => "Bundled ffmpeg",
        VideoEncoderSource.Configured => "Configured ffmpeg",
        _ => "PATH ffmpeg",
    };
}

internal sealed record VideoEncoderProbeResult(bool IsReady, string Message);
