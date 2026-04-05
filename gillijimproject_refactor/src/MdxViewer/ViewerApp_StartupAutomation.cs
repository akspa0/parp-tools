using System.Globalization;

namespace MdxViewer;

public partial class ViewerApp
{
    private sealed class StartupAutomationRequest
    {
        public string? GamePath { get; init; }
        public string? BuildVersion { get; init; }
        public string? LooseMapOverlayPath { get; init; }
        public string? WorldPath { get; init; }
        public string? CaptureShotName { get; init; }
        public string? CaptureOutputDir { get; init; }
        public bool CaptureIncludeUi { get; init; }
        public bool ExitAfterCapture { get; init; }
    }

    private void ApplyStartupAutomation(string[]? initialArgs)
    {
        StartupAutomationRequest request = ParseStartupAutomationRequest(initialArgs, out string? legacyPath);

        if (!string.IsNullOrWhiteSpace(request.GamePath))
        {
            if (!Directory.Exists(request.GamePath))
            {
                _statusMessage = $"Startup game path does not exist: {request.GamePath}";
                return;
            }

            LoadMpqDataSource(request.GamePath, null, request.BuildVersion);
        }

        if (!string.IsNullOrWhiteSpace(request.LooseMapOverlayPath))
        {
            if (!Directory.Exists(request.LooseMapOverlayPath))
            {
                _statusMessage = $"Startup loose overlay path does not exist: {request.LooseMapOverlayPath}";
                return;
            }

            AttachLooseMapOverlay(request.LooseMapOverlayPath);
        }

        string? startupTarget = request.WorldPath;
        if (string.IsNullOrWhiteSpace(startupTarget))
            startupTarget = legacyPath;

        if (!string.IsNullOrWhiteSpace(startupTarget))
            LoadStartupTarget(startupTarget);

        if (!string.IsNullOrWhiteSpace(request.CaptureOutputDir))
            _captureOutputDir = Path.GetFullPath(request.CaptureOutputDir);

        if (!string.IsNullOrWhiteSpace(request.CaptureShotName))
            QueueNamedStartupCapture(request.CaptureShotName, request.CaptureIncludeUi, request.ExitAfterCapture);
    }

    private StartupAutomationRequest ParseStartupAutomationRequest(string[]? initialArgs, out string? legacyPath)
    {
        legacyPath = null;
        if (initialArgs == null || initialArgs.Length == 0)
            return new StartupAutomationRequest();

        string? gamePath = null;
        string? buildVersion = null;
        string? looseMapOverlayPath = null;
        string? worldPath = null;
        string? captureShotName = null;
        string? captureOutputDir = null;
        bool captureIncludeUi = false;
        bool exitAfterCapture = false;

        for (int index = 0; index < initialArgs.Length; index++)
        {
            string arg = initialArgs[index];
            switch (arg.ToLowerInvariant())
            {
                case "--game-path":
                    if (!TryReadStartupOptionValue(initialArgs, ref index, arg, out gamePath))
                        return new StartupAutomationRequest();
                    break;

                case "--build":
                    if (!TryReadStartupOptionValue(initialArgs, ref index, arg, out buildVersion))
                        return new StartupAutomationRequest();
                    break;

                case "--loose-map-overlay":
                    if (!TryReadStartupOptionValue(initialArgs, ref index, arg, out looseMapOverlayPath))
                        return new StartupAutomationRequest();
                    break;

                case "--world":
                    if (!TryReadStartupOptionValue(initialArgs, ref index, arg, out worldPath))
                        return new StartupAutomationRequest();
                    break;

                case "--capture-shot":
                    if (!TryReadStartupOptionValue(initialArgs, ref index, arg, out captureShotName))
                        return new StartupAutomationRequest();
                    break;

                case "--capture-output":
                    if (!TryReadStartupOptionValue(initialArgs, ref index, arg, out captureOutputDir))
                        return new StartupAutomationRequest();
                    break;

                case "--capture-with-ui":
                    captureIncludeUi = true;
                    break;

                case "--capture-no-ui":
                    captureIncludeUi = false;
                    break;

                case "--exit-after-capture":
                    exitAfterCapture = true;
                    break;

                default:
                    if (arg.StartsWith("--", StringComparison.Ordinal))
                    {
                        _statusMessage = $"Unknown startup option: {arg}";
                        return new StartupAutomationRequest();
                    }

                    legacyPath ??= arg;
                    break;
            }
        }

        return new StartupAutomationRequest
        {
            GamePath = NormalizeOptionalPath(gamePath),
            BuildVersion = NormalizeOptionalValue(buildVersion),
            LooseMapOverlayPath = NormalizeOptionalPath(looseMapOverlayPath),
            WorldPath = NormalizeOptionalValue(worldPath),
            CaptureShotName = NormalizeOptionalValue(captureShotName),
            CaptureOutputDir = NormalizeOptionalPath(captureOutputDir),
            CaptureIncludeUi = captureIncludeUi,
            ExitAfterCapture = exitAfterCapture,
        };
    }

    private bool TryReadStartupOptionValue(string[] args, ref int index, string optionName, out string? value)
    {
        if (index + 1 >= args.Length)
        {
            value = null;
            _statusMessage = $"Missing value for startup option {optionName}.";
            return false;
        }

        index++;
        value = args[index];
        return true;
    }

    private void LoadStartupTarget(string startupTarget)
    {
        if (File.Exists(startupTarget))
        {
            LoadFileFromDisk(Path.GetFullPath(startupTarget));
            return;
        }

        if (_dataSource != null)
        {
            LoadFileFromDataSource(startupTarget);
            return;
        }

        _statusMessage = $"Startup world/file path was not found on disk and no data source is loaded: {startupTarget}";
    }

    private void QueueNamedStartupCapture(string shotName, bool includeUi, bool exitAfterCapture)
    {
        CameraShotPoint? shot = _cameraShotPoints.FirstOrDefault(candidate =>
            string.Equals(candidate.Name, shotName, StringComparison.OrdinalIgnoreCase));

        if (shot == null)
        {
            _statusMessage = $"Startup capture shot was not found: {shotName}";
            return;
        }

        EnqueueShotCapture(shot, includeUi, exitAfterCapture);
    }

    private static string? NormalizeOptionalPath(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return null;

        return Path.GetFullPath(value);
    }

    private static string? NormalizeOptionalValue(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return null;

        return value.Trim();
    }
}