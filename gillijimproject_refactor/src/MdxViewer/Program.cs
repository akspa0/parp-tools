using MdxViewer;
using MdxViewer.Logging;

/// <summary>
/// WoW Model Viewer entry point.
/// Launches with ImGui UI, menu bar, file browser, and 3D viewport.
/// Supports: MDX (Alpha 0.5.3), WMO (v14/v17), GLB export.
/// Data sources: loose files, MPQ archives.
/// Usage: MdxViewer [--verbose] [file ...]
/// </summary>
class Program
{
    static void Main(string[] args)
    {
        var filteredArgs = args.Where(a => !a.Equals("--verbose", StringComparison.OrdinalIgnoreCase)).ToArray();
        if (filteredArgs.Length != args.Length)
            ViewerLog.Verbose = true;

        using var app = new ViewerApp();
        app.Run(filteredArgs.Length > 0 ? filteredArgs : null);
    }
}
