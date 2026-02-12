using MdxViewer;
using MdxViewer.Logging;
using MdxLTool.Formats.Mdx;

/// <summary>
/// WoW Model Viewer entry point.
/// Launches with ImGui UI, menu bar, file browser, and 3D viewport.
/// Supports: MDX (Alpha 0.5.3), WMO (v14/v17), GLB export.
/// Data sources: loose files, MPQ archives.
/// Usage: MdxViewer [--verbose] [--full-load|--partial-load] [file ...]
/// </summary>
class Program
{
    static void Main(string[] args)
    {
        bool verbose = args.Any(a => a.Equals("--verbose", StringComparison.OrdinalIgnoreCase));
        bool fullLoad = args.Any(a => a.Equals("--full-load", StringComparison.OrdinalIgnoreCase));
        // AOI streaming is the default; --full-load loads all tiles at startup
        var filteredArgs = args
            .Where(a => !a.Equals("--verbose", StringComparison.OrdinalIgnoreCase)
                     && !a.Equals("--full-load", StringComparison.OrdinalIgnoreCase)
                     && !a.Equals("--partial-load", StringComparison.OrdinalIgnoreCase))
                     // keep filtering both flags for backwards compat
            .ToArray();

        if (verbose)
        {
            ViewerLog.Verbose = true;
            MdxFile.Verbose = true;
        }

        using var app = new ViewerApp();
        app.FullLoadMode = fullLoad;
        app.Run(filteredArgs.Length > 0 ? filteredArgs : null);
    }
}
