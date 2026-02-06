using MdxViewer;

/// <summary>
/// WoW Model Viewer entry point.
/// Launches with ImGui UI, menu bar, file browser, and 3D viewport.
/// Supports: MDX (Alpha 0.5.3), WMO (v14/v17), GLB export.
/// Data sources: loose files, MPQ archives.
/// </summary>
class Program
{
    static void Main(string[] args)
    {
        using var app = new ViewerApp();
        app.Run(args.Length > 0 ? args : null);
    }
}
