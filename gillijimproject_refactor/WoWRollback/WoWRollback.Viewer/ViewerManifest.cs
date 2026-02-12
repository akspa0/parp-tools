namespace WoWRollback.Viewer;

/// <summary>
/// Placeholder class for WoWRollback.Viewer project.
/// This project primarily serves to copy viewer assets to the output directory.
/// Future: Will contain plugin runtime and manifest builder.
/// </summary>
public static class ViewerManifest
{
    /// <summary>
    /// Gets the path to viewer assets relative to the assembly location.
    /// </summary>
    public static string AssetsPath => "assets";
    
    /// <summary>
    /// Version marker for tracking viewer asset changes.
    /// </summary>
    public const string Version = "1.0.0-phase1";
}
