namespace MdxViewer.DataSources;

/// <summary>
/// Abstraction for reading WoW game files from different sources
/// (loose files, MPQ archives, CASC storage).
/// </summary>
public interface IDataSource : IDisposable
{
    string Name { get; }
    bool IsLoaded { get; }
    
    /// <summary>
    /// Check if a virtual path exists in this data source.
    /// </summary>
    bool FileExists(string virtualPath);
    
    /// <summary>
    /// Read file bytes by virtual path (e.g. "World\Maps\...").
    /// </summary>
    byte[]? ReadFile(string virtualPath);

    /// <summary>
    /// Resolve a virtual path to a writable loose-file path when one exists.
    /// Returns false for archive-backed files with no loose source on disk.
    /// </summary>
    bool TryResolveWritablePath(string virtualPath, out string? fullPath);
    
    /// <summary>
    /// Get a filtered list of known file paths matching a pattern.
    /// Used for the file browser panel.
    /// </summary>
    IReadOnlyList<string> GetFileList(string? extensionFilter = null);
}
