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
    /// Get a filtered list of known file paths matching a pattern.
    /// Used for the file browser panel.
    /// </summary>
    IReadOnlyList<string> GetFileList(string? extensionFilter = null);
}
