namespace WowViewer.Core.IO.Files;

public interface IArchiveReader
{
    bool FileExists(string virtualPath);

    byte[]? ReadFile(string virtualPath);
}

public interface IArchiveCatalog : IArchiveReader, IDisposable
{
    void LoadArchives(IEnumerable<string> searchPaths);

    void LoadListfile(string path);

    void LoadListfileEntries(IEnumerable<string> entries);

    IReadOnlyList<string> ExtractInternalListfiles();

    IReadOnlyList<string> GetAllKnownFiles();

    /// <summary>
    /// Enumerates distinct raw copies of a virtual file across loaded archives in ascending
    /// priority order (lowest-priority base archives first). Used for patch-artifact base
    /// resolution where the base copy must be matched by content hash rather than priority.
    /// Default: the single highest-priority copy from <see cref="IArchiveReader.ReadFile"/>.
    /// </summary>
    IEnumerable<byte[]?> ReadFileCopiesLowestFirst(string virtualPath)
    {
        byte[]? data = ReadFile(virtualPath);
        return data is null ? [] : [data];
    }
}

/// <summary>
/// Optional provenance surface for archive-backed diagnostics. Implementations that cannot expose
/// an individual archive may omit this interface; callers must retain an explicit unknown state.
/// </summary>
public interface IArchiveFileSourceResolver
{
    bool TryResolveFileSource(string virtualPath, out string sourcePath);
}

public interface IArchiveCatalogFactory
{
    IArchiveCatalog Create();
}
