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
