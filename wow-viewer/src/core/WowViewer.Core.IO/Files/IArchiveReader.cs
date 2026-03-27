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

    IReadOnlyList<string> ExtractInternalListfiles();

    IReadOnlyList<string> GetAllKnownFiles();
}

public interface IArchiveCatalogFactory
{
    IArchiveCatalog Create();
}