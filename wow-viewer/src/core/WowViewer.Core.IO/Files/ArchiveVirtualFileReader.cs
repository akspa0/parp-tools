namespace WowViewer.Core.IO.Files;

public static class ArchiveVirtualFileReader
{
    public static byte[] ReadVirtualFile(
        string virtualPath,
        IEnumerable<string> archiveRoots,
        string? listfilePath = null,
        IArchiveCatalogFactory? archiveCatalogFactory = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(virtualPath);
        ArgumentNullException.ThrowIfNull(archiveRoots);

        archiveCatalogFactory ??= new MpqArchiveCatalogFactory();

        using IArchiveCatalog archiveCatalog = archiveCatalogFactory.Create();
        ArchiveCatalogBootstrapper.Bootstrap(archiveCatalog, archiveRoots, listfilePath);

        return archiveCatalog.ReadFile(virtualPath)
            ?? throw new FileNotFoundException($"Could not read virtual archive file '{virtualPath}'.", virtualPath);
    }
}