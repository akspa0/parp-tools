namespace WowViewer.Core.IO.Files;

public static class ArchiveVirtualFileReader
{
    public static byte[] ReadVirtualFile(
        string virtualPath,
        IEnumerable<string> archiveRoots,
        ArchiveCatalogBootstrapOptions? bootstrapOptions,
        IArchiveCatalogFactory? archiveCatalogFactory = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(virtualPath);
        ArgumentNullException.ThrowIfNull(archiveRoots);

        archiveCatalogFactory ??= new MpqArchiveCatalogFactory();

        using IArchiveCatalog archiveCatalog = archiveCatalogFactory.Create();
        ArchiveCatalogBootstrapper.Bootstrap(archiveCatalog, archiveRoots, bootstrapOptions);

        return archiveCatalog.ReadFile(virtualPath)
            ?? throw new FileNotFoundException($"Could not read virtual archive file '{virtualPath}'.", virtualPath);
    }

    public static byte[] ReadVirtualFile(
        string virtualPath,
        IEnumerable<string> archiveRoots,
        string? listfilePath = null,
        IArchiveCatalogFactory? archiveCatalogFactory = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(virtualPath);
        ArgumentNullException.ThrowIfNull(archiveRoots);
		return ReadVirtualFile(
			virtualPath,
			archiveRoots,
			new ArchiveCatalogBootstrapOptions(ExternalListfilePath: listfilePath),
			archiveCatalogFactory);
    }
}