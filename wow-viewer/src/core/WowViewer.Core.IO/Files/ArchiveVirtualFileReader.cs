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
        string[] archiveRootArray = archiveRoots.ToArray();

        ArchiveCatalogSession session = ArchiveCatalogSessionCache.GetOrCreate(
            archiveRootArray,
            bootstrapOptions,
            archiveCatalogFactory);

        byte[]? archiveBytes = session.ReadFile(virtualPath);

        if (archiveBytes is { Length: > 0 })
            return archiveBytes;

        byte[]? diskBytes = session.TryReadFileFromDisk(virtualPath);
        if (diskBytes is { Length: > 0 })
            return diskBytes;

        throw new FileNotFoundException($"Could not read virtual archive file '{virtualPath}'.", virtualPath);
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