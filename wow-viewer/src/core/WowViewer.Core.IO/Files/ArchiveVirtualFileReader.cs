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

        archiveCatalogFactory ??= new MpqArchiveCatalogFactory();

        using IArchiveCatalog archiveCatalog = archiveCatalogFactory.Create();
        ArchiveCatalogBootstrapper.Bootstrap(archiveCatalog, archiveRootArray, bootstrapOptions);

        string normalizedVirtualPath = virtualPath.Replace('/', '\\');
        byte[]? archiveBytes = archiveCatalog.ReadFile(virtualPath)
            ?? archiveCatalog.ReadFile(normalizedVirtualPath)
            ?? archiveCatalog.ReadFile(normalizedVirtualPath.Replace('\\', '/'));

        if (archiveBytes is { Length: > 0 })
            return archiveBytes;

        byte[]? diskBytes = TryReadVirtualFileFromDisk(normalizedVirtualPath, archiveRootArray);
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

    private static byte[]? TryReadVirtualFileFromDisk(string virtualPath, IEnumerable<string> archiveRoots)
    {
        foreach (string archiveRoot in archiveRoots)
        {
            if (string.IsNullOrWhiteSpace(archiveRoot))
                continue;

            foreach (string candidatePath in EnumerateDiskCandidates(archiveRoot, virtualPath))
            {
                byte[]? bytes = AlphaArchiveReader.ReadWithMpqFallback(candidatePath);
                if (bytes is { Length: > 0 })
                    return bytes;
            }
        }

        return null;
    }

    private static IEnumerable<string> EnumerateDiskCandidates(string archiveRoot, string virtualPath)
    {
        string fullArchiveRoot;
        try
        {
            fullArchiveRoot = Path.GetFullPath(archiveRoot);
        }
        catch
        {
            yield break;
        }

        string relativePath = virtualPath.Replace('/', Path.DirectorySeparatorChar).Replace('\\', Path.DirectorySeparatorChar);
        HashSet<string> yielded = new(StringComparer.OrdinalIgnoreCase);

        foreach (string candidate in new[]
        {
            Path.Combine(fullArchiveRoot, relativePath),
            Path.Combine(fullArchiveRoot, "Data", relativePath),
        })
        {
            if (yielded.Add(candidate))
                yield return candidate;
        }
    }
}