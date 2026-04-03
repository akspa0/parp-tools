namespace WowViewer.Core.IO.Files;

public sealed record ArchiveCatalogBootstrapResult(
    IReadOnlyList<string> InternalFiles,
    IReadOnlyList<string> KnownFiles,
    IReadOnlyList<string> ExternalListfileEntries,
    IReadOnlyList<string> CachedListfileEntries,
    string? ListfileCacheKey,
    string? ListfileCachePath)
{
    public IReadOnlyList<string> AllFiles { get; } = BuildAllFiles(InternalFiles, KnownFiles, ExternalListfileEntries, CachedListfileEntries);

    private static IReadOnlyList<string> BuildAllFiles(
        IReadOnlyList<string> internalFiles,
        IReadOnlyList<string> knownFiles,
        IReadOnlyList<string> externalListfileEntries,
        IReadOnlyList<string> cachedListfileEntries)
    {
        HashSet<string> files = new(StringComparer.OrdinalIgnoreCase);

        foreach (string file in internalFiles)
            files.Add(file);

        foreach (string file in knownFiles)
            files.Add(file);

        foreach (string file in externalListfileEntries)
            files.Add(file);

        foreach (string file in cachedListfileEntries)
            files.Add(file);

        return files.OrderBy(static file => file, StringComparer.OrdinalIgnoreCase).ToArray();
    }
}

public sealed record ArchiveCatalogBootstrapOptions(
    string? ExternalListfilePath = null,
    string? ListfileCacheKey = null,
    string? ListfileCacheDirectoryPath = null,
    bool LoadCachedEntries = true,
    bool PersistListfileCache = true);

public static class ArchiveCatalogBootstrapper
{
    public static ArchiveCatalogBootstrapResult Bootstrap(
        IArchiveCatalog archiveCatalog,
        IEnumerable<string> archiveRoots,
        string? listfilePath = null)
    {
        return Bootstrap(
            archiveCatalog,
            archiveRoots,
            new ArchiveCatalogBootstrapOptions(ExternalListfilePath: listfilePath));
    }

    public static ArchiveCatalogBootstrapResult Bootstrap(
        IArchiveCatalog archiveCatalog,
        IEnumerable<string> archiveRoots,
        ArchiveCatalogBootstrapOptions? options)
    {
        ArgumentNullException.ThrowIfNull(archiveCatalog);
        ArgumentNullException.ThrowIfNull(archiveRoots);

        options ??= new ArchiveCatalogBootstrapOptions();
        string[] roots = archiveRoots.Where(static path => !string.IsNullOrWhiteSpace(path)).ToArray();
        archiveCatalog.LoadArchives(roots);

        IReadOnlyList<string> cachedEntries = Array.Empty<string>();
        string? cachePath = null;
        if (options.LoadCachedEntries &&
            !string.IsNullOrWhiteSpace(options.ListfileCacheKey) &&
            !string.IsNullOrWhiteSpace(options.ListfileCacheDirectoryPath))
        {
            ArchiveListfileCacheManifest? manifest = ArchiveListfileCache.TryRead(options.ListfileCacheDirectoryPath, options.ListfileCacheKey);
            if (manifest is not null)
            {
                cachedEntries = manifest.AllEntries;
                archiveCatalog.LoadListfileEntries(cachedEntries);
                cachePath = ArchiveListfileCache.GetCachePath(options.ListfileCacheDirectoryPath, options.ListfileCacheKey);
            }
        }

        IReadOnlyList<string> internalFiles = archiveCatalog.ExtractInternalListfiles();
        if (internalFiles.Count > 0)
            archiveCatalog.LoadListfileEntries(internalFiles);

        IReadOnlyList<string> externalEntries = Array.Empty<string>();
        if (!string.IsNullOrWhiteSpace(options.ExternalListfilePath) && File.Exists(options.ExternalListfilePath))
        {
            archiveCatalog.LoadListfile(options.ExternalListfilePath);
            externalEntries = ParseExternalListfileLines(File.ReadLines(options.ExternalListfilePath));
            if (externalEntries.Count > 0)
                archiveCatalog.LoadListfileEntries(externalEntries);
        }

        IReadOnlyList<string> knownFiles = archiveCatalog.GetAllKnownFiles();

        if (options.PersistListfileCache &&
            !string.IsNullOrWhiteSpace(options.ListfileCacheKey) &&
            !string.IsNullOrWhiteSpace(options.ListfileCacheDirectoryPath))
        {
            cachePath = ArchiveListfileCache.Write(
                options.ListfileCacheDirectoryPath,
                options.ListfileCacheKey,
                roots,
                internalFiles,
                externalEntries);
        }

        return new ArchiveCatalogBootstrapResult(
            internalFiles,
            knownFiles,
            externalEntries,
            cachedEntries,
            options.ListfileCacheKey,
            cachePath);
    }

    public static IReadOnlyList<string> ParseExternalListfileLines(IEnumerable<string> lines)
    {
        ArgumentNullException.ThrowIfNull(lines);

        List<string> entries = [];
        foreach (string line in lines)
        {
            string entry = line.Trim();
            if (string.IsNullOrEmpty(entry))
                continue;

            if (entry.Contains(';', StringComparison.Ordinal))
            {
                string[] parts = entry.Split(';', 2, StringSplitOptions.None);
                if (parts.Length > 1)
                    entry = parts[1].Trim();
            }

            if (!string.IsNullOrWhiteSpace(entry))
                entries.Add(entry);
        }

        return entries;
    }
}