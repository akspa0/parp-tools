namespace WowViewer.Core.IO.Files;

public sealed record ArchiveCatalogBootstrapResult(
    IReadOnlyList<string> InternalFiles,
    IReadOnlyList<string> KnownFiles,
    IReadOnlyList<string> ExternalListfileEntries)
{
    public IReadOnlyList<string> AllFiles { get; } = BuildAllFiles(InternalFiles, KnownFiles, ExternalListfileEntries);

    private static IReadOnlyList<string> BuildAllFiles(
        IReadOnlyList<string> internalFiles,
        IReadOnlyList<string> knownFiles,
        IReadOnlyList<string> externalListfileEntries)
    {
        HashSet<string> files = new(StringComparer.OrdinalIgnoreCase);

        foreach (string file in internalFiles)
            files.Add(file);

        foreach (string file in knownFiles)
            files.Add(file);

        foreach (string file in externalListfileEntries)
            files.Add(file);

        return files.OrderBy(static file => file, StringComparer.OrdinalIgnoreCase).ToArray();
    }
}

public static class ArchiveCatalogBootstrapper
{
    public static ArchiveCatalogBootstrapResult Bootstrap(
        IArchiveCatalog archiveCatalog,
        IEnumerable<string> archiveRoots,
        string? listfilePath = null)
    {
        ArgumentNullException.ThrowIfNull(archiveCatalog);
        ArgumentNullException.ThrowIfNull(archiveRoots);

        string[] roots = archiveRoots.Where(static path => !string.IsNullOrWhiteSpace(path)).ToArray();
        archiveCatalog.LoadArchives(roots);

        IReadOnlyList<string> internalFiles = archiveCatalog.ExtractInternalListfiles();
        IReadOnlyList<string> knownFiles = archiveCatalog.GetAllKnownFiles();

        IReadOnlyList<string> externalEntries = Array.Empty<string>();
        if (!string.IsNullOrWhiteSpace(listfilePath) && File.Exists(listfilePath))
        {
            archiveCatalog.LoadListfile(listfilePath);
            externalEntries = ParseExternalListfileLines(File.ReadLines(listfilePath));
        }

        return new ArchiveCatalogBootstrapResult(internalFiles, knownFiles, externalEntries);
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