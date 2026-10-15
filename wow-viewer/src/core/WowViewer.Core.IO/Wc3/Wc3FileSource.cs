using WowViewer.Core.IO.Files;

namespace WowViewer.Core.IO.Wc3;

/// <summary>A read-only effective-file browser; no WC3 engine patch-order claim.</summary>
public sealed class Wc3FileSource : IDisposable
{
    private readonly object _gate = new();
    private readonly Dictionary<string, string> _loose;
    private readonly Dictionary<string, string> _storedNames;
    private readonly MpqArchiveCatalog? _catalog;
    private readonly string[] _archivePaths;
    private bool _disposed;

    private Wc3FileSource(string root, Dictionary<string, string> loose,
        Dictionary<string, string> storedNames, MpqArchiveCatalog? catalog, string[] archivePaths, string summary)
    {
        Root = root;
        _loose = loose;
        _storedNames = storedNames;
        _catalog = catalog;
        _archivePaths = archivePaths;
        Files = Array.AsReadOnly(storedNames.Values.OrderBy(static name => name, StringComparer.OrdinalIgnoreCase).ToArray());
        Summary = summary;
    }

    public string Root { get; }
    public IReadOnlyList<string> Files { get; }
    public string Summary { get; }

    public static Wc3FileSource Open(string root, string? listfilePath = null)
    {
        string fullRoot = Wc3PathSafety.FullPath(root);
        Wc3PathSafety.CheckAncestors(fullRoot);
        if (!Directory.Exists(fullRoot))
            throw new DirectoryNotFoundException($"WC3 source directory '{fullRoot}' does not exist. Select an extracted-data or MPQ directory.");
        fullRoot = Wc3PathGuard.CanonicalPath(fullRoot);
        using Wc3PathGuard sourceGuard = Wc3PathGuard.Pin(fullRoot);

        string? listfile = null;
        if (listfilePath is not null)
        {
            listfile = Wc3PathSafety.FullPath(listfilePath);
            Wc3PathSafety.CheckAncestors(listfile);
            if (!File.Exists(listfile))
                throw new FileNotFoundException("The selected WC3 listfile is missing. Select an existing local listfile, or omit it to use internal MPQ names/loose files.", listfile);
            sourceGuard.Add(listfile);
        }

        Dictionary<string, string> loose = new(StringComparer.OrdinalIgnoreCase);
        Dictionary<string, string> names = new(StringComparer.OrdinalIgnoreCase);
        List<string> archives = [];
        List<string> reparsePaths = [];
        int unsafeNames = 0;
        Stack<string> directories = new();
        directories.Push(fullRoot);
        while (directories.TryPop(out string? directory))
        {
            sourceGuard.Add(directory);
            foreach (string path in Directory.EnumerateFileSystemEntries(directory).Order(StringComparer.Ordinal))
            {
                FileAttributes attributes = File.GetAttributes(path);
                if ((attributes & FileAttributes.ReparsePoint) != 0)
                {
                    reparsePaths.Add(path);
                    continue;
                }
                if ((attributes & FileAttributes.Directory) != 0)
                {
                    directories.Push(path);
                    continue;
                }

                string stored = Path.GetRelativePath(fullRoot, path);
                string key;
                try { key = Wc3PathSafety.VirtualKey(stored); }
                catch (ArgumentException) { unsafeNames++; continue; }
                if (!loose.TryAdd(key, path))
                    throw new IOException($"WC3 loose files have ambiguous case/separator names: '{stored}'. Choose a source without duplicate virtual names.");
                names.Add(key, stored);
                if (Path.GetExtension(path).Equals(".mpq", StringComparison.OrdinalIgnoreCase))
                {
                    sourceGuard.Add(path);
                    archives.Add(path);
                }
            }
        }

        MpqArchiveCatalog? catalog = null;
        int confirmedArchiveNames = 0;
        int excludedArchiveDirectories = 0;
        try
        {
            if (archives.Count > 0)
            {
                // The shared catalog accepts directory roots, not an exact-file allowlist.
                // Do not let its implicit Data/World scans traverse skipped reparse points.
                string[] archiveRoots = archives.Select(static path => Path.GetDirectoryName(path)!)
                    .Distinct(StringComparer.OrdinalIgnoreCase)
                    .Where(directory =>
                    {
                        bool unsafeScan = reparsePaths.Any(path =>
                            Wc3PathSafety.Contains(Path.Combine(directory, "Data"), path) ||
                            Wc3PathSafety.Contains(Path.Combine(directory, "World"), path) ||
                            (Path.GetDirectoryName(path)!.Equals(directory, StringComparison.OrdinalIgnoreCase) &&
                             Path.GetExtension(path).Equals(".mpq", StringComparison.OrdinalIgnoreCase)));
                        if (unsafeScan) excludedArchiveDirectories++;
                        return !unsafeScan;
                    }).ToArray();
                if (archiveRoots.Length > 0)
                {
                    foreach (string archive in archives)
                        Wc3PathSafety.CheckAncestors(archive);
                    catalog = new MpqArchiveCatalog();
                    ArchiveCatalogBootstrapResult bootstrap = ArchiveCatalogBootstrapper.Bootstrap(catalog, archiveRoots,
                        new ArchiveCatalogBootstrapOptions(ExternalListfilePath: listfile,
                            LoadCachedEntries: false, PersistListfileCache: false));
                    HashSet<string> confirmed = new(StringComparer.OrdinalIgnoreCase);
                    foreach (string stored in bootstrap.AllFiles)
                    {
                        string key;
                        try { key = Wc3PathSafety.VirtualKey(stored); }
                        catch (ArgumentException) { unsafeNames++; continue; }
                        if (!catalog.FileExists(key))
                            continue;
                        if (catalog.TryResolveFileSource(key, out string archivePath))
                        {
                            if (!Wc3PathSafety.Contains(fullRoot, Wc3PathSafety.FullPath(archivePath)))
                                continue;
                            Wc3PathSafety.CheckAncestors(archivePath);
                        }
                        if (confirmed.Add(key)) confirmedArchiveNames++;
                        names.TryAdd(key, stored); // Actual loose stored name and bytes win.
                    }
                }
            }

            if (names.Count == 0)
                throw new NotSupportedException("No safe loose files or confirmed MPQ names were found. Select extracted WC3 data or a classic MPQ install with an existing WC3 listfile; CASC/Reforged layouts are not supported by this source.");

            string summary = $"{fullRoot}: {loose.Count} loose files; {archives.Count} MPQ containers; {confirmedArchiveNames} confirmed archive names. Loose files override archive bytes. ";
            summary += archives.Count == 0
                ? "Loose extracted-data mode; no archive bootstrap needed."
                : "Existing catalog ordering is unchanged (not WC3 engine patch semantics). Name coverage may be incomplete; select a local WC3 listfile for additional names. " +
                  (listfile is null ? "Internal listfiles only; no WoW default listfile." : $"Supplemental listfile: {listfile}.");
            if (archives.Count > 0 && confirmedArchiveNames == 0)
                summary += " No archive payload names were confirmed: archives may be unsupported/corrupt or lack usable listfiles. Raw loose containers remain extractable.";
            if (reparsePaths.Count > 0 || unsafeNames > 0 || excludedArchiveDirectories > 0)
                summary += $" Skipped {reparsePaths.Count} reparse paths, {unsafeNames} unsafe names, and {excludedArchiveDirectories} archive directories whose shared scans could follow reparse paths.";
            return new Wc3FileSource(fullRoot, loose, names, catalog, archives.ToArray(), summary);
        }
        catch
        {
            catalog?.Dispose();
            throw;
        }
    }

    public byte[] ReadFile(string virtualPath)
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            string key = Wc3PathSafety.VirtualKey(virtualPath);
            if (!_storedNames.ContainsKey(key))
                throw new FileNotFoundException($"'{virtualPath}' was not found in this WC3 source. Reopen the source after adding files/listfile names.", virtualPath);

            using Wc3PathGuard readGuard = Wc3PathGuard.Pin(Root);
            if (_loose.TryGetValue(key, out string? diskPath))
            {
                readGuard.Add(diskPath);
                using FileStream stream = new(diskPath, FileMode.Open, FileAccess.Read, FileShare.Read);
                Wc3PathSafety.CheckAncestors(diskPath);
                if (stream.Length > Array.MaxLength)
                    throw new NotSupportedException("This file is too large for the byte-array file API. Use an external streaming extractor.");
                byte[] bytes = new byte[checked((int)stream.Length)];
                stream.ReadExactly(bytes);
                return bytes;
            }

            // A shared read can fall back to a lower-priority archive; validate and
            // pin every original container, not just the first resolved match.
            if (_catalog is not null)
                foreach (string path in _archivePaths) readGuard.Add(path);
            if (_catalog is not null && _catalog.TryResolveFileSource(key, out string archivePath))
            {
                if (!Wc3PathSafety.Contains(Root, Wc3PathSafety.FullPath(archivePath)))
                    throw new IOException("The resolved archive is outside the source directory.");
                Wc3PathSafety.CheckAncestors(archivePath);
                byte[]? bytes = _catalog.ReadFile(key);
                if (bytes is not null)
                    return bytes;
                throw new NotSupportedException($"The existing MPQ reader could not read '{virtualPath}' (unsupported/corrupt/deleted payload). Try an independently extracted loose file; no reader changes were applied.");
            }
            throw new FileNotFoundException($"'{virtualPath}' is no longer available in the selected source.", virtualPath);
        }
    }

    public string ExtractFile(string virtualPath, string outputRoot, CancellationToken cancellationToken = default)
    {
        lock (_gate)
        {
            cancellationToken.ThrowIfCancellationRequested();
            ObjectDisposedException.ThrowIf(_disposed, this);
            string key = Wc3PathSafety.VirtualKey(virtualPath);
            if (!_storedNames.TryGetValue(key, out string? stored))
                throw new FileNotFoundException("Select a file listed in the current WC3 source before extracting.", virtualPath);
            string output = Wc3PathSafety.FullPath(outputRoot);
            Wc3PathSafety.CheckAncestors(Root);
            Wc3PathSafety.CheckAncestors(output);
            output = Wc3PathGuard.CanonicalPath(output);
            using Wc3PathGuard outputGuard = Wc3PathGuard.Pin(output);
            if (Wc3PathSafety.Contains(Root, output))
                throw new IOException("Extraction into the source tree is not allowed. Select a separate output directory.");
            string finalPath = Wc3PathSafety.Combine(output, Wc3PathSafety.VirtualKey(stored));
            if (Wc3PathSafety.Contains(Root, finalPath))
                throw new IOException("The selected output file is inside the source tree. Select a separate output directory.");
            Wc3PathSafety.CheckAncestors(finalPath);
            Wc3PathSafety.RejectCaseCollision(finalPath);
            byte[] bytes = ReadFile(key);
            string parent = Path.GetDirectoryName(finalPath)!;
            Wc3PathSafety.CreateDirectories(parent, outputGuard);
            outputGuard.Add(parent);
            string partialPath = Path.Combine(parent, $".wc3-{Guid.NewGuid():N}.partial");
            bool created = false;
            try
            {
                Wc3PathSafety.CheckAncestors(finalPath);
                Wc3PathSafety.RejectCaseCollision(finalPath);
                using (FileStream stream = new(partialPath, FileMode.CreateNew, FileAccess.Write, FileShare.None))
                {
                    created = true;
                    Wc3PathSafety.CheckAncestors(partialPath);
                    const int blockSize = 64 * 1024;
                    for (int offset = 0; offset < bytes.Length; offset += blockSize)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        stream.Write(bytes.AsSpan(offset, Math.Min(blockSize, bytes.Length - offset)));
                    }
                    stream.Flush(flushToDisk: true);
                }
                cancellationToken.ThrowIfCancellationRequested();
                Wc3PathSafety.CheckAncestors(finalPath);
                Wc3PathSafety.RejectCaseCollision(finalPath);
                File.Move(partialPath, finalPath, overwrite: false);
                created = false;
                return finalPath;
            }
            finally
            {
                if (created)
                {
                    // Do not follow a newly replaced output ancestor during cleanup.
                    Wc3PathSafety.CheckAncestors(partialPath);
                    File.Delete(partialPath);
                }
            }
        }
    }

    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed) return;
            _disposed = true;
            _catalog?.Dispose();
        }
    }
}
