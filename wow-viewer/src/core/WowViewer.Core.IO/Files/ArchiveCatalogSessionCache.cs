using System.Runtime.CompilerServices;

namespace WowViewer.Core.IO.Files;

public sealed class ArchiveCatalogSession : IDisposable
{
    private readonly string[] _archiveRoots;

    internal ArchiveCatalogSession(
        IArchiveCatalog archiveCatalog,
        IEnumerable<string> archiveRoots,
        ArchiveCatalogBootstrapOptions bootstrapOptions,
        int bootstrapCount)
    {
        ArchiveCatalog = archiveCatalog;
        _archiveRoots = archiveRoots.ToArray();
        BootstrapOptions = bootstrapOptions;
        BootstrapCount = bootstrapCount;
    }

    public IArchiveCatalog ArchiveCatalog { get; }

    public IReadOnlyList<string> ArchiveRoots => _archiveRoots;

    public ArchiveCatalogBootstrapOptions BootstrapOptions { get; }

    public int BootstrapCount { get; }

    public byte[]? ReadFile(string virtualPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(virtualPath);

        string normalizedVirtualPath = NormalizeVirtualPath(virtualPath);
        return ArchiveCatalog.ReadFile(virtualPath)
            ?? ArchiveCatalog.ReadFile(normalizedVirtualPath)
            ?? ArchiveCatalog.ReadFile(normalizedVirtualPath.Replace('\\', '/'));
    }

    public byte[]? TryReadFileFromDisk(string virtualPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(virtualPath);

        string normalizedVirtualPath = NormalizeVirtualPath(virtualPath);
        foreach (string archiveRoot in _archiveRoots)
        {
            if (string.IsNullOrWhiteSpace(archiveRoot))
                continue;

            foreach (string candidatePath in EnumerateDiskCandidates(archiveRoot, normalizedVirtualPath))
            {
                byte[]? bytes = AlphaArchiveReader.ReadWithMpqFallback(candidatePath);
                if (bytes is { Length: > 0 })
                    return bytes;
            }
        }

        return null;
    }

    public void Dispose()
    {
        ArchiveCatalog.Dispose();
    }

    internal static string NormalizeVirtualPath(string path)
    {
        return path.Replace('/', '\\');
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

public static class ArchiveCatalogSessionCache
{
    private sealed record CacheKey(
        string RootsSignature,
        string ExternalListfilePath,
        string ListfileCacheKey,
        string ListfileCacheDirectoryPath,
        bool LoadCachedEntries,
        bool PersistListfileCache,
        string FactoryToken);

    private static readonly object SyncRoot = new();
    private static readonly Dictionary<CacheKey, ArchiveCatalogSession> Sessions = [];
    private static int _bootstrapCount;

    public static ArchiveCatalogSession GetOrCreate(
        IEnumerable<string> archiveRoots,
        ArchiveCatalogBootstrapOptions? bootstrapOptions = null,
        IArchiveCatalogFactory? archiveCatalogFactory = null)
    {
        ArgumentNullException.ThrowIfNull(archiveRoots);

        string[] normalizedRoots = NormalizeRoots(archiveRoots);
        ArchiveCatalogBootstrapOptions normalizedOptions = NormalizeOptions(bootstrapOptions);
        CacheKey key = CreateCacheKey(normalizedRoots, normalizedOptions, archiveCatalogFactory);

        lock (SyncRoot)
        {
            if (Sessions.TryGetValue(key, out ArchiveCatalogSession? existing))
                return existing;

            archiveCatalogFactory ??= new MpqArchiveCatalogFactory();
            IArchiveCatalog archiveCatalog = archiveCatalogFactory.Create();
            ArchiveCatalogBootstrapper.Bootstrap(archiveCatalog, normalizedRoots, normalizedOptions);

            ArchiveCatalogSession created = new(
                archiveCatalog,
                normalizedRoots,
                normalizedOptions,
                ++_bootstrapCount);

            Sessions[key] = created;
            return created;
        }
    }

    public static void Invalidate(
        IEnumerable<string> archiveRoots,
        ArchiveCatalogBootstrapOptions? bootstrapOptions = null,
        IArchiveCatalogFactory? archiveCatalogFactory = null)
    {
        ArgumentNullException.ThrowIfNull(archiveRoots);

        string[] normalizedRoots = NormalizeRoots(archiveRoots);
        ArchiveCatalogBootstrapOptions normalizedOptions = NormalizeOptions(bootstrapOptions);
        CacheKey key = CreateCacheKey(normalizedRoots, normalizedOptions, archiveCatalogFactory);

        ArchiveCatalogSession? removed = null;
        lock (SyncRoot)
        {
            if (Sessions.Remove(key, out ArchiveCatalogSession? existing))
                removed = existing;
        }

        removed?.Dispose();
    }

    public static void InvalidateAll()
    {
        List<ArchiveCatalogSession> removed;
        lock (SyncRoot)
        {
            removed = Sessions.Values.ToList();
            Sessions.Clear();
        }

        foreach (ArchiveCatalogSession session in removed)
            session.Dispose();
    }

    private static CacheKey CreateCacheKey(
        IReadOnlyList<string> archiveRoots,
        ArchiveCatalogBootstrapOptions options,
        IArchiveCatalogFactory? archiveCatalogFactory)
    {
        string factoryToken = archiveCatalogFactory is null
            ? typeof(MpqArchiveCatalogFactory).FullName ?? nameof(MpqArchiveCatalogFactory)
            : $"{archiveCatalogFactory.GetType().FullName}:{RuntimeHelpers.GetHashCode(archiveCatalogFactory)}";

        return new CacheKey(
            string.Join("|", archiveRoots),
            NormalizeComparablePath(options.ExternalListfilePath),
            options.ListfileCacheKey?.Trim() ?? string.Empty,
            NormalizeComparablePath(options.ListfileCacheDirectoryPath),
            options.LoadCachedEntries,
            options.PersistListfileCache,
            factoryToken);
    }

    private static string[] NormalizeRoots(IEnumerable<string> archiveRoots)
    {
        return archiveRoots
            .Where(static root => !string.IsNullOrWhiteSpace(root))
            .Select(NormalizePath)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }

    private static ArchiveCatalogBootstrapOptions NormalizeOptions(ArchiveCatalogBootstrapOptions? bootstrapOptions)
    {
        ArchiveCatalogBootstrapOptions options = bootstrapOptions ?? new ArchiveCatalogBootstrapOptions();
        return options with
        {
            ExternalListfilePath = NormalizeOptionalPath(options.ExternalListfilePath),
            ListfileCacheDirectoryPath = NormalizeOptionalPath(options.ListfileCacheDirectoryPath),
            ListfileCacheKey = options.ListfileCacheKey?.Trim(),
        };
    }

    private static string NormalizeComparablePath(string? path)
    {
        return NormalizeOptionalPath(path) ?? string.Empty;
    }

    private static string? NormalizeOptionalPath(string? path)
    {
        return string.IsNullOrWhiteSpace(path) ? null : NormalizePath(path);
    }

    private static string NormalizePath(string path)
    {
        string trimmed = path.Trim();
        try
        {
            return Path.GetFullPath(trimmed).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        }
        catch
        {
            return trimmed.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        }
    }
}