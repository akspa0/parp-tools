using MdxViewer.Logging;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using WowViewer.Core.IO.Files;

namespace MdxViewer.DataSources;

public readonly record struct MpqDataSourceStats(
    long FileExistsRequests,
    long FileExistsCacheHits,
    long FileExistsLooseHits,
    long FileExistsAlphaHits,
    long FileExistsMpqHits,
    long FileExistsCanonicalHits,
    long FileExistsMisses,
    long ReadRequests,
    long ReadCacheHits,
    long ReadCacheMisses,
    long ReadLooseHits,
    long ReadAlphaHits,
    long ReadMpqHits,
    long ReadMisses,
    double AverageUncachedReadMs,
    long PrefetchRequests,
    long PrefetchCacheSkips,
    long PrefetchDuplicateSkips,
    long PrefetchEnqueued,
    long PrefetchCompleted,
    long PrefetchReadHits,
    long PrefetchReadMisses,
    double AveragePrefetchQueueMs,
    double AveragePrefetchReadMs,
    int ReadCacheEntryCount,
    long ReadCacheBytes,
    int PrefetchQueueDepth);

internal readonly record struct PrefetchRequest(string Path, long EnqueuedTimestamp);

internal enum MpqReadResolutionKind
{
    Loose,
    AlphaWrapper,
    Mpq,
    Miss,
}

/// <summary>
/// Data source backed by MPQ archives + loose files on disk.
/// Uses the shared archive-catalog boundary for MPQ access.
/// Builds file list from MPQ-internal (listfile) entries — accurate for any client version.
/// Supports Alpha 0.5.3, Classic, TBC, and WotLK 3.3.5 game folders.
/// </summary>
public class MpqDataSource : IDataSource
{
    private static readonly HashSet<string> IndexedLooseExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".mdl",
        ".mdx",
        ".wmo",
        ".m2",
        ".blp",
        ".skin",
        ".anim",
        ".dbc",
        ".wdt",
        ".adt",
        ".wlw",
        ".wlq",
        ".wlm",
        ".pm4"
    };

    private readonly IArchiveCatalog _archiveCatalog;
    private readonly IArchiveCatalogFactory _archiveCatalogFactory;
    private readonly string _gamePath;
    private readonly List<string> _overlayRoots = new();

    public IArchiveReader ArchiveReader => _archiveCatalog;
    private readonly HashSet<string> _fileSet = new(StringComparer.OrdinalIgnoreCase);
    private List<string> _fileList = new();
    private readonly Dictionary<string, string> _canonicalPathMap = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, IReadOnlyList<string>> _filesByExtension = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, string> _firstFileByExtensionAndBaseName = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, bool> _existsCache = new(StringComparer.OrdinalIgnoreCase);
    private readonly object _existsCacheLock = new();
    private bool _loaded;

    // Loose file roots to check (game folder structure)
    private readonly List<string> _looseRoots = new();

    // Alpha-era nested wrappers: virtual path or model alias → disk path for listfile-less .ext.MPQ files.
    private readonly Dictionary<string, string> _alphaMpqCache = new(StringComparer.OrdinalIgnoreCase);

    // Global raw-byte cache for repeated model/texture reads.
    private readonly Dictionary<string, byte[]?> _readCache = new(StringComparer.OrdinalIgnoreCase);
    private readonly LinkedList<string> _readCacheLru = new();
    private readonly Dictionary<string, LinkedListNode<string>> _readCacheLruMap = new(StringComparer.OrdinalIgnoreCase);
    private readonly object _readCacheLock = new();
    private long _readCacheBytes;
    private const int MaxReadCacheEntries = 4096;
    private const long MaxReadCacheBytes = 256L * 1024 * 1024;

    private readonly ConcurrentQueue<PrefetchRequest> _prefetchQueue = new();
    private readonly HashSet<string> _prefetchQueued = new(StringComparer.OrdinalIgnoreCase);
    private readonly object _prefetchLock = new();
    private SemaphoreSlim? _prefetchSignal;
    private CancellationTokenSource? _prefetchCts;
    private Task[]? _prefetchWorkers;
    private List<IArchiveCatalog>? _prefetchMpqServices;
    private const int PrefetchWorkerCount = 2;

    private long _fileExistsRequests;
    private long _fileExistsCacheHits;
    private long _fileExistsLooseHits;
    private long _fileExistsAlphaHits;
    private long _fileExistsMpqHits;
    private long _fileExistsCanonicalHits;
    private long _fileExistsMisses;
    private long _readRequests;
    private long _readCacheHits;
    private long _readCacheMisses;
    private long _readLooseHits;
    private long _readAlphaHits;
    private long _readMpqHits;
    private long _readMisses;
    private long _uncachedReadCount;
    private long _uncachedReadTicks;
    private long _prefetchRequests;
    private long _prefetchCacheSkips;
    private long _prefetchDuplicateSkips;
    private long _prefetchEnqueued;
    private long _prefetchCompleted;
    private long _prefetchReadHits;
    private long _prefetchReadMisses;
    private long _prefetchQueueTicks;
    private long _prefetchReadTicks;

    public string GamePath => _gamePath;
    public IReadOnlyList<string> OverlayRoots => _overlayRoots;
    public string Name => _overlayRoots.Count == 0
        ? $"Game: {Path.GetFileName(_gamePath)}"
        : $"Game: {Path.GetFileName(_gamePath)} + {_overlayRoots.Count} loose overlay(s)";
    public bool IsLoaded => _loaded;

    public MpqDataSourceStats GetStatsSnapshot()
    {
        int readCacheEntryCount;
        long readCacheBytes;
        lock (_readCacheLock)
        {
            readCacheEntryCount = _readCache.Count;
            readCacheBytes = _readCacheBytes;
        }

        long uncachedReadCount = Interlocked.Read(ref _uncachedReadCount);
        long uncachedReadTicks = Interlocked.Read(ref _uncachedReadTicks);
        long prefetchCompleted = Interlocked.Read(ref _prefetchCompleted);
        long prefetchQueueTicks = Interlocked.Read(ref _prefetchQueueTicks);
        long prefetchReadTicks = Interlocked.Read(ref _prefetchReadTicks);

        double avgUncachedReadMs = uncachedReadCount > 0
            ? (uncachedReadTicks * 1000.0) / Stopwatch.Frequency / uncachedReadCount
            : 0.0;
        double avgPrefetchQueueMs = prefetchCompleted > 0
            ? (prefetchQueueTicks * 1000.0) / Stopwatch.Frequency / prefetchCompleted
            : 0.0;
        double avgPrefetchReadMs = prefetchCompleted > 0
            ? (prefetchReadTicks * 1000.0) / Stopwatch.Frequency / prefetchCompleted
            : 0.0;

        return new MpqDataSourceStats(
            Interlocked.Read(ref _fileExistsRequests),
            Interlocked.Read(ref _fileExistsCacheHits),
            Interlocked.Read(ref _fileExistsLooseHits),
            Interlocked.Read(ref _fileExistsAlphaHits),
            Interlocked.Read(ref _fileExistsMpqHits),
            Interlocked.Read(ref _fileExistsCanonicalHits),
            Interlocked.Read(ref _fileExistsMisses),
            Interlocked.Read(ref _readRequests),
            Interlocked.Read(ref _readCacheHits),
            Interlocked.Read(ref _readCacheMisses),
            Interlocked.Read(ref _readLooseHits),
            Interlocked.Read(ref _readAlphaHits),
            Interlocked.Read(ref _readMpqHits),
            Interlocked.Read(ref _readMisses),
            avgUncachedReadMs,
            Interlocked.Read(ref _prefetchRequests),
            Interlocked.Read(ref _prefetchCacheSkips),
            Interlocked.Read(ref _prefetchDuplicateSkips),
            Interlocked.Read(ref _prefetchEnqueued),
            prefetchCompleted,
            Interlocked.Read(ref _prefetchReadHits),
            Interlocked.Read(ref _prefetchReadMisses),
            avgPrefetchQueueMs,
            avgPrefetchReadMs,
            readCacheEntryCount,
            readCacheBytes,
            _prefetchQueue.Count);
    }

    public MpqDataSource(string gamePath, string? listfilePath = null, IEnumerable<string>? overlayRoots = null, IArchiveCatalogFactory? archiveCatalogFactory = null)
    {
        _archiveCatalogFactory = archiveCatalogFactory ?? new MpqArchiveCatalogFactory();
        _archiveCatalog = _archiveCatalogFactory.Create();
        _gamePath = gamePath;

        ViewerLog.Important(ViewerLog.Category.MpqData, $"Loading game folder: {gamePath}");

        ArchiveCatalogBootstrapResult bootstrap = ArchiveCatalogBootstrapper.Bootstrap(_archiveCatalog, new[] { gamePath }, listfilePath);

        // 1. Extract files from MPQ internal (listfile) entries
        var internalFiles = bootstrap.InternalFiles;
        foreach (var file in internalFiles)
            _fileSet.Add(file);
        ViewerLog.Info(ViewerLog.Category.MpqData, $"Added {internalFiles.Count} files from MPQ internal listfiles.");

        // 2. Add any previously known files (from hash table / scanned)
        var knownFiles = bootstrap.KnownFiles;
        foreach (var file in knownFiles)
            _fileSet.Add(file);
        if (knownFiles.Count > 0)
            ViewerLog.Info(ViewerLog.Category.MpqData, $"Added {knownFiles.Count} previously known files.");

        // 3. Optionally add user-provided external listfile entries
        if (bootstrap.ExternalListfileEntries.Count > 0)
        {
            foreach (string file in bootstrap.ExternalListfileEntries)
                _fileSet.Add(file);

            ViewerLog.Info(ViewerLog.Category.MpqData, $"Added {bootstrap.ExternalListfileEntries.Count} listfile entries.");
        }

        // 4. Scan loose files on disk
        ScanLooseFiles(gamePath);
        
        // 5. Scan for Alpha 0.5.3 listfile-less .ext.MPQ archives (WMO, WDT, WDL)
        ScanAlphaNestedMpqArchives(gamePath);

        _fileList = _fileSet.OrderBy(f => f, StringComparer.OrdinalIgnoreCase).ToList();
        BuildLookupIndexes();

        if (overlayRoots != null)
        {
            foreach (string overlayRoot in overlayRoots)
            {
                if (!string.IsNullOrWhiteSpace(overlayRoot))
                    AddOverlayRoot(overlayRoot, out _, out _);
            }
        }

        _loaded = true;

        // Debug: show file counts by extension
        var extCounts = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        foreach (var f in _fileList)
        {
            var ext = Path.GetExtension(f).ToLowerInvariant();
            if (!extCounts.ContainsKey(ext)) extCounts[ext] = 0;
            extCounts[ext]++;
        }
        ViewerLog.Important(ViewerLog.Category.MpqData, $"Ready. {_fileList.Count} known files:");
        foreach (var ec in extCounts.OrderByDescending(x => x.Value).Take(10))
        {
            ViewerLog.Important(ViewerLog.Category.MpqData, $"  {ec.Key}: {ec.Value} files");
        }
    }

    /// <summary>
    /// Scans for Alpha-era listfile-less .ext.MPQ archives (terrain/WMO wrappers and per-model wrappers).
    /// These files wrap a single data file as file ID 1 inside an individual MPQ archive.
    /// Builds a virtual path/alias → disk path cache for fast reads.
    /// </summary>
    private void ScanAlphaNestedMpqArchives(string gamePath)
    {
        string[] validScanRoots = new[]
        {
            Path.Combine(gamePath, "Data"),
            Path.Combine(gamePath, "Data", "World"),
            Path.Combine(gamePath, "World"),
        };

        HashSet<string> excludeFolders = new(StringComparer.OrdinalIgnoreCase)
        {
            "wmos", "addons", "interface", "backup", "cache", "logs"
        };

        // Extensions that use listfile-less individual .ext.MPQ wrapping in Alpha-era clients.
        string[] nestedExts =
        {
            ".wmo.MPQ", ".wmo.mpq",
            ".wdt.MPQ", ".wdt.mpq",
            ".wdl.MPQ", ".wdl.mpq",
            ".mdx.MPQ", ".mdx.mpq",
            ".mdl.MPQ", ".mdl.mpq",
            ".m2.MPQ", ".m2.mpq"
        };

        var countByExt = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

        foreach (var root in validScanRoots)
        {
            if (!Directory.Exists(root)) continue;

            IEnumerable<string> allMpqs;
            try
            {
                allMpqs = Directory.EnumerateFiles(root, "*.MPQ", SearchOption.AllDirectories)
                    .Concat(Directory.EnumerateFiles(root, "*.mpq", SearchOption.AllDirectories))
                    .Distinct(StringComparer.OrdinalIgnoreCase);
            }
            catch { continue; }

            foreach (var mpqFile in allMpqs)
            {
                var fileName = Path.GetFileName(mpqFile);

                // Check if this matches any nested extension pattern
                string? matchedSuffix = null;
                foreach (var suffix in nestedExts)
                {
                    if (fileName.EndsWith(suffix, StringComparison.OrdinalIgnoreCase))
                    {
                        matchedSuffix = suffix;
                        break;
                    }
                }
                if (matchedSuffix == null) continue;

                // Check exclude folders
                var dir = Path.GetDirectoryName(mpqFile);
                bool shouldExclude = false;
                while (!string.IsNullOrEmpty(dir) && dir.Length > root.Length)
                {
                    var dirName = Path.GetFileName(dir);
                    if (excludeFolders.Contains(dirName))
                    {
                        shouldExclude = true;
                        break;
                    }
                    dir = Path.GetDirectoryName(dir);
                }
                if (shouldExclude) continue;

                // Generate virtual path: strip the trailing .MPQ/.mpq
                // e.g., "World\wmo\Azeroth\test.wmo.MPQ" → "World\wmo\Azeroth\test.wmo"
                var relativePath = Path.GetRelativePath(root, mpqFile);
                var virtualPath = relativePath;
                if (virtualPath.EndsWith(".MPQ", StringComparison.OrdinalIgnoreCase))
                    virtualPath = virtualPath[..^4];
                virtualPath = virtualPath.Replace('/', '\\');

                foreach (string registeredPath in EnumerateAlphaWrapperVirtualPaths(virtualPath))
                {
                    _fileSet.Add(registeredPath);
                    _alphaMpqCache[registeredPath] = mpqFile;

                    var ext = Path.GetExtension(registeredPath).ToLowerInvariant();
                    if (!countByExt.ContainsKey(ext)) countByExt[ext] = 0;
                    countByExt[ext]++;
                }
            }
        }

        ViewerLog.Info(ViewerLog.Category.MpqData, $"Alpha nested MPQ scan: {_alphaMpqCache.Count} files found");
        foreach (var kvp in countByExt.OrderByDescending(x => x.Value))
            ViewerLog.Info(ViewerLog.Category.MpqData, $"  {kvp.Key}: {kvp.Value} files");
    }

    private static IEnumerable<string> EnumerateAlphaWrapperVirtualPaths(string virtualPath)
    {
        yield return virtualPath;

        string extension = Path.GetExtension(virtualPath);
        if (!IsModelExtension(extension))
            yield break;

        string basePath = virtualPath[..^extension.Length];
        foreach (string aliasExtension in GetModelAliasExtensions(extension))
            yield return basePath + aliasExtension;
    }

    private static IEnumerable<string> GetModelAliasExtensions(string extension)
    {
        if (extension.Equals(".mdx", StringComparison.OrdinalIgnoreCase))
        {
            yield return ".mdl";
            yield return ".m2";
            yield break;
        }

        if (extension.Equals(".mdl", StringComparison.OrdinalIgnoreCase))
        {
            yield return ".mdx";
            yield return ".m2";
            yield break;
        }

        if (extension.Equals(".m2", StringComparison.OrdinalIgnoreCase))
        {
            yield return ".mdx";
            yield return ".mdl";
        }
    }

    private static bool IsModelExtension(string extension)
    {
        return extension.Equals(".mdx", StringComparison.OrdinalIgnoreCase)
            || extension.Equals(".mdl", StringComparison.OrdinalIgnoreCase)
            || extension.Equals(".m2", StringComparison.OrdinalIgnoreCase);
    }

    public bool AddOverlayRoot(string rootPath, out string normalizedRoot, out string message)
    {
        normalizedRoot = Path.GetFullPath(rootPath);

        if (!Directory.Exists(normalizedRoot))
        {
            message = $"Overlay folder not found: {normalizedRoot}";
            return false;
        }

        if (_overlayRoots.Contains(normalizedRoot, StringComparer.OrdinalIgnoreCase))
        {
            message = $"Overlay already attached: {normalizedRoot}";
            return false;
        }

        int fileCountBefore = _fileSet.Count;
        int alphaCountBefore = _alphaMpqCache.Count;

        int scannedLooseFiles = ScanLooseFiles(normalizedRoot);
        ScanAlphaNestedMpqArchives(normalizedRoot);

        int addedFiles = _fileSet.Count - fileCountBefore;
        int addedAlphaEntries = _alphaMpqCache.Count - alphaCountBefore;
        if (scannedLooseFiles <= 0 && addedAlphaEntries <= 0)
        {
            message = $"No supported loose map files were found under {normalizedRoot}";
            return false;
        }

        _overlayRoots.Add(normalizedRoot);
        _fileList = _fileSet.OrderBy(f => f, StringComparer.OrdinalIgnoreCase).ToList();
        BuildLookupIndexes();
        ClearExistsCache();
        ClearReadCache();

        message = $"Attached loose overlay {normalizedRoot} ({scannedLooseFiles} loose files scanned, {addedFiles} newly indexed paths, {addedAlphaEntries} alpha wrapper entries).";
        ViewerLog.Important(ViewerLog.Category.MpqData, message);
        return true;
    }

    private int ScanLooseFiles(string gamePath)
    {
        // Scan for loose files in the game directory structure
        // Alpha 0.5.3 has files directly in Data/ subfolders
        string[] scanRoots = new[]
        {
            gamePath,
            Path.Combine(gamePath, "Data"),
        };

        int totalAdded = 0;

        foreach (var root in scanRoots)
        {
            if (!Directory.Exists(root)) continue;

            ViewerLog.Info(ViewerLog.Category.MpqData, $"Scanning root: {root}");

            // Look for common WoW data subdirectories
            string[] dataDirs = { "World", "Creature", "Character", "Item", "Textures",
                                  "Interface", "Spells", "Environments", "Dungeons" };

            // Also scan for WMO files in World/wmo subdirectories
            string[] wmoScanDirs = { @"World\wmo", @"World\WMO" };

            bool foundAny = false;
            foreach (var subDir in dataDirs)
            {
                var fullDir = Path.Combine(root, subDir);
                if (!Directory.Exists(fullDir)) continue;

                if (!_looseRoots.Contains(root))
                    _looseRoots.Add(root);

                ViewerLog.Debug(ViewerLog.Category.MpqData, $"Scanning loose files: {fullDir}");
                int supportedFilesFound = 0;

                try
                {
                    foreach (var file in Directory.EnumerateFiles(fullDir, "*.*", SearchOption.AllDirectories))
                    {
                        var ext = Path.GetExtension(file).ToLowerInvariant();
                        if (IndexedLooseExtensions.Contains(ext))
                        {
                            supportedFilesFound++;
                            var virtualPath = Path.GetRelativePath(root, file).Replace('/', '\\');
                            _fileSet.Add(virtualPath);
                        }
                    }
                }
                catch (Exception ex)
                {
                    ViewerLog.Error(ViewerLog.Category.MpqData, $"Scan error in {fullDir}: {ex.Message}");
                }

                totalAdded += supportedFilesFound;
                ViewerLog.Debug(ViewerLog.Category.MpqData, $"  Found {supportedFilesFound} supported files in {subDir}/");
                foundAny |= supportedFilesFound > 0;
            }

            // Additional scan for WMO files in nested wmo directories
            foreach (var wmoDir in wmoScanDirs)
            {
                var fullWmoDir = Path.Combine(root, wmoDir);
                if (!Directory.Exists(fullWmoDir)) continue;

                ViewerLog.Debug(ViewerLog.Category.MpqData, $"Scanning WMO files: {fullWmoDir}");
                int supportedFilesFound = 0;

                try
                {
                    foreach (var file in Directory.EnumerateFiles(fullWmoDir, "*.wmo", SearchOption.AllDirectories))
                    {
                        supportedFilesFound++;
                        var virtualPath = Path.GetRelativePath(root, file).Replace('/', '\\');
                        _fileSet.Add(virtualPath);
                    }
                }
                catch (Exception ex)
                {
                    ViewerLog.Error(ViewerLog.Category.MpqData, $"WMO scan error in {fullWmoDir}: {ex.Message}");
                }

                totalAdded += supportedFilesFound;
                ViewerLog.Debug(ViewerLog.Category.MpqData, $"  Found {supportedFilesFound} WMO files");
            }

            if (foundAny) break;
        }

        return totalAdded;
    }

    public bool FileExists(string virtualPath)
    {
        if (string.IsNullOrWhiteSpace(virtualPath))
            return false;

        Interlocked.Increment(ref _fileExistsRequests);
        var normalized = virtualPath.Replace('/', '\\');

        lock (_existsCacheLock)
        {
            if (_existsCache.TryGetValue(normalized, out bool cached))
            {
                Interlocked.Increment(ref _fileExistsCacheHits);
                return cached;
            }
        }

        // Check loose files first (faster), then alpha MPQ cache, then file set, then StormLib
        if (TryResolveLoosePath(virtualPath) != null)
        {
            Interlocked.Increment(ref _fileExistsLooseHits);
            return CacheExistsResult(normalized, true);
        }

        // Alpha 0.5.3: .ext.MPQ archives (WMO, WDT, WDL) are indexed in _alphaMpqCache
        if (_alphaMpqCache.ContainsKey(normalized))
        {
            Interlocked.Increment(ref _fileExistsAlphaHits);
            return CacheExistsResult(normalized, true);
        }

        // Also try with .mpq suffix (e.g. "development.wdt" → "development.wdt.mpq" in cache)
        if (_alphaMpqCache.ContainsKey(normalized + ".mpq"))
        {
            Interlocked.Increment(ref _fileExistsAlphaHits);
            return CacheExistsResult(normalized, true);
        }

        // Check direct path in loaded MPQ archives.
        if (_archiveCatalog.FileExists(normalized))
        {
            Interlocked.Increment(ref _fileExistsMpqHits);
            return CacheExistsResult(normalized, true);
        }

        // Try canonical file-set spelling/casing as a fallback probe.
        if (_canonicalPathMap.TryGetValue(normalized, out string? canonicalPath))
        {
            if (_archiveCatalog.FileExists(canonicalPath))
            {
                Interlocked.Increment(ref _fileExistsCanonicalHits);
                return CacheExistsResult(normalized, true);
            }
        }

        Interlocked.Increment(ref _fileExistsMisses);
        return CacheExistsResult(normalized, false);
    }

    private bool CacheExistsResult(string normalizedPath, bool value)
    {
        lock (_existsCacheLock)
            _existsCache[normalizedPath] = value;

        return value;
    }

    private void ClearExistsCache()
    {
        lock (_existsCacheLock)
            _existsCache.Clear();
    }

    /// <summary>Find the actual path from the file set (case-insensitive, returns correctly-cased path).</summary>
    public string? FindInFileSet(string virtualPath)
    {
        string normalized = virtualPath.Replace('/', '\\');
        return _canonicalPathMap.TryGetValue(normalized, out var resolved) ? resolved : null;
    }

    public string? FindByBaseName(string baseName, IEnumerable<string> extensions)
    {
        foreach (var extension in extensions)
        {
            if (_firstFileByExtensionAndBaseName.TryGetValue(BuildBaseNameLookupKey(extension, baseName), out var resolved))
                return resolved;
        }

        return null;
    }

    public void PrefetchFile(string virtualPath)
    {
        string normalized = virtualPath.Replace('/', '\\');
        if (string.IsNullOrWhiteSpace(normalized))
            return;

        Interlocked.Increment(ref _prefetchRequests);
        if (TryGetCachedRead(normalized, out _))
        {
            Interlocked.Increment(ref _prefetchCacheSkips);
            return;
        }

        EnsurePrefetchWorkers();

        lock (_prefetchLock)
        {
            if (!_prefetchQueued.Add(normalized))
            {
                Interlocked.Increment(ref _prefetchDuplicateSkips);
                return;
            }
        }

        Interlocked.Increment(ref _prefetchEnqueued);
        _prefetchQueue.Enqueue(new PrefetchRequest(normalized, Stopwatch.GetTimestamp()));
        _prefetchSignal?.Release();
    }

    public byte[]? ReadFile(string virtualPath)
    {
        Interlocked.Increment(ref _readRequests);
        string normalized = virtualPath.Replace('/', '\\');
        if (TryGetCachedRead(normalized, out var cached))
        {
            Interlocked.Increment(ref _readCacheHits);
            return cached;
        }

        Interlocked.Increment(ref _readCacheMisses);
        var data = ReadFileUncached(normalized, _archiveCatalog, logFailures: true, isPrefetch: false);
        CacheRead(normalized, data);
        return data;
    }

    /// <summary>
    /// Reads the primary data file from an Alpha listfile-less .ext.MPQ archive.
    /// Uses AlphaArchiveReader which has smart block selection (name hash lookup, largest block fallback,
    /// magic byte checking) — critical for WMO MPQs that may contain multiple files.
    /// </summary>
    private byte[]? ReadFromAlphaMpq(string mpqDiskPath, string virtualPath)
    {
        // Build internal name candidates from the virtual path for hash-based lookup
        var candidates = AlphaArchiveReader.BuildInternalNameCandidates(virtualPath).ToList();
        // Also add just the filename
        var fileName = Path.GetFileName(virtualPath);
        if (!string.IsNullOrEmpty(fileName) && !candidates.Contains(fileName, StringComparer.OrdinalIgnoreCase))
            candidates.Insert(0, fileName);
        return AlphaArchiveReader.ReadFromMpq(mpqDiskPath, candidates);
    }


    private string? TryResolveLoosePath(string virtualPath)
    {
        var normalized = virtualPath.Replace('/', '\\').TrimStart('\\');

        // Search newest loose overlays first so attached overlays override earlier roots.
        for (int i = _looseRoots.Count - 1; i >= 0; i--)
        {
            var root = _looseRoots[i];
            var fullPath = Path.Combine(root, normalized);
            if (File.Exists(fullPath))
                return fullPath;
        }

        // Also check game path directly
        var directPath = Path.Combine(_gamePath, normalized);
        if (File.Exists(directPath))
            return directPath;

        var dataPath = Path.Combine(_gamePath, "Data", normalized);
        if (File.Exists(dataPath))
            return dataPath;

        // Trace WMO/PM4 misses because both are commonly supplied by loose overlays.
        if (virtualPath.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase)
            || virtualPath.EndsWith(".pm4", StringComparison.OrdinalIgnoreCase))
        {
            ViewerLog.Trace($"[MpqDataSource] TryResolveLoosePath FAILED for '{normalized}':");
            for (int i = _looseRoots.Count - 1; i >= 0; i--)
                ViewerLog.Trace($"  loose root[{i}]: {Path.Combine(_looseRoots[i], normalized)}");
            ViewerLog.Trace($"  gamePath: {directPath}");
            ViewerLog.Trace($"  dataPath: {dataPath}");
        }

        return null;
    }

    public bool TryResolveWritablePath(string virtualPath, out string? fullPath)
    {
        fullPath = TryResolveLoosePath(virtualPath);
        return fullPath != null;
    }

    public IReadOnlyList<string> GetFileList(string? extensionFilter = null)
    {
        if (extensionFilter != null)
            return _filesByExtension.TryGetValue(extensionFilter, out var files) ? files : Array.Empty<string>();

        return _fileList;
    }

    private void BuildLookupIndexes()
    {
        _canonicalPathMap.Clear();
        _filesByExtension.Clear();
        _firstFileByExtensionAndBaseName.Clear();

        var filesByExtension = new Dictionary<string, List<string>>(StringComparer.OrdinalIgnoreCase);

        foreach (var file in _fileList)
        {
            var normalized = file.Replace('/', '\\');
            _canonicalPathMap[normalized] = normalized;

            var extension = Path.GetExtension(normalized);
            if (!filesByExtension.TryGetValue(extension, out var files))
            {
                files = new List<string>();
                filesByExtension[extension] = files;
            }

            files.Add(normalized);

            var baseName = Path.GetFileNameWithoutExtension(normalized);
            if (!string.IsNullOrWhiteSpace(baseName))
            {
                var lookupKey = BuildBaseNameLookupKey(extension, baseName);
                if (!_firstFileByExtensionAndBaseName.ContainsKey(lookupKey))
                    _firstFileByExtensionAndBaseName[lookupKey] = normalized;
            }
        }

        foreach (var kvp in filesByExtension)
            _filesByExtension[kvp.Key] = kvp.Value;
    }

    private static string BuildBaseNameLookupKey(string extension, string baseName)
        => $"{extension.ToLowerInvariant()}|{baseName.ToLowerInvariant()}";

    private bool TryGetCachedRead(string normalizedPath, out byte[]? data)
    {
        lock (_readCacheLock)
        {
            if (_readCache.TryGetValue(normalizedPath, out data))
            {
                TouchReadCache_NoLock(normalizedPath);
                return true;
            }
        }

        data = null;
        return false;
    }

    private void CacheRead(string normalizedPath, byte[]? data)
    {
        lock (_readCacheLock)
        {
            if (_readCache.TryGetValue(normalizedPath, out var existing))
            {
                _readCacheBytes -= existing?.LongLength ?? 0;
                _readCache[normalizedPath] = data;
                _readCacheBytes += data?.LongLength ?? 0;
                TouchReadCache_NoLock(normalizedPath);
                EvictReadCacheIfNeeded_NoLock();
                return;
            }

            _readCache[normalizedPath] = data;
            _readCacheBytes += data?.LongLength ?? 0;
            var node = _readCacheLru.AddLast(normalizedPath);
            _readCacheLruMap[normalizedPath] = node;
            EvictReadCacheIfNeeded_NoLock();
        }
    }

    private void ClearReadCache()
    {
        lock (_readCacheLock)
        {
            _readCache.Clear();
            _readCacheLru.Clear();
            _readCacheLruMap.Clear();
            _readCacheBytes = 0;
        }
    }

    private void TouchReadCache(string normalizedPath)
    {
        lock (_readCacheLock)
            TouchReadCache_NoLock(normalizedPath);
    }

    private void TouchReadCache_NoLock(string normalizedPath)
    {
        if (!_readCacheLruMap.TryGetValue(normalizedPath, out var node))
            return;

        if (!ReferenceEquals(node, _readCacheLru.Last))
        {
            _readCacheLru.Remove(node);
            _readCacheLru.AddLast(node);
        }
    }

    private void EvictReadCacheIfNeeded()
    {
        lock (_readCacheLock)
            EvictReadCacheIfNeeded_NoLock();
    }

    private void EvictReadCacheIfNeeded_NoLock()
    {
        while (_readCache.Count > MaxReadCacheEntries || _readCacheBytes > MaxReadCacheBytes)
        {
            var oldest = _readCacheLru.First;
            if (oldest == null)
                break;

            var key = oldest.Value;
            _readCacheLru.RemoveFirst();
            _readCacheLruMap.Remove(key);

            if (_readCache.TryGetValue(key, out var data))
                _readCacheBytes -= data?.LongLength ?? 0;

            _readCache.Remove(key);
        }
    }

    private byte[]? CompleteRead(MpqReadResolutionKind resolutionKind, byte[]? data, long startTimestamp, bool isPrefetch)
    {
        long elapsedTicks = Stopwatch.GetTimestamp() - startTimestamp;

        if (isPrefetch)
        {
            Interlocked.Increment(ref _prefetchCompleted);
            Interlocked.Add(ref _prefetchReadTicks, elapsedTicks);
            if (resolutionKind == MpqReadResolutionKind.Miss)
                Interlocked.Increment(ref _prefetchReadMisses);
            else
                Interlocked.Increment(ref _prefetchReadHits);
        }
        else
        {
            Interlocked.Increment(ref _uncachedReadCount);
            Interlocked.Add(ref _uncachedReadTicks, elapsedTicks);
        }

        switch (resolutionKind)
        {
            case MpqReadResolutionKind.Loose:
                Interlocked.Increment(ref _readLooseHits);
                break;
            case MpqReadResolutionKind.AlphaWrapper:
                Interlocked.Increment(ref _readAlphaHits);
                break;
            case MpqReadResolutionKind.Mpq:
                Interlocked.Increment(ref _readMpqHits);
                break;
            case MpqReadResolutionKind.Miss:
                Interlocked.Increment(ref _readMisses);
                break;
        }

        return data;
    }

    private byte[]? ReadFileUncached(string virtualPath, IArchiveReader archiveReader, bool logFailures, bool isPrefetch)
    {
        long startTimestamp = Stopwatch.GetTimestamp();
        var loosePath = TryResolveLoosePath(virtualPath);
        if (loosePath != null)
        {
            ViewerLog.Trace($"[MpqDataSource] ReadFile '{virtualPath}' → loose file: {loosePath}");
            return CompleteRead(MpqReadResolutionKind.Loose, File.ReadAllBytes(loosePath), startTimestamp, isPrefetch);
        }

        string? alphaMpqPath = null;
        string? alphaMpqKey = null;

        if (_alphaMpqCache.TryGetValue(virtualPath, out alphaMpqPath))
            alphaMpqKey = virtualPath;
        else if (virtualPath.EndsWith(".mpq", StringComparison.OrdinalIgnoreCase) &&
                 _alphaMpqCache.TryGetValue(virtualPath[..^4], out alphaMpqPath))
            alphaMpqKey = virtualPath[..^4];

        if (alphaMpqPath != null && alphaMpqKey != null)
        {
            ViewerLog.Trace($"[MpqDataSource] ReadFile '{virtualPath}' → alpha MPQ: {alphaMpqPath}");
            var alphaData = ReadFromAlphaMpq(alphaMpqPath, alphaMpqKey);
            if (alphaData != null)
                return CompleteRead(MpqReadResolutionKind.AlphaWrapper, alphaData, startTimestamp, isPrefetch);

            ViewerLog.Trace($"[MpqDataSource] ReadFile '{virtualPath}' → alpha MPQ extraction FAILED");
        }

        var mpqData = archiveReader.ReadFile(virtualPath);
        if (mpqData != null)
        {
            ViewerLog.Trace($"[MpqDataSource] ReadFile '{virtualPath}' → standard MPQ ({mpqData.Length} bytes)");
            return CompleteRead(MpqReadResolutionKind.Mpq, mpqData, startTimestamp, isPrefetch);
        }

        if (logFailures)
            ViewerLog.Trace($"[MpqDataSource] ReadFile '{virtualPath}' → NOT FOUND (loose={_looseRoots.Count} roots, alphaMpq={_alphaMpqCache.ContainsKey(virtualPath)})");

        return CompleteRead(MpqReadResolutionKind.Miss, null, startTimestamp, isPrefetch);
    }

    private void EnsurePrefetchWorkers()
    {
        lock (_prefetchLock)
        {
            if (_prefetchWorkers != null)
                return;

            _prefetchSignal = new SemaphoreSlim(0);
            _prefetchCts = new CancellationTokenSource();
            _prefetchMpqServices = new List<IArchiveCatalog>(PrefetchWorkerCount);
            _prefetchWorkers = new Task[PrefetchWorkerCount];

            for (int i = 0; i < PrefetchWorkerCount; i++)
            {
                IArchiveCatalog archiveCatalog = _archiveCatalogFactory.Create();
                archiveCatalog.LoadArchives(new[] { _gamePath });
                _prefetchMpqServices.Add(archiveCatalog);
                _prefetchWorkers[i] = Task.Run(() => PrefetchWorkerLoop(archiveCatalog, _prefetchCts.Token));
            }
        }
    }

    private async Task PrefetchWorkerLoop(IArchiveReader archiveReader, CancellationToken cancellationToken)
    {
        if (_prefetchSignal == null)
            return;

        while (!cancellationToken.IsCancellationRequested)
        {
            try
            {
                await _prefetchSignal.WaitAsync(cancellationToken);
            }
            catch (OperationCanceledException)
            {
                break;
            }

            while (_prefetchQueue.TryDequeue(out var request))
            {
                string normalizedPath = request.Path;
                lock (_prefetchLock)
                    _prefetchQueued.Remove(normalizedPath);

                Interlocked.Add(ref _prefetchQueueTicks, Stopwatch.GetTimestamp() - request.EnqueuedTimestamp);

                if (TryGetCachedRead(normalizedPath, out _))
                    continue;

                var data = ReadFileUncached(normalizedPath, archiveReader, logFailures: false, isPrefetch: true);
                CacheRead(normalizedPath, data);
            }
        }
    }

    public void Dispose()
    {
        if (_prefetchCts != null)
        {
            _prefetchCts.Cancel();
            _prefetchSignal?.Release(PrefetchWorkerCount);
            try
            {
                if (_prefetchWorkers != null)
                    Task.WaitAll(_prefetchWorkers, TimeSpan.FromSeconds(2));
            }
            catch
            {
            }
        }

        if (_prefetchMpqServices != null)
        {
            foreach (var service in _prefetchMpqServices)
                service.Dispose();
        }

        _prefetchSignal?.Dispose();
        _prefetchCts?.Dispose();
        _archiveCatalog.Dispose();
    }
}
