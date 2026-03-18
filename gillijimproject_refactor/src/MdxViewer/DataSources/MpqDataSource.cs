using MdxViewer.Logging;
using WoWMapConverter.Core.Services;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;

namespace MdxViewer.DataSources;

/// <summary>
/// Data source backed by MPQ archives + loose files on disk.
/// Uses WoWMapConverter.Core's NativeMpqService for MPQ reading.
/// Builds file list from MPQ-internal (listfile) entries — accurate for any client version.
/// Supports Alpha 0.5.3, Classic, TBC, and WotLK 3.3.5 game folders.
/// </summary>
public class MpqDataSource : IDataSource
{
    private readonly NativeMpqService _mpq = new();
    private readonly string _gamePath;

    /// <summary>Exposes the underlying MPQ service for DBC provider access.</summary>
    public NativeMpqService MpqService => _mpq;
    private readonly HashSet<string> _fileSet = new(StringComparer.OrdinalIgnoreCase);
    private List<string> _fileList = new();
    private readonly Dictionary<string, string> _canonicalPathMap = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, IReadOnlyList<string>> _filesByExtension = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, string> _firstFileByExtensionAndBaseName = new(StringComparer.OrdinalIgnoreCase);
    private bool _loaded;

    // Loose file roots to check (game folder structure)
    private readonly List<string> _looseRoots = new();

    // Alpha 0.5.3: virtual path → disk path for listfile-less .ext.MPQ files (WMO, WDT, WDL)
    private readonly Dictionary<string, string> _alphaMpqCache = new(StringComparer.OrdinalIgnoreCase);

    // Global raw-byte cache for repeated model/texture reads.
    private readonly Dictionary<string, byte[]?> _readCache = new(StringComparer.OrdinalIgnoreCase);
    private readonly LinkedList<string> _readCacheLru = new();
    private readonly Dictionary<string, LinkedListNode<string>> _readCacheLruMap = new(StringComparer.OrdinalIgnoreCase);
    private readonly object _readCacheLock = new();
    private long _readCacheBytes;
    private const int MaxReadCacheEntries = 4096;
    private const long MaxReadCacheBytes = 256L * 1024 * 1024;

    private readonly ConcurrentQueue<string> _prefetchQueue = new();
    private readonly HashSet<string> _prefetchQueued = new(StringComparer.OrdinalIgnoreCase);
    private readonly object _prefetchLock = new();
    private SemaphoreSlim? _prefetchSignal;
    private CancellationTokenSource? _prefetchCts;
    private Task[]? _prefetchWorkers;
    private List<NativeMpqService>? _prefetchMpqServices;
    private const int PrefetchWorkerCount = 2;

    public string Name => $"Game: {Path.GetFileName(_gamePath)}";
    public bool IsLoaded => _loaded;

    public MpqDataSource(string gamePath, string? listfilePath = null)
    {
        _gamePath = gamePath;

        ViewerLog.Important(ViewerLog.Category.MpqData, $"Loading game folder: {gamePath}");

        // 1. Load MPQ archives (large MPQs with listfiles)
        _mpq.LoadArchives(new[] { gamePath });

        // 2. Extract files from MPQ internal (listfile) entries
        var internalFiles = _mpq.ExtractInternalListfiles();
        foreach (var file in internalFiles)
            _fileSet.Add(file);
        ViewerLog.Info(ViewerLog.Category.MpqData, $"Added {internalFiles.Count} files from MPQ internal listfiles.");

        // Also add any previously known files (from hash table / scanned)
        var knownFiles = _mpq.GetAllKnownFiles();
        foreach (var file in knownFiles)
            _fileSet.Add(file);
        if (knownFiles.Count > 0)
            ViewerLog.Info(ViewerLog.Category.MpqData, $"Added {knownFiles.Count} previously known files.");

        // 3. Optionally load user-provided external listfile
        if (!string.IsNullOrEmpty(listfilePath) && File.Exists(listfilePath))
        {
            ViewerLog.Info(ViewerLog.Category.MpqData, $"Loading listfile: {listfilePath}");
            _mpq.LoadListfile(listfilePath);
            AddExternalListfileEntries(listfilePath);
        }

        // 4. Scan loose files on disk
        ScanLooseFiles(gamePath);
        
        // 5. Scan for Alpha 0.5.3 listfile-less .ext.MPQ archives (WMO, WDT, WDL)
        ScanAlphaNestedMpqArchives(gamePath);

        _fileList = _fileSet.OrderBy(f => f, StringComparer.OrdinalIgnoreCase).ToList();
        BuildLookupIndexes();
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
    /// Scans for Alpha 0.5.3 listfile-less .ext.MPQ archives (WMO, WDT, WDL).
    /// These files wrap a single data file as file ID 1 inside an individual MPQ archive.
    /// Builds a virtual path → disk path cache for fast reads.
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

        // Extensions that use listfile-less individual .ext.MPQ wrapping in Alpha 0.5.3
        string[] nestedExts = { ".wmo.MPQ", ".wmo.mpq", ".wdt.MPQ", ".wdt.mpq", ".wdl.MPQ", ".wdl.mpq" };

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

                _fileSet.Add(virtualPath);
                _alphaMpqCache[virtualPath] = mpqFile;

                var ext = Path.GetExtension(virtualPath).ToLowerInvariant();
                if (!countByExt.ContainsKey(ext)) countByExt[ext] = 0;
                countByExt[ext]++;
            }
        }

        ViewerLog.Info(ViewerLog.Category.MpqData, $"Alpha nested MPQ scan: {_alphaMpqCache.Count} files found");
        foreach (var kvp in countByExt.OrderByDescending(x => x.Value))
            ViewerLog.Info(ViewerLog.Category.MpqData, $"  {kvp.Key}: {kvp.Value} files");
    }

    private void AddExternalListfileEntries(string listfilePath)
    {
        int count = 0;
        foreach (var line in File.ReadLines(listfilePath))
        {
            var name = line.Trim();
            if (string.IsNullOrEmpty(name)) continue;

            if (name.Contains(';'))
            {
                var parts = name.Split(';', 2);
                if (parts.Length > 1) name = parts[1].Trim();
            }

            if (!string.IsNullOrEmpty(name))
            {
                _fileSet.Add(name);
                count++;
            }
        }
        ViewerLog.Info(ViewerLog.Category.MpqData, $"Added {count} listfile entries.");
    }

    private void ScanLooseFiles(string gamePath)
    {
        // Scan for loose files in the game directory structure
        // Alpha 0.5.3 has files directly in Data/ subfolders
        string[] scanRoots = new[]
        {
            gamePath,
            Path.Combine(gamePath, "Data"),
        };

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
                int before = _fileSet.Count;

                try
                {
                    foreach (var file in Directory.EnumerateFiles(fullDir, "*.*", SearchOption.AllDirectories))
                    {
                        var ext = Path.GetExtension(file).ToLowerInvariant();
                        if (ext is ".mdx" or ".wmo" or ".m2" or ".blp" or ".skin" or ".anim" or ".dbc")
                        {
                            var virtualPath = Path.GetRelativePath(root, file).Replace('/', '\\');
                            _fileSet.Add(virtualPath);
                        }
                    }
                }
                catch (Exception ex)
                {
                    ViewerLog.Error(ViewerLog.Category.MpqData, $"Scan error in {fullDir}: {ex.Message}");
                }

                int found = _fileSet.Count - before;
                ViewerLog.Debug(ViewerLog.Category.MpqData, $"  Found {found} files in {subDir}/");
                foundAny = found > 0;
            }

            // Additional scan for WMO files in nested wmo directories
            foreach (var wmoDir in wmoScanDirs)
            {
                var fullWmoDir = Path.Combine(root, wmoDir);
                if (!Directory.Exists(fullWmoDir)) continue;

                ViewerLog.Debug(ViewerLog.Category.MpqData, $"Scanning WMO files: {fullWmoDir}");
                int before = _fileSet.Count;

                try
                {
                    foreach (var file in Directory.EnumerateFiles(fullWmoDir, "*.wmo", SearchOption.AllDirectories))
                    {
                        var virtualPath = Path.GetRelativePath(root, file).Replace('/', '\\');
                        _fileSet.Add(virtualPath);
                    }
                }
                catch (Exception ex)
                {
                    ViewerLog.Error(ViewerLog.Category.MpqData, $"WMO scan error in {fullWmoDir}: {ex.Message}");
                }

                int found = _fileSet.Count - before;
                ViewerLog.Debug(ViewerLog.Category.MpqData, $"  Found {found} WMO files");
            }

            if (foundAny) break;
        }
    }

    public bool FileExists(string virtualPath)
    {
        // Check loose files first (faster), then alpha MPQ cache, then file set, then StormLib
        if (TryResolveLoosePath(virtualPath) != null)
            return true;

        var normalized = virtualPath.Replace('/', '\\');

        // Alpha 0.5.3: .ext.MPQ archives (WMO, WDT, WDL) are indexed in _alphaMpqCache
        if (_alphaMpqCache.ContainsKey(normalized) || _alphaMpqCache.ContainsKey(virtualPath))
            return true;

        // Also try with .mpq suffix (e.g. "development.wdt" → "development.wdt.mpq" in cache)
        if (_alphaMpqCache.ContainsKey(normalized + ".mpq") || _alphaMpqCache.ContainsKey(virtualPath + ".mpq"))
            return true;

        // Check the master file set (includes all discovered files)
        if (_fileSet.Contains(normalized) || _fileSet.Contains(virtualPath))
            return true;

        return _mpq.FileExists(virtualPath);
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

        if (TryGetCachedRead(normalized, out _))
            return;

        EnsurePrefetchWorkers();

        lock (_prefetchLock)
        {
            if (!_prefetchQueued.Add(normalized))
                return;
        }

        _prefetchQueue.Enqueue(normalized);
        _prefetchSignal?.Release();
    }

    public byte[]? ReadFile(string virtualPath)
    {
        string normalized = virtualPath.Replace('/', '\\');
        if (TryGetCachedRead(normalized, out var cached))
            return cached;

        var data = ReadFileUncached(normalized, _mpq, logFailures: true);
        CacheRead(normalized, data);
        return data;
    }

    /// <summary>
    /// Reads the primary data file from an Alpha listfile-less .ext.MPQ archive.
    /// Uses AlphaMpqReader which has smart block selection (name hash lookup, largest block fallback,
    /// magic byte checking) — critical for WMO MPQs that may contain multiple files.
    /// </summary>
    private byte[]? ReadFromAlphaMpq(string mpqDiskPath, string virtualPath)
    {
        // Build internal name candidates from the virtual path for hash-based lookup
        var candidates = AlphaMpqReader.BuildInternalNameCandidates(virtualPath).ToList();
        // Also add just the filename
        var fileName = Path.GetFileName(virtualPath);
        if (!string.IsNullOrEmpty(fileName) && !candidates.Contains(fileName, StringComparer.OrdinalIgnoreCase))
            candidates.Insert(0, fileName);
        return AlphaMpqReader.ReadFromMpq(mpqDiskPath, candidates);
    }


    private string? TryResolveLoosePath(string virtualPath)
    {
        var normalized = virtualPath.Replace('/', '\\').TrimStart('\\');

        // Check each loose root
        foreach (var root in _looseRoots)
        {
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

        // Log failed resolution for .wmo files to help debug loose file issues
        if (virtualPath.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase))
        {
            ViewerLog.Trace($"[MpqDataSource] TryResolveLoosePath FAILED for '{normalized}':");
            foreach (var root in _looseRoots)
                ViewerLog.Trace($"  loose root: {Path.Combine(root, normalized)}");
            ViewerLog.Trace($"  gamePath: {directPath}");
            ViewerLog.Trace($"  dataPath: {dataPath}");
        }

        return null;
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

    private byte[]? ReadFileUncached(string virtualPath, NativeMpqService mpqService, bool logFailures)
    {
        var loosePath = TryResolveLoosePath(virtualPath);
        if (loosePath != null)
        {
            ViewerLog.Trace($"[MpqDataSource] ReadFile '{virtualPath}' → loose file: {loosePath}");
            return File.ReadAllBytes(loosePath);
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
                return alphaData;

            ViewerLog.Trace($"[MpqDataSource] ReadFile '{virtualPath}' → alpha MPQ extraction FAILED");
        }

        var mpqData = mpqService.ReadFile(virtualPath);
        if (mpqData != null)
        {
            ViewerLog.Trace($"[MpqDataSource] ReadFile '{virtualPath}' → standard MPQ ({mpqData.Length} bytes)");
            return mpqData;
        }

        if (logFailures)
            ViewerLog.Trace($"[MpqDataSource] ReadFile '{virtualPath}' → NOT FOUND (loose={_looseRoots.Count} roots, alphaMpq={_alphaMpqCache.ContainsKey(virtualPath)})");

        return null;
    }

    private void EnsurePrefetchWorkers()
    {
        lock (_prefetchLock)
        {
            if (_prefetchWorkers != null)
                return;

            _prefetchSignal = new SemaphoreSlim(0);
            _prefetchCts = new CancellationTokenSource();
            _prefetchMpqServices = new List<NativeMpqService>(PrefetchWorkerCount);
            _prefetchWorkers = new Task[PrefetchWorkerCount];

            for (int i = 0; i < PrefetchWorkerCount; i++)
            {
                var mpqService = new NativeMpqService();
                mpqService.LoadArchives(new[] { _gamePath });
                _prefetchMpqServices.Add(mpqService);
                _prefetchWorkers[i] = Task.Run(() => PrefetchWorkerLoop(mpqService, _prefetchCts.Token));
            }
        }
    }

    private async Task PrefetchWorkerLoop(NativeMpqService mpqService, CancellationToken cancellationToken)
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

            while (_prefetchQueue.TryDequeue(out var normalizedPath))
            {
                lock (_prefetchLock)
                    _prefetchQueued.Remove(normalizedPath);

                if (TryGetCachedRead(normalizedPath, out _))
                    continue;

                var data = ReadFileUncached(normalizedPath, mpqService, logFailures: false);
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
        _mpq.Dispose();
    }
}
