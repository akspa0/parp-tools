using System.Diagnostics;
using System.Numerics;
using MdxLTool.Formats.Mdx;
using MdxViewer.DataSources;
using MdxViewer.Logging;
using MdxViewer.Rendering;
using Silk.NET.OpenGL;
using WoWMapConverter.Core.Converters;

namespace MdxViewer.Terrain;

public readonly record struct WorldAssetReadStats(
    long ReadRequests,
    long FileCacheHits,
    long ResolvedPathCacheHits,
    long PathProbeAttempts,
    long PathProbeResolutions,
    long PathProbeMisses,
    int ResolvedPathCacheCount);

/// <summary>
/// Centralized asset manager for world scene rendering.
/// Ensures each model and texture is loaded exactly once into GPU memory,
/// then instanced via transforms for all placements.
/// 
/// Ownership: WorldAssetManager owns all GPU resources (renderers, textures).
/// WorldScene owns the instance lists (transforms) and delegates rendering here.
/// </summary>
public class WorldAssetManager : IDisposable
{
    private readonly GL _gl;
    private readonly IDataSource? _dataSource;
    private readonly ReplaceableTextureResolver? _texResolver;
    private string? _buildVersion;

    // ── Shared caches ──────────────────────────────────────────────────

    // Model path (normalized) → loaded renderer (null = load attempted but failed)
    private readonly Dictionary<string, MdxRenderer?> _mdxModels = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, WmoRenderer?> _wmoModels = new(StringComparer.OrdinalIgnoreCase);

    // LRU tracking — keys ordered by last access time (most recent at end)
    private readonly LinkedList<string> _mdxLru = new();
    private readonly Dictionary<string, LinkedListNode<string>> _mdxLruMap = new(StringComparer.OrdinalIgnoreCase);
    private readonly LinkedList<string> _wmoLru = new();
    private readonly Dictionary<string, LinkedListNode<string>> _wmoLruMap = new(StringComparer.OrdinalIgnoreCase);
    // World placements keep instance references long after the initial tile-load callback.
    // Bounded renderer eviction causes placed objects to disappear until the tile reloads,
    // so renderer residency defaults to unlimited while the raw file-data cache stays bounded.
    private static readonly int MaxMdxCached = 0; // 0 = unlimited
    private static readonly int MaxWmoCached = 0; // 0 = unlimited

    // Raw file data cache — avoids re-reading the same file from MPQ multiple times
    private readonly Dictionary<string, byte[]?> _fileDataCache = new(StringComparer.OrdinalIgnoreCase);
    private readonly LinkedList<string> _fileLru = new();
    private readonly Dictionary<string, LinkedListNode<string>> _fileLruMap = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, string> _resolvedReadPathCache = new(StringComparer.OrdinalIgnoreCase);
    private const int MaxFileCached = 1000; // Max raw file entries cached

    // Deferred world-asset loading keeps tile streaming responsive.
    private readonly Queue<string> _priorityMdxLoads = new();
    private readonly Queue<string> _pendingMdxLoads = new();
    private readonly HashSet<string> _queuedMdxLoads = new(StringComparer.OrdinalIgnoreCase);
    private readonly HashSet<string> _priorityQueuedMdxLoads = new(StringComparer.OrdinalIgnoreCase);
    private readonly Queue<string> _priorityWmoLoads = new();
    private readonly Queue<string> _pendingWmoLoads = new();
    private readonly HashSet<string> _queuedWmoLoads = new(StringComparer.OrdinalIgnoreCase);
    private readonly HashSet<string> _priorityQueuedWmoLoads = new(StringComparer.OrdinalIgnoreCase);
    private bool _preferWmoNext;
    private readonly Dictionary<string, string?> _bestSkinPathCache = new(StringComparer.OrdinalIgnoreCase);

    // Stats
    public int MdxModelsLoaded => _mdxModels.Count(kv => kv.Value != null);
    public int MdxModelsFailed => _mdxModels.Count(kv => kv.Value == null);
    public int WmoModelsLoaded => _wmoModels.Count(kv => kv.Value != null);
    public int WmoModelsFailed => _wmoModels.Count(kv => kv.Value == null);
    public int FileCacheCount => _fileDataCache.Count;
    public int PendingAssetLoadCount => _queuedMdxLoads.Count + _queuedWmoLoads.Count;

    private long _fileReadRequests;
    private long _fileReadCacheHits;
    private long _resolvedPathCacheHits;
    private long _pathProbeAttempts;
    private long _pathProbeResolutions;
    private long _pathProbeMisses;

    public WorldAssetManager(GL gl, IDataSource? dataSource, ReplaceableTextureResolver? texResolver = null, string? buildVersion = null)
    {
        _gl = gl;
        _dataSource = dataSource;
        _texResolver = texResolver;
        _buildVersion = buildVersion;
    }

    public void SetBuildVersion(string? buildVersion)
    {
        _buildVersion = buildVersion;
    }

    public WorldAssetReadStats GetReadStats()
        => new(
            _fileReadRequests,
            _fileReadCacheHits,
            _resolvedPathCacheHits,
            _pathProbeAttempts,
            _pathProbeResolutions,
            _pathProbeMisses,
            _resolvedReadPathCache.Count);

    /// <summary>
    /// Pre-register all model names referenced by the map so we know the full asset set.
    /// Does NOT load anything yet — just prepares the manifest.
    /// </summary>
    public AssetManifest BuildManifest(IReadOnlyList<string> mdxNames, IReadOnlyList<string> wmoNames,
        IReadOnlyList<MddfPlacement> mddfPlacements, IReadOnlyList<ModfPlacement> modfPlacements)
    {
        var manifest = new AssetManifest();

        // Collect unique MDX models actually referenced by placements
        foreach (var p in mddfPlacements)
        {
            if (p.NameIndex >= 0 && p.NameIndex < mdxNames.Count)
                manifest.ReferencedMdx.Add(NormalizeKey(mdxNames[p.NameIndex]));
        }

        // Collect unique WMO models actually referenced by placements
        foreach (var p in modfPlacements)
        {
            if (p.NameIndex >= 0 && p.NameIndex < wmoNames.Count)
                manifest.ReferencedWmo.Add(NormalizeKey(wmoNames[p.NameIndex]));
        }

        ViewerLog.Important(ViewerLog.Category.General, $"Manifest: {manifest.ReferencedMdx.Count} unique MDX, {manifest.ReferencedWmo.Count} unique WMO");
        ViewerLog.Info(ViewerLog.Category.General, $"Name tables: {mdxNames.Count} MDX names, {wmoNames.Count} WMO names");
        ViewerLog.Info(ViewerLog.Category.General, $"Placements: {mddfPlacements.Count} MDDF, {modfPlacements.Count} MODF");
        if (modfPlacements.Count > 0)
        {
            var p = modfPlacements[0];
            string name = p.NameIndex >= 0 && p.NameIndex < wmoNames.Count ? wmoNames[p.NameIndex] : $"BAD_INDEX({p.NameIndex})";
            ViewerLog.Debug(ViewerLog.Category.Wmo, $"First MODF: nameIdx={p.NameIndex} name=\"{name}\" key=\"{NormalizeKey(name)}\"");
        }
        if (mddfPlacements.Count > 0)
        {
            var p = mddfPlacements[0];
            string name = p.NameIndex >= 0 && p.NameIndex < mdxNames.Count ? mdxNames[p.NameIndex] : $"BAD_INDEX({p.NameIndex})";
            ViewerLog.Debug(ViewerLog.Category.Mdx, $"First MDDF: nameIdx={p.NameIndex} name=\"{name}\" key=\"{NormalizeKey(name)}\"");
        }
        return manifest;
    }

    /// <summary>
    /// Load all models in the manifest. Each model is loaded exactly once.
    /// </summary>
    public void LoadManifest(AssetManifest manifest)
    {
        int mdxOk = 0, mdxFail = 0;
        foreach (var key in manifest.ReferencedMdx)
        {
            if (_mdxModels.ContainsKey(key)) continue;
            var renderer = LoadMdxModel(key);
            _mdxModels[key] = renderer;
            if (renderer != null) mdxOk++; else mdxFail++;
        }

        int wmoOk = 0, wmoFail = 0;
        foreach (var key in manifest.ReferencedWmo)
        {
            if (_wmoModels.ContainsKey(key)) continue;
            var renderer = LoadWmoModel(key);
            _wmoModels[key] = renderer;
            if (renderer != null) wmoOk++; else wmoFail++;
        }

        ViewerLog.Important(ViewerLog.Category.General, $"Loaded: MDX {mdxOk} ok / {mdxFail} failed, WMO {wmoOk} ok / {wmoFail} failed");
    }

    /// <summary>
    /// Ensure an MDX model is loaded (lazy load on first reference).
    /// </summary>
    public void EnsureMdxLoaded(string normalizedKey)
    {
        if (_mdxModels.TryGetValue(normalizedKey, out var cachedRenderer) && cachedRenderer != null)
        {
            TouchLru(_mdxLru, _mdxLruMap, normalizedKey);
            return;
        }

        if (cachedRenderer == null && _mdxModels.ContainsKey(normalizedKey))
            ViewerLog.Debug(ViewerLog.Category.Mdx, $"Retrying cached failed MDX load: \"{normalizedKey}\"");

        var renderer = LoadMdxModel(normalizedKey);
        _mdxModels[normalizedKey] = renderer;
        TouchLru(_mdxLru, _mdxLruMap, normalizedKey);
        EvictMdxIfNeeded();
    }

    /// <summary>
    /// Ensure a WMO model is loaded (lazy load on first reference).
    /// </summary>
    public void EnsureWmoLoaded(string normalizedKey)
    {
        if (_wmoModels.TryGetValue(normalizedKey, out var cachedRenderer) && cachedRenderer != null)
        {
            TouchLru(_wmoLru, _wmoLruMap, normalizedKey);
            return;
        }

        if (cachedRenderer == null && _wmoModels.ContainsKey(normalizedKey))
            ViewerLog.Debug(ViewerLog.Category.Wmo, $"Retrying cached failed WMO load: \"{normalizedKey}\"");

        var renderer = LoadWmoModel(normalizedKey);
        _wmoModels[normalizedKey] = renderer;
        TouchLru(_wmoLru, _wmoLruMap, normalizedKey);
        EvictWmoIfNeeded();
    }

    /// <summary>
    /// Get a loaded MDX renderer by normalized key. Returns null if not loaded or failed.
    /// </summary>
    public MdxRenderer? GetMdx(string normalizedKey)
    {
        if (_mdxModels.TryGetValue(normalizedKey, out var r))
        {
            if (r != null)
                TouchLru(_mdxLru, _mdxLruMap, normalizedKey);
            return r;
        }

        EnsureMdxLoaded(normalizedKey);
        return _mdxModels.TryGetValue(normalizedKey, out r) ? r : null;
    }

    /// <summary>
    /// Get the model-space bounding box for a loaded MDX model.
    /// Returns false if the model is not loaded.
    /// </summary>
    public bool TryGetMdxBounds(string normalizedKey, out Vector3 boundsMin, out Vector3 boundsMax)
    {
        if (_mdxModels.TryGetValue(normalizedKey, out var r) && r != null)
        {
            boundsMin = r.BoundsMin;
            boundsMax = r.BoundsMax;
            return true;
        }
        boundsMin = boundsMax = Vector3.Zero;
        return false;
    }

    /// <summary>
    /// Get the bounding box center for a loaded MDX model.
    /// MDX geometry is offset from origin — the BB center is the effective pivot.
    /// Returns false if the model is not loaded.
    /// </summary>
    public bool TryGetMdxPivotOffset(string normalizedKey, out Vector3 pivotOffset)
    {
        if (_mdxModels.TryGetValue(normalizedKey, out var r) && r != null)
        {
            pivotOffset = (r.BoundsMin + r.BoundsMax) * 0.5f;
            return true;
        }
        pivotOffset = Vector3.Zero;
        return false;
    }

    /// <summary>
    /// Get the MOHD bounding box for a loaded WMO model (local space).
    /// Returns false if the model is not loaded.
    /// </summary>
    public bool TryGetWmoBounds(string normalizedKey, out Vector3 boundsMin, out Vector3 boundsMax)
    {
        if (_wmoModels.TryGetValue(normalizedKey, out var r) && r != null)
        {
            boundsMin = r.BoundsMin;
            boundsMax = r.BoundsMax;
            return true;
        }
        boundsMin = boundsMax = Vector3.Zero;
        return false;
    }

    /// <summary>
    /// Get a loaded WMO renderer by normalized key. Returns null if not loaded or failed.
    /// </summary>
    public WmoRenderer? GetWmo(string normalizedKey)
    {
        if (_wmoModels.TryGetValue(normalizedKey, out var r))
        {
            if (r != null)
                TouchLru(_wmoLru, _wmoLruMap, normalizedKey);
            return r;
        }

        EnsureWmoLoaded(normalizedKey);
        return _wmoModels.TryGetValue(normalizedKey, out r) ? r : null;
    }

    public bool TryGetLoadedMdx(string normalizedKey, out MdxRenderer? renderer)
    {
        if (_mdxModels.TryGetValue(normalizedKey, out renderer) && renderer != null)
        {
            TouchLru(_mdxLru, _mdxLruMap, normalizedKey);
            return true;
        }

        renderer = null;
        return false;
    }

    public bool TryGetLoadedWmo(string normalizedKey, out WmoRenderer? renderer)
    {
        if (_wmoModels.TryGetValue(normalizedKey, out renderer) && renderer != null)
        {
            TouchLru(_wmoLru, _wmoLruMap, normalizedKey);
            return true;
        }

        renderer = null;
        return false;
    }

    public void QueueMdxLoad(string normalizedKey)
    {
        normalizedKey = NormalizeKey(normalizedKey);

        if (_mdxModels.TryGetValue(normalizedKey, out var cachedRenderer) && cachedRenderer != null)
            return;

        if (_queuedMdxLoads.Add(normalizedKey))
        {
            PrefetchModelBytes(normalizedKey);
            _pendingMdxLoads.Enqueue(normalizedKey);
        }
    }

    public void PrioritizeMdxLoad(string normalizedKey)
    {
        normalizedKey = NormalizeKey(normalizedKey);

        if (_mdxModels.TryGetValue(normalizedKey, out var cachedRenderer) && cachedRenderer != null)
            return;

        if (_queuedMdxLoads.Add(normalizedKey))
        {
            PrefetchModelBytes(normalizedKey);
            _priorityMdxLoads.Enqueue(normalizedKey);
        }

        if (_priorityQueuedMdxLoads.Add(normalizedKey))
            _priorityMdxLoads.Enqueue(normalizedKey);
    }

    public void QueueWmoLoad(string normalizedKey)
    {
        normalizedKey = NormalizeKey(normalizedKey);

        if (_wmoModels.TryGetValue(normalizedKey, out var cachedRenderer) && cachedRenderer != null)
            return;

        if (_queuedWmoLoads.Add(normalizedKey))
        {
            PrefetchModelBytes(normalizedKey);
            _pendingWmoLoads.Enqueue(normalizedKey);
        }
    }

    public void PrioritizeWmoLoad(string normalizedKey)
    {
        normalizedKey = NormalizeKey(normalizedKey);

        if (_wmoModels.TryGetValue(normalizedKey, out var cachedRenderer) && cachedRenderer != null)
            return;

        if (_queuedWmoLoads.Add(normalizedKey))
        {
            PrefetchModelBytes(normalizedKey);
            _priorityWmoLoads.Enqueue(normalizedKey);
        }

        if (_priorityQueuedWmoLoads.Add(normalizedKey))
            _priorityWmoLoads.Enqueue(normalizedKey);
    }

    public int ProcessPendingLoads(int maxLoads = 2, double maxBudgetMs = 6.0)
    {
        if (maxLoads <= 0 || maxBudgetMs <= 0)
            return 0;

        var stopwatch = Stopwatch.StartNew();
        int loadsCompleted = 0;

        while (loadsCompleted < maxLoads && stopwatch.Elapsed.TotalMilliseconds < maxBudgetMs)
        {
            if (!TryDequeuePendingLoad(out bool isMdx, out string? key) || string.IsNullOrWhiteSpace(key))
                break;

            if (isMdx)
            {
                if (!_mdxModels.TryGetValue(key, out var cachedRenderer) || cachedRenderer == null)
                {
                    if (cachedRenderer == null && _mdxModels.ContainsKey(key))
                        ViewerLog.Debug(ViewerLog.Category.Mdx, $"Retrying deferred failed MDX load: \"{key}\"");

                    var renderer = LoadMdxModel(key);
                    _mdxModels[key] = renderer;
                    TouchLru(_mdxLru, _mdxLruMap, key);
                    EvictMdxIfNeeded();
                }
            }
            else
            {
                if (!_wmoModels.TryGetValue(key, out var cachedRenderer) || cachedRenderer == null)
                {
                    if (cachedRenderer == null && _wmoModels.ContainsKey(key))
                        ViewerLog.Debug(ViewerLog.Category.Wmo, $"Retrying deferred failed WMO load: \"{key}\"");

                    var renderer = LoadWmoModel(key);
                    _wmoModels[key] = renderer;
                    TouchLru(_wmoLru, _wmoLruMap, key);
                    EvictWmoIfNeeded();
                }
            }

            loadsCompleted++;
        }

        return loadsCompleted;
    }

    /// <summary>
    /// Read file data from the data source, with caching to avoid duplicate MPQ reads.
    /// </summary>
    public byte[]? ReadFileData(string virtualPath)
    {
        string key = NormalizeKey(virtualPath);
        _fileReadRequests++;
        if (_fileDataCache.TryGetValue(key, out var cached))
        {
            _fileReadCacheHits++;
            return cached;
        }

        byte[]? data = null;
        string? resolvedPath = null;

        if (_resolvedReadPathCache.TryGetValue(key, out string? cachedResolvedPath))
        {
            _resolvedPathCacheHits++;
            data = TryReadCandidate(cachedResolvedPath, out resolvedPath);
        }

        if (data == null)
        {
            foreach (string candidate in EnumerateReadCandidates(key))
            {
                if (!string.IsNullOrWhiteSpace(cachedResolvedPath) && candidate.Equals(cachedResolvedPath, StringComparison.OrdinalIgnoreCase))
                    continue;

                data = TryReadCandidate(candidate, out resolvedPath);
                if (data != null)
                    break;
            }
        }

        if (data != null && !string.IsNullOrWhiteSpace(resolvedPath))
            _resolvedReadPathCache[key] = NormalizeKey(resolvedPath);
        else if (data == null)
            _pathProbeMisses++;

        _fileDataCache[key] = data;
        TouchLru(_fileLru, _fileLruMap, key);
        EvictFileCacheIfNeeded();
        return data;
    }

    public static string NormalizeKey(string path) => path.Replace('/', '\\').ToLowerInvariant();

    private static string? SwapMdlMdxExtension(string path)
    {
        if (path.EndsWith(".mdl", StringComparison.OrdinalIgnoreCase))
            return path[..^4] + ".mdx";
        if (path.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase))
            return path[..^4] + ".mdl";
        // 3.x+ clients may reference .m2 while some archives/listfiles still expose .mdx.
        if (path.EndsWith(".m2", StringComparison.OrdinalIgnoreCase))
            return path[..^3] + ".mdx";
        return null;
    }

    private static IEnumerable<string> GetAlternateModelPaths(string path)
    {
        string? swapped = SwapMdlMdxExtension(path);
        if (!string.IsNullOrWhiteSpace(swapped))
            yield return swapped;

        if (path.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase))
            yield return path[..^4] + ".m2";
        else if (path.EndsWith(".mdl", StringComparison.OrdinalIgnoreCase))
            yield return path[..^4] + ".m2";
        else if (path.EndsWith(".m2", StringComparison.OrdinalIgnoreCase))
            yield return path[..^3] + ".mdl";
    }

    private string? TryResolveFromFileSet(string normalizedPath)
    {
        if (_dataSource is not MpqDataSource mpqDataSource)
            return null;

        foreach (var candidate in BuildFileSetCandidates(normalizedPath))
        {
            var found = mpqDataSource.FindInFileSet(candidate);
            if (!string.IsNullOrWhiteSpace(found))
                return found;
        }

        string baseName = Path.GetFileNameWithoutExtension(normalizedPath);
        if (string.IsNullOrWhiteSpace(baseName))
            return null;

        var indexedMatch = mpqDataSource.FindByBaseName(baseName, GetLikelyModelExtensions(normalizedPath));
        if (!string.IsNullOrWhiteSpace(indexedMatch))
            return NormalizeKey(indexedMatch);

        return null;
    }

    private static IEnumerable<string> BuildFileSetCandidates(string normalizedPath)
    {
        yield return normalizedPath;

        foreach (string alternate in GetAlternateModelPaths(normalizedPath))
            yield return alternate;

        string fileName = Path.GetFileName(normalizedPath);
        if (!string.IsNullOrWhiteSpace(fileName) && !fileName.Equals(normalizedPath, StringComparison.OrdinalIgnoreCase))
        {
            yield return fileName;

            foreach (string alternate in GetAlternateModelPaths(fileName))
                yield return alternate;
        }

        string baseName = Path.GetFileNameWithoutExtension(normalizedPath);
        if (!string.IsNullOrWhiteSpace(baseName))
        {
            yield return $"Creature\\{baseName}\\{baseName}.mdx";
            yield return $"Creature\\{baseName}\\{baseName}.m2";
            yield return $"Creature\\{baseName}\\{baseName}.mdl";
        }
    }

    private static IEnumerable<string> GetLikelyModelExtensions(string normalizedPath)
    {
        string ext = Path.GetExtension(normalizedPath);
        if (ext.Equals(".m2", StringComparison.OrdinalIgnoreCase))
        {
            yield return ".m2";
            yield return ".mdx";
            yield return ".mdl";
            yield break;
        }

        if (ext.Equals(".mdl", StringComparison.OrdinalIgnoreCase))
        {
            yield return ".mdl";
            yield return ".mdx";
            yield return ".m2";
            yield break;
        }

        yield return ".mdx";
        yield return ".mdl";
        yield return ".m2";
    }

    private byte[]? TryReadCandidate(string candidate, out string? resolvedPath)
    {
        resolvedPath = candidate;
        _pathProbeAttempts++;

        byte[]? data = _dataSource?.ReadFile(candidate);
        if (data != null)
        {
            _pathProbeResolutions++;
            return data;
        }

        resolvedPath = null;
        return null;
    }

    private IEnumerable<string> EnumerateReadCandidates(string normalizedPath)
    {
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        bool TryYield(string? candidate, out string yielded)
        {
            yielded = string.Empty;
            if (string.IsNullOrWhiteSpace(candidate))
                return false;

            string normalizedCandidate = NormalizeKey(candidate);
            if (!seen.Add(normalizedCandidate))
                return false;

            yielded = normalizedCandidate;
            return true;
        }

        if (TryYield(normalizedPath, out string exactPath))
            yield return exactPath;

        string? resolvedFileSetPath = TryResolveFromFileSet(normalizedPath);
        if (TryYield(resolvedFileSetPath, out string resolvedExactPath))
            yield return resolvedExactPath;

        foreach (string alternatePath in GetAlternateModelPaths(normalizedPath))
        {
            if (TryYield(alternatePath, out string yieldedAlternatePath))
                yield return yieldedAlternatePath;

            string? resolvedAlternatePath = TryResolveFromFileSet(alternatePath);
            if (TryYield(resolvedAlternatePath, out string yieldedResolvedAlternatePath))
                yield return yieldedResolvedAlternatePath;
        }

        string fileName = Path.GetFileName(normalizedPath);
        if (!string.IsNullOrWhiteSpace(fileName) && !fileName.Equals(normalizedPath, StringComparison.OrdinalIgnoreCase))
        {
            if (TryYield(fileName, out string yieldedFileName))
                yield return yieldedFileName;

            string? resolvedFileName = TryResolveFromFileSet(fileName);
            if (TryYield(resolvedFileName, out string yieldedResolvedFileName))
                yield return yieldedResolvedFileName;

            string[] prefixes = { "Creature\\", "World\\", "Environment\\" };
            foreach (string prefix in prefixes)
            {
                if (normalizedPath.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                    continue;

                if (TryYield(prefix + normalizedPath, out string yieldedPrefixedPath))
                    yield return yieldedPrefixedPath;

                if (TryYield(prefix + fileName, out string yieldedPrefixedFileName))
                    yield return yieldedPrefixedFileName;
            }
        }
    }

    private bool TryDequeuePendingLoad(out bool isMdx, out string? key)
    {
        bool tryWmoFirst = _preferWmoNext;
        _preferWmoNext = !_preferWmoNext;

        if (tryWmoFirst)
        {
            if (TryDequeueWmo(out key))
            {
                isMdx = false;
                return true;
            }

            if (TryDequeueMdx(out key))
            {
                isMdx = true;
                return true;
            }
        }
        else
        {
            if (TryDequeueMdx(out key))
            {
                isMdx = true;
                return true;
            }

            if (TryDequeueWmo(out key))
            {
                isMdx = false;
                return true;
            }
        }

        key = null;
        isMdx = false;
        return false;
    }

    private bool TryDequeueMdx(out string? key)
    {
        while (_priorityMdxLoads.TryDequeue(out key))
        {
            _priorityQueuedMdxLoads.Remove(key);

            if (_mdxModels.TryGetValue(key, out var renderer) && renderer != null)
                continue;

            _queuedMdxLoads.Remove(key);
            return true;
        }

        while (_pendingMdxLoads.TryDequeue(out key))
        {
            if (_mdxModels.TryGetValue(key, out var renderer) && renderer != null)
                continue;

            _queuedMdxLoads.Remove(key);
            return true;
        }

        key = null;
        return false;
    }

    private bool TryDequeueWmo(out string? key)
    {
        while (_priorityWmoLoads.TryDequeue(out key))
        {
            _priorityQueuedWmoLoads.Remove(key);

            if (_wmoModels.TryGetValue(key, out var renderer) && renderer != null)
                continue;

            _queuedWmoLoads.Remove(key);
            return true;
        }

        while (_pendingWmoLoads.TryDequeue(out key))
        {
            if (_wmoModels.TryGetValue(key, out var renderer) && renderer != null)
                continue;

            _queuedWmoLoads.Remove(key);
            return true;
        }

        key = null;
        return false;
    }

    // ── Private loading ────────────────────────────────────────────────

    private int _mdxLoadFailCount = 0;
    private MdxRenderer? LoadMdxModel(string normalizedKey)
    {
        try
        {
            string resolvedModelPath = ResolveCanonicalModelPath(normalizedKey);
            byte[]? data = ReadFileData(resolvedModelPath);
            if ((data == null || data.Length == 0) && !resolvedModelPath.Equals(normalizedKey, StringComparison.OrdinalIgnoreCase))
                data = ReadFileData(normalizedKey);
            if (data == null || data.Length == 0)
            {
                if (_mdxLoadFailCount++ < 5)
                    ViewerLog.Important(ViewerLog.Category.Mdx, $"MDX data null for: \"{normalizedKey}\"");
                return null;
            }

            bool isM2Family = resolvedModelPath.EndsWith(".m2", StringComparison.OrdinalIgnoreCase)
                || WarcraftNetM2Adapter.IsMd20(data)
                || WarcraftNetM2Adapter.IsMd21(data);

            // Match the final main-branch behavior first: adapt M2 + skin directly into the runtime model.
            // Keep byte-level conversion only as a fallback when the direct adapter path fails.
            if (isM2Family)
            {
                WarcraftNetM2Adapter.ValidateModelProfile(data, resolvedModelPath, _buildVersion);

                var candidatePaths = new List<string>(WarcraftNetM2Adapter.BuildSkinCandidates(resolvedModelPath));
                if (_dataSource != null)
                {
                    var bestSkinPath = ResolveBestSkinPath(resolvedModelPath);
                    if (!string.IsNullOrWhiteSpace(bestSkinPath))
                        candidatePaths.Add(bestSkinPath);
                }

                Exception? lastSkinError = null;
                bool anySkinFound = false;

                foreach (var skinPath in candidatePaths.Distinct(StringComparer.OrdinalIgnoreCase))
                {
                    var skinBytes = ReadFileData(skinPath);
                    if (skinBytes == null || skinBytes.Length == 0)
                        continue;

                    anySkinFound = true;

                    try
                    {
                        ViewerLog.Trace($"[M2] Trying skin for {Path.GetFileName(normalizedKey)}: {skinPath} ({skinBytes.Length} bytes)");
                        var adapted = WarcraftNetM2Adapter.BuildRuntimeModel(data, skinBytes, resolvedModelPath, _buildVersion);
                        string adaptedModelDir = Path.GetDirectoryName(resolvedModelPath) ?? "";
                        ViewerLog.Info(ViewerLog.Category.Mdx,
                            $"[M2] Selected skin for {Path.GetFileName(normalizedKey)}: {skinPath} ({skinBytes.Length} bytes)");
                        return new MdxRenderer(_gl, adapted, adaptedModelDir, _dataSource, _texResolver, resolvedModelPath, true, _buildVersion);
                    }
                    catch (Exception ex)
                    {
                        lastSkinError = ex;
                        ViewerLog.Debug(ViewerLog.Category.Mdx,
                            $"[M2] Skin candidate failed for {Path.GetFileName(normalizedKey)}: {skinPath} ({ex.Message})");
                    }
                }

                if (!anySkinFound)
                {
                    if (string.Equals(FormatProfileRegistry.ResolveModelProfile(_buildVersion)?.ProfileId, FormatProfileRegistry.M2Profile3018303.ProfileId, StringComparison.Ordinal))
                    {
                        try
                        {
                            var adapted = WarcraftNetM2Adapter.BuildRuntimeModel(data, null, resolvedModelPath, _buildVersion);
                            string adaptedModelDir = Path.GetDirectoryName(resolvedModelPath) ?? "";
                            ViewerLog.Info(ViewerLog.Category.Mdx,
                                $"[M2] Loaded embedded root-profile geometry for {Path.GetFileName(normalizedKey)} after no external .skin resolved");
                            return new MdxRenderer(_gl, adapted, adaptedModelDir, _dataSource, _texResolver, resolvedModelPath, true, _buildVersion);
                        }
                        catch (Exception ex)
                        {
                            lastSkinError = ex;
                            ViewerLog.Debug(ViewerLog.Category.Mdx,
                                $"[M2] Embedded root-profile world fallback failed for {Path.GetFileName(normalizedKey)}: {ex.Message}");
                        }
                    }

                    ViewerLog.Important(ViewerLog.Category.Mdx, $"[M2] Missing companion .skin for: {Path.GetFileName(normalizedKey)}");
                }

                if (WarcraftNetM2Adapter.IsMd20(data))
                {
                    var convertedBytes = ConvertM2ToMdx(data, resolvedModelPath);
                    if (convertedBytes != null && convertedBytes.Length > 0)
                    {
                        try
                        {
                            using var convertedStream = new MemoryStream(convertedBytes);
                            var convertedMdx = MdxFile.Load(convertedStream);
                            if (WarcraftNetM2Adapter.HasRenderableGeometry(convertedMdx))
                            {
                                string convertedModelDir = Path.GetDirectoryName(resolvedModelPath) ?? "";
                                ViewerLog.Info(ViewerLog.Category.Mdx,
                                    $"[M2] Falling back to M2->MDX conversion for {Path.GetFileName(normalizedKey)} after adapter failure");
                                return new MdxRenderer(_gl, convertedMdx, convertedModelDir, _dataSource, _texResolver, resolvedModelPath, true, _buildVersion);
                            }

                            lastSkinError = new InvalidDataException(
                                $"M2->MDX fallback produced no renderable geometry for {Path.GetFileName(normalizedKey)} ({WarcraftNetM2Adapter.SummarizeGeometry(convertedMdx)})");
                            ViewerLog.Debug(ViewerLog.Category.Mdx,
                                $"[M2] Rejecting converted world fallback for {Path.GetFileName(normalizedKey)}: {WarcraftNetM2Adapter.SummarizeGeometry(convertedMdx)}");
                        }
                        catch (Exception ex)
                        {
                            lastSkinError = ex;
                            ViewerLog.Debug(ViewerLog.Category.Mdx,
                                $"[M2] Converted world fallback load failed for {Path.GetFileName(normalizedKey)}: {ex.Message}");
                        }
                    }
                }

                if (lastSkinError != null)
                    throw new InvalidDataException($"All .skin candidates failed for M2: {Path.GetFileName(normalizedKey)}", lastSkinError);

                return null;
            }

            using var ms = new MemoryStream(data);
            var mdx = MdxFile.Load(ms);
            string modelDir = Path.GetDirectoryName(resolvedModelPath) ?? "";
            return new MdxRenderer(_gl, mdx, modelDir, _dataSource, _texResolver, resolvedModelPath);
        }
        catch (Exception ex)
        {
            if (_mdxLoadFailCount++ < 5)
                ViewerLog.Important(ViewerLog.Category.Mdx, $"MDX failed: {Path.GetFileName(normalizedKey)}\n{ex}");
            return null;
        }
    }

    private byte[]? ConvertM2ToMdx(byte[] m2Bytes, string normalizedKey)
    {
        try
        {
            byte[]? skinBytes = null;
            foreach (var skinPath in WarcraftNetM2Adapter.BuildSkinCandidates(normalizedKey).Distinct(StringComparer.OrdinalIgnoreCase))
            {
                skinBytes = ReadFileData(skinPath);
                if (skinBytes != null && skinBytes.Length > 0)
                {
                    ViewerLog.Trace($"[M2] Loaded skin for {Path.GetFileName(normalizedKey)} via converter: {skinPath} ({skinBytes.Length} bytes)");
                    break;
                }
            }

            var converter = new M2ToMdxConverter();
            byte[] mdxBytes = converter.ConvertToBytes(m2Bytes, skinBytes, _buildVersion);
            ViewerLog.Trace($"[M2] Converted {Path.GetFileName(normalizedKey)}: {m2Bytes.Length} -> {mdxBytes.Length} bytes");
            return mdxBytes;
        }
        catch (Exception ex)
        {
            ViewerLog.Debug(ViewerLog.Category.Mdx, $"[M2] M2->MDX converter fallback failed for {Path.GetFileName(normalizedKey)}: {ex.Message}");
            return null;
        }
    }

    private string ResolveCanonicalModelPath(string normalizedKey)
    {
        string? resolved = TryResolveFromFileSet(normalizedKey);
        if (!string.IsNullOrWhiteSpace(resolved))
            return NormalizeKey(resolved);

        string? swapped = SwapMdlMdxExtension(normalizedKey);
        if (!string.IsNullOrWhiteSpace(swapped))
        {
            resolved = TryResolveFromFileSet(swapped);
            if (!string.IsNullOrWhiteSpace(resolved))
                return NormalizeKey(resolved);
        }

        return normalizedKey;
    }

    private string? ResolveBestSkinPath(string resolvedModelPath)
    {
        if (_bestSkinPathCache.TryGetValue(resolvedModelPath, out var cachedPath))
            return cachedPath;

        string? resolvedPath = WarcraftNetM2Adapter.FindSkinInFileList(resolvedModelPath, _dataSource?.GetFileList(".skin") ?? Array.Empty<string>());
        _bestSkinPathCache[resolvedModelPath] = resolvedPath;
        return resolvedPath;
    }

    private void PrefetchModelBytes(string normalizedKey)
    {
        if (_dataSource is not MpqDataSource mpqDataSource)
            return;

        string canonicalModelPath = ResolveCanonicalModelPath(normalizedKey);
        mpqDataSource.PrefetchFile(canonicalModelPath);

        if (canonicalModelPath.Equals(normalizedKey, StringComparison.OrdinalIgnoreCase))
        {
            foreach (string alternatePath in GetAlternateModelPaths(normalizedKey))
            {
                string resolvedAlternatePath = ResolveCanonicalModelPath(alternatePath);
                if (!resolvedAlternatePath.Equals(canonicalModelPath, StringComparison.OrdinalIgnoreCase))
                    mpqDataSource.PrefetchFile(resolvedAlternatePath);
            }
        }

        string? bestSkinPath = ResolveBestSkinPath(canonicalModelPath);
        if (!string.IsNullOrWhiteSpace(bestSkinPath))
        {
            mpqDataSource.PrefetchFile(bestSkinPath);
            return;
        }

        foreach (string skinCandidate in WarcraftNetM2Adapter.BuildSkinCandidates(canonicalModelPath).Distinct(StringComparer.OrdinalIgnoreCase))
        {
            mpqDataSource.PrefetchFile(skinCandidate);
        }
    }

    private WmoRenderer? LoadWmoModel(string normalizedKey)
    {
        try
        {
            byte[]? data = ReadFileData(normalizedKey);
            if (data == null || data.Length == 0)
            {
                if (_wmoModels.Count < 3)
                    ViewerLog.Debug(ViewerLog.Category.Wmo, $"WMO data null for: \"{normalizedKey}\"");
                return null;
            }
            if (_wmoModels.Count < 3)
                ViewerLog.Debug(ViewerLog.Category.Wmo, $"WMO data found for: \"{normalizedKey}\" ({data.Length} bytes)");

            // Detect WMO version from bytes
            int version = DetectWmoVersion(data);

            WmoV14ToV17Converter.WmoV14Data wmo;

            if (version >= 17)
            {
                // v17+: parse directly into WmoV14Data — no lossy binary roundtrip
                var dir = Path.GetDirectoryName(normalizedKey)?.Replace('/', '\\') ?? "";
                var baseName = Path.GetFileNameWithoutExtension(normalizedKey);

                var groupBytesList = new List<byte[]>();
                for (int gi = 0; gi < 512; gi++)
                {
                    var groupName = $"{baseName}_{gi:D3}.wmo";
                    var groupPath = string.IsNullOrEmpty(dir) ? groupName : $"{dir}\\{groupName}";
                    var groupBytes = ReadFileData(groupPath);
                    if (groupBytes == null || groupBytes.Length == 0) break;
                    groupBytesList.Add(groupBytes);
                }

                var v17Parser = new WmoV17ToV14Converter();
                wmo = v17Parser.ParseV17ToModel(data, groupBytesList);
                ViewerLog.Trace($"[WMO] Parsed v{version} direct: {Path.GetFileName(normalizedKey)} ({wmo.Groups.Count} groups)");
            }
            else
            {
                // v14/v16: parse with existing pipeline
                string tmpPath = Path.Combine(Path.GetTempPath(), $"wmo_{Guid.NewGuid():N}.tmp");
                try
                {
                    File.WriteAllBytes(tmpPath, data);
                    var converter = new WmoV14ToV17Converter();
                    wmo = converter.ParseWmoV14(tmpPath);

                    // v14/v16 split format: load group files from data source
                    if (wmo.Groups.Count == 0 && wmo.GroupCount > 0 && _dataSource != null)
                    {
                        var wmoDir = Path.GetDirectoryName(normalizedKey)?.Replace('/', '\\') ?? "";
                        var wmoBase = Path.GetFileNameWithoutExtension(normalizedKey);
                        for (int gi = 0; gi < wmo.GroupCount; gi++)
                        {
                            var groupName = $"{wmoBase}_{gi:D3}.wmo";
                            var groupPath = string.IsNullOrEmpty(wmoDir) ? groupName : $"{wmoDir}\\{groupName}";
                            var groupBytes = ReadFileData(groupPath);
                            if (groupBytes != null && groupBytes.Length > 0)
                                converter.ParseGroupFile(groupBytes, wmo, gi);
                        }
                        for (int gi = 0; gi < wmo.Groups.Count; gi++)
                        {
                            if (wmo.Groups[gi].Name == null)
                                wmo.Groups[gi].Name = $"group_{gi}";
                        }
                    }
                }
                finally
                {
                    try { File.Delete(tmpPath); } catch { }
                }
            }

            string modelDir = Path.GetDirectoryName(normalizedKey) ?? "";
            return new WmoRenderer(_gl, wmo, modelDir, _dataSource, _texResolver, _buildVersion);
        }
        catch (Exception ex)
        {
            ViewerLog.Error(ViewerLog.Category.Wmo, $"WMO failed: {Path.GetFileName(normalizedKey)}\n{ex}");
            return null;
        }
    }

    /// <summary>
    /// Detect WMO version from raw bytes. Returns 14 for Alpha (MOMO container), version number for v17+, or 0.
    /// </summary>
    private static int DetectWmoVersion(byte[] data)
    {
        if (data.Length < 12) return 0;
        string magic = System.Text.Encoding.ASCII.GetString(data, 0, 4);
        string reversed = new string(magic.Reverse().ToArray());

        // v14 Alpha: starts with MOMO container
        if (magic == "MOMO" || reversed == "MOMO") return 14;

        // v17+: starts with MVER chunk
        if (magic == "MVER" || reversed == "MVER")
        {
            uint size = BitConverter.ToUInt32(data, 4);
            if (size >= 4 && data.Length >= 12)
                return (int)BitConverter.ToUInt32(data, 8);
        }
        return 0;
    }

    // ── LRU helpers ─────────────────────────────────────────────────────

    private static void TouchLru(LinkedList<string> lru, Dictionary<string, LinkedListNode<string>> map, string key)
    {
        if (map.TryGetValue(key, out var node))
        {
            lru.Remove(node);
            lru.AddLast(node);
        }
        else
        {
            var newNode = lru.AddLast(key);
            map[key] = newNode;
        }
    }

    private void EvictMdxIfNeeded()
    {
        if (MaxMdxCached <= 0)
            return;

        while (_mdxModels.Count > MaxMdxCached && _mdxLru.Count > 0)
        {
            string oldest = _mdxLru.First!.Value;
            _mdxLru.RemoveFirst();
            _mdxLruMap.Remove(oldest);
            if (_mdxModels.TryGetValue(oldest, out var r))
            {
                r?.Dispose();
                _mdxModels.Remove(oldest);
            }
        }
    }

    private void EvictWmoIfNeeded()
    {
        if (MaxWmoCached <= 0)
            return;

        while (_wmoModels.Count > MaxWmoCached && _wmoLru.Count > 0)
        {
            string oldest = _wmoLru.First!.Value;
            _wmoLru.RemoveFirst();
            _wmoLruMap.Remove(oldest);
            if (_wmoModels.TryGetValue(oldest, out var r))
            {
                r?.Dispose();
                _wmoModels.Remove(oldest);
            }
        }
    }

    private void EvictFileCacheIfNeeded()
    {
        while (_fileDataCache.Count > MaxFileCached && _fileLru.Count > 0)
        {
            string oldest = _fileLru.First!.Value;
            _fileLru.RemoveFirst();
            _fileLruMap.Remove(oldest);
            _fileDataCache.Remove(oldest);
        }
    }

    public void Dispose()
    {
        foreach (var r in _mdxModels.Values)
            r?.Dispose();
        _mdxModels.Clear();
        _mdxLru.Clear();
        _mdxLruMap.Clear();

        foreach (var r in _wmoModels.Values)
            r?.Dispose();
        _wmoModels.Clear();
        _wmoLru.Clear();
        _wmoLruMap.Clear();

        _fileDataCache.Clear();
        _fileLru.Clear();
        _fileLruMap.Clear();

        _pendingMdxLoads.Clear();
        _queuedMdxLoads.Clear();
        _priorityQueuedMdxLoads.Clear();
        _priorityWmoLoads.Clear();
        _pendingWmoLoads.Clear();
        _queuedWmoLoads.Clear();
        _priorityQueuedWmoLoads.Clear();
        _bestSkinPathCache.Clear();
        _priorityMdxLoads.Clear();
    }
}

/// <summary>
/// Describes the set of unique assets referenced by a map.
/// Built before loading so we know the full scope.
/// </summary>
public class AssetManifest
{
    public HashSet<string> ReferencedMdx { get; } = new(StringComparer.OrdinalIgnoreCase);
    public HashSet<string> ReferencedWmo { get; } = new(StringComparer.OrdinalIgnoreCase);
}
