using WoWViewer.DataSources;

namespace WoWViewer.Terrain;

/// <summary>
/// Handles path probing, candidate expansion, canonical path resolution, and raw path caching
/// for client assets across MPQ and CASC data sources.
/// </summary>
internal sealed class WorldAssetPathResolver
{
    private readonly IDataSource? _dataSource;
    private readonly Dictionary<string, string> _resolvedReadPathCache = new(StringComparer.OrdinalIgnoreCase);

    private long _resolvedPathCacheHits;
    private long _pathProbeAttempts;
    private long _pathProbeResolutions;
    private long _pathProbeMisses;

    public long ResolvedPathCacheHits => _resolvedPathCacheHits;
    public long PathProbeAttempts => _pathProbeAttempts;
    public long PathProbeResolutions => _pathProbeResolutions;
    public long PathProbeMisses => _pathProbeMisses;
    public int ResolvedPathCacheCount => _resolvedReadPathCache.Count;

    public WorldAssetPathResolver(IDataSource? dataSource)
    {
        _dataSource = dataSource;
    }

    public static string NormalizeKey(string path) => path.Replace('/', '\\').ToLowerInvariant();

    public bool TryGetCachedResolvedPath(string key, out string? cachedResolvedPath)
        => _resolvedReadPathCache.TryGetValue(key, out cachedResolvedPath);

    public byte[]? ResolveAndReadFile(string key, out string? resolvedPath)
    {
        byte[]? data = null;
        resolvedPath = null;

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

        return data;
    }

    public string ResolveCanonicalModelPath(string normalizedKey)
    {
        string? resolved = TryResolveFromFileSet(normalizedKey);
        if (!string.IsNullOrWhiteSpace(resolved))
            return NormalizeKey(resolved);

        foreach (string alternatePath in GetAlternateModelPaths(normalizedKey))
        {
            resolved = TryResolveFromFileSet(alternatePath);
            if (!string.IsNullOrWhiteSpace(resolved))
                return NormalizeKey(resolved);
        }

        if (_dataSource != null)
        {
            if (_dataSource.FileExists(normalizedKey))
                return normalizedKey;

            foreach (string alternatePath in GetAlternateModelPaths(normalizedKey))
            {
                if (_dataSource.FileExists(alternatePath))
                    return NormalizeKey(alternatePath);
            }
        }

        return normalizedKey;
    }

    public bool TryReadPreferredClassicModelData(string normalizedKey, out string resolvedPath, out byte[]? data)
    {
        resolvedPath = normalizedKey;
        data = null;

        if (!IsClassicModelRequest(normalizedKey))
            return false;

        foreach (string candidate in EnumeratePreferredClassicModelPaths(normalizedKey))
        {
            data = TryReadExactCandidate(candidate, out string? exactResolvedPath);
            if (data == null || data.Length == 0 || string.IsNullOrWhiteSpace(exactResolvedPath))
                continue;

            resolvedPath = NormalizeKey(exactResolvedPath);
            return true;
        }

        data = null;
        resolvedPath = normalizedKey;
        return false;
    }

    public IEnumerable<string> GetAlternateModelPaths(string path)
    {
        if (path.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase))
        {
            yield return path[..^4] + ".m2";
            yield return path[..^4] + ".mdl";
            yield break;
        }

        if (path.EndsWith(".mdl", StringComparison.OrdinalIgnoreCase))
        {
            yield return path[..^4] + ".mdx";
            yield return path[..^4] + ".m2";
            yield break;
        }

        if (path.EndsWith(".m2", StringComparison.OrdinalIgnoreCase))
        {
            yield return path[..^3] + ".mdx";
            yield return path[..^3] + ".mdl";
        }
    }

    public static string? SwapMdlMdxExtension(string path)
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

    private static bool IsClassicModelRequest(string path)
    {
        string extension = Path.GetExtension(path);
        return extension.Equals(".mdx", StringComparison.OrdinalIgnoreCase)
            || extension.Equals(".mdl", StringComparison.OrdinalIgnoreCase);
    }

    private static bool IsModelRequest(string path)
    {
        string extension = Path.GetExtension(path);
        return extension.Equals(".mdx", StringComparison.OrdinalIgnoreCase)
            || extension.Equals(".mdl", StringComparison.OrdinalIgnoreCase)
            || extension.Equals(".m2", StringComparison.OrdinalIgnoreCase);
    }

    private static IEnumerable<string> EnumeratePreferredClassicModelPaths(string normalizedPath)
    {
        yield return normalizedPath;

        foreach (string alt in GetAlternateModelPathsStatic(normalizedPath))
        {
            if (!alt.Equals(normalizedPath, StringComparison.OrdinalIgnoreCase))
                yield return alt;
        }
    }

    private static IEnumerable<string> GetAlternateModelPathsStatic(string path)
    {
        if (path.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase))
        {
            yield return path[..^4] + ".m2";
            yield return path[..^4] + ".mdl";
            yield break;
        }

        if (path.EndsWith(".mdl", StringComparison.OrdinalIgnoreCase))
        {
            yield return path[..^4] + ".mdx";
            yield return path[..^4] + ".m2";
            yield break;
        }

        if (path.EndsWith(".m2", StringComparison.OrdinalIgnoreCase))
        {
            yield return path[..^3] + ".mdx";
            yield return path[..^3] + ".mdl";
        }
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
        if (string.IsNullOrWhiteSpace(baseName) || !IsModelRequest(normalizedPath))
            return null;

        var indexedMatch = mpqDataSource.FindByBaseName(baseName, GetLikelyModelExtensions(normalizedPath));
        if (!string.IsNullOrWhiteSpace(indexedMatch))
            return NormalizeKey(indexedMatch);

        return null;
    }

    private static IEnumerable<string> BuildFileSetCandidates(string normalizedPath)
    {
        yield return normalizedPath;

        foreach (string alternate in GetAlternateModelPathsStatic(normalizedPath))
            yield return alternate;

        string fileName = Path.GetFileName(normalizedPath);
        if (!string.IsNullOrWhiteSpace(fileName) && !fileName.Equals(normalizedPath, StringComparison.OrdinalIgnoreCase))
        {
            yield return fileName;

            foreach (string alternate in GetAlternateModelPathsStatic(fileName))
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

    private byte[]? TryReadExactCandidate(string candidate, out string? resolvedPath)
    {
        resolvedPath = NormalizeKey(candidate);
        _pathProbeAttempts++;

        byte[]? data = _dataSource?.ReadFile(resolvedPath);
        if (data != null)
        {
            _pathProbeResolutions++;
            return data;
        }

        if (_dataSource is MpqDataSource mpqDataSource)
        {
            string? found = mpqDataSource.FindInFileSet(resolvedPath);
            if (!string.IsNullOrWhiteSpace(found))
            {
                string normalizedFound = NormalizeKey(found);
                if (!normalizedFound.Equals(resolvedPath, StringComparison.OrdinalIgnoreCase))
                    data = _dataSource.ReadFile(normalizedFound);

                if (data != null)
                {
                    resolvedPath = normalizedFound;
                    _pathProbeResolutions++;
                    return data;
                }
            }
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

        foreach (string alternatePath in GetAlternateModelPathsStatic(normalizedPath))
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

    public void Clear()
    {
        _resolvedReadPathCache.Clear();
    }
}
