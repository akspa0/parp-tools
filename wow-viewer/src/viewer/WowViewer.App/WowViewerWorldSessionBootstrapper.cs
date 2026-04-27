using System.Diagnostics;
using System.Collections.Concurrent;
using WowViewer.Core.IO.Dbc;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.App;

internal sealed record WowViewerWorldSessionOpenRequest(string ClientRoot, string MapInput, string BuildLabel, string LooseOverlayRoot);

internal readonly record struct WowViewerWorldSessionBootstrapTelemetry(
    WowViewerWorldSessionBootstrapResult Session,
    bool CacheHit,
    TimeSpan ResolveDuration);

internal sealed class WowViewerWorldSessionBootstrapResult
{
    public WowViewerWorldSessionBootstrapResult(
        string clientRoot,
        string requestedMapInput,
        string resolvedMapDirectory,
        bool resolvedViaDbc,
        bool usedMapDirectoryLookup,
        string buildLabel,
        string wdtVirtualPath,
        string wdtSourcePath,
        bool loadedFromArchive,
        string looseOverlayRoot,
        MapFileSummary fileSummary,
        WdtSummary wdtSummary,
        IReadOnlyList<WdtTileCoordinate> occupiedTiles,
        TimeSpan loadDuration)
    {
        ClientRoot = clientRoot;
        RequestedMapInput = requestedMapInput;
        ResolvedMapDirectory = resolvedMapDirectory;
        ResolvedViaDbc = resolvedViaDbc;
        UsedMapDirectoryLookup = usedMapDirectoryLookup;
        BuildLabel = buildLabel;
        WdtVirtualPath = wdtVirtualPath;
        WdtSourcePath = wdtSourcePath;
        LoadedFromArchive = loadedFromArchive;
        LooseOverlayRoot = looseOverlayRoot;
        FileSummary = fileSummary;
        WdtSummary = wdtSummary;
        OccupiedTiles = occupiedTiles;
        LoadDuration = loadDuration;
    }

    public string ClientRoot { get; }

    public string RequestedMapInput { get; }

    public string ResolvedMapDirectory { get; }

    public bool ResolvedViaDbc { get; }

    public bool UsedMapDirectoryLookup { get; }

    public string BuildLabel { get; }

    public string WdtVirtualPath { get; }

    public string WdtSourcePath { get; }

    public bool LoadedFromArchive { get; }

    public string LooseOverlayRoot { get; }

    public MapFileSummary FileSummary { get; }

    public WdtSummary WdtSummary { get; }

    public IReadOnlyList<WdtTileCoordinate> OccupiedTiles { get; }

    public TimeSpan LoadDuration { get; }
}

internal static class WowViewerWorldSessionBootstrapper
{
    private static readonly ConcurrentDictionary<string, WowViewerWorldSessionBootstrapResult> SessionCache = new(StringComparer.OrdinalIgnoreCase);

    public static WowViewerWorldSessionBootstrapResult Open(WowViewerWorldSessionOpenRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);

        string clientRoot = Path.GetFullPath(request.ClientRoot);
        if (!Directory.Exists(clientRoot))
            throw new DirectoryNotFoundException($"Client root does not exist: {clientRoot}");

        ArchiveCatalogSession session = ArchiveCatalogSessionCache.GetOrCreate(
            [clientRoot],
            WowViewerArchiveBootstrap.CreateBootstrapOptions(request.BuildLabel, clientRoot));

        return Open(request, session.ArchiveCatalog, clientRoot);
    }

    internal static WowViewerWorldSessionBootstrapResult Open(WowViewerWorldSessionOpenRequest request, IArchiveCatalog archiveCatalog)
    {
        return OpenWithTelemetry(request, archiveCatalog).Session;
    }

    internal static WowViewerWorldSessionBootstrapTelemetry OpenWithTelemetry(WowViewerWorldSessionOpenRequest request, IArchiveCatalog archiveCatalog)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(archiveCatalog);

        Stopwatch stopwatch = Stopwatch.StartNew();

        string cacheKey = BuildCacheKey(request);
        if (SessionCache.TryGetValue(cacheKey, out WowViewerWorldSessionBootstrapResult? cached))
            return new WowViewerWorldSessionBootstrapTelemetry(cached, CacheHit: true, stopwatch.Elapsed);

        string clientRoot = Path.GetFullPath(request.ClientRoot);
        if (!Directory.Exists(clientRoot))
            throw new DirectoryNotFoundException($"Client root does not exist: {clientRoot}");

        WowViewerWorldSessionBootstrapResult result = Open(request, archiveCatalog, clientRoot);
        SessionCache[cacheKey] = result;
        return new WowViewerWorldSessionBootstrapTelemetry(result, CacheHit: false, stopwatch.Elapsed);
    }

    private static WowViewerWorldSessionBootstrapResult Open(WowViewerWorldSessionOpenRequest request, IArchiveCatalog archiveCatalog, string clientRoot)
    {
        ArgumentNullException.ThrowIfNull(request);

        string looseOverlayRoot = string.IsNullOrWhiteSpace(request.LooseOverlayRoot)
            ? string.Empty
            : Path.GetFullPath(request.LooseOverlayRoot);
        if (!string.IsNullOrWhiteSpace(looseOverlayRoot) && !Directory.Exists(looseOverlayRoot))
            throw new DirectoryNotFoundException($"Loose overlay root does not exist: {looseOverlayRoot}");

        if (string.IsNullOrWhiteSpace(request.MapInput))
            throw new ArgumentException("Provide a map directory, map id, or Map.dbc name via --map.", nameof(request.MapInput));

        Stopwatch stopwatch = Stopwatch.StartNew();

        MapDirectoryLookup directoryLookup = new();
        directoryLookup.Load(
        new[]
        {
            looseOverlayRoot,
            clientRoot,
        }.Where(static path => !string.IsNullOrWhiteSpace(path)), archiveCatalog);

        string requestedMapInput = request.MapInput.Trim();
        string directDirectory = ExtractMapDirectory(requestedMapInput);
        string? resolvedFromDbc = directoryLookup.ResolveDirectory(requestedMapInput);
        if (string.IsNullOrWhiteSpace(resolvedFromDbc) && !string.Equals(directDirectory, requestedMapInput, StringComparison.OrdinalIgnoreCase))
            resolvedFromDbc = directoryLookup.ResolveDirectory(directDirectory);

        string? archiveAlias = TryResolveArchiveMapDirectoryAlias(requestedMapInput, archiveCatalog.GetAllKnownFiles());
        if (string.IsNullOrWhiteSpace(archiveAlias) && !string.Equals(directDirectory, requestedMapInput, StringComparison.OrdinalIgnoreCase))
            archiveAlias = TryResolveArchiveMapDirectoryAlias(directDirectory, archiveCatalog.GetAllKnownFiles());

        string canonicalMapDirectory = !string.IsNullOrWhiteSpace(resolvedFromDbc)
            ? resolvedFromDbc
            : !string.IsNullOrWhiteSpace(archiveAlias)
                ? archiveAlias
                : directDirectory;

        string resolvedMapDirectory = canonicalMapDirectory;
        string wdtVirtualPath = $@"World\Maps\{resolvedMapDirectory}\{resolvedMapDirectory}.wdt";
        string sourcePath = string.Empty;
        bool loadedFromArchive = false;
        MapFileSummary fileSummary;
        WdtSummary wdtSummary;
        IReadOnlyList<WdtTileCoordinate> occupiedTiles = Array.Empty<WdtTileCoordinate>();
        FileNotFoundException? missingWdt = null;

        foreach (string candidateMapDirectory in DistinctMapDirectories(directDirectory, canonicalMapDirectory))
        {
            string candidateWdtVirtualPath = $@"World\Maps\{candidateMapDirectory}\{candidateMapDirectory}.wdt";

            try
            {
                (byte[] data, sourcePath, loadedFromArchive) = ReadWdt(clientRoot, looseOverlayRoot, candidateMapDirectory, candidateWdtVirtualPath, archiveCatalog);

                using MemoryStream stream = new(data, writable: false);
                fileSummary = MapFileSummaryReader.Read(stream, loadedFromArchive ? candidateWdtVirtualPath : sourcePath);
                stream.Position = 0;
                wdtSummary = WdtSummaryReader.Read(stream, fileSummary);
                stream.Position = 0;
                occupiedTiles = WdtTileIndexReader.ReadOccupiedTiles(stream, fileSummary);
                resolvedMapDirectory = candidateMapDirectory;
                wdtVirtualPath = candidateWdtVirtualPath;
                goto Success;
            }
            catch (FileNotFoundException ex)
            {
                missingWdt = ex;
            }
        }

        foreach (string candidateMapDirectory in DistinctMapDirectories(directDirectory, canonicalMapDirectory))
        {
            if (!TryProbeOccupiedAdtTiles(clientRoot, looseOverlayRoot, candidateMapDirectory, archiveCatalog, out occupiedTiles, out string adtProbeSourcePath, out loadedFromArchive))
                continue;

            resolvedMapDirectory = candidateMapDirectory;
            wdtVirtualPath = $@"World\Maps\{candidateMapDirectory}\{candidateMapDirectory}.wdt";
            sourcePath = adtProbeSourcePath;
            string syntheticSourcePath = $@"{wdtVirtualPath} [synthesized from ADT probes]";
            fileSummary = new MapFileSummary(syntheticSourcePath, MapFileKind.Unknown, null, Array.Empty<MapChunkLocation>());
            wdtSummary = new WdtSummary(
                syntheticSourcePath,
                isWmoBased: false,
                tilesWithData: occupiedTiles.Count,
                totalTiles: 64 * 64,
                mainCellSizeBytes: 0,
                doodadNameCount: 0,
                worldModelNameCount: 0,
                doodadPlacementCount: 0,
                worldModelPlacementCount: 0,
                mainFlags: null);
            goto Success;
        }

        throw missingWdt ?? new FileNotFoundException($"Could not find WDT for map '{canonicalMapDirectory}' under client root '{clientRoot}'.");

    Success:
        stopwatch.Stop();
        return new WowViewerWorldSessionBootstrapResult(
            clientRoot,
            requestedMapInput,
            resolvedMapDirectory,
            !string.IsNullOrWhiteSpace(resolvedFromDbc),
            directoryLookup.IsLoaded,
            request.BuildLabel?.Trim() ?? string.Empty,
            wdtVirtualPath,
            sourcePath,
            loadedFromArchive,
            looseOverlayRoot,
            fileSummary,
            wdtSummary,
            occupiedTiles,
            stopwatch.Elapsed);
    }

    private static string BuildCacheKey(WowViewerWorldSessionOpenRequest request)
    {
        ViewerIoSourceKey sourceKey = ViewerIoSourceKey.Create(request.ClientRoot, request.BuildLabel, request.LooseOverlayRoot);
        return string.Join('|', sourceKey.Signature, request.MapInput.Trim());
    }

    private static (byte[] Data, string SourcePath, bool LoadedFromArchive) ReadWdt(string clientRoot, string looseOverlayRoot, string mapDirectory, string wdtVirtualPath, IArchiveCatalog archiveCatalog)
    {
        if (VirtualAssetOverlayResolver.TryReadLooseVirtualFile(wdtVirtualPath, looseOverlayRoot, out byte[]? overlayData, out string overlaySourcePath)
            && overlayData is { Length: > 0 })
        {
            return (overlayData, overlaySourcePath, false);
        }

        foreach ((string path, bool isPerAssetMpq) in EnumerateDiskWdtCandidates(clientRoot, mapDirectory))
        {
            if (!File.Exists(path))
                continue;

            if (!isPerAssetMpq)
                return (File.ReadAllBytes(path), Path.GetFullPath(path), false);

            if (archiveCatalog is MpqArchiveCatalog mpqArchiveCatalog)
            {
                byte[]? payload = mpqArchiveCatalog.ReadFile0FromPath(path, wdtVirtualPath, $"{mapDirectory}.wdt");
                if (payload is { Length: > 0 })
                    return (payload, Path.GetFullPath(path), true);
            }
        }

        byte[]? archiveData = archiveCatalog.ReadFile(wdtVirtualPath) ?? archiveCatalog.ReadFile(wdtVirtualPath.Replace('\\', '/'));
        if (archiveData is { Length: > 0 })
            return (archiveData, wdtVirtualPath, true);

        string wdtMpqVirtualPath = wdtVirtualPath + ".MPQ";
        byte[]? perAssetArchiveData = archiveCatalog.ReadFile(wdtMpqVirtualPath) ?? archiveCatalog.ReadFile(wdtMpqVirtualPath.Replace('\\', '/'));
        if (perAssetArchiveData is { Length: > 0 })
            return (perAssetArchiveData, wdtMpqVirtualPath, true);

        throw new FileNotFoundException($"Could not find WDT for map '{mapDirectory}' at '{wdtVirtualPath}' under client root '{clientRoot}'.", Path.Combine(clientRoot, "Data", "World", "Maps", mapDirectory, $"{mapDirectory}.wdt.MPQ"));
    }

    private static IEnumerable<(string Path, bool IsPerAssetMpq)> EnumerateDiskWdtCandidates(string clientRoot, string mapDirectory)
    {
        string[] baseDirectories =
        [
            Path.Combine(clientRoot, "World", "Maps", mapDirectory),
            Path.Combine(clientRoot, "Data", "World", "Maps", mapDirectory),
        ];

        foreach (string baseDirectory in baseDirectories.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            yield return (Path.Combine(baseDirectory, $"{mapDirectory}.wdt"), false);
            yield return (Path.Combine(baseDirectory, $"{mapDirectory}.wdt.MPQ"), true);
        }
    }

    private static bool TryProbeOccupiedAdtTiles(
        string clientRoot,
        string looseOverlayRoot,
        string mapDirectory,
        IArchiveCatalog archiveCatalog,
        out IReadOnlyList<WdtTileCoordinate> occupiedTiles,
        out string sourcePath,
        out bool loadedFromArchive)
    {
        List<WdtTileCoordinate> discovered = [];
        sourcePath = string.Empty;
        loadedFromArchive = false;

        for (int tileY = 0; tileY < 64; tileY++)
        {
            for (int tileX = 0; tileX < 64; tileX++)
            {
                string adtVirtualPath = BuildStandardAdtVirtualPath(mapDirectory, tileX, tileY);
                if (!TryResolveAdtProbeSource(clientRoot, looseOverlayRoot, adtVirtualPath, mapDirectory, tileX, tileY, archiveCatalog, out string tileSourcePath, out bool tileLoadedFromArchive))
                    continue;

                discovered.Add(new WdtTileCoordinate(tileX, tileY));
                if (string.IsNullOrWhiteSpace(sourcePath))
                {
                    sourcePath = tileSourcePath;
                    loadedFromArchive = tileLoadedFromArchive;
                }
            }
        }

        occupiedTiles = discovered;
        return discovered.Count > 0;
    }

    private static bool TryResolveAdtProbeSource(
        string clientRoot,
        string looseOverlayRoot,
        string adtVirtualPath,
        string mapDirectory,
        int tileX,
        int tileY,
        IArchiveCatalog archiveCatalog,
        out string sourcePath,
        out bool loadedFromArchive)
    {
        loadedFromArchive = false;
        if (VirtualAssetOverlayResolver.TryReadLooseVirtualFile(adtVirtualPath, looseOverlayRoot, out _, out sourcePath))
            return true;

        foreach (string path in EnumerateDiskAdtCandidates(clientRoot, mapDirectory, tileX, tileY))
        {
            if (!File.Exists(path))
                continue;

            sourcePath = Path.GetFullPath(path);
            return true;
        }

        byte[]? archiveData = archiveCatalog.ReadFile(adtVirtualPath) ?? archiveCatalog.ReadFile(adtVirtualPath.Replace('\\', '/'));
        if (archiveData is { Length: > 0 })
        {
            sourcePath = adtVirtualPath;
            loadedFromArchive = true;
            return true;
        }

        sourcePath = string.Empty;
        return false;
    }

    private static IEnumerable<string> EnumerateDiskAdtCandidates(string clientRoot, string mapDirectory, int tileX, int tileY)
    {
        string fileName = $"{mapDirectory}_{tileY}_{tileX}.adt";
        string[] baseDirectories =
        [
            Path.Combine(clientRoot, "World", "Maps", mapDirectory),
            Path.Combine(clientRoot, "Data", "World", "Maps", mapDirectory),
        ];

        foreach (string baseDirectory in baseDirectories.Distinct(StringComparer.OrdinalIgnoreCase))
            yield return Path.Combine(baseDirectory, fileName);
    }

    private static string BuildStandardAdtVirtualPath(string mapDirectory, int tileX, int tileY)
    {
        // Match MdxViewer's row-major convention: tileX is row (y), tileY is column (x),
        // while on-disk ADT families are named Map_x_y.
        return $@"World\Maps\{mapDirectory}\{mapDirectory}_{tileY}_{tileX}.adt";
    }

    private static string ExtractMapDirectory(string mapInput)
    {
        string normalized = mapInput.Trim().Replace('/', '\\');
        if (normalized.EndsWith(".wdt", StringComparison.OrdinalIgnoreCase))
            return Path.GetFileNameWithoutExtension(normalized);

        string trimmed = normalized.TrimEnd('\\');
        if (trimmed.Contains('\\', StringComparison.Ordinal))
            return Path.GetFileName(trimmed);

        return trimmed;
    }

    private static IEnumerable<string> DistinctMapDirectories(string requestedDirectory, string resolvedDirectory)
    {
        HashSet<string> seen = new(StringComparer.OrdinalIgnoreCase);
        if (!string.IsNullOrWhiteSpace(requestedDirectory) && seen.Add(requestedDirectory))
            yield return requestedDirectory;

        if (!string.IsNullOrWhiteSpace(resolvedDirectory) && seen.Add(resolvedDirectory))
            yield return resolvedDirectory;
    }

    private static string? TryResolveArchiveMapDirectoryAlias(string mapName, IReadOnlyList<string> knownFiles)
    {
        if (string.IsNullOrWhiteSpace(mapName) || knownFiles.Count == 0)
            return null;

        string requestedToken = NormalizeMapToken(mapName);
        if (requestedToken.Length == 0)
            return null;

        HashSet<string> candidates = new(StringComparer.OrdinalIgnoreCase);
        foreach (string knownFile in knownFiles)
        {
            if (string.IsNullOrWhiteSpace(knownFile))
                continue;

            string normalizedPath = knownFile.Replace('\\', '/');
            if (!normalizedPath.StartsWith("World/Maps/", StringComparison.OrdinalIgnoreCase)
                || !normalizedPath.EndsWith(".wdt", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            string[] segments = normalizedPath.Split('/', StringSplitOptions.RemoveEmptyEntries);
            if (segments.Length < 4)
                continue;

            string directoryName = segments[2];
            string fileName = Path.GetFileNameWithoutExtension(segments[^1]);
            if (!string.Equals(directoryName, fileName, StringComparison.OrdinalIgnoreCase))
                continue;

            candidates.Add(directoryName);
        }

        foreach (string candidate in candidates)
        {
            if (string.Equals(NormalizeMapToken(candidate), requestedToken, StringComparison.Ordinal))
                return candidate;
        }

        string? bestCandidate = null;
        int bestDistance = int.MaxValue;
        bool isAmbiguous = false;

        foreach (string candidate in candidates)
        {
            string candidateToken = NormalizeMapToken(candidate);
            if (candidateToken.Length == 0 || candidateToken[0] != requestedToken[0])
                continue;

            if (Math.Abs(candidateToken.Length - requestedToken.Length) > 2)
                continue;

            int distance = ComputeLevenshteinDistance(requestedToken, candidateToken, 2);
            if (distance > 2)
                continue;

            if (distance < bestDistance)
            {
                bestDistance = distance;
                bestCandidate = candidate;
                isAmbiguous = false;
            }
            else if (distance == bestDistance && !string.Equals(bestCandidate, candidate, StringComparison.OrdinalIgnoreCase))
            {
                isAmbiguous = true;
            }
        }

        return isAmbiguous ? null : bestCandidate;
    }

    private static string NormalizeMapToken(string value)
    {
        Span<char> buffer = stackalloc char[value.Length];
        int length = 0;
        foreach (char ch in value)
        {
            if (!char.IsLetterOrDigit(ch))
                continue;

            buffer[length++] = char.ToLowerInvariant(ch);
        }

        return length == 0 ? string.Empty : new string(buffer[..length]);
    }

    private static int ComputeLevenshteinDistance(string source, string target, int maxDistance)
    {
        int sourceLength = source.Length;
        int targetLength = target.Length;

        if (sourceLength == 0)
            return targetLength;
        if (targetLength == 0)
            return sourceLength;
        if (Math.Abs(sourceLength - targetLength) > maxDistance)
            return maxDistance + 1;

        int[] previous = new int[targetLength + 1];
        int[] current = new int[targetLength + 1];

        for (int j = 0; j <= targetLength; j++)
            previous[j] = j;

        for (int i = 1; i <= sourceLength; i++)
        {
            current[0] = i;
            int rowMin = current[0];

            for (int j = 1; j <= targetLength; j++)
            {
                int substitutionCost = source[i - 1] == target[j - 1] ? 0 : 1;
                current[j] = Math.Min(
                    Math.Min(previous[j] + 1, current[j - 1] + 1),
                    previous[j - 1] + substitutionCost);
                rowMin = Math.Min(rowMin, current[j]);
            }

            if (rowMin > maxDistance)
                return maxDistance + 1;

            (previous, current) = (current, previous);
        }

        return previous[targetLength];
    }
}