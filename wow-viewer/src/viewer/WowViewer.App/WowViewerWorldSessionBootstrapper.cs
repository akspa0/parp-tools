using System.Diagnostics;
using WowViewer.Core.IO.Dbc;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.App;

internal sealed record WowViewerWorldSessionOpenRequest(string ClientRoot, string MapInput, string BuildLabel, string LooseOverlayRoot);

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
    public static WowViewerWorldSessionBootstrapResult Open(WowViewerWorldSessionOpenRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);

        string clientRoot = Path.GetFullPath(request.ClientRoot);
        if (!Directory.Exists(clientRoot))
            throw new DirectoryNotFoundException($"Client root does not exist: {clientRoot}");

        string looseOverlayRoot = string.IsNullOrWhiteSpace(request.LooseOverlayRoot)
            ? string.Empty
            : Path.GetFullPath(request.LooseOverlayRoot);
        if (!string.IsNullOrWhiteSpace(looseOverlayRoot) && !Directory.Exists(looseOverlayRoot))
            throw new DirectoryNotFoundException($"Loose overlay root does not exist: {looseOverlayRoot}");

        if (string.IsNullOrWhiteSpace(request.MapInput))
            throw new ArgumentException("Provide a map directory, map id, or Map.dbc name via --map.", nameof(request.MapInput));

        Stopwatch stopwatch = Stopwatch.StartNew();
        using IArchiveCatalog archiveCatalog = new MpqArchiveCatalogFactory().Create();
        ArchiveCatalogBootstrapper.Bootstrap(archiveCatalog, [clientRoot], new ArchiveCatalogBootstrapOptions());

        MapDirectoryLookup directoryLookup = new();
        directoryLookup.Load([clientRoot], archiveCatalog);

        string requestedMapInput = request.MapInput.Trim();
        string directDirectory = ExtractMapDirectory(requestedMapInput);
        string? resolvedFromDbc = directoryLookup.ResolveDirectory(requestedMapInput);
        if (string.IsNullOrWhiteSpace(resolvedFromDbc) && !string.Equals(directDirectory, requestedMapInput, StringComparison.OrdinalIgnoreCase))
            resolvedFromDbc = directoryLookup.ResolveDirectory(directDirectory);

        string resolvedMapDirectory = string.IsNullOrWhiteSpace(resolvedFromDbc) ? directDirectory : resolvedFromDbc;
        string wdtVirtualPath = $@"World\Maps\{resolvedMapDirectory}\{resolvedMapDirectory}.wdt";
        (byte[] data, string sourcePath, bool loadedFromArchive) = ReadWdt(clientRoot, looseOverlayRoot, resolvedMapDirectory, wdtVirtualPath, archiveCatalog);

        using MemoryStream stream = new(data, writable: false);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, loadedFromArchive ? wdtVirtualPath : sourcePath);
        stream.Position = 0;
        WdtSummary wdtSummary = WdtSummaryReader.Read(stream, fileSummary);
        stream.Position = 0;
        IReadOnlyList<WdtTileCoordinate> occupiedTiles = WdtTileIndexReader.ReadOccupiedTiles(stream, fileSummary);

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
}