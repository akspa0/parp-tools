using System.Diagnostics;
using WowViewer.Core.IO.Dbc;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.App;

internal sealed record WowViewerWorldSessionOpenRequest(string ClientRoot, string MapInput, string BuildLabel);

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
        (byte[] data, string sourcePath, bool loadedFromArchive) = ReadWdt(clientRoot, resolvedMapDirectory, wdtVirtualPath, archiveCatalog);

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
            fileSummary,
            wdtSummary,
            occupiedTiles,
            stopwatch.Elapsed);
    }

    private static (byte[] Data, string SourcePath, bool LoadedFromArchive) ReadWdt(string clientRoot, string mapDirectory, string wdtVirtualPath, IArchiveCatalog archiveCatalog)
    {
        string loosePath = Path.Combine(clientRoot, "World", "Maps", mapDirectory, $"{mapDirectory}.wdt");
        if (File.Exists(loosePath))
            return (File.ReadAllBytes(loosePath), Path.GetFullPath(loosePath), false);

        byte[]? archiveData = archiveCatalog.ReadFile(wdtVirtualPath) ?? archiveCatalog.ReadFile(wdtVirtualPath.Replace('\\', '/'));
        if (archiveData is { Length: > 0 })
            return (archiveData, wdtVirtualPath, true);

        throw new FileNotFoundException($"Could not find WDT for map '{mapDirectory}' at '{wdtVirtualPath}' under client root '{clientRoot}'.", loosePath);
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