using WowViewer.Core.IO.Dbc;
using WowViewer.Core.IO.Files;

namespace WowViewer.Core.IO.Maps;

public sealed record DiscoveredLooseWorldMap(
    int Id,
    string Directory,
    string Name,
    bool HasLooseWdt,
    bool HasLooseWdl,
    string? LooseSourceDirectory)
{
    public bool HasLooseFiles => HasLooseWdt || HasLooseWdl;
}

public static class LooseWorldMapDiscovery
{
    public static IReadOnlyList<DiscoveredLooseWorldMap> Discover(string clientRoot, string? looseOverlayRoot = null)
    {
        string normalizedClientRoot = Path.GetFullPath(clientRoot);
        if (!Directory.Exists(normalizedClientRoot))
            return Array.Empty<DiscoveredLooseWorldMap>();

        string normalizedLooseOverlayRoot = string.IsNullOrWhiteSpace(looseOverlayRoot)
            ? string.Empty
            : Path.GetFullPath(looseOverlayRoot);

        string[] lookupSearchPaths = BuildLookupSearchPaths(normalizedClientRoot, normalizedLooseOverlayRoot);
        using IArchiveCatalog archiveCatalog = new MpqArchiveCatalogFactory().Create();
        ArchiveCatalogBootstrapper.Bootstrap(archiveCatalog, [normalizedClientRoot], new ArchiveCatalogBootstrapOptions());

        MapDirectoryLookup mapLookup = new();
        mapLookup.Load(lookupSearchPaths, archiveCatalog);
        if (!mapLookup.IsLoaded)
            return Array.Empty<DiscoveredLooseWorldMap>();

        return mapLookup.Entries
            .Select(entry => CreateDiscoveredMap(entry, lookupSearchPaths))
            .OrderBy(static map => map.Name, StringComparer.OrdinalIgnoreCase)
            .ThenBy(static map => map.Directory, StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }

    private static string[] BuildLookupSearchPaths(string clientRoot, string looseOverlayRoot)
    {
        return new[] { looseOverlayRoot, clientRoot }
            .Where(static path => !string.IsNullOrWhiteSpace(path))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }

    private static DiscoveredLooseWorldMap CreateDiscoveredMap(MapDirectoryEntry entry, IReadOnlyList<string> searchRoots)
    {
        foreach (string root in searchRoots)
        {
            foreach (string directoryPath in EnumerateCandidateMapDirectories(root, entry.Directory))
            {
                bool hasLooseWdt = File.Exists(Path.Combine(directoryPath, entry.Directory + ".wdt"))
                    || File.Exists(Path.Combine(directoryPath, entry.Directory + ".wdt.MPQ"));
                bool hasLooseWdl = File.Exists(Path.Combine(directoryPath, entry.Directory + ".wdl"))
                    || File.Exists(Path.Combine(directoryPath, entry.Directory + ".wdl.MPQ"));

                if (!hasLooseWdt && !hasLooseWdl)
                    continue;

                return new DiscoveredLooseWorldMap(
                    entry.Id,
                    entry.Directory,
                    entry.Name,
                    hasLooseWdt,
                    hasLooseWdl,
                    directoryPath);
            }
        }

        return new DiscoveredLooseWorldMap(entry.Id, entry.Directory, entry.Name, false, false, null);
    }

    private static IEnumerable<string> EnumerateCandidateMapDirectories(string root, string directory)
    {
        foreach (string candidate in new[]
        {
            Path.Combine(root, "World", "Maps", directory),
            Path.Combine(root, "Data", "World", "Maps", directory),
        }.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            if (Directory.Exists(candidate))
                yield return candidate;
        }
    }
}