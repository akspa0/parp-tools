using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;

namespace WowViewer.Core.Tests;

internal static class AdtRealDataTestCatalog
{
    private static readonly string[] CandidateMapNames =
    [
        "azeroth",
        "kalimdor",
        "northrend",
        "outland",
        "expansion01",
        "expansion02",
        "development",
    ];

    public static string ListfilePath => Path.Combine(GetWowViewerRoot(), "libs", "wowdev", "wow-listfile", "listfile.txt");

    public static IReadOnlyList<StagedClientSurface> GetStagedClients()
    {
        string stagedRoot = Path.Combine(GetWowViewerRoot(), "..", "output", "tmp", "wowarchive-clients");
        if (!Directory.Exists(stagedRoot))
            return [];

        return Directory.GetDirectories(stagedRoot)
            .Select(static path => new DirectoryInfo(path))
            .Select(static dir => new StagedClientSurface(dir.Name.Replace('_', '.'), Path.Combine(dir.FullName, "World of Warcraft", "Data")))
            .Where(static client => Directory.Exists(client.DataPath))
            .OrderBy(static client => client.BuildVersion)
            .ToArray();
    }

    public static MpqArchiveCatalog CreateCatalog(StagedClientSurface client)
    {
        MpqArchiveCatalog catalog = new();
        catalog.LoadArchives([client.DataPath]);
        catalog.LoadListfile(ListfilePath);
        return catalog;
    }

    public static IReadOnlyList<string> GetCandidateRootAdts(MpqArchiveCatalog catalog, int maxTilesPerMap = 256)
    {
        List<string> candidates = [];
        foreach (string mapName in CandidateMapNames)
        {
            string wdtVirtualPath = $"world\\maps\\{mapName}\\{mapName}.wdt";
            byte[]? wdtBytes = catalog.ReadFile(wdtVirtualPath);
            if (wdtBytes is null)
                continue;

            string tempWdtPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}_{mapName}.wdt");
            try
            {
                File.WriteAllBytes(tempWdtPath, wdtBytes);
                candidates.AddRange(WdtTileIndexReader.ReadOccupiedTiles(tempWdtPath)
                    .Select(tile => $"world\\maps\\{mapName}\\{mapName}_{tile.TileX}_{tile.TileY}.adt")
                    .Take(maxTilesPerMap));
            }
            catch (InvalidDataException)
            {
            }
            finally
            {
                if (File.Exists(tempWdtPath))
                    File.Delete(tempWdtPath);
            }
        }

        return candidates;
    }

    private static string GetWowViewerRoot()
    {
        DirectoryInfo? current = new(AppContext.BaseDirectory);
        while (current is not null)
        {
            if (File.Exists(Path.Combine(current.FullName, "WowViewer.slnx")))
                return current.FullName;

            current = current.Parent;
        }

        throw new DirectoryNotFoundException("Could not locate the wow-viewer repository root from the test output directory.");
    }
}

internal sealed record StagedClientSurface(string BuildVersion, string DataPath);