using WowViewer.Core.IO.Files;

namespace WowViewer.Tool.Inspect;

/// <summary>
/// Audits <see cref="MpqArchiveCatalog.ScanWmoMpqArchives"/> — the scan that finds every per-asset
/// <c>*.wmo.mpq</c> container in a loose-tree build — against an independent, unscoped walk of the
/// same game root, to check whether the seven hardcoded candidate directories that scan searches
/// actually cover every container that exists.
/// </summary>
/// <remarks>
/// Spec 155's first real sweep of 0.5.3.3368 examined 492 world objects against an earlier, informally
/// obtained figure of 532 per-asset containers. The 492 figure comes directly from
/// <see cref="MpqArchiveCatalog.ScanWmoMpqArchives"/>, unmodified — this command does not reimplement
/// that scan, it calls it, then separately walks the whole game root for the same filename pattern and
/// reports anything the scoped scan's candidate directories would have missed.
/// </remarks>
public static class ArchiveWmoContainerAuditCommandSupport
{
    public static void Run(string[] args)
    {
        string? gamePath = GetOption(args, "--archive-root", "-r");
        if (string.IsNullOrWhiteSpace(gamePath))
        {
            Console.Error.WriteLine("Error: archive scan-wmo-containers requires --archive-root <game dir>.");
            Environment.ExitCode = 1;
            return;
        }

        gamePath = Path.GetFullPath(gamePath);
        if (!Directory.Exists(gamePath))
        {
            Console.Error.WriteLine($"Error: archive root '{gamePath}' does not exist.");
            Environment.ExitCode = 1;
            return;
        }

        using MpqArchiveCatalog catalog = new();
        IReadOnlyList<string> productionVirtualPaths = catalog.ScanWmoMpqArchives(gamePath);
        HashSet<string> productionSet = new(productionVirtualPaths, StringComparer.OrdinalIgnoreCase);

        List<string> rawFiles = Directory.GetFiles(gamePath, "*.wmo.mpq", SearchOption.AllDirectories)
            .Concat(Directory.GetFiles(gamePath, "*.WMO.MPQ", SearchOption.AllDirectories))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();

        Dictionary<string, List<string>> predictedVirtualPathToRawFiles = new(StringComparer.OrdinalIgnoreCase);
        foreach (string rawFile in rawFiles)
        {
            string predicted = PredictVirtualPath(rawFile);
            if (!predictedVirtualPathToRawFiles.TryGetValue(predicted, out List<string>? sources))
            {
                sources = [];
                predictedVirtualPathToRawFiles[predicted] = sources;
            }

            sources.Add(rawFile);
        }

        List<string> uncoveredByProductionScan = predictedVirtualPathToRawFiles.Keys
            .Where(virtualPath => !productionSet.Contains(virtualPath))
            .Order(StringComparer.OrdinalIgnoreCase)
            .ToList();

        List<string> claimedByProductionButNotSeenRaw = productionSet
            .Where(virtualPath => !predictedVirtualPathToRawFiles.ContainsKey(virtualPath))
            .Order(StringComparer.OrdinalIgnoreCase)
            .ToList();

        Console.WriteLine($"GAME ROOT:                          {gamePath}");
        Console.WriteLine();
        Console.WriteLine("Scoped scan (production ScanWmoMpqArchives, 7 candidate directories):");
        Console.WriteLine($"  Virtual paths found:               {productionSet.Count}");
        Console.WriteLine();
        Console.WriteLine("Unscoped scan (*.wmo.mpq / *.WMO.MPQ anywhere under the game root):");
        Console.WriteLine($"  Raw container files found:         {rawFiles.Count}");
        Console.WriteLine($"  Distinct predicted virtual paths:  {predictedVirtualPathToRawFiles.Count}");
        Console.WriteLine();
        Console.WriteLine($"Containers the scoped scan MISSED:   {uncoveredByProductionScan.Count}");
        foreach (string virtualPath in uncoveredByProductionScan.Take(100))
        {
            foreach (string source in predictedVirtualPathToRawFiles[virtualPath])
                Console.WriteLine($"  {virtualPath}  <-  {source}");
        }

        if (uncoveredByProductionScan.Count > 100)
            Console.WriteLine($"  ... {uncoveredByProductionScan.Count - 100} more");

        if (claimedByProductionButNotSeenRaw.Count > 0)
        {
            // Should never happen — every candidate directory the scoped scan searches is a
            // descendant of the game root, so the unscoped walk is a strict superset. If this prints,
            // the two scans disagree on something more fundamental than directory coverage.
            Console.WriteLine();
            Console.WriteLine($"UNEXPECTED — scoped scan found paths the unscoped walk did not see: {claimedByProductionButNotSeenRaw.Count}");
            foreach (string virtualPath in claimedByProductionButNotSeenRaw.Take(50))
                Console.WriteLine($"  {virtualPath}");
        }
    }

    /// <summary>
    /// Reproduces <see cref="MpqArchiveCatalog"/>'s own container-name-to-virtual-path transform so a
    /// raw file found by the unscoped walk can be checked for membership in the scoped scan's result
    /// set. This is filename munging, not format decoding — it does not read the container's contents.
    /// </summary>
    private static string PredictVirtualPath(string mpqPath)
    {
        string nameWithoutMpq = Path.GetFileNameWithoutExtension(mpqPath);
        string nameWithoutWmo = nameWithoutMpq.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase)
            ? nameWithoutMpq[..^4]
            : nameWithoutMpq;

        return $"World\\wmo\\{nameWithoutWmo}.wmo";
    }

    private static string? GetOption(string[] args, string longName, string shortName)
    {
        for (int index = 0; index < args.Length - 1; index++)
        {
            if (string.Equals(args[index], longName, StringComparison.OrdinalIgnoreCase)
                || string.Equals(args[index], shortName, StringComparison.OrdinalIgnoreCase))
            {
                return args[index + 1];
            }
        }

        return null;
    }
}
