using System.Numerics;
using System.Text.Json;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

internal sealed record MapUniqueIdReport(
    string BuildLabel,
    IReadOnlyList<string> InputPaths,
    int ScannedFileCount,
    int PlacementFileCount,
    int ModelPlacementCount,
    int WorldModelPlacementCount,
    int DistinctUniqueIdCount,
    int DuplicateUniqueIdCount,
    int MaxReuseCount,
    IReadOnlyList<MapUniqueIdBuildSummary> Builds,
    IReadOnlyList<MapUniqueIdSourceSummary> Sources,
    IReadOnlyList<MapUniqueIdDuplicateSummary> DuplicateUniqueIds,
    IReadOnlyList<MapUniqueIdRangeClusterSummary> RangeClusters,
    IReadOnlyList<MapUniqueIdPlacementRecord> Placements,
    IReadOnlyList<MapUniqueIdReadFailure> Failures,
    IReadOnlyList<string> Notes);

internal sealed record MapUniqueIdBuildSummary(
    string BuildLabel,
    string InputPath,
    int ScannedFileCount,
    int PlacementFileCount,
    int ModelPlacementCount,
    int WorldModelPlacementCount,
    int DistinctUniqueIdCount,
    int DuplicateUniqueIdCount,
    int MaxReuseCount);

internal sealed record MapUniqueIdSourceSummary(
    string SourcePath,
    MapFileKind Kind,
    int ModelPlacementCount,
    int WorldModelPlacementCount,
    int DistinctUniqueIdCount,
    int DuplicateUniqueIdCount);

internal sealed record MapUniqueIdDuplicateSummary(
    int UniqueId,
    int Count,
    IReadOnlyList<string> PlacementKinds,
    IReadOnlyList<string> ModelPaths,
    IReadOnlyList<string> SourcePaths);

internal sealed record MapUniqueIdPlacementRecord(
    string BuildLabel,
    string SourcePath,
    MapFileKind SourceKind,
    string PlacementKind,
    int NameId,
    string ModelPath,
    int UniqueId,
    Vector3 Position,
    Vector3 Rotation,
    float? Scale,
    Vector3? BoundsMin,
    Vector3? BoundsMax,
    ushort? Flags);

internal sealed record MapUniqueIdReadFailure(
    string BuildLabel,
    string SourcePath,
    string Error);

internal sealed record MapUniqueIdRangeClusterSummary(
    int ClusterIndex,
    int StartUniqueId,
    int EndUniqueId,
    int DistinctUniqueIdCount,
    int PlacementCount,
    IReadOnlyList<string> BuildLabels,
    IReadOnlyList<string> PlacementKinds,
    IReadOnlyList<string> SampleModelPaths);

internal static class MapUniqueIdReportSupport
{
    public static MapUniqueIdReport Build(string inputPath, string? buildLabel)
    {
        return Build([inputPath], buildLabel);
    }

    public static MapUniqueIdReport Build(IReadOnlyList<string> inputPaths, string? buildLabel)
    {
        ArgumentNullException.ThrowIfNull(inputPaths);
        if (inputPaths.Count == 0)
            throw new ArgumentException("At least one input path is required.", nameof(inputPaths));

        List<string> resolvedInputPaths = inputPaths
            .Where(static path => !string.IsNullOrWhiteSpace(path))
            .Select(Path.GetFullPath)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();
        if (resolvedInputPaths.Count == 0)
            throw new ArgumentException("At least one non-empty input path is required.", nameof(inputPaths));

        string resolvedBuildLabel = ResolveReportBuildLabel(resolvedInputPaths, buildLabel);
        List<MapUniqueIdBuildSummary> builds = new(resolvedInputPaths.Count);
        List<MapUniqueIdSourceSummary> sources = [];
        List<MapUniqueIdPlacementRecord> placements = [];
        List<MapUniqueIdReadFailure> failures = [];

        foreach (string resolvedInputPath in resolvedInputPaths)
        {
            string inputBuildLabel = string.IsNullOrWhiteSpace(buildLabel)
                ? DeriveBuildLabel(resolvedInputPath)
                : buildLabel.Trim();
            BuildInputResult inputResult = BuildForInput(resolvedInputPath, inputBuildLabel);
            builds.Add(inputResult.Build);
            sources.AddRange(inputResult.Sources);
            placements.AddRange(inputResult.Placements);
            failures.AddRange(inputResult.Failures);
        }

        List<MapUniqueIdDuplicateSummary> duplicateUniqueIds = placements
            .GroupBy(static placement => placement.UniqueId)
            .Where(static group => group.Count() > 1)
            .OrderByDescending(static group => group.Count())
            .ThenBy(static group => group.Key)
            .Select(static group => new MapUniqueIdDuplicateSummary(
                group.Key,
                group.Count(),
                group.Select(static placement => placement.PlacementKind).Distinct(StringComparer.OrdinalIgnoreCase).OrderBy(static value => value, StringComparer.OrdinalIgnoreCase).ToArray(),
                group.Select(static placement => placement.ModelPath).Distinct(StringComparer.OrdinalIgnoreCase).OrderBy(static value => value, StringComparer.OrdinalIgnoreCase).ToArray(),
                group.Select(static placement => placement.SourcePath).Distinct(StringComparer.OrdinalIgnoreCase).OrderBy(static value => value, StringComparer.OrdinalIgnoreCase).ToArray()))
            .ToList();

        List<MapUniqueIdRangeClusterSummary> rangeClusters = BuildRangeClusters(placements);

        List<string> notes =
        [
            "This report captures raw MDDF and MODF UniqueId values from ADT or ADTOBJ placement data.",
            "Repeated --input values are aggregated into one report so multiple builds can be compared from a single artifact.",
            "RangeClusters groups nearby UniqueId values to seed later development-era clustering and timeline diff work; it is heuristic, not a final historical truth.",
            "UniqueId reuse inside a build is reported explicitly and should not be assumed impossible or invalid by default."
        ];

        if (failures.Count > 0)
            notes.Add($"{failures.Count} placement files failed to read; inspect the failures list before treating this report as complete.");

        int modelPlacementCount = placements.Count(static placement => string.Equals(placement.PlacementKind, "m2", StringComparison.OrdinalIgnoreCase));
        int worldModelPlacementCount = placements.Count - modelPlacementCount;

        return new MapUniqueIdReport(
            resolvedBuildLabel,
            resolvedInputPaths,
            builds.Sum(static build => build.ScannedFileCount),
            sources.Count(static source => source.ModelPlacementCount > 0 || source.WorldModelPlacementCount > 0),
            modelPlacementCount,
            worldModelPlacementCount,
            placements.Select(static placement => placement.UniqueId).Distinct().Count(),
            duplicateUniqueIds.Count,
            duplicateUniqueIds.Count > 0 ? duplicateUniqueIds[0].Count : 1,
            builds,
            sources,
            duplicateUniqueIds,
            rangeClusters,
            placements,
            failures,
            notes);
    }

    public static string Write(MapUniqueIdReport report, string? outputPath)
    {
        ArgumentNullException.ThrowIfNull(report);

        string resolvedOutputPath = string.IsNullOrWhiteSpace(outputPath)
            ? GetDefaultOutputPath(report.BuildLabel)
            : Path.GetFullPath(outputPath);
        string? outputDirectory = Path.GetDirectoryName(resolvedOutputPath);
        if (!string.IsNullOrWhiteSpace(outputDirectory))
            Directory.CreateDirectory(outputDirectory);

        File.WriteAllText(resolvedOutputPath, JsonSerializer.Serialize(report, CreateJsonOptions()));
        return resolvedOutputPath;
    }

    public static void PrintSummary(MapUniqueIdReport report, string outputPath)
    {
        Console.WriteLine("WowViewer.Tool.Inspect map uniqueid-report");
        Console.WriteLine($"Build: {report.BuildLabel}");
        Console.WriteLine($"Inputs: {report.InputPaths.Count}");
        Console.WriteLine($"Scanned files: {report.ScannedFileCount} placement-bearing files: {report.PlacementFileCount}");
        Console.WriteLine($"Placements: m2={report.ModelPlacementCount} wmo={report.WorldModelPlacementCount} total={report.Placements.Count}");
        Console.WriteLine($"UniqueIds: distinct={report.DistinctUniqueIdCount} duplicates={report.DuplicateUniqueIdCount} maxReuse={report.MaxReuseCount}");
        Console.WriteLine($"Builds: {report.Builds.Count} clusters={report.RangeClusters.Count}");
        Console.WriteLine($"Failures: {report.Failures.Count}");
        Console.WriteLine($"Wrote {outputPath}");
    }

    private static BuildInputResult BuildForInput(string inputPath, string buildLabel)
    {
        List<string> placementFiles = ResolvePlacementFiles(inputPath);
        List<MapUniqueIdSourceSummary> sources = new(placementFiles.Count);
        List<MapUniqueIdPlacementRecord> placements = [];
        List<MapUniqueIdReadFailure> failures = [];

        foreach (string placementFile in placementFiles)
        {
            try
            {
                AdtPlacementCatalog catalog = AdtPlacementReader.Read(placementFile);
                List<MapUniqueIdPlacementRecord> sourcePlacements = BuildPlacementRecords(buildLabel, catalog);
                placements.AddRange(sourcePlacements);

                int distinctUniqueIds = sourcePlacements
                    .Select(static placement => placement.UniqueId)
                    .Distinct()
                    .Count();
                int duplicateUniqueIdCount = sourcePlacements
                    .GroupBy(static placement => placement.UniqueId)
                    .Count(static group => group.Count() > 1);

                sources.Add(new MapUniqueIdSourceSummary(
                    catalog.SourcePath,
                    catalog.Kind,
                    catalog.ModelPlacements.Count,
                    catalog.WorldModelPlacements.Count,
                    distinctUniqueIds,
                    duplicateUniqueIdCount));
            }
            catch (Exception ex) when (ex is IOException or InvalidDataException or UnauthorizedAccessException)
            {
                failures.Add(new MapUniqueIdReadFailure(buildLabel, placementFile, ex.Message));
            }
        }

        int modelPlacementCount = placements.Count(static placement => string.Equals(placement.PlacementKind, "m2", StringComparison.OrdinalIgnoreCase));
        int worldModelPlacementCount = placements.Count - modelPlacementCount;
        int duplicateUniqueIdCountForBuild = placements
            .GroupBy(static placement => placement.UniqueId)
            .Count(static group => group.Count() > 1);
        int maxReuseCount = placements
            .GroupBy(static placement => placement.UniqueId)
            .Select(static group => group.Count())
            .DefaultIfEmpty(1)
            .Max();

        return new BuildInputResult(
            new MapUniqueIdBuildSummary(
                buildLabel,
                inputPath,
                placementFiles.Count,
                sources.Count(static source => source.ModelPlacementCount > 0 || source.WorldModelPlacementCount > 0),
                modelPlacementCount,
                worldModelPlacementCount,
                placements.Select(static placement => placement.UniqueId).Distinct().Count(),
                duplicateUniqueIdCountForBuild,
                maxReuseCount),
            sources,
            placements,
            failures);
    }

    private static List<MapUniqueIdPlacementRecord> BuildPlacementRecords(string buildLabel, AdtPlacementCatalog catalog)
    {
        List<MapUniqueIdPlacementRecord> records = new(catalog.ModelPlacements.Count + catalog.WorldModelPlacements.Count);

        foreach (AdtModelPlacement placement in catalog.ModelPlacements)
        {
            records.Add(new MapUniqueIdPlacementRecord(
                buildLabel,
                catalog.SourcePath,
                catalog.Kind,
                "m2",
                placement.NameId,
                placement.ModelPath,
                placement.UniqueId,
                placement.Position,
                placement.Rotation,
                placement.Scale,
                null,
                null,
                null));
        }

        foreach (AdtWorldModelPlacement placement in catalog.WorldModelPlacements)
        {
            records.Add(new MapUniqueIdPlacementRecord(
                buildLabel,
                catalog.SourcePath,
                catalog.Kind,
                "wmo",
                placement.NameId,
                placement.ModelPath,
                placement.UniqueId,
                placement.Position,
                placement.Rotation,
                null,
                placement.BoundsMin,
                placement.BoundsMax,
                placement.Flags));
        }

        return records;
    }

    private static List<string> ResolvePlacementFiles(string inputPath)
    {
        if (File.Exists(inputPath))
        {
            string extension = Path.GetExtension(inputPath);
            if (extension.Equals(".wdt", StringComparison.OrdinalIgnoreCase))
                return ResolvePlacementFilesFromWdt(inputPath);

            if (extension.Equals(".adt", StringComparison.OrdinalIgnoreCase) && IsPlacementCandidateAdtPath(inputPath))
                return [inputPath];

            throw new InvalidDataException($"Input '{inputPath}' must be a .wdt, an .adt, or a directory containing ADT tiles.");
        }

        if (Directory.Exists(inputPath))
        {
            return Directory.EnumerateFiles(inputPath, "*.adt", SearchOption.AllDirectories)
                .Where(IsPlacementCandidateAdtPath)
                .OrderBy(static path => path, StringComparer.OrdinalIgnoreCase)
                .ToList();
        }

        throw new FileNotFoundException($"Input '{inputPath}' does not exist.", inputPath);
    }

    private static List<string> ResolvePlacementFilesFromWdt(string wdtPath)
    {
        string directory = Path.GetDirectoryName(wdtPath)
            ?? throw new InvalidDataException($"Could not resolve the map directory for '{wdtPath}'.");
        string mapName = Path.GetFileNameWithoutExtension(wdtPath);

        return Directory.EnumerateFiles(directory, $"{mapName}_*.adt", SearchOption.TopDirectoryOnly)
            .Where(IsPlacementCandidateAdtPath)
            .OrderBy(static path => path, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    private static bool IsPlacementCandidateAdtPath(string path)
    {
        string fileName = Path.GetFileName(path);
        return !fileName.EndsWith("_tex0.adt", StringComparison.OrdinalIgnoreCase)
            && !fileName.EndsWith("_lod.adt", StringComparison.OrdinalIgnoreCase);
    }

    private static string DeriveBuildLabel(string inputPath)
    {
        if (Directory.Exists(inputPath))
            return new DirectoryInfo(inputPath).Name;

        string extension = Path.GetExtension(inputPath);
        if (extension.Equals(".wdt", StringComparison.OrdinalIgnoreCase))
            return Path.GetFileNameWithoutExtension(inputPath);

        return Path.GetFileNameWithoutExtension(inputPath);
    }

    private static string GetDefaultOutputPath(string buildLabel)
    {
        string root = FindWowViewerRoot();
        string safeBuildLabel = string.Concat(buildLabel.Select(static ch => Path.GetInvalidFileNameChars().Contains(ch) ? '_' : ch));
        return Path.Combine(root, "output", "reports", "map-uniqueids", $"{safeBuildLabel}.json");
    }

    private static string FindWowViewerRoot()
    {
        DirectoryInfo? current = new(AppContext.BaseDirectory);
        while (current is not null)
        {
            if (File.Exists(Path.Combine(current.FullName, "WowViewer.slnx")))
                return current.FullName;

            current = current.Parent;
        }

        return Path.GetFullPath(Directory.GetCurrentDirectory());
    }

    private static string ResolveReportBuildLabel(IReadOnlyList<string> inputPaths, string? buildLabel)
    {
        if (!string.IsNullOrWhiteSpace(buildLabel))
            return buildLabel.Trim();

        if (inputPaths.Count == 1)
            return DeriveBuildLabel(inputPaths[0]);

        return "multi-build";
    }

    private static List<MapUniqueIdRangeClusterSummary> BuildRangeClusters(IReadOnlyList<MapUniqueIdPlacementRecord> placements)
    {
        if (placements.Count == 0)
            return [];

        const int maxGap = 512;
        List<MapUniqueIdPlacementRecord> ordered = placements
            .OrderBy(static placement => placement.UniqueId)
            .ThenBy(static placement => placement.BuildLabel, StringComparer.OrdinalIgnoreCase)
            .ToList();

        List<MapUniqueIdRangeClusterSummary> clusters = [];
        List<MapUniqueIdPlacementRecord> clusterPlacements = [ordered[0]];

        for (int index = 1; index < ordered.Count; index++)
        {
            MapUniqueIdPlacementRecord placement = ordered[index];
            MapUniqueIdPlacementRecord previous = ordered[index - 1];
            if (placement.UniqueId - previous.UniqueId > maxGap)
            {
                clusters.Add(BuildCluster(clusters.Count + 1, clusterPlacements));
                clusterPlacements = [];
            }

            clusterPlacements.Add(placement);
        }

        clusters.Add(BuildCluster(clusters.Count + 1, clusterPlacements));
        return clusters;
    }

    private static MapUniqueIdRangeClusterSummary BuildCluster(int clusterIndex, IReadOnlyList<MapUniqueIdPlacementRecord> placements)
    {
        return new MapUniqueIdRangeClusterSummary(
            clusterIndex,
            placements.Min(static placement => placement.UniqueId),
            placements.Max(static placement => placement.UniqueId),
            placements.Select(static placement => placement.UniqueId).Distinct().Count(),
            placements.Count,
            placements.Select(static placement => placement.BuildLabel).Distinct(StringComparer.OrdinalIgnoreCase).OrderBy(static value => value, StringComparer.OrdinalIgnoreCase).ToArray(),
            placements.Select(static placement => placement.PlacementKind).Distinct(StringComparer.OrdinalIgnoreCase).OrderBy(static value => value, StringComparer.OrdinalIgnoreCase).ToArray(),
            placements.Select(static placement => placement.ModelPath).Distinct(StringComparer.OrdinalIgnoreCase).OrderBy(static value => value, StringComparer.OrdinalIgnoreCase).Take(8).ToArray());
    }

    private static JsonSerializerOptions CreateJsonOptions()
    {
        return new JsonSerializerOptions
        {
            WriteIndented = true,
            IncludeFields = true,
        };
    }

    private sealed record BuildInputResult(
        MapUniqueIdBuildSummary Build,
        IReadOnlyList<MapUniqueIdSourceSummary> Sources,
        IReadOnlyList<MapUniqueIdPlacementRecord> Placements,
        IReadOnlyList<MapUniqueIdReadFailure> Failures);
}