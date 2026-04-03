using System.Numerics;
using System.Text.Json;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

internal sealed record MapUniqueIdReport(
    string BuildLabel,
    string InputPath,
    int ScannedFileCount,
    int PlacementFileCount,
    int ModelPlacementCount,
    int WorldModelPlacementCount,
    int DistinctUniqueIdCount,
    int DuplicateUniqueIdCount,
    int MaxReuseCount,
    IReadOnlyList<MapUniqueIdSourceSummary> Sources,
    IReadOnlyList<MapUniqueIdDuplicateSummary> DuplicateUniqueIds,
    IReadOnlyList<MapUniqueIdPlacementRecord> Placements,
    IReadOnlyList<MapUniqueIdReadFailure> Failures,
    IReadOnlyList<string> Notes);

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
    string SourcePath,
    string Error);

internal static class MapUniqueIdReportSupport
{
    public static MapUniqueIdReport Build(string inputPath, string? buildLabel)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(inputPath);

        string resolvedInputPath = Path.GetFullPath(inputPath);
        string resolvedBuildLabel = string.IsNullOrWhiteSpace(buildLabel)
            ? DeriveBuildLabel(resolvedInputPath)
            : buildLabel.Trim();

        List<string> placementFiles = ResolvePlacementFiles(resolvedInputPath);
        List<MapUniqueIdSourceSummary> sources = new(placementFiles.Count);
        List<MapUniqueIdPlacementRecord> placements = [];
        List<MapUniqueIdReadFailure> failures = [];

        foreach (string placementFile in placementFiles)
        {
            try
            {
                AdtPlacementCatalog catalog = AdtPlacementReader.Read(placementFile);
                List<MapUniqueIdPlacementRecord> sourcePlacements = BuildPlacementRecords(resolvedBuildLabel, catalog);
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
                failures.Add(new MapUniqueIdReadFailure(placementFile, ex.Message));
            }
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

        List<string> notes =
        [
            "This report captures raw MDDF and MODF UniqueId values from ADT or ADTOBJ placement data.",
            "Use one report per build as the evidence source for later added/removed-object timeline diffs.",
            "UniqueId reuse inside a build is reported explicitly and should not be assumed impossible or invalid by default."
        ];

        if (failures.Count > 0)
            notes.Add($"{failures.Count} placement files failed to read; inspect the failures list before treating this report as complete.");

        int modelPlacementCount = placements.Count(static placement => string.Equals(placement.PlacementKind, "m2", StringComparison.OrdinalIgnoreCase));
        int worldModelPlacementCount = placements.Count - modelPlacementCount;

        return new MapUniqueIdReport(
            resolvedBuildLabel,
            resolvedInputPath,
            placementFiles.Count,
            sources.Count(static source => source.ModelPlacementCount > 0 || source.WorldModelPlacementCount > 0),
            modelPlacementCount,
            worldModelPlacementCount,
            placements.Select(static placement => placement.UniqueId).Distinct().Count(),
            duplicateUniqueIds.Count,
            duplicateUniqueIds.Count > 0 ? duplicateUniqueIds[0].Count : 1,
            sources,
            duplicateUniqueIds,
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
        Console.WriteLine($"Input: {report.InputPath}");
        Console.WriteLine($"Scanned files: {report.ScannedFileCount} placement-bearing files: {report.PlacementFileCount}");
        Console.WriteLine($"Placements: m2={report.ModelPlacementCount} wmo={report.WorldModelPlacementCount} total={report.Placements.Count}");
        Console.WriteLine($"UniqueIds: distinct={report.DistinctUniqueIdCount} duplicates={report.DuplicateUniqueIdCount} maxReuse={report.MaxReuseCount}");
        Console.WriteLine($"Failures: {report.Failures.Count}");
        Console.WriteLine($"Wrote {outputPath}");
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

    private static JsonSerializerOptions CreateJsonOptions()
    {
        return new JsonSerializerOptions
        {
            WriteIndented = true,
            IncludeFields = true,
        };
    }
}