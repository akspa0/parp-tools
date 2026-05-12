using System.Text.Json;

internal sealed record MapUniqueIdFilterOptions(
    int? MinUniqueId,
    int? MaxUniqueId,
    IReadOnlyList<string> BuildLabels,
    string PlacementKind,
    bool Invert);

internal sealed record MapUniqueIdFilterReport(
    string SourceReportPath,
    string SourceBuildLabel,
    int? MinUniqueId,
    int? MaxUniqueId,
    IReadOnlyList<string> BuildLabels,
    string PlacementKind,
    bool Invert,
    int SourcePlacementCount,
    int SelectedPlacementCount,
    int SelectedDistinctUniqueIdCount,
    IReadOnlyList<MapUniqueIdFilterBuildSummary> Builds,
    IReadOnlyList<MapUniqueIdPlacementRecord> Placements,
    IReadOnlyList<string> Notes);

internal sealed record MapUniqueIdFilterBuildSummary(
    string BuildLabel,
    int PlacementCount,
    int DistinctUniqueIdCount,
    int? MinUniqueId,
    int? MaxUniqueId);

internal static class MapUniqueIdFilterSupport
{
    public static MapUniqueIdFilterReport Filter(string reportPath, MapUniqueIdFilterOptions options)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(reportPath);
        ArgumentNullException.ThrowIfNull(options);

        string resolvedReportPath = Path.GetFullPath(reportPath);
        if (!File.Exists(resolvedReportPath))
            throw new FileNotFoundException($"UniqueId report '{resolvedReportPath}' does not exist.", resolvedReportPath);

        MapUniqueIdReport sourceReport = LoadReport(resolvedReportPath);
        HashSet<string> buildLabelSet = options.BuildLabels.Count == 0
            ? []
            : options.BuildLabels.ToHashSet(StringComparer.OrdinalIgnoreCase);
        string normalizedKind = NormalizePlacementKind(options.PlacementKind);

        List<MapUniqueIdPlacementRecord> selectedPlacements = sourceReport.Placements
            .Where(placement => Matches(placement, options, buildLabelSet, normalizedKind))
            .ToList();

        List<MapUniqueIdFilterBuildSummary> buildSummaries = selectedPlacements
            .GroupBy(static placement => placement.BuildLabel, StringComparer.OrdinalIgnoreCase)
            .OrderBy(static group => group.Key, StringComparer.OrdinalIgnoreCase)
            .Select(static group => new MapUniqueIdFilterBuildSummary(
                group.Key,
                group.Count(),
                group.Select(static placement => placement.UniqueId).Distinct().Count(),
                group.Min(static placement => placement.UniqueId),
                group.Max(static placement => placement.UniqueId)))
            .ToList();

        List<string> notes =
        [
            "This filter operates on the JSON artifact emitted by map uniqueid-report.",
            "Selected placements are intended to seed later era-aware patching or viewer toggles, not to claim a final historical reconstruction by themselves."
        ];

        if (options.Invert)
            notes.Add("Invert=true means placements outside the provided criteria were selected.");

        return new MapUniqueIdFilterReport(
            resolvedReportPath,
            sourceReport.BuildLabel,
            options.MinUniqueId,
            options.MaxUniqueId,
            options.BuildLabels,
            normalizedKind,
            options.Invert,
            sourceReport.Placements.Count,
            selectedPlacements.Count,
            selectedPlacements.Select(static placement => placement.UniqueId).Distinct().Count(),
            buildSummaries,
            selectedPlacements,
            notes);
    }

    public static string Write(MapUniqueIdFilterReport report, string? outputPath)
    {
        ArgumentNullException.ThrowIfNull(report);

        string resolvedOutputPath = string.IsNullOrWhiteSpace(outputPath)
            ? GetDefaultOutputPath(report.SourceBuildLabel)
            : Path.GetFullPath(outputPath);
        string? outputDirectory = Path.GetDirectoryName(resolvedOutputPath);
        if (!string.IsNullOrWhiteSpace(outputDirectory))
            Directory.CreateDirectory(outputDirectory);

        File.WriteAllText(resolvedOutputPath, JsonSerializer.Serialize(report, CreateJsonOptions()));
        return resolvedOutputPath;
    }

    public static void PrintSummary(MapUniqueIdFilterReport report, string outputPath)
    {
        Console.WriteLine("WowViewer.Tool.Inspect map uniqueid-filter");
        Console.WriteLine($"Source: {report.SourceReportPath}");
        Console.WriteLine($"Placements: selected={report.SelectedPlacementCount} source={report.SourcePlacementCount}");
        Console.WriteLine($"UniqueIds: distinct={report.SelectedDistinctUniqueIdCount}");
        Console.WriteLine($"Builds: {report.Builds.Count}");
        Console.WriteLine($"Wrote {outputPath}");
    }

    private static MapUniqueIdReport LoadReport(string path)
    {
        string json = File.ReadAllText(path);
        return JsonSerializer.Deserialize<MapUniqueIdReport>(json, CreateJsonOptions())
            ?? throw new InvalidDataException($"Could not deserialize UniqueId report '{path}'.");
    }

    private static bool Matches(MapUniqueIdPlacementRecord placement, MapUniqueIdFilterOptions options, HashSet<string> buildLabelSet, string normalizedKind)
    {
        bool matches = true;

        if (options.MinUniqueId.HasValue)
            matches &= placement.UniqueId >= options.MinUniqueId.Value;
        if (options.MaxUniqueId.HasValue)
            matches &= placement.UniqueId <= options.MaxUniqueId.Value;
        if (buildLabelSet.Count > 0)
            matches &= buildLabelSet.Contains(placement.BuildLabel);
        if (!string.Equals(normalizedKind, "all", StringComparison.OrdinalIgnoreCase))
            matches &= string.Equals(placement.PlacementKind, normalizedKind, StringComparison.OrdinalIgnoreCase);

        return options.Invert ? !matches : matches;
    }

    private static string NormalizePlacementKind(string? placementKind)
    {
        if (string.IsNullOrWhiteSpace(placementKind))
            return "all";

        string normalized = placementKind.Trim().ToLowerInvariant();
        return normalized switch
        {
            "all" or "m2" or "wmo" => normalized,
            _ => throw new InvalidDataException($"Unsupported placement kind '{placementKind}'. Expected all, m2, or wmo.")
        };
    }

    private static string GetDefaultOutputPath(string buildLabel)
    {
        string root = FindWowViewerRoot();
        string safeBuildLabel = string.Concat(buildLabel.Select(static ch => Path.GetInvalidFileNameChars().Contains(ch) ? '_' : ch));
        return Path.Combine(root, "output", "reports", "map-uniqueids", $"{safeBuildLabel}.filtered.json");
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
            PropertyNameCaseInsensitive = true,
            WriteIndented = true,
            IncludeFields = true,
        };
    }
}