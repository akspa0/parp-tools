using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using WoWRollback.Core.Models;

namespace WoWRollback.Core.Services;

public static class VersionComparisonService
{
    private static readonly char[] InvalidFileNameChars = Path.GetInvalidFileNameChars();

    public static VersionComparisonResult CompareVersions(
        string? root,
        IEnumerable<string> versionIdentifiers,
        IEnumerable<string>? mapFilter = null)
    {
        if (versionIdentifiers is null)
            throw new ArgumentNullException(nameof(versionIdentifiers));

        var requestedVersions = versionIdentifiers
            .Select(v => v?.Trim())
            .Where(v => !string.IsNullOrWhiteSpace(v))
            .Select(v => v!)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();

        if (requestedVersions.Count == 0)
            throw new ArgumentException("At least one version must be provided.", nameof(versionIdentifiers));

        var rootDirectory = string.IsNullOrWhiteSpace(root)
            ? Path.Combine(Directory.GetCurrentDirectory(), "rollback_outputs")
            : root!;

        if (!Directory.Exists(rootDirectory))
            throw new DirectoryNotFoundException($"Comparison root directory not found: {rootDirectory}");

        HashSet<string>? mapFilterSet = null;
        if (mapFilter is not null)
            mapFilterSet = new HashSet<string>(mapFilter.Select(m => m.Trim()), StringComparer.OrdinalIgnoreCase);

        var warnings = new List<string>();
        var versionDirectoryMap = ResolveVersionDirectories(rootDirectory, requestedVersions, warnings);
        if (versionDirectoryMap.Count == 0)
            return EmptyResult(rootDirectory, warnings);

        var orderedVersions = versionDirectoryMap.Keys
            .OrderBy(v => v, StringComparer.OrdinalIgnoreCase)
            .ToList();

        var allRanges = new List<VersionRangeEntry>();
        var rangesByVersion = new Dictionary<string, List<VersionRangeEntry>>(StringComparer.OrdinalIgnoreCase);
        var assetsByVersion = new Dictionary<string, List<PlacementAsset>>(StringComparer.OrdinalIgnoreCase);
        var mapSummaries = new List<MapVersionSummary>();

        foreach (var version in orderedVersions)
        {
            var versionDirectory = versionDirectoryMap[version];
            if (!Directory.Exists(versionDirectory))
            {
                warnings.Add($"Directory missing for version {version}: {versionDirectory}");
                continue;
            }

            var mapDirectories = Directory.GetDirectories(versionDirectory);
            if (mapDirectories.Length == 0)
            {
                warnings.Add($"No map directories found for version {version} in {versionDirectory}.");
                continue;
            }

            foreach (var mapDirectory in mapDirectories)
            {
                var mapName = Path.GetFileName(mapDirectory);
                if (mapFilterSet is not null && !mapFilterSet.Contains(mapName))
                    continue;

                var mapEntries = LoadMapEntries(version, mapName, mapDirectory, warnings);
                if (mapEntries.Ranges.Count == 0 && mapEntries.Assets.Count == 0)
                {
                    warnings.Add($"No placement ranges found for {version}/{mapName}.");
                    continue;
                }

                if (!rangesByVersion.TryGetValue(version, out var versionRanges))
                {
                    versionRanges = new List<VersionRangeEntry>();
                    rangesByVersion[version] = versionRanges;
                }

                if (!assetsByVersion.TryGetValue(version, out var versionAssets))
                {
                    versionAssets = new List<PlacementAsset>();
                    assetsByVersion[version] = versionAssets;
                }

                versionRanges.AddRange(mapEntries.Ranges);
                versionAssets.AddRange(mapEntries.Assets);
                allRanges.AddRange(mapEntries.Ranges);

                AppendMapSummary(version, mapName, mapEntries, mapSummaries);
            }
        }

        var actualVersions = orderedVersions
            .Where(v => rangesByVersion.ContainsKey(v) || assetsByVersion.ContainsKey(v))
            .ToList();

        if (actualVersions.Count == 0)
        {
            warnings.Add("No comparable versions contained placement data.");
            return EmptyResult(rootDirectory, warnings, mapSummaries);
        }

        var versionOrder = actualVersions
            .Select((name, index) => (name, index))
            .ToDictionary(t => t.name, t => t.index, StringComparer.OrdinalIgnoreCase);

        var overlaps = ComputeOverlaps(actualVersions, rangesByVersion);
        var assetFirstSeen = ComputeAssetFirstSeen(actualVersions, rangesByVersion, versionOrder);
        var assetFolderSummaries = ComputeAssetFolderSummaries(rangesByVersion, assetsByVersion);
        var assetFolderTimeline = ComputeAssetFolderTimeline(assetsByVersion, versionOrder);
        var assetTimeline = BuildAssetTimeline(assetsByVersion, versionOrder);
        var designKitAssets = BuildDesignKitAssets(assetsByVersion, versionOrder);
        var designKitRanges = BuildDesignKitRanges(rangesByVersion);
        var designKitSummaries = BuildDesignKitSummaries(designKitAssets);
        var designKitTimeline = BuildDesignKitTimeline(designKitAssets, versionOrder);

        var comparisonKey = BuildComparisonKey(actualVersions);

        var orderedRangeEntries = allRanges
            .Where(entry => versionOrder.ContainsKey(entry.Version))
            .OrderBy(entry => versionOrder[entry.Version])
            .ThenBy(entry => entry.Map, StringComparer.OrdinalIgnoreCase)
            .ThenBy(entry => entry.TileRow)
            .ThenBy(entry => entry.TileCol)
            .ThenBy(entry => entry.MinUniqueId)
            .ToList();

        return new VersionComparisonResult(
            rootDirectory,
            comparisonKey,
            actualVersions,
            orderedRangeEntries,
            mapSummaries,
            overlaps,
            assetFirstSeen,
            assetFolderSummaries,
            assetFolderTimeline,
            assetTimeline,
            designKitAssets,
            designKitRanges,
            designKitSummaries,
            designKitTimeline,
            warnings);
    }

    private static Dictionary<string, string> ResolveVersionDirectories(
        string rootDirectory,
        IReadOnlyList<string> requestedVersions,
        List<string> warnings)
    {
        var map = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var versionIdentifier in requestedVersions)
        {
            var resolvedDirectory = ResolveVersionDirectory(rootDirectory, versionIdentifier, warnings);
            if (resolvedDirectory is null)
                continue;
            var folderName = Path.GetFileName(resolvedDirectory);
            if (!map.ContainsKey(folderName))
                map[folderName] = resolvedDirectory;
        }
        return map;
    }

    private static string? ResolveVersionDirectory(string rootDirectory, string versionIdentifier, List<string> warnings)
    {
        var directPath = Path.Combine(rootDirectory, versionIdentifier);
        if (Directory.Exists(directPath))
            return directPath;

        var candidates = Directory.GetDirectories(rootDirectory)
            .Where(dir =>
            {
                var name = Path.GetFileName(dir);
                if (name.Equals(versionIdentifier, StringComparison.OrdinalIgnoreCase))
                    return true;
                return name.StartsWith(versionIdentifier + ".", StringComparison.OrdinalIgnoreCase);
            })
            .OrderBy(dir => dir, StringComparer.OrdinalIgnoreCase)
            .ToList();

        if (candidates.Count == 0)
            return null;

        if (candidates.Count > 1)
            warnings.Add($"Multiple build directories match '{versionIdentifier}'. Using '{Path.GetFileName(candidates[0])}'.");

        return candidates[0];
    }

    private static MapEntries LoadMapEntries(
        string version,
        string map,
        string mapDirectory,
        List<string> warnings)
    {
        var assetLedgerPath = Directory.GetFiles(mapDirectory, "assets_*.csv").FirstOrDefault();
        var assetData = LoadAssetData(assetLedgerPath);

        var ranges = new List<VersionRangeEntry>();
        var timelineAssetsPath = Directory.GetFiles(mapDirectory, "timeline_assets_*.csv").FirstOrDefault();
        var timelinePath = Directory.GetFiles(mapDirectory, "timeline_*.csv").FirstOrDefault();
        var rangePath = Directory.GetFiles(mapDirectory, "id_ranges_by_map_*.csv").FirstOrDefault();

        if (timelineAssetsPath is not null)
        {
            ranges.AddRange(ParseTimelineAssets(version, map, timelineAssetsPath, warnings));
        }
        else if (timelinePath is not null)
        {
            ranges.AddRange(ParseTimeline(version, map, timelinePath, assetData.Lookup, warnings));
        }
        else if (rangePath is not null)
        {
            ranges.AddRange(ParseIdRanges(version, map, rangePath, assetData.Lookup, warnings));
        }
        else
        {
            warnings.Add($"No timeline or range CSV found for {version}/{map} in {mapDirectory}.");
        }

        if (ranges.Count == 0 && assetLedgerPath is not null && assetData.Assets.Count > 0)
        {
            warnings.Add($"Assets detected for {version}/{map} but no ranges were established.");
        }

        return new MapEntries(ranges, assetData.Assets);
    }

    private static void AppendMapSummary(
        string version,
        string map,
        MapEntries entries,
        List<MapVersionSummary> summaries)
    {
        var distinctAssets = new HashSet<string>(entries.Assets.Select(a => a.AssetPath), StringComparer.OrdinalIgnoreCase);
        uint min = uint.MaxValue;
        uint max = 0;
        foreach (var range in entries.Ranges)
        {
            if (range.MinUniqueId < min) min = range.MinUniqueId;
            if (range.MaxUniqueId > max) max = range.MaxUniqueId;
        }
        foreach (var asset in entries.Assets)
        {
            if (!asset.UniqueId.HasValue) continue;
            if (asset.UniqueId.Value < min) min = asset.UniqueId.Value;
            if (asset.UniqueId.Value > max) max = asset.UniqueId.Value;
        }
        if (min == uint.MaxValue) min = 0;
        summaries.Add(new MapVersionSummary(version, map, entries.Ranges.Count, min, max, distinctAssets.Count));
    }

    private static AssetData LoadAssetData(string? assetLedgerPath)
    {
        var assets = new List<PlacementAsset>();
        var lookup = new Dictionary<(string Map, int Row, int Col, PlacementKind Kind), HashSet<string>>();
        if (assetLedgerPath is null || !File.Exists(assetLedgerPath))
            return new AssetData(assets, lookup);

        foreach (var line in File.ReadLines(assetLedgerPath).Skip(1))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            var parts = line.Split(',', StringSplitOptions.None);
            if (parts.Length < 7) continue;
            try
            {
                var mapName = parts[0].Trim();
                var kind = ParseKind(parts[1]);
                var uniqueId = string.IsNullOrWhiteSpace(parts[2]) ? (uint?)null : ParseUInt(parts[2]);
                var assetPath = NormalizeAssetPath(parts[3]);
                if (!int.TryParse(parts[4], NumberStyles.Integer, CultureInfo.InvariantCulture, out var tileRow)) continue;
                if (!int.TryParse(parts[5], NumberStyles.Integer, CultureInfo.InvariantCulture, out var tileCol)) continue;
                var filePath = parts.Length > 6 ? parts[6].Trim() : string.Empty;
                assets.Add(new PlacementAsset(mapName, tileRow, tileCol, kind, uniqueId, assetPath, filePath));
                var key = (mapName, tileRow, tileCol, kind);
                if (!lookup.TryGetValue(key, out var set))
                {
                    set = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                    lookup[key] = set;
                }
                if (!string.IsNullOrWhiteSpace(assetPath)) set.Add(assetPath);
            }
            catch { }
        }
        return new AssetData(assets, lookup);
    }

    private static List<VersionRangeEntry> ParseTimelineAssets(
        string version,
        string map,
        string timelineAssetsPath,
        List<string> warnings)
    {
        var entries = new List<VersionRangeEntry>();
        foreach (var line in File.ReadLines(timelineAssetsPath).Skip(1))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            var parts = line.Split(',', StringSplitOptions.None);
            if (parts.Length < 9)
            {
                warnings.Add($"Timeline asset row malformed in {timelineAssetsPath}: {line}");
                continue;
            }
            try
            {
                var kind = ParseKind(parts[1]);
                var minId = ParseUInt(parts[2]);
                var maxId = ParseUInt(parts[3]);
                var assetList = ParseAssets(parts[5]);
                var tileRow = int.Parse(parts[6], CultureInfo.InvariantCulture);
                var tileCol = int.Parse(parts[7], CultureInfo.InvariantCulture);
                var filePath = parts[8].Trim();
                entries.Add(new VersionRangeEntry(version, parts[0].Trim(), tileRow, tileCol, kind, minId, maxId, filePath, assetList));
            }
            catch (Exception ex)
            {
                warnings.Add($"Failed to parse timeline asset row in {timelineAssetsPath}: {ex.Message}");
            }
        }
        return entries;
    }

    private static List<VersionRangeEntry> ParseTimeline(
        string version,
        string map,
        string timelinePath,
        Dictionary<(string Map, int Row, int Col, PlacementKind Kind), HashSet<string>> assetLookup,
        List<string> warnings)
    {
        var entries = new List<VersionRangeEntry>();
        foreach (var line in File.ReadLines(timelinePath).Skip(1))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            var parts = line.Split(',', StringSplitOptions.None);
            if (parts.Length < 8)
            {
                warnings.Add($"Timeline row malformed in {timelinePath}: {line}");
                continue;
            }
            try
            {
                var kind = ParseKind(parts[1]);
                var minId = ParseUInt(parts[2]);
                var maxId = ParseUInt(parts[3]);
                var tileRow = int.Parse(parts[5], CultureInfo.InvariantCulture);
                var tileCol = int.Parse(parts[6], CultureInfo.InvariantCulture);
                var filePath = parts[7].Trim();
                var key = (parts[0].Trim(), tileRow, tileCol, kind);
                assetLookup.TryGetValue(key, out var assetsForKey);
                entries.Add(new VersionRangeEntry(version, parts[0].Trim(), tileRow, tileCol, kind, minId, maxId, filePath, assetsForKey?.ToList() ?? new List<string>()));
            }
            catch (Exception ex)
            {
                warnings.Add($"Failed to parse timeline row in {timelinePath}: {ex.Message}");
            }
        }
        return entries;
    }

    private static List<VersionRangeEntry> ParseIdRanges(
        string version,
        string map,
        string rangePath,
        Dictionary<(string Map, int Row, int Col, PlacementKind Kind), HashSet<string>> assetLookup,
        List<string> warnings)
    {
        var entries = new List<VersionRangeEntry>();
        foreach (var line in File.ReadLines(rangePath).Skip(1))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            var parts = line.Split(',', StringSplitOptions.None);
            if (parts.Length < 8)
            {
                warnings.Add($"Range row malformed in {rangePath}: {line}");
                continue;
            }
            try
            {
                var kind = ParseKind(parts[3]);
                var minId = ParseUInt(parts[5]);
                var maxId = ParseUInt(parts[6]);
                var tileRow = int.Parse(parts[1], CultureInfo.InvariantCulture);
                var tileCol = int.Parse(parts[2], CultureInfo.InvariantCulture);
                var filePath = parts[7].Trim();
                var key = (parts[0].Trim(), tileRow, tileCol, kind);
                assetLookup.TryGetValue(key, out var assetsForKey);
                entries.Add(new VersionRangeEntry(version, parts[0].Trim(), tileRow, tileCol, kind, minId, maxId, filePath, assetsForKey?.ToList() ?? new List<string>()));
            }
            catch (Exception ex)
            {
                warnings.Add($"Failed to parse id range row in {rangePath}: {ex.Message}");
            }
        }
        return entries;
    }

    private static List<RangeOverlapEntry> ComputeOverlaps(
        IReadOnlyList<string> versions,
        Dictionary<string, List<VersionRangeEntry>> rangesByVersion)
    {
        var overlaps = new List<RangeOverlapEntry>();
        for (int outer = 0; outer < versions.Count; outer++)
        {
            var versionA = versions[outer];
            var rangesA = rangesByVersion.ContainsKey(versionA) ? rangesByVersion[versionA] : new List<VersionRangeEntry>();
            for (int inner = outer + 1; inner < versions.Count; inner++)
            {
                var versionB = versions[inner];
                var rangesB = rangesByVersion.ContainsKey(versionB) ? rangesByVersion[versionB] : new List<VersionRangeEntry>();
                foreach (var rangeA in rangesA)
                {
                    foreach (var rangeB in rangesB)
                    {
                        var overlapMin = Math.Max(rangeA.MinUniqueId, rangeB.MinUniqueId);
                        var overlapMax = Math.Min(rangeA.MaxUniqueId, rangeB.MaxUniqueId);
                        if (overlapMin > overlapMax) continue;
                        overlaps.Add(new RangeOverlapEntry(
                            versionA, rangeA.Map, rangeA.TileRow, rangeA.TileCol, rangeA.Kind, rangeA.MinUniqueId, rangeA.MaxUniqueId,
                            versionB, rangeB.Map, rangeB.TileRow, rangeB.TileCol, rangeB.Kind, rangeB.MinUniqueId, rangeB.MaxUniqueId,
                            overlapMin, overlapMax));
                    }
                }
            }
        }
        return overlaps;
    }

    private static List<AssetFirstSeenEntry> ComputeAssetFirstSeen(
        IReadOnlyList<string> versions,
        Dictionary<string, List<VersionRangeEntry>> rangesByVersion,
        IReadOnlyDictionary<string, int> versionOrder)
    {
        var firstSeen = new Dictionary<string, AssetFirstSeenEntry>(StringComparer.OrdinalIgnoreCase);
        foreach (var version in versions)
        {
            if (!rangesByVersion.TryGetValue(version, out var ranges)) continue;
            foreach (var range in ranges.OrderBy(r => r.MinUniqueId))
            {
                foreach (var asset in range.Assets)
                {
                    if (string.IsNullOrWhiteSpace(asset)) continue;
                    if (!firstSeen.TryGetValue(asset, out var existing))
                    {
                        firstSeen[asset] = new AssetFirstSeenEntry(asset, version, range.Map, range.TileRow, range.TileCol, range.Kind, range.MinUniqueId, range.MaxUniqueId);
                        continue;
                    }
                    var existingOrder = versionOrder[existing.Version];
                    var currentOrder = versionOrder[version];
                    if (currentOrder < existingOrder || (currentOrder == existingOrder && range.MinUniqueId < existing.MinUniqueId))
                    {
                        firstSeen[asset] = new AssetFirstSeenEntry(asset, version, range.Map, range.TileRow, range.TileCol, range.Kind, range.MinUniqueId, range.MaxUniqueId);
                    }
                }
            }
        }
        return firstSeen.Values
            .OrderBy(entry => versionOrder[entry.Version])
            .ThenBy(entry => entry.MinUniqueId)
            .ThenBy(entry => entry.AssetPath, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    private static List<AssetFolderSummary> ComputeAssetFolderSummaries(
        Dictionary<string, List<VersionRangeEntry>> rangesByVersion,
        Dictionary<string, List<PlacementAsset>> assetsByVersion)
    {
        var folderMap = new Dictionary<(string Version, string Map, int Row, int Col, PlacementKind Kind, string Folder), HashSet<string>>();
        foreach (var (version, ranges) in rangesByVersion)
        {
            foreach (var range in ranges)
            {
                foreach (var asset in range.Assets)
                {
                    if (string.IsNullOrWhiteSpace(asset)) continue;
                    AppendAsset(folderMap, version, range.Map, range.TileRow, range.TileCol, range.Kind, asset);
                }
            }
        }
        foreach (var (version, assets) in assetsByVersion)
        {
            foreach (var asset in assets)
            {
                if (string.IsNullOrWhiteSpace(asset.AssetPath)) continue;
                AppendAsset(folderMap, version, asset.Map, asset.TileRow, asset.TileCol, asset.Kind, asset.AssetPath);
            }
        }
        return folderMap
            .Select(kvp => new AssetFolderSummary(kvp.Key.Version, kvp.Key.Map, kvp.Key.Row, kvp.Key.Col, kvp.Key.Kind, kvp.Key.Folder, kvp.Value.Count))
            .OrderBy(summary => summary.Version, StringComparer.OrdinalIgnoreCase)
            .ThenBy(summary => summary.Map, StringComparer.OrdinalIgnoreCase)
            .ThenBy(summary => summary.TileRow)
            .ThenBy(summary => summary.TileCol)
            .ThenBy(summary => summary.Folder, StringComparer.OrdinalIgnoreCase)
            .ToList();

        static void AppendAsset(
            Dictionary<(string Version, string Map, int Row, int Col, PlacementKind Kind, string Folder), HashSet<string>> folderMap,
            string version,
            string map,
            int row,
            int col,
            PlacementKind kind,
            string assetPath)
        {
            var folder = ExtractFolder(assetPath);
            var key = (version, map, row, col, kind, folder);
            if (!folderMap.TryGetValue(key, out var set))
            {
                set = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                folderMap[key] = set;
            }
            set.Add(assetPath);
        }
    }

    private static List<AssetFolderTimelineEntry> ComputeAssetFolderTimeline(
        Dictionary<string, List<PlacementAsset>> assetsByVersion,
        IReadOnlyDictionary<string, int> versionOrder)
    {
        var aggregate = new Dictionary<string, AssetFolderTimelineBuilder>(StringComparer.OrdinalIgnoreCase);
        foreach (var (version, assets) in assetsByVersion)
        {
            foreach (var asset in assets)
            {
                if (string.IsNullOrWhiteSpace(asset.AssetPath)) continue;
                var folder = ExtractFolder(asset.AssetPath, depth: 2);
                var key = string.Concat(version, "||", folder);
                if (!aggregate.TryGetValue(key, out var builder))
                {
                    builder = new AssetFolderTimelineBuilder(version, folder);
                    aggregate[key] = builder;
                }
                builder.Register(asset);
            }
        }
        return aggregate.Values
            .OrderBy(builder => versionOrder.TryGetValue(builder.Version, out var order) ? order : int.MaxValue)
            .ThenBy(builder => builder.Folder, StringComparer.OrdinalIgnoreCase)
            .Select(builder => builder.Build())
            .ToList();
    }

    private static List<AssetTimelineEntry> BuildAssetTimeline(
        Dictionary<string, List<PlacementAsset>> assetsByVersion,
        IReadOnlyDictionary<string, int> versionOrder)
    {
        var entries = new List<AssetTimelineEntry>();
        foreach (var (version, assets) in assetsByVersion)
        {
            foreach (var asset in assets)
            {
                if (string.IsNullOrWhiteSpace(asset.AssetPath)) continue;
                entries.Add(new AssetTimelineEntry(
                    version,
                    asset.Map,
                    asset.TileRow,
                    asset.TileCol,
                    asset.Kind,
                    asset.UniqueId ?? 0,
                    asset.AssetPath,
                    ExtractFolder(asset.AssetPath, depth: 2),
                    ExtractCategory(asset.AssetPath),
                    ExtractSubcategory(asset.AssetPath)));
            }
        }
        return entries
            .OrderBy(entry => versionOrder.TryGetValue(entry.Version, out var order) ? order : int.MaxValue)
            .ThenBy(entry => entry.Folder, StringComparer.OrdinalIgnoreCase)
            .ThenBy(entry => entry.AssetPath, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    private static VersionComparisonResult EmptyResult(string rootDirectory, IReadOnlyList<string> warnings) =>
        EmptyResult(rootDirectory, warnings, Array.Empty<MapVersionSummary>());

    private static VersionComparisonResult EmptyResult(
        string rootDirectory,
        IReadOnlyList<string> warnings,
        IReadOnlyList<MapVersionSummary> mapSummaries) =>
        new(
            rootDirectory,
            "no_versions",
            Array.Empty<string>(),
            Array.Empty<VersionRangeEntry>(),
            mapSummaries,
            Array.Empty<RangeOverlapEntry>(),
            Array.Empty<AssetFirstSeenEntry>(),
            Array.Empty<AssetFolderSummary>(),
            Array.Empty<AssetFolderTimelineEntry>(),
            Array.Empty<AssetTimelineEntry>(),
            Array.Empty<DesignKitAssetEntry>(),
            Array.Empty<DesignKitRangeEntry>(),
            Array.Empty<DesignKitSummaryEntry>(),
            Array.Empty<DesignKitTimelineEntry>(),
            warnings);

    private static PlacementKind ParseKind(string value) =>
        value.Trim().Equals("M2", StringComparison.OrdinalIgnoreCase) ? PlacementKind.M2 : PlacementKind.WMO;

    private static uint ParseUInt(string value) =>
        uint.Parse(value.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture);

    private static IReadOnlyList<string> ParseAssets(string value) =>
        string.IsNullOrWhiteSpace(value)
            ? Array.Empty<string>()
            : value.Split('|', StringSplitOptions.RemoveEmptyEntries)
                .Select(NormalizeAssetPath)
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToList();

    private static string NormalizeAssetPath(string value) => value.Replace('\\', '/').Trim();

    private static string ExtractFolder(string assetPath) => ExtractFolder(assetPath, 2);

    private static string ExtractFolder(string assetPath, int depth)
    {
        var normalized = NormalizeAssetPath(assetPath);
        var segments = normalized.Split('/', StringSplitOptions.RemoveEmptyEntries);
        if (segments.Length == 0) return "(root)";
        var actualDepth = Math.Clamp(depth, 1, segments.Length);
        return string.Join('/', segments.Take(actualDepth));
    }

    private static string ExtractCategory(string assetPath)
    {
        var normalized = NormalizeAssetPath(assetPath);
        var segments = normalized.Split('/', StringSplitOptions.RemoveEmptyEntries);
        if (segments.Length >= 3) return segments[2];
        if (segments.Length >= 2) return segments[1];
        if (segments.Length >= 1) return segments[0];
        return "(root)";
    }

    private static string ExtractSubcategory(string assetPath)
    {
        var normalized = NormalizeAssetPath(assetPath);
        var segments = normalized.Split('/', StringSplitOptions.RemoveEmptyEntries);
        if (segments.Length >= 4) return segments[3];
        if (segments.Length >= 3) return segments[2];
        return "(none)";
    }

    private static (string Kit, string Rule) InferDesignKit(string assetPath)
    {
        var normalized = NormalizeAssetPath(assetPath);
        var segments = normalized.Split('/', StringSplitOptions.RemoveEmptyEntries);
        if (segments.Length == 0) return ("(unknown)", "fallback");

        // world/<map>/<zone>/...
        if (segments[0].Equals("world", StringComparison.OrdinalIgnoreCase))
        {
            if (segments.Length >= 3 && !segments[1].Equals("wmo", StringComparison.OrdinalIgnoreCase))
            {
                return (segments[2], "world-map-zone");
            }
            // world/wmo/<map>/<zoneOrCategory>/...
            if (segments.Length >= 4 && segments[1].Equals("wmo", StringComparison.OrdinalIgnoreCase))
            {
                var zoneOrCat = segments[3];
                if (!zoneOrCat.Equals("buildings", StringComparison.OrdinalIgnoreCase) &&
                    !zoneOrCat.Equals("doodads", StringComparison.OrdinalIgnoreCase))
                {
                    return (zoneOrCat, "wmo-map-zone");
                }
                return (segments[2], "wmo-map-only");
            }
            if (segments.Length >= 2)
            {
                // world/<map>/...
                return (segments[1], "world-map-only");
            }
        }

        // Generic: pick first top-level folder
        return (segments[0], "generic-top");
    }

    private static List<DesignKitAssetEntry> BuildDesignKitAssets(
        Dictionary<string, List<PlacementAsset>> assetsByVersion,
        IReadOnlyDictionary<string, int> versionOrder)
    {
        var list = new List<DesignKitAssetEntry>();
        foreach (var (version, assets) in assetsByVersion)
        {
            foreach (var a in assets)
            {
                if (string.IsNullOrWhiteSpace(a.AssetPath)) continue;
                var (kit, rule) = InferDesignKit(a.AssetPath);
                list.Add(new DesignKitAssetEntry(
                    version,
                    a.Map,
                    a.TileRow,
                    a.TileCol,
                    a.Kind,
                    a.UniqueId ?? 0,
                    a.AssetPath,
                    kit,
                    rule));
            }
        }
        return list
            .OrderBy(e => versionOrder.TryGetValue(e.Version, out var ord) ? ord : int.MaxValue)
            .ThenBy(e => e.DesignKit, StringComparer.OrdinalIgnoreCase)
            .ThenBy(e => e.AssetPath, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    private static List<DesignKitRangeEntry> BuildDesignKitRanges(
        Dictionary<string, List<VersionRangeEntry>> rangesByVersion)
    {
        var list = new List<DesignKitRangeEntry>();
        foreach (var (version, ranges) in rangesByVersion)
        {
            foreach (var r in ranges)
            {
                var byKit = new Dictionary<string, (int Count, HashSet<string> Distinct, HashSet<string> Rules)>(StringComparer.OrdinalIgnoreCase);
                foreach (var asset in r.Assets)
                {
                    if (string.IsNullOrWhiteSpace(asset)) continue;
                    var (kit, rule) = InferDesignKit(asset);
                    if (!byKit.TryGetValue(kit, out var agg)) agg = (0, new HashSet<string>(StringComparer.OrdinalIgnoreCase), new HashSet<string>(StringComparer.OrdinalIgnoreCase));
                    agg.Count++;
                    agg.Distinct.Add(asset);
                    agg.Rules.Add(rule);
                    byKit[kit] = agg;
                }
                foreach (var (kit, agg) in byKit)
                {
                    var sourceRule = agg.Rules.Count == 1 ? agg.Rules.First() : "mixed";
                    list.Add(new DesignKitRangeEntry(
                        version,
                        r.Map,
                        r.TileRow,
                        r.TileCol,
                        r.Kind,
                        r.MinUniqueId,
                        r.MaxUniqueId,
                        kit,
                        agg.Count,
                        agg.Distinct.Count,
                        sourceRule));
                }
            }
        }
        return list
            .OrderBy(e => e.Version, StringComparer.OrdinalIgnoreCase)
            .ThenBy(e => e.DesignKit, StringComparer.OrdinalIgnoreCase)
            .ThenBy(e => e.Map, StringComparer.OrdinalIgnoreCase)
            .ThenBy(e => e.TileRow)
            .ThenBy(e => e.TileCol)
            .ThenBy(e => e.MinUniqueId)
            .ToList();
    }

    private static List<DesignKitSummaryEntry> BuildDesignKitSummaries(
        List<DesignKitAssetEntry> designKitAssets)
    {
        var list = new List<DesignKitSummaryEntry>();
        foreach (var byVersion in designKitAssets.GroupBy(a => a.Version, StringComparer.OrdinalIgnoreCase))
        {
            foreach (var g in byVersion.GroupBy(a => a.DesignKit, StringComparer.OrdinalIgnoreCase))
            {
                var maps = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                var tiles = new HashSet<(string Map, int Row, int Col)>();
                uint minId = uint.MaxValue, maxId = 0;
                foreach (var a in g)
                {
                    maps.Add(a.Map);
                    tiles.Add((a.Map, a.TileRow, a.TileCol));
                    if (a.UniqueId > 0)
                    {
                        if (a.UniqueId < minId) minId = a.UniqueId;
                        if (a.UniqueId > maxId) maxId = a.UniqueId;
                    }
                }
                if (minId == uint.MaxValue) minId = 0;
                list.Add(new DesignKitSummaryEntry(
                    byVersion.Key,
                    g.Key,
                    g.Select(a => a.AssetPath).Distinct(StringComparer.OrdinalIgnoreCase).Count(),
                    g.Count(),
                    maps.Count,
                    tiles.Count,
                    minId,
                    maxId));
            }
        }
        return list
            .OrderBy(e => e.Version, StringComparer.OrdinalIgnoreCase)
            .ThenBy(e => e.DesignKit, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    private static List<DesignKitTimelineEntry> BuildDesignKitTimeline(
        List<DesignKitAssetEntry> designKitAssets,
        IReadOnlyDictionary<string, int> versionOrder)
    {
        var list = new List<DesignKitTimelineEntry>();
        foreach (var byKit in designKitAssets.GroupBy(a => a.DesignKit, StringComparer.OrdinalIgnoreCase))
        {
            foreach (var g in byKit.GroupBy(a => a.Version, StringComparer.OrdinalIgnoreCase))
            {
                var maps = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                var tiles = new HashSet<(string Map, int Row, int Col)>();
                uint minId = uint.MaxValue, maxId = 0;
                foreach (var a in g)
                {
                    maps.Add(a.Map);
                    tiles.Add((a.Map, a.TileRow, a.TileCol));
                    if (a.UniqueId > 0)
                    {
                        if (a.UniqueId < minId) minId = a.UniqueId;
                        if (a.UniqueId > maxId) maxId = a.UniqueId;
                    }
                }
                if (minId == uint.MaxValue) minId = 0;
                list.Add(new DesignKitTimelineEntry(
                    byKit.Key,
                    g.Key,
                    g.Count(),
                    maps.Count,
                    tiles.Count,
                    minId,
                    maxId));
            }
        }
        return list
            .OrderBy(e => e.DesignKit, StringComparer.OrdinalIgnoreCase)
            .ThenBy(e => versionOrder.TryGetValue(e.Version, out var ord) ? ord : int.MaxValue)
            .ToList();
    }

    private static int FolderDepth(string folder)
    {
        if (string.IsNullOrWhiteSpace(folder) || folder.Equals("(root)", StringComparison.OrdinalIgnoreCase)) return 0;
        return folder.Count(c => c == '/') + 1;
    }

    private static string BuildComparisonKey(IReadOnlyList<string> versions)
    {
        if (versions.Count == 0) return "no_versions";
        return string.Join("_vs_", versions.Select(Sanitize));
        string Sanitize(string value)
        {
            var builder = new StringBuilder(value.Length);
            foreach (var ch in value)
            {
                if (Array.IndexOf(InvalidFileNameChars, ch) >= 0 || ch == Path.DirectorySeparatorChar || ch == Path.AltDirectorySeparatorChar)
                    builder.Append('_');
                else
                    builder.Append(ch);
            }
            return builder.ToString().Replace('.', '_');
        }
    }

    private sealed record AssetData(
        IReadOnlyList<PlacementAsset> Assets,
        Dictionary<(string Map, int Row, int Col, PlacementKind Kind), HashSet<string>> Lookup);

    private sealed class AssetFolderTimelineBuilder
    {
        public string Version { get; }
        public string Folder { get; }
        private readonly HashSet<string> _assets = new(StringComparer.OrdinalIgnoreCase);
        private readonly HashSet<string> _maps = new(StringComparer.OrdinalIgnoreCase);
        private readonly HashSet<(int Row, int Col)> _tiles = new();
        private readonly HashSet<string> _subfolders = new(StringComparer.OrdinalIgnoreCase);
        private uint _minId = uint.MaxValue;
        private uint _maxId = 0;
        public AssetFolderTimelineBuilder(string version, string folder)
        {
            Version = version;
            Folder = folder;
        }
        public void Register(PlacementAsset asset)
        {
            _assets.Add(asset.AssetPath);
            _maps.Add(asset.Map);
            _tiles.Add((asset.TileRow, asset.TileCol));
            if (asset.UniqueId.HasValue)
            {
                if (asset.UniqueId.Value < _minId) _minId = asset.UniqueId.Value;
                if (asset.UniqueId.Value > _maxId) _maxId = asset.UniqueId.Value;
            }
            var subfolder = ExtractSubcategory(asset.AssetPath);
            if (!string.IsNullOrEmpty(subfolder) && !subfolder.Equals("(none)", StringComparison.OrdinalIgnoreCase))
                _subfolders.Add(subfolder);
        }
        public AssetFolderTimelineEntry Build()
        {
            if (_minId == uint.MaxValue) _minId = 0;
            return new AssetFolderTimelineEntry(
                Version,
                Folder,
                FolderDepth(Folder),
                _assets.Count,
                _maps.Count,
                _tiles.Count,
                _minId,
                _maxId,
                _maps.OrderBy(m => m, StringComparer.OrdinalIgnoreCase).ToList(),
                _subfolders.OrderBy(s => s, StringComparer.OrdinalIgnoreCase).ToList());
        }
    }
}
