using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using WoWRollback.Core.Models;

namespace WoWRollback.Core.Services.Viewer;

/// <summary>
/// Orchestrates viewer asset generation (minimaps, overlays, diffs, index/config).
/// </summary>
public sealed class ViewerReportWriter
{
    private readonly MinimapComposer _minimapComposer;
    private readonly OverlayBuilder _overlayBuilder;
    private readonly OverlayDiffBuilder _diffBuilder;

    public ViewerReportWriter()
        : this(new MinimapComposer(), new OverlayBuilder(), new OverlayDiffBuilder())
    {
    }

    public ViewerReportWriter(
        MinimapComposer minimapComposer,
        OverlayBuilder overlayBuilder,
        OverlayDiffBuilder diffBuilder)
    {
        _minimapComposer = minimapComposer ?? throw new ArgumentNullException(nameof(minimapComposer));
        _overlayBuilder = overlayBuilder ?? throw new ArgumentNullException(nameof(overlayBuilder));
        _diffBuilder = diffBuilder ?? throw new ArgumentNullException(nameof(diffBuilder));
    }

    /// <summary>
    /// Generates viewer artifacts for the supplied comparison result.
    /// </summary>
    /// <returns>Absolute path to the viewer root (or empty string when skipped).</returns>
    public string Generate(
        string comparisonDirectory,
        VersionComparisonResult result,
        ViewerOptions? options,
        (string Baseline, string Comparison)? diffPair = null)
    {
        ArgumentNullException.ThrowIfNull(comparisonDirectory);
        ArgumentNullException.ThrowIfNull(result);
        var resolvedOptions = options ?? ViewerOptions.CreateDefault();

        if (result.AssetTimelineDetailed is null || result.AssetTimelineDetailed.Count == 0)
            return string.Empty;

        var viewerRoot = Path.Combine(comparisonDirectory, "viewer");
        Directory.CreateDirectory(viewerRoot);

        var overlaysRoot = Path.Combine(viewerRoot, "overlays");
        var diffsRoot = Path.Combine(viewerRoot, "diffs");
        Directory.CreateDirectory(overlaysRoot);
        Directory.CreateDirectory(diffsRoot);

        var entries = result.AssetTimelineDetailed;
        var maps = entries
            .GroupBy(e => e.Map, StringComparer.OrdinalIgnoreCase)
            .OrderBy(g => g.Key, StringComparer.OrdinalIgnoreCase)
            .ToList();

        var entriesByVersion = entries
            .GroupBy(e => e.Version, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(g => g.Key, g => g.ToList(), StringComparer.OrdinalIgnoreCase);

        var chosenDefaultVersion = SelectDefaultVersion(resolvedOptions, result);
        var effectiveDiffPair = ResolveDiffPair(diffPair, resolvedOptions, result);
        var minimapLocator = MinimapLocator.Build(result.RootDirectory, result.Versions);

        var mapTileCatalog = new Dictionary<string, List<TileDescriptor>>(StringComparer.OrdinalIgnoreCase);

        foreach (var mapGroup in maps)
        {
            var mapName = mapGroup.Key;
            var safeMap = Sanitize(mapName);
            var mapOverlayDir = Path.Combine(overlaysRoot, safeMap);
            var mapDiffDir = Path.Combine(diffsRoot, safeMap);
            Directory.CreateDirectory(mapOverlayDir);
            Directory.CreateDirectory(mapDiffDir);

            var tiles = mapGroup
                .GroupBy(e => (e.TileRow, e.TileCol))
                .OrderBy(g => g.Key.TileRow)
                .ThenBy(g => g.Key.TileCol)
                .ToList();

            foreach (var tileGroup in tiles)
            {
                var (row, col) = tileGroup.Key;

                // Export minimap for each version that has this tile
                var versionsForTile = tileGroup
                    .Select(e => e.Version)
                    .Distinct(StringComparer.OrdinalIgnoreCase)
                    .OrderBy(v => v, StringComparer.OrdinalIgnoreCase)
                    .ToList();

                foreach (var version in versionsForTile)
                {
                    var safeVersion = Sanitize(version);
                    var versionMinimapRoot = Path.Combine(viewerRoot, "minimap", safeVersion);
                    var versionMapMinimapDir = Path.Combine(versionMinimapRoot, safeMap);
                    Directory.CreateDirectory(versionMapMinimapDir);

                    var hasTile = minimapLocator.TryGetTile(version, mapName, row, col, out var tileDescriptor);
                    var minimapFile = $"{mapName}_{col}_{row}.png";
                    var minimapPath = Path.Combine(versionMapMinimapDir, minimapFile);

                    if (hasTile)
                    {
                        using var tileStream = tileDescriptor.Open();
                        _minimapComposer.ComposeAsync(tileStream, minimapPath, resolvedOptions).GetAwaiter().GetResult();
                    }
                    else
                    {
                        _minimapComposer.WritePlaceholderAsync(minimapPath, resolvedOptions).GetAwaiter().GetResult();
                    }
                }

                var overlayPath = Path.Combine(mapOverlayDir, $"tile_r{row}_c{col}.json");
                var overlayJson = _overlayBuilder.BuildOverlayJson(mapName, row, col, entries, resolvedOptions);
                File.WriteAllText(overlayPath, overlayJson);

                if (effectiveDiffPair is { } pair &&
                    entriesByVersion.TryGetValue(pair.Baseline, out var baselineEntries) &&
                    entriesByVersion.TryGetValue(pair.Comparison, out var comparisonEntries))
                {
                    var diffJson = _diffBuilder.BuildDiffJson(mapName, row, col, baselineEntries, comparisonEntries, resolvedOptions);
                    var diffPath = Path.Combine(mapDiffDir, $"tile_r{row}_c{col}.json");
                    File.WriteAllText(diffPath, diffJson);
                }

                if (!mapTileCatalog.TryGetValue(mapName, out var tileList))
                {
                    tileList = new List<TileDescriptor>();
                    mapTileCatalog[mapName] = tileList;
                }

                tileList.Add(new TileDescriptor(row, col, versionsForTile));
            }
        }

        WriteIndexJson(viewerRoot, result, mapTileCatalog, chosenDefaultVersion, effectiveDiffPair);
        WriteConfigJson(viewerRoot, resolvedOptions, chosenDefaultVersion, effectiveDiffPair);
        CopyViewerAssets(viewerRoot);

        return viewerRoot;
    }

    private static string SelectDefaultVersion(ViewerOptions options, VersionComparisonResult result)
    {
        var requested = options.DefaultVersion;
        if (!string.IsNullOrWhiteSpace(requested))
        {
            var match = result.Versions.FirstOrDefault(v => v.Equals(requested, StringComparison.OrdinalIgnoreCase));
            if (!string.IsNullOrWhiteSpace(match))
                return match;
        }

        return result.Versions
            .OrderBy(v => v, StringComparer.OrdinalIgnoreCase)
            .FirstOrDefault() ?? result.Versions.First();
    }

    private static (string Baseline, string Comparison)? ResolveDiffPair(
        (string Baseline, string Comparison)? explicitPair,
        ViewerOptions options,
        VersionComparisonResult result)
    {
        static (string Baseline, string Comparison)? Normalize(
            (string Baseline, string Comparison)? pair,
            IReadOnlyList<string> versions)
        {
            if (pair is null) return null;
            var baseMatch = versions.FirstOrDefault(v => v.Equals(pair.Value.Baseline, StringComparison.OrdinalIgnoreCase));
            var compMatch = versions.FirstOrDefault(v => v.Equals(pair.Value.Comparison, StringComparison.OrdinalIgnoreCase));
            if (string.IsNullOrWhiteSpace(baseMatch) || string.IsNullOrWhiteSpace(compMatch)) return null;
            return (baseMatch, compMatch);
        }

        var normalized = Normalize(explicitPair, result.Versions)
            ?? Normalize(options.DiffPair, result.Versions);

        if (normalized is not null)
            return normalized;

        if (result.Versions.Count >= 2)
        {
            var ordered = result.Versions.OrderBy(v => v, StringComparer.OrdinalIgnoreCase).ToList();
            return (ordered[0], ordered[^1]);
        }

        return null;
    }

    private static void WriteIndexJson(
        string viewerRoot,
        VersionComparisonResult result,
        Dictionary<string, List<TileDescriptor>> mapTiles,
        string defaultVersion,
        (string Baseline, string Comparison)? diffPair)
    {
        var maps = mapTiles
            .OrderBy(kvp => kvp.Key, StringComparer.OrdinalIgnoreCase)
            .Select(kvp => new
            {
                map = kvp.Key,
                tiles = kvp.Value
                    .OrderBy(t => t.Row)
                    .ThenBy(t => t.Col)
                    .Select(t => new
                    {
                        row = t.Row,
                        col = t.Col,
                        versions = t.Versions
                    })
                    .ToList()
            })
            .ToList();

        var index = new
        {
            comparisonKey = result.ComparisonKey,
            defaultVersion,
            diff = diffPair is null ? null : new { baseline = diffPair.Value.Baseline, comparison = diffPair.Value.Comparison },
            versions = result.Versions,
            maps
        };

        var options = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            WriteIndented = true,
            Converters = { new JsonStringEnumConverter() }
        };

        var indexPath = Path.Combine(viewerRoot, "index.json");
        File.WriteAllText(indexPath, JsonSerializer.Serialize(index, options));
    }

    private static void WriteConfigJson(
        string viewerRoot,
        ViewerOptions options,
        string defaultVersion,
        (string Baseline, string Comparison)? diffPair)
    {
        var config = new
        {
            defaultVersion,
            diff = diffPair is null ? null : new { baseline = diffPair.Value.Baseline, comparison = diffPair.Value.Comparison },
            minimap = new { width = options.MinimapWidth, height = options.MinimapHeight },
            thresholds = new
            {
                distance = options.DiffDistanceThreshold,
                moveEpsilonRatio = options.MoveEpsilonRatio
            }
        };

        var jsonOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            WriteIndented = true
        };

        var configPath = Path.Combine(viewerRoot, "config.json");
        File.WriteAllText(configPath, JsonSerializer.Serialize(config, jsonOptions));
    }

    private static void CopyViewerAssets(string viewerRoot)
    {
        // Find ViewerAssets directory relative to this assembly
        var assemblyDir = Path.GetDirectoryName(typeof(ViewerReportWriter).Assembly.Location);
        if (assemblyDir is null) return;

        // Search upward from assembly location for ViewerAssets
        var currentDir = assemblyDir;
        string? assetsPath = null;

        for (int i = 0; i < 6; i++) // Search up to 6 levels
        {
            var candidate = Path.Combine(currentDir, "ViewerAssets");
            if (Directory.Exists(candidate))
            {
                assetsPath = candidate;
                break;
            }

            var parent = Directory.GetParent(currentDir);
            if (parent is null) break;
            currentDir = parent.FullName;
        }

        if (assetsPath is null || !Directory.Exists(assetsPath)) return;

        try
        {
            // Copy HTML files
            CopyIfExists(Path.Combine(assetsPath, "index.html"), Path.Combine(viewerRoot, "index.html"));
            CopyIfExists(Path.Combine(assetsPath, "tile.html"), Path.Combine(viewerRoot, "tile.html"));
            CopyIfExists(Path.Combine(assetsPath, "test.html"), Path.Combine(viewerRoot, "test.html"));
            CopyIfExists(Path.Combine(assetsPath, "styles.css"), Path.Combine(viewerRoot, "styles.css"));
            CopyIfExists(Path.Combine(assetsPath, "README.md"), Path.Combine(viewerRoot, "README.md"));
            CopyIfExists(Path.Combine(assetsPath, "start-viewer.ps1"), Path.Combine(viewerRoot, "start-viewer.ps1"));

            // Copy JS directory
            var jsSource = Path.Combine(assetsPath, "js");
            var jsTarget = Path.Combine(viewerRoot, "js");

            if (Directory.Exists(jsSource))
            {
                Directory.CreateDirectory(jsTarget);
                foreach (var file in Directory.GetFiles(jsSource, "*.js"))
                {
                    var fileName = Path.GetFileName(file);
                    File.Copy(file, Path.Combine(jsTarget, fileName), overwrite: true);
                }
            }
        }
        catch
        {
            // Fail silently - viewer will just be missing interactive assets
        }
    }

    private static void CopyIfExists(string source, string destination)
    {
        if (File.Exists(source))
        {
            File.Copy(source, destination, overwrite: true);
        }
    }

    private static string Sanitize(string value)
    {
        if (string.IsNullOrWhiteSpace(value)) return "unnamed";
        var invalid = Path.GetInvalidFileNameChars();
        var chars = value.Select(c => invalid.Contains(c) ? '_' : c).ToArray();
        var sanitized = new string(chars).Trim();
        return string.IsNullOrWhiteSpace(sanitized) ? "unnamed" : sanitized;
    }

    private sealed record TileDescriptor(int Row, int Col, IReadOnlyList<string> Versions);
}
