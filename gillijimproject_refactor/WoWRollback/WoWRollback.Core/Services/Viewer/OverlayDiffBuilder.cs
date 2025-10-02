using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using WoWRollback.Core.Models;

namespace WoWRollback.Core.Services.Viewer;

/// <summary>
/// Computes diff overlays between two viewer versions (added/removed/moved/changed).
/// </summary>
public sealed class OverlayDiffBuilder
{
    private const double TileSpanYards = 533.33333;

    /// <summary>
    /// Produces a diff JSON document for the supplied tile.
    /// </summary>
    public string BuildDiffJson(
        string map,
        int tileRow,
        int tileCol,
        IEnumerable<AssetTimelineDetailedEntry> baseline,
        IEnumerable<AssetTimelineDetailedEntry> comparison,
        ViewerOptions options)
    {
        ArgumentNullException.ThrowIfNull(map);
        ArgumentNullException.ThrowIfNull(baseline);
        ArgumentNullException.ThrowIfNull(comparison);
        options ??= ViewerOptions.CreateDefault();

        var baselineList = FilterByTile(baseline, map, tileRow, tileCol);
        var comparisonList = FilterByTile(comparison, map, tileRow, tileCol);

        var baselineBuckets = BuildBuckets(baselineList);
        var matchedPairs = new List<(AssetTimelineDetailedEntry Base, AssetTimelineDetailedEntry Comp)>();
        var added = new List<AssetTimelineDetailedEntry>();

        foreach (var candidate in comparisonList)
        {
            var match = FindBestMatch(candidate, baselineBuckets, options.DiffDistanceThreshold);
            if (match is null)
            {
                added.Add(candidate);
                continue;
            }

            matchedPairs.Add((match.Value.BaseEntry, candidate));
        }

        var removed = baselineBuckets.SelectMany(kvp => kvp.Value).ToList();

        var moveThreshold = options.MoveEpsilonRatio * TileSpanYards;
        var moved = new List<object>();
        var changed = new List<object>();

        foreach (var (baseEntry, compEntry) in matchedPairs)
        {
            removed.Remove(baseEntry);

            var delta = ComputeDistance(baseEntry, compEntry);
            var basePoint = BuildPointPayload(baseEntry, tileRow, tileCol, options);
            var compPoint = BuildPointPayload(compEntry, tileRow, tileCol, options);

            if (delta > moveThreshold)
            {
                moved.Add(new
                {
                    from = basePoint,
                    to = compPoint,
                    distance = Math.Round(delta, 4, MidpointRounding.AwayFromZero)
                });
                continue;
            }

            if (HasMetadataChange(baseEntry, compEntry))
            {
                changed.Add(new
                {
                    from = basePoint,
                    to = compPoint
                });
            }
        }

        var payload = new
        {
            map,
            tile = new { row = tileRow, col = tileCol },
            thresholds = new
            {
                distance = options.DiffDistanceThreshold,
                move = moveThreshold
            },
            added = added.Select(e => BuildPointPayload(e, tileRow, tileCol, options)).ToList(),
            removed = removed.Select(e => BuildPointPayload(e, tileRow, tileCol, options)).ToList(),
            moved,
            changed
        };

        return JsonSerializer.Serialize(payload, new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            WriteIndented = true,
            Converters = { new JsonStringEnumConverter() },
            NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals
        });
    }

    private static List<AssetTimelineDetailedEntry> FilterByTile(
        IEnumerable<AssetTimelineDetailedEntry> entries,
        string map,
        int tileRow,
        int tileCol)
    {
        return entries
            .Where(e => e.Map.Equals(map, StringComparison.OrdinalIgnoreCase) && e.TileRow == tileRow && e.TileCol == tileCol)
            .ToList();
    }

    private static Dictionary<string, List<AssetTimelineDetailedEntry>> BuildBuckets(IEnumerable<AssetTimelineDetailedEntry> entries)
    {
        var comparer = StringComparer.OrdinalIgnoreCase;
        return entries
            .GroupBy(e => e.AssetPath ?? string.Empty, comparer)
            .ToDictionary(g => g.Key, g => g.ToList(), comparer);
    }

    private static (AssetTimelineDetailedEntry BaseEntry, double Distance)? FindBestMatch(
        AssetTimelineDetailedEntry candidate,
        Dictionary<string, List<AssetTimelineDetailedEntry>> baselineBuckets,
        double distanceThreshold)
    {
        var key = candidate.AssetPath ?? string.Empty;
        if (!baselineBuckets.TryGetValue(key, out var candidates) || candidates.Count == 0)
            return null;

        AssetTimelineDetailedEntry? best = null;
        var bestDistance = double.MaxValue;
        foreach (var entry in candidates)
        {
            var distance = ComputeDistance(entry, candidate);
            if (distance > distanceThreshold) continue;
            if (distance < bestDistance)
            {
                bestDistance = distance;
                best = entry;
            }
        }

        if (best is null)
            return null;

        candidates.Remove(best);
        if (candidates.Count == 0)
            baselineBuckets.Remove(key);

        return (best, bestDistance);
    }

    private static double ComputeDistance(AssetTimelineDetailedEntry a, AssetTimelineDetailedEntry b)
    {
        var dx = a.WorldX - b.WorldX;
        var dy = a.WorldY - b.WorldY;
        return Math.Sqrt((dx * dx) + (dy * dy));
    }

    private static bool HasMetadataChange(AssetTimelineDetailedEntry a, AssetTimelineDetailedEntry b)
    {
        return !string.Equals(a.DesignKit, b.DesignKit, StringComparison.OrdinalIgnoreCase)
            || !string.Equals(a.SubkitPath, b.SubkitPath, StringComparison.OrdinalIgnoreCase)
            || !string.Equals(a.SourceRule, b.SourceRule, StringComparison.OrdinalIgnoreCase)
            || !string.Equals(a.Category, b.Category, StringComparison.OrdinalIgnoreCase)
            || !string.Equals(a.Subcategory, b.Subcategory, StringComparison.OrdinalIgnoreCase);
    }

    private static object BuildPointPayload(AssetTimelineDetailedEntry entry, int tileRow, int tileCol, ViewerOptions options)
    {
        static double Sanitize(double value) => double.IsNaN(value) || double.IsInfinity(value) ? 0.0 : value;

        var worldX = Sanitize(entry.WorldX);
        var worldY = Sanitize(entry.WorldY);
        var worldZ = Sanitize(entry.WorldZ);

        var (localX, localY) = CoordinateTransformer.ComputeLocalCoordinates(worldX, worldY, tileRow, tileCol);
        var (pixelX, pixelY) = CoordinateTransformer.ToPixels(localX, localY, options.MinimapWidth, options.MinimapHeight);

        return new
        {
            version = entry.Version,
            kind = entry.Kind,
            uniqueId = entry.UniqueId,
            assetPath = entry.AssetPath,
            folder = entry.Folder,
            category = entry.Category,
            subcategory = entry.Subcategory,
            designKit = entry.DesignKit,
            sourceRule = entry.SourceRule,
            kitRoot = entry.KitRoot,
            subkitPath = entry.SubkitPath,
            subkitTop = entry.SubkitTop,
            subkitDepth = entry.SubkitDepth,
            fileName = entry.FileName,
            fileStem = entry.FileStem,
            extension = entry.Extension,
            world = new { x = worldX, y = worldY, z = worldZ },
            local = new
            {
                x = Math.Round(localX, 6, MidpointRounding.AwayFromZero),
                y = Math.Round(localY, 6, MidpointRounding.AwayFromZero)
            },
            pixel = new
            {
                x = Math.Round(pixelX, 2, MidpointRounding.AwayFromZero),
                y = Math.Round(pixelY, 2, MidpointRounding.AwayFromZero)
            }
        };
    }
}
